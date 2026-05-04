"""Build the two training datasets for hybrid two-phase training.

Runs three resumable stages:
  1. Stream the Lichess HF dataset, filter by Elo + Termination, save two
     disjoint raw-game subsets (movetext + Result only).
  2. Build the shared tokenizer from the outcome subset and generate
     outcome-labeled samples ({+1, 0, -1} from game Result).
  3. Run parallel Stockfish on the disjoint subset to produce precisely
     labeled samples (tanh(cp/400)).

Each stage skips if its output files already exist. Use --force to re-run.

Outputs (under --out-dir):
  games_outcome.pt       raw outcome-subset games
  games_stockfish.pt     raw stockfish-subset games
  tokenizer.pt           shared Tokenizer (built from outcome games)
  outcome_samples.pt     list[(token_ids, outcome_label)]
  stockfish_samples.pt   list[(token_ids, stockfish_label)]
"""

import argparse
import random
from pathlib import Path

import chess
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from model import CLS_TOKEN
from train import (
    build_tokenizer_from_games,
    generate_samples_stockfish_parallel,
    parse_movetext,
)

# Lichess Result → outcome label from white's perspective.
RESULT_TO_LABEL = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def _save_as_memmap(
    samples: list[tuple[list[int], float]], out_dir: Path, name: str, max_seq_len: int = 128
) -> None:
    """Save samples as memory-mapped arrays for fast DataLoader access.

    Sequences longer than max_seq_len are truncated (keeps the most recent tokens,
    since the CLS token is at position 0 we keep ids[:max_seq_len]).

    Produces three files:
      {name}_tokens.bin   — (N, max_seq_len) int32, zero-padded
      {name}_labels.bin   — (N,) float32
      {name}_lengths.bin  — (N,) int32, actual sequence length per sample (capped at max_seq_len)
      {name}_meta.pt      — dict with 'n' and 'max_len'
    """
    n = len(samples)
    max_len = min(max(len(ids) for ids, _ in samples), max_seq_len)
    print(f"  memmap {name}: {n:,} samples, max_seq_len={max_len}")

    tokens = np.memmap(out_dir / f"{name}_tokens.bin", dtype=np.int32, mode="w+", shape=(n, max_len))
    labels = np.memmap(out_dir / f"{name}_labels.bin", dtype=np.float32, mode="w+", shape=(n,))
    lengths = np.memmap(out_dir / f"{name}_lengths.bin", dtype=np.int32, mode="w+", shape=(n,))

    for i, (ids, label) in enumerate(tqdm(samples, desc=f"  writing {name}", unit="sample")):
        ids = ids[:max_len]
        l = len(ids)
        tokens[i, :l] = ids
        labels[i] = label
        lengths[i] = l

    tokens.flush()
    labels.flush()
    lengths.flush()
    torch.save({"n": n, "max_len": max_len}, out_dir / f"{name}_meta.pt")
    size_gb = (tokens.nbytes + labels.nbytes + lengths.nbytes) / 1024 ** 3
    print(f"  memmap {name} saved ({size_gb:.2f} GB)")


def stage1_collect_games(args: argparse.Namespace) -> None:
    policy_games_path = args.out_dir / "games_outcome.pt"
    reward_games_path = args.out_dir / "games_stockfish.pt"
    if policy_games_path.exists() and reward_games_path.exists() and not args.force:
        print(f"Stage 1: skipping — {policy_games_path.name} and {reward_games_path.name} exist.")
        return

    lower_elo = min(args.reward_min_elo, args.policy_min_elo)
    print(
        f"Stage 1: streaming Lichess/standard-chess-games (Termination == 'Normal'), "
        f"reward Elo >= {args.reward_min_elo} (target {args.reward_games:,}), "
        f"policy Elo >= {args.policy_min_elo} (target {args.policy_games:,})..."
    )

    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    # Pre-filter by the lower of the two thresholds to skip clearly ineligible games.
    ds = ds.filter(
        lambda r: (
            r.get("WhiteElo") is not None
            and r.get("BlackElo") is not None
            and r["WhiteElo"] >= lower_elo
            and r["BlackElo"] >= lower_elo
            and r.get("Termination") == "Normal"
        )
    )

    policy_games: list[dict] = []
    reward_games: list[dict] = []
    keep_keys = ("movetext", "Result")

    for row in tqdm(ds, desc="Stage 1: streaming", unit="game"):
        white_elo = row.get("WhiteElo", 0)
        black_elo = row.get("BlackElo", 0)
        minimal = {k: row.get(k) for k in keep_keys}

        if len(reward_games) < args.reward_games and white_elo >= args.reward_min_elo and black_elo >= args.reward_min_elo:
            reward_games.append(minimal)

        if len(policy_games) < args.policy_games and white_elo >= args.policy_min_elo and black_elo >= args.policy_min_elo:
            policy_games.append(minimal)

        if len(reward_games) >= args.reward_games and len(policy_games) >= args.policy_games:
            break

    if len(reward_games) < args.reward_games or len(policy_games) < args.policy_games:
        print(
            f"  WARNING: dataset exhausted before target — "
            f"got {len(reward_games):,} reward + {len(policy_games):,} policy games."
        )

    print(f"Stage 1: saving {reward_games_path} ({len(reward_games):,} games)...")
    torch.save(reward_games, reward_games_path)
    print(f"Stage 1: saving {policy_games_path} ({len(policy_games):,} games)...")
    torch.save(policy_games, policy_games_path)


def _generate_outcome_samples(games, tokenizer, max_positions_per_game, skip_ply):
    """Build (token_ids, outcome_label) samples for the phase-1 dataset."""
    cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
    samples: list[tuple[list[int], float]] = []
    with tqdm(games, desc="Stage 2: outcome samples", unit="game") as pbar:
        for idx, game in enumerate(pbar):
            result = game.get("Result")
            if result not in RESULT_TO_LABEL:
                continue
            label = RESULT_TO_LABEL[result]

            movetext = game.get("movetext", "")
            if not movetext:
                continue
            move_sans = parse_movetext(movetext)
            if len(move_sans) < max(2, skip_ply + 1):
                continue

            eligible = list(range(skip_ply, len(move_sans)))
            num_positions = min(max_positions_per_game, len(eligible))
            rng = random.Random(idx)
            sample_indices = set(rng.sample(eligible, num_positions))

            board = chess.Board()
            valid_moves: list[str] = []
            for i, san in enumerate(move_sans):
                try:
                    move = board.parse_san(san)
                    board.push(move)
                    valid_moves.append(move.uci())
                except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                    break
                if i in sample_indices:
                    token_ids = [cls_id] + tokenizer.encode_moves(valid_moves)
                    samples.append((token_ids, label))

            if (idx + 1) % 50_000 == 0:
                pbar.set_postfix(samples=f"{len(samples):,}")

    return samples


def stage2_outcome_samples(args: argparse.Namespace) -> None:
    tokenizer_path = args.out_dir / "tokenizer.pt"
    meta_path = args.out_dir / "outcome_meta.pt"
    if tokenizer_path.exists() and meta_path.exists() and not args.force:
        print(f"Stage 2: skipping — {tokenizer_path.name} and {meta_path.name} exist.")
        return

    raw_games_path = args.out_dir / "games_outcome.pt"
    print(f"Stage 2: loading outcome games from {raw_games_path}...")
    games = torch.load(raw_games_path, weights_only=False)

    print("Stage 2: building tokenizer from all UCI moves...")
    tokenizer = build_tokenizer_from_games()
    print(f"Stage 2: tokenizer vocab size = {tokenizer.language_size}")
    torch.save(tokenizer, tokenizer_path)

    print("Stage 2: generating outcome samples (up to 20 per game)...")
    samples = _generate_outcome_samples(
        games,
        tokenizer,
        max_positions_per_game=20,
        skip_ply=0,
    )
    print(f"Stage 2: saving {len(samples):,} outcome samples as memmap...")
    _save_as_memmap(samples, args.out_dir, "outcome", max_seq_len=args.max_seq_len)


def _generate_policy_sequences(games, tokenizer, max_seq_len: int = 128) -> list[list[int]]:
    """Tokenize full game sequences for policy training.

    Each output sequence is [CLS, m1, m2, ..., mN], truncated to max_seq_len.
    Games with fewer than 2 valid UCI moves are skipped.
    """
    cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
    sequences: list[list[int]] = []
    with tqdm(games, desc="Stage 4: policy sequences", unit="game") as pbar:
        for game in pbar:
            movetext = game.get("movetext", "")
            if not movetext:
                continue
            move_sans = parse_movetext(movetext)
            if len(move_sans) < 2:
                continue
            board = chess.Board()
            move_ucis: list[str] = []
            for san in move_sans:
                try:
                    move = board.parse_san(san)
                    board.push(move)
                    move_ucis.append(move.uci())
                except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                    break
            if len(move_ucis) < 2:
                continue
            move_ucis = move_ucis[:max_seq_len - 1]
            sequences.append([cls_id] + tokenizer.encode_moves(move_ucis))
    return sequences


def _save_policy_memmap(
    sequences: list[list[int]], out_dir: Path, name: str, max_seq_len: int = 128
) -> None:
    """Save policy sequences as memory-mapped arrays (no labels).

    Produces:
      {name}_tokens.bin   — (N, max_len) int32, zero-padded
      {name}_lengths.bin  — (N,) int32, actual sequence length per sample
      {name}_meta.pt      — dict with 'n' and 'max_len'
    """
    n = len(sequences)
    max_len = min(max(len(s) for s in sequences), max_seq_len)
    print(f"  memmap {name}: {n:,} sequences, max_seq_len={max_len}")

    tokens = np.memmap(out_dir / f"{name}_tokens.bin", dtype=np.int32, mode="w+", shape=(n, max_len))
    lengths = np.memmap(out_dir / f"{name}_lengths.bin", dtype=np.int32, mode="w+", shape=(n,))

    for i, seq in enumerate(tqdm(sequences, desc=f"  writing {name}", unit="seq")):
        seq = seq[:max_len]
        l = len(seq)
        tokens[i, :l] = seq
        lengths[i] = l

    tokens.flush()
    lengths.flush()
    torch.save({"n": n, "max_len": max_len}, out_dir / f"{name}_meta.pt")
    size_gb = (tokens.nbytes + lengths.nbytes) / 1024 ** 3
    print(f"  memmap {name} saved ({size_gb:.3f} GB)")


def _process_puzzle(
    row: dict,
    tokenizer_symbol_map: dict,
    cls_id: int,
) -> list[int] | None:
    """Parse one Lichess puzzle row into a token sequence.

    Sequence layout: [CLS, setup_move, solver_move1, opp_response, solver_move2, ...]

    The setup move (Moves[0]) is included as context so the model conditions on it
    when predicting the solution. During training the loss on the setup move position
    is masked out — we model P[m_n | S, m_{<n}] where S is the setup move.

    Returns None if any move is illegal, unknown to the tokenizer, or the sequence
    has fewer than 3 tokens (CLS + setup + at least one solver move).
    """
    fen = row.get("FEN", "")
    moves_str = row.get("Moves", "")
    if not fen or not moves_str:
        return None
    uci_moves = moves_str.strip().split()
    if len(uci_moves) < 2:  # need setup + at least one solver move
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    # Tokenize all moves: setup first (as context), then the full solution.
    token_ids: list[int] = [cls_id]
    for uci in uci_moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        if uci not in tokenizer_symbol_map:
            return None
        token_ids.append(tokenizer_symbol_map[uci])
        board.push(move)

    if len(token_ids) < 3:  # CLS + setup + at least one solver move
        return None
    return token_ids


def stage3_stockfish_samples(args: argparse.Namespace) -> None:
    meta_path = args.out_dir / "stockfish_meta.pt"
    if meta_path.exists() and not args.force:
        print(f"Stage 3: skipping — {meta_path.name} exists.")
        return

    games_path = args.out_dir / "games_stockfish.pt"
    tokenizer_path = args.out_dir / "tokenizer.pt"
    print(f"Stage 3: loading {games_path} and {tokenizer_path}...")
    games = torch.load(games_path, weights_only=False)
    tokenizer = torch.load(tokenizer_path, weights_only=False)

    print(
        f"Stage 3: running parallel Stockfish ({args.workers} workers, "
        f"depth {args.stockfish_depth}) on {len(games):,} games..."
    )
    samples = generate_samples_stockfish_parallel(
        games,
        tokenizer,
        num_workers=args.workers,
        stockfish_depth=args.stockfish_depth,
        sample_rate=args.sample_rate,
        skew_exponent=args.position_skew,
    )

    print(f"Stage 3: saving {len(samples):,} stockfish samples as memmap...")
    _save_as_memmap(samples, args.out_dir, "stockfish", max_seq_len=args.max_seq_len)


def stage4_policy_sequences(args: argparse.Namespace) -> None:
    meta_path = args.out_dir / "policy_meta.pt"
    if meta_path.exists() and not args.force:
        print(f"Stage 4: skipping — {meta_path.name} exists.")
        return

    games_path = args.out_dir / "games_outcome.pt"
    tokenizer_path = args.out_dir / "tokenizer.pt"
    print(f"Stage 4: loading {games_path} and {tokenizer_path}...")
    games = torch.load(games_path, weights_only=False)
    tokenizer = torch.load(tokenizer_path, weights_only=False)

    print(f"Stage 4: tokenizing {len(games):,} games into policy sequences...")
    sequences = _generate_policy_sequences(games, tokenizer, max_seq_len=args.max_seq_len)
    print(f"Stage 4: saving {len(sequences):,} policy sequences as memmap...")
    _save_policy_memmap(sequences, args.out_dir, "policy", max_seq_len=args.max_seq_len)


def _write_test_subset_reward(out_dir: Path, src_name: str, dst_name: str, indices: np.ndarray) -> None:
    """Write a subset of a reward memmap (tokens+labels+lengths) to new files."""
    meta = torch.load(out_dir / f"{src_name}_meta.pt", weights_only=True)
    n_src, max_len = meta["n"], meta["max_len"]
    src_tokens = np.memmap(out_dir / f"{src_name}_tokens.bin", dtype=np.int32, mode="r", shape=(n_src, max_len))
    src_labels = np.memmap(out_dir / f"{src_name}_labels.bin", dtype=np.float32, mode="r", shape=(n_src,))
    src_lengths = np.memmap(out_dir / f"{src_name}_lengths.bin", dtype=np.int32, mode="r", shape=(n_src,))
    n_test = len(indices)
    dst_tokens = np.memmap(out_dir / f"{dst_name}_tokens.bin", dtype=np.int32, mode="w+", shape=(n_test, max_len))
    dst_labels = np.memmap(out_dir / f"{dst_name}_labels.bin", dtype=np.float32, mode="w+", shape=(n_test,))
    dst_lengths = np.memmap(out_dir / f"{dst_name}_lengths.bin", dtype=np.int32, mode="w+", shape=(n_test,))
    for i, idx in enumerate(tqdm(indices, desc=f"  writing {dst_name}", unit="sample")):
        dst_tokens[i] = src_tokens[idx]
        dst_labels[i] = src_labels[idx]
        dst_lengths[i] = src_lengths[idx]
    dst_tokens.flush()
    dst_labels.flush()
    dst_lengths.flush()
    torch.save({"n": n_test, "max_len": max_len}, out_dir / f"{dst_name}_meta.pt")
    print(f"  {dst_name}: {n_test:,} samples written")


def _write_test_subset_policy(out_dir: Path, src_name: str, dst_name: str, indices: np.ndarray) -> None:
    """Write a subset of a policy memmap (tokens+lengths, no labels) to new files."""
    meta = torch.load(out_dir / f"{src_name}_meta.pt", weights_only=True)
    n_src, max_len = meta["n"], meta["max_len"]
    src_tokens = np.memmap(out_dir / f"{src_name}_tokens.bin", dtype=np.int32, mode="r", shape=(n_src, max_len))
    src_lengths = np.memmap(out_dir / f"{src_name}_lengths.bin", dtype=np.int32, mode="r", shape=(n_src,))
    n_test = len(indices)
    dst_tokens = np.memmap(out_dir / f"{dst_name}_tokens.bin", dtype=np.int32, mode="w+", shape=(n_test, max_len))
    dst_lengths = np.memmap(out_dir / f"{dst_name}_lengths.bin", dtype=np.int32, mode="w+", shape=(n_test,))
    for i, idx in enumerate(tqdm(indices, desc=f"  writing {dst_name}", unit="seq")):
        dst_tokens[i] = src_tokens[idx]
        dst_lengths[i] = src_lengths[idx]
    dst_tokens.flush()
    dst_lengths.flush()
    torch.save({"n": n_test, "max_len": max_len}, out_dir / f"{dst_name}_meta.pt")
    print(f"  {dst_name}: {n_test:,} sequences written")


def stage_build_test_splits(args: argparse.Namespace, out_dir: Path) -> None:
    """Build held-out test sets for reward and policy models from existing memmaps.

    Uses a fixed random seed (42) so the same indices are always selected.
    Saves the chosen indices to {name}_test_indices.npy so the corresponding
    training memmap loader can exclude them — making train and test disjoint
    even though they share the underlying .bin file.

    Produces:
      stockfish_test_*.bin / stockfish_test_meta.pt / stockfish_test_indices.npy
      policy_test_*.bin    / policy_test_meta.pt    / policy_test_indices.npy
    """
    rng = np.random.default_rng(42)

    # Reward test set
    reward_test_meta = out_dir / "stockfish_test_meta.pt"
    sf_meta_path = out_dir / "stockfish_meta.pt"
    if (not reward_test_meta.exists() or args.force) and sf_meta_path.exists():
        print(f"Test splits: building stockfish_test ({args.reward_test_size:,} samples)...")
        sf_meta = torch.load(sf_meta_path, weights_only=True)
        n = sf_meta["n"]
        test_n = min(args.reward_test_size, n)
        idx = rng.choice(n, size=test_n, replace=False)
        idx.sort()
        _write_test_subset_reward(out_dir, "stockfish", "stockfish_test", idx)
        np.save(out_dir / "stockfish_test_indices.npy", idx)
        print(f"  saved stockfish_test_indices.npy ({test_n:,} indices excluded from training)")
    elif reward_test_meta.exists():
        print("Test splits: stockfish_test already exists, skipping.")

    # Policy test set
    policy_test_meta = out_dir / "policy_test_meta.pt"
    pol_meta_path = out_dir / "policy_meta.pt"
    if (not policy_test_meta.exists() or args.force) and pol_meta_path.exists():
        print(f"Test splits: building policy_test ({args.policy_test_size:,} sequences)...")
        pol_meta = torch.load(pol_meta_path, weights_only=True)
        n = pol_meta["n"]
        test_n = min(args.policy_test_size, n)
        idx = rng.choice(n, size=test_n, replace=False)
        idx.sort()
        _write_test_subset_policy(out_dir, "policy", "policy_test", idx)
        np.save(out_dir / "policy_test_indices.npy", idx)
        print(f"  saved policy_test_indices.npy ({test_n:,} indices excluded from training)")
    elif policy_test_meta.exists():
        print("Test splits: policy_test already exists, skipping.")


def stage5_puzzle_samples(args: argparse.Namespace, tokenizer, out_dir: Path) -> None:
    train_done = (out_dir / "puzzle_meta.pt").exists()
    test_done = (out_dir / "puzzle_test_meta.pt").exists()
    if train_done and test_done and not args.force:
        print("Stage 5: skipping — puzzle_meta.pt and puzzle_test_meta.pt exist.")
        return

    print("Stage 5: loading Lichess/chess-puzzles from HuggingFace...")
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    min_pop = args.min_puzzle_popularity
    min_plays = args.min_puzzle_plays
    cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
    sym_map = tokenizer.symbol_to_token
    test_seqs: list[list[int]] = []
    train_seqs: list[list[int]] = []
    skipped = 0
    test_target = args.puzzle_test_size
    train_target = args.puzzle_count

    with tqdm(ds, desc="Stage 5: puzzles", unit="puzzle") as pbar:
        for row in pbar:
            if min_pop is not None and row.get("Popularity", 0) < min_pop:
                continue
            if min_plays is not None and row.get("NbPlays", 0) < min_plays:
                continue
            seq = _process_puzzle(row, sym_map, cls_id)
            if seq is None:
                skipped += 1
                continue
            if len(test_seqs) < test_target:
                test_seqs.append(seq)
            else:
                train_seqs.append(seq)
            pbar.set_postfix(test=len(test_seqs), train=len(train_seqs), skipped=skipped)
            if train_target is not None and len(train_seqs) >= train_target:
                break

    print(
        f"Stage 5: test={len(test_seqs):,} puzzles, "
        f"train={len(train_seqs):,} puzzles, "
        f"skipped={skipped:,} invalid."
    )
    if test_seqs and not test_done:
        _save_policy_memmap(test_seqs, out_dir, "puzzle_test", max_seq_len=args.max_seq_len)
    if train_seqs and not train_done:
        _save_policy_memmap(train_seqs, out_dir, "puzzle", max_seq_len=args.max_seq_len)
    elif not train_seqs:
        print("Stage 5: WARNING — no training puzzles collected.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--policy-games", type=int, default=1_000_000,
                        help="Number of games to collect for policy model training")
    parser.add_argument("--reward-games", type=int, default=1_000_000,
                        help="Number of games to collect for reward model (Stockfish eval)")
    parser.add_argument("--policy-min-elo", type=int, default=1800,
                        help="Min Elo for both players in policy training games")
    parser.add_argument("--reward-min-elo", type=int, default=1500,
                        help="Min Elo for both players in reward model training games")
    parser.add_argument("--sample-rate", type=float, default=0.25,
                        help="Fraction of positions to sample per game (scales with game length)")
    parser.add_argument("--position-skew", type=float, default=1.5,
                        help="Power-law exponent weighting later positions; 1.0=linear, higher=more mid/late")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Truncate token sequences to this length when writing .bin files")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if their outputs already exist",
    )
    parser.add_argument("--puzzle-count", type=int, default=None, dest="puzzle_count",
        help="Max puzzles to include (default: all ~4.99M)")
    parser.add_argument("--min-puzzle-popularity", type=int, default=None, dest="min_puzzle_popularity",
        help="Min Lichess Popularity score (0-100 scale)")
    parser.add_argument("--min-puzzle-plays", type=int, default=None, dest="min_puzzle_plays",
        help="Min NbPlays for a puzzle to be included")
    parser.add_argument("--skip-puzzles", action="store_true",
        help="Skip Stage 5 puzzle processing")
    parser.add_argument("--puzzle-test-size", type=int, default=100_000, dest="puzzle_test_size",
        help="Number of puzzle sequences held out for the test set (default: 100000)")
    parser.add_argument("--reward-test-size", type=int, default=50_000, dest="reward_test_size",
        help="Number of reward positions held out for the test set (default: 50000)")
    parser.add_argument("--policy-test-size", type=int, default=50_000, dest="policy_test_size",
        help="Number of policy sequences held out for the test set (default: 50000)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage1_collect_games(args)
    stage2_outcome_samples(args)
    stage3_stockfish_samples(args)
    stage4_policy_sequences(args)

    tokenizer_path = args.out_dir / "tokenizer.pt"
    if not args.skip_puzzles and tokenizer_path.exists():
        tokenizer = torch.load(tokenizer_path, weights_only=False)
        stage5_puzzle_samples(args, tokenizer, args.out_dir)
    elif not args.skip_puzzles:
        print("Stage 5: skipping — tokenizer.pt not found (run stages 1-2 first).")

    stage_build_test_splits(args, args.out_dir)

    print("\nAll stages complete. Artifacts:")
    for name in (
        "games_outcome.pt",
        "games_stockfish.pt",
        "tokenizer.pt",
        "outcome_tokens.bin",
        "outcome_labels.bin",
        "outcome_lengths.bin",
        "outcome_meta.pt",
        "stockfish_tokens.bin",
        "stockfish_labels.bin",
        "stockfish_lengths.bin",
        "stockfish_meta.pt",
        "policy_tokens.bin",
        "policy_lengths.bin",
        "policy_meta.pt",
        "puzzle_tokens.bin",
        "puzzle_lengths.bin",
        "puzzle_meta.pt",
        "puzzle_test_tokens.bin",
        "puzzle_test_lengths.bin",
        "puzzle_test_meta.pt",
        "stockfish_test_tokens.bin",
        "stockfish_test_labels.bin",
        "stockfish_test_lengths.bin",
        "stockfish_test_meta.pt",
        "policy_test_tokens.bin",
        "policy_test_lengths.bin",
        "policy_test_meta.pt",
    ):
        path = args.out_dir / name
        size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
