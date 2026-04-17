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
import torch
from datasets import load_dataset

from model import CLS_TOKEN
from train import (
    build_tokenizer_from_games,
    generate_samples_stockfish_parallel,
    parse_movetext,
)

# Lichess Result → outcome label from white's perspective.
RESULT_TO_LABEL = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def stage1_collect_games(args: argparse.Namespace) -> None:
    out_games_path = args.out_dir / "games_outcome.pt"
    sf_games_path = args.out_dir / "games_stockfish.pt"
    if out_games_path.exists() and sf_games_path.exists() and not args.force:
        print(f"Stage 1: skipping — {out_games_path.name} and {sf_games_path.name} exist.")
        return

    total = args.outcome_games + args.stockfish_games
    print(
        f"Stage 1: streaming Lichess/standard-chess-games "
        f"(Elo >= {args.min_elo}, Termination == 'Normal'), "
        f"collecting {args.outcome_games:,} outcome + {args.stockfish_games:,} stockfish games..."
    )

    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)
    ds = ds.filter(
        lambda r: (
            r.get("WhiteElo") is not None
            and r.get("BlackElo") is not None
            and r["WhiteElo"] >= args.min_elo
            and r["BlackElo"] >= args.min_elo
            and r.get("Termination") == "Normal"
        )
    )

    outcome_games: list[dict] = []
    stockfish_games: list[dict] = []
    keep_keys = ("movetext", "Result")

    for i, row in enumerate(ds):
        minimal = {k: row.get(k) for k in keep_keys}
        if i < args.outcome_games:
            outcome_games.append(minimal)
        else:
            stockfish_games.append(minimal)

        if (i + 1) % 50_000 == 0:
            print(f"  collected {i + 1:,}/{total:,} games...")
        if i + 1 >= total:
            break

    if len(outcome_games) < args.outcome_games or len(stockfish_games) < args.stockfish_games:
        print(
            f"  WARNING: dataset exhausted before target — "
            f"got {len(outcome_games):,} outcome + {len(stockfish_games):,} stockfish."
        )

    print(f"Stage 1: saving {out_games_path} ({len(outcome_games):,} games)...")
    torch.save(outcome_games, out_games_path)
    print(f"Stage 1: saving {sf_games_path} ({len(stockfish_games):,} games)...")
    torch.save(stockfish_games, sf_games_path)


def _generate_outcome_samples(games, tokenizer, max_positions_per_game, skip_ply):
    """Build (token_ids, outcome_label) samples for the phase-1 dataset."""
    cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
    samples: list[tuple[list[int], float]] = []
    for idx, game in enumerate(games):
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
                valid_moves.append(san)
            except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                break
            if i in sample_indices:
                try:
                    token_ids = [cls_id] + tokenizer.encode_moves(valid_moves)
                except KeyError:
                    continue
                samples.append((token_ids, label))

        if (idx + 1) % 50_000 == 0:
            print(f"  processed {idx + 1:,}/{len(games):,} games, {len(samples):,} samples")

    return samples


def stage2_outcome_samples(args: argparse.Namespace) -> None:
    tokenizer_path = args.out_dir / "tokenizer.pt"
    samples_path = args.out_dir / "outcome_samples.pt"
    if tokenizer_path.exists() and samples_path.exists() and not args.force:
        print(f"Stage 2: skipping — {tokenizer_path.name} and {samples_path.name} exist.")
        return

    raw_games_path = args.out_dir / "games_outcome.pt"
    sf_games_path = args.out_dir / "games_stockfish.pt"
    print(f"Stage 2: loading outcome games from {raw_games_path}...")
    games = torch.load(raw_games_path, weights_only=False)

    # Build tokenizer from BOTH game sets so the vocab covers moves in the
    # disjoint Stockfish subset too. Without this, rare SAN disambiguations
    # like 'Qfxd8+' that only appear in the smaller set cause KeyErrors.
    all_games_for_vocab = list(games)
    if sf_games_path.exists():
        sf_games = torch.load(sf_games_path, weights_only=False)
        all_games_for_vocab.extend(sf_games)
        print(f"Stage 2: building tokenizer from {len(all_games_for_vocab):,} games (outcome + stockfish)...")
    else:
        print(f"Stage 2: building tokenizer from {len(all_games_for_vocab):,} games (outcome only)...")
    tokenizer = build_tokenizer_from_games(all_games_for_vocab, max_language_size=args.max_language_size)
    print(f"Stage 2: tokenizer vocab size = {tokenizer.language_size}")
    torch.save(tokenizer, tokenizer_path)

    print(
        f"Stage 2: generating outcome samples "
        f"(up to {args.max_positions_per_game} per game, skip_ply={args.skip_ply})..."
    )
    samples = _generate_outcome_samples(
        games,
        tokenizer,
        max_positions_per_game=args.max_positions_per_game,
        skip_ply=args.skip_ply,
    )
    print(f"Stage 2: saving {len(samples):,} outcome samples to {samples_path}...")
    torch.save(samples, samples_path)


def stage3_stockfish_samples(args: argparse.Namespace) -> None:
    samples_path = args.out_dir / "stockfish_samples.pt"
    if samples_path.exists() and not args.force:
        print(f"Stage 3: skipping — {samples_path.name} exists.")
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
        max_positions_per_game=args.max_positions_per_game,
        skip_ply=args.sf_skip_ply,
    )

    print(f"Stage 3: saving {len(samples):,} stockfish samples to {samples_path}...")
    torch.save(samples, samples_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--outcome-games", type=int, default=1_000_000)
    parser.add_argument("--stockfish-games", type=int, default=30_000)
    parser.add_argument("--min-elo", type=int, default=1500)
    parser.add_argument("--max-positions-per-game", type=int, default=20)
    parser.add_argument("--skip-ply", type=int, default=10,
                        help="Plies to drop from outcome dataset (noisy opening labels)")
    parser.add_argument("--sf-skip-ply", type=int, default=0,
                        help="Plies to drop from Stockfish dataset (0 = include openings)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--max-language-size", type=int, default=2000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages even if their outputs already exist",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stage1_collect_games(args)
    stage2_outcome_samples(args)
    stage3_stockfish_samples(args)

    print("\nAll stages complete. Artifacts:")
    for name in (
        "games_outcome.pt",
        "games_stockfish.pt",
        "tokenizer.pt",
        "outcome_samples.pt",
        "stockfish_samples.pt",
    ):
        path = args.out_dir / name
        size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
