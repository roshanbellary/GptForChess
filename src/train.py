import argparse
import atexit
import contextlib
import math  # noqa: F401 (used in eval_policy)
import multiprocessing as mp
import re
import shutil
import time
import numpy as np
from pathlib import Path
import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from tokenizer import Tokenizer
from model import ChessRewardModel, ChessPolicyModel, DummyRewardModel, CLS_TOKEN, PAD_TOKEN

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/local/bin/stockfish"

RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MOVE_NUMBER_RE = re.compile(r"^\d+\.(\.\.)?$")
# Brace-delimited PGN comments like {[%eval 0.37]} and {[%clk 0:05:00]}.
# Non-greedy to handle multiple comments in one movetext.
BRACE_COMMENT_RE = re.compile(r"\{[^}]*\}")


def normalize_cp(centipawns: int) -> float:
    """Map centipawn score to [-1, 1] using tanh scaling."""
    return math.tanh(centipawns / 400.0)


def material_eval(board: chess.Board) -> float:
    """Material-count evaluation as a fallback for Stockfish."""
    return DummyRewardModel()(board)


class StockfishEvaluator:
    """Wraps a persistent Stockfish engine for batch evaluation."""
    def __init__(self, engine_path: str = STOCKFISH_PATH, depth: int = 15):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth

    def __call__(self, board: chess.Board) -> float:
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white()
        if score.is_mate():
            return 1.0 if score.mate() > 0 else -1.0
        return normalize_cp(score.score())

    def close(self):
        self.engine.quit()


def parse_movetext(movetext: str) -> list[str]:
    """Parse PGN movetext into a list of SAN moves.

    Handles Lichess-style annotations like '{[%eval 0.37]}' and '{[%clk 0:05:00]}'
    by stripping them before tokenization. Without this, annotated games get
    truncated mid-replay when parse_san chokes on comment fragments.

    Input format: '1. d4 {[%eval 0.13]} d5 2. Nf3 ... 1-0'
    Returns: ['d4', 'd5', 'Nf3', ...]
    """
    cleaned = BRACE_COMMENT_RE.sub(" ", movetext)
    tokens = cleaned.split()
    moves = []
    for tok in tokens:
        if tok in RESULT_TOKENS:
            continue
        if MOVE_NUMBER_RE.match(tok):
            continue
        moves.append(tok)
    return moves


def load_filtered_dataset(min_elo: int = 1500, min_rows: int = 100_000):
    """Load and filter the Lichess HuggingFace dataset.

    Filters:
    - WhiteElo >= min_elo AND BlackElo >= min_elo
    - Termination == 'Normal'

    Returns the filtered dataset and raises ValueError if too few rows.
    """
    print("Loading Lichess dataset from HuggingFace...")
    ds = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)

    print(f"Filtering for Elo >= {min_elo} and Termination == 'Normal'...")
    ds_filtered = ds.filter(
        lambda row: (
            row["WhiteElo"] is not None
            and row["BlackElo"] is not None
            and row["WhiteElo"] >= min_elo
            and row["BlackElo"] >= min_elo
            and row.get("Termination") == "Normal"
        )
    )

    # Materialize enough rows to validate the threshold
    print(f"Collecting at least {min_rows:,} filtered games...")
    rows = []
    for row in ds_filtered:
        rows.append(row)
        if len(rows) % 50_000 == 0:
            print(f"  collected {len(rows):,} games so far...")
        if len(rows) >= min_rows:
            break

    if len(rows) < min_rows:
        raise ValueError(
            f"Only found {len(rows):,} games matching filters, "
            f"need at least {min_rows:,}."
        )

    print(f"Collected {len(rows):,} games (target met).")
    return rows


def _enumerate_all_uci_moves() -> list[str]:
    """Enumerate every UCI move string that can legally appear in a chess game.

    Uses direct geometric enumeration rather than board simulation to avoid
    edge cases where king placement blocks valid destination squares.
    Covers all piece movement patterns: lines (rook/queen), diagonals
    (bishop/queen), L-shapes (knight), and pawn promotions.
    """
    seen: set[str] = set()
    for from_sq in chess.SQUARES:
        fr = chess.square_rank(from_sq)
        ff = chess.square_file(from_sq)
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            tr = chess.square_rank(to_sq)
            tf = chess.square_file(to_sq)
            dr = abs(tr - fr)
            df = abs(tf - ff)

            is_line   = (dr == 0 or df == 0)                           # rook / queen
            is_diag   = (dr == df)                                      # bishop / queen
            is_knight = (dr == 2 and df == 1) or (dr == 1 and df == 2)

            if not (is_line or is_diag or is_knight):
                continue

            seen.add(chess.Move(from_sq, to_sq).uci())

            # Promotion variants: pawn on 7th rank advancing to 8th (or 2nd→1st)
            if ((fr == 6 and tr == 7) or (fr == 1 and tr == 0)) and df <= 1:
                for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    seen.add(chess.Move(from_sq, to_sq, promotion=promo).uci())

    return list(seen)


def _weighted_sample(eligible: list[int], k: int, skew_exponent: float, seed: int) -> set[int]:
    """Sample k positions from eligible without replacement, skewed toward later positions.

    Weights grow as (position_rank + 1)^skew_exponent so later positions in a game
    are proportionally more likely to be selected. skew_exponent=1.0 gives linear
    weighting; higher values concentrate more mass at the end of the game.
    """
    n = len(eligible)
    k = min(k, n)
    if k == n:
        return set(eligible)
    weights = np.array([(i + 1) ** skew_exponent for i in range(n)], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=k, replace=False, p=weights)
    return {eligible[i] for i in chosen}


def build_tokenizer_from_games(games: list[dict] | None = None) -> Tokenizer:
    """Build a move-level tokenizer covering all 1968 UCI moves."""
    uci_moves = _enumerate_all_uci_moves()
    print(f"  building tokenizer from {len(set(uci_moves)):,} UCI moves (no BPE)")
    tokenizer = Tokenizer()
    tokenizer.train_tokenizer(uci_moves, max_language_size=len(set(uci_moves)))
    tokenizer.add_special_tokens([CLS_TOKEN, PAD_TOKEN])
    return tokenizer


def _load_train_idx(out_dir: Path, name: str, n: int) -> np.ndarray | None:
    """If `{name}_test_indices.npy` exists, return the complement (training-only
    indices into the full memmap). Returns None when no test split is recorded —
    in that case the caller indexes into the memmap directly.

    Names ending in '_test' always return None (the test memmap should not
    exclude itself).
    """
    if name.endswith("_test"):
        return None
    test_idx_file = out_dir / f"{name}_test_indices.npy"
    if not test_idx_file.exists():
        return None
    test_idx = np.load(test_idx_file)
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    return np.where(mask)[0]


class ChessPositionDataset(Dataset):
    def __init__(
        self,
        games: list[dict],
        tokenizer: Tokenizer,
        eval_fn=material_eval,
        sample_rate: float = 0.25,
        skew_exponent: float = 1.5,
    ):
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.samples: list[tuple[list[int], float]] = []
        self._memmap = False
        self._train_idx: np.ndarray | None = None
        self._generate_samples(games, eval_fn, sample_rate, skew_exponent)

    def _generate_samples(self, games, eval_fn, sample_rate, skew_exponent):
        for idx, game in enumerate(games):
            movetext = game.get("movetext", "")
            if not movetext:
                continue
            move_sans = parse_movetext(movetext)
            if len(move_sans) < 2:
                continue

            board = chess.Board()
            eligible = list(range(len(move_sans)))
            # Scale sample count with game length — longer games have more
            # evaluation swings and contribute proportionally more samples.
            num_positions = max(1, int(len(move_sans) * sample_rate))
            # Deterministic weighted sampling seeded by game index so serial
            # and parallel paths produce identical sample sets for the same input.
            sample_indices = _weighted_sample(eligible, num_positions, skew_exponent, seed=idx)

            valid_moves = []
            for i, san in enumerate(move_sans):
                try:
                    move = board.parse_san(san)
                    board.push(move)
                    valid_moves.append(move.uci())
                except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                    break

                if i in sample_indices:
                    token_ids = [self.cls_id] + self.tokenizer.encode_moves(valid_moves)
                    score = eval_fn(board)
                    self.samples.append((token_ids, score))

            if (idx + 1) % 10_000 == 0:
                print(f"  processed {idx + 1:,} games, {len(self.samples):,} positions...")

    def __len__(self) -> int:
        if self._memmap:
            if self._train_idx is not None:
                return len(self._train_idx)
            return len(self._mm_labels)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self._memmap:
            if self._train_idx is not None:
                idx = int(self._train_idx[idx])
            tokens = torch.from_numpy(np.array(self._mm_tokens[idx], dtype=np.int32)).long()
            length = int(self._mm_lengths[idx])
            mask = torch.arange(tokens.shape[0]) >= length  # True = padded
            return tokens, mask, float(self._mm_labels[idx])
        token_ids, score = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), score

    @classmethod
    def from_samples(cls, samples, tokenizer: Tokenizer):
        """Build a dataset from pre-generated (token_ids, score) samples."""
        inst = cls.__new__(cls)
        inst.tokenizer = tokenizer
        inst.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        inst.samples = list(samples)
        inst._memmap = False
        return inst

    @classmethod
    def from_file(cls, samples_path: str, tokenizer: Tokenizer):
        """Load (token_ids, score) samples from a torch.save file."""
        samples = torch.load(samples_path, weights_only=False)
        return cls.from_samples(samples, tokenizer)

    @classmethod
    def from_memmap(cls, out_dir: Path, name: str, tokenizer: Tokenizer):
        """Load pre-padded samples from memory-mapped arrays (fast DataLoader path).

        If a sibling file `{name}_test_indices.npy` exists, those indices are
        excluded from this dataset — used to make training disjoint from the
        held-out test split that shares the same underlying .bin file.
        """
        meta = torch.load(out_dir / f"{name}_meta.pt", weights_only=True)
        n, max_len = meta["n"], meta["max_len"]
        inst = cls.__new__(cls)
        inst.tokenizer = tokenizer
        inst.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        inst._memmap = True
        inst._mm_tokens = np.memmap(out_dir / f"{name}_tokens.bin", dtype=np.int32, mode="r", shape=(n, max_len))
        inst._mm_labels = np.memmap(out_dir / f"{name}_labels.bin", dtype=np.float32, mode="r", shape=(n,))
        inst._mm_lengths = np.memmap(out_dir / f"{name}_lengths.bin", dtype=np.int32, mode="r", shape=(n,))
        inst._train_idx = _load_train_idx(out_dir, name, n)
        return inst


# ---------------------------------------------------------------------------
# Parallel Stockfish-backed sample generation.
#
# One Stockfish subprocess per worker process. Each worker:
#   1. receives a game + its index (used to seed a local random.Random)
#   2. replays the game, samples positions, tokenizes move prefixes
#   3. evaluates each sampled position with its own Stockfish engine
#   4. returns a list of (token_ids, score) tuples
#
# The main process collects results via imap_unordered and flattens them.
# ---------------------------------------------------------------------------

# Module-level state populated by _init_worker in each spawned process.
_worker_engine = None
_worker_tokenizer = None
_worker_cls_id = None
_worker_sample_rate = None
_worker_skew = None
_worker_depth = None


def _shutdown_worker():
    """Called at worker exit to cleanly quit the Stockfish engine."""
    global _worker_engine
    if _worker_engine is not None:
        try:
            _worker_engine.quit()
        except Exception:
            pass
        _worker_engine = None


def _init_worker(engine_path, depth, tokenizer, cls_id, sample_rate, skew_exponent):
    """Pool initializer: create one Stockfish engine per worker.

    If engine_path is None, workers fall back to material_eval. This lets
    tests exercise the parallel machinery without requiring Stockfish.
    """
    global _worker_engine, _worker_tokenizer, _worker_cls_id
    global _worker_sample_rate, _worker_skew, _worker_depth
    _worker_tokenizer = tokenizer
    _worker_cls_id = cls_id
    _worker_sample_rate = sample_rate
    _worker_skew = skew_exponent
    _worker_depth = depth
    if engine_path is not None:
        _worker_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        atexit.register(_shutdown_worker)
    else:
        _worker_engine = None


def _worker_eval(board: chess.Board) -> float:
    if _worker_engine is None:
        return material_eval(board)
    info = _worker_engine.analyse(board, chess.engine.Limit(depth=_worker_depth))
    score = info["score"].white()
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    return normalize_cp(score.score())


def _process_game(game_with_seed):
    """Worker task: parse, replay, sample, tokenize, and evaluate one game."""
    game, seed = game_with_seed
    movetext = game.get("movetext", "")
    if not movetext:
        return []
    move_sans = parse_movetext(movetext)
    if len(move_sans) < 2:
        return []

    eligible = list(range(len(move_sans)))
    num_positions = max(1, int(len(move_sans) * _worker_sample_rate))
    sample_indices = _weighted_sample(eligible, num_positions, _worker_skew, seed=seed)

    samples = []
    board = chess.Board()
    valid_moves = []
    for i, san in enumerate(move_sans):
        try:
            move = board.parse_san(san)
            board.push(move)
            valid_moves.append(move.uci())
        except (chess.InvalidMoveError, chess.AmbiguousMoveError):
            break

        if i in sample_indices:
            token_ids = [_worker_cls_id] + _worker_tokenizer.encode_moves(valid_moves)
            score = _worker_eval(board)
            samples.append((token_ids, score))

    return samples


def generate_samples_stockfish_parallel(
    games: list[dict],
    tokenizer: Tokenizer,
    num_workers: int = 8,
    stockfish_depth: int = 12,
    sample_rate: float = 0.25,
    skew_exponent: float = 1.5,
    engine_path: str | None = STOCKFISH_PATH,
    chunksize: int = 8,
    progress_every: int = 1000,
) -> list[tuple[list[int], float]]:
    """Parallel Stockfish-backed sample generation.

    Spawns `num_workers` processes, each owning one Stockfish subprocess.
    If `engine_path` is None, workers use material_eval instead of Stockfish
    (used by tests to verify the parallel machinery without the binary).

    Each game contributes `max(1, game_length * sample_rate)` positions,
    weighted toward mid/late game by `skew_exponent`. Sampling is seeded
    per-game-index for determinism across runs and worker counts.
    """
    cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
    tasks = [(game, idx) for idx, game in enumerate(games)]

    # spawn context: safest across macOS/Linux and avoids fork-safety issues
    # with chess.engine's subprocess.
    ctx = mp.get_context("spawn")
    samples: list[tuple[list[int], float]] = []
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(engine_path, stockfish_depth, tokenizer, cls_id, sample_rate, skew_exponent),
    ) as pool:
        for i, game_samples in enumerate(
            pool.imap_unordered(_process_game, tasks, chunksize=chunksize)
        ):
            samples.extend(game_samples)
            if progress_every and (i + 1) % progress_every == 0:
                print(
                    f"  processed {i + 1:,}/{len(games):,} games, "
                    f"{len(samples):,} positions..."
                )

    return samples


def collate_fn(batch):
    """Pad token sequences and create attention mask."""
    tokens, labels = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
    attention_mask = torch.ones(len(tokens), max_len, dtype=torch.bool)  # True = masked

    for i, t in enumerate(tokens):
        padded[i, :len(t)] = t
        attention_mask[i, :len(t)] = False

    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return padded, attention_mask, labels_tensor


def collate_fn_memmap(batch):
    """Collate pre-padded memmap samples — just stack, no per-batch padding needed."""
    tokens, masks, labels = zip(*batch)
    return torch.stack(tokens), torch.stack(masks), torch.tensor(labels, dtype=torch.float)


def collate_fn_policy(batch):
    """Pad token sequences for policy training — yields (tokens, mask), no labels."""
    max_len = max(len(t) for t in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True = padded
    for i, t in enumerate(batch):
        padded[i, :len(t)] = t
        mask[i, :len(t)] = False
    return padded, mask


class ChessPolicyDataset(Dataset):
    """Full game sequences for next-move prediction training.

    Each sample is [CLS, m1, m2, ..., mN] — the complete tokenized move list
    for one game, truncated to max_seq_len. Unlike ChessPositionDataset there
    are no external labels; targets are derived from the sequence itself in
    _run_epoch_policy.
    """
    def __init__(self, games: list[dict], tokenizer: Tokenizer, max_seq_len: int = 128):
        cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self._memmap = False
        self._train_idx: np.ndarray | None = None
        self.samples: list[list[int]] = []
        for game in games:
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
            move_ucis = move_ucis[:max_seq_len - 1]  # reserve slot for CLS
            self.samples.append([cls_id] + tokenizer.encode_moves(move_ucis))

    def __len__(self) -> int:
        if self._memmap:
            if self._train_idx is not None:
                return len(self._train_idx)
            return len(self._mm_lengths)
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._memmap:
            if self._train_idx is not None:
                idx = int(self._train_idx[idx])
            length = int(self._mm_lengths[idx])
            return torch.from_numpy(np.array(self._mm_tokens[idx, :length], dtype=np.int32)).long()
        return torch.tensor(self.samples[idx], dtype=torch.long)

    @classmethod
    def from_memmap(cls, out_dir: Path, tokenizer: Tokenizer, name: str = "policy"):
        """Load pre-tokenized policy sequences from memory-mapped arrays.

        Args:
            name: filename prefix; use 'puzzle' to load puzzle_*.bin files.

        If a sibling file `{name}_test_indices.npy` exists, those indices are
        excluded from this dataset — used to make training disjoint from the
        held-out test split that shares the same underlying .bin file.
        """
        meta = torch.load(out_dir / f"{name}_meta.pt", weights_only=True)
        n, max_len = meta["n"], meta["max_len"]
        inst = cls.__new__(cls)
        inst._memmap = True
        inst._mm_tokens = np.memmap(out_dir / f"{name}_tokens.bin", dtype=np.int32, mode="r", shape=(n, max_len))
        inst._mm_lengths = np.memmap(out_dir / f"{name}_lengths.bin", dtype=np.int32, mode="r", shape=(n,))
        inst._train_idx = _load_train_idx(out_dir, name, n)
        return inst


def _fmt_duration(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _amp_ctx(device):
    """BF16 autocast on CUDA, no-op elsewhere.

    BF16 is preferred over FP16 here: same dynamic range as FP32 (no GradScaler
    needed) and full tensor-core acceleration on Ampere+ / Ada / Blackwell.
    On Blackwell (RTX PRO 6000 / B200) this typically gives 2-3x training speedup
    on transformer matmuls.
    """
    dev = device if isinstance(device, str) else getattr(device, "type", "cpu")
    if dev == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _run_epoch_reward(model, loader, optimizer, device, writer, global_step, epoch_idx):
    """Single training epoch: MSE against Stockfish labels."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    epoch_start = time.time()

    for i, (batch_tokens, batch_mask, batch_labels) in enumerate(loader):
        batch_tokens = batch_tokens.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        with _amp_ctx(device):
            predictions = model(batch_tokens, attention_mask=batch_mask)
            loss = F.mse_loss(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar("train/reward_batch_loss", loss.item(), global_step)
        global_step += 1

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            elapsed = time.time() - epoch_start
            batches_done = i + 1
            eta = elapsed / batches_done * (n_batches - batches_done)
            samples_per_sec = batches_done * batch_tokens.size(0) / elapsed
            avg_so_far = total_loss / batches_done
            print(
                f"    batch {batches_done:,}/{n_batches:,}  "
                f"loss={avg_so_far:.4f}  "
                f"{samples_per_sec:,.0f} samples/s  "
                f"eta {_fmt_duration(eta)}"
            )

    epoch_elapsed = time.time() - epoch_start
    avg = total_loss / n_batches
    writer.add_scalar("train/reward_epoch_loss", avg, epoch_idx)
    return avg, global_step, epoch_elapsed

def _run_epoch_policy(model, loader, optimizer, device, writer, global_step, epoch_idx, pad_id):
    """Single training epoch: GPT-style cross-entropy at every position.

    For each sequence [CLS, m1, ..., mN], the model receives [CLS, m1, ..., m_{N-1}]
    and must predict [m1, m2, ..., mN] at every position simultaneously.
    PAD positions are excluded from the loss via ignore_index.

    Loader must yield (batch_tokens, batch_mask) — no external labels.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    epoch_start = time.time()

    for i, (batch_tokens, batch_mask) in enumerate(loader):
        batch_tokens = batch_tokens.to(device)
        batch_mask = batch_mask.to(device)

        # input: all tokens except last; target: all tokens except first (shifted by 1)
        input_tokens = batch_tokens[:, :-1]         # (batch, seq_len-1)
        input_mask = batch_mask[:, :-1]             # (batch, seq_len-1)
        targets = batch_tokens[:, 1:].contiguous()  # (batch, seq_len-1)

        with _amp_ctx(device):
            logits = model(input_tokens, attention_mask=input_mask)  # (batch, seq_len-1, vocab_size)
            # flatten to (batch*(seq_len-1), vocab_size) — ignore_index skips PAD targets
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar("train_policy/batch_loss", loss.item(), global_step)
        global_step += 1

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            elapsed = time.time() - epoch_start
            batches_done = i + 1
            eta = elapsed / batches_done * (n_batches - batches_done)
            samples_per_sec = batches_done * batch_tokens.size(0) / elapsed
            avg_so_far = total_loss / batches_done
            print(
                f"    batch {batches_done:,}/{n_batches:,}  "
                f"loss={avg_so_far:.4f}  "
                f"{samples_per_sec:,.0f} samples/s  "
                f"eta {_fmt_duration(eta)}"
            )

    epoch_elapsed = time.time() - epoch_start
    avg = total_loss / n_batches
    writer.add_scalar("train_policy/epoch_loss", avg, epoch_idx)
    return avg, global_step, epoch_elapsed


def _run_epoch_policy_puzzle(model, loader, optimizer, device, writer, global_step, epoch_idx, pad_id):
    """Puzzle fine-tuning epoch: cross-entropy on solver and opp-defensive moves.

    Sequence layout: [CLS, S, m1, opp1, m2, opp2, m3, ...]
    Only the setup move S is masked from loss — it's the opponent's forcing move
    that defines the puzzle, given as context (P[m_n | S, m_{<n}]).

    Solver moves (m1, m2, ...) and Lichess's chosen opponent responses (opp1,
    opp2, ...) both contribute to loss. The opp moves aren't arbitrary — Lichess
    picks the engine's best defensive continuation (the move that delays loss
    longest or holds the position), so training on them teaches the model what
    strong defensive play looks like. That structural signal is genuinely useful
    and doesn't appear cleanly in full-game training data.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    epoch_start = time.time()

    for i, (batch_tokens, batch_mask) in enumerate(loader):
        batch_tokens = batch_tokens.to(device)
        batch_mask = batch_mask.to(device)

        input_tokens = batch_tokens[:, :-1]
        input_mask = batch_mask[:, :-1]
        targets = batch_tokens[:, 1:].contiguous()
        targets[:, 0] = pad_id  # mask only the setup move (S) from loss

        with _amp_ctx(device):
            logits = model(input_tokens, attention_mask=input_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar("train_puzzle/batch_loss", loss.item(), global_step)
        global_step += 1

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            elapsed = time.time() - epoch_start
            batches_done = i + 1
            eta = elapsed / batches_done * (n_batches - batches_done)
            samples_per_sec = batches_done * batch_tokens.size(0) / elapsed
            avg_so_far = total_loss / batches_done
            print(
                f"    batch {batches_done:,}/{n_batches:,}  "
                f"loss={avg_so_far:.4f}  "
                f"{samples_per_sec:,.0f} samples/s  "
                f"eta {_fmt_duration(eta)}"
            )

    epoch_elapsed = time.time() - epoch_start
    avg = total_loss / n_batches
    writer.add_scalar("train_puzzle/epoch_loss", avg, epoch_idx)
    return avg, global_step, epoch_elapsed


def eval_reward(model, loader, device) -> dict:
    """Evaluate reward model on a test loader. Returns MSE, MAE, and Pearson r."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), _amp_ctx(device):
        for batch_tokens, batch_mask, batch_labels in loader:
            preds = model(batch_tokens.to(device), attention_mask=batch_mask.to(device))
            all_preds.append(preds.float().cpu())
            all_labels.append(batch_labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    mse = F.mse_loss(preds, labels).item()
    mae = (preds - labels).abs().mean().item()
    # Pearson r
    p_centered = preds - preds.mean()
    l_centered = labels - labels.mean()
    denom = (p_centered.norm() * l_centered.norm()).clamp(min=1e-8)
    pearson_r = (p_centered * l_centered).sum() / denom
    return {"mse": mse, "mae": mae, "pearson_r": pearson_r.item()}


def eval_policy(model, loader, device, pad_id: int) -> dict:
    """Evaluate policy model on a test loader. Returns loss, perplexity, top-1/top-5 acc."""
    model.eval()
    total_loss = 0.0
    total_correct1 = 0
    total_correct5 = 0
    total_positions = 0
    with torch.no_grad(), _amp_ctx(device):
        for batch_tokens, batch_mask in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            input_tokens = batch_tokens[:, :-1]
            input_mask = batch_mask[:, :-1]
            targets = batch_tokens[:, 1:].contiguous()
            logits = model(input_tokens, attention_mask=input_mask)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            valid = flat_targets != pad_id
            total_loss += F.cross_entropy(flat_logits, flat_targets, ignore_index=pad_id, reduction="sum").item()
            total_positions += valid.sum().item()
            top5 = flat_logits[valid].topk(5, dim=-1).indices
            valid_targets = flat_targets[valid]
            total_correct1 += (top5[:, 0] == valid_targets).sum().item()
            total_correct5 += (top5 == valid_targets.unsqueeze(1)).any(dim=1).sum().item()
    avg_loss = total_loss / max(total_positions, 1)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20)),
        "top1_acc": total_correct1 / max(total_positions, 1),
        "top5_acc": total_correct5 / max(total_positions, 1),
    }


def eval_puzzle_solve_rate(model, loader, device, pad_id: int) -> dict:
    """Evaluate puzzle solve rate: % of solver positions where model's top-1 matches
    ground truth. Sequence layout: [CLS, setup, solver1, opp1, solver2, ...]

    Solver moves are at token positions 2, 4, 6, ... (logit positions 1, 3, 5, ...).
    The setup move at token position 1 (logit 0) is excluded — it's context, not a
    prediction target. Also reports first-move solve rate (logit position 1 only).
    """
    model.eval()
    first_correct = 0
    first_total = 0
    all_correct = 0
    all_total = 0
    with torch.no_grad(), _amp_ctx(device):
        for batch_tokens, batch_mask in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            input_tokens = batch_tokens[:, :-1]
            input_mask = batch_mask[:, :-1]
            logits = model(input_tokens, attention_mask=input_mask)
            seq_len = batch_tokens.size(1)
            # Solver logit positions: 1, 3, 5, ... → target positions: 2, 4, 6, ...
            for solver_logit_pos in range(1, seq_len - 1, 2):
                solver_token_pos = solver_logit_pos + 1
                if solver_token_pos >= seq_len:
                    break
                targets = batch_tokens[:, solver_token_pos]
                valid = targets != pad_id
                if not valid.any():
                    continue
                preds = logits[:, solver_logit_pos].argmax(dim=-1)
                correct = (preds[valid] == targets[valid]).sum().item()
                n_valid = valid.sum().item()
                all_correct += correct
                all_total += n_valid
                if solver_logit_pos == 1:
                    first_correct += correct
                    first_total += n_valid
    return {
        "first_move_solve_rate": first_correct / max(first_total, 1),
        "all_moves_solve_rate": all_correct / max(all_total, 1),
    }


def train(
    tokenizer_path,
    stockfish_samples_path,
    outcome_games_path,
    epochs,
    policy_epochs,
    batch_size,
    learning_rate,
    max_seq_len,
    log_dir,
    num_workers,
    puzzle_data_dir=None,
    puzzle_epochs=5,
):
    """Train the reward model then the policy model.

    Phase 1: MSE on Stockfish-labeled positions (reward model).
    Phase 2: Cross-entropy next-move prediction on full game sequences (policy model).

    Requires stockfish memmap files and outcome games built by src/build_datasets.py.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = "bfloat16 autocast" if device == "cuda" else "fp32 (CPU)"
    print(f"Using device: {device} ({amp_dtype})")

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    vocab_size = tokenizer.language_size
    pad_id = tokenizer.symbol_to_token[PAD_TOKEN]

    writer = SummaryWriter(log_dir=log_dir)

    # ── Test loaders (optional, skip silently if test sets not built yet) ───────
    out_dir = Path(stockfish_samples_path).parent
    reward_test_loader = None
    if (out_dir / "stockfish_test_meta.pt").exists():
        reward_test_ds = ChessPositionDataset.from_memmap(out_dir, "stockfish_test", tokenizer)
        reward_test_loader = DataLoader(
            reward_test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_memmap,
            num_workers=num_workers, pin_memory=True,
        )
        print(f"Reward test set: {len(reward_test_ds):,} positions loaded")

    policy_data_dir_early = Path(outcome_games_path).parent
    policy_test_loader = None
    if (policy_data_dir_early / "policy_test_meta.pt").exists():
        policy_test_ds = ChessPolicyDataset.from_memmap(policy_data_dir_early, tokenizer, name="policy_test")
        policy_test_loader = DataLoader(
            policy_test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_policy,
            num_workers=num_workers, pin_memory=True,
        )
        print(f"Policy test set: {len(policy_test_ds):,} sequences loaded")

    puzzle_test_loader = None
    _puzzle_dir = Path(puzzle_data_dir) if puzzle_data_dir is not None else policy_data_dir_early
    if (_puzzle_dir / "puzzle_test_meta.pt").exists():
        puzzle_test_ds = ChessPolicyDataset.from_memmap(_puzzle_dir, tokenizer, name="puzzle_test")
        puzzle_test_loader = DataLoader(
            puzzle_test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_policy,
            num_workers=num_workers, pin_memory=True,
        )
        print(f"Puzzle test set: {len(puzzle_test_ds):,} sequences loaded")

    # ── Phase 1: reward model ────────────────────────────────────────────────
    global_step = 0
    reward_model = None
    if Path("reward_model.pt").exists():
        print("\n── Phase 1: reward_model.pt already exists, skipping reward training")
    else:
        sf_meta = out_dir / "stockfish_meta.pt"
        if sf_meta.exists():
            print(f"Loading Stockfish samples from memmap ({out_dir}/stockfish_*)...")
            sf_ds = ChessPositionDataset.from_memmap(out_dir, "stockfish", tokenizer)
            sf_collate = collate_fn_memmap
        else:
            print(f"Loading Stockfish samples from {stockfish_samples_path}...")
            sf_ds = ChessPositionDataset.from_file(stockfish_samples_path, tokenizer)
            sf_collate = collate_fn
        print(f"Reward dataset: {len(sf_ds):,} positions")

        sf_loader = DataLoader(
            sf_ds, batch_size=batch_size, shuffle=True, collate_fn=sf_collate,
            num_workers=num_workers, pin_memory=True,
        )

        reward_model = ChessRewardModel(vocab_size=vocab_size, max_seq_len=max_seq_len).to(device)
        reward_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)

        print(f"\n── Phase 1: reward model — {epochs} epochs, lr={learning_rate}")
        phase1_start = time.time()
        for epoch in range(epochs):
            epoch_num = epoch + 1
            print(f"  [epoch {epoch_num}/{epochs}] starting...")
            avg_loss, global_step, epoch_secs = _run_epoch_reward(
                reward_model, sf_loader, reward_optimizer, device, writer, global_step, epoch
            )
            epochs_left = epochs - epoch_num
            print(
                f"  [epoch {epoch_num}/{epochs}]  "
                f"loss={avg_loss:.4f}  "
                f"epoch_time={_fmt_duration(epoch_secs)}  "
                f"eta={_fmt_duration(epoch_secs * epochs_left)}"
            )
            if reward_test_loader is not None:
                m = eval_reward(reward_model, reward_test_loader, device)
                writer.add_scalar("test/reward_mse", m["mse"], epoch_num)
                writer.add_scalar("test/reward_mae", m["mae"], epoch_num)
                writer.add_scalar("test/reward_pearson_r", m["pearson_r"], epoch_num)
                print(
                    f"  [test]  mse={m['mse']:.4f}  mae={m['mae']:.4f}  r={m['pearson_r']:.4f}"
                )

        print(f"Phase 1 complete in {_fmt_duration(time.time() - phase1_start)}")
        torch.save(reward_model.state_dict(), "reward_model.pt")
        print("Reward model saved to reward_model.pt")

    # ── Phase 2a: policy model on game sequences ─────────────────────────────
    policy_data_dir = policy_data_dir_early  # already computed above
    policy_meta = policy_data_dir / "policy_meta.pt"
    if policy_meta.exists():
        print(f"Loading policy sequences from memmap ({policy_data_dir}/policy_*)...")
        policy_ds = ChessPolicyDataset.from_memmap(policy_data_dir, tokenizer)
    else:
        print(f"Loading outcome games from {outcome_games_path} (tokenizing on-the-fly)...")
        outcome_games = torch.load(outcome_games_path, weights_only=False)
        policy_ds = ChessPolicyDataset(outcome_games, tokenizer, max_seq_len=max_seq_len)
    print(f"Policy dataset: {len(policy_ds):,} sequences")

    policy_loader = DataLoader(
        policy_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_policy,
        num_workers=num_workers, pin_memory=True,
    )

    policy_model = ChessPolicyModel(vocab_size=vocab_size, max_seq_len=max_seq_len).to(device)
    policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    global_step = 0

    def _run_policy_test(epoch_num: int, tb_prefix: str) -> None:
        if policy_test_loader is not None:
            m = eval_policy(policy_model, policy_test_loader, device, pad_id)
            writer.add_scalar(f"{tb_prefix}/policy_loss", m["loss"], epoch_num)
            writer.add_scalar(f"{tb_prefix}/policy_perplexity", m["perplexity"], epoch_num)
            writer.add_scalar(f"{tb_prefix}/policy_top1_acc", m["top1_acc"], epoch_num)
            writer.add_scalar(f"{tb_prefix}/policy_top5_acc", m["top5_acc"], epoch_num)
            print(
                f"  [policy test]  loss={m['loss']:.4f}  ppl={m['perplexity']:.2f}"
                f"  top1={m['top1_acc']:.3f}  top5={m['top5_acc']:.3f}"
            )
        if puzzle_test_loader is not None:
            m = eval_puzzle_solve_rate(policy_model, puzzle_test_loader, device, pad_id)
            writer.add_scalar(f"{tb_prefix}/puzzle_first_move", m["first_move_solve_rate"], epoch_num)
            writer.add_scalar(f"{tb_prefix}/puzzle_all_moves", m["all_moves_solve_rate"], epoch_num)
            print(
                f"  [puzzle test]  first_move={m['first_move_solve_rate']:.3f}"
                f"  all_moves={m['all_moves_solve_rate']:.3f}"
            )

    print(f"\n── Phase 2a: policy model (games) — {policy_epochs} epochs, lr={learning_rate}")
    phase2a_start = time.time()
    for epoch in range(policy_epochs):
        epoch_num = epoch + 1
        print(f"  [epoch {epoch_num}/{policy_epochs}] starting...")
        avg_loss, global_step, epoch_secs = _run_epoch_policy(
            policy_model, policy_loader, policy_optimizer, device, writer, global_step, epoch, pad_id
        )
        epochs_left = policy_epochs - epoch_num
        print(
            f"  [epoch {epoch_num}/{policy_epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"epoch_time={_fmt_duration(epoch_secs)}  "
            f"eta={_fmt_duration(epoch_secs * epochs_left)}"
        )
        _run_policy_test(epoch_num, "test_2a")

    print(f"Phase 2a complete in {_fmt_duration(time.time() - phase2a_start)}")
    torch.save(policy_model.state_dict(), "policy_model.pt")
    print("Policy model saved to policy_model.pt")

    # ── Phase 2b: fine-tune policy model on puzzle sequences ─────────────────
    if puzzle_data_dir is not None and puzzle_epochs > 0:
        pdir = Path(puzzle_data_dir)
        puzzle_train_meta = pdir / "puzzle_meta.pt"
        if puzzle_train_meta.exists():
            puzzle_ds = ChessPolicyDataset.from_memmap(pdir, tokenizer, name="puzzle")
            print(f"\nPuzzle training set: {len(puzzle_ds):,} sequences")
            puzzle_loader = DataLoader(
                puzzle_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_policy,
                num_workers=num_workers, pin_memory=True,
            )
            puzzle_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

            print(f"\n── Phase 2b: puzzle fine-tune — {puzzle_epochs} epochs, lr={learning_rate}")
            phase2b_start = time.time()
            for epoch in range(puzzle_epochs):
                epoch_num = epoch + 1
                print(f"  [epoch {epoch_num}/{puzzle_epochs}] starting...")
                avg_loss, global_step, epoch_secs = _run_epoch_policy_puzzle(
                    policy_model, puzzle_loader, puzzle_optimizer, device, writer, global_step,
                    policy_epochs + epoch, pad_id,
                )
                epochs_left = puzzle_epochs - epoch_num
                print(
                    f"  [epoch {epoch_num}/{puzzle_epochs}]  "
                    f"loss={avg_loss:.4f}  "
                    f"epoch_time={_fmt_duration(epoch_secs)}  "
                    f"eta={_fmt_duration(epoch_secs * epochs_left)}"
                )
                _run_policy_test(epoch_num, "test_2b")

            print(f"Phase 2b complete in {_fmt_duration(time.time() - phase2b_start)}")
            torch.save(policy_model.state_dict(), "policy_model.pt")
            print("Policy model (puzzle fine-tuned) saved to policy_model.pt")
        else:
            print(f"WARNING: --puzzle-data given but {puzzle_train_meta} not found; skipping Phase 2b.")

    return reward_model, policy_model, tokenizer


def _build_argparser():
    p = argparse.ArgumentParser(description=train.__doc__)
    p.add_argument("--tokenizer-path", default="data/tokenizer.pt")
    p.add_argument("--stockfish-samples-path", default="data/stockfish_samples.pt")
    p.add_argument("--outcome-games-path", default="data/games_outcome.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--policy-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--log-dir", default="runs/chess_models")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--puzzle-data", default=None, dest="puzzle_data_dir",
        help="Directory containing puzzle_tokens.bin / puzzle_lengths.bin / puzzle_meta.pt")
    p.add_argument("--puzzle-epochs", type=int, default=5, dest="puzzle_epochs",
        help="Number of fine-tuning epochs on puzzle sequences in Phase 2b (default 5)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train(
        tokenizer_path=args.tokenizer_path,
        stockfish_samples_path=args.stockfish_samples_path,
        outcome_games_path=args.outcome_games_path,
        epochs=args.epochs,
        policy_epochs=args.policy_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
        puzzle_data_dir=args.puzzle_data_dir,
        puzzle_epochs=args.puzzle_epochs,
    )
