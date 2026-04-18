import argparse
import atexit
import copy
import math
import multiprocessing as mp
import re
import random
import shutil
import time
import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from tokenizer import Tokenizer
from model import ChessRewardModel, DummyRewardModel, CLS_TOKEN, PAD_TOKEN

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


def build_tokenizer_from_games(games: list[dict], max_language_size: int = 2000) -> Tokenizer:
    """Build a move-level tokenizer from filtered HuggingFace games."""
    all_moves = []
    for game in games:
        movetext = game.get("movetext", "")
        if not movetext:
            continue
        all_moves.extend(parse_movetext(movetext))

    tokenizer = Tokenizer()
    unique_count = len(set(all_moves))
    lang_size = max(unique_count, max_language_size)
    tokenizer.train_tokenizer(all_moves, max_language_size=lang_size)
    tokenizer.add_special_tokens([CLS_TOKEN, PAD_TOKEN])
    return tokenizer


class ChessPositionDataset(Dataset):
    def __init__(
        self,
        games: list[dict],
        tokenizer: Tokenizer,
        eval_fn=material_eval,
        max_positions_per_game: int = 10,
        skip_ply: int = 0,
    ):
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.samples: list[tuple[list[int], float]] = []
        self._generate_samples(games, eval_fn, max_positions_per_game, skip_ply)

    def _generate_samples(self, games, eval_fn, max_positions_per_game, skip_ply):
        for idx, game in enumerate(games):
            movetext = game.get("movetext", "")
            if not movetext:
                continue
            move_sans = parse_movetext(movetext)
            if len(move_sans) < max(2, skip_ply + 1):
                continue

            board = chess.Board()
            eligible = list(range(skip_ply, len(move_sans)))
            num_positions = min(max_positions_per_game, len(eligible))
            # Per-game deterministic sampling (seeded by game index) so serial
            # and parallel paths produce identical sample sets for the same input.
            rng = random.Random(idx)
            sample_indices = set(rng.sample(eligible, num_positions))

            valid_moves = []
            for i, san in enumerate(move_sans):
                try:
                    move = board.parse_san(san)
                    board.push(move)
                    valid_moves.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                    break

                if i in sample_indices:
                    try:
                        token_ids = [self.cls_id] + self.tokenizer.encode_moves(valid_moves)
                    except KeyError:
                        continue
                    score = eval_fn(board)
                    self.samples.append((token_ids, score))

            if (idx + 1) % 10_000 == 0:
                print(f"  processed {idx + 1:,} games, {len(self.samples):,} positions...")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        token_ids, score = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), score

    @classmethod
    def from_samples(cls, samples, tokenizer: Tokenizer):
        """Build a dataset from pre-generated (token_ids, score) samples."""
        inst = cls.__new__(cls)
        inst.tokenizer = tokenizer
        inst.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        inst.samples = list(samples)
        return inst

    @classmethod
    def from_file(cls, samples_path: str, tokenizer: Tokenizer):
        """Load (token_ids, score) samples from a torch.save file.

        Expected format: a list[tuple[list[int], float]] pickled with torch.save.
        """
        samples = torch.load(samples_path, weights_only=False)
        return cls.from_samples(samples, tokenizer)


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
_worker_max_positions = None
_worker_depth = None
_worker_skip_ply = 0


def _shutdown_worker():
    """Called at worker exit to cleanly quit the Stockfish engine."""
    global _worker_engine
    if _worker_engine is not None:
        try:
            _worker_engine.quit()
        except Exception:
            pass
        _worker_engine = None


def _init_worker(engine_path, depth, tokenizer, cls_id, max_positions_per_game, skip_ply):
    """Pool initializer: create one Stockfish engine per worker.

    If engine_path is None, workers fall back to material_eval. This lets
    tests exercise the parallel machinery without requiring Stockfish.
    """
    global _worker_engine, _worker_tokenizer, _worker_cls_id
    global _worker_max_positions, _worker_depth, _worker_skip_ply
    _worker_tokenizer = tokenizer
    _worker_cls_id = cls_id
    _worker_max_positions = max_positions_per_game
    _worker_depth = depth
    _worker_skip_ply = skip_ply
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
    if len(move_sans) < max(2, _worker_skip_ply + 1):
        return []

    rng = random.Random(seed)
    eligible = list(range(_worker_skip_ply, len(move_sans)))
    num_positions = min(_worker_max_positions, len(eligible))
    sample_indices = set(rng.sample(eligible, num_positions))

    samples = []
    board = chess.Board()
    valid_moves = []
    for i, san in enumerate(move_sans):
        try:
            move = board.parse_san(san)
            board.push(move)
            valid_moves.append(san)
        except (chess.InvalidMoveError, chess.AmbiguousMoveError):
            break

        if i in sample_indices:
            try:
                token_ids = [_worker_cls_id] + _worker_tokenizer.encode_moves(valid_moves)
            except KeyError:
                continue  # rare SAN not in vocab — skip this position
            score = _worker_eval(board)
            samples.append((token_ids, score))

    return samples


def generate_samples_stockfish_parallel(
    games: list[dict],
    tokenizer: Tokenizer,
    num_workers: int = 8,
    stockfish_depth: int = 12,
    max_positions_per_game: int = 10,
    engine_path: str | None = STOCKFISH_PATH,
    chunksize: int = 8,
    progress_every: int = 1000,
    skip_ply: int = 0,
) -> list[tuple[list[int], float]]:
    """Parallel Stockfish-backed sample generation.

    Spawns `num_workers` processes, each owning one Stockfish subprocess.
    If `engine_path` is None, workers use material_eval instead of Stockfish
    (used by tests to verify the parallel machinery without the binary).

    `skip_ply` drops the first N plies of each game before sampling (useful
    for discarding opening-theory noise where position labels carry weak
    signal).

    Sampling is seeded per-game-index, so the returned sample set is
    deterministic across runs and across worker counts (though the order
    is not, since we use imap_unordered).
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
        initargs=(engine_path, stockfish_depth, tokenizer, cls_id, max_positions_per_game, skip_ply),
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


def _fmt_duration(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _run_epoch_phase1(model, loader, optimizer, device, writer, global_step, epoch_idx):
    """Single epoch of phase-1 outcome MSE training."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    epoch_start = time.time()

    for i, (batch_tokens, batch_mask, batch_labels) in enumerate(loader):
        batch_tokens = batch_tokens.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_tokens, attention_mask=batch_mask)
        loss = F.mse_loss(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        writer.add_scalar("phase1/batch_loss", loss.item(), global_step)
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
    writer.add_scalar("phase1/epoch_loss", avg, epoch_idx)
    return avg, global_step, epoch_elapsed


def _run_epoch_phase2(
    model, teacher, loader, optimizer, device, writer, global_step, epoch_idx, distill_lambda
):
    """Single epoch of phase-2 Stockfish training with distillation regularizer.

    loss = MSE(pred, stockfish_label) + λ * MSE(pred, teacher(x).detach())
    """
    model.train()
    total_task = total_distill = total_combined = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)
    epoch_start = time.time()

    for i, (batch_tokens, batch_mask, batch_labels) in enumerate(loader):
        batch_tokens = batch_tokens.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_tokens, attention_mask=batch_mask)
        with torch.no_grad():
            teacher_preds = teacher(batch_tokens, attention_mask=batch_mask)

        task_loss = F.mse_loss(predictions, batch_labels)
        distill_loss = F.mse_loss(predictions, teacher_preds)
        loss = task_loss + distill_lambda * distill_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_task += task_loss.item()
        total_distill += distill_loss.item()
        total_combined += loss.item()

        writer.add_scalar("phase2/batch_task_mse", task_loss.item(), global_step)
        writer.add_scalar("phase2/batch_distill", distill_loss.item(), global_step)
        writer.add_scalar("phase2/batch_total", loss.item(), global_step)
        global_step += 1

        if (i + 1) % log_every == 0 or (i + 1) == n_batches:
            elapsed = time.time() - epoch_start
            batches_done = i + 1
            eta = elapsed / batches_done * (n_batches - batches_done)
            samples_per_sec = batches_done * batch_tokens.size(0) / elapsed
            print(
                f"    batch {batches_done:,}/{n_batches:,}  "
                f"task={total_task/batches_done:.4f}  "
                f"distill={total_distill/batches_done:.4f}  "
                f"{samples_per_sec:,.0f} samples/s  "
                f"eta {_fmt_duration(eta)}"
            )

    epoch_elapsed = time.time() - epoch_start
    n = n_batches
    avg_task = total_task / n
    avg_distill = total_distill / n
    avg_total = total_combined / n
    writer.add_scalar("phase2/epoch_task_mse", avg_task, epoch_idx)
    writer.add_scalar("phase2/epoch_distill", avg_distill, epoch_idx)
    writer.add_scalar("phase2/epoch_total", avg_total, epoch_idx)
    return avg_task, avg_distill, avg_total, global_step, epoch_elapsed


def train(
    tokenizer_path: str = "data/tokenizer.pt",
    outcome_samples_path: str = "data/outcome_samples.pt",
    stockfish_samples_path: str = "data/stockfish_samples.pt",
    phase1_epochs: int = 10,
    phase2_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 3e-5,
    phase2_lr: float = 1e-5,
    distill_lambda: float = 0.05,
    max_seq_len: int = 512,
    log_dir: str = "runs/chess_reward_model",
):
    """Two-phase hybrid training.

    Phase 1: MSE against game-outcome labels ({+1, 0, -1}) on a large dataset.
    Phase 2: MSE against Stockfish labels + λ·MSE-to-frozen-phase-1-teacher
             on a smaller disjoint dataset.

    Both datasets must be built first by src/build_datasets.py.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    vocab_size = tokenizer.language_size

    # -------------------- Phase 1: outcome pretraining -----------------------
    print(f"Loading outcome samples from {outcome_samples_path}...")
    outcome_ds = ChessPositionDataset.from_file(outcome_samples_path, tokenizer)
    print(f"Phase 1 dataset size: {len(outcome_ds):,} positions")

    outcome_loader = DataLoader(
        outcome_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=8, pin_memory=True,
    )

    model = ChessRewardModel(vocab_size=vocab_size, max_seq_len=max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    print(f"\nPhase 1: {phase1_epochs} epochs, lr={learning_rate}")
    phase1_start = time.time()
    for epoch in range(phase1_epochs):
        epoch_num = epoch + 1
        print(f"  [phase1] epoch {epoch_num}/{phase1_epochs} starting...")
        avg_loss, global_step, epoch_secs = _run_epoch_phase1(
            model, outcome_loader, optimizer, device, writer, global_step, epoch
        )
        epochs_left = phase1_epochs - epoch_num
        eta = epoch_secs * epochs_left
        print(
            f"  [phase1] epoch {epoch_num}/{phase1_epochs}  "
            f"loss={avg_loss:.4f}  "
            f"epoch_time={_fmt_duration(epoch_secs)}  "
            f"eta={_fmt_duration(eta)}"
        )

    phase1_total = time.time() - phase1_start
    print(f"Phase 1 complete in {_fmt_duration(phase1_total)}")
    torch.save(model.state_dict(), "reward_model_phase1.pt")
    print("Phase 1 checkpoint saved to reward_model_phase1.pt")

    # -------------------- Phase 2: Stockfish fine-tuning ---------------------
    print(f"\nLoading Stockfish samples from {stockfish_samples_path}...")
    sf_ds = ChessPositionDataset.from_file(stockfish_samples_path, tokenizer)
    print(f"Phase 2 dataset size: {len(sf_ds):,} positions")

    sf_loader = DataLoader(
        sf_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=8, pin_memory=True,
    )

    # Frozen teacher — a deep copy of the phase-1 model held in eval mode.
    # Gradients disabled so teacher forward passes are cheap and don't leak grads.
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Fresh optimizer for phase 2 with a lower LR so Stockfish fine-tuning
    # doesn't clobber phase-1 representations.
    phase2_optimizer = torch.optim.AdamW(model.parameters(), lr=phase2_lr)

    print(
        f"\nPhase 2: {phase2_epochs} epochs, lr={phase2_lr}, "
        f"distill_lambda={distill_lambda}"
    )
    phase2_start = time.time()
    for epoch in range(phase2_epochs):
        epoch_num = epoch + 1
        print(f"  [phase2] epoch {epoch_num}/{phase2_epochs} starting...")
        avg_task, avg_distill, avg_total, global_step, epoch_secs = _run_epoch_phase2(
            model, teacher, sf_loader, phase2_optimizer, device,
            writer, global_step, epoch, distill_lambda,
        )
        epochs_left = phase2_epochs - epoch_num
        eta = epoch_secs * epochs_left
        print(
            f"  [phase2] epoch {epoch_num}/{phase2_epochs}  "
            f"task_mse={avg_task:.4f}  distill={avg_distill:.4f}  total={avg_total:.4f}  "
            f"epoch_time={_fmt_duration(epoch_secs)}  eta={_fmt_duration(eta)}"
        )

    phase2_total = time.time() - phase2_start
    print(f"Phase 2 complete in {_fmt_duration(phase2_total)}")
    writer.close()

    torch.save(model.state_dict(), "reward_model.pt")
    print("\nFinal model saved to reward_model.pt")
    return model, tokenizer


def _build_argparser():
    p = argparse.ArgumentParser(description=train.__doc__)
    p.add_argument("--tokenizer-path", default="data/tokenizer.pt")
    p.add_argument("--outcome-samples-path", default="data/outcome_samples.pt")
    p.add_argument("--stockfish-samples-path", default="data/stockfish_samples.pt")
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase2-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--phase2-lr", type=float, default=1e-5)
    p.add_argument("--distill-lambda", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--log-dir", default="runs/chess_reward_model")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    train(
        tokenizer_path=args.tokenizer_path,
        outcome_samples_path=args.outcome_samples_path,
        stockfish_samples_path=args.stockfish_samples_path,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        phase2_lr=args.phase2_lr,
        distill_lambda=args.distill_lambda,
        max_seq_len=args.max_seq_len,
        log_dir=args.log_dir,
    )
