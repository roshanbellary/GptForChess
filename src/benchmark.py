"""Evaluate trained reward and policy models on held-out test sets.

Test sets are built by src/build_datasets.py (run once before benchmarking):
  stockfish_test_*.bin  — 50K Stockfish-labeled positions  (reward model)
  policy_test_*.bin     — 50K game sequences               (policy model)
  puzzle_test_*.bin     — 100K puzzle sequences            (puzzle solve rate)

Metrics:
  Reward:  MSE, MAE, Pearson r
  Policy:  loss, perplexity, top-1 move accuracy, top-5 move accuracy
  Puzzle:  first-move solve rate (top-1), all-moves solve rate

Usage:
  arch -arm64 poetry run python src/benchmark.py
  arch -arm64 poetry run python src/benchmark.py \\
    --reward-model reward_model.pt \\
    --policy-model policy_model.pt \\
    --data-dir data/ \\
    --batch-size 512
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import ChessRewardModel, ChessPolicyModel, PAD_TOKEN
from train import (
    ChessPositionDataset,
    ChessPolicyDataset,
    collate_fn_memmap,
    collate_fn_policy,
    eval_reward,
    eval_policy,
    eval_puzzle_solve_rate,
)


def _fmt(v: float, pct: bool = False) -> str:
    return f"{v * 100:.2f}%" if pct else f"{v:.4f}"


def run_benchmark(
    data_dir: Path,
    reward_model_path: str | None,
    policy_model_path: str | None,
    batch_size: int = 512,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    """Run all available benchmarks and return a results dict.

    Returns a dict with keys 'reward', 'policy', 'puzzle' (each a sub-dict of
    metrics), or omits a key if the test set / model is unavailable.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmark device: {device}")

    tokenizer_path = data_dir / "tokenizer.pt"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = torch.load(tokenizer_path, weights_only=False)
    vocab_size = tokenizer.language_size
    pad_id = tokenizer.symbol_to_token[PAD_TOKEN]

    results = {}

    # ── Reward model ──────────────────────────────────────────────────────────
    reward_test_meta = data_dir / "stockfish_test_meta.pt"
    if reward_model_path and Path(reward_model_path).exists() and reward_test_meta.exists():
        print(f"\nLoading reward model from {reward_model_path}...")
        reward_model = ChessRewardModel(vocab_size=vocab_size).to(device)
        reward_model.load_state_dict(torch.load(reward_model_path, map_location=device, weights_only=True))

        print("Loading reward test set...")
        reward_test_ds = ChessPositionDataset.from_memmap(data_dir, "stockfish_test", tokenizer)
        reward_test_loader = DataLoader(
            reward_test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn_memmap, num_workers=num_workers, pin_memory=True,
        )
        print(f"  {len(reward_test_ds):,} test positions")

        t0 = time.time()
        m = eval_reward(reward_model, reward_test_loader, device)
        print(
            f"  Reward  |  MSE={_fmt(m['mse'])}  MAE={_fmt(m['mae'])}"
            f"  Pearson r={_fmt(m['pearson_r'])}  ({time.time()-t0:.1f}s)"
        )
        results["reward"] = m
    elif not reward_test_meta.exists():
        print("\nReward test set not found (run build_datasets.py to create it). Skipping.")
    elif not reward_model_path or not Path(reward_model_path).exists():
        print(f"\nReward model not found at {reward_model_path}. Skipping.")

    # ── Policy model ──────────────────────────────────────────────────────────
    policy_test_meta = data_dir / "policy_test_meta.pt"
    puzzle_test_meta = data_dir / "puzzle_test_meta.pt"
    policy_model = None

    if policy_model_path and Path(policy_model_path).exists():
        print(f"\nLoading policy model from {policy_model_path}...")
        policy_model = ChessPolicyModel(vocab_size=vocab_size).to(device)
        policy_model.load_state_dict(torch.load(policy_model_path, map_location=device, weights_only=True))

    if policy_model is not None and policy_test_meta.exists():
        print("Loading policy test set...")
        policy_test_ds = ChessPolicyDataset.from_memmap(data_dir, tokenizer, name="policy_test")
        policy_test_loader = DataLoader(
            policy_test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn_policy, num_workers=num_workers, pin_memory=True,
        )
        print(f"  {len(policy_test_ds):,} test sequences")

        t0 = time.time()
        m = eval_policy(policy_model, policy_test_loader, device, pad_id)
        print(
            f"  Policy  |  loss={_fmt(m['loss'])}  ppl={m['perplexity']:.2f}"
            f"  top1={_fmt(m['top1_acc'], pct=True)}  top5={_fmt(m['top5_acc'], pct=True)}"
            f"  ({time.time()-t0:.1f}s)"
        )
        results["policy"] = m
    elif policy_model is None:
        print(f"\nPolicy model not found at {policy_model_path}. Skipping policy + puzzle eval.")
    elif not policy_test_meta.exists():
        print("\nPolicy test set not found (run build_datasets.py to create it). Skipping.")

    if policy_model is not None and puzzle_test_meta.exists():
        print("Loading puzzle test set...")
        puzzle_test_ds = ChessPolicyDataset.from_memmap(data_dir, tokenizer, name="puzzle_test")
        puzzle_test_loader = DataLoader(
            puzzle_test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn_policy, num_workers=num_workers, pin_memory=True,
        )
        print(f"  {len(puzzle_test_ds):,} test puzzles")

        t0 = time.time()
        m = eval_puzzle_solve_rate(policy_model, puzzle_test_loader, device, pad_id)
        print(
            f"  Puzzle  |  first_move={_fmt(m['first_move_solve_rate'], pct=True)}"
            f"  all_moves={_fmt(m['all_moves_solve_rate'], pct=True)}"
            f"  ({time.time()-t0:.1f}s)"
        )
        results["puzzle"] = m
    elif policy_model is not None and not puzzle_test_meta.exists():
        print("\nPuzzle test set not found (run build_datasets.py to create it). Skipping.")

    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    if "reward" in results:
        m = results["reward"]
        print(f"  Reward model  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  Pearson r={m['pearson_r']:.4f}")
    if "policy" in results:
        m = results["policy"]
        print(
            f"  Policy model  loss={m['loss']:.4f}  perplexity={m['perplexity']:.2f}"
            f"  top-1={m['top1_acc']*100:.2f}%  top-5={m['top5_acc']*100:.2f}%"
        )
    if "puzzle" in results:
        m = results["puzzle"]
        print(
            f"  Puzzle eval   first-move solve={m['first_move_solve_rate']*100:.2f}%"
            f"  all-moves solve={m['all_moves_solve_rate']*100:.2f}%"
        )
    if not results:
        print("  No results — check that models and test sets exist.")
    print("=" * 60)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward-model", default="reward_model.pt",
        help="Path to reward_model.pt (default: reward_model.pt)")
    p.add_argument("--policy-model", default="policy_model.pt",
        help="Path to policy_model.pt (default: policy_model.pt)")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
        help="Directory containing test set .bin files and tokenizer.pt (default: data/)")
    p.add_argument("--batch-size", type=int, default=512,
        help="Batch size for evaluation (default: 512)")
    p.add_argument("--num-workers", type=int, default=4,
        help="DataLoader worker count (default: 4)")
    p.add_argument("--device", default=None,
        help="Device override, e.g. 'cpu' or 'cuda:0' (default: auto-detect)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    results = run_benchmark(
        data_dir=args.data_dir,
        reward_model_path=args.reward_model,
        policy_model_path=args.policy_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    _print_summary(results)
