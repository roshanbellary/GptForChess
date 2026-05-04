# Deployment

End-to-end workflow on a rented GPU: build datasets → train all phases → benchmark → pull artifacts back for local inference. Everything runs on the remote because Stockfish labeling at 1M-game scale is impractically slow on consumer CPUs (~30 hr on an M2 Pro vs ~5 hr on a 64-core EPYC).

```
 LOCAL (Mac)                       VAST (rented GPU)
 ───────────                       ─────────────────
                                   1. setup deps
                                   2. build_datasets.py  (Stages 1–5)
                                   3. train.py           (Phases 1, 2a, 2b)
                                   4. benchmark.py       (post-training eval)
                       ← scp artifacts
 5. local inference / minimax
```

---

## Prerequisites

**Local:**
- Repo cloned, Poetry installed (only needed for local inference at the end)
- SSH key generated and added to Vast.ai under *Account → SSH keys*

**Remote (Vast.ai):**
- Account with positive balance
- A template preloaded with **Python 3.11+, PyTorch 2.3+, CUDA 12.x**
  (e.g., `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` or any official Vast ML template)
- **GPU recommendation**: RTX PRO 6000 Blackwell (96 GB VRAM, $0.972/hr) or RTX 4090 (24 GB, ~$0.40/hr). Both work; the PRO 6000 is ~1.7× faster on transformer training.

---

## Step 1 — Rent the GPU

**Search filters on Vast.ai:**
- GPU: RTX PRO 6000 Blackwell *or* RTX 4090
- VRAM ≥ 24 GB
- CPU cores ≥ 32 (you'll want them for parallel Stockfish in Stage 3)
- Disk ≥ 60 GB
- Reliability ≥ 98%
- Bandwidth ≥ 500 Mbps (for streaming Lichess data from HuggingFace)

**Launch and grab SSH info from the Instances tab.** It looks like:
```
ssh -p 12345 root@ssh5.vast.ai
```

---

## Step 2 — Set up the remote box (~5 min)

SSH in and clone the repo:

```bash
ssh -p <PORT> root@<HOST>

# on remote:
apt-get update && apt-get install -y git stockfish
git clone https://github.com/<your-user>/GptForChess.git
cd GptForChess
```

Install Python deps **without overwriting the template's CUDA-enabled PyTorch**:

```bash
pip install python-chess datasets tqdm tensorboard numpy
```

> **Why not `poetry install`?** Vast templates ship a CUDA PyTorch build. Running `poetry install` can replace it with the CPU-only build from PyPI, breaking GPU training. Installing only the non-torch deps preserves the template's CUDA stack.

Sanity-check CUDA is available:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.4.0+cu124 True NVIDIA RTX PRO 6000 Blackwell
```

---

## Step 3 — Build datasets (~5–8 hr, mostly Stockfish)

Run the full resumable pipeline. Stage 3 (Stockfish labeling) is the bottleneck — scale `--workers` to your core count.

```bash
PYTHONPATH=src python src/build_datasets.py \
    --reward-games 1000000 \
    --policy-games 1000000 \
    --workers 56 \
    --stockfish-depth 12 \
    --min-puzzle-popularity 75 \
    --min-puzzle-plays 5000 \
    --reward-test-size 50000 \
    --policy-test-size 50000 \
    --puzzle-test-size 100000
```

**What runs (each stage skips if its outputs exist; use `--force` to re-run):**

| Stage | What it does | Time on 64-core EPYC |
|---|---|---|
| 1 | Stream Lichess games, filter by Elo + Termination, save raw subsets | 30–60 min (network bound) |
| 2 | Build the fixed UCI tokenizer + outcome samples | <5 min |
| 3 | **Parallel Stockfish labeling on 1M reward games (depth 12)** | **4–6 hr** |
| 4 | Tokenize 1M policy games into full sequences | 10–20 min |
| 5 | Stream + tokenize ~1.5M filtered Lichess puzzles | 30–60 min |
| Splits | Build disjoint test sets (reward, policy, puzzle) | <2 min |

**Tunable knobs:**
- `--reward-games`: drop to 500_000 to **halve Stage 3 time** (the dominant cost). The ~10M training samples this produces is plenty for the 58 M-param reward model.
- `--workers`: leave 8 cores for the OS — on a 64-core box, use 56.
- `--stockfish-depth 10` is ~2× faster than 12 with only minor quality loss.

**Expected artifacts in `data/` after Stage 5 completes:**

| File family | Description | Approx size |
|---|---|---|
| `tokenizer.pt` | Fixed UCI vocabulary (~1970 tokens) | <1 MB |
| `stockfish_{tokens,labels,lengths}.bin` + `_meta.pt` | Reward training data (~20 M positions) | ~10 GB |
| `policy_{tokens,lengths}.bin` + `_meta.pt` | Policy training data (~1 M sequences) | ~500 MB |
| `puzzle_{tokens,lengths}.bin` + `_meta.pt` | Puzzle training data (~1.4 M sequences) | ~60 MB |
| `stockfish_test_*.bin` + `_test_indices.npy` | Reward test set (50 K, disjoint from train) | ~25 MB |
| `policy_test_*.bin` + `_test_indices.npy` | Policy test set (50 K, disjoint from train) | ~25 MB |
| `puzzle_test_*.bin` | Puzzle test set (100 K, disjoint from train) | ~5 MB |

The `*_test_indices.npy` files cause `train.py` to automatically skip those rows in training, making train and test genuinely disjoint.

**Sanity check:**
```bash
PYTHONPATH=src python -c "
import torch, numpy as np
tok = torch.load('data/tokenizer.pt', weights_only=False)
sf = torch.load('data/stockfish_meta.pt', weights_only=True)
pol = torch.load('data/policy_meta.pt', weights_only=True)
puz = torch.load('data/puzzle_meta.pt', weights_only=True)
print(f'vocab size:        {tok.language_size}')
print(f'reward train:      {sf[\"n\"]:,} positions  (max_len={sf[\"max_len\"]})')
print(f'policy train:      {pol[\"n\"]:,} sequences (max_len={pol[\"max_len\"]})')
print(f'puzzle train:      {puz[\"n\"]:,} sequences (max_len={puz[\"max_len\"]})')
"
```

Expected:
```
vocab size:        1970
reward train:      ~20,000,000 positions  (max_len=128)
policy train:      ~1,000,000 sequences (max_len=128)
puzzle train:      ~1,400,000 sequences (max_len=~10)
```

---

## Step 4 — Train (~20–30 hr on RTX PRO 6000)

Three sequential phases, all in one command:

```bash
PYTHONPATH=src python src/train.py \
    --epochs 10 \
    --policy-epochs 12 \
    --puzzle-epochs 5 \
    --batch-size 4096 \
    --learning-rate 4e-5 \
    --num-workers 16 \
    --puzzle-data data/
```

**What runs:**

| Phase | Data | Loss | Time on PRO 6000 (BF16) |
|---|---|---|---|
| **1** Reward model | ~20 M Stockfish-labeled positions × 10 epochs | MSE | 17–25 hr |
| **2a** Policy model on games | ~1 M game sequences × 12 epochs | causal cross-entropy | 1.5–2.5 hr |
| **2b** Policy model fine-tune on puzzles | ~1.4 M puzzles × 5 epochs (setup move masked) | causal cross-entropy | 30–60 min |

**Confirmation that BF16 is active**: at startup the trainer prints
```
Using device: cuda (bfloat16 autocast)
```
If you see `(fp32)` instead, GPU isn't detected — re-check the CUDA install.

**Live test metrics**: after each epoch the trainer evaluates on the held-out test sets and logs:
- Reward: MSE, MAE, Pearson r
- Policy (each phase): cross-entropy, perplexity, top-1 / top-5 accuracy
- Puzzles (each policy phase): first-move solve rate, all-moves solve rate

### Watch live with TensorBoard

From your **local** machine, open an SSH tunnel:
```bash
ssh -p <PORT> -L 6006:localhost:6006 root@<HOST>
```

In a second shell on the remote:
```bash
cd ~/GptForChess
python -m tensorboard.main --logdir runs/chess_models --bind_all
```

Then open `http://localhost:6006` locally.

**Healthy trajectories to expect:**
- `train/reward_epoch_loss` — descends steadily toward ~0.05–0.10
- `test/reward_mse` — slightly above train (target ~0.10–0.13)
- `test/reward_pearson_r` — climbs toward 0.7+ over training
- `train_policy/epoch_loss` — descends toward ~3.5–4.5 (cross-entropy in nats)
- `test_2a/policy_top1_acc` — climbs to ~30–40% by epoch 12
- `test_2b/puzzle_first_move` — should jump during Phase 2b; target 25–40%

---

## Step 5 — Run the formal benchmark

After training, all three test metrics in one command:

```bash
PYTHONPATH=src python src/benchmark.py \
    --reward-model reward_model.pt \
    --policy-model policy_model.pt \
    --data-dir data/ \
    --batch-size 1024
```

This prints final test numbers for the reward model, policy model (post-Phase 2b), and puzzle solve rates. **Save this output** — it's the canonical record of the run.

---

## Step 6 — Pull artifacts back to local

From your **local** machine:

```bash
# model checkpoints (~232 MB each in FP32)
scp -P <PORT> \
    root@<HOST>:~/GptForChess/reward_model.pt \
    root@<HOST>:~/GptForChess/policy_model.pt \
    .

# tensorboard logs (browse offline)
scp -P <PORT> -r root@<HOST>:~/GptForChess/runs/ ./runs/

# tokenizer (needed at inference time)
scp -P <PORT> root@<HOST>:~/GptForChess/data/tokenizer.pt ./data/
```

You generally don't need the `.bin` test sets locally — they're recoverable by re-running Step 3 with the same seed.

---

## Step 7 — Destroy the Vast instance

**Critical — instances bill until destroyed.** From the Instances tab on vast.ai:
- Click **Destroy** (not just Stop — stopped instances still incur storage fees)
- Confirm the instance disappears from the list

---

## Step 8 — Local inference with the trained reward + policy models

```python
# inference_demo.py
import chess
import torch

from model import (
    ChessRewardModel,
    ChessPolicyModel,
    RewardModelInference,
    PolicyModelInference,
)
from mcts import MinimaxSearch

tokenizer = torch.load("data/tokenizer.pt", weights_only=False)
vocab_size = tokenizer.language_size

reward_model = ChessRewardModel(vocab_size=vocab_size)
reward_model.load_state_dict(torch.load("reward_model.pt", map_location="cpu", weights_only=True))
reward_model.eval()

policy_model = ChessPolicyModel(vocab_size=vocab_size)
policy_model.load_state_dict(torch.load("policy_model.pt", map_location="cpu", weights_only=True))
policy_model.eval()

reward_fn = RewardModelInference(reward_model, tokenizer, device="cpu")
policy_fn = PolicyModelInference(policy_model, tokenizer, device="cpu")

board = chess.Board()
print(f"starting eval:  {reward_fn(board):+.3f}")        # should be ≈ 0
print(f"policy choice:  {policy_fn(board)}")             # e.g., 'e2e4'

# Combine with minimax search (top-N pruning, depth 3)
search = MinimaxSearch(reward_fn=reward_fn, depth=3, top_n=5)
best_move = search.search(board)
print(f"minimax pick:   {best_move}")
```

Run from the repo root:
```bash
PYTHONPATH=src arch -arm64 poetry run python inference_demo.py
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `scp: Permission denied (publickey)` | SSH key not registered. Add under *Account → SSH keys*, then destroy + recreate the instance (existing instances don't pick up new keys). |
| OOM during training | Reduce `--batch-size`. On 24 GB card try 1024. On 96 GB you should never hit this even at 8192. |
| `torch.cuda.is_available()` returns False | Template lacks CUDA PyTorch. Verify with `python -c "import torch; print(torch.__version__)"` — needs `+cu` suffix. If missing: `pip install torch --index-url https://download.pytorch.org/whl/cu124`. |
| `bfloat16 autocast` not shown at startup | GPU not detected. Same fix as above. |
| TensorBoard connection refused | `--bind_all` flag missing on the remote tensorboard, or forgot `-L 6006:localhost:6006` in the SSH command. |
| Test loss climbs while train loss falls | Overfitting. With 58 M params + 20 M reward samples this is rare; if it happens, lower `--epochs` or add weight decay. |
| Memmap shape mismatch error | `.bin` and `_meta.pt` files out of sync. Delete and re-run the relevant stage with `--force`. |
| Stage 3 absurdly slow | Confirm `--workers 56` (or appropriate for core count) — default is 16. Check `htop` to verify Stockfish processes are running. |
| Phase 2b crashes with shape error | `puzzle_meta.pt` missing. Run `build_datasets.py` Stage 5 (or full pipeline). |
| Training resumes but skips Phase 1 | `reward_model.pt` already exists in cwd from a prior run. Delete it or move it before re-launching. |

---

## Cost reality check

At **$0.972/hr** for the RTX PRO 6000 Blackwell:

| Configuration | Wall-clock | Cost |
|---|---|---|
| Default config (1M reward games, 10 epochs) | 25–35 hr | **$24–34** |
| Aggressive (500K reward games, 10 epochs) | 18–25 hr | **$18–24** |
| Conservative (1M games, 15 epochs) | 30–45 hr | **$29–44** |
| Idle instance you forgot to destroy | $23 / day | 💸 |

**Always destroy when done.** Stopped instances still bill for storage.

At **$0.40/hr** for an RTX 4090, total cost is ~$15–20 but wall-clock is ~1.7× longer. Pick based on whether you value money or hours more.

---

## Quick reference: full sequence

```bash
# 1. SSH to remote
ssh -p <PORT> root@<HOST>

# 2. Setup (one-time per instance)
apt-get install -y git stockfish && \
git clone https://github.com/<your-user>/GptForChess.git && \
cd GptForChess && \
pip install python-chess datasets tqdm tensorboard numpy

# 3. Datasets (~5–8 hr)
PYTHONPATH=src python src/build_datasets.py \
    --workers 56 --stockfish-depth 12 \
    --min-puzzle-popularity 75 --min-puzzle-plays 5000

# 4. Train (~20–30 hr)
PYTHONPATH=src python src/train.py \
    --epochs 10 --policy-epochs 12 --puzzle-epochs 5 \
    --batch-size 4096 --learning-rate 4e-5 \
    --num-workers 16 --puzzle-data data/

# 5. Benchmark
PYTHONPATH=src python src/benchmark.py \
    --reward-model reward_model.pt \
    --policy-model policy_model.pt \
    --data-dir data/

# 6. (back on local) pull artifacts
scp -P <PORT> root@<HOST>:~/GptForChess/{reward,policy}_model.pt .
scp -P <PORT> root@<HOST>:~/GptForChess/data/tokenizer.pt data/

# 7. (in Vast UI) destroy the instance
```
