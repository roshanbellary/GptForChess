# Deployment

End-to-end workflow: build datasets locally → train on a rented Vast.ai GPU → pull the trained model back for local inference via MCTS.

```
 LOCAL (Mac)                            VAST (rented GPU)
 ───────────                            ─────────────────
 1. build_datasets.py    scp data/ →    4. poetry install
 2. verify artifacts                    5. train.py (two-phase)
                         ← scp model    6. shutdown
 7. load + run inference
```

---

## Prerequisites

**Local:**
- Repo cloned, Poetry installed, Stockfish installed (`brew install stockfish`)
- Python 3.11+

**Remote (Vast.ai):**
- Account with positive balance
- SSH key added to Vast under *Account → SSH keys*
- A template that gives you **Python 3.11+ and PyTorch 2.0+ with CUDA** (e.g., `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` or one of Vast's official ML templates)

---

## Step 1 — Build the datasets locally (~60–90 min)

Run the full resumable builder with defaults (1M outcome + 30K Stockfish games, depth 12, 8 workers):

```bash
poetry run python src/build_datasets.py
```

Key defaults:
- `--skip-ply=10` — outcome dataset drops first 10 plies (opening positions carry noisy labels when labeled by game result)
- `--sf-skip-ply=0` — Stockfish dataset includes all positions including openings (Stockfish labels are precise per-position, so the model learns accurate opening evaluations during phase-2 fine-tuning)

Expected artifacts in `data/`:

| File | Size |
|---|---|
| `games_outcome.pt` | ~100 MB |
| `games_stockfish.pt` | ~3 MB |
| `tokenizer.pt` | <1 MB |
| `outcome_samples.pt` | ~3–4 GB |
| `stockfish_samples.pt` | ~120 MB |

Only the last three are needed for training.

**Resuming after a crash:** just re-run the same command — each stage skips if its output exists. Use `--force` to re-run everything.

**Smaller smoke run (2 min):**
```bash
poetry run python src/build_datasets.py \
  --outcome-games=500 --stockfish-games=50 \
  --stockfish-depth=6 --workers=4 --out-dir=data_smoke
```

## Step 2 — Verify artifacts before renting

Catching a bad dataset *before* you pay for a GPU is worth 30 seconds:

```bash
poetry run python -c "
import torch
tok = torch.load('data/tokenizer.pt', weights_only=False)
out = torch.load('data/outcome_samples.pt', weights_only=False)
sf  = torch.load('data/stockfish_samples.pt', weights_only=False)
print(f'vocab size: {tok.language_size}')
print(f'outcome samples: {len(out):,}  labels: {set(s[1] for s in out[:10000])}')
print(f'stockfish samples: {len(sf):,}  score range: '
      f'[{min(s[1] for s in sf):.3f}, {max(s[1] for s in sf):.3f}]')
"
```

Expected roughly:
```
vocab size: ~2000
outcome samples: ~15–20M  labels: {-1.0, 0.0, 1.0}
stockfish samples: ~400–600K  score range: [-1.0, 1.0]
```

## Step 3 — Rent a GPU on Vast.ai

**Search filters:**
- GPU RAM ≥ 32 GB
- Disk ≥ 50 GB
- Reliability ≥ 98%
- DLPerf ≥ 150 (rules out ancient cards)
- An ML template preloaded with PyTorch + CUDA

**Launch, then grab SSH connection info from the Instances tab.** It will look like:
```
ssh -p 12345 root@ssh5.vast.ai
```

## Step 4 — Set up the remote box (~5 min)

SSH in and clone the repo:

```bash
ssh -p <PORT> root@<HOST>

# on remote:
apt-get update && apt-get install -y git
git clone https://github.com/<your-user>/GptForChess.git
cd GptForChess

# Install Poetry if not present
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies (Poetry will pick up the CUDA torch from the template)
poetry install --no-root
```

**Note:** Stockfish is *not* required on the remote — you're using pre-labeled datasets. Skip that install.

## Step 5 — Transfer datasets from local to remote (~2–10 min)

Back on your **local** machine, from the repo root:

```bash
ssh -p <PORT> root@<HOST> "mkdir -p ~/GptForChess/data"
scp -P <PORT> \
  data/tokenizer.pt \
  data/outcome_samples.pt \
  data/stockfish_samples.pt \
  root@<HOST>:~/GptForChess/data/
```

Total transfer: ~3–4 GB. At 100 Mbps upload this is ~5 min.

## Step 6 — Train (~1–3 hr on a 32GB card)

On the remote:

```bash
cd ~/GptForChess
poetry run python src/train.py
```

Default schedule:
- Phase 1: 10 epochs on ~20M outcome samples → saves `reward_model_phase1.pt`
- Phase 2: 10 epochs on ~600K Stockfish samples with distillation regularizer (λ=0.05) → saves `reward_model.pt`

**Watch training live with TensorBoard.** From your **local** machine, set up an SSH tunnel:

```bash
ssh -p <PORT> -L 6006:localhost:6006 root@<HOST>

# on remote (in a second shell):
cd ~/GptForChess
poetry run tensorboard --logdir runs/chess_reward_model --bind_all
```

Then open `http://localhost:6006` locally. Expected trajectories:
- `phase1/epoch_loss` — descends toward ~0.3–0.5 (noisy outcome labels)
- `phase2/epoch_task_mse` — descends faster (precise Stockfish labels)
- `phase2/epoch_distill` — small but nonzero (slight drift from phase-1 teacher, as intended)

**Common flag overrides:**
```bash
poetry run python src/train.py \
  --batch-size=256 \
  --phase1-epochs=15 --phase2-epochs=5 \
  --distill-lambda=0.1   # stronger anti-drift
```

## Step 7 — Pull the model back to local

From **local** machine:

```bash
scp -P <PORT> \
  root@<HOST>:~/GptForChess/reward_model.pt \
  root@<HOST>:~/GptForChess/reward_model_phase1.pt \
  .

# optional — pull tensorboard logs so you can browse offline
scp -P <PORT> -r \
  root@<HOST>:~/GptForChess/runs/ ./runs/
```

Model weights: ~160 MB (40M params × 4 bytes). Transfer takes seconds.

## Step 8 — Shut down the Vast instance

**Important — instances bill until destroyed.** Go to the Instances tab on vast.ai and click *Destroy* (not just *Stop* — stopped instances still incur a small storage fee).

## Step 9 — Local inference with MCTS

Load the trained reward model and use it with MCTS. Create a small script:

```python
# inference_demo.py
import chess
import torch

from model import ChessRewardModel, RewardModelInference
from mcts import MinimaxSearch  # or whatever your search class is named

# Load artifacts
tokenizer = torch.load("data/tokenizer.pt", weights_only=False)
model = ChessRewardModel(vocab_size=tokenizer.language_size)
model.load_state_dict(torch.load("reward_model.pt", map_location="cpu", weights_only=False))
model.eval()

reward_fn = RewardModelInference(model, tokenizer, device="cpu")

# Sanity check on known positions
board = chess.Board()
print(f"starting position: {reward_fn(board):+.3f}")  # should be ≈ 0

board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(f"white up a queen:  {reward_fn(board):+.3f}")  # should be clearly positive

# Run MCTS to find a move
board = chess.Board()
searcher = MinimaxSearch(reward_fn=reward_fn, ...)  # use your existing search API
best_move = searcher.find_best_move(board)
print(f"best move: {best_move}")
```

Run from the repo root with `src/` on the path:

```bash
PYTHONPATH=src poetry run python inference_demo.py
```

The reward model runs fast enough on CPU for single-board evaluation (~10–50 ms per call). If you're doing deep MCTS rollouts, consider batching evals — but that's a future optimization.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `scp: Permission denied (publickey)` | SSH key not registered on Vast. Add under *Account → SSH keys*, destroy+recreate instance. |
| OOM during training | Reduce `--batch-size` (try 64, then 32). Current 40M model should fit batch=128 on 32 GB easily. |
| `torch.cuda.is_available()` returns False on remote | Template lacks CUDA PyTorch. `poetry run python -c "import torch; print(torch.__version__)"` — reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu121`. |
| TensorBoard connection refused | `--bind_all` flag missing, or forgot `-L 6006:localhost:6006` on the ssh command. |
| Phase 2 `task_mse` not descending | `distill_lambda` may be too high — try 0.01 or 0. If phase-1 weights are bad, something is off upstream. |
| Instance destroyed mid-training | `reward_model_phase1.pt` saves after phase 1 — resume by loading that checkpoint and running only phase 2 (requires a small code tweak to skip phase 1 if a checkpoint is passed in). |

---

## Cost reality check

At ~$0.50/hr for the ~32 GB DLPerf-199 card we scoped:
- Full training run: ~$1.50–3.00
- Idle instance while you sleep: **$12/day** — always destroy when done.

Build datasets locally (CPU-only work), do the GPU-intensive parts in a single focused session, and tear down.
