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

Each dataset is saved as memory-mapped binary arrays (`.bin`) for fast DataLoader access. Expected artifacts in `data/`:

| File | Description | Approx size |
|---|---|---|
| `games_outcome.pt` | Raw outcome-subset games | ~100 MB |
| `games_stockfish.pt` | Raw Stockfish-subset games | ~3 MB |
| `tokenizer.pt` | Shared BPE tokenizer | <1 MB |
| `outcome_tokens.bin` | (N, max_len) int16 token ids, zero-padded | ~6–10 GB |
| `outcome_labels.bin` | (N,) float32 outcome labels {−1, 0, +1} | ~76 MB |
| `outcome_lengths.bin` | (N,) int16 actual sequence lengths | ~38 MB |
| `outcome_meta.pt` | Dict: `n` (sample count), `max_len` | <1 MB |
| `stockfish_tokens.bin` | (N, max_len) int16 token ids, zero-padded | ~200–300 MB |
| `stockfish_labels.bin` | (N,) float32 Stockfish labels in [−1, 1] | ~2 MB |
| `stockfish_lengths.bin` | (N,) int16 actual sequence lengths | ~1 MB |
| `stockfish_meta.pt` | Dict: `n` (sample count), `max_len` | <1 MB |

Only the `tokenizer.pt`, `outcome_*`, and `stockfish_*` files are needed for training. The `games_*.pt` files are intermediate artifacts.

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
PYTHONPATH=src poetry run python -c "
import torch
import numpy as np

tok = torch.load('data/tokenizer.pt', weights_only=False)
out_meta = torch.load('data/outcome_meta.pt', weights_only=True)
sf_meta  = torch.load('data/stockfish_meta.pt', weights_only=True)

out_labels = np.memmap('data/outcome_labels.bin', dtype=np.float32, mode='r',
                        shape=(out_meta['n'],))
sf_labels  = np.memmap('data/stockfish_labels.bin', dtype=np.float32, mode='r',
                        shape=(sf_meta['n'],))

print(f'vocab size:        {tok.language_size}')
print(f'outcome samples:   {out_meta[\"n\"]:,}  max_seq_len={out_meta[\"max_len\"]}  '
      f'labels: {set(out_labels[:10000].round(1).tolist())}')
print(f'stockfish samples: {sf_meta[\"n\"]:,}  max_seq_len={sf_meta[\"max_len\"]}  '
      f'score range: [{sf_labels.min():.3f}, {sf_labels.max():.3f}]')
"
```

Expected roughly:
```
vocab size:        ~2000
outcome samples:   ~15–20M  max_seq_len=~150–250  labels: {-1.0, 0.0, 1.0}
stockfish samples: ~400–600K  max_seq_len=~150–250  score range: [-1.0, 1.0]
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

## Step 5 — Transfer datasets from local to remote (~5–20 min)

Back on your **local** machine, from the repo root:

```bash
ssh -p <PORT> root@<HOST> "mkdir -p ~/GptForChess/data"
scp -P <PORT> \
  data/tokenizer.pt \
  data/outcome_tokens.bin \
  data/outcome_labels.bin \
  data/outcome_lengths.bin \
  data/outcome_meta.pt \
  data/stockfish_tokens.bin \
  data/stockfish_labels.bin \
  data/stockfish_lengths.bin \
  data/stockfish_meta.pt \
  root@<HOST>:~/GptForChess/data/
```

Total transfer: ~7–11 GB. At 100 Mbps upload this is ~10–15 min. The bulk is `outcome_tokens.bin` — the padded token array is larger than the old list-of-lists `.pt` because every row is padded to `max_len`.

## Step 6 — Train (~1–3 hr on a 32GB card)

On the remote:

```bash
cd ~/GptForChess
poetry run python src/train.py
```

The trainer detects `outcome_meta.pt` / `stockfish_meta.pt` and automatically loads both datasets via memory-mapped arrays (`ChessPositionDataset.from_memmap`), which avoids the multi-minute Python deserialization of the old `.pt` format.

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

Model weights: ~160 MB. Transfer takes seconds.

## Step 8 — Shut down the Vast instance

**Important — instances bill until destroyed.** Go to the Instances tab on vast.ai and click *Destroy* (not just *Stop* — stopped instances still incur a small storage fee).

## Step 9 — Local inference with MCTS

Load the trained reward model and use it with MCTS. Create a small script:

```python
# inference_demo.py
import chess
import torch

from model import ChessRewardModel, RewardModelInference
from mcts import MinimaxSearch

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

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `scp: Permission denied (publickey)` | SSH key not registered on Vast. Add under *Account → SSH keys*, destroy+recreate instance. |
| OOM during training | Reduce `--batch-size` (try 64, then 32). Current ~10M model should fit batch=128 on 32 GB easily. |
| `torch.cuda.is_available()` returns False on remote | Template lacks CUDA PyTorch. `poetry run python -c "import torch; print(torch.__version__)"` — reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu121`. |
| TensorBoard connection refused | `--bind_all` flag missing, or forgot `-L 6006:localhost:6006` on the ssh command. |
| Phase 2 `task_mse` not descending | `distill_lambda` may be too high — try 0.01 or 0. If phase-1 weights are bad, something is off upstream. |
| Instance destroyed mid-training | `reward_model_phase1.pt` saves after phase 1 — resume by loading that checkpoint and running only phase 2 (requires a small code tweak to skip phase 1 if a checkpoint is passed in). |
| Memmap shape mismatch error | The `.bin` files and `_meta.pt` must match — if you re-ran `build_datasets.py` with `--force`, re-transfer all `outcome_*` and `stockfish_*` files. |

---

## Cost reality check

At ~$0.50/hr for the ~32 GB DLPerf-199 card we scoped:
- Full training run: ~$1.50–3.00
- Idle instance while you sleep: **$12/day** — always destroy when done.

Build datasets locally (CPU-only work), do the GPU-intensive parts in a single focused session, and tear down.
