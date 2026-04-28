# Deployment

End-to-end workflow: build Stockfish dataset locally → train on a rented Vast.ai GPU → pull the trained model back for local inference via MCTS.

```
 LOCAL (Mac)                            VAST (rented GPU)
 ───────────                            ─────────────────
 1. build_datasets.py    scp data/ →    3. install deps
 2. verify artifacts                    4. train.py
                         ← scp model    5. shutdown
 6. load + run inference
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

## Step 1 — Build the Stockfish dataset locally

Run the full resumable builder. Stockfish labeling is the slow step — scale `--workers` to your core count and budget time accordingly (~15 hrs on 8 cores, ~5 hrs on 32 cores):

```bash
arch -arm64 poetry run python src/build_datasets.py \
  --stockfish-games 500000 \
  --workers 10 \
  --stockfish-depth 12
```

Key flags:
- `--stockfish-games` — number of games to label with Stockfish. 500K × 20 positions = ~10M training samples.
- `--workers` — one Stockfish subprocess per core. Set to (physical core count − 2).
- `--stockfish-depth` — depth 12 is a good quality/speed tradeoff. Depth 10 is ~4x faster with only a small quality drop.

Each stage is resumable — re-running skips completed stages. Use `--force` to re-run everything.

Expected artifacts in `data/` after completion:

| File | Description | Approx size |
|---|---|---|
| `tokenizer.pt` | Shared BPE tokenizer | <1 MB |
| `stockfish_tokens.bin` | (N, max_len) int32 token ids, zero-padded | ~5 GB |
| `stockfish_labels.bin` | (N,) float32 Stockfish labels in [−1, 1] | ~40 MB |
| `stockfish_lengths.bin` | (N,) int32 actual sequence lengths | ~20 MB |
| `stockfish_meta.pt` | Dict: `n` (sample count), `max_len` | <1 MB |

`games_outcome.pt`, `games_stockfish.pt` are intermediate artifacts — not needed for training.

## Step 2 — Verify artifacts before renting

```bash
PYTHONPATH=src arch -arm64 poetry run python -c "
import torch
import numpy as np

tok = torch.load('data/tokenizer.pt', weights_only=False)
sf_meta = torch.load('data/stockfish_meta.pt', weights_only=True)
sf_labels = np.memmap('data/stockfish_labels.bin', dtype=np.float32, mode='r',
                       shape=(sf_meta['n'],))

print(f'vocab size:        {tok.language_size}')
print(f'stockfish samples: {sf_meta[\"n\"]:,}  max_seq_len={sf_meta[\"max_len\"]}')
print(f'score range:       [{sf_labels.min():.3f}, {sf_labels.max():.3f}]')
"
```

Expected roughly:
```
vocab size:        ~30000
stockfish samples: ~8–10M  max_seq_len=128
score range:       [-1.000, 1.000]
```

## Step 3 — Rent a GPU on Vast.ai

**Search filters:**
- GPU RAM ≥ 24 GB
- Disk ≥ 30 GB
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
```

Install dependencies without overwriting the template's CUDA-enabled PyTorch:

```bash
pip install python-chess datasets tqdm tensorboard
```

> **Why not `poetry install`?** Vast templates ship a CUDA PyTorch build. Running `poetry install` can replace it with a CPU-only build from PyPI, breaking GPU training. Installing only the non-torch deps avoids this.

## Step 5 — Transfer datasets from local to remote (~5–15 min)

Back on your **local** machine, from the repo root:

```bash
ssh -p <PORT> root@<HOST> "mkdir -p ~/GptForChess/data"
scp -P <PORT> \
  data/tokenizer.pt \
  data/stockfish_tokens.bin \
  data/stockfish_labels.bin \
  data/stockfish_lengths.bin \
  data/stockfish_meta.pt \
  root@<HOST>:~/GptForChess/data/
```

Total transfer: ~5 GB. At 100 Mbps upload this is ~7–10 min.

## Step 6 — Train (~1–3 hr on a 24GB card)

On the remote:

```bash
cd ~/GptForChess
PYTHONPATH=src python src/train.py --epochs 15 --batch-size 1024
```

The trainer detects `stockfish_meta.pt` and loads the dataset via memory-mapped arrays, which avoids multi-minute Python deserialization.

**Watch training live with TensorBoard.** From your **local** machine, set up an SSH tunnel:

```bash
ssh -p <PORT> -L 6006:localhost:6006 root@<HOST>

# on remote (in a second shell):
cd ~/GptForChess
PYTHONPATH=src python -m tensorboard.main --logdir runs/chess_reward_model --bind_all
```

Then open `http://localhost:6006` locally. Expected trajectory:
- `train/epoch_loss` — descends steadily toward ~0.05–0.15

**Common flag overrides:**
```bash
PYTHONPATH=src python src/train.py \
  --epochs 20 \
  --batch-size 512 \
  --learning-rate 1e-4
```

## Step 7 — Pull the model back to local

From **local** machine:

```bash
scp -P <PORT> \
  root@<HOST>:~/GptForChess/reward_model.pt \
  .

# optional — pull tensorboard logs so you can browse offline
scp -P <PORT> -r \
  root@<HOST>:~/GptForChess/runs/ ./runs/
```

Model weights: ~160 MB. Transfer takes seconds.

## Step 8 — Shut down the Vast instance

**Important — instances bill until destroyed.** Go to the Instances tab on vast.ai and click *Destroy* (not just *Stop* — stopped instances still incur a small storage fee).

## Step 9 — Local inference with MCTS

```python
# inference_demo.py
import chess
import torch

from model import ChessRewardModel, RewardModelInference
from mcts import MinimaxSearch

tokenizer = torch.load("data/tokenizer.pt", weights_only=False)
model = ChessRewardModel(vocab_size=tokenizer.language_size)
model.load_state_dict(torch.load("reward_model.pt", map_location="cpu", weights_only=False))
model.eval()

reward_fn = RewardModelInference(model, tokenizer, device="cpu")

board = chess.Board()
print(f"starting position: {reward_fn(board):+.3f}")  # should be ≈ 0
```

Run from the repo root:

```bash
PYTHONPATH=src arch -arm64 poetry run python inference_demo.py
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `scp: Permission denied (publickey)` | SSH key not registered on Vast. Add under *Account → SSH keys*, destroy+recreate instance. |
| OOM during training | Reduce `--batch-size` (try 256, then 128). The ~10M param model should fit batch=512 on 24 GB easily. |
| `torch.cuda.is_available()` returns False | Template lacks CUDA PyTorch. Run `python -c "import torch; print(torch.__version__)"` — if no `+cu` suffix, reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu121`. |
| TensorBoard connection refused | `--bind_all` flag missing, or forgot `-L 6006:localhost:6006` on the ssh command. |
| `train/epoch_loss` not descending | Learning rate may be too low/high — try `--learning-rate 1e-4`. |
| Memmap shape mismatch error | The `.bin` files and `_meta.pt` must match — if you re-ran `build_datasets.py --force`, re-transfer all `stockfish_*` files. |

---

## Cost reality check

At ~$0.50/hr for a 24 GB card:
- Full training run: ~$1–2
- Idle instance while you sleep: **$12/day** — always destroy when done.

Build datasets locally (CPU-only work), do the GPU-intensive parts in a single focused session, and tear down.
