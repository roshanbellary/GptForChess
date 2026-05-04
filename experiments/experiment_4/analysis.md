# Experiment 4: Puzzle-Augmented Policy Training + Systematic Benchmarking

## Hypothesis

Augmenting policy model training with ~1.5M high-quality Lichess chess puzzles will meaningfully improve tactical move accuracy specifically the puzzle first-move solve rate and top-1 next-move accuracy without degrading strategic/positional play learned from full games. Puzzle data provides uniquely clean supervision: every solver move is the "only" correct move, which is rare in standard game sequences where many moves are roughly equivalent. This hard labeling should sharpen the policy's confidence at critical decision points.

Secondary hypothesis: a formal held-out benchmark (rather than ad-hoc interactive play) will reveal where the model actually struggles, likely mid-game tactics, and give a repeatable metric to track across future experiments.

---

## Procedure

### 1. Policy data augmentation with chess puzzles

Added a Stage 5 pipeline to `build_datasets.py` that streams the Lichess/chess-puzzles dataset from HuggingFace (~4.99M entries). Each puzzle contains a FEN (position before opponent's setup move) and a Moves string (UCI solution sequence). Applied two quality filters to get a reliable signal:

- `Popularity >= 75` — removes poorly-rated or disputed puzzles
- `NbPlays >= 5000` — ensures enough attempts for popularity to be meaningful

These filters yield approximately **1.5M high-quality puzzles** from the full 4.99M.

#### Puzzle structure: setup move + solution

Each puzzle decomposes into two parts:

- **Setup move S**: the opponent's forcing move (`Moves[0]`) — applied to the FEN to reach the puzzle position. This is the position the player must respond to. We treat S as **context**, not a prediction target — the model is never asked to guess what the opponent did.
- **Move set / solution**: `Moves[1:]` — the alternating sequence of (solver move, forced opponent response, solver move, ...). Every solver move is the "only correct" move; this is the supervision signal we want.

Each puzzle is tokenized as:

```
[CLS, S, m1, opp1, m2, opp2, ...]
```

All moves are validated against a `chess.Board` initialized from the FEN — any illegal move discards the entire puzzle.

### 2. Fine-tuning objective: P[m_n | S, M_{<n}]

The puzzle phase fine-tunes the policy model to learn:

> Given the setup move S and prior moves M_{<n}, predict the next move m_n.

In other words, we model the conditional distribution P[m_n | S, M_{<n}], where S is fixed context and the loss accumulates only over the solution moves.

#### Training step (per puzzle batch)

For input sequence `x = [CLS, S, m1, opp1, m2, ...]`:

1. **Input** to the policy model: `x[:-1]` = `[CLS, S, m1, opp1, ...]`
2. **Targets** for cross-entropy: `x[1:]` = `[S, m1, opp1, m2, ...]`
3. **Mask the setup move from loss**: `targets[:, 0] = pad_id`. Cross-entropy is called with `ignore_index=pad_id`, so the model is **not** penalized for failing to predict S from `[CLS]` — that prediction would be ill-posed (S is the opponent's move, not a model decision).
4. All remaining target positions (m1, opp1, m2, ...) contribute to the loss. The model updates its weights to match the unique solver moves and the forced opponent responses.

This is implemented in `_run_epoch_policy_puzzle` in `src/train.py`. The non-puzzle policy epoch (`_run_epoch_policy`) is unchanged — it computes loss at every non-PAD position because in full games there is no "context-only" prefix.

#### Two-phase Phase 2 training

Rather than mixing puzzle and game sequences in the same batches (originally planned 50/50), training runs **sequentially**:

| Phase | Data | Epochs (default) | Loss | Function |
|---|---|---|---|---|
| **2a** | ~950K full game sequences | `--policy-epochs 15` | cross-entropy at every position | `_run_epoch_policy` |
| **2b** | ~1.4M puzzle sequences | `--puzzle-epochs 5` | cross-entropy with setup move masked | `_run_epoch_policy_puzzle` |

Phase 2a establishes broad positional play from full games. Phase 2b then fine-tunes the resulting weights on puzzle solutions — the model arrives at puzzle training already knowing how chess sequences work and learns to sharpen its move distribution toward the unique correct answer at each solver position. After Phase 2a `policy_model.pt` is saved, then Phase 2b overwrites it with the fine-tuned weights.

This sequential structure also makes ablation easy: skip `--puzzle-data` to get a games-only baseline, or run `benchmark.py` against the post-Phase-2a checkpoint before fine-tuning to measure the puzzle-induced delta.


### 3. Formal benchmark system

The main weakness identified in Experiment 3 was the absence of a systematic test set: performance was judged by ad-hoc play rather than reproducible metrics. This experiment addresses that directly.

Built three held-out test sets in `build_datasets.py`:

| Test Set | Size | Source | Purpose |
|---|---|---|---|
| `stockfish_test_*.bin` | 50K positions | Sampled from stockfish memmap (fixed seed 42) | Reward model MSE/MAE/Pearson r |
| `policy_test_*.bin` | 50K sequences | Sampled from policy memmap (fixed seed 42) | Policy loss, perplexity, top-1/top-5 accuracy |
| `puzzle_test_*.bin` | 100K puzzles | First 100K collected before training puzzles | Puzzle first-move solve rate |

All three test sets are **genuinely disjoint** from training:
- Puzzle test set: the streaming pipeline collects the first 100K quality-filtered puzzles as test, then continues to collect training puzzles.
- Reward and policy test sets: a fixed-seed (42) random sample of indices is drawn from the underlying memmap and saved to `{name}_test_indices.npy`. The training dataset loaders read these files and skip those indices, so the model never trains on positions/sequences that appear in the test set. The fixed seed guarantees the same indices are always evaluated across runs.

Metrics per test set:
- **Reward model**: MSE, MAE, Pearson r (predicted vs Stockfish eval)
- **Policy model**: cross-entropy loss, perplexity, top-1 accuracy, top-5 accuracy
- **Puzzle eval**: first-move solve rate (top-1 at the solver's key move), all-moves solve rate

These metrics are logged to TensorBoard after every epoch and are also available via the standalone `benchmark.py`:

```bash
arch -arm64 poetry run python src/benchmark.py \
    --reward-model reward_model.pt \
    --policy-model policy_model.pt \
    --data-dir data/
```

---

## What Was Built (No Training Run Yet)

The expected workflow on the 4090:
```bash
# 1. Build puzzle data + test sets (run once)
python src/build_datasets.py \
    --min-puzzle-popularity 75 \
    --min-puzzle-plays 5000

# 2. Train: Phase 1 (reward) → Phase 2a (policy on games) → Phase 2b (puzzle fine-tune)
python src/train.py \
    --policy-epochs 15 \
    --puzzle-data data/ \
    --puzzle-epochs 5

# 3. Benchmark after training
python src/benchmark.py
```

---

## Expected Outcomes and Baselines

Based on Experiment 3's reward model MSE of 0.08 (train), I expect:
- **Reward test MSE**: ~0.10–0.13 (test set will be higher than train; this gives the true number)
- **Policy top-1 accuracy**: unknown baseline — this is the first time we measure it
- **Puzzle first-move solve rate**: 30–40% would be strong for a model this size; random baseline is ~0.05% (1/1968 moves)

The critical comparison after training: does puzzle augmentation improve first-move solve rate relative to a baseline trained on game sequences only? If the solve rate is above ~25%, tactical augmentation is working. If it's below 15%, the puzzle sequences are too short to provide meaningful context to the model.

---

## Avenues for Improvement (Pre-Training Observations)

**Puzzle context problem**: The model has no FEN encoder. It sees `[CLS, setup_move, solver_move1, ...]` with no knowledge of the board state before the puzzle begins. This limits how much it can generalize — it's learning tactical move patterns ("if opponent plays X, respond Y") rather than board state evaluation. A future improvement is to prepend a sequence of reconstructed moves from the full game (using the GameId field in the puzzle data) so the model has board context before the puzzle position. This would require fetching game histories from Lichess.

**Rating stratification**: The 1.5M quality-filtered puzzles are heavily skewed toward 1200–1800 Elo difficulty. The model will be overfit to intermediate-level tactics and may miss grandmaster-level combinational patterns. Future: sample uniformly across rating bands to cover beginner through master tactics.

**Puzzle solve rate decomposition**: The benchmark currently reports an aggregate solve rate. More useful would be solve rate broken down by puzzle theme (mate, fork, pin, etc.) and by rating band. This would reveal exactly which tactical patterns the model has and hasn't learned.
