# Experiment 4: Puzzle-Augmented Policy Training + Systematic Benchmarking


## Hypothesis
This is more of a given but I plan on updating puzzle training to take in an encoded board state given by the FEN encoding in the puzzle data. This will dramatically improve the errors that model had with puzzle reasoning by being able to properly give attention to moves.

I will also need to udpate policy training on regular game data by having a FEN String for board state and a sequence of moves after that. It shouldn't be a problem though.


## Procedure

Set up a CNN that gets trained in unison with the encoder. The CNN takes the
19-plane board representation (`board_to_planes`) and outputs a single
context-rich `d_model` vector, which replaces the position-0 (CLS) embedding in
the move sequence. The transformer then runs over the same `(B, T, d_model)`
shape it did before — only the contents of slot 0 change.

### Architecture choice: why a single pooled board embedding (option a)

Three fusion strategies were on the table:

- **(a) Single pooled board vector** at position 0 of the move sequence.
- **(b) 64 per-square tokens** prepended to the move sequence.
- **(c) Cross-attention** with board features as keys/values and moves as queries.

Option (b) was rejected because it conflates two semantically different entities
in the same stream: each of the 64 squares becomes its own "token" the
transformer reasons about alongside the moves. The model would have to learn —
purely from positional encodings — to distinguish "this is a square" from "this
is a move." That's wasted capacity, and it inflates the sequence length from
`T` to `T + 64` (roughly 1.5x at `T=128`), making attention noticeably more
expensive for a benefit the CNN was supposed to provide in the first place.

Option (c) is architecturally the most expressive (no bottleneck, separate
streams, moves can selectively pull spatial detail), but it requires a
cross-attention layer per block and a more invasive rewrite of the encoder
stack. We're deferring it as a fallback if (a) underperforms.

Option (a) was chosen because:

1. **Separation of concerns.** Spatial reasoning lives entirely in the CNN;
   temporal/sequential reasoning lives entirely in the transformer. Each
   component does what it's good at.
2. **The sequence stays "pure."** Every position in the transformer represents
   a move (or, at position 0, the game state). No type-discrimination burden
   on the attention layers.
3. **No sequence-length inflation.** We reuse the existing CLS slot — the
   causal mask, positional encoding, padding mask, and downstream training code
   all work unchanged.
4. **Implementation cost is minimal.** A ResNet trunk with average pooling and
   a linear projection. ~1.8M params at 6 blocks × 128 channels — negligible
   next to the transformer.

### Acknowledged tradeoff

Pooling 8×8×128 = 8192 spatial features into a single `d_model`=768 vector is a
genuine information bottleneck. The CNN has to decide *up front* what's
relevant about the board, before it knows which move the policy is about to
score. If puzzle accuracy stalls in a way that suggests the model is missing
specific tactical squares (e.g., not seeing a pinned piece on a square it
hasn't "summarized" well), that is the signal to migrate to option (c).

### Implementation summary

- `BoardCNN` in `src/model.py`: stem conv → 6 residual blocks (128 channels) →
  adaptive avg pool → linear projection to `d_model`.
- `ChessPolicyModel.forward` now takes `(token_ids, board_planes, attention_mask)`.
  The board vector overwrites the position-0 embedding; the rest of the path
  (positional encoding, causal mask, encoder, prob head) is unchanged.
- `PolicyModelInference` builds planes via `board_to_planes(board)` alongside
  the existing token sequence.
- Fixed two bugs in `board_to_planes`: `board.tun` → `board.turn`, and
  `planes[12].fill(1.0)` → `planes[12].fill_(1.0)` (the non-underscore version
  is not in-place).


## Mixed training: games + puzzles in the same batches

### Why mixed instead of sequential (Phase 2a / Phase 2b)

Experiment 4 used sequential training — full-game policy first, then puzzle
fine-tuning — and observed catastrophic forgetting on regular-game play. The
new CNN-conditioned architecture makes sequential training *worse* for two
reasons:

1. **BatchNorm running-stats drift.** The original CNN sketch used
   `BatchNorm2d`. BN accumulates running mean/variance across training; if
   Phase 1 sees only game-derived board distributions and Phase 2 sees only
   puzzle distributions, the inference-time BN buffers stop matching what
   the conv filters expect. To remove this concern entirely we swapped
   `BatchNorm2d` → `GroupNorm` in `BoardCNN`, which is batch-composition
   independent.
2. **The CNN itself can forget.** In Experiment 4, the only thing that could
   "forget" was the transformer. Now the CNN's conv filters are also part of
   the learned board representation, and sequential phases would specialize
   them to one distribution then re-specialize to another.

Mixed training keeps both data sources visible throughout, eliminating the
forgetting failure mode by construction.

### Anchoring the CNN's board input

The CNN takes exactly one board planes tensor per sample (per the option (a)
design). For each data source we anchor it to the *starting* board of that
sequence:

- **Games:** `board_planes = board_to_planes(chess.Board())` — the standard
  chess starting position. The CNN signal for games is therefore a constant
  vector and provides effectively no information beyond "this is the start
  of a regular game." That's fine: games already have a rich move-history
  signal in the token sequence, and the transformer's positional encoding
  handles the temporal reasoning.
- **Puzzles:** `board_planes = board_to_planes(chess.Board(fen))` where the
  FEN is the puzzle's starting position. This is where the CNN earns its
  keep — without it, the model would have to reason about a tactical mid-game
  position from a near-empty token sequence (`[CLS]` + setup only). With it,
  the position-0 embedding carries the full board state.

Importantly, this anchor choice avoids the information-leak failure mode
discussed earlier: the CNN board never depends on a token the model is being
asked to predict, so the existing multi-position LM-style training paradigm
(loss at every non-pad position) carries over unchanged.

### Data pipeline changes

`build_datasets.py` now persists puzzle FENs alongside the token memmaps:

- `_process_puzzle` returns `(token_sequence, fen)` instead of just the
  sequence.
- `_save_policy_memmap` accepts an optional `fens` list and, when provided,
  writes `{name}_fens.bin` — `(N, fen_len)` uint8 holding zero-padded ASCII
  FEN strings.
- Stage 5 collects puzzle FENs and passes them through. `puzzle_fens.bin`
  and `puzzle_test_fens.bin` are now part of the standard build output.

`ChessPolicyDataset.from_memmap` accepts `source_tag` and `loss_weight`
arguments (defaults 0 / 1.0 for game data). When `{name}_fens.bin` exists,
the FEN at each sample index is decoded and used to build `board_planes`
via `chess.Board(fen)` → `board_to_planes(board)`. When the FEN file is
absent, the dataset falls back to the chess starting board planes (cached
once per dataset instance).

Each sample now yields `(tokens, board_planes, weight, source_tag)`, and
`collate_fn_policy` stacks the four pieces into batch-level tensors.

### Hard-balanced batches

`MixedBatchSampler` (in `train.py`) yields index lists into a
`ConcatDataset([games, puzzles])` such that every batch contains exactly
`game_ratio * batch_size` game indices and `(1 - game_ratio) * batch_size`
puzzle indices. Defaults are 80% game / 20% puzzle. Both pools are shuffled
independently; the (smaller) puzzle pool is re-shuffled whenever exhausted,
so puzzles are effectively oversampled to match the game stream.

This is preferred over stochastic mixing because:

1. The gradient signal per batch is consistent — every step "sees" puzzles.
2. BatchNorm-related composition concerns are moot (resolved by GroupNorm),
   but loss-level balance still benefits from determinism.
3. The puzzle ratio is an explicit, tunable knob rather than an emergent
   property of dataset sizes.

### Per-sample loss weighting

`_run_epoch_policy_mixed` replaces the separate `_run_epoch_policy` /
`_run_epoch_policy_puzzle` functions. The training step:

1. Builds inputs and targets (`targets = batch_tokens[:, 1:]`).
2. Masks the setup-move target for puzzle rows (`source==1`): the setup move
   is given as context, not predicted.
3. Computes per-position cross-entropy with `reduction='none'` so each
   position's loss is available before reduction.
4. Multiplies by `(position_is_valid * sample_weight)` and takes a weighted
   mean, where `sample_weight` is 1.0 for games and 5.0 for puzzles by
   default (configurable via `--puzzle-loss-weight`).

The puzzle weight is multiplicative with the batch composition — at 20%
puzzle ratio and 5× weight, puzzle positions account for ~50% of the
gradient norm per batch despite being a quarter of the samples. This is
deliberate: puzzles are the only source of high-quality tactical labels,
and the model should listen to them.

### CLI additions

- `--puzzle-loss-weight FLOAT` (default 5.0): per-sample loss weight for
  puzzle rows in the mixed training loop.
- `--puzzle-ratio FLOAT` (default 0.2): fraction of each mixed batch drawn
  from the puzzle pool.
- `--puzzle-epochs` is retained for CLI backward compatibility but ignored;
  mixed training has a single Phase 2 controlled by `--policy-epochs`.

### Rebuild required

Existing puzzle memmaps from prior runs do not contain FENs and will not
drive CNN-conditioned puzzle training correctly. Re-running
`build_datasets.py` Stage 5 produces the new `puzzle_fens.bin` /
`puzzle_test_fens.bin` files alongside the existing token/length files.
A loud warning is printed if puzzle data is loaded without the FEN sidecar.