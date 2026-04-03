# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A chess AI project combining a BERT-style reward model with Monte Carlo Tree Search. A custom BPE tokenizer handles chess move sequences, the reward model evaluates board positions from move history, and MCTS searches for optimal moves using the reward model (or a material-count heuristic).

## Commands

- **Install dependencies:** `poetry install --no-root`
- **Run all tests:** `poetry run pytest`
- **Run a single test:** `poetry run pytest tests/test_mcts.py::TestMCTS::test_search_returns_legal_move`
- **Run tests with output:** `poetry run pytest -v -s`
- **Train reward model:** `poetry run python train.py` (uses material eval as mock; Stockfish not yet integrated)

## Architecture

### Core modules
- `tokenizer.py` — `Tokenizer` (BPE tokenizer for chess move notation) and `DataLoader` (file reader). The tokenizer supports both character-level `encode()`/`decode()` and move-level `encode_moves()` for chess SAN strings. `add_special_tokens()` registers `[CLS]`/`[PAD]` tokens.
- `model.py` — `ChessRewardModel` (BERT-style TransformerEncoder → CLS token → linear head → tanh, outputs scalar in [-1,1]). `DummyRewardModel` (material-count heuristic for testing). `RewardModelInference` (wraps model+tokenizer, takes `chess.Board` → float, for use in MCTS).
- `mcts.py` — `MCTSNode` (tree node with UCB1 selection) and `MCTS` (select/expand/simulate/backpropagate). Uses negamax sign-flipping since evaluation is always from white's perspective. `dummy_reward_fn` for standalone testing.
- `train.py` — `ChessPositionDataset` (loads CSV, replays moves on board, evaluates positions), `build_tokenizer()` (builds move-level tokenizer from dataset), training loop with MSE loss. Currently uses `material_eval` as placeholder; `stockfish_eval` is stubbed with TODO.

### Data flow
1. `build_tokenizer()` collects all SAN moves from `data/games.csv`, trains tokenizer with moves as atomic symbols
2. `ChessPositionDataset` replays games, samples positions, evaluates with `eval_fn`, tokenizes move prefixes
3. Model receives `[CLS] + move_tokens` with padding mask, outputs scalar reward
4. MCTS accepts any `Callable[[chess.Board], float]` as reward function — works with both `DummyRewardModel` and `RewardModelInference`

### Key design decisions
- Reward model output is tanh-bounded to [-1, 1] (positive = white advantage)
- MCTS backpropagation flips sign at each level (negamax for zero-sum game)
- Stockfish integration is stubbed — currently uses material count heuristic everywhere. When Stockfish is installed, update `stockfish_eval()` in `train.py` and pass it as `eval_fn` to the dataset.

## Data

- `data/games.csv` — Lichess dataset with space-separated SAN moves (e.g., `"d4 d5 c4 c6 Nf3"`), ratings, victory status
- The tokenizer treats each chess move as an atomic symbol (no sub-move BPE splitting for model training)
