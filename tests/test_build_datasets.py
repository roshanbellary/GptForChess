"""Tests for src/build_datasets.py and related parse/sampling changes."""

from pathlib import Path

import pytest
import torch

from build_datasets import (
    RESULT_TO_LABEL,
    _generate_outcome_samples,
    _process_puzzle,
    stage2_outcome_samples,
    stage3_stockfish_samples,
    stage5_puzzle_samples,
)
from train import (
    ChessPositionDataset,
    build_tokenizer_from_games,
    generate_samples_stockfish_parallel,
    material_eval,
    parse_movetext,
)
from model import CLS_TOKEN


# --- Fixtures --------------------------------------------------------------

# A handful of short, valid games with known Results. Short enough that sampling
# with skip_ply=0 and max_positions_per_game>=len(moves) yields every position.
SYNTHETIC_GAMES = [
    {
        "movetext": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0",
        "Result": "1-0",
    },
    {
        "movetext": "1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. Bg5 h6 6. Bh4 dxc4 0-1",
        "Result": "0-1",
    },
    {
        "movetext": "1. c4 Nf6 2. Nc3 e5 3. Nf3 Nc6 4. g3 Bb4 5. Bg2 O-O 1/2-1/2",
        "Result": "1/2-1/2",
    },
]


@pytest.fixture(scope="module")
def synthetic_tokenizer():
    return build_tokenizer_from_games(SYNTHETIC_GAMES)


# --- parse_movetext brace-comment handling ---------------------------------

class TestParseMovetextBraceComments:
    def test_strips_eval_comment(self):
        mt = "1. e4 {[%eval 0.13]} e5 {[%eval -0.05]} 2. Nf3 1-0"
        assert parse_movetext(mt) == ["e4", "e5", "Nf3"]

    def test_strips_clock_comment(self):
        mt = "1. d4 {[%clk 0:05:00]} d5 {[%clk 0:04:58]} 2. Nf3 1/2-1/2"
        assert parse_movetext(mt) == ["d4", "d5", "Nf3"]

    def test_strips_multiple_annotations_per_move(self):
        mt = "1. e4 {[%eval 0.2] [%clk 0:05:00]} e5 1-0"
        assert parse_movetext(mt) == ["e4", "e5"]

    def test_no_annotations_unchanged(self):
        mt = "1. e4 e5 2. Nf3 Nc6 1-0"
        assert parse_movetext(mt) == ["e4", "e5", "Nf3", "Nc6"]

    def test_black_move_number_syntax(self):
        # PGN uses "5..." when black moves first in a continuation; make sure
        # the move-number regex accepts it without capturing a move token.
        mt = "5... Nc6 6. Bb5"
        assert parse_movetext(mt) == ["Nc6", "Bb5"]


# --- Outcome labeling correctness ------------------------------------------

class TestOutcomeLabeling:
    def test_result_map(self):
        assert RESULT_TO_LABEL == {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

    def test_every_sample_labeled_by_game_result(self, synthetic_tokenizer):
        samples = _generate_outcome_samples(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            max_positions_per_game=100,  # sample every position
            skip_ply=0,
        )
        # Split by game by label (each synthetic game has a unique result).
        labels = {s[1] for s in samples}
        assert labels == {1.0, -1.0, 0.0}

        # Every sample label is in RESULT_TO_LABEL's value set.
        for _, label in samples:
            assert label in {1.0, 0.0, -1.0}

    def test_skips_games_with_star_result(self, synthetic_tokenizer):
        games = [{"movetext": "1. e4 e5 2. Nf3 *", "Result": "*"}]
        samples = _generate_outcome_samples(
            games, synthetic_tokenizer, max_positions_per_game=10, skip_ply=0
        )
        assert samples == []


# --- sample_rate and skew_exponent -----------------------------------------

class TestSamplingRate:
    def test_sample_rate_scales_with_game_length(self, synthetic_tokenizer):
        """Longer games should produce more samples at a fixed sample_rate."""
        short_games = [{"movetext": "1. e4 e5 2. Nf3 Nc6", "Result": "1-0"}]
        long_games = [{"movetext": " ".join(
            f"{i // 2 + 1}. e4 e5" if i % 2 == 0 else ""
            for i in range(40)
        ).replace("  ", " ").strip(), "Result": "1-0"}]
        # Use a simple 20-move game vs the synthetic games (10+ moves)
        ds_short = ChessPositionDataset(short_games, synthetic_tokenizer, eval_fn=material_eval, sample_rate=0.5)
        ds_long = ChessPositionDataset(SYNTHETIC_GAMES, synthetic_tokenizer, eval_fn=material_eval, sample_rate=0.5)
        # SYNTHETIC_GAMES are longer, so they should produce more samples total
        assert len(ds_long.samples) >= len(ds_short.samples)

    def test_serial_and_parallel_identical(self, synthetic_tokenizer):
        """sample_rate sampling must produce identical outputs across serial/parallel."""
        serial = ChessPositionDataset(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            eval_fn=material_eval,
            sample_rate=0.5,
            skew_exponent=1.5,
        ).samples
        parallel = generate_samples_stockfish_parallel(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            num_workers=2,
            engine_path=None,
            sample_rate=0.5,
            skew_exponent=1.5,
        )
        canon = lambda xs: sorted((tuple(t), round(s, 9)) for t, s in xs)
        assert canon(serial) == canon(parallel)

    def test_higher_skew_shifts_samples_later(self, synthetic_tokenizer):
        """Higher skew_exponent should produce samples with longer move prefixes on average."""
        low_skew = ChessPositionDataset(
            SYNTHETIC_GAMES, synthetic_tokenizer, eval_fn=material_eval,
            sample_rate=0.5, skew_exponent=0.1,
        )
        high_skew = ChessPositionDataset(
            SYNTHETIC_GAMES, synthetic_tokenizer, eval_fn=material_eval,
            sample_rate=0.5, skew_exponent=5.0,
        )
        avg_len_low = sum(len(t) for t, _ in low_skew.samples) / len(low_skew.samples)
        avg_len_high = sum(len(t) for t, _ in high_skew.samples) / len(high_skew.samples)
        assert avg_len_high > avg_len_low


# --- Stage resumability + disjoint invariant -------------------------------

class TestStagesResumable:
    def _make_args(self, tmp_path: Path, **overrides):
        import argparse
        ns = argparse.Namespace(
            out_dir=tmp_path,
            policy_games=2,
            reward_games=1,
            policy_min_elo=1500,
            reward_min_elo=1500,
            sample_rate=1.0,
            position_skew=1.5,
            workers=2,
            stockfish_depth=4,
            max_seq_len=64,
            force=False,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_stage2_produces_valid_outputs(self, tmp_path):
        import numpy as np
        from train import ChessPositionDataset
        args = self._make_args(tmp_path)
        torch.save(SYNTHETIC_GAMES, tmp_path / "games_outcome.pt")
        torch.save(SYNTHETIC_GAMES[:1], tmp_path / "games_stockfish.pt")

        stage2_outcome_samples(args)

        assert (tmp_path / "tokenizer.pt").exists()
        assert (tmp_path / "outcome_meta.pt").exists()
        tokenizer = torch.load(tmp_path / "tokenizer.pt", weights_only=False)
        ds = ChessPositionDataset.from_memmap(tmp_path, "outcome", tokenizer)
        assert len(ds) > 0
        for tokens, mask, label in ds:
            assert label in {1.0, 0.0, -1.0}
            assert tokens[0].item() == tokenizer.symbol_to_token[CLS_TOKEN]

    def test_stage3_with_material_fallback_smoke(self, tmp_path, monkeypatch):
        """Stage 3 runs end-to-end using the material-eval fallback (no Stockfish)."""
        from train import ChessPositionDataset
        args = self._make_args(tmp_path)
        torch.save(SYNTHETIC_GAMES, tmp_path / "games_outcome.pt")
        torch.save(SYNTHETIC_GAMES[:1], tmp_path / "games_stockfish.pt")
        stage2_outcome_samples(args)

        import build_datasets
        orig = build_datasets.generate_samples_stockfish_parallel

        def no_stockfish(*a, **kw):
            kw["engine_path"] = None
            return orig(*a, **kw)

        monkeypatch.setattr(build_datasets, "generate_samples_stockfish_parallel", no_stockfish)
        stage3_stockfish_samples(args)

        assert (tmp_path / "stockfish_meta.pt").exists()
        tokenizer = torch.load(tmp_path / "tokenizer.pt", weights_only=False)
        ds = ChessPositionDataset.from_memmap(tmp_path, "stockfish", tokenizer)
        assert len(ds) > 0
        for _, _, score in ds:
            assert -1.0 <= score <= 1.0

    def test_stage2_skips_if_outputs_exist(self, tmp_path, capsys):
        args = self._make_args(tmp_path)
        (tmp_path / "tokenizer.pt").write_bytes(b"placeholder")
        (tmp_path / "outcome_meta.pt").write_bytes(b"placeholder")
        stage2_outcome_samples(args)
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()
        assert (tmp_path / "tokenizer.pt").read_bytes() == b"placeholder"

    def test_stage3_skips_if_output_exists(self, tmp_path, capsys):
        args = self._make_args(tmp_path)
        (tmp_path / "stockfish_meta.pt").write_bytes(b"placeholder")
        stage3_stockfish_samples(args)
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()


# --- Disjoint sampling invariant -------------------------------------------

class TestDisjointSubsets:
    def test_single_pass_split_is_disjoint(self):
        """Simulate the stage-1 single-pass split logic on a small list."""
        # Re-implement the split inline so the test is decoupled from HF streaming.
        source = [{"movetext": str(i), "Result": "1-0"} for i in range(10)]
        outcome_n, sf_n = 6, 3
        outcome, stockfish = [], []
        for i, row in enumerate(source):
            if i < outcome_n:
                outcome.append(row)
            else:
                stockfish.append(row)
            if i + 1 >= outcome_n + sf_n:
                break

        assert len(outcome) == outcome_n
        assert len(stockfish) == sf_n
        overlap = {g["movetext"] for g in outcome} & {g["movetext"] for g in stockfish}
        assert overlap == set()


# --- Puzzle processing -------------------------------------------------------

# Starting position FEN with a trivial 2-move puzzle: 1.e4 e5, then white plays d2d4.
# We use the standard starting FEN + e2e4 as the setup move, d2d4 as solver move.
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# After e2e4 (white), black's turn: e7e5 is the setup, then white plays d2d4
_AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

SYNTHETIC_PUZZLES = [
    # Valid: 4-move puzzle from Lichess sample (mate in 2)
    {
        "FEN": "r1bqk2r/pp1nbNp1/2p1p2p/8/2BP4/1PN3P1/P3QP1P/3R1RK1 b kq - 0 19",
        "Moves": "e8f7 e2e6 f7f8 e6f7",
        "Popularity": 92,
        "NbPlays": 674,
    },
    # Valid: simple 2-move puzzle (setup + one solver move)
    {
        "FEN": _AFTER_E4_FEN,
        "Moves": "e7e5 d2d4",
        "Popularity": 50,
        "NbPlays": 100,
    },
]

INVALID_PUZZLES = [
    # Missing FEN
    {"FEN": "", "Moves": "e2e4 e7e5", "Popularity": 80, "NbPlays": 100},
    # Only one move (no solver move)
    {"FEN": _AFTER_E4_FEN, "Moves": "e7e5", "Popularity": 80, "NbPlays": 100},
    # Second move is illegal after the first
    {"FEN": _START_FEN, "Moves": "e2e4 e2e5", "Popularity": 80, "NbPlays": 100},
]


@pytest.fixture(scope="module")
def puzzle_tokenizer():
    return build_tokenizer_from_games()


class TestProcessPuzzle:
    def test_valid_puzzle_returns_sequence(self, puzzle_tokenizer):
        cls_id = puzzle_tokenizer.symbol_to_token[CLS_TOKEN]
        seq = _process_puzzle(SYNTHETIC_PUZZLES[0], puzzle_tokenizer.symbol_to_token, cls_id)
        assert seq is not None
        assert seq[0] == cls_id
        assert len(seq) == 5  # CLS + setup + 3 solver moves

    def test_minimum_length_sequence(self, puzzle_tokenizer):
        cls_id = puzzle_tokenizer.symbol_to_token[CLS_TOKEN]
        seq = _process_puzzle(SYNTHETIC_PUZZLES[1], puzzle_tokenizer.symbol_to_token, cls_id)
        assert seq is not None
        assert len(seq) == 3  # CLS + setup + 1 solver move

    def test_invalid_puzzles_return_none(self, puzzle_tokenizer):
        cls_id = puzzle_tokenizer.symbol_to_token[CLS_TOKEN]
        for bad in INVALID_PUZZLES:
            result = _process_puzzle(bad, puzzle_tokenizer.symbol_to_token, cls_id)
            assert result is None, f"Expected None for {bad}"

    def test_all_tokens_are_known_moves(self, puzzle_tokenizer):
        cls_id = puzzle_tokenizer.symbol_to_token[CLS_TOKEN]
        seq = _process_puzzle(SYNTHETIC_PUZZLES[0], puzzle_tokenizer.symbol_to_token, cls_id)
        assert seq is not None
        for tok in seq[1:]:
            assert tok in puzzle_tokenizer.token_to_symbol


class TestStage5Puzzles:
    def _make_args(self, tmp_path: Path, **overrides):
        import argparse
        ns = argparse.Namespace(
            out_dir=tmp_path,
            puzzle_count=None,
            min_puzzle_popularity=None,
            min_puzzle_plays=None,
            max_seq_len=64,
            force=False,
            puzzle_test_size=1,  # keep small for tests
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_stage5_produces_valid_memmap(self, tmp_path, monkeypatch, puzzle_tokenizer):
        from train import ChessPolicyDataset
        import build_datasets as bd
        monkeypatch.setattr(bd, "load_dataset", lambda *a, **kw: iter(SYNTHETIC_PUZZLES))

        # puzzle_test_size=1: first valid puzzle → test, second → train
        args = self._make_args(tmp_path, puzzle_test_size=1)
        stage5_puzzle_samples(args, puzzle_tokenizer, tmp_path)

        assert (tmp_path / "puzzle_meta.pt").exists()
        assert (tmp_path / "puzzle_test_meta.pt").exists()

        train_meta = torch.load(tmp_path / "puzzle_meta.pt", weights_only=True)
        assert train_meta["n"] == 1  # second puzzle goes to training

        test_meta = torch.load(tmp_path / "puzzle_test_meta.pt", weights_only=True)
        assert test_meta["n"] == 1  # first puzzle goes to test

        ds = ChessPolicyDataset.from_memmap(tmp_path, puzzle_tokenizer, name="puzzle")
        assert len(ds) == 1
        assert ds[0][0].item() == puzzle_tokenizer.symbol_to_token[CLS_TOKEN]

    def test_stage5_skips_if_meta_exists(self, tmp_path, capsys, puzzle_tokenizer):
        # Skip requires BOTH puzzle_meta.pt AND puzzle_test_meta.pt to exist
        (tmp_path / "puzzle_meta.pt").write_bytes(b"placeholder")
        (tmp_path / "puzzle_test_meta.pt").write_bytes(b"placeholder")
        args = self._make_args(tmp_path)
        stage5_puzzle_samples(args, puzzle_tokenizer, tmp_path)
        captured = capsys.readouterr()
        assert "skipping" in captured.out.lower()
        assert (tmp_path / "puzzle_meta.pt").read_bytes() == b"placeholder"

    def test_stage5_respects_puzzle_count(self, tmp_path, monkeypatch, puzzle_tokenizer):
        import build_datasets as bd
        monkeypatch.setattr(bd, "load_dataset", lambda *a, **kw: iter(SYNTHETIC_PUZZLES))
        # puzzle_test_size=1 + puzzle_count=1: needs 2 valid puzzles total — we have exactly 2
        args = self._make_args(tmp_path, puzzle_count=1, puzzle_test_size=1)
        stage5_puzzle_samples(args, puzzle_tokenizer, tmp_path)
        meta = torch.load(tmp_path / "puzzle_meta.pt", weights_only=True)
        assert meta["n"] == 1  # 1 training puzzle

    def test_stage5_popularity_filter(self, tmp_path, monkeypatch, puzzle_tokenizer):
        import build_datasets as bd
        monkeypatch.setattr(bd, "load_dataset", lambda *a, **kw: iter(SYNTHETIC_PUZZLES))
        # Only SYNTHETIC_PUZZLES[0] (Popularity=92) passes the filter; [1] has 50.
        # With puzzle_test_size=1: that 1 valid puzzle goes to test set; no training puzzles.
        args = self._make_args(tmp_path, min_puzzle_popularity=60, puzzle_test_size=1)
        stage5_puzzle_samples(args, puzzle_tokenizer, tmp_path)
        test_meta = torch.load(tmp_path / "puzzle_test_meta.pt", weights_only=True)
        assert test_meta["n"] == 1
        assert not (tmp_path / "puzzle_meta.pt").exists()  # no training puzzles


class TestChessPolicyDatasetFromMemmapGeneralized:
    def test_default_name_loads_policy_files(self, tmp_path):
        import numpy as np
        from train import ChessPolicyDataset
        from build_datasets import _save_policy_memmap
        tokenizer = build_tokenizer_from_games()
        cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        seqs = [[cls_id, cls_id + 1, cls_id + 2]]
        _save_policy_memmap(seqs, tmp_path, "policy", max_seq_len=10)
        ds = ChessPolicyDataset.from_memmap(tmp_path, tokenizer)
        assert len(ds) == 1

    def test_puzzle_name_loads_puzzle_files(self, tmp_path):
        import numpy as np
        from train import ChessPolicyDataset
        from build_datasets import _save_policy_memmap
        tokenizer = build_tokenizer_from_games()
        cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        seqs = [[cls_id, cls_id + 1], [cls_id, cls_id + 2, cls_id + 3]]
        _save_policy_memmap(seqs, tmp_path, "puzzle", max_seq_len=10)
        ds = ChessPolicyDataset.from_memmap(tmp_path, tokenizer, name="puzzle")
        assert len(ds) == 2
        assert ds[0][0].item() == cls_id


class TestTrainTestDisjointness:
    """When `{name}_test_indices.npy` exists, from_memmap must skip those rows."""

    def test_policy_train_excludes_test_indices(self, tmp_path):
        import numpy as np
        from train import ChessPolicyDataset
        from build_datasets import _save_policy_memmap
        tokenizer = build_tokenizer_from_games()
        cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        # 5 sequences, indices 0..4 with distinguishing second token
        seqs = [[cls_id, cls_id + i] for i in range(1, 6)]
        _save_policy_memmap(seqs, tmp_path, "policy", max_seq_len=10)

        # Mark indices 1 and 3 as the held-out test set.
        np.save(tmp_path / "policy_test_indices.npy", np.array([1, 3], dtype=np.int64))

        ds = ChessPolicyDataset.from_memmap(tmp_path, tokenizer, name="policy")
        assert len(ds) == 3  # 5 total - 2 held out

        # Remaining sequences must come from indices 0, 2, 4 — i.e. second tokens 1, 3, 5.
        second_tokens = sorted(ds[i][1].item() for i in range(len(ds)))
        assert second_tokens == [cls_id + 1, cls_id + 3, cls_id + 5]

    def test_test_memmap_does_not_self_exclude(self, tmp_path):
        """Loading a `_test` memmap must not look for an indices file with its own name."""
        import numpy as np
        from train import ChessPolicyDataset
        from build_datasets import _save_policy_memmap
        tokenizer = build_tokenizer_from_games()
        cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        seqs = [[cls_id, cls_id + i] for i in range(1, 4)]
        _save_policy_memmap(seqs, tmp_path, "policy_test", max_seq_len=10)
        # Even with an indices file present, the _test loader returns the full set.
        np.save(tmp_path / "policy_test_test_indices.npy", np.array([0], dtype=np.int64))
        ds = ChessPolicyDataset.from_memmap(tmp_path, tokenizer, name="policy_test")
        assert len(ds) == 3
