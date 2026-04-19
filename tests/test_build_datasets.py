"""Tests for src/build_datasets.py and related parse/sampling changes."""

from pathlib import Path

import pytest
import torch

from build_datasets import (
    RESULT_TO_LABEL,
    _generate_outcome_samples,
    stage2_outcome_samples,
    stage3_stockfish_samples,
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


# --- skip_ply in both serial + parallel paths ------------------------------

class TestSkipPly:
    def test_serial_respects_skip_ply(self, synthetic_tokenizer):
        """Every sampled position's move-prefix length must be > skip_ply."""
        skip_ply = 5
        ds = ChessPositionDataset(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            eval_fn=material_eval,
            max_positions_per_game=100,
            skip_ply=skip_ply,
        )
        # token_ids = [CLS] + move tokens. sampled at index i means prefix length = i+1.
        # We require i >= skip_ply, i.e. len(move_tokens) >= skip_ply + 1 (excluding CLS).
        for token_ids, _ in ds.samples:
            move_tokens = token_ids[1:]  # strip CLS
            assert len(move_tokens) >= skip_ply + 1

    def test_parallel_respects_skip_ply(self, synthetic_tokenizer):
        skip_ply = 5
        samples = generate_samples_stockfish_parallel(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            num_workers=2,
            engine_path=None,  # material fallback, no Stockfish needed
            max_positions_per_game=100,
            skip_ply=skip_ply,
        )
        for token_ids, _ in samples:
            move_tokens = token_ids[1:]
            assert len(move_tokens) >= skip_ply + 1

    def test_serial_and_parallel_identical_with_skip_ply(self, synthetic_tokenizer):
        """skip_ply must produce identical outputs across serial/parallel paths."""
        skip_ply = 3
        serial = ChessPositionDataset(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            eval_fn=material_eval,
            max_positions_per_game=100,
            skip_ply=skip_ply,
        ).samples
        parallel = generate_samples_stockfish_parallel(
            SYNTHETIC_GAMES,
            synthetic_tokenizer,
            num_workers=2,
            engine_path=None,
            max_positions_per_game=100,
            skip_ply=skip_ply,
        )
        canon = lambda xs: sorted((tuple(t), round(s, 9)) for t, s in xs)
        assert canon(serial) == canon(parallel)


# --- Stage resumability + disjoint invariant -------------------------------

class TestStagesResumable:
    def _make_args(self, tmp_path: Path, **overrides):
        import argparse
        ns = argparse.Namespace(
            out_dir=tmp_path,
            outcome_games=2,
            stockfish_games=1,
            min_elo=1500,
            max_positions_per_game=100,
            skip_ply=0,
            sf_skip_ply=0,
            workers=2,
            stockfish_depth=4,
            # Use a small vocab target so BPE doesn't try to merge past the
            # tiny synthetic corpus's unique-token count (build_tokenizer_from_games
            # uses max(unique, max_language_size); we want the unique count to win).
            max_language_size=10,
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
