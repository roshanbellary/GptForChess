import math
import os
import shutil
import time
import torch
import pytest
import pandas as pd
from train import (
    normalize_cp,
    collate_fn,
    build_tokenizer_from_games,
    parse_movetext,
    ChessPositionDataset,
    generate_samples_stockfish_parallel,
    material_eval,
)
from model import CLS_TOKEN, PAD_TOKEN


class TestNormalizeCp:
    def test_zero(self):
        assert normalize_cp(0) == 0.0

    def test_positive(self):
        result = normalize_cp(400)
        assert abs(result - math.tanh(1.0)) < 1e-6

    def test_negative(self):
        result = normalize_cp(-400)
        assert abs(result - math.tanh(-1.0)) < 1e-6

    def test_bounded(self):
        for cp in [-10000, -100, 0, 100, 10000]:
            assert -1.0 <= normalize_cp(cp) <= 1.0


class TestCollateFn:
    def test_pads_to_max_length(self):
        batch = [
            (torch.tensor([1, 2, 3]), 0.5),
            (torch.tensor([4, 5]), -0.3),
        ]
        padded, mask, labels = collate_fn(batch)
        assert padded.shape == (2, 3)
        assert mask.shape == (2, 3)

    def test_mask_true_on_padding(self):
        batch = [
            (torch.tensor([1, 2, 3]), 0.5),
            (torch.tensor([4, 5]), -0.3),
        ]
        padded, mask, labels = collate_fn(batch)
        # Second sample padded at position 2
        assert mask[1, 2].item() == True
        assert mask[1, 0].item() == False
        assert mask[0, 0].item() == False

    def test_labels_tensor(self):
        batch = [
            (torch.tensor([1]), 0.5),
            (torch.tensor([2]), -0.3),
        ]
        _, _, labels = collate_fn(batch)
        assert labels.shape == (2,)
        assert labels.dtype == torch.float


class TestParseMovetext:
    def test_basic(self):
        mt = "1. d4 d5 2. Nf3 Nf6 1-0"
        assert parse_movetext(mt) == ["d4", "d5", "Nf3", "Nf6"]

    def test_draw_result(self):
        mt = "1. e4 e5 1/2-1/2"
        assert parse_movetext(mt) == ["e4", "e5"]

    def test_empty(self):
        assert parse_movetext("") == []

    def test_star_result(self):
        mt = "1. d4 *"
        assert parse_movetext(mt) == ["d4"]


class TestBuildTokenizer:
    def _games_from_csv(self, csv_path="data/games.csv"):
        """Convert local CSV games to the HuggingFace row format."""
        df = pd.read_csv(csv_path)
        games = []
        for _, row in df.iterrows():
            moves_str = row.get("moves", "")
            if not isinstance(moves_str, str) or not moves_str.strip():
                continue
            # Convert space-separated SAN to PGN movetext
            sans = moves_str.split()
            parts = []
            for i, san in enumerate(sans):
                if i % 2 == 0:
                    parts.append(f"{i // 2 + 1}.")
                parts.append(san)
            games.append({"movetext": " ".join(parts)})
        return games

    def test_builds_from_games(self):
        games = self._games_from_csv()
        tokenizer = build_tokenizer_from_games(games)
        assert tokenizer.language_size > 0
        assert CLS_TOKEN in tokenizer.symbol_to_token
        assert PAD_TOKEN in tokenizer.symbol_to_token

    def test_can_encode_common_moves(self):
        games = self._games_from_csv()
        tokenizer = build_tokenizer_from_games(games)
        for move in ["e2e4", "d2d4", "g1f3", "e7e5"]:
            assert move in tokenizer.symbol_to_token


# ---------------------------------------------------------------------------
# Helpers shared by parallel-sampler tests.
# ---------------------------------------------------------------------------

def _games_from_csv(csv_path="data/games.csv", limit=500):
    """Load a subset of local CSV games and convert to HF-row format."""
    df = pd.read_csv(csv_path).head(limit)
    games = []
    for _, row in df.iterrows():
        moves_str = row.get("moves", "")
        if not isinstance(moves_str, str) or not moves_str.strip():
            continue
        sans = moves_str.split()
        parts = []
        for i, san in enumerate(sans):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}.")
            parts.append(san)
        games.append({"movetext": " ".join(parts)})
    return games


def _canonicalize(samples):
    """Convert a list of (token_ids, score) into a sorted, hashable form
    so two sample sets produced in different orders can be compared."""
    return sorted((tuple(t), round(s, 9)) for t, s in samples)


@pytest.fixture(scope="module")
def shared_tokenizer():
    """Module-scoped tokenizer trained on a subset of games.csv."""
    games = _games_from_csv(limit=500)
    return build_tokenizer_from_games(games)


@pytest.fixture(scope="module")
def shared_games():
    """A small set of real games reused across parallel tests."""
    return _games_from_csv(limit=30)


class TestParallelSampleGeneration:
    """Tests for generate_samples_stockfish_parallel.

    The first three tests use engine_path=None so they exercise the full
    multiprocessing machinery (spawn, pool init, imap_unordered) without
    requiring a Stockfish binary — useful when this runs in CI.

    The Stockfish-backed tests are skipped when the binary is absent.
    """

    def test_parallel_produces_samples(self, shared_tokenizer, shared_games):
        """Smoke test: pool spins up, workers run, samples come back."""
        samples = generate_samples_stockfish_parallel(
            shared_games,
            shared_tokenizer,
            num_workers=4,
            engine_path=None,  # material_eval fallback, no Stockfish needed
            sample_rate=0.5,
        )
        assert len(samples) > 0, "parallel pool produced no samples"
        for token_ids, score in samples:
            assert isinstance(token_ids, list)
            assert len(token_ids) >= 2  # CLS + at least one move
            assert all(isinstance(t, int) for t in token_ids)
            assert -1.0 <= score <= 1.0

    def test_parallel_matches_serial(self, shared_tokenizer, shared_games):
        """Parallel output is identical to serial output (same eval, same seeds)."""
        serial_ds = ChessPositionDataset(
            shared_games,
            shared_tokenizer,
            eval_fn=material_eval,
            sample_rate=0.5,
        )
        parallel = generate_samples_stockfish_parallel(
            shared_games,
            shared_tokenizer,
            num_workers=4,
            engine_path=None,
            sample_rate=0.5,
        )
        assert _canonicalize(serial_ds.samples) == _canonicalize(parallel)

    def test_worker_count_invariant(self, shared_tokenizer, shared_games):
        """Different worker counts must produce the same sample set."""
        s1 = generate_samples_stockfish_parallel(
            shared_games, shared_tokenizer,
            num_workers=1, engine_path=None, sample_rate=0.5,
        )
        s8 = generate_samples_stockfish_parallel(
            shared_games, shared_tokenizer,
            num_workers=8, engine_path=None, sample_rate=0.5,
        )
        assert _canonicalize(s1) == _canonicalize(s8)

    @pytest.mark.skipif(
        shutil.which("stockfish") is None,
        reason="Stockfish binary not on PATH",
    )
    def test_parallel_stockfish_end_to_end(self, shared_tokenizer, shared_games):
        """End-to-end parallel run with real Stockfish at shallow depth."""
        samples = generate_samples_stockfish_parallel(
            shared_games[:8],
            shared_tokenizer,
            num_workers=4,
            stockfish_depth=4,  # shallow for test speed
            sample_rate=0.15,
        )
        assert len(samples) > 0
        for token_ids, score in samples:
            assert len(token_ids) >= 2
            assert -1.0 <= score <= 1.0

    @pytest.mark.skipif(
        shutil.which("stockfish") is None,
        reason="Stockfish binary not on PATH",
    )
    def test_parallel_faster_than_serial(self, shared_tokenizer):
        """Parallel (4 workers) must be meaningfully faster than serial (1 worker).

        Uses real Stockfish at depth 12 across ~300 positions. Smaller
        workloads are dominated by spawn startup (~2-3s per worker for the
        heavy import chain) and produce misleading numbers.

        Gated behind RUN_PERF_TESTS=1 because it takes ~25-30s.
        """
        if os.getenv("RUN_PERF_TESTS") != "1":
            pytest.skip("set RUN_PERF_TESTS=1 to run the perf comparison")

        games = _games_from_csv(limit=100)

        t0 = time.time()
        _ = generate_samples_stockfish_parallel(
            games, shared_tokenizer,
            num_workers=1, stockfish_depth=12, sample_rate=0.15,
            chunksize=2, progress_every=0,
        )
        serial_time = time.time() - t0

        t0 = time.time()
        _ = generate_samples_stockfish_parallel(
            games, shared_tokenizer,
            num_workers=4, stockfish_depth=12, sample_rate=0.15,
            chunksize=2, progress_every=0,
        )
        parallel_time = time.time() - t0

        speedup = serial_time / parallel_time
        print(
            f"\n  serial(1w)={serial_time:.2f}s  parallel(4w)={parallel_time:.2f}s  "
            f"speedup={speedup:.2f}x"
        )
        # Conservative floor: 4 workers should be at least 1.4x faster
        # than 1 worker at this scale (expected ~1.6-2.0x from benchmarks).
        # At full training scale spawn overhead amortizes to zero and the
        # speedup approaches N_workers.
        assert speedup > 1.4, f"expected >1.4x speedup, got {speedup:.2f}x"
