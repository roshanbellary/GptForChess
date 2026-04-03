import math
import torch
import pytest
from train import normalize_cp, collate_fn, build_tokenizer
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


class TestBuildTokenizer:
    def test_builds_from_csv(self):
        tokenizer = build_tokenizer("data/games.csv")
        assert tokenizer.language_size > 0
        assert CLS_TOKEN in tokenizer.symbol_to_token
        assert PAD_TOKEN in tokenizer.symbol_to_token

    def test_can_encode_common_moves(self):
        tokenizer = build_tokenizer("data/games.csv")
        for move in ["e4", "d4", "Nf3", "e5"]:
            assert move in tokenizer.symbol_to_token
