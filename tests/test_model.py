import chess
import torch
import pytest
from model import ChessRewardModel, DummyRewardModel, PositionalEncoding


class TestChessRewardModel:
    def setup_method(self):
        self.vocab_size = 100
        self.model = ChessRewardModel(vocab_size=self.vocab_size, d_model=64, nhead=4, num_layers=2)

    def test_output_shape(self):
        token_ids = torch.randint(0, self.vocab_size, (4, 10))
        output = self.model(token_ids)
        assert output.shape == (4,)

    def test_output_bounded(self):
        token_ids = torch.randint(0, self.vocab_size, (8, 20))
        output = self.model(token_ids)
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()

    def test_with_attention_mask(self):
        token_ids = torch.randint(0, self.vocab_size, (2, 15))
        mask = torch.zeros(2, 15, dtype=torch.bool)
        mask[0, 10:] = True  # pad last 5 tokens of first sample
        mask[1, 12:] = True
        output = self.model(token_ids, attention_mask=mask)
        assert output.shape == (2,)

    def test_single_token_input(self):
        token_ids = torch.randint(0, self.vocab_size, (1, 1))
        output = self.model(token_ids)
        assert output.shape == (1,)

    def test_variable_length_sequences(self):
        short = torch.randint(0, self.vocab_size, (1, 5))
        long = torch.randint(0, self.vocab_size, (1, 50))
        out_short = self.model(short)
        out_long = self.model(long)
        assert out_short.shape == out_long.shape == (1,)


class TestDummyRewardModel:
    def test_starting_position_zero(self):
        model = DummyRewardModel()
        assert model(chess.Board()) == 0.0

    def test_returns_float(self):
        model = DummyRewardModel()
        assert isinstance(model(chess.Board()), float)

    def test_bounded(self):
        model = DummyRewardModel()
        board = chess.Board()
        score = model(board)
        assert -1.0 <= score <= 1.0

    def test_white_material_advantage(self):
        model = DummyRewardModel()
        # White has an extra queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assert model(board) > 0.0


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 50, 64)
        out = pe(x)
        assert out.shape == (2, 50, 64)

    def test_adds_to_input(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        # Output should not be all zeros (positional encoding was added)
        assert not torch.allclose(out, x)
