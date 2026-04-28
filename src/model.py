import math
import torch
import torch.nn as nn
import chess

from tokenizer import Tokenizer

CLS_TOKEN = "[CLS]"
PAD_TOKEN = "[PAD]"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ChessRewardModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) int tensor with CLS prepended
            attention_mask: (batch, seq_len) bool tensor, True where padded
        Returns:
            (batch,) float tensor bounded to [-1, 1]
        """
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        cls_hidden = x[:, 0, :]  # CLS token at position 0
        reward = self.reward_head(cls_hidden).squeeze(-1)
        return torch.tanh(reward)


class DummyRewardModel:
    """Material-count heuristic for MCTS testing."""
    def __call__(self, board: chess.Board) -> float:
        score = 0.0
        for piece_type in PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        return math.tanh(score / 10.0)


class RewardModelInference:
    """Wraps ChessRewardModel + Tokenizer for use in MCTS."""
    def __init__(self, model: ChessRewardModel, tokenizer: Tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.pad_id = tokenizer.symbol_to_token[PAD_TOKEN]
        self.model.eval()

    @torch.no_grad()
    def __call__(self, board: chess.Board, max_seq_len: int = 128) -> float:
        temp_board = chess.Board()
        moves_san = []
        for move in board.move_stack:
            moves_san.append(temp_board.san(move))
            temp_board.push(move)

        # Keep the most recent moves to stay within the training sequence length.
        # CLS occupies position 0, so cap move history at max_seq_len - 1.
        moves_san = moves_san[-(max_seq_len - 1):]
        token_ids = [self.cls_id] + self.tokenizer.encode_moves(moves_san)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        reward = self.model(token_tensor)
        return reward.item()
