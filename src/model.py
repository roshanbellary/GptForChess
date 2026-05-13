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

BOARD_PLANES = 19

def board_to_planes(board: chess.Board) -> torch.Tensor:
    """chess.Board -> (19, 8, 8) float tensor."""
    planes = torch.zeros(BOARD_PLANES, 8, 8, dtype=torch.float32)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]
    piece_to_plane = {(piece, color) : 6 * color_num + piece_num  for piece_num, piece in enumerate(pieces) for color_num, color in enumerate(colors)}

    for sq, piece in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)

        planes[piece_to_plane[(piece.piece_type, piece.color)], r, c] = 1.0
    
    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)
    
    if board.has_kingside_castling_rights(chess.WHITE):  planes[13].fill_(1.0)                   
    if board.has_queenside_castling_rights(chess.WHITE): planes[14].fill_(1.0)                   
    if board.has_kingside_castling_rights(chess.BLACK):  planes[15].fill_(1.0)                   
    if board.has_queenside_castling_rights(chess.BLACK): planes[16].fill_(1.0)                   
    if board.ep_square is not None:                                                              
        r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        planes[17, r, c] = 1.0                                                                   
    planes[18].fill_(min(board.halfmove_clock, 100) / 100.0)
    
    return planes

def _group_norm(channels: int, groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return torch.relu(h + x)


class BoardCNN(nn.Module):
    """(B, 19, 8, 8) -> (B, d_model): one context-rich vector per board."""
    def __init__(self, d_model: int, channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, channels, 3, padding=1, bias=False),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, d_model)

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        x = self.stem(planes)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)   # (B, channels)
        return self.proj(x)           # (B, d_model)


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
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        dim_feedforward: int = 3072,
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
    
class ChessPolicyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 8,
        dim_feedforward: int = 3072,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        cnn_channels: int = 128,
        cnn_blocks: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.board_cnn = BoardCNN(d_model, cnn_channels, cnn_blocks)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prob_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        board_planes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) int tensor with CLS as BOS at position 0
            board_planes: (batch, 19, 8, 8) float tensor of the current board
            attention_mask: (batch, seq_len) bool tensor, True where padded
        Returns:
            (batch, seq_len, vocab_size) raw logits at every position
        """
        move_emb = self.token_embedding(token_ids)              # (B, T, d)
        board_emb = self.board_cnn(board_planes).unsqueeze(1)   # (B, 1, d)
        # Replace position-0 (CLS) with the board summary so the move sequence
        # is conditioned on the current state without growing the sequence.
        x = torch.cat([board_emb, move_emb[:, 1:, :]], dim=1)   # (B, T, d)
        x = self.pos_encoding(x)
        seq_len = token_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=token_ids.device)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=attention_mask)
        return self.prob_head(x)  # (batch, seq_len, vocab_size)


class DummyRewardModel:
    """Material-count heuristic for MCTS testing."""
    def __call__(self, board: chess.Board) -> float:
        score = 0.0
        for piece_type in PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        return math.tanh(score / 10.0)


class RewardModelInference:
    """Wraps ChessRewardModel + Tokenizer for use in minimax"""
    def __init__(self, model: ChessRewardModel, tokenizer: Tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.pad_id = tokenizer.symbol_to_token[PAD_TOKEN]
        self.model.eval()

    @torch.no_grad()
    def __call__(self, board: chess.Board, max_seq_len: int = 128) -> float:
        moves_uci = [move.uci() for move in board.move_stack]

        # Keep the most recent moves to stay within the training sequence length.
        # CLS occupies position 0, so cap move history at max_seq_len - 1.
        moves_uci = moves_uci[-(max_seq_len - 1):]
        token_ids = [self.cls_id] + self.tokenizer.encode_moves(moves_uci)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        reward = self.model(token_tensor)
        return reward.item()

class PolicyModelInference:
    """Wraps ChessPolicyModel + Tokenizer"""

    def __init__(self, model: ChessPolicyModel, tokenizer: Tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.pad_id = tokenizer.symbol_to_token[PAD_TOKEN]
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, board: chess.Board, max_seq_len: int = 128) -> str:
        moves_uci = [move.uci() for move in board.move_stack]

        moves_uci = moves_uci[-(max_seq_len - 1):]
        token_ids = [self.cls_id] + self.tokenizer.encode_moves(moves_uci)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        planes = board_to_planes(board).unsqueeze(0).to(self.device)

        logits = self.model(token_tensor, planes)  # (1, seq_len, vocab_size)
        last_logits = logits[0, -1]                # (vocab_size,) — last position has full history

        legal_move_ids = [self.tokenizer.symbol_to_token[move.uci()] for move in board.legal_moves]
        mask = torch.full((self.tokenizer.language_size,), float('-inf'), device=self.device)
        mask[legal_move_ids] = 0.0
        best_move_idx = (last_logits + mask).argmax().item()

        return self.tokenizer.token_to_symbol[best_move_idx]