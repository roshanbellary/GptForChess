import chess
import math
from typing import Callable

from model import PIECE_VALUES


def dummy_reward_fn(board: chess.Board) -> float:
    """Material-count heuristic: positive favors white."""
    score = 0.0
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return math.tanh(score / 10.0)


class MinimaxSearch:
    """Minimax search with top-N move pruning.

    At each node, evaluates all legal moves with the reward function,
    keeps the top N candidates, and recurses to the given depth.
    Alternates between maximizing (white) and minimizing (black).
    """

    def __init__(
        self,
        reward_fn: Callable[[chess.Board], float],
        depth: int = 3,
        top_n: int = 5,
    ):
        self.reward_fn = reward_fn
        self.depth = depth
        self.top_n = top_n

    def search(self, board: chess.Board) -> chess.Move:
        """Return the best move for the current side to play."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        if len(legal_moves) == 1:
            return legal_moves[0]

        maximizing = board.turn == chess.WHITE

        # Score every legal move with a shallow reward evaluation
        scored_moves = []
        for move in legal_moves:
            board.push(move)
            score = self.reward_fn(board)
            board.pop()
            scored_moves.append((score, move))

        # Keep top N candidates (best for current side)
        scored_moves.sort(key=lambda x: x[0], reverse=maximizing)
        candidates = scored_moves[: self.top_n]

        # Recurse on each candidate to find the best
        best_move = candidates[0][1]
        best_value = float("-inf") if maximizing else float("inf")

        for _, move in candidates:
            board.push(move)
            value = self._minimax(board, self.depth - 1, not maximizing)
            board.pop()

            if maximizing and value > best_value:
                best_value = value
                best_move = move
            elif not maximizing and value < best_value:
                best_value = value
                best_move = move

        return best_move

    def _minimax(self, board: chess.Board, depth: int, maximizing: bool) -> float:
        if depth <= 0 or board.is_game_over():
            return self._terminal_eval(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._terminal_eval(board)

        # Score all moves, keep top N for the current side
        scored_moves = []
        for move in legal_moves:
            board.push(move)
            score = self.reward_fn(board)
            board.pop()
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: x[0], reverse=maximizing)
        candidates = scored_moves[: self.top_n]

        if maximizing:
            best = float("-inf")
            for _, move in candidates:
                board.push(move)
                best = max(best, self._minimax(board, depth - 1, False))
                board.pop()
            return best
        else:
            best = float("inf")
            for _, move in candidates:
                board.push(move)
                best = min(best, self._minimax(board, depth - 1, True))
                board.pop()
            return best

    def _terminal_eval(self, board: chess.Board) -> float:
        """Evaluate a terminal or leaf node."""
        if board.is_checkmate():
            # The side to move is checkmated
            return -1.0 if board.turn == chess.WHITE else 1.0
        if board.is_game_over():
            return 0.0
        return self.reward_fn(board)
