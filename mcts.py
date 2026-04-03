import chess
import math
from typing import Callable

from model import PIECE_VALUES


class MCTSNode:
    def __init__(
        self,
        board: chess.Board,
        parent: "MCTSNode | None" = None,
        move: chess.Move | None = None,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.untried_moves: list[chess.Move] = list(board.legal_moves)
        self.visits: int = 0
        self.value: float = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def ucb1(self, exploration: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + explore


class MCTS:
    def __init__(
        self,
        reward_fn: Callable[[chess.Board], float],
        num_simulations: int = 800,
        exploration: float = 1.41,
    ):
        self.reward_fn = reward_fn
        self.num_simulations = num_simulations
        self.exploration = exploration

    def search(self, root_board: chess.Board) -> chess.Move:
        root = MCTSNode(board=root_board.copy())
        for _ in range(self.num_simulations):
            node = self._select(root)
            node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)
        return max(root.children, key=lambda c: c.visits).move

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and not node.is_terminal():
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried_moves and not node.is_terminal():
            move = node.untried_moves.pop()
            new_board = node.board.copy()
            new_board.push(move)
            child = MCTSNode(board=new_board, parent=node, move=move)
            node.children.append(child)
            return child
        return node

    def _simulate(self, node: MCTSNode) -> float:
        return self.reward_fn(node.board)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent


def dummy_reward_fn(board: chess.Board) -> float:
    """Material-count heuristic: positive favors white."""
    score = 0.0
    for piece_type in PIECE_VALUES:
        score += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return math.tanh(score / 10.0)
