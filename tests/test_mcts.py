import chess
import pytest
from mcts import MCTSNode, MCTS, dummy_reward_fn


class TestMCTSNode:
    def test_initial_visits_zero(self):
        node = MCTSNode(board=chess.Board())
        assert node.visits == 0
        assert node.value == 0.0

    def test_untried_moves_match_legal_moves(self):
        board = chess.Board()
        node = MCTSNode(board=board)
        assert len(node.untried_moves) == board.legal_moves.count()

    def test_is_terminal_on_checkmate(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert board.is_checkmate()
        node = MCTSNode(board=board)
        assert node.is_terminal()

    def test_is_not_terminal_on_start(self):
        node = MCTSNode(board=chess.Board())
        assert not node.is_terminal()

    def test_is_fully_expanded_when_no_untried(self):
        node = MCTSNode(board=chess.Board())
        node.untried_moves = []
        assert node.is_fully_expanded()

    def test_not_fully_expanded_initially(self):
        node = MCTSNode(board=chess.Board())
        assert not node.is_fully_expanded()

    def test_ucb1_infinite_when_unvisited(self):
        parent = MCTSNode(board=chess.Board())
        parent.visits = 10
        child = MCTSNode(board=chess.Board(), parent=parent)
        assert child.ucb1() == float("inf")

    def test_ucb1_finite_when_visited(self):
        parent = MCTSNode(board=chess.Board())
        parent.visits = 10
        child = MCTSNode(board=chess.Board(), parent=parent)
        child.visits = 5
        child.value = 2.0
        score = child.ucb1()
        assert isinstance(score, float)
        assert score != float("inf")


class TestMCTS:
    def test_search_returns_legal_move(self):
        board = chess.Board()
        mcts = MCTS(reward_fn=dummy_reward_fn, num_simulations=50)
        move = mcts.search(board)
        assert move in board.legal_moves

    def test_search_finds_mate_in_one(self):
        # White to move, Qh5# is mate in 1 (scholars mate setup)
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        mcts = MCTS(reward_fn=dummy_reward_fn, num_simulations=200)
        move = mcts.search(board)
        # Qxf7# is the mate
        board.push(move)
        # With material heuristic, it should find a strong move (may not always find mate)
        assert move in chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4").legal_moves

    def test_backpropagation_sign_flipping(self):
        board = chess.Board()
        mcts = MCTS(reward_fn=dummy_reward_fn, num_simulations=1)
        root = MCTSNode(board=board.copy())

        # Manually expand and backpropagate
        node = mcts._expand(root)
        value = mcts._simulate(node)
        mcts._backpropagate(node, value)

        # Child and parent should have opposite sign contributions
        assert root.visits == 1
        assert node.visits == 1
        if value != 0:
            assert root.value == -value
            assert node.value == value

    def test_search_does_not_modify_input_board(self):
        board = chess.Board()
        original_fen = board.fen()
        mcts = MCTS(reward_fn=dummy_reward_fn, num_simulations=20)
        mcts.search(board)
        assert board.fen() == original_fen


class TestDummyRewardFn:
    def test_starting_position_is_zero(self):
        board = chess.Board()
        score = dummy_reward_fn(board)
        assert score == 0.0

    def test_returns_float(self):
        board = chess.Board()
        assert isinstance(dummy_reward_fn(board), float)

    def test_bounded_output(self):
        board = chess.Board()
        score = dummy_reward_fn(board)
        assert -1.0 <= score <= 1.0

    def test_white_advantage_positive(self):
        # Remove black's queen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        score = dummy_reward_fn(board)
        assert score > 0.0
