import chess
import pytest
from mcts import MinimaxSearch, dummy_reward_fn


class TestMinimaxSearch:
    def test_search_returns_legal_move(self):
        board = chess.Board()
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=2, top_n=5)
        move = engine.search(board)
        assert move in board.legal_moves

    def test_search_finds_mate_in_one(self):
        # White to move, Qxf7# is mate in 1
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=1, top_n=10)
        move = engine.search(board)
        board.push(move)
        assert board.is_checkmate()

    def test_search_does_not_modify_input_board(self):
        board = chess.Board()
        original_fen = board.fen()
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=2, top_n=5)
        engine.search(board)
        assert board.fen() == original_fen

    def test_returns_only_move_when_one_legal(self):
        # King has only one escape square
        board = chess.Board("k7/8/1K6/8/8/8/8/1R6 b - - 0 1")
        legal = list(board.legal_moves)
        if len(legal) == 1:
            engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=3, top_n=5)
            move = engine.search(board)
            assert move == legal[0]

    def test_minimax_prefers_capture(self):
        # White queen on e4 can capture undefended black queen on d5
        board = chess.Board("rnb1kbnr/pppppppp/8/3q4/4Q3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1")
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=1, top_n=10)
        move = engine.search(board)
        assert board.is_capture(move)

    def test_depth_zero_still_works(self):
        board = chess.Board()
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=0, top_n=5)
        move = engine.search(board)
        assert move in board.legal_moves

    def test_black_to_move(self):
        board = chess.Board()
        board.push_san("e4")
        engine = MinimaxSearch(reward_fn=dummy_reward_fn, depth=2, top_n=5)
        move = engine.search(board)
        assert move in board.legal_moves


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
