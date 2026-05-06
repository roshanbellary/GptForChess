#!/usr/bin/env python3
"""Pygame chess demo — drag pieces to play against GptForChess.

Defaults to the canonical Experiment 4 checkpoints. Run from repo root with no
arguments to use them, or override any path explicitly:

    PYTHONPATH=src poetry run python demo.py
    PYTHONPATH=src poetry run python demo.py \
        --policy-model experiments/experiment_4/policy_model_phase2a.pt \
        --reward-model experiments/experiment_4/reward_model.pt \
        --tokenizer data/tokenizer.pt
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import argparse
import threading
import chess
import torch
import pygame

# ── board colours ──────────────────────────────────────────────────────────
LIGHT   = (240, 217, 181)
DARK    = (181, 136,  99)
HILITE  = (205, 210,  56)   # selected square
LEGAL   = (130, 151,  98)   # legal-move dot
CHECK   = (220,  50,  50)   # king-in-check tint
BG      = ( 40,  40,  40)
TEXT_FG = (220, 220, 220)
BAR_W   = (255, 255, 255)
BAR_B   = ( 30,  30,  30)
BAR_BG  = ( 80,  80,  80)
AI_CLR  = (100, 180, 100)
ERR_CLR = (220,  80,  80)

SQ   = 90        # pixels per square
BORD = SQ * 8    # board pixel size
W    = BORD + 220
H    = BORD + 40

PIECE_CHARS = {
    (chess.KING,   chess.WHITE): '♔', (chess.QUEEN,  chess.WHITE): '♕',
    (chess.ROOK,   chess.WHITE): '♖', (chess.BISHOP, chess.WHITE): '♗',
    (chess.KNIGHT, chess.WHITE): '♘', (chess.PAWN,   chess.WHITE): '♙',
    (chess.KING,   chess.BLACK): '♚', (chess.QUEEN,  chess.BLACK): '♛',
    (chess.ROOK,   chess.BLACK): '♜', (chess.BISHOP, chess.BLACK): '♝',
    (chess.KNIGHT, chess.BLACK): '♞', (chess.PAWN,   chess.BLACK): '♟',
}


def sq_to_xy(sq: int, flipped: bool) -> tuple[int, int]:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if flipped:
        col, row = 7 - file, rank
    else:
        col, row = file, 7 - rank
    return col * SQ + 20, row * SQ + 20


def xy_to_sq(x: int, y: int, flipped: bool) -> int | None:
    col = (x - 20) // SQ
    row = (y - 20) // SQ
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    if flipped:
        return chess.square(7 - col, row)
    else:
        return chess.square(col, 7 - row)


def draw_board(surf, board, piece_font, label_font, flipped,
               selected, legal_dests, drag_sq, drag_pos, ai_move,
               eval_score, status_msg, status_color):

    surf.fill(BG)

    # squares
    for sq in chess.SQUARES:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        col  = (7 - file) if flipped else file
        row  = rank       if flipped else (7 - rank)
        x, y = col * SQ + 20, row * SQ + 20
        light = (file + rank) % 2 == 1
        color = LIGHT if light else DARK

        # tints
        if board.is_check() and board.king(board.turn) == sq:
            color = CHECK
        elif sq == selected:
            color = HILITE
        elif ai_move and sq in (ai_move.from_square, ai_move.to_square):
            color = (100, 160, 220)

        pygame.draw.rect(surf, color, (x, y, SQ, SQ))

        # legal move dots
        if sq in legal_dests:
            cx, cy = x + SQ // 2, y + SQ // 2
            if board.piece_at(sq):
                pygame.draw.rect(surf, LEGAL, (x, y, SQ, SQ), 5)
            else:
                pygame.draw.circle(surf, LEGAL, (cx, cy), SQ // 7)

    # rank / file labels
    files = 'hgfedcba' if flipped else 'abcdefgh'
    ranks = '12345678' if flipped else '87654321'
    for i in range(8):
        fl = label_font.render(files[i], True, DARK if i % 2 == 0 else LIGHT)
        surf.blit(fl, (20 + i * SQ + 3, 20 + 7 * SQ + SQ - 14))
        rl = label_font.render(ranks[i], True, DARK if i % 2 == 0 else LIGHT)
        surf.blit(rl, (20 + 3, 20 + i * SQ + 3))

    # pieces (skip dragged piece)
    for sq in chess.SQUARES:
        if sq == drag_sq:
            continue
        piece = board.piece_at(sq)
        if not piece:
            continue
        x, y = sq_to_xy(sq, flipped)
        glyph = piece_font.render(PIECE_CHARS[(piece.piece_type, piece.color)], True,
                                  (255, 255, 255) if piece.color == chess.WHITE else (20, 20, 20))
        # shadow
        surf.blit(piece_font.render(PIECE_CHARS[(piece.piece_type, piece.color)], True, (0, 0, 0)),
                  (x + SQ // 2 - glyph.get_width() // 2 + 2,
                   y + SQ // 2 - glyph.get_height() // 2 + 2))
        surf.blit(glyph, (x + SQ // 2 - glyph.get_width() // 2,
                          y + SQ // 2 - glyph.get_height() // 2))

    # dragged piece follows cursor
    if drag_sq is not None and drag_pos:
        piece = board.piece_at(drag_sq)
        if piece:
            glyph = piece_font.render(PIECE_CHARS[(piece.piece_type, piece.color)], True,
                                      (255, 255, 255) if piece.color == chess.WHITE else (20, 20, 20))
            surf.blit(glyph, (drag_pos[0] - glyph.get_width() // 2,
                               drag_pos[1] - glyph.get_height() // 2))

    # ── side panel ──────────────────────────────────────────────────────────
    px = BORD + 30

    # eval bar
    bar_h = BORD
    bar_x, bar_y = px, 20
    pygame.draw.rect(surf, BAR_BG, (bar_x, bar_y, 24, bar_h))
    clamped = max(-1.0, min(1.0, eval_score))
    white_h = int((clamped + 1.0) / 2.0 * bar_h)
    pygame.draw.rect(surf, BAR_B, (bar_x, bar_y,          24, bar_h - white_h))
    pygame.draw.rect(surf, BAR_W, (bar_x, bar_y + bar_h - white_h, 24, white_h))
    score_lbl = label_font.render(f'{eval_score:+.2f}', True, TEXT_FG)
    surf.blit(score_lbl, (bar_x, bar_y + bar_h + 4))

    # status message
    if status_msg:
        msg_surf = label_font.render(status_msg, True, status_color)
        surf.blit(msg_surf, (px + 34, 20))

    # move history (last 12 half-moves)
    hist = []
    tmp = board.copy()
    moves = list(tmp.move_stack)
    tmp2 = chess.Board()
    sans = []
    for m in moves:
        sans.append(tmp2.san(m))
        tmp2.push(m)

    move_y = 60
    for i in range(max(0, len(sans) - 12), len(sans), 2):
        num = i // 2 + 1
        w_san = sans[i] if i < len(sans) else ''
        b_san = sans[i + 1] if i + 1 < len(sans) else ''
        line = label_font.render(f'{num}. {w_san}  {b_san}', True, TEXT_FG)
        surf.blit(line, (px + 34, move_y))
        move_y += 18

    # turn indicator
    turn_str = 'White to move' if board.turn == chess.WHITE else 'Black to move'
    turn_surf = label_font.render(turn_str, True,
                                  (240, 240, 240) if board.turn == chess.WHITE else (160, 160, 160))
    surf.blit(turn_surf, (px + 34, H - 30))

    pygame.display.flip()


def promote_dialog(surf, label_font, color: chess.Color) -> chess.PieceType:
    """Blocking promotion picker — returns chosen piece type."""
    options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    labels  = ['Queen', 'Rook', 'Bishop', 'Knight']
    rects   = []
    bx, by  = W // 2 - 160, H // 2 - 30
    for i, lbl in enumerate(labels):
        r = pygame.Rect(bx + i * 82, by, 78, 50)
        rects.append(r)

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for i, r in enumerate(rects):
                    if r.collidepoint(ev.pos):
                        return options[i]

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))
        for i, (r, lbl) in enumerate(zip(rects, labels)):
            pygame.draw.rect(surf, LIGHT if i % 2 == 0 else DARK, r, border_radius=6)
            txt = label_font.render(lbl, True, (30, 30, 30))
            surf.blit(txt, txt.get_rect(center=r.center))
        pygame.display.flip()

    return chess.QUEEN


def parse_args():
    parser = argparse.ArgumentParser(description='GptForChess pygame demo')
    parser.add_argument('--policy-model', type=str,
                        default='experiments/experiment_4/policy_model_phase2a.pt',
                        help='Path to ChessPolicyModel state_dict (used to choose AI moves). '
                             'Defaults to the Phase 2a checkpoint from Experiment 4 — the strong '
                             'games-only baseline (top-1 65.4%, top-5 81.9%).')
    parser.add_argument('--reward-model', type=str,
                        default='experiments/experiment_4/reward_model.pt',
                        help='Path to ChessRewardModel state_dict (used for the eval bar). '
                             'Defaults to the Experiment 4 reward model.')
    parser.add_argument('--tokenizer', type=str, default='data/tokenizer.pt',
                        help='Path to the shared Tokenizer object')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device for inference (cpu / cuda / mps)')
    return parser.parse_args()


def main():
    from model import (
        ChessRewardModel, ChessPolicyModel,
        RewardModelInference, PolicyModelInference,
    )

    args = parse_args()

    pygame.init()
    surf = pygame.display.set_mode((W, H))
    pygame.display.set_caption('GptForChess')

    piece_font = pygame.font.SysFont('segoeuisymbol,applesymbols,dejavusans', 62)
    label_font = pygame.font.SysFont('helveticaneue,arial', 14)

    # loading screen
    surf.fill(BG)
    loading = label_font.render('Loading models...', True, TEXT_FG)
    surf.blit(loading, loading.get_rect(center=(W // 2, H // 2)))
    pygame.display.flip()

    tokenizer = torch.load(args.tokenizer, weights_only=False)

    reward_model = ChessRewardModel(vocab_size=tokenizer.language_size)
    reward_model.load_state_dict(
        torch.load(args.reward_model, map_location=args.device, weights_only=False)
    )
    reward_model.eval()
    reward_fn = RewardModelInference(reward_model, tokenizer, device=args.device)

    policy_model = ChessPolicyModel(vocab_size=tokenizer.language_size)
    policy_model.load_state_dict(
        torch.load(args.policy_model, map_location=args.device, weights_only=False)
    )
    policy_model.eval()
    policy_fn = PolicyModelInference(policy_model, tokenizer, device=args.device)

    board        = chess.Board()

    # colour picker screen
    surf.fill(BG)
    for txt, pos in [('Play as...', (W//2, H//2 - 60)),
                     ('W  —  White', (W//2, H//2)),
                     ('B  —  Black', (W//2, H//2 + 30))]:
        s = label_font.render(txt, True, TEXT_FG)
        surf.blit(s, s.get_rect(center=pos))
    pygame.display.flip()
    player_color = chess.WHITE
    picking = True
    while picking:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_b:
                    player_color = chess.BLACK; picking = False
                elif ev.key == pygame.K_w:
                    picking = False

    flipped = (player_color == chess.BLACK)

    selected     = None          # square the user clicked/is dragging from
    legal_dests  = set()
    drag_sq      = None
    drag_pos     = None

    ai_move      = None
    eval_score   = 0.0
    color_str    = 'White' if player_color == chess.WHITE else 'Black'
    status_msg   = f'You play {color_str}  |  R = restart  |  F = flip board'
    status_color = TEXT_FG

    ai_thinking  = False
    ai_result    = [None]

    clock = pygame.time.Clock()

    def run_ai():
        move_uci = policy_fn(board)
        ai_result[0] = chess.Move.from_uci(move_uci)

    # if human plays black, AI goes first
    if player_color == chess.BLACK:
        status_msg   = 'AI thinking...'
        status_color = AI_CLR
        ai_thinking  = True
        threading.Thread(target=run_ai, daemon=True).start()

    while True:
        clock.tick(60)

        # ── AI thread result ──────────────────────────────────────────────
        if ai_thinking and ai_result[0] is not None:
            move = ai_result[0]
            ai_result[0] = None
            ai_thinking   = False
            if move in board.legal_moves:
                ai_move = move
                board.push(move)
                eval_score = reward_fn(board)
                if board.is_game_over():
                    status_msg   = game_over_msg(board)
                    status_color = ERR_CLR
                else:
                    status_msg   = 'Your turn'
                    status_color = TEXT_FG

        # ── events ───────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    board = chess.Board(); selected = None
                    legal_dests = set(); drag_sq = None
                    ai_move = None; eval_score = 0.0
                    status_msg = 'You play White  |  R = restart  |  F = flip board'
                    status_color = TEXT_FG; ai_thinking = False; ai_result[0] = None
                if ev.key == pygame.K_f:
                    flipped = not flipped

            if ai_thinking or board.is_game_over():
                continue

            if board.turn != player_color:
                continue

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                sq = xy_to_sq(*ev.pos, flipped)
                if sq is not None and board.piece_at(sq) and board.piece_at(sq).color == player_color:
                    drag_sq     = sq
                    selected    = sq
                    drag_pos    = ev.pos
                    legal_dests = {m.to_square for m in board.legal_moves if m.from_square == sq}

            if ev.type == pygame.MOUSEMOTION and drag_sq is not None:
                drag_pos = ev.pos

            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and drag_sq is not None:
                target = xy_to_sq(*ev.pos, flipped)
                moved  = False

                if target is not None and target in legal_dests:
                    # check promotion
                    piece = board.piece_at(drag_sq)
                    promo = None
                    if (piece and piece.piece_type == chess.PAWN and
                            chess.square_rank(target) in (0, 7)):
                        promo = promote_dialog(surf, label_font, player_color)

                    move = chess.Move(drag_sq, target, promotion=promo)
                    if move in board.legal_moves:
                        board.push(move)
                        eval_score   = reward_fn(board)
                        moved        = True
                        selected     = None
                        legal_dests  = set()
                        ai_move      = None

                        if board.is_game_over():
                            status_msg   = game_over_msg(board)
                            status_color = ERR_CLR
                        else:
                            status_msg   = 'AI thinking...'
                            status_color = AI_CLR
                            ai_thinking  = True
                            ai_result[0] = None
                            threading.Thread(target=run_ai, daemon=True).start()

                if not moved:
                    # dropped on invalid square — check if clicking a new piece
                    if target is not None and board.piece_at(target) and \
                            board.piece_at(target).color == player_color and target != drag_sq:
                        selected    = target
                        legal_dests = {m.to_square for m in board.legal_moves if m.from_square == target}
                    else:
                        selected    = None
                        legal_dests = set()

                drag_sq  = None
                drag_pos = None

        draw_board(surf, board, piece_font, label_font, flipped,
                   selected, legal_dests, drag_sq, drag_pos, ai_move,
                   eval_score, status_msg, status_color)


def game_over_msg(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = 'Black' if board.turn == chess.WHITE else 'White'
        return f'Checkmate — {winner} wins!'
    if board.is_stalemate():   return 'Stalemate — draw'
    if board.is_insufficient_material(): return 'Draw — insufficient material'
    return 'Draw'


if __name__ == '__main__':
    main()
