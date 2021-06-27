import numpy as np
import random
from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict

seq_board = state.board
plr_state = state.agents[agent_id]
colour = plr_state.colour
pieces = state.board.plr_coords[colour]
opposite_colour = plr_state.opp_colour
opponent_pieces = state.board.plr_coords[opposite_colour]
hand = plr_state.hand
draft = state.board.draft

BOARD = [['jk','2s','3s','4s','5s','6s','7s','8s','9s','jk'],
         ['6c','5c','4c','3c','2c','ah','kh','qh','th','ts'],
         ['7c','as','2d','3d','4d','5d','6d','7d','9h','qs'],
         ['8c','ks','6c','5c','4c','3c','2c','8d','8h','ks'],
         ['9c','qs','7c','6h','5h','4h','ah','9d','7h','as'],
         ['tc','ts','8c','7h','2h','3h','kh','td','6h','2d'],
         ['qc','9s','9c','8h','9h','th','qh','qd','5h','3d'],
         ['kc','8s','tc','qc','kc','ac','ad','kd','4h','4d'],
         ['ac','7s','6s','5s','4s','3s','2s','2h','3h','5d'],
         ['jk','ad','kd','qd','td','9d','8d','7d','6d','jk']]


COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))


def card_to_cord(hand):
    co_ords = []
    for card in hand:
        co_ords.append(COORDS[card])
    return co_ords

co_ord_hand = card_to_cord(hand)
co_ord_draft = card_to_cord(draft)

def pieces_to_board(board, pieces, player):
    for co_ord in pieces:
        board[co_ord] = player

pieces_to_board(board, pieces, 1)
pieces_to_board(board, opponent_pieces, -1)

class PlaySeq:
    """
    A simple game of sequence where you play anywhere
    """

    def __init__(self):
        pass

    def get_init_board(self):
        b = np.zeros((10, 10), dtype=np.int)
        # corners = [0,0], [9,9], [0,9], [9,0]
        b[0,0] = 2
        b[9,9] = 2
        b[0,9] = 2
        b[9,0] = 2
        return b

    def get_next_state(self, board, player, action):
        b = np.copy(board)
        b[action[0], action[1]] = player

        # Return the new game, but
        # change the perspective of the game with negative
        return (b, -player)

    def has_legal_moves(self, board):
        for piece in np.nditer(board):
            if piece == 0:
                return True
        return False

    def get_valid_moves(self, board, hand):
        # checks
        valid_moves = []
        for card in hand:
            if board[card[0], card[1]] == 0:
                valid_moves.append(card)
        return valid_moves

    def is_win(self, board, player):

        # centre check
        if board[4,4] == player and board[4,5] == player and board[5,4] == player and board[5,5] == player:
            return True

        # check sequences
        sequences = 0
        for row in board:
            count = 0
            for element in row:
                if element == player or element == 2:
                    count += 1
                    if count == 5:
                        sequences += 1
                        if sequences >= 2:
                            return True
                        count = 0
                else:
                    count = 0

        for column in board.transpose():
            count = 0
            for element in column:
                if element == player or element == 2:
                    count += 1
                    if count == 5:
                        sequences += 1
                        if sequences >= 2:
                            return True
                        count = 0
                else:
                    count = 0

        a = np.copy(board)
        diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
        diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
        diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]

        for diag in diags[4:-4]:
            count = 0
            for element in diag:
                if element == player or element == 2:
                    count += 1
                    if count == 5:
                        sequences += 1
                        if sequences >= 2:
                            return True
                        count = 0
                else:
                    count = 0

        a = np.rot90(a)
        diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
        diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
        diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]

        for diag in diags[4:-4]:
            count = 0
            for element in diag:
                if element == player or element == 2:
                    count += 1
                    if count == 5:
                        sequences += 1
                        if sequences >= 2:
                            return True
                        count = 0
                else:
                    count = 0

        return False

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None

        return 0

    def get_canonical_board(self, board, player):
        return player * board



p1 = PlaySeq()
hand = [(4,4),(4,5), (5,4), (5,5)]
board = p1.get_init_board()
#print(board)
player = 1
valid_moves = p1.get_valid_moves(board, hand)

for move in valid_moves:
    board = p1.get_next_state(board, player, move)[0]
print(board)


def board_score(board, player):

    centre_tiles = [(4,4,), (4,5), (5,4), (5,5)]

    centre_count = 0

    for place in centre_tiles:
        if board[place[0], place[1]] == player:
            centre_count += 1

    centre_score = centre_count ** 4

    sequences_score = 0
    for row in board:
        count = 0
        for element in row:
            if element == player or element == 2:
                count += 1
                if count == 5:
                    sequences_score += 2 ** 5
                    count = 0
            else:
                if count > 0:
                    sequences_score += 2 ** count
                count = 0

        if count > 0:
            sequences_score += 2 ** count

    for column in board.transpose():
        count = 0
        for element in column:
            if element == player or element == 2:
                count += 1
                if count == 5:
                    sequences_score += 2 ** 5
                    count = 0
            else:
                if count > 0:
                    sequences_score += 2 ** count
                count = 0

        if count > 0:
            sequences_score += 2 ** count

    a = np.copy(board)
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]

    for diag in diags[4:-4]:
        count = 0
        for element in diag:
            if element == player or element == 2:
                count += 1
                if count == 5:
                    sequences_score += 2 ** 5
                    count = 0
            else:
                if count > 0:
                    sequences_score += 2 ** count
                count = 0

        if count > 0:
            sequences_score += 2 ** count

    a = np.rot90(a)
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]

    for diag in diags[4:-4]:
        count = 0
        for element in diag:
            if element == player or element == 2:
                count += 1
                if count == 5:
                    sequences_score += 2 ** 5
                    count = 0
            else:
                if count > 0:
                    sequences_score += 2 ** count
                count = 0

        if count > 0:
            sequences_score += 2 ** count

    return centre_score + sequences_score

print(board_score(board,1))
print(board)

red = [(0, 4), (8, 3), (0, 5), (8, 2), (0, 6), (8, 1)]

for move in valid_moves:
    board = p1.get_next_state(board, player, move)[0]
print(board)

for co_ord in red:
    board[co_ord] = 1

for co_ord in red:
    print(BOARD[co_ord[0]][co_ord[1]])


