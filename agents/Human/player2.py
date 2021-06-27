from template import Agent
import random
from template import Agent
import random
from collections import defaultdict
from Sequence.sequence_model import SequenceGameRule, SequenceState
import traceback
from copy import deepcopy
import operator
import numpy as np

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

def windows_of_five(mylist):
    windows = []
    for i in range(len(mylist)):
        if i <= len(mylist) - 5:
            windows.append(mylist[i:i+5])

    return windows


def rows_to_scores(board, player):
    all_windows = []
    for row in board:
        for window in windows_of_five(row):
            all_windows.append(window)

    total_score = 0

    for window in all_windows:
        flag = True
        score = 1
        for element in window:
            if element == player or element == 2:
                score  = score*2
            if element == -player:
                flag = False
                break
        if flag:
            total_score += score

    return total_score


def get_diags(a):
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    return diags[4:-4]


def hearts_to_score(board, player):
    centre_tiles = [(4,4,), (4,5), (5,4), (5,5)]
    score = 1
    for tile in centre_tiles:
        if board[tile[0], tile[1]] == player:
            score = score * 8
        elif board[tile[0], tile[1]] == -player:
            return 0
    return score


def get_board_score(board, player):
    score = 0
    score += rows_to_scores(board, player)
    score += rows_to_scores(board.transpose(), player)
    score += rows_to_scores(get_diags(board), player)
    score += rows_to_scores(get_diags(np.rot90(board)), player)
    score += hearts_to_score(board, player)
    return score


def get_win_percentage(board):

    you = 1
    them = -1

    your_score = get_board_score(board, you)
    their_score = get_board_score(board, them)

    return your_score / (your_score + their_score)

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        state = deepcopy(game_state)
        agent_id = self.id
        plr_state = state.agents[agent_id]
        colour = plr_state.colour
        seq_colour = plr_state.seq_colour

        pieces = state.board.plr_coords[colour]
        opposite_colour = plr_state.opp_colour
        opponent_pieces = state.board.plr_coords[opposite_colour]

        #print(state.board.chips)

        highest_score = 0
        best_move = actions[0]

        for action in actions:
            card = action['play_card']
            draft = action['draft_card']
            move = action['type']
            (r, c) = action['coords']

            # create new board
            board = np.zeros((10, 10), dtype=np.int)
            # corners 
            board[0,0] = 2
            board[9,9] = 2
            board[0,9] = 2
            board[9,0] = 2

            self.pieces_to_board(board, pieces, 1)
            self.pieces_to_board(board, opponent_pieces, -1)

            score = 0

            if move == 'place':
                board[r,c] = 1
                score = get_win_percentage(board)

            if move == 'remove':
                board[r,c] = 0
                score = get_win_percentage(board)

            if move == 'trade:':
                score = 0
                for co_ord in self.get_possible_moves(board, draft):
                    board[co_ord] = 1
                    if get_win_percentage(board) > score:
                        score = get_win_percentage(board)
                    board[co_ord] = 0

            if score > highest_score:
                best_move = action

        return best_move

    def card_to_cord(self, hand):
        co_ords = []
        for card in hand:
            co_ords.append(COORDS[card])
        return co_ords

    def pieces_to_board(self, board, pieces, player):
        for co_ord in pieces:
            board[co_ord] = player

    # Doesn't account for jacks
    # TODO extend this function to include jacks, placement, and removal
    def get_possible_moves(self, board, card):
        locations = []
        co_ord = COORDS[card]
        if board[co_ord] == 0:
            locations.append(co_ord)

        return locations
