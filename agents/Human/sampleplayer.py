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
import random
from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict


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

def rows_to_scores(board, your_char, their_char, special_char):
    all_windows = []
    for row in board:
        for window in windows_of_five(row):
            all_windows.append(window)

    total_score = 0

    for window in all_windows:
        flag = True
        score = 0
        your_char_normal = 0
        your_char_seq = 0
        normal_char = 0

        for element in window:
            if element == their_char[0] or element == their_char[1]:
                score = 0
                break

            elif element == special_char[1]:
                normal_char += 1

            if element == your_char[0] or element == special_char[0]:
                your_char_normal += 1

            elif element == your_char[1]:
                your_char_seq += 1

            if element == their_char[0] or element == their_char[1]:
                flag = False
                break

        if flag:
            if your_char_seq == 5:
                score = 2**8

            elif your_char_seq <= 1:
                score = (your_char_normal + your_char_seq) ** 2 + normal_char

        total_score += score

    return total_score


def get_diags(a):
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    return diags[4:-4]


def hearts_to_score(board, your_char, their_char):
    centre_tiles = [(4,4,), (4,5), (5,4), (5,5)]
    score = 0
    tiles_controlled = 0
    flag = True
    for tile in centre_tiles:
        if board[tile[0], tile[1]] == your_char[0] or board[tile[0], tile[1]] == your_char[1]:
            tiles_controlled += 1
        elif board[tile[0], tile[1]] == their_char[0] or board[tile[0], tile[1]] == their_char[1]:
            flag = False

    if flag:
        score = tiles_controlled ** 4
    else:
        score = tiles_controlled * 6

    return score


def get_board_score(board, your_char, their_char, special_char):
    score = 0
    score += rows_to_scores(board, your_char, their_char, special_char)
    score += rows_to_scores(board.transpose(), your_char, their_char, special_char)
    score += rows_to_scores(get_diags(board), your_char, their_char, special_char)
    score += rows_to_scores(get_diags(np.rot90(board)), your_char, their_char, special_char)
    score += hearts_to_score(board, your_char, their_char)
    return score


def get_win_percentage(board, your_char, their_char, special_char):

    board = np.array(board)

    your_score = get_board_score(board, your_char, their_char, special_char)
    their_score = get_board_score(board, their_char, your_char, special_char)

    return your_score / (your_score + their_score)

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)

        board = np.zeros((10, 10), dtype=np.int)
        # corners = [0,0], [9,9], [0,9], [9,0]
        board[0,0] = 2
        board[9,9] = 2
        board[0,9] = 2
        board[9,0] = 2



    def SelectAction(self,actions,game_state):
        state = deepcopy(game_state)
        agent_id = self.id
        plr_state = state.agents[agent_id]
        colour = plr_state.colour


        pieces = state.board.plr_coords[colour]
        opposite_colour = plr_state.opp_colour
        opponent_pieces = state.board.plr_coords[opposite_colour]
        hand = plr_state.hand
        draft = state.board.draft

        co_ord_hand = self.card_to_cord(hand)
        co_ord_draft = self.card_to_cord(draft)

        self.pieces_to_board(self.board, pieces, 1)
        self.pieces_to_board(self.board, opponent_pieces, -1)

        return actions[0]

    def card_to_cord(hand):
        co_ords = []
        for card in hand:
            co_ords.append(COORDS[card])
        return co_ords

    def pieces_to_board(board, pieces, player):
        for co_ord in pieces:
            board[co_ord] = player

