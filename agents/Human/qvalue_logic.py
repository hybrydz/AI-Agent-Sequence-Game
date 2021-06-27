import numpy as np
import random
from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict


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