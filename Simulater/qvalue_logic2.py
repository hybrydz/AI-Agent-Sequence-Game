example_list = [['#', '_', 'r', 'b', '_', 'b', 'b', 'r', 'b', '#'], ['_', 'r', 'r', 'b', 'b', 'b', 'b', '_', '_', '_'], ['b', '_', '_', 'b', '_', '_', 'r', 'r', 'r', '_'], ['_', 'b', 'b', 'b', 'r', '_', 'r', 'b', '_', '_'], ['b', 'r', 'b', 'r', '_', 'b', 'b', 'r', 'b', 'r'], ['b', 'X', 'b', '_', 'b', 'r', '_', '_', 'r', 'b'], ['_', 'X', '_', 'r', 'r', 'r', 'b', '_', 'b', '_'], ['r', 'X', 'r', '_', 'b', '_', 'b', '_', 'b', 'r'], ['b', 'X', 'b', 'r', 'r', 'r', 'r', '_', 'b', 'r'], ['#', 'X', 'b', 'b', 'r', '_', 'b', 'r', 'r', '#']]
import numpy as np
import random
from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict

red_char = ['r', 'X']
blue_char = ['b', 'O']
special_char = ['#','_']

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


x = np.array(example_list)

print(x)
print(get_win_percentage(x, red_char, blue_char, special_char))