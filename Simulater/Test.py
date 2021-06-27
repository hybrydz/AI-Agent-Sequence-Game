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

hand = ['5s', '6s', '7s', 'jd']
new_hand = []

for card in hand:
    print(card)
    new_hand.append(COORDS[card])

print(new_hand)



'''
b = np.zeros((10, 10), dtype=np.int)

b[0,0] = 2
b[9,9] = 2
b[0,9] = 2
b[9,0] = 2

b[0:5] = 1

x,y = 3,4

def get_diags(a):
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    return diags

print(b)
for diag in get_diags(b)[4:-4]:
    print(diag)
    count = 0
'''
'''
    for element in diag:
        if element == 1:
            count += 1
            if count >= 5:
                print("win")
                break
        else:
            count = 0
'''
# a = np.rot90(a)


'''
for row in b:
    count = 0
    for element in row:
        if element == 1:
            count += 1
            if count >= 5:
                print("win")
                break
        else:
            count = 0

for column in b.transpose():
    count = 0
    for element in column:
        if element == 1:
            count += 1
            print(count)
            if count >= 5:
                print("win")
                break
        else:
            count = 0
'''