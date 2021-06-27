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

RED     = 'r'
BLU     = 'b'
RED_SEQ = 'X'
BLU_SEQ = 'O'
JOKER   = '#'
EMPTY   = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3

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
        red_char = ['r', 'X']
        blue_char = ['b', 'O']
        special_char = ['#','_']

        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        state = deepcopy(game_state)
        agent_id = self.id
        plr_state = state.agents[agent_id]
        colour = plr_state.colour

        special_char = ['#','_']
        if colour == 'r':
            your_char = ['r', 'X']
            their_char = ['b', 'O']
        else:
            your_char = ['b', 'O']
            their_char = ['r', 'X']

        highest_score = 0
        best_move = actions[0]

        #handle trade case first
        if actions[0]['type'] == 'trade':
            for action in actions:
                draft = action['draft_card']
                hypothetical = deepcopy(state)
                hypothetical.agents[agent_id].hand = [draft]
                hypothetical_actions = self.getLegalActions(hypothetical, agent_id)
                for hypo_action in hypothetical_actions:
                    succ_state = self.generateSuccessor(hypothetical,hypo_action,agent_id)
                    succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
                    if succ_score > highest_score:
                        best_move = action
                        highest_score = succ_score

            return best_move

        # handle play remove cases
        # find best card
        nodraft = deepcopy(state)
        nodraft.board.draft = ['2s']

        jacks = ['jd','jc','jh','js']
        jack_threshold = 0.01
        jack_flag = False
        jack_list = []

        # if there is a jack change the hand to seperate jacks
        # if (set(nodraft.agents[agent_id].hand) and set(jacks)):

        for card in nodraft.agents[agent_id].hand:
            if card in jacks:
                jack_flag = True

        if jack_flag:
            for card in nodraft.agents[agent_id].hand:
                if card in jacks:
                    jack_list.append(card)
                    nodraft.agents[agent_id].hand.remove(card)


        nodraft_actions = self.getLegalActions(nodraft, agent_id)
        best_card = None
        best_cord = None
        best_state = None


        for action in nodraft_actions:
            play_hypothetical = deepcopy(nodraft)
            succ_state = self.generateSuccessor(play_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > highest_score:
                best_card = action['play_card']
                best_cord = action['coords']
                best_type = action['type']
                best_state = deepcopy(succ_state)
                highest_score = succ_score

        # if Jacks, find the best jack move
        if jack_flag:
            highest_score_jack = 0
            nodraft.agents[agent_id].hand = jack_list
            nodraft_actions = self.getLegalActions(nodraft, agent_id)
            for action in nodraft_actions:
                play_hypothetical = deepcopy(nodraft)
                succ_state = self.generateSuccessor(play_hypothetical,action,agent_id)
                succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
                if succ_score > highest_score_jack:
                    best_card_jac = action['play_card']
                    best_cord_jack = action['coords']
                    best_type_jack = action['type']
                    best_state_jack = deepcopy(succ_state)
                    highest_score_jack = succ_score
        

        if jack_flag:
            if highest_score_jack > (highest_score + jack_threshold):
                #print(highest_score + jack_threshold)
                #print(highest_score_jack)
                #print("picking jack")
                best_card = best_card_jac
                best_cord = best_cord_jack
                best_type = best_type_jack
                best_state = best_state_jack
            elif highest_score_jack > highest_score:
                print("this is the move played, but jack was mildly better")
                print(best_card)
                print(best_cord)
                print(best_type)
                print(type)
                print("this is the best jack move that was not played")
                print(best_card_jac)
                print(best_cord_jack)
                print(best_type_jack)
                print(best_state_jack)
            
            

        # find best draft card for best state
        # make draft cards the hand in the new board state

        best_state.agents[agent_id].hand = state.board.draft
        best_state.board.draft = ["2s"]
        best_state_actions = self.getLegalActions(best_state, agent_id)

        draft_flag = True

        while draft_flag:
            if best_state_actions[0]['type'] == 'trade':
                best_state.agents[agent_id].hand.remove(best_state_actions[0]['play_card'])
                best_state_actions = self.getLegalActions(best_state, agent_id)
            else:
                draft_flag = False

        future_highest_score = 0
        for action in best_state_actions:
            future_hypothetical = deepcopy(best_state)
            succ_state = self.generateSuccessor(future_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > future_highest_score:
                best_draft_card = action['play_card']
                future_highest_score = succ_score

        best_move = {'play_card':best_card, 'draft_card':best_draft_card, 'type':best_type, 'coords':best_cord}

        #print(best_move)

        return best_move

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        plr_state = state.agents[agent_id]
        plr_state.last_action = action #Record last action such that other agents can make use of this information.
        reward = 0

        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)                 #Remove card from hand.
            plr_state.discard = card                    #Add card to discard pile.
            state.deck.discards.append(card)            #Add card to global list of discards (some agents might find tracking this helpful).
            state.board.draft.remove(draft)             #Remove draft from draft selection.
            plr_state.hand.append(draft)                #Add draft to player hand.
            # state.board.draft.extend(state.deck.deal()) #Replenish draft selection.

        #If action was to trade in a dead card, action is complete, and agent gets to play another card.
        if action['type']=='trade':
            plr_state.trade = True #Switch trade flag to prohibit agent performing a second trade this turn.
            return state

        #Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r,c = action['coords']
        if action['type']=='place':
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type']=='remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")


        #Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type']=='place':
            seq,seq_type = self.checkSeq(state.board.chips, plr_state, (r,c))
            if seq:
                reward += seq['num_seq']
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r,c in sequence:
                        if state.board.chips[r][c] != JOKER: #Joker spaces stay jokers.
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except: #Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        plr_state.trade = False #Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action,reward)) #Log this turn's action and any resultant score.
        plr_state.score += reward
        return state

    def getLegalActions(self, game_state, agent_id):
        actions = []
        agent_state = game_state.agents[agent_id]

        #First, give the agent the option to trade a dead card, if they haven't just done so.
        if not agent_state.trade:
            for card in agent_state.hand:
                if card[0]!='j':
                    free_spaces = 0
                    for r,c in COORDS[card]:
                        if game_state.board.chips[r][c]==EMPTY:
                            free_spaces+=1
                    if not free_spaces: #No option to place, so card is considered dead and can be traded.
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'trade', 'coords':None})

            if len(actions): #If trade actions available, return those, along with the option to forego the trade.
                actions.append({'play_card':None, 'draft_card':None, 'type':'trade', 'coords':None})
                return actions

        #If trade is prohibited, or no trades available, add action/s for each card in player's hand.
        #For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
        for card in agent_state.hand:
            if card in ['jd','jc']: #two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==EMPTY:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})

            elif card in ['jh','js']: #one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==agent_state.opp_colour:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'remove', 'coords':(r,c)})

            else: #regular cards
                for r,c in COORDS[card]:
                    if game_state.board.chips[r][c]==EMPTY:
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})

        return actions

    def checkSeq(self, chips, plr_state, last_coords):
        clr,sclr   = plr_state.colour, plr_state.seq_colour
        oc,os      = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type   = TRADSEQ
        seq_coords = []
        seq_found  = {'vr':0, 'hz':0, 'd1':0, 'd2':0, 'hb':0}
        found      = False
        nine_chip  = lambda x,clr : len(x)==9 and len(set(x))==1 and clr in x
        lr,lc      = last_coords

        #All joker spaces become player chips for the purposes of sequence checking.
        for r,c in COORDS['jk']:
            chips[r][c] = clr

        #First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4,4),(4,5),(5,4),(5,5)]
        heart_chips = [chips[y][x] for x,y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb']+=2
            seq_coords.append(coord_list)

        #Search vertical, horizontal, and both diagonals.
        vr = [(-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0)]
        hz = [(0,-4),(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),(0,4)]
        d1 = [(-4,-4),(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3),(4,4)]
        d2 = [(-4,4),(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3),(4,-4)]
        for seq,seq_name in [(vr,'vr'), (hz,'hz'), (d1,'d1'), (d2,'d2')]:
            coord_list = [(r+lr, c+lc) for r,c in seq]
            coord_list = [i for i in coord_list if 0<=min(i) and 9>=max(i)] #Sequences must stay on the board.
            chip_str   = ''.join([chips[r][c] for r,c in coord_list])
            #Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name]+=2
                seq_coords.append(coord_list)
            #If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx    = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i+1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx+5])
                        break
            else: #Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr*5, clr*4+sclr, clr*3+sclr+clr, clr*2+sclr+clr*2, clr+sclr+clr*3, sclr+clr*4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx+5] == pattern:
                            seq_found[seq_name]+=1
                            seq_coords.append(coord_list[start_idx:start_idx+5])
                            found = True
                            break
                    if found:
                        break

        for r,c in COORDS['jk']:
            chips[r][c] = JOKER #Joker spaces reset after sequence checking.

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return ({'num_seq':num_seq, 'orientation':[k for k,v in seq_found.items() if v], 'coords':seq_coords}, seq_type) if num_seq else (None,None)