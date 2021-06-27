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

import time
import math
import queue
import heapq
import csv


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

def generateSuccessor(state, action, agent_id):
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
        seq,seq_type = checkSeq(state.board.chips, plr_state, (r,c))
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

def getLegalActions(game_state, agent_id):
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

def checkSeq(chips, plr_state, last_coords):
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

def my_get_legal_actions(state, agent_id):

    actions = getLegalActions(state, agent_id)

    # handle trade case
    draft_flag = True
    while draft_flag:
        if actions[0]['type'] == 'trade':
            state.agents[agent_id].hand.remove(actions[0]['play_card'])
            actions = getLegalActions(state, agent_id)
        else:
            draft_flag = False

    return actions

def my_get_successor_state(state, action, agent_id):
    new_state = generateSuccessor(state, action, agent_id)
    new_state.agents[agent_id].hand.remove('##')
    new_state.board.draft = ['##']

    return new_state

def opponentMoveGenerator(state, agent_id, moves):

    if agent_id == 0 or agent_id == 2:
        opp_id = 1
    else:
        opp_id = 0

    special_char = ['#','_']
    if agent_id == 0 or agent_id == 2:
        your_char = ['r', 'X']
        their_char = ['b', 'O']
    else:
        your_char = ['b', 'O']
        their_char = ['r', 'X']

    best_opp_actions = []

    opp_state = deepcopy(state)
    opp_state.agents[opp_id].hand = ['jd']
    opp_state.board.draft = ['##']
    for action in getLegalActions(opp_state, opp_id):
        hypostate = deepcopy(opp_state)
        succ_state = my_get_successor_state(hypostate, action, opp_id)
        succ_score = get_win_percentage(succ_state.board.chips, their_char, your_char, special_char)
        best_opp_actions.append([succ_score, action,deepcopy(succ_state)])

    sorted_moves = sorted(best_opp_actions, key=lambda x: x[0], reverse=True)[0:moves]
    possible_states = []
    for action in sorted_moves:
        possible_states.append(action[2])

    return possible_states

def MDPexecute(state, action, agent_id, moves, all = False):
    succ_state = deepcopy(state)
    future_state = my_get_successor_state(succ_state, action, agent_id)
    possible_state = opponentMoveGenerator(future_state, agent_id, moves)

    if all:
        return possible_state

    return random.choice(possible_state)

class MultiArmedBandit():

    '''
        Select an action given Q-values for each action.
    '''
    def select(self, actions, qValues):
        pass

    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''

class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0  #number of times a choice has been made
        self.N = dict() #number of times each action has been chosen

    def select(self, actions, qValues):

        # First execute each action one time
        for action in actions:
            if not action in self.N.keys():
                self.N[action] = 1
                self.total += 1
                return action

        maxActions = []
        maxValue = float('-inf')
        for action in actions:
            N = self.N[action]
            value = qValues[action] + math.sqrt((2 * math.log(self.total)) / N)
            if value > maxValue:
                maxActions = [action]
                maxValue = value
            elif value == maxValue:
                maxActions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(maxActions)
        self.N[result] = self.N[result] + 1
        self.total += 1
        return result



class Node():

    # record a unique node id to distinguish duplicated states for visualisation
    nextNodeID = 0

    def __init__(self, parent, state, agent_id):
        self.parent = parent
        self.state = state
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0
        self.agent_id = agent_id

    '''
    Return the value of this node
    '''
    def getValue(self):
        return self.value

class StateNode(Node):

    def __init__(self, parent, state, agent_id, reward = 0, probability = 1.0, bandit = UpperConfidenceBounds()):
        super().__init__(parent, state, agent_id)

        # a dictionary from actions to an environment node
        self.children = {}

        # the reward received for this state
        self.reward = reward

        # the probability of this node being chosen from its parent
        self.probability = probability

        # a multi-armed bandit for this node
        self.bandit = bandit


    '''
    Return true if and only if all child actions have been expanded
    '''
    def isFullyExpanded(self):
        validActions = my_get_legal_actions(self.state, self.agent_id)
        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self):
        if not self.isFullyExpanded():
            return self
        else:
            actions = list(self.children.keys())
            qValues = dict()
            for action in actions:
                #get the Q values from all outcome nodes
                qValues[action] = self.children[tuple(sorted(action.items()))].getValue()
            bestAction = self.bandit.select(actions, qValues)
            return self.children[tuple(sorted(bestAction.items()))].select()

    def expand(self):
        #randomly select an unexpanded action to expand
        a = my_get_legal_actions(self.state, self.agent_id)
        b = list(self.children.keys())
        actions = [action for action in a if action not in b]

        action = random.choice(list(actions))

        #choose an outcome
        newChild = EnvironmentNode(self, self.state, action, self.agent_id)
        newStateNode = newChild.expand()
        action_tuple = tuple(sorted(action.items()))
        self.children[action_tuple] = newChild
        return newStateNode

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits)

        if self.parent != None:
            self.parent.backPropagate(reward)

    def getQFunction(self, highest = False):
        qValues = {}
        highest_score = 0
        best_action = None
        for action in self.children.keys():
            if highest:
                score = round(self.children[action].getValue(), 3)
                if score > highest_score:
                    highest = score
                    best_action = action
            else:
                qValues[(self.state, action)] = round(self.children[action].getValue(), 3)
        if highest:
            return best_action
        else:
            return qValues

class EnvironmentNode(Node):

    def __init__(self, parent, state, action, agent_id):
        super().__init__(parent, state, agent_id)
        self.outcomes = {}
        self.action = action

        # a set of outcomes
        self.children = []

    def select(self):

        newState = MDPexecute(deepcopy(self.state), self.action, self.agent_id, 5)
        reward = get_reward(newState, self.agent_id)

        #find the corresponding state
        for child in self.children:
            if newState.board.chips == child.state.board.chips:
                return child.select()

    def addChild(self, action, newState, reward, probability):
        child = StateNode(self, newState, reward, probability)
        self.children += [child]
        return child

    def expand(self):
        # choose one outcome based on transition probabilities

        newState = MDPexecute(self.state, self.action, self.agent_id, 5)
        reward = get_reward(newState, self.agent_id)

        # expand all outcomes
        selected = None

        all_possibilities = MDPexecute(self.state, self.action, self.agent_id, 5, True)

        num_poss = len(all_possibilities)

        transitions = []

        for poss in all_possibilities:
            transitions.append([poss, get_reward(poss, self.agent_id)])

        total_score = 0

        for poss, score in transitions:
            total_score += score

        updated_transitions = []

        for poss, score in transitions:
            updated_transitions.append((poss, score/total_score))

        for (outcome, probability) in updated_transitions:
            newChild = self.addChild(self.action, outcome, reward, probability)
            # find the child node correponding to the new state
            if outcome.board.chips == newState.board.chips:
                selected = newChild

        return selected

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((reward - self.value) / self.visits)

        # TODO CHANGE DISCOUNT FACTORS
        self.parent.backPropagate(reward * 0.9)

def terminalState(state, agent_id):
    if len(state.agents[agent_id].hand) < 3:
        return True
    else:
        return False

def MDPtuple(state, action, agent_id):
    newState = MDPexecute(state, action, agent_id, 5)
    reward = get_reward(newState, agent_id)

    return (newState, reward)

class MCTS():

    def __init__(self, agent_id, initial_state):
        self.agent_id = agent_id
        self.initial_state = initial_state

    '''
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    '''
    def mcts(self, agent_id, timeout = 1):
        rootNode = StateNode(None, self.initial_state, self.agent_id)
        '''
        startTime = int(time.time() * 1000)
        currentTime = int(time.time() * 1000)
        
        while currentTime < startTime + timeout * 1000:
            # find a state node to expand
            selectedNode = rootNode.select()
            if not terminalState(selectedNode.state, agent_id):
                child = selectedNode.expand()
                reward = self.simulate(child)
                child.backPropagate(reward)

            currentTime = int(time.time() * 1000)
        '''

        iterations = 0
        while iterations < 12:
            # find a state node to expand
            selectedNode = rootNode.select()
            if not terminalState(selectedNode.state, agent_id):
                child = selectedNode.expand()
                reward = self.simulate(child)
                child.backPropagate(reward)
            iterations += 1

        return rootNode

    '''
        Choose a random action. Heustics can be used here to improve simulations.
    '''
    # TODO PUT Q FUCNTION IN AS HEURISTIC
    def choose(self, state):
        return random.choice(my_get_legal_actions(state, self.agent_id))

    '''
        Simulate until a terminal state
    '''
    def simulate(self, node):
        state = deepcopy(node.state)
        cumulativeReward = 0.0
        depth = 0
        while not terminalState(state, self.agent_id):
            #choose an action to execute
            action = self.choose(state)

            # execute the action
            (newState, reward) = MDPtuple(state, action, self.agent_id)

            # discount the reward
            cumulativeReward += pow(0.9, depth) * reward

            depth += 1
            state = newState

        return cumulativeReward


def get_reward(state, agent_id):

    '''
    if terminalState(state, agent_id):
        special_char = ['#','_']
        if agent_id == 0 or agent_id == 2:
            your_char = ['r', 'X']
            their_char = ['b', 'O']
        else:
            your_char = ['b', 'O']
            their_char = ['r', 'X']

        return get_win_percentage(state.board.chips, your_char, their_char, special_char)

    else:
        return 0
    '''

    special_char = ['#','_']
    if agent_id == 0 or agent_id == 2:
        your_char = ['r', 'X']
        their_char = ['b', 'O']
    else:
        your_char = ['b', 'O']
        their_char = ['r', 'X']

    return get_win_percentage(state.board.chips, your_char, their_char, special_char)


class myAgent(Agent):
    def __init__(self,_id):
        red_char = ['r', 'X']
        blue_char = ['b', 'O']
        special_char = ['#','_']

        super().__init__(_id)

    def SelectAction(self, actions, game_state):
        player6Action = {}
        bfsAction = {}
        aStarAction= {}
        mctsAction = {}

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
        hypothetical = deepcopy(state)
        astarHypothetical = deepcopy(state)
        mctsState = deepcopy(state)
        

        # PLAYER 6 -------------------------------------
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
        player6BestState =[]
        future_highest_score = 0
        for action in best_state_actions:
            future_hypothetical = deepcopy(best_state)
            succ_state = self.generateSuccessor(future_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > future_highest_score:
                best_draft_card = action['play_card']
                future_highest_score = succ_score
                player6BestState = succ_state
        # PLAYER 6 ACTION
        player6Action = {'play_card':best_card, 'draft_card':best_draft_card, 'type':best_type, 'coords':best_cord}

        # print("player6: ",player6Action)

        # BFS EXPAND FUNCTION
        def expand_node(BFSstate,flag="bfs"):
            # this sets depth of search (cards in hand left)
            if len(BFSstate.agents[agent_id].hand) == 5:
                return []

            actions = self.getLegalActions(BFSstate, agent_id)


            # handle trade case
            draft_flag = True
            while draft_flag:
                if actions[0]['type'] == 'trade':
                    BFSstate.agents[agent_id].hand.remove(actions[0]['play_card'])
                    actions = self.getLegalActions(BFSstate, agent_id)
                else:
                    draft_flag = False

            newnodes = []

            for action in actions:
                new_state = deepcopy(BFSstate)
                new_state = self.generateSuccessor(new_state, action, agent_id)
                new_state.agents[agent_id].hand.remove('##')
                new_state.board.draft = ['##']
                if flag == "astar":
                    cost = 1
                    node = (new_state, action, cost)
                else:
                    node = (new_state, action)
                newnodes.append(node)

            return newnodes
        
        # BFS ---------------------------------------

        highest_future_state_score = 0
        depth = 0
        
        hypothetical.board.draft = ["##"]

        best_card = None
        best_cord = None
        best_state = None

        myqueue = queue.Queue()
        startNode = (hypothetical, '', [])
        myqueue.put(startNode)
        visited = set()

        # TODO PRIORITY QUEUE NOT WORKING
        #priorQueue = queue.PriorityQueue()
        final_nodes = []

        while not myqueue.empty():
            node = myqueue.get()
            BFSstate, BFSaction, BFSpath = node
            if BFSstate not in visited:
                visited.add(BFSstate)
                succNodes = expand_node(BFSstate)
                if not succNodes:
                    BFSpath = BFSpath + [(BFSstate, BFSaction)]
                    #print(get_win_percentage(BFSstate.board.chips, your_char, their_char, special_char))
                    final_nodes.append((BFSstate, BFSpath))
                for succNode in succNodes:
                    succ_state, succAction = succNode
                    newNode = (succ_state, succAction, BFSpath + [(BFSstate, BFSaction)])
                    #newNode = (succ_state, succAction, BFSpath + [(state,BFSaction)])
                    myqueue.put(newNode)


        # find highest score amongst terminal nodes
        highest_score = 0
        for node in final_nodes:
            finalstate = node[0]
            path = node[1]
            finalstatescore = get_win_percentage(finalstate.board.chips, your_char, their_char, special_char)
            if finalstatescore > highest_score:
                action = path[1][1]
                best_card = action['play_card']
                best_cord = action['coords']
                best_type = action['type']
                best_state = deepcopy(path[1][0])
                highest_score = finalstatescore



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
        bfsBestState = []
        for action in best_state_actions:
            future_hypothetical = deepcopy(best_state)
            succ_state = self.generateSuccessor(future_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > future_highest_score:
                best_draft_card = action['play_card']
                future_highest_score = succ_score
                bfsBestState = succ_state

        bfsAction = {'play_card':best_card, 'draft_card':best_draft_card, 'type':best_type, 'coords':best_cord}
        # print("bfsAction: ",bfsAction)
        # --------------------------------------------------------------

        # A* ------------------------------------------------------------
        highest_future_state_score = 0
        depth = 0
        # astarHypothetical = deepcopy(state)
        astarHypothetical.board.draft = ["##"]
        best_card = None
        best_cord = None
        best_state = None
        priorQueue = PriorityQueue()

        startNode = (astarHypothetical, '',0, [])
        priorQueue.push(startNode,0)
        visited = set()
        final_nodes = []

        # print("astar here")
        try:
            while priorQueue:
                node = priorQueue.pop()
                Astate, Aaction, Acost, Apath = node
                if Astate not in visited:
                    visited.add(Astate)
                    succNodes = expand_node(Astate,"astar")
                    if not succNodes:
                        Apath = Apath + [(Astate, Aaction)]
                        final_nodes.append((Astate, Apath))
                        break;
                    for succNode in succNodes:
                        succState, succAction, succCost = succNode
                        newCost = Acost + succCost
                        priority = self.seqHeuristic(succState,agent_id) + newCost
                        # print("Priority",priority)
                        # priority = 0 + newCost
                        newNode = (succState, succAction,newCost, Apath + [(Astate,Aaction)])
                        priorQueue.push(newNode, priority)
        except Exception:
            traceback.print_exc()

        # find highest score amongst terminal nodes
        highest_score = 0
        for node in final_nodes:
            finalstate = node[0]
            path = node[1]
            finalstatescore = get_win_percentage(finalstate.board.chips, your_char, their_char, special_char)
            if finalstatescore > highest_score:
                action = path[1][1]
                best_card = action['play_card']
                best_cord = action['coords']
                best_type = action['type']
                best_state = deepcopy(path[1][0])
                highest_score = finalstatescore


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
        astarBestState = []
        for action in best_state_actions:
            future_hypothetical = deepcopy(best_state)
            succ_state = self.generateSuccessor(future_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > future_highest_score:
                best_draft_card = action['play_card']
                future_highest_score = succ_score
                astarBestState = succ_state

        aStarAction = {'play_card':best_card, 'draft_card':best_draft_card, 'type':best_type, 'coords':best_cord}
        
        # MCTS ------------------------------------------------------------
        # nodraft = deepcopy(state)
        mctsState.board.draft = ['##']
        rootnode = MCTS(agent_id, mctsState).mcts(agent_id)

        action = dict(rootnode.getQFunction(highest=True))
        best_card = action['play_card']
        best_cord = action['coords']
        best_type = action['type']
        best_state = my_get_successor_state(mctsState, action, agent_id)

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
        mctBestState = []
        for action in best_state_actions:
            future_hypothetical = deepcopy(best_state)
            succ_state = self.generateSuccessor(future_hypothetical,action,agent_id)
            succ_score = get_win_percentage(succ_state.board.chips, your_char, their_char, special_char)
            if succ_score > future_highest_score:
                best_draft_card = action['play_card']
                future_highest_score = succ_score
                mctBestState = succ_state
        mctsAction = {'play_card':best_card, 'draft_card':best_draft_card, 'type':best_type, 'coords':best_cord}

        print("player6:", player6Action)
        p6chips = player6BestState.board.chips
        p6hand = player6BestState.agents[agent_id].hand

        print("BFS:",bfsAction)
        bfschips = bfsBestState.board.chips
        bfshand = bfsBestState.agents[agent_id].hand

        print("Astar:", aStarAction)
        astarchips = astarBestState.board.chips
        astarhand = astarBestState.agents[agent_id].hand

        print("MCTS:", mctsAction)
        mctschips = mctBestState.board.chips
        mctshand = mctBestState.agents[agent_id].hand

        data = [p6hand,bfshand,astarhand,mctshand,player6Action,bfsAction,aStarAction,mctsAction,p6chips,bfschips,astarchips,mctschips]
        with open('playerAll.csv','a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

        return player6Action

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
   
    def seqHeuristic(self,state,agent_id):
        board=state.board.chips
        board = np.array(board)
        colour=state.agents[agent_id].colour
        special_char = ['#','_']
        if colour == 'r':
            your_char = ['r', 'X']
            their_char = ['b', 'O']
        else:
            your_char = ['b', 'O']
            their_char = ['r', 'X']
        maxScore=0
        seqScore=0
        # secondMaxScore=0
        all_windows=[]
        SEQ_CHIPS=9
        heart=[(4,4),(4,5),(5,4),(5,5)]
        opp_coords=state.board.plr_coords[state.agents[agent_id].opp_colour]
        our_coords=state.board.plr_coords[state.agents[agent_id].colour]
        check=any(items in opp_coords for items in heart)
        if check:
            transposeBoard=board.transpose()
            for row in board:
                for window in windows_of_five(row):
                    all_windows.append(window)
            for row in transposeBoard:
                for window in windows_of_five(row):
                    all_windows.append(window)
            for row in get_diags(board):
                for window in windows_of_five(row):
                    all_windows.append(window)
            for row in get_diags(np.rot90(board)):
                for window in windows_of_five(row):
                    all_windows.append(window)
            for window in all_windows:
                score=0
                flag=True
                for element in window:
                    if element == their_char[0] or element == their_char[1]:
                        flag=False
                        break
                    elif element == your_char[0]:
                        score+=1
                    elif element == your_char[1]:
                        seqScore=5
                if(score>=maxScore and flag==True):
                    # secondMaxScore=maxScore
                    maxScore=score
                # elif(score>secondMaxScore and flag==True):
                #     secondMaxScore=score
            heuristicValue=SEQ_CHIPS-(maxScore+seqScore)
        else:
            count=0
            for i in our_coords:
                if i==(4,4):
                    count+=1
                elif i==(4,5):
                    count+=1
                elif i==(5,4):
                    count+=1
                elif i==(5,5):
                    count+=1
            heuristicValue=4-count
        return heuristicValue


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)