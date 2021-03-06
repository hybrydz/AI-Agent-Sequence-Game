from template import Agent
import random
from collections import defaultdict
from Sequence.sequence_model import SequenceGameRule, SequenceState
import traceback
from copy import deepcopy
import operator

HEURISTIC_BOARD = [[0,35,25,25,25,25,25,25,35,0],
                   [35,25,25,25,25,25,25,25,25,35],
                   [25,25,25,25,25,25,25,25,25,25],
                   [25,25,25,25,25,25,25,25,25,25],
                   [25,25,25,25,100,100,25,25,25,25],
                   [25,25,25,25,100,100,25,25,25,25],
                   [25,25,25,25,25,25,25,25,25,25],
                   [25,25,25,25,25,25,25,25,25,25],
                   [35,25,25,25,25,25,25,25,25,35],
                   [0,35,25,25,25,25,25,25,35,0]]

HEURISTIC_COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        HEURISTIC_COORDS[(row,col)].append(HEURISTIC_BOARD[row][col])

RED     = 'r'
BLU     = 'b'
RED_SEQ = 'X'
BLU_SEQ = 'O'
JOKER   = '#'
EMPTY   = '_'
TRADSEQ = 1
HOTBSEQ = 2
MULTSEQ = 3

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

#Store dict of cards and their coordinates for fast lookup.
COORDS = defaultdict(list)
for row in range(10):
    for col in range(10):
        COORDS[BOARD[row][col]].append((row,col))

class myAgent(Agent):

    def __init__(self,_id):
        super().__init__(_id)
        self.score=0

    def SelectAction(self,actions,game_state):

        new_state = self.generateSuccessor(game_state, actions[0], self.id)
        print(len(actions))
        print(len(self.getLegalActions(new_state, self.id)))
        quit()

        maxAction={}
        maxScore=0
        try:
            for action in actions:
                if(action["play_card"] not in ['jh','js','jd','jc']):
                    copyGameState=deepcopy(game_state)
                    self.run_simulator(copyGameState,self.id,action,0)
                    if(self.score>maxScore):
                        maxScore=self.score
                        maxAction=action
                        self.score=0
        except Exception:
            traceback.print_exc()
        return maxAction

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

    def isTerminalNode(self,state,action,agent_id):
        r,c=action["coords"]
        if self.checkSeq(state.board.chips,state.agent[agent_id],(r,c))!=(None,None):
            return False
        else:
            return True

    def run_simulator(self,state,agent_id,action,iterations):
        if(iterations<5):
            nextState=self.generateSuccessor(state,action,agent_id)
            nextActions=self.getLegalActions(nextState,agent_id)
            if nextActions:
                nextAction=random.choice(nextActions)
                for item in self.neighbour(nextAction) :
                    if(item in nextState.board.plr_coords[RED]):
                        self.score+=1
                    if(item in [(4,4),(4,5),(5,4),(5,5)]):
                        self.score+=3
                iterations+=1
                self.run_simulator(nextState,agent_id,nextAction,iterations)