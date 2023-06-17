import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import sys
sys.path.insert(0, '../')
from game.players import BasePokerPlayer

import logging
import time

import numpy as np

from src.util import *

import json
import pickle

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.wins = 0
        self.losses = 0
        self.n_simulations = 1000

    def declare_action(self, valid_actions, hole_card, round_state):
        # Estimate the win rate
        win_rate = estimate_hole_card_win_rate(self.n_simulations, len(round_state['seats']), hole_card, round_state['community_card'])
        # Check whether it is possible to call
        can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
        if can_call:
            # If so, compute the amount that needs to be called
            call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
        else:
            call_amount = 0

        amount = None

        # If the win rate is large enough, then raise
        if win_rate > 0.5:
            raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
            if win_rate > 0.85:
                # If it is extremely likely to win, then raise as much as possible
                action = 'raise'
                amount = raise_amount_options['max']
            elif win_rate > 0.75:
                # If it is likely to win, then raise by the minimum amount possible
                action = 'raise'
                amount = raise_amount_options['min']
            else:
                # If there is a chance to win, then call
                action = 'call'
        else:
            action = 'call' if can_call and call_amount == 0 else 'fold'

        # Set the amount
        if amount is None:
            items = [item for item in valid_actions if item['action'] == action]
            amount = items[0]['amount']

        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = self.uuid in [item['uuid'] for item in winners]
        self.wins += int(is_winner)
        self.losses += int(not is_winner)
        
def setup_ai():
    return MonteCarloPlayer()
