import numpy as np
import sys
sys.path.insert(0, '../')
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.hand_evaluator import HandEvaluator

import random


suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())

def gen_card_im(card):
    a = np.zeros((4, 13))
    s = suits.index(card.suit)
    r = ranks.index(card.rank)
    a[s, r] = 1
    return np.pad(a, ((6, 7), (2, 2)), 'constant', constant_values=0)

streep_map = {
    'preflop': 0,
    'flop': 1,
    'turn': 2,
    'river': 3
}

def get_street(s):
    val = [0, 0, 0, 0]
    val[streep_map[s]] = 1
    return val

def process_img(img):
    return np.reshape(img, [17 * 17 * 1])

class ExperienceBuffer():
    def __init__(self, buffer_size=5_000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(self.buffer) + len(experience) - self.buffer_size] = []
        self.buffer.extend(experience)
    
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 7])


def get_action_by_num(action_num, valid_actions, is_train=True):
    if action_num == 0:
        action, amount = valid_actions[0]['action'], valid_actions[0]['amount']
    elif action_num == 1:
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    elif action_num == 2:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']
    elif action_num == 3:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']
    elif action_num == 4:
        action, amount = valid_actions[2]['action'], int(valid_actions[2]['amount']['max'] // 2)
        
    if not is_train and amount == -1:
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    return action, amount

def img_from_state(hole_card, round_state):
    imgs = np.zeros((8, 17, 17))
    for i, c in enumerate(hole_card):
        imgs[i] = gen_card_im(Card.from_str(c))

    for i, c in enumerate(round_state['community_card']):
        imgs[i + 2] = gen_card_im(Card.from_str(c))

    imgs[7] = imgs[:7].sum(axis=0)
#     return imgs
    return np.swapaxes(imgs, 0, 2)[:, :, -1:]

def gen_cards(cards_str):
#     print(type(cards_str[0]))
    return [Card.from_str(s) for s in cards_str]

def estimate_hole_card_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if not community_card: community_card = []
    community_card = gen_cards(community_card)
    hole_card = gen_cards(hole_card)
    win_count = sum([_montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation

def gen_deck(exclude_cards=None):
    deck_ids = range(1, 53)
    if exclude_cards:
        assert isinstance(exclude_cards, list)
        if isinstance(exclude_cards[0], str):
            exclude_cards = [Card.from_str(s) for s in exclude_cards]
        exclude_ids = [card.to_id() for card in exclude_cards]
        deck_ids = [i for i in deck_ids if not i in exclude_ids]
    return Deck(deck_ids)

def evaluate_hand(hole_card, community_card):
    assert len(hole_card)==2 and len(community_card)==5
    hand_info = HandEvaluator.gen_hand_rank_info(hole_card, community_card)
    return {
            "hand": hand_info["hand"]["strength"],
            "strength": HandEvaluator.eval_hand(hole_card, community_card)
            }

def _montecarlo_simulation(nb_player, hole_card, community_card):
    community_card = _fill_community_card(community_card, used_card=hole_card+community_card)
    unused_cards = _pick_unused_card((nb_player-1)*2, hole_card + community_card)
    opponents_hole = [unused_cards[2*i:2*i+2] for i in range(nb_player-1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

def _fill_community_card(base_cards, used_card):
    need_num = 5 - len(base_cards)
    return base_cards + _pick_unused_card(need_num, used_card)

def _pick_unused_card(card_num, used_card):
    used = [card.to_id() for card in used_card]
    unused = [card_id for card_id in range(1, 53) if card_id not in used]
    choiced = random.sample(unused, card_num)
    return [Card.from_id(card_id) for card_id in choiced]