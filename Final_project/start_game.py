import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from src.agent import setup_ai as DQN_ai
from src.MC_player import setup_ai as MC_ai
from baseline5 import setup_ai as baseline_ai

config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=20)
config.register_player(name="p1", algorithm=baseline_ai())
# config.register_player(name="p2", algorithm=random_ai())
config.register_player(name="p3", algorithm=DQN_ai())
## Play in interactive mode if uncomment
#config.register_player(name="me", algorithm=console_ai())
game_result = start_poker(config, verbose=1)

#print(json.dumps(game_result, indent=4))
