import numpy as np
from tqdm import tqdm
from board import WHITE, BLACK, play_game
from agents.human import HumanAgent
from agents.alphabeta import AlphaBetaAgent
from agents.hybrid import HybridAgent
from agents.qlearning import QLearningAgent
from agents.dqn import AIAgentDQN
from agents.greedy import GreedyAgent
from agents.human import HumanAgent
from agents.mcts import MCTSAgent
from agents.random import RandomAgent
from agents.mdp import MDPValueIterationWrapper
from agents.multiarmedbandid import MCTSAgentMultiArmed
from agents.policygradient import PolicyGradientAgent

np.random.seed = 42

def game_menu():
    print("Welcome to Othello (Reversi)!")
    
    choice = input("black player: Human (1) or AI(2)? ").strip()
    if choice == "1":
        black_player = HumanAgent()
    elif choice == "2":
        depth = int(input("Enter AI difficulty level (1-5, higher is harder): ").strip())
        black_player = AlphaBetaAgent(depth)

    choice = input("white player: Human (1) or AI(2)? ").strip()
    if choice == "1":
        white_player = HumanAgent()
    elif choice == "2":
        depth = int(input("Enter AI difficulty level (1-5, higher is harder): ").strip())
        white_player = AlphaBetaAgent(depth)
    
    winner_player = play_game(black_player, white_player, verbose=True)

    print(f"winner: {winner_player}")

def game_simulation(player_one_agent, player_two_agent, trial_count):
    player_one_win = 0
    player_two_win = 0

    for trial in tqdm(range(trial_count)):
        if trial % 2 == 0:
            winner = play_game(black_player=player_one_agent, white_player=player_two_agent, verbose=False)
            if winner == BLACK:
                player_one_win+= 1
            elif winner == WHITE:
                player_two_win+= 1
        else:
            winner = play_game(black_player=player_two_agent, white_player=player_one_agent, verbose=False)
            if winner == BLACK:
                player_two_win+= 1
            elif winner == WHITE:
                player_one_win+= 1
    
    print(f"player 1 wins: {player_one_win}, player 2 wins: {player_two_win}, total games: {trial_count}")
    return {
        "player1": player_one_win / trial_count,
        "player2": player_two_win / trial_count
    }

if __name__ == "__main__":
    # player_one_agent = QLearningAgent(epsilon=0.05, q_file="q_values/q_values.pkl") # 3
    # player_one_agent = AIAgentDQN() # 4
    player_one_agent = AlphaBetaAgent(depth=4) # 1
    # player_one_agent = AlphaBetaAgent(depth=3)
    # player_two_agent = GreedyAgent()
    # player_two_agent = HybridAgent()
    # player_two_agent = RandomAgent()
    # player_two_agent = MCTSAgent() # 5
    # player_two_agent = MDPValueIterationWrapper() # 2
    # player_two_agent = MCTSAgentMultiArmed() # 1
    player_two_agent = PolicyGradientAgent()
    game_simulation(player_one_agent=player_one_agent, player_two_agent=player_two_agent, trial_count=10)

    # game_menu()
    pass
