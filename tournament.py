import numpy as np
from tqdm import tqdm
from board import WHITE, BLACK, play_game
from agents.human import HumanAgent
from agents.alphabeta import AlphaBetaAgent
from agents.hybrid import HybridAgent
from agents.qlearning import QLearningAgent
from agents.dqn import AIAgentDQN
from agents.greedy import GreedyAgent
from agents.mcts import MCTSAgent
from agents.random import RandomAgent
from agents.mdp import MDPValueIterationWrapper
from agents.multiarmedbandid import MCTSAgentMultiArmed
from agents.policygradient import PolicyGradientAgent
from interface import game_simulation
import json

np.random.seed = 42

def initialize_agents():
    """Initialize all agents with their parameters"""
    agents = {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
        "AlphaBeta1": AlphaBetaAgent(depth=1),
        "AlphaBeta3": AlphaBetaAgent(depth=3),
        "AlphaBeta4": AlphaBetaAgent(depth=4),
        "MCTS": MCTSAgent(),
        "MCTS-MAB": MCTSAgentMultiArmed(),
        "MDP-VI": MDPValueIterationWrapper(),
        "QLearning": QLearningAgent(epsilon=0.05, q_file="q_values/q_values.pkl"),
        "DQN": AIAgentDQN(),
        "Hybrid": HybridAgent(),
        "PolicyGradient": PolicyGradientAgent()
    }
    return agents

def run_tournament(agents, games_per_match=10):
    """
    Run a round-robin tournament where each agent plays every other agent
    Args:
        agents: Dictionary of agent names to agent instances
        games_per_match: Number of games per agent pair
    Returns:
        Dictionary of dictionaries containing win percentages
    """
    results = {agent_name: {} for agent_name in agents.keys()}
    
    agent_names = list(agents.keys())
    
    for i, name1 in enumerate(agent_names):
        for j, name2 in enumerate(agent_names):
            if i == j:
                continue  # Skip self-play
                
            if name2 in results[name1]:
                continue  # Skip duplicate matches
                
            print(f"\nMatchup: {name1} vs {name2}")
            agent1 = agents[name1]
            agent2 = agents[name2]
            
            # Run the simulation
            matchup_results = game_simulation(
                player_one_agent=agent1,
                player_two_agent=agent2,
                trial_count=games_per_match
            )
            
            # Store results for both perspectives
            results[name1][name2] = matchup_results["player1"]
            results[name2][name1] = matchup_results["player2"]
    
    return results

def print_tournament_results(results):
    """Pretty print the tournament results"""
    print("\nTournament Results:")
    print("="*80)
    
    # Print header
    header = "Agent".ljust(20)
    agent_names = list(results.keys())
    for name in agent_names:
        header += name[:8].rjust(10)  # Truncate long names
    print(header)
    print("-"*80)
    
    # Print each row
    for agent, opponents in results.items():
        row = agent.ljust(20)
        for opponent in agent_names:
            if opponent == agent:
                row += "    -    ".rjust(10)
            else:
                win_pct = opponents.get(opponent, 0)
                row += f"{win_pct:.1%}".rjust(10)
        print(row)
    
    print("="*80)

    json.dump(results, open("tournament_result.json", "w"), indent=4)

if __name__ == "__main__":
    # Initialize all agents
    agents = initialize_agents()
    
    # Run the tournament (10 games per matchup)
    tournament_results = run_tournament(agents, games_per_match=10)
    
    # Print the results
    print_tournament_results(tournament_results)
    
    # You can also access specific results programmatically:
    # For example: tournament_results["AlphaBeta3"]["MCTS"] gives AlphaBeta3's win % against MCTS