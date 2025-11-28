import pickle
import os
import numpy as np
from tqdm import tqdm
import time
from agents.agent import Agent
from board import WHITE, BLACK, play_game, OthelloGame
from agents.alphabeta import AlphaBetaAgent
import random
random.seed(42)

class MDPValueIterationAgent():
    def __init__(self, gamma=0.9, theta=1e-6, max_iterations=1000, 
                 value_file="mdp_values.pkl", training=False):
        self.color = None
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.max_iterations = max_iterations  # Max iterations for value iteration
        self.value_file = value_file  # File to store learned values
        self.state_values = {}  # V(s) - state value function
        self.policy = {}  # π(s) - policy function
        self.training = training
        
        if not training and os.path.exists(value_file):
            self.load_values()
    
    def save_values(self):
        """Save the learned state values and policy to disk"""
        with open(self.value_file, 'wb') as f:
            pickle.dump({'state_values': self.state_values, 'policy': self.policy}, f)
    
    def load_values(self):
        """Load state values and policy from disk"""
        with open(self.value_file, 'rb') as f:
            data = pickle.load(f)
            self.state_values = data['state_values']
            self.policy = data['policy']
    
    def get_state_key(self, game):
        """Convert game state to a compact representation"""
        # Use board configuration and current player as state
        return (tuple(game.board.flatten().tolist()), game.current_player)
    
    def get_reward(self, game, player):
        """Calculate immediate reward for a state"""
        if game.is_game_over():
            winner = game.get_winner()
            if winner == player:
                return 1  # Win
            elif winner == 0:
                return 0.5  # Draw
            else:
                return -1  # Loss
        
        # Intermediate rewards based on piece difference
        my_pieces = np.sum(game.board == player)
        opp_pieces = np.sum(game.board == (WHITE if player == BLACK else BLACK))
        return (my_pieces - opp_pieces) / (my_pieces + opp_pieces + 1e-6)
    
    def value_iteration(self):
        """Perform value iteration to learn state values"""
        print("Starting value iteration...")
        
        # Initialize state values
        self.state_values = {}
        self.policy = {}
        
        # We'll need to explore all possible states through simulation
        # This is computationally intensive, so we'll limit the depth
        max_depth = 4  # Look ahead 4 moves
        
        for _ in tqdm(range(self.max_iterations)):
            delta = 0
            
            # Simulate games to explore state space
            game = OthelloGame()
            states_visited = set()
            
            # Use DFS to explore states up to max_depth
            stack = [(game.copy(), 0)]
            
            while stack:
                current_game, depth = stack.pop()
                state = self.get_state_key(current_game)
                
                if state in states_visited or depth > max_depth:
                    continue
                
                states_visited.add(state)
                
                # Initialize state value if not present
                if state not in self.state_values:
                    self.state_values[state] = 0
                
                # Store old value for convergence check
                old_value = self.state_values[state]
                
                if current_game.is_game_over():
                    # Terminal state - value is the reward
                    self.state_values[state] = self.get_reward(current_game, self.color)
                    delta = max(delta, abs(old_value - self.state_values[state]))
                    continue
                
                # Get all possible actions (moves)
                valid_moves = current_game.get_valid_moves(current_game.current_player)
                
                if not valid_moves:
                    # No valid moves - skip to opponent's turn
                    new_game = current_game.copy()
                    new_game.current_player = WHITE if current_game.current_player == BLACK else BLACK
                    stack.append((new_game, depth + 1))
                    continue
                
                max_value = -float('inf')
                best_action = None
                
                for action in valid_moves:
                    # Simulate the action
                    new_game = current_game.copy()
                    new_game.make_move(action[0], action[1], new_game.current_player)
                    
                    # Get the new state
                    new_state = self.get_state_key(new_game)
                    
                    # Calculate the value
                    reward = self.get_reward(new_game, self.color)
                    new_value = reward + self.gamma * self.state_values.get(new_state, 0)
                    
                    if new_value > max_value:
                        max_value = new_value
                        best_action = action
                
                # Update state value and policy
                self.state_values[state] = max_value
                self.policy[state] = best_action
                delta = max(delta, abs(old_value - self.state_values[state]))
                
                # Push successor states to stack
                for action in valid_moves:
                    new_game = current_game.copy()
                    new_game.make_move(action[0], action[1], new_game.current_player)
                    stack.append((new_game, depth + 1))
            
            # Check for convergence
            if delta < self.theta:
                print(f"Value iteration converged after {_ + 1} iterations")
                break
        
        self.save_values()
        print("Value iteration completed")
    
    def get_move(self, game, verbose):
        """Get the best move based on learned policy"""
        state = self.get_state_key(game)
        valid_moves = game.get_valid_moves(self.color)
        
        if not valid_moves:
            return None
        
        # If we have a policy for this state, use it
        if state in self.policy and self.policy[state] in valid_moves:
            return self.policy[state]
        
        # Otherwise fall back to greedy policy based on state values
        best_move = None
        best_value = -float('inf')
        
        for move in valid_moves:
            new_game = game.copy()
            new_game.make_move(move[0], move[1], self.color)
            new_state = self.get_state_key(new_game)
            value = self.state_values.get(new_state, 0)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move is not None else random.choice(valid_moves)
    
class MDPValueIterationWrapper(Agent):
    def __init__(self):
        self.color = None
        self.mdp_black = MDPValueIterationAgent(value_file="mdp_values/mdp_values_black.pkl")
        self.mdp_black.color = BLACK
        
        self.mdp_white = MDPValueIterationAgent(value_file="mdp_values/mdp_values_white.pkl")
        self.mdp_white.color = WHITE
    
    def get_move(self, game, verbose):
        if self.color == BLACK:
            return self.mdp_black.get_move(game, verbose=verbose)
        elif self.color == WHITE:
            return self.mdp_white.get_move(game, verbose)
        else:
            print("wrong color")
            exit(0)

def train_mdp_agent():
    """Train the MDP agent using value iteration"""
    # Create and train a black player agent
    black_agent = MDPValueIterationAgent(gamma=0.9, theta=1e-4, 
                                       max_iterations=100,
                                       value_file="mdp_values/mdp_values_black.pkl",
                                       training=True)
    black_agent.color = BLACK
    black_agent.value_iteration()
    
    # Create and train a white player agent
    white_agent = MDPValueIterationAgent(gamma=0.9, theta=1e-4,
                                       max_iterations=100,
                                       value_file="mdp_values/mdp_values_white.pkl",
                                       training=True)
    white_agent.color = WHITE
    white_agent.value_iteration()

if __name__ == "__main__":
    # First train the MDP agents
    train_mdp_agent()
