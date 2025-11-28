import pickle
import os
import numpy as np
from tqdm import tqdm
import random
from agents.agent import Agent
from agents.alphabeta import AlphaBetaAgent
from board import OthelloGame,WHITE, BLACK, play_game

class QLearningAgent(Agent):
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9, q_file="q_values/q_values.pkl", training=False):
        self.color = None
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_file = q_file    # File to save/load Q-values
        self.q_values = {}      # State-action values
        self.last_state = None
        self.last_action = None
        self.training = training  # Whether agent is in training mode
        
        # Load Q-values if file exists and not in training mode
        if not training and os.path.exists(q_file):
            with open(q_file, 'rb') as f:
                self.q_values = pickle.load(f)
    
    def save_q_values(self):
        """Save Q-values to disk"""
        with open(self.q_file, 'wb') as f:
            pickle.dump(self.q_values, f)
    
    def get_state_key(self, game):
        """Convert the board state to a hashable key"""
        # Use a more compact representation - just the board and current player
        return (tuple(game.board.flatten().tolist()), game.current_player)
    
    def get_move(self, game, verbose):
        """Get the best move based on Q-values or explore randomly"""
        state = self.get_state_key(game)
        valid_moves = game.get_valid_moves(self.color)
        
        if not valid_moves:
            return None
        
        # During training, use epsilon-greedy policy
        if self.training and random.random() < self.epsilon:
            move = random.choice(valid_moves)
        else:
            # Get Q-values for all valid moves (default to 0 if not found)
            q_values = [self.q_values.get((state, move), 0) for move in valid_moves]
            max_q = max(q_values)
            best_moves = [move for move, q in zip(valid_moves, q_values) if q == max_q]
            move = random.choice(best_moves)  # Random choice if multiple best moves
        
        # Store last state and action for Q-learning update
        self.last_state = state
        self.last_action = move
        
        return move
    
    def update_q_values(self, reward, new_game):
        """Update Q-values based on reward and new state"""
        if not self.training or self.last_state is None or self.last_action is None:
            return
        
        new_state = self.get_state_key(new_game)
        new_valid_moves = new_game.get_valid_moves(self.color)
        
        # Calculate max Q-value for new state
        if new_valid_moves:
            max_q_new = max([self.q_values.get((new_state, move), 0) for move in new_valid_moves])
        else:
            max_q_new = 0
        
        # Get current Q-value
        current_q = self.q_values.get((self.last_state, self.last_action), 0)
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_new - current_q)
        self.q_values[(self.last_state, self.last_action)] = new_q
        
        # Reset last state and action
        self.last_state = None
        self.last_action = None
    
    def game_over(self, result):
        """Handle end of game with final reward"""
        if not self.training:
            return
        
        if result == self.color:
            reward = 1  # Win
        elif result == 0:
            reward = 0.5  # Draw
        else:
            reward = -1  # Loss
        
        # Update Q-values with final reward
        if self.last_state is not None and self.last_action is not None:
            current_q = self.q_values.get((self.last_state, self.last_action), 0)
            new_q = current_q + self.alpha * (reward - current_q)
            self.q_values[(self.last_state, self.last_action)] = new_q
        
        # Save Q-values periodically during training
        if random.random() < 0.1:  # 10% chance to save after each game
            self.save_q_values()

def train_q_agent(episodes=1000, q_file="q_values/q_values.pkl"):
    """Train the Q-learning agent against itself"""
    q_agent_black = QLearningAgent(epsilon=0.3, alpha=0.2, gamma=0.9, 
                                  q_file=q_file, training=True)
    q_agent_white = QLearningAgent(epsilon=0.3, alpha=0.2, gamma=0.9, 
                                  q_file=q_file, training=True)
    
    # Load existing Q-values if available
    if os.path.exists(q_file):
        with open(q_file, 'rb') as f:
            q_values = pickle.load(f)
            q_agent_black.q_values = q_values
            q_agent_white.q_values = q_values
    
    print(f"Training Q-learning agent for {episodes} episodes...")
    for episode in tqdm(range(episodes)):
        # Alternate who goes first
        if episode % 2 == 0:
            black_player, white_player = q_agent_black, q_agent_white
        else:
            black_player, white_player = q_agent_white, q_agent_black
        
        game = OthelloGame()
        players = {BLACK: black_player, WHITE: white_player}
        
        while True:
            current_player = players[game.current_player]
            
            # Check if game is over
            if game.is_game_over(game.current_player):
                game.current_player = WHITE if game.current_player == BLACK else BLACK
                if game.is_game_over(game.current_player):
                    break
                continue
            
            # Get move and make it
            move = current_player.get_move(game, False)
            if move is None:
                game.current_player = WHITE if game.current_player == BLACK else BLACK
                continue
            
            row, col = move
            old_game = game.copy()
            game.make_move(row, col)
            
            # Update Q-values with intermediate reward (0)
            current_player.update_q_values(0, game)
        
        # Final game result
        winner = game.get_winner()
        black_player.game_over(winner)
        white_player.game_over(winner)
    
    # Save final Q-values
    q_agent_black.save_q_values()
    print("Training complete. Q-values saved to", q_file)

if __name__ == "__main__":
    # First train the Q-learning agent
    train_q_agent(episodes=5000)
