import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from agents.agent import Agent
from board import BLACK, WHITE, EMPTY, OthelloGame
from tqdm import tqdm

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class PolicyGradientAgent(Agent):
    def __init__(self, learning_rate=0.0001, gamma=0.99, model_path="policy_gradients_saved/policy_gradient_model.pth"):
        self.color = None
        self.gamma = gamma
        self.model_path = model_path
        
        # Initialize network
        self.policy_net = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model()
        
        # Episode data
        self.saved_log_probs = []
        self.rewards = []
        self.episode_states = []
    
    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
    
    def load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_path))
        self.policy_net.eval()
    
    def board_to_tensor(self, game):
        """Convert the game board to a tensor representation"""
        # Flatten the board and convert to player's perspective
        board = game.board.flatten()
        if self.color == WHITE:
            board = np.where(board == BLACK, WHITE, np.where(board == WHITE, BLACK, EMPTY))
        return torch.FloatTensor(board)
    
    def get_move(self, game, verbose=False):
        current_color = self.color
        valid_moves = game.get_valid_moves(current_color)
        
        if not valid_moves:
            return None  # No valid moves means we have to pass
        
        # Convert board to tensor
        state = self.board_to_tensor(game)
        
        # Get action probabilities
        probs = self.policy_net(state)
        
        # Create a mask for valid moves
        action_mask = torch.zeros(64)
        for move in valid_moves:
            row, col = move
            action_idx = row * 8 + col
            action_mask[action_idx] = 1
        
        # Apply mask and renormalize
        masked_probs = probs * action_mask
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            # If all valid moves have zero probability, fall back to uniform distribution
            masked_probs = action_mask / action_mask.sum()
        
        # Sample an action
        m = Categorical(masked_probs)
        action_idx = m.sample()
        
        # Save log probability for training
        log_prob = m.log_prob(action_idx)
        self.saved_log_probs.append(log_prob)
        self.episode_states.append(state)
        
        # Convert action index back to board coordinates
        row = action_idx.item() // 8
        col = action_idx.item() % 8
        return (row, col)
    
    def update_policy(self, final_result):
        """Update policy based on episode outcome"""
        if not self.saved_log_probs:  # No moves were made in this episode
            return
        
        # Calculate discounted rewards
        R = 0
        discounted_rewards = []
        
        # The final result is 1 for win, 0 for draw, -1 for loss
        reward = final_result
        
        # Distribute the final reward to all steps in the episode
        for _ in reversed(range(len(self.saved_log_probs))):
            R = self.gamma * R + reward
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        
        # Only normalize if we have more than one reward
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        # Only update if we have losses to backpropagate
        if policy_loss:
            # Update network
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()  # Changed from cat to stack
            policy_loss.backward()
            self.optimizer.step()
        
        # Reset episode data
        self.saved_log_probs = []
        self.rewards = []
        self.episode_states = []
    
    def train(self, num_episodes=1000, opponent=None, verbose=False):
        """Train the agent by playing against itself or another opponent"""
        if opponent is None:
            opponent = PolicyGradientAgent()
            opponent.color = WHITE if self.color == BLACK else BLACK
        
        for episode in tqdm(range(num_episodes)):
            # Alternate who starts first
            if episode % 2 == 0:
                black_player = self
                white_player = opponent
                self.color = BLACK
                opponent.color = WHITE
            else:
                black_player = opponent
                white_player = self
                self.color = WHITE
                opponent.color = BLACK
            
            # Play the game
            game = OthelloGame()
            
            while True:
                if game.is_game_over():
                    break
                
                current_player = black_player if game.current_player == BLACK else white_player
                move = current_player.get_move(game, verbose)
                
                if move is None:
                    # Player has to pass
                    game.current_player = WHITE if game.current_player == BLACK else BLACK
                    continue
                
                game.make_move(move[0], move[1])
            
            # Determine the result from our perspective
            winner = game.get_winner()
            if winner == self.color:
                final_result = 1
            elif winner == EMPTY:
                final_result = 0
            else:
                final_result = -1
            
            # Update policy
            self.update_policy(final_result)
            if isinstance(opponent, PolicyGradientAgent):
                opponent.update_policy(-final_result)
            
            # Save model periodically
            if episode % 100 == 0:
                self.save_model()
                if verbose:
                    print(f"Episode {episode}, result: {final_result}")
        
        self.save_model()

if __name__ == "__main__":
    # First train the agent (this would take time)
    pg_agent = PolicyGradientAgent()
    pg_agent.train(num_episodes=1000)  # Train against itself
