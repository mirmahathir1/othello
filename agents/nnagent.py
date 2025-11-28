import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
import os
from board import WHITE, BLACK, EMPTY, OthelloGame, play_game
from agents.agent import Agent
from agents.multiarmedbandid import MCTSAgentMultiArmed
from tqdm import tqdm
import pickle

class OthelloDataset(Dataset):
    def __init__(self, states, policies):
        self.states = states
        self.policies = policies
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx]

class OthelloNN(nn.Module):
    def __init__(self):
        super(OthelloNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc_policy = nn.Linear(512, 8 * 8)  # Outputs move probabilities
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 8, 8)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        
        policy = self.fc_policy(x)
        policy = policy.view(-1, 8, 8)  # Reshape to (batch_size, 8, 8)
        
        return policy

class NeuralNetAgent(Agent):
    def __init__(self, model_path="data_nn/othello_nn.pth"):
        self.color = None
        self.model_path = model_path
        
        # Load or initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OthelloNN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Initialized new model (no saved model found)")
        
        self.model.eval()
    
    def get_move(self, game, verbose):
        # Convert game state to input tensor
        input_tensor = self.game_state_to_tensor(game)
        
        # Get model prediction
        with torch.no_grad():
            policy = self.model(input_tensor)
            policy = policy.squeeze(0).cpu().numpy()  # Remove batch dim and convert to numpy
        
        # Get valid moves and mask invalid ones
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        # Create mask for valid moves
        move_probs = np.zeros((8, 8))
        for move in valid_moves:
            row, col = move
            move_probs[row, col] = policy[row, col]
        
        # Softmax over valid moves
        valid_indices = [(move[0], move[1]) for move in valid_moves]
        rows, cols = zip(*valid_indices)
        valid_probs = policy[rows, cols]
        valid_probs = np.exp(valid_probs) / np.sum(np.exp(valid_probs))
        
        # Select move with highest probability
        best_idx = np.argmax(valid_probs)
        best_move = valid_moves[best_idx]
        
        if verbose:
            print("Move probabilities:")
            for move, prob in zip(valid_moves, valid_probs):
                print(f"  {move}: {prob:.2%}")
        
        return best_move
    
    def game_state_to_tensor(self, game):
        # Create a 3-channel board representation:
        # Channel 0: our pieces (1 where our color, 0 elsewhere)
        # Channel 1: opponent pieces (1 where opponent color, 0 elsewhere)
        # Channel 2: empty spaces (1 where empty, 0 elsewhere)
        
        board = game.board
        our_color = self.color
        opponent_color = WHITE if our_color == BLACK else BLACK
        
        our_pieces = (board == our_color).astype(np.float32)
        opponent_pieces = (board == opponent_color).astype(np.float32)
        empty = (board == EMPTY).astype(np.float32)
        
        # Stack channels and add batch dimension
        state = np.stack([our_pieces, opponent_pieces, empty], axis=0)
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        
        return state
    
    @staticmethod
    def generate_training_data(teacher_agent=MCTSAgentMultiArmed(),
                             num_games=1000,
                             data_path="data_nn/othello_training_data.pkl"):
        """Generate and save training data from teacher agent"""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        states = []
        policies = []
        
        print(f"Generating training data from {num_games} games...")
        for game_num in tqdm(range(num_games)):
            game = OthelloGame()
            teacher_agent.color = BLACK if game_num % 2 == 0 else WHITE
            
            while True:
                # Check if game is over
                current_player = game.current_player
                if game.is_game_over(current_player):
                    # Switch player and check if they also have no moves
                    game.current_player = WHITE if current_player == BLACK else BLACK
                    if game.is_game_over(game.current_player):
                        break  # Game over
                    continue
                
                # Get teacher move probabilities
                if current_player == teacher_agent.color:
                    # Get all valid moves
                    valid_moves = game.get_valid_moves(current_player)
                    
                    # Get teacher's move probabilities
                    move_stats = {move: {'wins': 0, 'plays': 0} for move in valid_moves}
                    total_plays = 100  # Run 100 iterations to get good probabilities
                    
                    for _ in range(total_plays):
                        # Select a move randomly (since we're just collecting stats)
                        move = random.choice(valid_moves)
                        
                        # Simulate the game from this move
                        result = teacher_agent.simulate_game(game, move)
                        
                        # Update statistics
                        move_stats[move]['plays'] += 1
                        if result == teacher_agent.color:
                            move_stats[move]['wins'] += 1
                    
                    # Create policy vector
                    policy = np.zeros((8, 8))
                    total_wins = sum(stats['wins'] for stats in move_stats.values())
                    
                    if total_wins > 0:
                        for move, stats in move_stats.items():
                            row, col = move
                            policy[row, col] = stats['wins'] / stats['plays'] if stats['plays'] > 0 else 0
                    
                    # Normalize policy
                    policy_sum = policy.sum()
                    if policy_sum > 0:
                        policy /= policy_sum
                    
                    # Save state and policy
                    state = NeuralNetAgent.game_state_to_tensor_static(game, teacher_agent.color)
                    states.append(state.squeeze(0).cpu().numpy())
                    policies.append(policy)
                
                # Make a move (teacher or random)
                valid_moves = game.get_valid_moves(current_player)
                if current_player == teacher_agent.color:
                    # Use teacher's actual move (not the probabilities we calculated)
                    move = teacher_agent.get_move(game, verbose=False)
                else:
                    # Random move for opponent
                    move = random.choice(valid_moves) if valid_moves else None
                
                if move is None:
                    game.current_player = WHITE if current_player == BLACK else BLACK
                    continue
                
                row, col = move
                game.make_move(row, col)
            
            if (game_num + 1) % 10 == 0:
                print(f"Generated {game_num + 1}/{num_games} games")
        
        # Save the data
        data = {
            'states': np.array(states),
            'policies': np.array(policies)
        }
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Training data saved to {data_path}")
        return data
    
    @staticmethod
    def train(data_path="data_nn/othello_training_data.pkl",
              batch_size=64,
              learning_rate=0.0001,
              model_path="data_nn/othello_nn.pth",
              epochs=10):
        """Train the model using pre-generated training data"""
        
        # Load training data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No training data found at {data_path}. "
                                   "Please generate data first using generate_training_data().")
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        states = data['states']
        policies = data['policies']
        
        # Initialize model and training components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create dataset and dataloader
        dataset = OthelloDataset(states, policies)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train the model
        print("Training model...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (state_batch, policy_batch) in enumerate(dataloader):
                state_batch = state_batch.to(device)
                policy_batch = policy_batch.to(device)
                
                # Flatten the policy for CrossEntropyLoss
                batch_size = state_batch.size(0)
                policy_batch = policy_batch.view(batch_size, -1)  # Flatten to (batch, 64)
                
                optimizer.zero_grad()
                output = model(state_batch)
                output = output.view(batch_size, -1)  # Flatten to (batch, 64)
                
                loss = criterion(output, policy_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    @staticmethod
    def game_state_to_tensor_static(game, our_color):
        # Static version of game_state_to_tensor for use during training
        board = game.board
        opponent_color = WHITE if our_color == BLACK else BLACK
        
        our_pieces = (board == our_color).astype(np.float32)
        opponent_pieces = (board == opponent_color).astype(np.float32)
        empty = (board == EMPTY).astype(np.float32)
        
        # Stack channels and add batch dimension
        state = np.stack([our_pieces, opponent_pieces, empty], axis=0)
        state = torch.from_numpy(state).unsqueeze(0).float()
        
        return state

if __name__ == '__main__':
    teacher = MCTSAgentMultiArmed()
    NeuralNetAgent.generate_training_data(teacher_agent=teacher, num_games=1000)
    NeuralNetAgent.train(epochs=100)
    nn_agent = NeuralNetAgent()
    result = play_game(MCTSAgentMultiArmed(), nn_agent, verbose=True)
    if result == BLACK:
        print("MCTSMAB won")
    elif result == WHITE:
        print("NN won")
