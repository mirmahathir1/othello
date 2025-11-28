import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import losses
from tqdm import tqdm

from board import WHITE, BLACK, EMPTY, OthelloGame, play_game
from agents.agent import Agent
from agents.alphabeta import AlphaBetaAgent

class DQNAgent:
    def __init__(self, color, state_size=(8, 8), action_size=64, memory_size=100000, batch_size=64, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.00001):
        self.color = color
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(8, 8, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        # Use string identifier for loss function to avoid serialization issues
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_move(self, game, verbose=False):
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        state = self._preprocess_state(game)
        
        if np.random.rand() <= self.epsilon:
            if verbose:
                print("Random move (exploration)")
            return random.choice(valid_moves)
        else:
            act_values = self.model.predict(state[np.newaxis, ...], verbose=0)
            
            # Convert all possible actions to Q-values
            q_values = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    action_idx = i * 8 + j
                    q_values[i][j] = act_values[0][action_idx]
            
            # Filter only valid moves and select the best one
            max_q = -float('inf')
            best_move = None
            for move in valid_moves:
                row, col = move
                if q_values[row][col] > max_q:
                    max_q = q_values[row][col]
                    best_move = move
            
            if verbose:
                print(f"Predicted move (exploitation) with Q-value: {max_q:.4f}")
            return best_move
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([self._preprocess_memory(t[0]) for t in minibatch])
        next_states = np.array([self._preprocess_memory(t[3]) for t in minibatch])
        
        # Predict Q-values for current and next states
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        X = []
        y = []
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            row, col = action
            action_idx = row * 8 + col
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(next_q[i])
            
            # Update the target for current action
            current_q[i][action_idx] = target
            
            X.append(self._preprocess_memory(state))
            y.append(current_q[i])
        
        # Train the model
        self.model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _preprocess_state(self, game):
        # Convert the game board to a numerical representation from the agent's perspective
        board = np.zeros((8, 8, 1))
        
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == self.color:
                    board[i][j][0] = 1  # Current player's pieces
                elif game.board[i][j] != EMPTY:
                    board[i][j][0] = -1  # Opponent's pieces
        
        return board
    
    def _preprocess_memory(self, game_state):
        # Convert stored game state to neural network input format
        board = np.zeros((8, 8, 1))
        
        for i in range(8):
            for j in range(8):
                if game_state[i][j] == self.color:
                    board[i][j][0] = 1
                elif game_state[i][j] != EMPTY:
                    board[i][j][0] = -1
        
        return board
    
    def save_model(self, filename):
        # Use the new .keras format instead of .h5
        if not filename.endswith('.keras'):
            filename += '.keras'
        self.model.save(filename)
    
    def load_model(self, filename):
        # Handle both .keras and .h5 formats for backward compatibility
        if not filename.endswith('.keras'):
            filename += '.keras'
        self.model = models.load_model(filename)
        self.update_target_model()

class AIAgentDQN(Agent):
    def __init__(self):
        self.color = None
        self.black_dqn = DQNAgent(BLACK)
        self.black_dqn.load_model('dqn_saved/othello_black_dqn.keras')
        self.white_dqn = DQNAgent(WHITE)
        self.white_dqn.load_model('dqn_saved/othello_white_dqn.keras')
    
    def get_move(self, game, verbose):
        if self.color == BLACK:
            return self.black_dqn.get_move(game=game, verbose=verbose)
        elif self.color == WHITE:
            return self.white_dqn.get_move(game=game, verbose=verbose)
        else:
            print("Invalid color")
            exit(0)

def train_dqn_agent(episodes=1000, batch_size=64, verbose=1):
    # Create agents
    player1 = DQNAgent(BLACK)
    player2 = DQNAgent(WHITE)
    
    stats = {
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'black_scores': [],
        'white_scores': []
    }
    
    for e in range(episodes):
        game = OthelloGame()
        state = game.board.copy()
        done = False
        
        # Alternate who starts first
        if e % 2 == 0:
            current_player = BLACK
            opponent = WHITE
        else:
            current_player = WHITE
            opponent = BLACK
        
        while not done:
            # Current player's turn
            if current_player == BLACK:
                move = player1.get_move(game)
            else:
                move = player2.get_move(game)
            
            if move is None:
                # No valid moves, switch player
                current_player, opponent = opponent, current_player
                continue
            
            # Make the move
            row, col = move
            game.make_move(row, col, current_player)
            next_state = game.board.copy()
            
            # Calculate reward
            if game.is_game_over():
                winner = game.get_winner()
                if winner == current_player:
                    reward = 100
                elif winner == opponent:
                    reward = -100
                else:
                    reward = 50  # draw
                done = True
            else:
                # Intermediate reward based on piece difference
                current_score = game.get_score(current_player)
                opponent_score = game.get_score(opponent)
                reward = current_score - opponent_score
            
            # Store the experience
            if current_player == BLACK:
                player1.remember(state, (row, col), reward, next_state, done)
            else:
                player2.remember(state, (row, col), reward, next_state, done)
            
            state = next_state
            current_player, opponent = opponent, current_player
        
        # Update statistics
        black_score = game.get_score(BLACK)
        white_score = game.get_score(WHITE)
        stats['black_scores'].append(black_score)
        stats['white_scores'].append(white_score)
        
        if black_score > white_score:
            stats['black_wins'] += 1
        elif white_score > black_score:
            stats['white_wins'] += 1
        else:
            stats['draws'] += 1
        
        # Train both agents
        player1.replay(batch_size)
        player2.replay(batch_size)
        
        # Update target networks periodically
        if e % 10 == 0:
            player1.update_target_model()
            player2.update_target_model()
        
        if verbose and (e % 100 == 0 or e == episodes - 1):
            print(f"Episode: {e}/{episodes}")
            print(f"Black Wins: {stats['black_wins']}, White Wins: {stats['white_wins']}, Draws: {stats['draws']}")
            print(f"Avg Black Score: {np.mean(stats['black_scores'][-100:]):.1f}, Avg White Score: {np.mean(stats['white_scores'][-100:]):.1f}")
            print(f"Epsilon (Black): {player1.epsilon:.4f}, Epsilon (White): {player2.epsilon:.4f}")
    
    return player1, player2, stats

# # Train the DQN agents (both Black and White)
# black_dqn, white_dqn, stats = train_dqn_agent(episodes=10000, verbose=1)

# # Save the trained models using the new format
# black_dqn.save_model('dqn_saved/othello_black_dqn')  # Will save as .keras
# white_dqn.save_model('dqn_saved/othello_white_dqn')
