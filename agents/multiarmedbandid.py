import math
import random
import time
from agents.agent import Agent
from board import WHITE, BLACK

class MCTSAgentMultiArmed(Agent):
    def __init__(self, iterations=100, exploration_weight=math.sqrt(2)):
        self.color = None
        self.iterations = iterations
        self.exploration_weight = exploration_weight
    
    def get_move(self, game, verbose):
        start_time = time.time()
        
        # Get all valid moves
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        # If only one move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Initialize move statistics
        move_stats = {move: {'wins': 0, 'plays': 0} for move in valid_moves}
        
        # Run simulations
        for _ in range(self.iterations):
            # Select a move using UCB
            total_plays = sum(stat['plays'] for stat in move_stats.values())
            if total_plays == 0:
                # First play - select randomly
                move = random.choice(valid_moves)
            else:
                # Use UCB to select a move
                best_score = -float('inf')
                best_move = None
                
                for move in valid_moves:
                    if move_stats[move]['plays'] == 0:
                        # If a move hasn't been tried yet, give it maximum priority
                        best_move = move
                        break
                    
                    # UCB formula
                    win_rate = move_stats[move]['wins'] / move_stats[move]['plays']
                    exploration = math.sqrt(math.log(total_plays) / move_stats[move]['plays'])
                    score = win_rate + self.exploration_weight * exploration
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                move = best_move
            
            # Simulate the game from this move
            result = self.simulate_game(game, move)
            
            # Update statistics
            move_stats[move]['plays'] += 1
            if result == self.color:
                move_stats[move]['wins'] += 1
        
        # Select the move with the highest win rate
        best_move = max(move_stats.keys(), 
                       key=lambda m: move_stats[m]['wins'] / move_stats[m]['plays'] if move_stats[m]['plays'] > 0 else 0)
        
        end_time = time.time()
        if verbose:
            print(f"MCTS Agent ({'X' if self.color == BLACK else 'O'}) thought for {end_time - start_time:.2f} seconds")
            print("Move statistics:")
            for move, stats in move_stats.items():
                if stats['plays'] > 0:
                    print(f"  {move}: {stats['wins']}/{stats['plays']} ({stats['wins']/stats['plays']:.2%})")
        
        return best_move
    
    def simulate_game(self, game, first_move):
        """Simulate a game starting with the given move, returning the winner"""
        sim_game = game.copy()
        sim_game.make_move(first_move[0], first_move[1], self.color)
        
        current_player = sim_game.current_player
        opponent = WHITE if current_player == BLACK else BLACK
        
        # Alternate between players making random moves until game ends
        while True:
            # Check if current player has moves
            valid_moves = sim_game.get_valid_moves(current_player)
            if not valid_moves:
                # Check if opponent also has no moves
                opponent_moves = sim_game.get_valid_moves(opponent)
                if not opponent_moves:
                    break  # Game over
                
                # Switch player
                current_player, opponent = opponent, current_player
                continue
            
            # Make a random move
            move = random.choice(valid_moves)
            sim_game.make_move(move[0], move[1], current_player)
            
            # Switch players
            current_player, opponent = opponent, current_player
        
        # Return the winner from our perspective
        winner = sim_game.get_winner()
        return winner