from agents.agent import Agent
from board import WHITE, BLACK
import numpy as np

class MCTSAgent(Agent):
    def __init__(self, simulations=100):
        self.color = None
        self.simulations = simulations
    
    def get_move(self, game, verbose):
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        move_wins = {move: 0 for move in valid_moves}
        
        for move in valid_moves:
            for _ in range(self.simulations // len(valid_moves)):
                sim_game = game.copy()
                sim_game.make_move(move[0], move[1], self.color)
                winner = self.simulate_random_game(sim_game)
                if winner == self.color:
                    move_wins[move] += 1
        
        return max(move_wins.items(), key=lambda x: x[1])[0]
    
    def simulate_random_game(self, game):
        while not game.is_game_over():
            valid_moves = game.get_valid_moves(game.current_player)
            if valid_moves:
                move = valid_moves[np.random.choice(len(valid_moves))]
                game.make_move(move[0], move[1], game.current_player)
            else:
                game.current_player = WHITE if game.current_player == BLACK else BLACK
        return game.get_winner()

