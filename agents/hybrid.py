import numpy as np
from agents.agent import Agent
from agents.alphabeta import AlphaBetaAgent
from board import EMPTY
class HybridAgent(Agent):
    def __init__(self, early_depth=3, late_depth=5, threshold=12):
        self.color = None
        self.early_depth = early_depth
        self.late_depth = late_depth
        self.threshold = threshold  # When to switch from early to late game
    
    def get_move(self, game, verbose):
        empty_count = np.sum(game.board == EMPTY)
        
        if empty_count > self.threshold:
            # Early game - use simpler evaluation
            return AlphaBetaAgent(self.early_depth).get_move(game, verbose)
        else:
            # Late game - use deeper search
            return AlphaBetaAgent(self.late_depth).get_move(game, verbose)
