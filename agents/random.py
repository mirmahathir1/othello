from agents.agent import Agent
import numpy as np

class RandomAgent(Agent):
    def get_move(self, game, verbose):
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        return valid_moves[np.random.choice(len(valid_moves))]

