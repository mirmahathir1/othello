from agents.agent import Agent
class GreedyAgent(Agent):
    def get_move(self, game, verbose):
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        max_flips = -1
        best_move = None
        
        for move in valid_moves:
            temp_game = game.copy()
            temp_game.make_move(move[0], move[1], self.color)
            flipped = temp_game.get_score(self.color) - game.get_score(self.color) - 1
            if flipped > max_flips:
                max_flips = flipped
                best_move = move
        
        return best_move

