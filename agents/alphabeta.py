from agents.agent import Agent
import time
from board import WHITE, BLACK
class AlphaBetaAgent(Agent):
    def __init__(self, depth):
        self.color = None
        self.depth = depth
    
    def get_move(self, game, verbose):
        start_time = time.time()
        _, best_move = self.minimax(game, self.depth, True, -float('inf'), float('inf'))
        end_time = time.time()
        if verbose == True:
            print(f"AI ({'X' if self.color == BLACK else 'O'}) thought for {end_time - start_time:.2f} seconds")
        return best_move
    
    def minimax(self, game, depth, maximizing_player, alpha, beta):
        current_color = self.color if maximizing_player else (WHITE if self.color == BLACK else BLACK)
        
        if depth == 0 or game.is_game_over():
            return game.evaluate_board_advanced(current_color), None
        
        valid_moves = game.get_valid_moves(current_color)
        
        if not valid_moves:
            new_game = game.copy()
            new_game.current_player = WHITE if current_color == BLACK else BLACK
            return self.minimax(new_game, depth - 1, not maximizing_player, alpha, beta)
        
        best_move = None
        if maximizing_player:
            max_eval = -float('inf')
            for move in valid_moves:
                new_game = game.copy()
                new_game.make_move(move[0], move[1], current_color)
                current_eval, _ = self.minimax(new_game, depth - 1, False, alpha, beta)
                
                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = move
                
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_game = game.copy()
                new_game.make_move(move[0], move[1], current_color)
                current_eval, _ = self.minimax(new_game, depth - 1, True, alpha, beta)
                
                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = move
                
                beta = min(beta, current_eval)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
