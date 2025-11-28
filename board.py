import numpy as np
import copy
# Constants
EMPTY = 0
BLACK = 1
WHITE = 2
BOARD_SIZE = 8

class OthelloGame:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.current_player = BLACK
        
    def print_board(self):
        print("   ", end="")
        for i in range(BOARD_SIZE):
            print(f" {i}", end="")
        print("\n  +" + "--" * BOARD_SIZE + "+")
        
        for i in range(BOARD_SIZE):
            print(f"{i} |", end="")
            for j in range(BOARD_SIZE):
                if self.board[i][j] == EMPTY:
                    print(" .", end="")
                elif self.board[i][j] == BLACK:
                    print(" X", end="")
                else:
                    print(" O", end="")
            print(" |")
        
        print("  +" + "--" * BOARD_SIZE + "+")
        print(f"Current player: {'X' if self.current_player == BLACK else 'O'}")
    
    def is_valid_move(self, row, col, player=None):
        if player is None:
            player = self.current_player
        
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE or self.board[row][col] != EMPTY:
            return False
        
        opponent = WHITE if player == BLACK else BLACK
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent:
                r += dr
                c += dc
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent:
                    r += dr
                    c += dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                    return True
        return False
    
    def get_valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        
        valid_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
    
    def make_move(self, row, col, player=None):
        if player is None:
            player = self.current_player
        
        if not self.is_valid_move(row, col, player):
            return False
        
        self.board[row][col] = player
        opponent = WHITE if player == BLACK else BLACK
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == player:
                    for flip_r, flip_c in to_flip:
                        self.board[flip_r][flip_c] = player
                    break
        
        self.current_player = opponent
        return True
    
    def is_game_over(self, player=None):
        if player is None:
            player = self.current_player
        return len(self.get_valid_moves(player)) == 0
    
    def get_winner(self):
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        
        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return EMPTY  # Draw
    
    def get_score(self, player):
        return np.sum(self.board == player)
    
    def evaluate_board_advanced(self, player):
        # More advanced evaluation function with positional weights
        weights = [
            [120, -20, 20, 5, 5, 20, -20, 120],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [5, -5, 3, 3, 3, 3, -5, 5],
            [20, -5, 15, 3, 3, 15, -5, 20],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [120, -20, 20, 5, 5, 20, -20, 120]
        ]
        
        opponent = WHITE if player == BLACK else BLACK
        score = 0
        
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == player:
                    score += weights[i][j]
                elif self.board[i][j] == opponent:
                    score -= weights[i][j]
        
        # Add mobility factor
        player_moves = len(self.get_valid_moves(player))
        opponent_moves = len(self.get_valid_moves(opponent))
        score += (player_moves - opponent_moves) * 5
        
        return score
    
    def copy(self):
        new_game = OthelloGame()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game

def play_game(black_player, white_player, verbose):
    black_player.color = BLACK
    white_player.color = WHITE

    game = OthelloGame()
    
    players = {
        BLACK: black_player,
        WHITE: white_player
    }
    
    while True:
        if verbose == True:
            game.print_board()
        
        current_player = players[game.current_player]
        
        # Check if current player has valid moves
        if game.is_game_over(game.current_player):
            if verbose == True:
                print(f"Player {'X' if game.current_player == BLACK else 'O'} has no valid moves.")
            
            # Switch player and check if they also have no moves
            game.current_player = WHITE if game.current_player == BLACK else BLACK
            if game.is_game_over(game.current_player):
                break  # Both players have no moves - game over
            continue
        
        move = current_player.get_move(game, verbose)
        if move is None:
            # This shouldn't happen since we checked is_game_over already
            if verbose == True:
                print("Unexpected pass - no valid moves but we thought there were some")
            game.current_player = WHITE if game.current_player == BLACK else BLACK
            continue
        
        row, col = move
        game.make_move(row, col)

    # Game over
    if verbose == True:
        game.print_board()
    winner = game.get_winner()
    black_count = game.get_score(BLACK)
    white_count = game.get_score(WHITE)
    
    if verbose == True:
        print("\nGame over!")
        print(f"Final score - X: {black_count}, O: {white_count}")
        if winner == BLACK:
            print("X wins!")
        elif winner == WHITE:
            print("O wins!")
        else:
            print("It's a draw!")

    return winner