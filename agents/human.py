from agents.agent import Agent
from board import BLACK
class HumanAgent(Agent):
    def __init__(self):
        self.color = None
    
    def get_move(self, game, verbose):
        valid_moves = game.get_valid_moves(self.color)
        if not valid_moves:
            return None  # No valid moves, will pass
        
        print(f"\nPlayer {'X' if self.color == BLACK else 'O'}'s turn (Human)")
        print("Valid moves:", valid_moves)
        
        while True:
            move = input("Enter your move (row col): ").strip().split()
            try:
                if len(move) == 2:
                    row, col = map(int, move)
                    if (row, col) in valid_moves:
                        return (row, col)
                print("Invalid move. Try again.")
            except:
                print("Invalid input. Please enter two numbers separated by space.")

