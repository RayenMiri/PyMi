import chess
from State import State
from Evaulator import Evaluator
from opening_book import OpeningBook

class AlphaBeta:
    def __init__(self, depth):
        self.depth = depth
        self.evaluator = Evaluator()
        self.opening_book = OpeningBook()
        

    def search(self, state):
        board = state.board
        book_move = self.opening_book.get_move(board)
        if book_move:
            return book_move

        is_maximizing = board.turn == chess.WHITE
        best_move, _ = self.alpha_beta(board, self.depth, float('-inf'), float('inf'), is_maximizing)
       
        return best_move

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        

        if depth == 0 or board.is_game_over():
            return None, self.evaluator.evaluate(board)

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            moves = board.legal_moves
            moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
            for move in moves:
                board.push(move)
                _, eval_score = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                  
                    break
            return best_move, max_eval
        else:
            min_eval = float('inf')
            moves = board.legal_moves
            moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
            for move in moves:
                board.push(move)
                _, eval_score = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    
                    break
            return best_move, min_eval

if __name__ == "__main__":
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
    
    state = State(board)
    
    alpha_beta = AlphaBeta(5)
    
    best_move = alpha_beta.search(state)
    print(f"Best move found: {board.san(best_move)}")
    
    evaluator = Evaluator()
    evaluation = evaluator.evaluate(board)
    print(f"\nPosition evaluation: {evaluation}")

    # Make the best move
    board.push(best_move)
    print(f"\nPosition after best move:")
    print(board)
    
    # Evaluate the new position
    new_evaluation = evaluator.evaluate(board)
    print(f"\nNew position evaluation: {new_evaluation}")