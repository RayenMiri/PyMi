import chess
import time
from State import State
from Evaluator import Evaluator
from opening_book import OpeningBook

class AlphaBeta:
    def __init__(self, depth):
        self.depth = depth
        self.evaluator = Evaluator()
        self.opening_book = OpeningBook()
        self.transposition_table = {}
        self.history_heuristic = {}
        self.killer_moves = [[None, None] for _ in range(depth + 1)]

    def search(self, state):
        board = state.board
        # Check opening book for quick move
        book_move = self.opening_book.get_move(board)
        if book_move:
            return book_move

        is_maximizing = board.turn == chess.WHITE
        best_move, _ = self.alpha_beta(state, self.depth, float('-inf'), float('inf'), is_maximizing)
        return best_move

    def alpha_beta(self, state, depth, alpha, beta, maximizing_player):
        serialized_state = state.serialize().tobytes()
        # Check the transposition table
        if serialized_state in self.transposition_table:
            tt_entry = self.transposition_table[serialized_state]
            return tt_entry

        board = state.board
        if depth == 0 or board.is_game_over():
            evaluation = self.evaluator.evaluate(board)
            self.transposition_table[serialized_state] = (None, evaluation)
            return None, evaluation

        best_move = None
        moves = list(board.legal_moves)
        
        # Move ordering
        moves = sorted(
            moves,
            key=lambda move: (
                move in self.killer_moves[depth],
                board.is_castling(move),
                board.is_capture(move),
                self.history_heuristic.get((board.piece_at(move.from_square), move.to_square), 0)
            ),
            reverse=True
        )

        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                if board.is_repetition(3) or board.is_fifty_moves():
                    board.pop()
                    continue

                next_state = State(board)
                
                # Late Move Reduction
                reduced_depth = depth - 1
                if not board.is_capture(move) and not board.is_check():
                    reduced_depth = max(0, depth - 2)

                _, eval_score = self.alpha_beta(next_state, reduced_depth, alpha, beta, False)
                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if not board.is_capture(move):
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    # Update history heuristic
                    self.history_heuristic[(board.piece_at(move.from_square), move.to_square)] = depth * depth
                    break

            # Update transposition table
            self.transposition_table[serialized_state] = (best_move, max_eval)
            return best_move, max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                if board.is_repetition(3) or board.is_fifty_moves():
                    board.pop()
                    continue

                next_state = State(board)
                
                # Late Move Reduction
                reduced_depth = depth - 1
                if not board.is_capture(move) and not board.is_check():
                    reduced_depth = max(0, depth - 2)

                _, eval_score = self.alpha_beta(next_state, reduced_depth, alpha, beta, True)
                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if not board.is_capture(move):
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    # Update history heuristic
                    self.history_heuristic[(board.piece_at(move.from_square), move.to_square)] = depth * depth
                    break

            # Update transposition table
            self.transposition_table[serialized_state] = (best_move, min_eval)
            return best_move, min_eval

if __name__ == "__main__":
    # Initialize the chess board with a given position
    board = chess.Board("rnbq1b1r/1p1n1ppp/p2pk3/3N1pP1/4P3/4B3/PPP2P1P/R2QKB1R w KQ - 2 11")
    state = State(board)
    alpha_beta = AlphaBeta(5)

    # Start the timer
    start_time = time.time()
    
    best_move = alpha_beta.search(state)
    
    # End the timer
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Best move found: {board.san(best_move)}")
    print(f"Time taken to find best move: {elapsed_time:.4f} seconds")

    evaluator = Evaluator()
    evaluation = evaluator.evaluate(board)
    print(f"\nPosition evaluation: {evaluation}")

    # Apply the best move to the board
    board.push(best_move)
    print(f"\nPosition after best move:")
    print(board)

    # Evaluate the new position after the best move
    new_evaluation = evaluator.evaluate(board)
    print(f"\nNew position evaluation: {new_evaluation}")
