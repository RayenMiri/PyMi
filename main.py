

# Piece-square tables and PIECE_VALUES remain unchanged

class State:
    def __init__(self, board=None):
        self.board = board if board else chess.Board()

    @lru_cache(maxsize=1000)
    def serialize(self):
        bstate = np.zeros(64, dtype=np.uint8)
        piece_map = {
            "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
            "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14
        }

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                bstate[i] = piece_map[piece.symbol()]

        if self.board.has_queenside_castling_rights(chess.WHITE):
            bstate[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            bstate[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            bstate[56] = 15
        if self.board.has_kingside_castling_rights(chess.BLACK):
            bstate[63] = 15

        if self.board.ep_square is not None:
            bstate[self.board.ep_square] = 8

        bstate = bstate.reshape(8, 8)
        state = np.zeros((5, 8, 8), dtype=np.uint8)
        state[0] = (bstate >> 3) & 1
        state[1] = (bstate >> 2) & 1
        state[2] = (bstate >> 1) & 1
        state[3] = bstate & 1
        state[4] = np.full((8, 8), int(self.board.turn))
        return state

class PieceEvaluator:
    @staticmethod
    def get_piece_value(piece_type):
        return PIECE_VALUES.get(piece_type, 0)

    @staticmethod
    def get_position_score(piece_type, square, endgame):
        if piece_type == chess.PAWN:
            return PAWN_TABLE[square]
        elif piece_type == chess.KNIGHT:
            return KNIGHT_TABLE[square]
        elif piece_type == chess.BISHOP:
            return BISHOP_TABLE[square]
        elif piece_type == chess.ROOK:
            return ROOK_TABLE[square]
        elif piece_type == chess.QUEEN:
            return QUEEN_TABLE[square]
        elif piece_type == chess.KING:
            return KING_END_GAME_TABLE[square] if endgame else KING_MIDDLE_GAME_TABLE[square]
        return 0

class MinimaxAlgorithm:
    def __init__(self, state, depth):
        self.state = state
        self.depth = depth

    def minimax(self, depth, maximizing_player, alpha=float('-inf'), beta=float('inf'), evaluate_fn=None):
        if depth == 0 or self.state.board.is_game_over():
            return evaluate_fn(self.state.board) if evaluate_fn else self.evaluate_position(), None

        best_move = None
        legal_moves = list(self.state.board.legal_moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                self.state.board.push(move)
                eval_score, _ = self.minimax(depth - 1, False, alpha, beta, evaluate_fn)
                self.state.board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                self.state.board.push(move)
                eval_score, _ = self.minimax(depth - 1, True, alpha, beta, evaluate_fn)
                self.state.board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate_position(self):
        if self.state.board.is_checkmate():
            return -10000 if self.state.board.turn == chess.WHITE else 10000
        if self.state.board.is_stalemate() or self.state.board.is_insufficient_material():
            return 0

        score = 0
        endgame = self.is_endgame()

        for square in chess.SQUARES:
            piece = self.state.board.piece_at(square)
            if piece is not None:
                piece_value = PieceEvaluator.get_piece_value(piece.piece_type)
                pos_score = PieceEvaluator.get_position_score(piece.piece_type, square, endgame)
                if piece.color == chess.WHITE:
                    score += piece_value + pos_score
                else:
                    score -= piece_value + pos_score

        return score

    def is_endgame(self):
        white_queens = len(self.state.board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = len(self.state.board.pieces(chess.QUEEN, chess.BLACK))
        white_pieces = len(self.state.board.pieces(chess.KNIGHT, chess.WHITE)) + \
                       len(self.state.board.pieces(chess.BISHOP, chess.WHITE)) + \
                       len(self.state.board.pieces(chess.ROOK, chess.WHITE))
        black_pieces = len(self.state.board.pieces(chess.KNIGHT, chess.BLACK)) + \
                       len(self.state.board.pieces(chess.BISHOP, chess.BLACK)) + \
                       len(self.state.board.pieces(chess.ROOK, chess.BLACK))

        return (white_queens == 0 and black_queens == 0) or \
               (white_queens == 1 and white_pieces <= 1 and black_queens == 0) or \
               (black_queens == 1 and black_pieces <= 1 and white_queens == 0)

class ImprovedChessBot:
    def __init__(self, depth=1):  # Reduced depth for training
        self.depth = depth
        self.model = SGDRegressor(max_iter=1000, tol=1e-3, warm_start=True)
        self.feature_weights = np.ones(7)  # Initial weights for piece values and position

    def extract_features(self, board):
        state = State(board)
        serialized_state = state.serialize()
        features = np.zeros(7)
        
        # Extract piece counts using vectorized operations
        piece_counts = np.bincount(serialized_state.flatten(), minlength=16)
        features[:6] = piece_counts[1:7] - piece_counts[9:15]
        
        # Calculate position score
        features[6] = self.calculate_position_score(board)
        
        return features

    def calculate_position_score(self, board):
        score = 0
        endgame = self.is_endgame(board)
        for square, piece in board.piece_map().items():
            if piece.color == chess.WHITE:
                square_index = square
            else:
                square_index = chess.square_mirror(square)
            
            score += PieceEvaluator.get_position_score(piece.piece_type, square_index, endgame)
            
            if piece.color == chess.BLACK:
                score = -score
        
        return score

    def is_endgame(self, board):
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        total_pieces = len(list(board.pieces(chess.KNIGHT, chess.WHITE))) + len(list(board.pieces(chess.BISHOP, chess.WHITE))) + \
                       len(list(board.pieces(chess.ROOK, chess.WHITE))) + \
                       len(list(board.pieces(chess.KNIGHT, chess.BLACK))) + len(list(board.pieces(chess.BISHOP, chess.BLACK))) + \
                       len(list(board.pieces(chess.ROOK, chess.BLACK)))
        return queens == 0 or (queens == 2 and total_pieces <= 3)

    def evaluate_position(self, board):
        features = self.extract_features(board)
        return np.dot(features, self.feature_weights)

    def process_puzzle_batch(self, puzzles):
        features_batch = []
        targets_batch = []
        correct_moves = 0

        for puzzle in puzzles:
            board = chess.Board(puzzle['fen'])
            correct_move = chess.Move.from_uci(puzzle['moves'][0])
            
            state = State(board)
            minimax = MinimaxAlgorithm(state, self.depth)
            _, bot_move = minimax.minimax(self.depth, state.board.turn, evaluate_fn=self.evaluate_position)
            
            features = self.extract_features(board)
            target = 1 if bot_move == correct_move else -1
            
            features_batch.append(features)
            targets_batch.append(target)
            
            if bot_move == correct_move:
                correct_moves += 1

        return np.array(features_batch), np.array(targets_batch), correct_moves

    def train(self, puzzle_file, epochs=5, batch_size=1000):
        puzzles = self.load_puzzles(puzzle_file)
        total_puzzles = len(puzzles)
        sovled_puzzles = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            np.random.shuffle(puzzles)
            
            total_correct = 0
            for i in range(0, total_puzzles, batch_size):
                batch = puzzles[i:i+batch_size]
                features_batch, targets_batch, correct_moves = self.process_puzzle_batch(batch)
                
                self.model.partial_fit(features_batch, targets_batch)
                self.feature_weights = self.model.coef_
                
                total_correct += correct_moves
                sovled_puzzles+= len(batch)
                print(f"Solved {sovled_puzzles}/{total_puzzles} puzzles.")

                if (i + batch_size) % 10000 == 0 or (i + batch_size) >= total_puzzles:
                    accuracy = total_correct / (i + batch_size)
                    print(f"Processed {i + batch_size}/{total_puzzles} puzzles. Current accuracy: {accuracy:.2%}")
            
            epoch_accuracy = total_correct / total_puzzles
            print(f"Epoch {epoch + 1} completed. Accuracy: {epoch_accuracy:.2%}")
            print(f"Updated weights: {self.feature_weights}")
            print()

    def get_best_move(self, board):
        state = State(board)
        minimax = MinimaxAlgorithm(state, self.depth)
        _, best_move = minimax.minimax(self.depth, state.board.turn, evaluate_fn=self.evaluate_position)
        return best_move

    @staticmethod
    def load_puzzles(file_path):
        df = pd.read_csv(file_path)
        return [{'fen': row['FEN'], 'moves': row['Moves'].split()} for _, row in df.iterrows() if type(row['Moves']) != float]
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def play_against_bot(bot):
    board = chess.Board()
    
    while not board.is_game_over():
        print(board)
        
        if board.turn == chess.WHITE:
            move = input("Enter your move (in UCI format, e.g., 'e2e4'): ")
            try:
                board.push_uci(move)
            except ValueError:
                print("Invalid move. Try again.")
                continue
        else:
            bot_move = bot.get_best_move(board)
            print(f"Bot's move: {bot_move}")
            board.push(bot_move)
    
    print("Game over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    # Training the bot
    bot = ImprovedChessBot(depth=1)  
    puzzle_file = "lichess_db_puzzle.csv"  
    bot.train(puzzle_file, epochs=5, batch_size=1000)

    # Saving the trained model
    bot.save_model("improved_chess_bot_model.pkl")
    print("Model saved successfully.")

    # Loading the saved model
    loaded_bot = ImprovedChessBot.load_model("improved_chess_bot_model.pkl")
    loaded_bot.depth = 3  # Increase depth for gameplay
    print("Model loaded successfully.")

    # Play against the bot
    play_against_bot(loaded_bot)