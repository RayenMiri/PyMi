import chess

class Evaluator:
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    BISHOP_TABLE = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    ROOK_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ]
    QUEEN_TABLE = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]
    KING_MIDDLE_GAME_TABLE = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
    KING_END_GAME_TABLE = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
    EXTENDED_CENTER_SQUARES = [
        chess.C3, chess.D3, chess.E3, chess.F3,
        chess.C4, chess.D4, chess.E4, chess.F4,
        chess.C5, chess.D5, chess.E5, chess.F5,
        chess.C6, chess.D6, chess.E6, chess.F6
    ]
    DEVELOPMENT_BONUS = 10
    EARLY_QUEEN_PENALTY = -20
    BLOCKED_CENTER_PAWN_PENALTY = -15
    TEMPO_VALUE = 33  
    
    @staticmethod
    def evaluate(board):
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                piece_color = piece.color

                # Calculating material score
                piece_value = Evaluator.PIECE_VALUES[piece_type]
                score += piece_value if piece_color == chess.WHITE else -piece_value

                # Calculating positional score based on piece-square tables
                table = Evaluator.get_piece_square_table(piece_type, Evaluator.is_endgame(board), piece_color)
                table_score = table[square] if piece_color == chess.WHITE else table[chess.square_mirror(square)]
                score += table_score if piece_color == chess.WHITE else -table_score

        score += Evaluator.mobility(board)
        score += Evaluator.king_safety(board)
        score += Evaluator.pawn_structure(board)
        score += Evaluator.center_control(board)
        score += Evaluator.development(board)
        score += Evaluator.checkmate_threat(board)

        return score
    
    @staticmethod
    def get_piece_square_table(piece_type, endgame, color):
        if piece_type == chess.PAWN:
            return Evaluator.PAWN_TABLE
        elif piece_type == chess.KNIGHT:
            return Evaluator.KNIGHT_TABLE
        elif piece_type == chess.BISHOP:
            return Evaluator.BISHOP_TABLE
        elif piece_type == chess.ROOK:
            return Evaluator.ROOK_TABLE
        elif piece_type == chess.QUEEN:
            return Evaluator.QUEEN_TABLE
        elif piece_type == chess.KING:
            return Evaluator.KING_END_GAME_TABLE if endgame else Evaluator.KING_MIDDLE_GAME_TABLE

    @staticmethod
    def mobility(board):
        white_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        black_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        return white_moves - black_moves

    @staticmethod
    def evaluate_king_safety(board, king_square, color):
        safety_score = 0
        
        # Pawn shield
        pawn_shield_score = Evaluator.evaluate_pawn_shield(board, king_square, color)
        safety_score += pawn_shield_score
        
        # King attackers
        attacker_score = Evaluator.evaluate_king_attackers(board, king_square, color)
        safety_score -= attacker_score
        
        # Open files near king
        open_file_score = Evaluator.evaluate_open_files(board, king_square, color)
        safety_score -= open_file_score
        
        return safety_score

    @staticmethod
    def king_safety(board):
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        white_king_safety = Evaluator.evaluate_king_safety(board, white_king_square, chess.WHITE)
        black_king_safety = Evaluator.evaluate_king_safety(board, black_king_square, chess.BLACK)
        
        return white_king_safety - black_king_safety
    
    @staticmethod
    def pawn_structure(board):
        # Simplified pawn structure evaluation
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        white_doubled_pawns = sum([1 for square in white_pawns if board.piece_at(square + 8) == chess.PAWN])
        black_doubled_pawns = sum([1 for square in black_pawns if board.piece_at(square - 8) == chess.PAWN])
        return black_doubled_pawns - white_doubled_pawns

    @staticmethod
    def center_control(board):
        score = 0
        
        # Evaluate control of center squares
        for square in Evaluator.CENTER_SQUARES:
            score += 10 if board.is_attacked_by(chess.WHITE, square) else 0
            score -= 10 if board.is_attacked_by(chess.BLACK, square) else 0

        # Evaluate control of extended center squares
        for square in Evaluator.EXTENDED_CENTER_SQUARES:
            score += 5 if board.is_attacked_by(chess.WHITE, square) else 0
            score -= 5 if board.is_attacked_by(chess.BLACK, square) else 0

        # Bonus for pieces in the center
        for square in Evaluator.CENTER_SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 15
                else:
                    score -= 15

        # Bonus for pawns in the extended center
        for square in Evaluator.EXTENDED_CENTER_SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    score += 10
                else:
                    score -= 10

        return score
    
    @staticmethod
    def development(board):
        score = 0
        
        # Count developed minor pieces and evaluate tempo
        white_developed = 0
        black_developed = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, chess.WHITE):
                if chess.square_rank(square) > 1:  # If the piece has moved
                    white_developed += 1
            for square in board.pieces(piece_type, chess.BLACK):
                if chess.square_rank(square) < 6:  # If the piece has moved
                    black_developed += 1
        
        # Add development bonus
        score += (white_developed - black_developed) * Evaluator.DEVELOPMENT_BONUS
        
        # Evaluate tempo (assuming white starts)
        tempo_difference = board.fullmove_number * 2 - (1 if board.turn == chess.WHITE else 0)
        tempo_score = (white_developed - black_developed - tempo_difference // 2) * Evaluator.TEMPO_VALUE
        score += tempo_score

        # Penalize early queen development
        if board.piece_at(chess.D1) is None and board.piece_type_at(chess.D8) == chess.QUEEN:
            score += Evaluator.EARLY_QUEEN_PENALTY
        if board.piece_at(chess.D8) is None and board.piece_type_at(chess.D1) == chess.QUEEN:
            score -= Evaluator.EARLY_QUEEN_PENALTY

        # Penalize blocked center pawns
        for square in [chess.D2, chess.E2]:
            if board.piece_type_at(square) == chess.PAWN and board.piece_at(square + 8):
                score += Evaluator.BLOCKED_CENTER_PAWN_PENALTY
        for square in [chess.D7, chess.E7]:
            if board.piece_type_at(square) == chess.PAWN and board.piece_at(square - 8):
                score -= Evaluator.BLOCKED_CENTER_PAWN_PENALTY

        return score
    
    @staticmethod
    def is_endgame(board):
        # Consider it endgame if both sides have no queens or if every side which has a queen has additionally no other pieces or only 
        # one minor piece
        white_queen = len(board.pieces(chess.QUEEN, chess.WHITE))
        black_queen = len(board.pieces(chess.QUEEN, chess.BLACK))
        white_pieces = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.WHITE))
        black_pieces = len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + len(board.pieces(chess.ROOK, chess.BLACK))
        
        return (white_queen == 0 and black_queen == 0) or \
               (white_queen == 0 and black_pieces <= 1) or \
               (black_queen == 0 and white_pieces <= 1)

    @staticmethod
    def checkmate_threat(board):
        score = 0
        #check if the current player has checkmate pattern
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                score += 500
            board.pop()
        
        #check if the opponent has checkmat pattern 
        board.turn = not board.turn
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                score -= 500
            board.pop()
        board.turn = not board.turn
        
        return score 
                
    @staticmethod
    def evaluate_pawn_shield(board, king_square, color):
        score = 0
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        for f in range(max(0, file - 1), min(8, file + 2)):
            if color == chess.WHITE:
                for r in range(rank, min(8, rank + 3)):
                    if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.WHITE):
                        score += 10  
            else:
                for r in range(rank, max(-1, rank - 3), -1):
                    if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.BLACK):
                        score += 10  
        
        return score
    
    @staticmethod
    def evaluate_king_attackers(board, king_square, color):
        score = 0
        opponent_color = not color
        
        for square in board.attacks(king_square):
            attacker = board.piece_at(square)
            if attacker and attacker.color == opponent_color:
                score += Evaluator.PIECE_VALUES[attacker.piece_type]
        
        return score
    
    @staticmethod
    def evaluate_open_files(board, king_square, color):
        score = 0
        file = chess.square_file(king_square)
        
        for f in range(max(0, file - 1), min(8, file + 2)):
            if Evaluator.is_open_file(board, f):
                score += 20  # Penalty for each open file near the king
        
        return score

    @staticmethod
    def is_open_file(board, file):
        for rank in range(8):
            square = chess.square(file, rank)
            if board.piece_type_at(square) == chess.PAWN:
                return False
        return True
board = chess.Board()  
board.set_fen("rnbqkb1r/pppppppp/8/8/3Pn3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")  
eval = Evaluator()
print(eval.evaluate(board))

board.set_fen("rnbqkb1r/ppppppp1/5n2/7p/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3")  
eval = Evaluator()
print(eval.evaluate(board))

