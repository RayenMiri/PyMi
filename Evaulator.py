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

    @staticmethod
    def evaluate(board):
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
                table = Evaluator.get_piece_square_table(piece_type, board.is_checkmate(), piece_color)
                table_score = table[square] if piece_color == chess.WHITE else table[chess.square_mirror(square)]
                score += table_score if piece_color == chess.WHITE else -table_score

        # Additional dynamic factors
        score += Evaluator.mobility(board)
        score += Evaluator.king_safety(board)
        score += Evaluator.pawn_structure(board)

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
    def king_safety(board):
        # Simplified king safety evaluation
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        white_king_safety = sum([1 for move in board.attacks(white_king_square) if board.color_at(move) == chess.BLACK])
        black_king_safety = sum([1 for move in board.attacks(black_king_square) if board.color_at(move) == chess.WHITE])
        return black_king_safety - white_king_safety

    @staticmethod
    def pawn_structure(board):
        # Simplified pawn structure evaluation
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        white_doubled_pawns = sum([1 for square in white_pawns if board.piece_at(square + 8) == chess.PAWN])
        black_doubled_pawns = sum([1 for square in black_pawns if board.piece_at(square - 8) == chess.PAWN])
        return black_doubled_pawns - white_doubled_pawns