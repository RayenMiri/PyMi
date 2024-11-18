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
    HANGING_PIECE_PENALTY = 50
    LOOSE_PIECE_PENALTY = 20
    UNDEFENDED_PAWN_PENALTY = 10
    CASTLING_BONUS = 50
    CASTLED_BONUS = 100
    DELAYED_CASTLING_PENALTY = -30  
    KING_EXPOSURE_PENALTY = -20  
    ATTACKER_WEIGHTS = {
        chess.PAWN: 1,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 1.3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING : 1
    }
    PAWN_SHIELD_BONUS = 10  
    PAWN_STORM_PENALTY = -5 

    
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
        score += Evaluator.castling_evaluation(board)
        score += Evaluator.pawn_structure(board)
        score += Evaluator.center_control(board)
        score += Evaluator.development(board)
        score += Evaluator.checkmate_threat(board)
        score += Evaluator.evaluate_hanging_pieces(board)

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
        score = 0
        white_moves = len(list(board.generate_legal_moves()))
        board.turn = not board.turn
        black_moves = len(list(board.generate_legal_moves()))
        board.turn = not board.turn
        score += (white_moves - black_moves) * Evaluator.TEMPO_VALUE
        return score

    @staticmethod
    def castling_evaluation(board):
        score = 0
        is_endgame = Evaluator.is_endgame(board)

        # Evaluate white's castling situation
        if board.has_castling_rights(chess.WHITE):
            score += Evaluator.CASTLING_BONUS
        elif Evaluator.has_castled(board, chess.WHITE):
            score += Evaluator.CASTLED_BONUS
        elif not is_endgame:
            score += Evaluator.DELAYED_CASTLING_PENALTY

        # Evaluate black's castling situation
        if board.has_castling_rights(chess.BLACK):
            score -= Evaluator.CASTLING_BONUS
        elif Evaluator.has_castled(board, chess.BLACK):
            score -= Evaluator.CASTLED_BONUS
        elif not is_endgame:
            score -= Evaluator.DELAYED_CASTLING_PENALTY

        return score

    @staticmethod
    def has_castled(board, color):
        king_file = chess.square_file(board.king(color))
        return king_file in [2, 6]  # King on c-file or g-file indicates castling

    @staticmethod
    def king_safety(board):
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        white_king_safety = Evaluator.evaluate_king_safety(board, white_king_square, chess.WHITE)
        black_king_safety = Evaluator.evaluate_king_safety(board, black_king_square, chess.BLACK)
        
        return white_king_safety - black_king_safety

    @staticmethod
    def evaluate_king_safety(board, king_square, color):
        safety_score = 0
        shield_bonus = 0
        storm_penalty = 0
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type == chess.PAWN:
                if abs(chess.square_rank(square) - king_rank) <= 1 and abs(chess.square_file(square) - king_file) <= 1:
                    shield_bonus += Evaluator.PAWN_SHIELD_BONUS

                # Penalize pawns advancing in front of the king
                if color == chess.WHITE and chess.square_rank(square) > king_rank:
                    storm_penalty += Evaluator.PAWN_STORM_PENALTY
                elif color == chess.BLACK and chess.square_rank(square) < king_rank:
                    storm_penalty += Evaluator.PAWN_STORM_PENALTY

        safety_score += shield_bonus + storm_penalty
        return safety_score

    @staticmethod
    def evaluate_pawn_shield(board, king_square, color):
        score = 0
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        for f in range(max(0, file - 1), min(8, file + 2)):
            if color == chess.WHITE:
                for r in range(rank, min(8, rank + 3)):
                    piece = board.piece_at(chess.square(f, r))
                    if piece == chess.Piece(chess.PAWN, chess.WHITE):
                        score += Evaluator.PAWN_SHIELD_BONUS
                    elif piece == chess.Piece(chess.PAWN, chess.BLACK):
                        score += Evaluator.PAWN_STORM_PENALTY
            else:
                for r in range(rank, max(-1, rank - 3), -1):
                    piece = board.piece_at(chess.square(f, r))
                    if piece == chess.Piece(chess.PAWN, chess.BLACK):
                        score += Evaluator.PAWN_SHIELD_BONUS
                    elif piece == chess.Piece(chess.PAWN, chess.WHITE):
                        score += Evaluator.PAWN_STORM_PENALTY
        
        return score

    @staticmethod
    def evaluate_king_attackers(board, king_square, color):
        score = 0
        opponent_color = not color
        
        attackers = board.attackers(opponent_color, king_square)
        for attacker_square in attackers:
            attacker = board.piece_at(attacker_square)
            score += Evaluator.ATTACKER_WEIGHTS[attacker.piece_type]
        
        return score * 10  # Multiply by 10 to give more weight to attackers

    @staticmethod
    def evaluate_open_files(board, king_square, color):
        score = 0
        file = chess.square_file(king_square)
        
        for f in range(max(0, file - 1), min(8, file + 2)):
            if Evaluator.is_open_file(board, f):
                score += 20  # Penalty for each open file near the king
        
        return score

    @staticmethod
    def evaluate_king_exposure(board, king_square, color):
        score = 0
        is_endgame = Evaluator.is_endgame(board)
        
        if not is_endgame:
            # Penalize exposed king in middle game
            if color == chess.WHITE and chess.square_rank(king_square) > 1:
                score += Evaluator.KING_EXPOSURE_PENALTY
            elif color == chess.BLACK and chess.square_rank(king_square) < 6:
                score += Evaluator.KING_EXPOSURE_PENALTY
        else:
            # Encourage king activity in endgame
            central_distance = abs(3.5 - chess.square_file(king_square)) + abs(3.5 - chess.square_rank(king_square))
            score -= central_distance * 5  # Bonus for centralized king in endgame
        
        return score

    @staticmethod
    def pawn_structure(board):
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if board.is_attacked_by(not piece.color, square):
                    score -= Evaluator.UNDEFENDED_PAWN_PENALTY if piece.color == chess.WHITE else -Evaluator.UNDEFENDED_PAWN_PENALTY
        return score

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
    
    @staticmethod
    def evaluate_hanging_pieces(board):
        score = 0
        is_endgame = Evaluator.is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                defenders = board.attackers(piece.color, square)
                attackers = board.attackers(not piece.color, square)
                
                if attackers:
                    attacker_count = len(attackers)
                    defender_count = len(defenders)
                    
                    # Evaluate the exchange
                    attacker_values = sorted([Evaluator.PIECE_VALUES[board.piece_at(attacker).piece_type] for attacker in attackers])
                    defender_values = sorted([Evaluator.PIECE_VALUES[board.piece_at(defender).piece_type] for defender in defenders] + [Evaluator.PIECE_VALUES[piece.piece_type]])
                    
                    exchange_value = sum(defender_values[:attacker_count]) - sum(attacker_values[:defender_count])
                    
                    if exchange_value < 0:
                        penalty = abs(exchange_value)
                        if piece.color == chess.WHITE:
                            score += penalty //2
                        else: 
                            score -= penalty //2
                
                elif not defenders and piece.piece_type != chess.KING:
                    penalty = Evaluator.LOOSE_PIECE_PENALTY * Evaluator.PIECE_VALUES[piece.piece_type] // 100
                    if piece.color == chess.WHITE:
                        score -= penalty //2
                    else:
                        score += penalty //2
        
        if is_endgame:
            score *= 1.2  # Slightly increase the importance of hanging pieces in endgame
        
        return score 
    
    @staticmethod
    def evaluate_forks(board):
        fork_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:  # Focus on the active player's pieces
                attackers = board.attackers(piece.color, square)
                for target_square in attackers:
                    target_piece = board.piece_at(target_square)
                    if target_piece and target_piece.color != piece.color:
                        # If the piece attacks multiple valuable targets, assign a score
                        fork_score += (
                            Evaluator.PIECE_VALUES.get(target_piece.piece_type, 0) // 2
                        )
        return fork_score

eval = Evaluator()  
print(eval.evaluate(chess.Board("rnbq1b1r/1p1n1ppp/p2p4/3N1kP1/8/4B3/PPP2P1P/R2QKB1R w KQ - 0 12")))

#print(eval.evaluate(chess.Board("rnbq1b1r/1p1n1ppp/p2pk3/5pP1/4PN2/4B3/PPP2P1P/R2QKB1R b KQ - 3 11")))

#print(eval.evaluate(chess.Board("r3qrk1/pppb2P1/2n5/3p4/8/2N1PN2/PPP2PP1/R2QKB1R b KQ - 0 15")))

