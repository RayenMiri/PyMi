import numpy as np
import chess 

class State:
    def __init__(self, board=None):
        self.board = board if board else chess.Board()

   
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