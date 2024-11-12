import chess
import chess.polyglot

class OpeningBook:
    def __init__(self, book_path="data/gm2001.bin"):
        self.book_path = book_path

    def get_move(self, board):
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entry = reader.weighted_choice(board)
                if entry is not None:
                    return entry.move
        except:
            #pass for errors
            pass
        return None