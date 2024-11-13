import sys
import cv2
import numpy as np
import pyautogui as pg
import chess
import time
from State import State
from AlphaBeta import AlphaBeta
import os

# Constants for the board
BOARD_WIDTH = 680
BOARD_HEIGHT = 650
CELL_WIDTH = int(BOARD_WIDTH / 8)
CELL_HEIGHT = int(BOARD_HEIGHT / 8)
BOARD_TOP_COORD = 140
BOARD_LEFT_COORD = 250
CONFIDENCE = 0.7  # Lowered confidence to improve detection
DETECTION_NOISE_THRESHOLD = 15  # Increased noise threshold to allow more detections

PIECES_PATH = './new_pieces/'

# players
WHITE = 0
BLACK = 1

# side to move
side_to_move = 0

# read argv if available
try:
    if sys.argv[1] == 'black': side_to_move = BLACK
except:
    print('usage: "chessbot.py white" or "chessbot.py black"')
    sys.exit(0)

# square to coords
square_to_coords = []

# array to convert board square indices to coordinates (black)
get_square = [
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1'
]

# map piece names to FEN chars
piece_names = {
    'black_king': 'k',
    'black_queen': 'q',
    'black_rook': 'r',
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_pawn': 'p',
    'white_knight': 'N',
    'white_pawn': 'P',
    'white_king': 'K',
    'white_queen': 'Q',
    'white_rook': 'R',
    'white_bishop': 'B'
}

# locate piece on image
def locate_piece(screenshot, piece_location):
    for index in range(len(piece_location)):
        piece = piece_location[index]
        cv2.rectangle(
            screenshot,
            (piece.left, piece.top),
            (piece.left + piece.width, piece.top + piece.height),
            (0, 0, 255),
            2
        )
    cv2.imshow('Screenshot', screenshot)
    cv2.waitKey(0)

# get coordinates of chess pieces
def recognize_position():
    piece_locations = {
        'black_king': [], 'black_queen': [], 'black_rook': [], 'black_bishop': [], 'black_knight': [], 'black_pawn': [],
        'white_knight': [], 'white_pawn': [], 'white_king': [], 'white_queen': [], 'white_rook': [], 'white_bishop': []
    }
    
    # Take screenshot of the board region only
    screenshot = cv2.cvtColor(np.array(pg.screenshot(region=(
        BOARD_LEFT_COORD,
        BOARD_TOP_COORD,
        BOARD_WIDTH,
        BOARD_HEIGHT
    ))), cv2.COLOR_RGB2BGR)
    
    for piece in piece_names.keys():
        template = cv2.imread(os.path.join(PIECES_PATH, piece + '.png'), 0)
        if template is None:
            print(f"Failed to load template for {piece}")
            continue
        
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
        
        # For debugging: Show the result matrix
        print(f"Template match result for {piece}: {result.shape}")
        
        locations = np.where(result >= CONFIDENCE)
        print(f"Locations found for {piece}: {list(zip(*locations[::-1]))}")
        
        for pt in zip(*locations[::-1]):
            noise = False
            for position in piece_locations[piece]:
                if abs(position['x'] - pt[0]) < DETECTION_NOISE_THRESHOLD and \
                   abs(position['y'] - pt[1]) < DETECTION_NOISE_THRESHOLD:
                    noise = True
                    break
            if noise:
                continue
            
            # Store locations as dictionaries with x, y coordinates
            piece_locations[piece].append({
                'x': pt[0] + BOARD_LEFT_COORD,  # Add board offset
                'y': pt[1] + BOARD_TOP_COORD,   # Add board offset
                'width': template.shape[1],
                'height': template.shape[0]
            })
            print('detecting:', piece, pt)
            
            # Draw rectangle around the detected piece (for debugging)
            cv2.rectangle(screenshot, pt,
                          (pt[0] + template.shape[1], pt[1] + template.shape[0]),
                          (0, 0, 255), 2)
    
    # Show the screenshot with detected pieces (for debugging)
    cv2.imshow('Detected Pieces', screenshot)
    cv2.waitKey(1)  # Changed to 1ms wait to not block execution
    
    return screenshot, piece_locations

def locations_to_fen(piece_locations):
    fen = ''
    x = BOARD_LEFT_COORD
    y = BOARD_TOP_COORD
    for row in range(8):
        empty = 0
        for col in range(8):
            square = row * 8 + col
            is_piece = False
            for piece_type in piece_locations.keys():
                for piece in piece_locations[piece_type]:
                    if abs(piece['x'] - x) < DETECTION_NOISE_THRESHOLD and \
                       abs(piece['y'] - y) < DETECTION_NOISE_THRESHOLD:
                        if empty:
                            fen += str(empty)
                            empty = 0
                        fen += piece_names[piece_type]
                        is_piece = True
            if not is_piece:
                empty += 1
            x += CELL_WIDTH
        if empty: fen += str(empty)
        if row < 7: fen += '/'
        x = BOARD_LEFT_COORD
        y += CELL_HEIGHT
    fen += ' ' + ('b' if side_to_move else 'w')
    fen += ' KQkq - 0 1'
    return fen

# search position for a best move using AlphaBeta
def search(fen):
    print('Searching best move for this position:')
    print(fen)
    board = chess.Board(fen=fen)
    state = State(board)
    alpha_beta = AlphaBeta(3)
    best_move = alpha_beta.search(state)
    return best_move

################################
#
#        Init coordinates
#
################################
x = BOARD_LEFT_COORD
y = BOARD_TOP_COORD
for row in range(8):
    for col in range(8):
        square = row * 8 + col
        square_to_coords.append((int(x + CELL_WIDTH / 2), int(y + CELL_HEIGHT / 2)))
        x += CELL_WIDTH
    x = BOARD_LEFT_COORD
    y += CELL_HEIGHT

################################
#
#          Main driver
#
################################
while True:
    try:
        screenshot, piece_locations = recognize_position()
        fen = locations_to_fen(piece_locations)
        best_move = search(fen)
        print(fen)
        print('Best move:', best_move)
        from_sq = square_to_coords[get_square.index(best_move.uci()[:2])]
        to_sq = square_to_coords[get_square.index(best_move.uci()[2:])]
        pg.moveTo(from_sq)
        pg.click()
        pg.moveTo(to_sq)
        pg.click()
        time.sleep(3)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(0)
