import sys
import pyautogui as pg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import chess
import chess.engine
from State import State
from AlphaBeta import AlphaBeta

# Constants for the board
BOARD_WIDTH = 680
BOARD_HEIGHT = 650
CELL_WIDTH = int(BOARD_WIDTH / 8)
CELL_HEIGHT = int(BOARD_HEIGHT / 8)
BOARD_TOP_COORD = 140
BOARD_LEFT_COORD = 230

# Players
WHITE = 0
BLACK = 1

# Side to move
side_to_move = 0

# Read argv if available
try:
    if sys.argv[1] == 'black':
        side_to_move = BLACK  # User plays black
    elif sys.argv[1] == 'white':
        side_to_move = WHITE  # Bot plays white
    else:
        print('usage: "chessbot.py white" or "chessbot.py black"')
        sys.exit(0)
except IndexError:
    print('usage: "chessbot.py white" or "chessbot.py black"')
    sys.exit(0)

# Initialize coordinates of squares on the board
square_to_coords = []
x = BOARD_LEFT_COORD
y = BOARD_TOP_COORD
for row in range(8):
    for col in range(8):
        square_to_coords.append((int(x + CELL_WIDTH / 2), int(y + CELL_HEIGHT / 2)))
        x += CELL_WIDTH
    x = BOARD_LEFT_COORD
    y += CELL_HEIGHT

# Map square names (like 'a1', 'h8') to indices in square_to_coords
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

# Capture and display board screenshot
def capture_board_screenshot():
    screenshot = pg.screenshot(region=(BOARD_LEFT_COORD, BOARD_TOP_COORD, BOARD_WIDTH, BOARD_HEIGHT))
    # Convert the screenshot to a NumPy array for OpenCV processing
    screenshot_np = np.array(screenshot)
    # Convert the image from RGB (pyautogui) to BGR (OpenCV)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    # Display the image using matplotlib
    #plt.imshow(cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2RGB))  # Convert to RGB for correct display
    #plt.axis('off')  # Hide axes for clean display
    #plt.show()

# Search position for the best move using AlphaBeta
def search(fen):
    print('Searching best move for this position:')
    print(fen)
    board = chess.Board(fen=fen)
    state = State(board)
    alpha_beta = AlphaBeta(4)
    best_move = alpha_beta.search(state)
    return best_move

# Function to perform a move using pyautogui
def perform_move(from_sq, to_sq):
    pg.moveTo(from_sq)
    pg.click()
    pg.moveTo(to_sq)
    pg.click()
    time.sleep(3)

# Function to parse SAN move to square coordinates
def san_to_coords(board, san_move):
    move = board.parse_san(san_move)
    start_square = move.uci()[:2]
    end_square = move.uci()[2:]
    start_idx = get_square.index(start_square)
    end_idx = get_square.index(end_square)
    return square_to_coords[start_idx], square_to_coords[end_idx]

# Main loop
board = chess.Board()

# If white, bot moves first; if black, user moves first
if side_to_move == WHITE:
    print("Bot is playing first (White).")
    # Get FEN for the initial position
    fen = board.fen()
    # Calculate the best move for the bot
    best_move = search(fen)
    best_move_san = board.san(best_move)
    print(f"Bot's first move: {best_move_san}")
    
    # Get coordinates for the bot's move and perform it
    from_sq, to_sq = san_to_coords(board, best_move_san)
    perform_move(from_sq, to_sq)
    
    # Apply bot's move to board
    board.push(best_move)
    side_to_move = BLACK  # Switch to user's turn

# Main game loop
while True:
    try:
        # Capture and show the board screenshot
        capture_board_screenshot()
        
        if side_to_move == BLACK:
            # Ask the user for a move input in SAN format
            user_move_san = input(f"Your move (in SAN format, e.g., 'e4', 'Nf3'): ").strip()
            
            try:
                user_move = board.parse_san(user_move_san)
            except ValueError:
                print(f"Invalid move: {user_move_san}")
                continue
            
            # Perform the user's move
            #from_sq, to_sq = san_to_coords(board, user_move_san)
            #perform_move(from_sq, to_sq)
            
            # Apply user's move to board
            board.push(user_move)
            side_to_move = WHITE  # Switch to bot's turn
        
        if side_to_move == WHITE:
            # Get FEN after the user's move
            fen = board.fen()
            
            # Calculate the best move for the bot
            best_move = search(fen)
            best_move_san = board.san(best_move)
            print(f"Best move for bot: {best_move_san}")
            
            # Get coordinates for the bot's move and perform it
            from_sq, to_sq = san_to_coords(board, best_move_san)
            perform_move(from_sq, to_sq)
            
            # Apply bot's move to board
            board.push(best_move)
            side_to_move = BLACK  # Switch to user's turn

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(0)
