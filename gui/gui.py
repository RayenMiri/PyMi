import pygame
import chess
import chess.engine

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
DIMENSION = 8  # Chessboard is 8x8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

# Load images
def load_images():
    pieces = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(f"gui/pieces/{piece}.png"), (SQ_SIZE, SQ_SIZE))

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess')

# Initialize the clock
clock = pygame.time.Clock()

# Initialize the board
board = chess.Board()

# Load images
load_images()

# Draw the board
def draw_board(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Draw the pieces
def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_key = f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"
            piece_image = IMAGES.get(piece_key)  
            if piece_image:  
                # Adjust the row calculation to flip it for the display
                screen.blit(piece_image, pygame.Rect(chess.square_file(square) * SQ_SIZE, (7 - chess.square_rank(square)) * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            else:
                print(f"Image for {piece_key} not found.")  
                
# Main game loop
def main():
    running = True
    selected_square = None
    player_clicks = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()
                col = location[0] // SQ_SIZE
                row = location[1] // SQ_SIZE
                square = chess.square(col, 7 - row)
                if selected_square is None:
                    selected_square = square
                    player_clicks.append(square)
                else:
                    player_clicks.append(square)
                    move = chess.Move(player_clicks[0], player_clicks[1])
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None
                    player_clicks = []

        draw_board(screen)
        draw_pieces(screen, board)
        pygame.display.flip()
        clock.tick(MAX_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()