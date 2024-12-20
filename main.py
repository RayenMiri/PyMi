import chess
import chess.pgn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ============================
# Neural Network Definition
# ============================
class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        # Input: 64 squares, 12 pieces (one-hot encoding per piece), total 768 features
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Output: single evaluation score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output: value between -1 (black win) and 1 (white win)
        return x

# ============================
# Helper Functions
# ============================
def board_to_tensor(board):
    """Converts a chess board to a tensor representation."""
    pieces = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            tensor[pieces[piece.symbol()], row, col] = 1
    return torch.tensor(tensor.flatten(), dtype=torch.float32)

def evaluate_board(nn_model, board):
    """Evaluates the board position using the neural network."""
    tensor_input = board_to_tensor(board)
    with torch.no_grad():
        return nn_model(tensor_input).item()

def legal_moves_with_evaluation(board, nn_model):
    """Returns a list of legal moves with their evaluations."""
    moves = list(board.legal_moves)
    evaluations = []
    for move in moves:
        board.push(move)
        evaluations.append(evaluate_board(nn_model, board))
        board.pop()
    return list(zip(moves, evaluations))

def select_best_move(board, nn_model):
    """Selects the best move using the neural network."""
    moves_evaluated = legal_moves_with_evaluation(board, nn_model)
    if board.turn:  # White's turn, maximize
        best_move = max(moves_evaluated, key=lambda x: x[1])[0]
    else:  # Black's turn, minimize
        best_move = min(moves_evaluated, key=lambda x: x[1])[0]
    return best_move

# ============================
# Model Save and Load
# ============================
MODEL_SAVE_PATH = "chess_nn_model.pth"

def save_model(nn_model, optimizer, epoch, path=MODEL_SAVE_PATH):
    """Saves the model and optimizer state to a file."""
    torch.save({
        'model_state_dict': nn_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Model saved to {path}")

def load_model(nn_model, optimizer, path=MODEL_SAVE_PATH):
    """Loads the model and optimizer state from a file."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        nn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Model loaded from {path}, starting from epoch {start_epoch + 1}")
        return start_epoch
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0

# ============================
# Game Logic and Training
# ============================
def play_game(nn_model, max_moves=100):
    """Simulates a single game between two NN players."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    while not board.is_game_over() and len(list(board.move_stack)) < max_moves:
        move = select_best_move(board, nn_model)
        board.push(move)
        print(board)
        node = node.add_variation(move)
    result = board.result()
    print(f"Game over! Result: {result}")
    print(board)
    return result

def train_nn(nn_model, optimizer, epochs=50, batch_size=32):
    """Trains the neural network using self-play games."""
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        board = chess.Board()
        positions = []
        evaluations = []

        # Self-play to generate training data
        for _ in range(batch_size):
            while not board.is_game_over():
                move = select_best_move(board, nn_model)
                board.push(move)
                positions.append(board_to_tensor(board))
                evaluations.append(random.uniform(-1, 1))  # Random target for now
                if board.is_game_over():
                    break
            board.reset()

        # Prepare dataset
        x_train = torch.stack(positions)
        y_train = torch.tensor(evaluations, dtype=torch.float32)

        # Training step
        optimizer.zero_grad()
        predictions = nn_model(x_train).squeeze()
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

        # Save model after each epoch
        save_model(nn_model, optimizer, epoch)

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # Create and initialize the neural network
    nn_model = ChessNN()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

    # Load model if available
    start_epoch = load_model(nn_model, optimizer,path='chess_nn_model.pth')

    # Train the NN
    print("Training the neural network...")
    train_nn(nn_model, optimizer, epochs=50)

    # Play a game
    print("Playing a game between two instances of the NN...")
    play_game(nn_model)
