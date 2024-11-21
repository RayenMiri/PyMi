import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import chess
import chess.pgn
import re
import os
from sklearn.model_selection import train_test_split

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Dataset Class (unchanged)
class ChessMoveDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.positions = []
        self.best_moves = []
        self.move_vocab = {}

        for idx, row in self.data.iterrows():
            try:
                moves = self.clean_moves(row['AN'])
                board = chess.Board()
                valid_game = True

                # Validate the sequence of moves
                for move_san in moves:
                    try:
                        board.push_san(move_san)
                    except ValueError as e:
                        print(f"Invalid move '{move_san}' in row {idx}: {e}")
                        valid_game = False
                        break

                if not valid_game:
                    continue  # Skip this game

                # Process valid moves
                board = chess.Board()  # Reset board
                for i in range(len(moves) - 1):
                    try:
                        board.push_san(moves[i])
                        position = self.board_to_array(board)
                        next_move = moves[i + 1]

                        if next_move not in self.move_vocab:
                            self.move_vocab[next_move] = len(self.move_vocab)

                        self.positions.append(position)
                        self.best_moves.append(self.move_vocab[next_move])
                    except ValueError as e:
                        print(f"Skipping invalid move '{moves[i]}' at index {i}: {e}")
                        break
            except Exception as e:
                print(f"Skipping row {idx} due to error: {e}")

        self.positions = np.array(self.positions)
        self.best_moves = np.array(self.best_moves)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = torch.tensor(self.positions[idx], dtype=torch.float32)
        best_move = torch.tensor(self.best_moves[idx], dtype=torch.long)
        return position, best_move

    @staticmethod
    def board_to_array(board):
        board_state = np.zeros((8, 8), dtype=int)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            piece_type = piece.piece_type
            color = piece.color
            value = piece_type * (1 if color == chess.WHITE else -1)
            row, col = divmod(square, 8)
            board_state[row, col] = value
        return board_state
    
    @staticmethod
    def clean_moves(moves):
        moves = re.sub(r'\{[^}]*\}', '', moves)
        moves = re.sub(r'[?!]+', '', moves)
        moves = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', moves)
        moves = re.sub(r'\s+', ' ', moves).strip()
        cleaned_moves = [move for move in moves.split() if not move.endswith('.')]
        return cleaned_moves

# 2. Neural Network Architecture
class ChessMoveCNN(nn.Module):
    def __init__(self, move_vocab_size):
        super(ChessMoveCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, move_vocab_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.mean(x, dim=(2, 3))
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Training the Model
def train_model(model, train_loader, val_loader, move_vocab, epochs=10, lr=0.001, model_path='chess_move_model.pth'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_epoch = 0

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No saved model found. Starting training from scratch.")

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        for position, best_move in train_loader:
            position = position.unsqueeze(1).to(device)
            best_move = best_move.to(device)
            optimizer.zero_grad()

            outputs = model(position)
            loss = criterion(outputs, best_move)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'move_vocab': move_vocab,
            'epoch': epoch + 1
        }, model_path)

def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for position, best_move in val_loader:
            position = position.unsqueeze(1).to(device)
            best_move = best_move.to(device)
            outputs = model(position)
            _, predicted = torch.max(outputs, 1)
            total += best_move.size(0)
            correct += (predicted == best_move).sum().item()
    return 100 * correct / total

# 4. Playing Against AI
def load_model(model_path='chess_move_model.pth'):
    checkpoint = torch.load(model_path, map_location=device)
    move_vocab = checkpoint['move_vocab']
    model = ChessMoveCNN(len(move_vocab)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, move_vocab

def get_ai_move(model, board, move_vocab):
    reverse_vocab = {v: k for k, v in move_vocab.items()}
    board_array = ChessMoveDataset.board_to_array(board)
    board_tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(board_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        sorted_moves = sorted(enumerate(probabilities.cpu().tolist()), key=lambda x: x[1], reverse=True)
        for move_index, _ in sorted_moves:
            best_move_san = reverse_vocab[move_index]
            try:
                board.parse_san(best_move_san)
                return best_move_san
            except:
                pass

        return "0000"

def play_against_ai():
    model, move_vocab = load_model()
    board = chess.Board()
    print("Welcome to Chess AI!")
    print("Enter your moves in SAN format (e.g., e4). Type 'quit' to exit.")

    while not board.is_game_over():
        print("\nCurrent board:")
        print(board)

        if board.turn == chess.WHITE:
            move_san = input("Your move: ")
            if move_san.lower() == 'quit':
                break
            try:
                board.push_san(move_san)
            except:
                print("Invalid move. Try again.")
                continue
        else:
            print("AI is thinking...")
            ai_move = get_ai_move(model, board, move_vocab)
            board.push_san(ai_move)
            print(f"AI move: {ai_move}")

    print("\nGame Over")
    print("Final board:")
    print(board)
    print(f"Result: {board.result()}")

# 5. Main Execution
def main():
    csv_file = './data/10KRowsGames/chess_games_155.csv'
    dataset = ChessMoveDataset(csv_file)
    if len(dataset) == 0:
        print("No valid games found in the dataset. Exiting...")
        return

    train_data, val_data = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_data)
    val_dataset = torch.utils.data.Subset(dataset, val_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = ChessMoveCNN(len(dataset.move_vocab))
    train_model(model, train_loader, val_loader, dataset.move_vocab, epochs=10, lr=0.001)

if __name__ == "__main__":
    main()
    # Uncomment the line below to play against the AI after training
    # play_against_ai()