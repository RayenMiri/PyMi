# PyMi
PyMi is a chess bot for first versions is gonna be simple based neural network 
This is the Version 0.1

to represent all of board : 
    64 squares => 8 * 8 bits 
    12 types of pieces => 12 states
    to represent 12 states we do log_2(12) => 3.58496250072 = 4
    so we need like 8 * 8 * 4 = 256 bits 
    and a 1 bit for turn representation (1 for while 0 for black or the reverse)
    ==>257 bits 

Key Serialization Steps

bstate Initialization:

The code initializes bstate as a 64-element NumPy array, where each entry corresponds to a square on the board. 
Each square is encoded with an integer representing the piece on that square (or 0 if empty).
The dictionary maps each piece to a unique integer: white pieces use values from 1 to 6, while black pieces use 9 to 14, 
which creates a distinct range for each color.

Castling Rights Encoding:

The code uses special values in bstate to indicate castling rights. If a rook is eligible for castling, 
the value is updated to 7 (white) or 15 (black) on its respective square. This way, castling information is embedded directly 
in the board representation.

En Passant Square:

If there’s an en passant square available, the code marks it with an 8. This way, en passant is tracked without requiring additional information outside the board.
Binary Representation:

state is a 5x8x8 NumPy array where each “layer” in state encodes a different aspect of the board:
Layers 0-3 represent each square in 4-bit binary format. Each bit in state corresponds to one of the binary positions in bstate.
Layer 4 represents whose turn it is (1 for white, 0 for black), with all entries in this layer being the same.
Resulting state Format:

The state variable, which has shape (5, 8, 8), effectively holds the entire game state in a compact, 257-bit equivalent representation:
256 bits represent the board and castling/en passant information.
1 bit in layer 4 indicates whose turn it is.


Here are several methods to optimize the training process for a chess bot:

Batching Puzzle Training
Instead of training on each puzzle individually, process puzzles in batches. This approach can help update the model more efficiently, especially when using a stochastic gradient descent (SGD) model like SGDRegressor. To implement batching, accumulate puzzle features and target labels in a batch, then call partial_fit once per batch, which could save time compared to updating weights puzzle-by-puzzle.

Parallel Processing of Puzzles
Leverage parallel processing by running multiple puzzles through the training process concurrently. Python’s multiprocessing library or libraries like joblib (for now we'll be using multiprocessing library we may use joblib next time) can be helpful here

Using Efficient Vectorized Operations
Minimize any looping within extract_features and calculate_position_score methods. Use NumPy vectorized operations wherever possible to avoid repeated iteration over the board, which could lead to significant speedup.

Early Stopping Based on Accuracy
Set an accuracy threshold for each epoch, and if the model achieves sufficient accuracy before the end of the epoch, stop further training for that epoch. This way, unnecessary computations are avoided if the model is already performing well.

Caching Serialized States
If similar board states or FEN positions are encountered repeatedly, caching the serialized output could save processing time. Use a dictionary to store and retrieve precomputed states rather than recalculating them each time.

Downsampling Puzzles or Epochs
Although five epochs are good for improving model robustness, reducing the number of epochs might be an option if each epoch takes too long. Alternatively, downsample puzzles by randomly selecting a subset for training each epoch.

Utilize a Faster Model Variant
The SGDRegressor can be slow with high-dimensional inputs. For faster convergence, we will consider other fast, linear models like Ridge with a warm start or even try gradient-boosting methods, which might give better performance and faster training but now we'll see the performance of SGDRegressor

Preprocessing and Feature Engineering
Further streamline extract_features to reduce redundancy or consider engineered features that are faster to compute while still encapsulating critical information for decision-making.

Model Compression
After training on all epochs, consider pruning less impactful features or re-weighting using a compressed feature representation if any redundant patterns are identified.
