import math
import random
import shutil
import pandas as pd
import chess
import chess.engine
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer
from model import ChessRewardModel, DummyRewardModel, CLS_TOKEN, PAD_TOKEN

STOCKFISH_PATH = shutil.which("stockfish") or "/usr/local/bin/stockfish"


def normalize_cp(centipawns: int) -> float:
    """Map centipawn score to [-1, 1] using tanh scaling."""
    return math.tanh(centipawns / 400.0)


def material_eval(board: chess.Board) -> float:
    """Material-count evaluation as a fallback for Stockfish."""
    return DummyRewardModel()(board)


class StockfishEvaluator:
    """Wraps a persistent Stockfish engine for batch evaluation."""
    def __init__(self, engine_path: str = STOCKFISH_PATH, depth: int = 15):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.depth = depth

    def __call__(self, board: chess.Board) -> float:
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white()
        if score.is_mate():
            return 1.0 if score.mate() > 0 else -1.0
        return normalize_cp(score.score())

    def close(self):
        self.engine.quit()


def build_tokenizer(csv_path: str, max_language_size: int = 2000) -> Tokenizer:
    """Build a move-level tokenizer from all moves in the dataset."""
    df = pd.read_csv(csv_path)
    all_moves = []
    for move_str in df["moves"].dropna():
        all_moves.extend(move_str.split())
    tokenizer = Tokenizer()
    unique_count = len(set(all_moves))
    lang_size = max(unique_count, max_language_size)
    tokenizer.train_tokenizer(all_moves, max_language_size=lang_size)
    tokenizer.add_special_tokens([CLS_TOKEN, PAD_TOKEN])
    return tokenizer


class ChessPositionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: Tokenizer,
        eval_fn=material_eval,
        max_positions_per_game: int = 10,
        max_games: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.symbol_to_token[CLS_TOKEN]
        self.samples: list[tuple[list[int], float]] = []
        self._generate_samples(csv_path, eval_fn, max_positions_per_game, max_games)

    def _generate_samples(self, csv_path, eval_fn, max_positions_per_game, max_games):
        df = pd.read_csv(csv_path)
        if max_games is not None:
            df = df.head(max_games)

        for _, row in df.iterrows():
            moves_str = row.get("moves", "")
            if not isinstance(moves_str, str) or not moves_str.strip():
                continue
            move_sans = moves_str.split()
            board = chess.Board()

            # Pick random positions to evaluate
            num_positions = min(max_positions_per_game, len(move_sans))
            sample_indices = sorted(random.sample(range(len(move_sans)), num_positions))

            valid_moves = []
            for i, san in enumerate(move_sans):
                try:
                    move = board.parse_san(san)
                    board.push(move)
                    valid_moves.append(san)
                except (chess.InvalidMoveError, chess.AmbiguousMoveError):
                    break

                if i in sample_indices:
                    token_ids = [self.cls_id] + self.tokenizer.encode_moves(valid_moves)
                    score = eval_fn(board)
                    self.samples.append((token_ids, score))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        token_ids, score = self.samples[idx]
        return torch.tensor(token_ids, dtype=torch.long), score


def collate_fn(batch):
    """Pad token sequences and create attention mask."""
    tokens, labels = zip(*batch)
    max_len = max(len(t) for t in tokens)
    pad_id = 0  # will be overwritten; just need consistent shape
    padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
    attention_mask = torch.ones(len(tokens), max_len, dtype=torch.bool)  # True = masked

    for i, t in enumerate(tokens):
        padded[i, :len(t)] = t
        attention_mask[i, :len(t)] = False

    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return padded, attention_mask, labels_tensor


def train(
    csv_path: str = "data/games.csv",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_seq_len: int = 512,
    d_model: int = 256,
    max_games: int | None = None,
    use_stockfish: bool = True,
    stockfish_depth: int = 15,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(csv_path)
    vocab_size = tokenizer.language_size

    eval_fn = None
    evaluator = None
    if use_stockfish:
        print(f"Starting Stockfish engine (depth={stockfish_depth})...")
        evaluator = StockfishEvaluator(depth=stockfish_depth)
        eval_fn = evaluator
    else:
        print("Using material count heuristic (no Stockfish).")
        eval_fn = material_eval

    print("Generating dataset...")
    dataset = ChessPositionDataset(csv_path, tokenizer, eval_fn=eval_fn, max_games=max_games)
    print(f"Dataset size: {len(dataset)} positions")

    if evaluator is not None:
        evaluator.close()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ChessRewardModel(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_tokens, batch_mask, batch_labels in loader:
            batch_tokens = batch_tokens.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_tokens, attention_mask=batch_mask)
            loss = loss_fn(predictions, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "reward_model.pt")
    print("Model saved to reward_model.pt")
    return model, tokenizer


if __name__ == "__main__":
    train(max_games=100)
