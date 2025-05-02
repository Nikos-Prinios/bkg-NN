from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

###############################################################################
#  CONFIG                                                                     #
###############################################################################

DB_PATH     = "matches.db"  # SQLite file from parse_gnubg.py
BATCH_SIZE  = 2048
EPOCHS      = 20
LR          = 3e-4
VAL_SPLIT   = 0.1
SEED        = 42
NUM_WORKERS = 4  # Set 0 for debug, up to CPU cores
NUM_CHECKERS= 15 # Standard Backgammon
INPUT_DIM   = 54 # MUST MATCH BKG_30_04.py and model architecture

###############################################################################
#  FEATURE ENCODING (Mirrors BKG_30_04.py state_to_tensor)                    #
###############################################################################

def calculate_pip_simple(board: List[int], white_bar: int, black_bar: int, player: str) -> int:
    """Basic Python pip calculation (like in BackgammonGame)."""
    pip = 0
    p_sign = 1 if player == 'w' else -1
    bar_cnt = white_bar if player == 'w' else black_bar
    for i, count in enumerate(board): # board is 0-23 index
        pos = i + 1 # Point number 1-24
        if count * p_sign > 0:
            dist = (25 - pos) if player == 'w' else pos
            pip += dist * abs(count)
    pip += bar_cnt * 25
    return pip


def encode_state_for_nn(row: Dict[str, Any]) -> np.ndarray | None:
    """
    Encodes a row from the database (representing state *before* move)
    into the 54-feature vector for the NN.
    Returns None if data is invalid.
    """
    try:
        board = json.loads(row["board_pts"]) # List of 24 ints
        if len(board) != 24: return None

        # Clamp board values just in case
        board_np = np.array(board, dtype=np.int8)

        # Get bar/off counts, default to 0 if None/missing
        w_bar = int(row.get("bar_w") or 0)
        b_bar = int(row.get("bar_b") or 0)
        w_off = int(row.get("off_w") or 0)
        b_off = int(row.get("off_b") or 0)

        move_num = row.get("move_number")
        if move_num is None: return None # Cannot determine turn
        player_turn = 'w' if move_num % 2 == 0 else 'b'
        turn_feature = 1.0 if player_turn == 'w' else 0.0

        # Calculate pip difference for this state
        pip_w = calculate_pip_simple(board, w_bar, b_bar, 'w')
        pip_b = calculate_pip_simple(board, w_bar, b_bar, 'b')
        # Normalize: (pip_b - pip_w) / 100.0
        pip_diff_feature = (pip_b - pip_w) / 100.0

        # Normalize board features
        whites_norm = np.clip(board_np, 0, NUM_CHECKERS) / NUM_CHECKERS
        blacks_norm = np.clip(-board_np, 0, NUM_CHECKERS) / NUM_CHECKERS

        # Normalize bar/off features
        bar_off_norm = np.array([
            w_bar / NUM_CHECKERS,
            b_bar / NUM_CHECKERS,
            w_off / NUM_CHECKERS,
            b_off / NUM_CHECKERS
        ], dtype=np.float32)

        features = np.concatenate([
            whites_norm,        # Features 0-23
            blacks_norm,        # Features 24-47
            bar_off_norm,       # Features 48-51
            np.array([turn_feature], dtype=np.float32),      # Feature 52
            np.array([pip_diff_feature], dtype=np.float32)   # Feature 53
        ])

        if len(features) != INPUT_DIM:
             print(f"Warning: Feature length mismatch for row {row.get('file_uid', '')}/{row.get('game_id', '')}/{row.get('move_number', '')}. Expected {INPUT_DIM}, got {len(features)}. Skipping.")
             return None

        return features.astype(np.float32)

    except Exception as e:
        # Log error with more context if possible
        fuid = row.get('file_uid', 'unknown')
        gid = row.get('game_id', 'unknown')
        m_num = row.get('move_number', 'unknown')
        print(f"Error encoding state for {fuid}/{gid}/{m_num}: {e}")
        return None

###############################################################################
#  DATASET                                                                    #
###############################################################################

class EquityDataset(Dataset):
    """
    Lazy dataset reading pre-move states and post-move equity from SQLite.
    Handles per-worker DB connection.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cached_data = None # Cache data after first load
        self._load_data()

    def _load_data(self):
        """Loads data from SQLite into memory."""
        if self._cached_data is not None:
            return

        print(f"Loading data from {self.db_path}...")
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        query = """
            SELECT m.file_uid, m.game_id, m.move_number, -- Identifiers
                   m.board_pts, m.bar_b, m.bar_w, m.off_b, m.off_w, -- State features
                   (SELECT equity FROM evaluations e
                      WHERE e.file_uid = m.file_uid
                        AND e.game_id  = m.game_id
                        AND e.move_number = m.move_number
                        AND e.rank = 1) AS best_eq -- Target value
            FROM moves m
            WHERE best_eq IS NOT NULL -- Only include rows where target exists
        """
        cur = conn.execute(query)
        # Store raw rows first
        raw_rows = [dict(r) for r in cur]
        conn.close()
        print(f"Loaded {len(raw_rows)} raw rows from DB.")

        # Process rows into features (X) and targets (y)
        processed_data = []
        invalid_count = 0
        for i, row in enumerate(raw_rows):
            x_features = encode_state_for_nn(row)
            if x_features is not None:
                try:
                    y_target = float(row["best_eq"])
                    processed_data.append({'x': x_features, 'y': y_target})
                except (TypeError, ValueError):
                    invalid_count += 1
            else:
                invalid_count += 1 # encode_state_for_nn failed

        if invalid_count > 0:
            print(f"Warning: Skipped {invalid_count} rows due to invalid features or target.")

        self._cached_data = processed_data
        print(f"Successfully processed {len(self._cached_data)} data points.")

    def __len__(self):
        return len(self._cached_data) if self._cached_data else 0

    def __getitem__(self, idx):
        if self._cached_data is None:
            raise RuntimeError("Dataset not loaded properly.")
        data = self._cached_data[idx]
        # Return tensors
        return torch.from_numpy(data['x']), torch.tensor(data['y'], dtype=torch.float32)

###############################################################################
#  MODEL (Ensure INPUT_DIM matches)                                           #
###############################################################################

class MiniMaxHelper(pl.LightningModule):
    def __init__(self, input_dim: int = INPUT_DIM, learning_rate: float = LR):
        super().__init__()
        
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Define the network architecture
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.loss_fn = nn.MSELoss() # --> Mean Squared Error for regression

        print(f"Initialized MiniMaxHelper model with input_dim={self.input_dim}")

    def forward(self, x):
        return self.net(x).squeeze(1) # --> Remove the last dimension

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        mae = torch.abs(y_pred - y).mean()
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

###############################################################################
#  MAIN TRAINING LOOP                                                         #
###############################################################################

def main():
    print("Starting training process...")
    pl.seed_everything(SEED)

    # --- Dataset and Dataloaders ---
    print("Initializing dataset...")
    full_ds = EquityDataset(DB_PATH)
    if len(full_ds) == 0:
        print("ERROR: No data loaded from the database. Cannot train.")
        return

    val_len = int(len(full_ds) * VAL_SPLIT)
    train_len = len(full_ds) - val_len
    print(f"Splitting data: Train={train_len}, Validation={val_len}")
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    persistent = NUM_WORKERS > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )
    print("Dataloaders created.")

    # --- Model and Trainer ---
    print("Initializing model...")
    model = MiniMaxHelper(input_dim=INPUT_DIM, learning_rate=LR)

    print("Initializing trainer...")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',        # Metric to monitor
        dirpath='checkpoints/',    # Directory to save checkpoints
        filename='minimax-helper-{epoch:02d}-{val_loss:.4f}', # Filename format
        save_top_k=1,              # Save only the best model
        mode='min',                # Minimize validation loss
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,     # Prevent exploding gradients
        log_every_n_steps=50,      # How often to log within an epoch
        callbacks=[checkpoint_callback] # Add callbacks
    )

    # --- Training ---
    print("Starting training...")
    trainer.fit(model, train_dl, val_dl)
    print("Training finished.")

    # --- Save the Best Model ---
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

    # Load the best model checkpoint
    best_model = MiniMaxHelper.load_from_checkpoint(best_model_path)

    output_filename = "minimax_helper.pt"
    out = Path(output_filename)
    torch.save(best_model.state_dict(), out)
    print(f"Final model state_dict saved to {out}")

if __name__ == "__main__":
    # Check if DB exists before starting
    if not Path(DB_PATH).exists():
        print(f"ERROR: Database file '{DB_PATH}' not found.")
        print("Please run the parsing script (e.g., parse_gnubg.py) first.")
    else:
        main()
