# -*- coding: utf-8 -*-
"""train_minimax_helper.py

Entraîne un réseau neuronal léger pour approximer l'équité de GNU Backgammon
(meilleur coup, 2-ply) pour une position donnée. Le modèle guidera ensuite
une recherche minimax en élaguant les branches dont l'équité prédite est faible.

Les données sont lues depuis la base de données SQLite produite par
`parse_to_database.py`.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MeanAbsoluteError, R2Score, PearsonCorrCoef

###############################################################################
#  CONFIG (Defaults, can be overridden by CLI arguments)                      #
###############################################################################
DB_PATH_DEFAULT     = "matches.db"  # Database path
BATCH_SIZE_DEFAULT  = 1024
EPOCHS_DEFAULT      = 50    # Max epochs (EarlyStopping might stop sooner)
LR_DEFAULT          = 3e-4  # Learning rate
VAL_SPLIT_DEFAULT   = 0.15  # Validation set proportion
TEST_SPLIT_DEFAULT  = 0.15  # Test set proportion
SEED_DEFAULT        = 42    # Seed for reproducibility
NUM_WORKERS_DEFAULT = 8     # DataLoader workers (adjust based on CPU cores)
HIDDEN_DIM_DEFAULT  = 256   # Hidden layer dimension
NUM_LAYERS_DEFAULT  = 2     # Number of *hidden* layers
DROPOUT_DEFAULT     = 0.1   # Dropout probability
OUTPUT_ACTIVATION_DEFAULT = "none" # Output activation ('none' or 'tanh')
OPTIMIZER_DEFAULT   = "adamw" # Optimizer ('adam' or 'adamw')
WEIGHT_DECAY_DEFAULT = 1e-5   # Weight decay for AdamW
OUTPUT_FILE_DEFAULT = "minimax_helper.pt" # Output model file
CHECKPOINT_DIR_DEFAULT = "checkpoints/" # Directory for model checkpoints
LOG_DIR_DEFAULT     = "tb_logs/"        # Directory for TensorBoard logs

###############################################################################
#  FEATURE ENCODING (Aligned with BKG_2_05.py for 54 features)               #
#  CONVENTION: 'O' = WHITE, 'X' = BLACK                                      #
###############################################################################

INPUT_DIM_CONFIGURABLE = 54 # Fixed input dimension for the model

NORM_FACTOR_CHECKERS = 15.0
NORM_FACTOR_PIP_DIFF = 100.0

def encode_position_to_tensor(
    board_pts_json: str,    # JSON list, DB convention: + is Black(X), - is White(O)
    current_player_db: str, # 'O' (White) or 'X' (Black) from DB
    db_bar_w: int, db_bar_b: int,   # DB convention: _w = White(O), _b = Black(X)
    db_off_w: int, db_off_b: int,
    db_pip_w: int, db_pip_b: int
) -> np.ndarray:
    """
    Encodes the full position into a 54-feature tensor.
    Aligns with the encoding expected by BKG_2_05.py.
    Target Tensor Convention: feature[0-23]=White(+), feature[24-47]=Black(-),
                              feature[52]=Turn (1.0 if White's turn, 0.0 if Black's turn).
    """
    board_list = json.loads(board_pts_json)
    # DB stores + for Black(X), - for White(O)
    board_array_black_positive = np.array(board_list, dtype=np.int8)

    # --- Convert DB board representation to Target Tensor representation ---
    # Target features 0-23: White checkers (positive values in target tensor)
    # White checkers in DB are negative in board_array_black_positive.
    white_checkers_on_board_norm = np.clip(-board_array_black_positive, 0, NORM_FACTOR_CHECKERS) / NORM_FACTOR_CHECKERS
    # Target features 24-47: Black checkers (positive values for count, will be handled by model)
    # Black checkers in DB are positive in board_array_black_positive.
    black_checkers_on_board_norm = np.clip(board_array_black_positive, 0, NORM_FACTOR_CHECKERS) / NORM_FACTOR_CHECKERS

    board_features = np.concatenate([white_checkers_on_board_norm, black_checkers_on_board_norm])

    # --- Bar/Off features ---
    # Use DB values directly as DB _w corresponds to White(O) and DB _b to Black(X)
    bar_off_features = np.array([
        (db_bar_w or 0) / NORM_FACTOR_CHECKERS, # 48: white_bar
        (db_bar_b or 0) / NORM_FACTOR_CHECKERS, # 49: black_bar
        (db_off_w or 0) / NORM_FACTOR_CHECKERS, # 50: white_off
        (db_off_b or 0) / NORM_FACTOR_CHECKERS, # 51: black_off
    ], dtype=np.float32)

    # --- Turn feature ---
    # 1.0 if White's ('O') turn, 0.0 if Black's ('X') turn
    turn_feature_val = 1.0 if current_player_db == 'O' else 0.0
    turn_feature_np = np.array([turn_feature_val], dtype=np.float32) # 52: turn

    # --- Pip difference feature ---
    # Target: (pip_black - pip_white) / NORM_FACTOR_PIP_DIFF
    pip_diff_val = (db_pip_b - db_pip_w) / NORM_FACTOR_PIP_DIFF
    pip_diff_feature = np.array([pip_diff_val], dtype=np.float32) # 53: pip_diff

    # --- Concatenate all features in the correct order ---
    final_features = np.concatenate([
        board_features,      # Features 0-47
        bar_off_features,    # Features 48-51
        turn_feature_np,     # Feature 52
        pip_diff_feature     # Feature 53
    ]).astype(np.float32)

    if final_features.shape[0] != INPUT_DIM_CONFIGURABLE:
        raise ValueError(f"Incorrect feature encoding dimension: {final_features.shape[0]} vs {INPUT_DIM_CONFIGURABLE}")
    return final_features

###############################################################################
#  DATASET                                                                    #
###############################################################################

class EquityDataset(Dataset):
    """Loads data from SQLite on the fly for training."""
    def __init__(self, db_path: str, ids_list: List[int]):
        self.db_path = db_path
        if not ids_list:
            raise ValueError("ids_list cannot be empty for EquityDataset")
        self.rowids = ids_list
        self.conn = None # Connection established per worker

    @staticmethod
    def get_base_query_ids(db_path_for_query: str) -> List[int]:
        """Queries DB to get all valid rowids based on necessary columns."""
        print(f"Querying valid rowids from {db_path_for_query}...")
        conn_temp = sqlite3.connect(db_path_for_query)
        query = """
            SELECT m.rowid
            FROM moves m
            WHERE m.board_pts IS NOT NULL
              AND m.player IS NOT NULL AND (m.player = 'X' OR m.player = 'O') -- Check for corrected player symbol
              AND m.bar_b IS NOT NULL AND m.bar_w IS NOT NULL
              AND m.off_b IS NOT NULL AND m.off_w IS NOT NULL
              AND m.pip_b IS NOT NULL AND m.pip_w IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM evaluations e
                  WHERE e.file_uid = m.file_uid
                    AND e.game_id  = m.game_id
                    AND e.move_number = m.move_number
                    AND e.rank = 1
                    AND e.equity IS NOT NULL
              )
        """
        try:
            cur = conn_temp.execute(query)
            rowids = [r[0] for r in cur]
        except sqlite3.Error as e:
            print(f"Erreur SQLite lors de la récupération des rowids: {e}")
            print("Vérifiez le chemin de la BDD et la structure des tables/colonnes.")
            sys.exit(1)
        finally:
            conn_temp.close()

        if not rowids:
            print("WARNING: Aucune position valide trouvée avec la requête `get_base_query_ids`.")
            print("         Vérifiez que la BDD contient les données attendues (y compris 'O'/'X' dans moves.player).")
        else:
            print(f"Found {len(rowids)} valid positions.")
        return rowids

    def _open_connection(self):
        """Opens SQLite connection (used by DataLoader workers)."""
        if self.conn is None:
            # Connect in read-only mode if possible, increase timeout
            try:
                # URI allows read-only mode, requires Python 3.4+ sqlite3
                db_uri = f"file:{self.db_path}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=20, check_same_thread=False)
            except sqlite3.OperationalError:
                # Fallback if URI or read-only mode is not supported
                print(f"Worker {os.getpid()}: Read-only connection failed, connecting normally.")
                self.conn = sqlite3.connect(self.db_path, timeout=20, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["conn"] = None # Don't pickle the connection object
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        # Connection will be re-established in __getitem__ if needed

    def __len__(self) -> int:
        return len(self.rowids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._open_connection()
        rowid = self.rowids[idx]
        try:
            cur = self.conn.execute("""
                SELECT m.board_pts, m.player,
                       m.bar_w, m.bar_b,  -- DB _w = White (O), DB _b = Black (X)
                       m.off_w, m.off_b,
                       m.pip_w, m.pip_b,
                       (SELECT equity FROM evaluations e
                        WHERE e.file_uid = m.file_uid
                          AND e.game_id  = m.game_id
                          AND e.move_number = m.move_number
                          AND e.rank = 1) AS target_equity
                FROM moves m
                WHERE m.rowid = ?
            """, (rowid,))
            row = cur.fetchone()
        except sqlite3.Error as e:
             print(f"Erreur SQLite dans __getitem__ pour rowid {rowid}: {e}")
             # Return dummy data or raise error? Raising is safer.
             raise RuntimeError(f"Erreur SQLite pour rowid {rowid}") from e


        if row is None:
            # This should ideally not happen if rowids are validated first
            raise IndexError(f"No data found for rowid {rowid} at index {idx}")

        try:
            x_np = encode_position_to_tensor(
                board_pts_json=row["board_pts"],
                current_player_db=row["player"], # 'O' or 'X'
                db_bar_w=row["bar_w"], db_bar_b=row["bar_b"],
                db_off_w=row["off_w"], db_off_b=row["off_b"],
                db_pip_w=row["pip_w"], db_pip_b=row["pip_b"]
            )
            y_val = float(row["target_equity"])

            return torch.from_numpy(x_np), torch.tensor(y_val, dtype=torch.float32)
        except Exception as e:
            print(f"Erreur d'encodage pour rowid {rowid}: {e}")
            # Again, raise error to stop potentially bad training
            raise RuntimeError(f"Erreur d'encodage pour rowid {rowid}") from e

###############################################################################
#  MODEL                                                                      #
###############################################################################

class MiniMaxHelper(pl.LightningModule):
    def __init__(self,
                 input_dim: int = INPUT_DIM_CONFIGURABLE, # Use the configured dimension
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 2,
                 dropout_p: float = 0.1,
                 learning_rate: float = 3e-4,
                 optimizer_name: str = "adamw",
                 weight_decay: float = 1e-5,
                 output_activation: str = "none"):
        super().__init__()
        if input_dim != INPUT_DIM_CONFIGURABLE:
             print(f"WARNING: Model input_dim ({input_dim}) differs from expected ({INPUT_DIM_CONFIGURABLE})")
        self.save_hyperparameters() # Saves args to hparams attribute

        layers = []
        current_dim = self.hparams.input_dim # Access via hparams
        # Hidden layers
        for _ in range(self.hparams.num_hidden_layers):
            layers.append(nn.Linear(current_dim, self.hparams.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
            layers.append(nn.ReLU())
            if self.hparams.dropout_p > 0:
                layers.append(nn.Dropout(self.hparams.dropout_p))
            current_dim = self.hparams.hidden_dim
        # Output layer
        layers.append(nn.Linear(current_dim, 1))

        if self.hparams.output_activation == "tanh":
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

        # --- Metrics Initialization ---
        # Use ModuleDict for proper device handling by Lightning
        metrics = {
            "mae": MeanAbsoluteError(),
            "r2": R2Score(),
            "pearson": PearsonCorrCoef()
        }
        self.train_metrics = nn.ModuleDict({"train_" + k: v.clone() for k, v in metrics.items()})
        self.val_metrics = nn.ModuleDict({"val_" + k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({"test_" + k: v.clone() for k, v in metrics.items()})


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) # Squeeze the last dimension

    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred, y_true

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, y_pred, y_true = self._common_step(batch)
        # Log loss immediately
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Update metrics
        for metric_name, metric in self.train_metrics.items():
            metric.update(y_pred, y_true)
        # Log metrics on epoch end
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, y_pred, y_true = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric_name, metric in self.val_metrics.items():
            metric.update(y_pred, y_true)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, y_pred, y_true = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        for metric_name, metric in self.test_metrics.items():
            metric.update(y_pred, y_true)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hparams.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=7,
            min_lr=1e-7
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

###############################################################################
#  MAIN                                                                       #
###############################################################################

def main(args: argparse.Namespace):
    pl.seed_everything(args.seed, workers=True)

    print("--- Training Configuration ---")
    for k, v in vars(args).items(): print(f"  {k:<20}: {v}")
    print(f"  {'Input Dimension':<20}: {INPUT_DIM_CONFIGURABLE}")
    print("-" * 30)

    # --- Create output directories ---
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # --- Load ALL valid rowids ONCE ---
    all_rowids = EquityDataset.get_base_query_ids(args.db_path)
    if not all_rowids:
        print("\nERROR: No valid data found in the database. Exiting.")
        sys.exit(1)

    # --- Shuffle and Split rowids ---
    np.random.shuffle(all_rowids)
    n_total = len(all_rowids)
    n_test = int(n_total * args.test_split)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        print(f"\nERROR: Not enough data ({n_total}) for the specified splits "
              f"(Train: {n_train}, Val: {n_val}, Test: {n_test}). "
              f"Adjust --val_split or --test_split.")
        sys.exit(1)

    train_ids = all_rowids[:n_train]
    val_ids = all_rowids[n_train : n_train + n_val]
    test_ids = all_rowids[n_train + n_val :]

    print(f"\nData Split: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}")

    # --- Create Datasets ---
    try:
        train_ds = EquityDataset(args.db_path, train_ids)
        val_ds = EquityDataset(args.db_path, val_ids)
        test_ds = EquityDataset(args.db_path, test_ids)
    except ValueError as e: # Catch empty ids_list error
        print(f"Error creating dataset: {e}")
        sys.exit(1)


    # --- Quick dimension check on one sample ---
    try:
        sample_x, sample_y = train_ds[0]
        if sample_x.shape[0] != INPUT_DIM_CONFIGURABLE:
             raise ValueError(f"Dimension mismatch: Expected {INPUT_DIM_CONFIGURABLE}, got {sample_x.shape[0]}")
        print(f"Sample input dimension check OK: {sample_x.shape}")
        print(f"Sample target value: {sample_y.item():.4f}")
    except Exception as e:
        print(f"\nERROR verifying dataset sample: {e}")
        print("Please check the `encode_position_to_tensor` function and database consistency.")
        sys.exit(1)

    # --- Create DataLoaders ---
    # Determine optimal num_workers if set to -1 (auto)
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = os.cpu_count() or 0
        print(f"Auto-detected num_workers: {num_workers}")
    # Ensure num_workers isn't 1 if persistent_workers is True, as it can cause issues.
    if num_workers == 1 and sys.platform != "win32": # Persistent workers generally okay on Windows with 1 worker
         print("Warning: Setting num_workers=1 with persistent_workers=True can sometimes lead to hangs. Consider using 0 or >=2.")
         # persistent_workers = False # Option: disable persistence if workers=1
    elif num_workers > 0 :
        persistent_workers = True
    else: # num_workers == 0
        persistent_workers = False

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                         num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True)

    # --- Initialize Model ---
    model = MiniMaxHelper(
        input_dim=INPUT_DIM_CONFIGURABLE, # Use the fixed constant
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        dropout_p=args.dropout_p,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        output_activation=args.output_activation
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.checkpoint_dir,
        filename="equity-{epoch:02d}-{val_loss:.4f}-{val_mae:.4f}", # Include MAE in filename
        save_top_k=3,
        mode="min",
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001, # Smaller delta for potentially finer improvements
        patience=15,      # Increased patience
        verbose=True,
        mode="min"
    )
    rich_progress_bar = RichProgressBar()

    # --- Logger ---
    tensorboard_logger = TensorBoardLogger(args.log_dir, name="minimax_helper_equity_run")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=max(1, len(train_dl) // 100), # Log ~100 times per epoch
        callbacks=[checkpoint_callback, early_stop_callback, rich_progress_bar],
        logger=tensorboard_logger,
        deterministic="warn" # Use 'warn' for better performance unless strict reproducibility is needed
    )

    # --- Training ---
    print("\n--- Starting Training ---")
    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        print("--- Training Finished ---")
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Testing ---
    print("\n--- Evaluating on Test Set ---")
    best_model_path = checkpoint_callback.best_model_path
    final_model_for_testing = model # Default to current model state

    if best_model_path and Path(best_model_path).exists():
        print(f"Loading best model from checkpoint: {best_model_path}")
        try:
            final_model_for_testing = MiniMaxHelper.load_from_checkpoint(best_model_path)
        except Exception as e:
            print(f"Warning: Failed to load best checkpoint ({e}). Testing with last model state.")
            # Try loading last checkpoint if best fails
            if checkpoint_callback.last_model_path and Path(checkpoint_callback.last_model_path).exists():
                 print(f"Attempting to load last checkpoint: {checkpoint_callback.last_model_path}")
                 try:
                     final_model_for_testing = MiniMaxHelper.load_from_checkpoint(checkpoint_callback.last_model_path)
                 except Exception as e_last:
                     print(f"Warning: Failed to load last checkpoint ({e_last}). Testing with model in memory.")
            else:
                print("No valid last checkpoint found either. Testing with model in memory.")

    elif checkpoint_callback.last_model_path and Path(checkpoint_callback.last_model_path).exists():
        print(f"Best model checkpoint not found. Loading last model from: {checkpoint_callback.last_model_path}")
        try:
             final_model_for_testing = MiniMaxHelper.load_from_checkpoint(checkpoint_callback.last_model_path)
        except Exception as e_last:
             print(f"Warning: Failed to load last checkpoint ({e_last}). Testing with model in memory.")
    else:
        print("No best or last checkpoint found. Testing with final model state in memory.")

    try:
        test_results = trainer.test(final_model_for_testing, dataloaders=test_dl, verbose=False) # verbose=False to avoid double printing
        if test_results:
            print("\n--- Test Set Results ---")
            for key, value in test_results[0].items():
                print(f"  {key:<20}: {value:.4f}")
            print("-" * 30)
        else:
            print("WARNING: No results returned from trainer.test().")
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


    # --- Save Final Model State Dictionary ---
    out_path = Path(args.output_file)
    try:
        torch.save(final_model_for_testing.state_dict(), out_path)
        print(f"\nFinal model state_dict saved successfully to: {out_path}")
    except Exception as e:
        print(f"\nERROR saving final model state_dict: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MiniMaxHelper model for Backgammon equity prediction.")
    # --- Data/Paths ---
    parser.add_argument("--db_path", type=str, default=DB_PATH_DEFAULT, help="Path to the SQLite database.")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE_DEFAULT, help="Path to save the final trained model state_dict.")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR_DEFAULT, help="Directory to save model checkpoints.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR_DEFAULT, help="Directory for TensorBoard logs.")
    # --- Training Params ---
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT_DEFAULT, help="Validation set split ratio (0.0 to 1.0).")
    parser.add_argument("--test_split", type=float, default=TEST_SPLIT_DEFAULT, help="Test set split ratio (0.0 to 1.0).")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT, help="Number of DataLoader workers (-1 for auto).")
    # --- Model Architecture ---
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM_DEFAULT, help="Dimension of hidden layers.")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS_DEFAULT, help="Number of *hidden* layers.")
    parser.add_argument("--dropout_p", type=float, default=DROPOUT_DEFAULT, help="Dropout probability (0 to disable).")
    parser.add_argument("--output_activation", type=str, default=OUTPUT_ACTIVATION_DEFAULT, choices=["none", "tanh"], help="Activation function for the output layer.")
    # --- Optimizer ---
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER_DEFAULT, choices=["adam", "adamw"], help="Optimizer.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT, help="Weight decay for AdamW optimizer.")

    cli_args = parser.parse_args()

    main(cli_args)
