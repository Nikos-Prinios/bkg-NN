# Ci-après, le meilleur Backgammon de tous les temps
# ---------------------------------------------------------------------------
# 1) DÉPENDANCES
# ---------------------------------------------------------------------------
import os, sys, time, math, random, copy, traceback
from sentences import PhraseManager # Gestion des phrases d'ambiance
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from functools import lru_cache
from numpy.typing import NDArray
BoardArr = NDArray[np.int8]
import cProfile
import pstats
import copy as std_copy

# Initialisation Cython
calculate_pip_fast = None
_get_single_moves_for_die_cy = None
make_move_base_logic_cy = None
check_all_pieces_home_cy = None
compute_zobrist_cy_func = None

try:
    import speedups
    CYTHON_OK = True
    print("Fonctions Cython chargées.")
    calculate_pip_fast = speedups.calculate_pip
    _get_single_moves_for_die_cy = speedups.get_single_moves_for_die
    make_move_base_logic_cy = speedups.make_move_base_logic
    check_all_pieces_home_cy = speedups.check_all_pieces_home
    compute_zobrist_cy_func = speedups.compute_zobrist_cy
except ImportError:
    CYTHON_OK = False
    print("ERREUR CRITIQUE : Module Cython 'speedups' introuvable.")
    print("Veuillez compiler l'extension Cython (speedups.pyx).")
    sys.exit(1)

# — PyTorch —
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("❌ PyTorch n’est pas installé ; faites : pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 0) DEVICE (CPU / MPS (Apple Silicon) / CUDA)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"- Utilisation du device : {DEVICE}")

# ---------------------------------------------------------------------------
# 2) CONSTANTES GLOBALES
# ---------------------------------------------------------------------------
SHOW_STATS          = False      # Afficher les stats de performance ?
MAX_DEPTH           = 4          # Profondeur de recherche minimax
NUM_DICE_SAMPLES    = 21         # Nb. de lancers simulés (nœuds chance)
NN_CHECKPOINT       = "minimax_helper.pt" # Fichier du modèle NN
NN_WEIGHT           = 0.50       # Poids du NN vs Heuristique (0=Heuristique, 1=NN)
BOARD_SIZE          = 24
NUM_CHECKERS        = 15
BATCH_BUF: list[tuple['BackgammonGame',str]] = [] # Buffer d'évaluation NN
BATCH_KEYS: list[tuple] = []                     # Clés associées au buffer
BATCH_SIZE = 256                                 # Taille du buffer NN
AI_MODE = "MINIMAX"                              # Mode IA: "MINIMAX", "NN_ONLY"

INPUT_DIM = 54                  # Dimension d'entrée fixe
HIDDEN = 256                  # Dimension cachée utilisée à l'entraînement
NUM_HIDDEN_LAYERS = 2           # Nombre de couches CACHÉES utilisées à l'entraînement
DROPOUT_P = 0.1                 # Dropout utilisé à l'entraînement (sera inactif en mode eval)
OUTPUT_ACTIVATION = "none"      # Activation de sortie utilisée à l'entraînement
# ---------------------------------------------------------------
# 2-bis) Table des coups unitaires (src,dst) pré-calculée
# ---------------------------------------------------------------
MOVE_TABLE: dict[int, list[tuple[int,int]]] = {d: [] for d in range(1,7)}
for src in range(1,25):
    for d in range(1,7):
        dst_w = src + d
        dst_b = src - d
        if dst_w <= 24: MOVE_TABLE[d].append( (src, dst_w) )
        if dst_b >= 1 : MOVE_TABLE[d].append( (src, dst_b) )

# ---------------------------------------------------------------
# 2-ter) ZOBRIST : Nombres aléatoires pour hash incrémental (reproductible)
# ---------------------------------------------------------------
RND64 = np.random.default_rng(2025).integers(
           low=0, high=2**63, dtype=np.uint64, size=(24, 31))
RND_BAR   = np.random.default_rng(42).integers(
           low=0, high=2**63, dtype=np.uint64, size=(2,16))
RND_OFF   = np.random.default_rng(123).integers(
           low=0, high=2**63, dtype=np.uint64, size=(2,16))
RND_TURN  = np.random.default_rng(314).integers(
           low=0, high=2**63, dtype=np.uint64, size=2)

# ---------------------------------------------------------------------------
# 3) MINIMAX
# ---------------------------------------------------------------------------
class MiniMaxHelper(nn.Module):
    """ Architecture du réseau neuronal pour l'inférence, alignée sur l'entraînement."""
    def __init__(self,
                 input_dim: int = INPUT_DIM, # Utilise la constante globale
                 hidden_dim: int = HIDDEN,
                 num_hidden_layers: int = NUM_HIDDEN_LAYERS,
                 dropout_p: float = DROPOUT_P,
                 output_activation: str = OUTPUT_ACTIVATION):
        super().__init__()
        layers = []
        current_dim = input_dim
        # Couches cachées
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        if output_activation == "tanh":
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_single_input = False
        if x.ndim == 1:
           x = x.unsqueeze(0)
           is_single_input = True
        output = self.net(x)
        output = output.squeeze(-1)
        if is_single_input:
             pass
        return output
_nn_model: MiniMaxHelper | None = None

# ---------------------------------------------------------------------------
# Table de Transposition (mémoire pour Minimax)
# ---------------------------------------------------------------------------
TT: dict[tuple, float] = {}    # (hash_zobrist, profondeur, joueur) -> score

# ---------------------------------------------------------------------------
# Fonction de chargement du modèle (Utilise la nouvelle classe)
# ---------------------------------------------------------------------------
def _load_nn_model() -> MiniMaxHelper | None:
    global _nn_model
    if _nn_model is not None:
        return _nn_model
    if not os.path.exists(NN_CHECKPOINT):
        print(f"⚠️ Modèle {NN_CHECKPOINT} introuvable – Heuristique seule.")
        return None

    print(f"Chargement du modèle NN '{NN_CHECKPOINT}' (Input: {INPUT_DIM})")
    try:
        _nn_model = MiniMaxHelper(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            dropout_p=DROPOUT_P,
            output_activation=OUTPUT_ACTIVATION
        )

        state_dict = torch.load(NN_CHECKPOINT, map_location=DEVICE)
        _nn_model.load_state_dict(state_dict)
        _nn_model.to(DEVICE)
        _nn_model.eval()
        print("Modèle NN chargé et prêt pour l'inférence.")
        return _nn_model
    except FileNotFoundError:
         print(f"❌ Erreur: Fichier modèle {NN_CHECKPOINT} non trouvé.")
         print("   Utilisation de l'heuristique seule.")
         return None
    except RuntimeError as e: # Attrape spécifiquement les erreurs de chargement state_dict
        print(f"❌ Erreur chargement state_dict modèle {NN_CHECKPOINT}: {e}")
        print(f"   Cela indique généralement une INCOHÉRENCE D'ARCHITECTURE entre l'entraînement et ici.")
        print(f"   Vérifiez les constantes : INPUT_DIM={INPUT_DIM}, HIDDEN={HIDDEN}, NUM_HIDDEN_LAYERS={NUM_HIDDEN_LAYERS}, etc.")
        print("   Utilisation de l'heuristique seule.")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e: # Attrape les autres erreurs potentielles
        print(f"❌ Erreur chargement modèle {NN_CHECKPOINT}: {e}")
        print("   Utilisation de l'heuristique seule.")
        import traceback
        traceback.print_exc()
        return None

NET = _load_nn_model()

# Couleur CLI
YEL = "\033[33m"; RESET = "\033[0m"
def color(s, clr): return f"{clr}{s}{RESET}" # Utilitaire couleur

# ---------------------------------------------------------------------------
# 4) FONCTION UTILITAIRE : encodage plateau -> tenseur NN
# ---------------------------------------------------------------------------
def state_to_tensor(game: 'BackgammonGame') -> torch.Tensor:
    """
    Encode l'état du jeu en tenseur 54 features pour le NN.
    Convention BKG_2_05: Blancs='w'(+), Noirs='b'(-).
    Tenseur: feats[0-23]=Blancs, feats[24-47]=Noirs, feat[52]=Tour(1.0=Blanc)
    """
    # Board: BKG_2_05 stocke Blancs (+), Noirs (-)
    white_checkers_on_board_norm = np.clip(game.board, 0, 15.0) / 15.0
    black_checkers_on_board_norm = np.clip(-game.board, 0, 15.0) / 15.0
    board_features = np.concatenate([white_checkers_on_board_norm, black_checkers_on_board_norm])

    # Bar/Off
    bar_off_features = np.array([
        game.white_bar / 15.0, # 48: white_bar
        game.black_bar / 15.0, # 49: black_bar
        game.white_off / 15.0, # 50: white_off
        game.black_off / 15.0, # 51: black_off
    ], dtype=np.float32)

    # Turn feature: 1.0 pour Blanc ('w'), 0.0 pour Noir ('b')
    turn_feature_val = 1.0 if game.current_player == 'w' else 0.0
    turn_feature_np = np.array([turn_feature_val], dtype=np.float32) # 52: turn

    # Pip difference feature: (pip_noir - pip_blanc) / 100.0
    # Utilise la méthode de calcul de pip de la classe (potentiellement Cython)
    pip_w = game.calculate_pip('w')
    pip_b = game.calculate_pip('b')
    pip_diff_val = (pip_b - pip_w) / 100.0
    pip_diff_feature = np.array([pip_diff_val], dtype=np.float32) # 53: pip_diff

    # Concatenate all features
    final_features = np.concatenate([
        board_features,      # Features 0-47
        bar_off_features,    # Features 48-51
        turn_feature_np,     # Feature 52
        pip_diff_feature     # Feature 53
    ]).astype(np.float32)

    if final_features.shape[0] != INPUT_DIM:
        raise ValueError(f"Erreur interne: Taille tenseur {final_features.shape[0]} != INPUT_DIM {INPUT_DIM}")

    return torch.tensor(final_features, device=DEVICE)

# ---------------------------------------------------------------------------
# 4-bis) Tensor pour la perspective d'un joueur (+ pour W, - pour B)
# ---------------------------------------------------------------------------
def _tensor_for_player(game: 'BackgammonGame', player: str) -> torch.Tensor:
    """
    Retourne un tenseur normalisé du point de vue du joueur spécifié.
    Si player='b', inverse la représentation.
    """
    t_np = state_to_tensor(game).cpu().numpy() # Obtenir comme numpy array

    if player == 'b': # Si on veut la perspective du Noir
        # Inverser features plateau (Blancs <=> Noirs)
        whites_orig = t_np[0:24].copy()
        blacks_orig = t_np[24:48].copy()
        t_np[0:24] = blacks_orig  # Noirs deviennent features 0-23
        t_np[24:48] = whites_orig # Blancs deviennent features 24-47

        # Inverser features bar/off
        t_np[48], t_np[49] = t_np[49], t_np[48] # bar_w <=> bar_b
        t_np[50], t_np[51] = t_np[51], t_np[50] # off_w <=> off_b

        # Inverser feature de tour (1=adversaire(Blanc), 0=joueur(Noir))
        t_np[52] = 1.0 - t_np[52]

        # Inverser différence de pip (pip_b - pip_w) -> -(pip_b - pip_w) = (pip_w - pip_b)
        t_np[53] = -t_np[53]

    return torch.tensor(t_np, device=DEVICE)

# ---------------------------------------------------------------------------
# 5) HEURISTIQUE
# ---------------------------------------------------------------------------
@dataclass
class HeuristicWeights:
    # --- Facteurs de base ---
    PIP_SCORE_FACTOR: float = 1.0
    OFF_SCORE_FACTOR: float = 10.0
    HIT_BONUS: float = 30.0
    BAR_PENALTY: float = -20.0

    # --- Structure: Points, Ancres ---
    POINT_BONUS: float = 2.0
    HOME_BOARD_POINT_BONUS: float = 3.0
    INNER_HOME_POINT_BONUS: float = 2.0
    ANCHOR_BONUS: float = 5.0
    FIVE_POINT_BONUS: float = 0.0 # Défaut = 0.0

    # --- Structure: Primes ---
    PRIME_BASE_BONUS: float = 4.0
    SIX_PRIME_BONUS: float = 0.0 # Défaut = 0.0
    TRAPPED_CHECKER_BONUS: float = 8.0

    # --- Structure: Distribution ---
    STACKING_PENALTY_FACTOR: float = 0.0 # Défaut = 0.0 (Facteur négatif dans instances)
    BUILDER_BONUS: float = 0.0 # Défaut = 0.0

    # --- Blots ---
    DIRECT_SHOT_PENALTY_FACTOR: float = -1.5
    BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR: float = 0.5
    HIT_FAR_BLOT_PENALTY_MULTIPLIER: float = 1.0 # Défaut = 1.0 (pas de majoration)
    STRATEGIC_BLOT_PENALTY_REDUCTION: float = 1.0 # Défaut = 1.0 (pas de réduction)

    # --- Situationnel / Contrôle ---
    MIDGAME_HOME_PRISON_BONUS: float = 0.0 # Défaut = 0.0
    CLOSEOUT_BONUS: float = 0.0 # Défaut = 0.0

    # --- Pénalités spécifiques Phase / Course ---
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR: float = 0.0 # Défaut = 0.0
    ENDGAME_BACK_CHECKER_PENALTY_FACTOR: float = 0.0 # Défaut = 0.0
    ENDGAME_STRAGGLER_PENALTY_FACTOR: float = 0.0 # Défaut = 0.0

    # --- Mode Course ---
    RACE_MODE_FACTOR: float = 1.0 # Défaut = 1.0 (pas d'effet)
    RACE_MODE_OTHER_FACTOR_REDUCTION: float = 1.0 # Défaut = 1.0 (pas de réduction)

# --- Poids différents selon la phase de jeu ---
OPENING_WEIGHTS = HeuristicWeights(
    # --- Facteurs de base ---
    PIP_SCORE_FACTOR=0.8, OFF_SCORE_FACTOR=5.0, HIT_BONUS=35.0, BAR_PENALTY=-25.0,
    # --- Structure: Points, Ancres ---
    POINT_BONUS=3.0, HOME_BOARD_POINT_BONUS=2.0, INNER_HOME_POINT_BONUS=1.0,
    ANCHOR_BONUS=8.0, FIVE_POINT_BONUS=5.0, # Important tôt
    # --- Structure: Primes ---
    PRIME_BASE_BONUS=5.0, SIX_PRIME_BONUS=10.0, # Très fort si réussi tôt
    TRAPPED_CHECKER_BONUS=6.0,
    # --- Structure: Distribution ---
    STACKING_PENALTY_FACTOR=-0.5, BUILDER_BONUS=1.5, # Encourager flexibilité
    # --- Blots ---
    DIRECT_SHOT_PENALTY_FACTOR=-1.0, BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.6,
    HIT_FAR_BLOT_PENALTY_MULTIPLIER=1.2, STRATEGIC_BLOT_PENALTY_REDUCTION=0.4,
    # --- Situationnel / Contrôle ---
    MIDGAME_HOME_PRISON_BONUS=15.0, CLOSEOUT_BONUS=5.0, # Moins probable mais utile
    # --- Pénalités spécifiques Phase / Course ---
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=-0.5,
    ENDGAME_BACK_CHECKER_PENALTY_FACTOR=0.0, # Non pertinent
    ENDGAME_STRAGGLER_PENALTY_FACTOR=0.0, # Non pertinent
    # --- Mode Course ---
    RACE_MODE_FACTOR=1.0, RACE_MODE_OTHER_FACTOR_REDUCTION=1.0
)

MIDGAME_WEIGHTS = HeuristicWeights(
    # --- Facteurs de base ---
    PIP_SCORE_FACTOR=1.2, OFF_SCORE_FACTOR=15.0, HIT_BONUS=40.0, BAR_PENALTY=-25.0,
    # --- Structure: Points, Ancres ---
    POINT_BONUS=3.0, HOME_BOARD_POINT_BONUS=5.0, INNER_HOME_POINT_BONUS=3.0,
    ANCHOR_BONUS=3.0, FIVE_POINT_BONUS=6.0, # Toujours crucial
    # --- Structure: Primes ---
    PRIME_BASE_BONUS=5.0, SIX_PRIME_BONUS=15.0, # Valeur maximale ici
    TRAPPED_CHECKER_BONUS=8.0,
    # --- Structure: Distribution ---
    STACKING_PENALTY_FACTOR=-0.7, BUILDER_BONUS=1.0,
    # --- Blots ---
    DIRECT_SHOT_PENALTY_FACTOR=-1.5, BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.5,
    HIT_FAR_BLOT_PENALTY_MULTIPLIER=1.5, STRATEGIC_BLOT_PENALTY_REDUCTION=0.3, # Forte réduction stratégique
    # --- Situationnel / Contrôle ---
    MIDGAME_HOME_PRISON_BONUS=20.0, CLOSEOUT_BONUS=15.0, # Très important
    # --- Pénalités spécifiques Phase / Course ---
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=-0.7,
    ENDGAME_BACK_CHECKER_PENALTY_FACTOR=0.0, # Non pertinent
    ENDGAME_STRAGGLER_PENALTY_FACTOR=0.0, # Non pertinent
    # --- Mode Course ---
    RACE_MODE_FACTOR=1.0, RACE_MODE_OTHER_FACTOR_REDUCTION=1.0,
)

ENDGAME_WEIGHTS = HeuristicWeights(
    # --- Facteurs de base ---
    PIP_SCORE_FACTOR=3.0, OFF_SCORE_FACTOR=30.0, HIT_BONUS=50.0, BAR_PENALTY=-50.0,
    # --- Structure: Points, Ancres ---
    POINT_BONUS=0.5, HOME_BOARD_POINT_BONUS=0.5, INNER_HOME_POINT_BONUS=0.2, # Moins importants
    ANCHOR_BONUS=0.1, FIVE_POINT_BONUS=1.0, # Garde un peu de valeur
    # --- Structure: Primes ---
    PRIME_BASE_BONUS=1.0, SIX_PRIME_BONUS=0.0, # Normalement cassées
    TRAPPED_CHECKER_BONUS=0.5, # Moins pertinent
    # --- Structure: Distribution ---
    STACKING_PENALTY_FACTOR=0.0, BUILDER_BONUS=0.1, # Peu utiles
    # --- Blots ---
    DIRECT_SHOT_PENALTY_FACTOR=-2.5, BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR=0.2,
    HIT_FAR_BLOT_PENALTY_MULTIPLIER=1.8, STRATEGIC_BLOT_PENALTY_REDUCTION=1,
    # --- Situationnel / Contrôle ---
    MIDGAME_HOME_PRISON_BONUS=0.0, # Non pertinent
    CLOSEOUT_BONUS=10.0, # Toujours utile si ça arrive
    # --- Pénalités spécifiques Phase / Course ---
    FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR=0.0, # Remplacé par les suivants
    ENDGAME_BACK_CHECKER_PENALTY_FACTOR=-3.0,
    ENDGAME_STRAGGLER_PENALTY_FACTOR=-3.0, # Ajouté
    # --- Mode Course ---
    RACE_MODE_FACTOR=4.0, RACE_MODE_OTHER_FACTOR_REDUCTION=0.0 # Valeurs agressives
)
PHASE_WEIGHTS = {"OPENING": OPENING_WEIGHTS, "MIDGAME": MIDGAME_WEIGHTS, "ENDGAME": ENDGAME_WEIGHTS}

YELLOW = "\033[33m"; RESET = "\033[0m"
def yellow(text: str) -> str: return f"{YELLOW}{text}{RESET}"

class BackgammonGame:
    """ Gère l'état du jeu, les règles, l'évaluation. """
    OPENING_EXIT_TOTAL_PIP_THRESHOLD = 280 # Seuil pip pour passer de Opening à Midgame

    def __init__(self, human_player=None, initial_game=True):
        """ Initialise le plateau et l'état du jeu. """
        self.board: BoardArr = np.array([
             2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, # Flèches 1-12
            -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2  # Flèches 13-24
        ], dtype=np.int8)
        self.white_bar = 0
        self.black_bar = 0
        self.white_off = 0
        self.black_off = 0
        self.winner = None
        self.current_player = 'w' # Blanc commence toujours
        self.dice = []
        self.available_moves = []
        self.human_player = human_player
        self.ai_player = 'b' if human_player == 'w' else ('w' if human_player == 'b' else None)
        self.current_phase = self.determine_game_phase()
        self.white_last_turn_sequence = [] # Séquence (src, dst) du dernier tour Blanc
        self.black_last_turn_sequence = [] # Séquence (src, dst) du dernier tour Noir

    def copy(self):
        """ Crée une copie profonde pour la simulation IA. """
        new_game = object.__new__(BackgammonGame)
        new_game.board = self.board.copy()  # Copie profonde pour numpy array
        new_game.white_bar = self.white_bar
        new_game.black_bar = self.black_bar
        new_game.white_off = self.white_off
        new_game.black_off = self.black_off
        new_game.winner = self.winner
        new_game.current_player = self.current_player
        new_game.dice = list(self.dice)
        new_game.available_moves = list(self.available_moves)
        new_game.human_player = self.human_player
        new_game.ai_player = self.ai_player
        new_game.current_phase = self.current_phase
        new_game.white_last_turn_sequence = list(self.white_last_turn_sequence)
        new_game.black_last_turn_sequence = list(self.black_last_turn_sequence)
        return new_game

    def calculate_pip(self, player: str) -> int:
        """ Calcule le pip count pour un joueur (utilise Cython si possible). """
        try:
            # Utilise la fonction Cython optimisée si disponible
            return calculate_pip_fast(ord(player), self.board, self.white_bar, self.black_bar)
        except Exception as e:
            print(f"ERREUR pendant l'appel Cython calculate_pip: {e}")
            raise e # Propage l'erreur

    def is_game_over(self):
        """ Vérifie si la partie est terminée. """
        if self.white_off >= NUM_CHECKERS:
            self.winner = 'w'; return True
        if self.black_off >= NUM_CHECKERS:
            self.winner = 'b'; return True
        return False

    def board_tuple(self):
        """ Retourne une représentation hachable de l'état clé du jeu. """
        return (
            tuple(self.board),
            self.white_bar, self.black_bar,
            self.white_off, self.black_off,
            self.current_player
        )

    def get_total_checker_count(self):
        """ Compte tous les pions pour validation (devrait être 15 par joueur). """
        w_board = sum(c for c in self.board if c > 0)
        b_board = sum(abs(c) for c in self.board if c < 0)
        w_total = w_board + self.white_bar + self.white_off
        b_total = b_board + self.black_bar + self.black_off
        return w_total, b_total

    def parse_move(self, move_str, current_player_hint=None):
        """ Analyse l'entrée 'src/dst' en (joueur, src, dst). """
        try:
            move_str = move_str.lower().strip()
            if '/' not in move_str: return None, None, None
            src_str, dst_str = move_str.split('/', 1)
            player = current_player_hint
            src = 'bar' if src_str == 'bar' else (int(src_str) if src_str.isdigit() and 1 <= int(src_str) <= 24 else None)
            dst = 'off' if dst_str == 'off' else (int(dst_str) if dst_str.isdigit() and 1 <= int(dst_str) <= 24 else None)

            if src is None or dst is None: return None, None, None
            if src == 'bar' and dst == 'off': return None, None, None
            if isinstance(src, int) and isinstance(dst, int) and src == dst: return None, None, None

            return player, src, dst
        except Exception:
            return None, None, None

    def _check_all_pieces_home(self, player, game_state):
        """ Vérifie si tous les pions du joueur sont dans son jan intérieur ou sortis. """
        if check_all_pieces_home_cy is not None:
            try:
                # Utilise Cython
                return check_all_pieces_home_cy(
                    ord(player), game_state.board,
                    game_state.white_bar, game_state.black_bar)
            except Exception as e:
                print(f"Warning: Cython check_all_pieces_home échoué: {e}. Fallback Python.")
                # Continue avec Python si Cython échoue

        # --- Fallback Python ---
        p_sign = 1 if player == 'w' else -1
        bar_count = game_state.white_bar if player == 'w' else game_state.black_bar
        if bar_count > 0: return False
        outside_range = range(1, 19) if player == 'w' else range(7, 25) # Points hors du jan intérieur
        for pos in outside_range:
            if game_state.board[pos - 1] * p_sign > 0: return False
        return True

    def _can_bear_off(self, player, checker_pos, die_value, game_state):
        """ Vérifie si un pion peut légalement sortir avec ce dé (suppose all_home=True). """
        p_sign = 1 if player == 'w' else -1
        board = game_state.board
        home_start, home_end = (19, 24) if player == 'w' else (1, 6)

        if not (home_start <= checker_pos <= home_end): return False # Sanity check

        target_point_for_exact_off = 25 if player == 'w' else 0
        required_dist = abs(target_point_for_exact_off - checker_pos)

        # 1. Sortie exacte
        if die_value == required_dist: return True

        # 2. Sortie avec dé supérieur (overshoot)
        is_overshoot = (player == 'w' and checker_pos + die_value > 24) or \
                       (player == 'b' and checker_pos - die_value < 1)
        if is_overshoot and die_value > required_dist:
            # Légal seulement si aucun pion ami n'est plus loin de la sortie
            check_behind_range = range(home_start, checker_pos) if player == 'w' \
                                 else range(checker_pos + 1, home_end + 1)
            for p in check_behind_range:
                if board[p - 1] * p_sign > 0: return False # Trouvé pion derrière -> illégal
            return True # Aucun pion derrière -> légal
        return False

    def _get_die_for_move(self, src, dst, player):
        """ Détermine la valeur nominale du dé pour un coup (src, dst). Ignore overshoot. """
        try:
            if isinstance(src, int) and isinstance(dst, int): # Coup normal
                diff = (dst - src) if player == 'w' else (src - dst)
                return diff if 1 <= diff <= 6 else None
            elif src == 'bar' and isinstance(dst, int): # Entrée depuis la barre
                 die = dst if player == 'w' else (25 - dst)
                 return die if 1 <= die <= 6 else None
            elif isinstance(src, int) and dst == 'off': # Sortie (exacte seulement ici)
                 if not ((player == 'w' and 19 <= src <= 24) or (player == 'b' and 1 <= src <= 6)): return None
                 needed = (25 - src) if player == 'w' else src
                 return needed if 1 <= needed <= 6 else None
            else: return None # Combinaison invalide
        except Exception: return None

    def _get_single_moves_for_die(self, player, die_value, current_state):
        """ Génère tous les coups simples légaux pour un dé donné (utilise Cython si possible). """
        if _get_single_moves_for_die_cy is not None:
            # Utilise Cython directement (plus rapide si ça marche)
            all_home = self._check_all_pieces_home(player, current_state)
            p_bar = current_state.white_bar if player == 'w' else current_state.black_bar
            return _get_single_moves_for_die_cy(
                ord(player), die_value, current_state.board, all_home, p_bar
            )
            # Remarque : pas de try/except ici pour maximiser la vitesse. Si ça plante, ça plante.

        # --- Fallback Python lent ---
        moves: list[tuple[int | str, int | str]] = []
        p_sign = 1 if player == 'w' else -1
        board = current_state.board
        p_bar = current_state.white_bar if player == 'w' else current_state.black_bar

        # 1) Entrée depuis la barre (prioritaire)
        if p_bar > 0:
            entry_point = die_value if player == 'w' else (25 - die_value)
            idx = entry_point - 1
            dest_count = board[idx]
            if dest_count * p_sign >= 0 or abs(dest_count) == 1: # Libre, ami, ou blot adverse
                moves.append(('bar', entry_point))
            return moves # Si sur la barre, seule l'entrée est possible

        # 2) Coups normaux / Sortie (si pas sur la barre)
        all_home = self._check_all_pieces_home(player, current_state)
        for src, dst in MOVE_TABLE.get(die_value, []): # Utilise la table pré-calculée
            if player == 'w' and dst <= src: continue # Vérifie direction
            if player == 'b' and dst >= src: continue
            src_index = src - 1
            if board[src_index] * p_sign <= 0: continue # Doit avoir un pion à la source

            if 1 <= dst <= 24: # Destination sur le plateau
                dest_index = dst - 1
                dest_count = board[dest_index]
                if dest_count * p_sign >= 0 or abs(dest_count) == 1: # Pas bloqué par >= 2 adverses
                    moves.append((src, dst))
            elif all_home: # Destination 'off' (sortie)
                if self._can_bear_off(player, src, die_value, current_state):
                    moves.append((src, 'off'))
        return moves

    def _get_strictly_playable_dice(self, current_state_obj, dice_list, player):
        """ Détermine quels dés DOIVENT être joués selon les règles (dé le plus fort si choix). """
        if not dice_list: return []
        possible_moves_by_die = {}
        individually_playable_dice = []
        unique_dice = sorted(list(set(dice_list)), reverse=True)

        # Vérifie si chaque dé peut jouer individuellement
        for die in unique_dice:
            moves = self._get_single_moves_for_die(player, die, current_state_obj)
            if moves:
                possible_moves_by_die[die] = moves
                individually_playable_dice.append(die)
        if not individually_playable_dice: return [] # Aucun dé ne peut jouer

        # Règle spéciale pour non-doubles
        is_non_double = len(dice_list) == 2 and dice_list[0] != dice_list[1]
        if is_non_double:
            d1, d2 = dice_list[0], dice_list[1]
            smaller_die, larger_die = min(d1, d2), max(d1, d2)
            can_play_larger = larger_die in individually_playable_dice
            can_play_smaller = smaller_die in individually_playable_dice

            if can_play_larger and not can_play_smaller: return [larger_die] # Doit jouer le plus grand
            if not can_play_larger and can_play_smaller: return [smaller_die] # Doit jouer le plus petit
            if can_play_larger and can_play_smaller:
                # Vérifie si les deux peuvent être joués en séquence
                can_play_both_sequence = False
                # Essai: Grand puis petit
                if larger_die in possible_moves_by_die:
                    for move_lg in possible_moves_by_die[larger_die]:
                        temp_game = current_state_obj.copy()
                        if temp_game.make_move_base_logic(player, move_lg[0], move_lg[1]):
                            if temp_game._get_single_moves_for_die(player, smaller_die, temp_game):
                                can_play_both_sequence = True; break
                # Essai: Petit puis grand (si premier échec)
                if not can_play_both_sequence and smaller_die in possible_moves_by_die:
                     for move_sm in possible_moves_by_die[smaller_die]:
                        temp_game = current_state_obj.copy()
                        if temp_game.make_move_base_logic(player, move_sm[0], move_sm[1]):
                            if temp_game._get_single_moves_for_die(player, larger_die, temp_game):
                                can_play_both_sequence = True; break
                # Si impossible de jouer les deux, mais les deux jouables individuellement -> jouer le plus grand
                if not can_play_both_sequence:
                    return [larger_die]
                else:
                    return [larger_die, smaller_die] # Les deux sont options

        # Doubles, ou cas non-double où la règle ne s'applique pas
        return individually_playable_dice

    def get_legal_actions(self):
        """ Calcule tous les coups *simples* légaux possibles avec les dés actuels. """
        player = self.current_player
        if not self.dice: return []

        # Détermine les dés jouables/obligatoires
        playable_dice_values = self._get_strictly_playable_dice(self, list(self.dice), player)

        # Collecte les coups possibles avec ces dés
        final_moves = set()
        for die_val in playable_dice_values:
             moves_for_this_die = self._get_single_moves_for_die(player, die_val, self)
             final_moves.update(moves_for_this_die)
        return list(final_moves)

    def make_move(self, src, dst):
        """ Applique un coup unique validé (src, dst) à l'état du jeu. """
        player = self.current_player
        if not self.dice: return False # Pas de dés

        # Trouve le dé à utiliser
        die_to_remove = None
        nominal_die = self._get_die_for_move(src, dst, player) # Essai correspondance exacte
        if nominal_die is not None and nominal_die in self.dice:
            die_to_remove = nominal_die
        elif dst == 'off' and isinstance(src, int): # Cas sortie (y compris overshoot)
             needed_dist = (25 - src) if player == 'w' else src
             possible_dice = sorted([d for d in self.dice if d >= needed_dist]) # Dés >= distance
             all_home = self._check_all_pieces_home(player, self)
             if all_home:
                 for potential_die in possible_dice:
                     if self._can_bear_off(player, src, potential_die, self):
                          die_to_remove = potential_die; break # Trouvé le plus petit dé valide

        if die_to_remove is None:
            return False # Coup illégal dans ce contexte

        # Sauvegarde état (au cas où make_move_base_logic échoue)
        board_before = self.board.copy(); bars_before = (self.white_bar, self.black_bar)
        off_before = (self.white_off, self.black_off); dice_before = list(self.dice)

        # Applique la logique de base du mouvement
        move_success = self.make_move_base_logic(player, src, dst)

        if move_success:
            # Met à jour l'état du jeu APRES succès
            try:
                self.dice.remove(die_to_remove) # Retire le dé utilisé
            except ValueError:
                 print(f"ERREUR CRITIQUE: Dé {die_to_remove} absent de {dice_before} après coup {src}/{dst} réussi!")
                 # Tentative de rollback
                 self.board = board_before; self.white_bar, self.black_bar = bars_before
                 self.white_off, self.black_off = off_before; self.dice = dice_before
                 self.available_moves = self.get_legal_actions()
                 return False

            self.available_moves = self.get_legal_actions() # Recalcule coups possibles
            game_over = self.is_game_over() # Vérifie fin de partie

            # Vérification nb pions (sécurité)
            w_final, b_final = self.get_total_checker_count()
            if w_final != NUM_CHECKERS or b_final != NUM_CHECKERS:
                 print(f"!! CRITIQUE: Nb pions invalide après {src}/{dst} -> W:{w_final}, B:{b_final}")
                 self.winner = "ERROR"; return False
            return True # Coup réussi

        else:
            self.dice = dice_before # Restaurer les dés est important
            self.available_moves = self.get_legal_actions() # Recalculer
            return False

    def make_move_base_logic(self, player: str, src, dst) -> bool:
        """ Logique de base: maj plateau, bar, off pour un coup. Sans toucher aux dés. """
        if make_move_base_logic_cy is not None:
            try:
                # Utilise Cython
                src_cy = 0 if src == 'bar' else src
                dst_cy = dst
                success, wb, bb, wo, bo = make_move_base_logic_cy(
                    ord(player), src_cy, dst_cy, self.board,
                    self.white_bar, self.black_bar, self.white_off, self.black_off
                )
                if success:
                    self.white_bar, self.black_bar = wb, bb
                    self.white_off, self.black_off = wo, bo
                return bool(success)
            except Exception as e:
                print(f"Warning: Cython make_move_base_logic échoué: {e}. Fallback Python.")

        # ----------- Fallback Python -----------
        p_sign = 1 if player == 'w' else -1
        opp_bar_attr = 'black_bar' if player == 'w' else 'white_bar'
        p_off_attr = 'white_off' if player == 'w' else 'black_off'

        # Sauvegarde pour rollback
        save_board = self.board.copy(); save_w_bar = self.white_bar; save_b_bar = self.black_bar
        save_w_off = self.white_off; save_b_off = self.black_off

        try:
            # --- 1: Retirer pion de la source ---
            if src == 'bar':
                bar_attr = 'white_bar' if player == 'w' else 'black_bar'
                if getattr(self, bar_attr) == 0: raise ValueError(f"Barre {player} vide")
                setattr(self, bar_attr, getattr(self, bar_attr) - 1)
            else: # Source sur plateau
                src_index = src - 1
                if self.board[src_index] * p_sign <= 0: raise ValueError(f"Pas de pion {player} sur {src}")
                self.board[src_index] -= p_sign

            # --- 2: Placer pion sur destination ---
            if dst == 'off':
                setattr(self, p_off_attr, getattr(self, p_off_attr) + 1)
            else: # Destination sur plateau
                dst_index = dst - 1
                dest_count = self.board[dst_index]
                # Vérifie si case bloquée (>= 2 adverses)
                if dest_count * p_sign < 0 and abs(dest_count) >= 2:
                    raise ValueError(f"Case {dst} bloquée")
                # Vérifie si frappe (exactement 1 adverse)
                if dest_count * p_sign < 0 and abs(dest_count) == 1:
                    self.board[dst_index] = 0 # Enlève pion adverse
                    setattr(self, opp_bar_attr, getattr(self, opp_bar_attr) + 1) # Met sur barre
                    self.board[dst_index] = p_sign # Place pion joueur
                else:
                    # Case vide ou déjà occupée par joueur
                    self.board[dst_index] += p_sign
            return True # Succès

        except ValueError as e:
            # Rollback si erreur
            self.board = save_board; self.white_bar = save_w_bar; self.black_bar = save_b_bar
            self.white_off = save_w_off; self.black_off = save_b_off
            return False # Échec

    def determine_game_phase(self):
        """ Détermine la phase de jeu (Opening, Midgame, Endgame). """
        w_home = self._check_all_pieces_home('w', self)
        b_home = self._check_all_pieces_home('b', self)
        if self.white_off > 0 or self.black_off > 0 or (w_home and b_home):
            phase = 'ENDGAME'
        else:
            total_pip = self.calculate_pip('w') + self.calculate_pip('b')
            phase = 'OPENING' if total_pip > self.OPENING_EXIT_TOTAL_PIP_THRESHOLD else 'MIDGAME'
        self.current_phase = phase
        return phase

    def roll_dice(self):
        """ Lance les dés, met à jour l'état et calcule les coups légaux initiaux. """
        d1, d2 = random.randint(1, 6), random.randint(1, 6)
        self.dice = [d1] * 4 if d1 == d2 else [d1, d2]
        self.available_moves = self.get_legal_actions()
        return self.dice

    def switch_player(self):
        """ Change le joueur courant et réinitialise dés/coups. """
        self.current_player = 'b' if self.current_player == 'w' else 'w'
        self.dice = []
        self.available_moves = []

    def draw_board(self):
        """ Crée une représentation textuelle du plateau. """
        board_template = [
            list("   13 14 15 16 17 18 |BAR| 19 20 21 22 23 24    "),
            list("   +-----------------+---+-------------------+  "),
            list("   |                 |   |                   |  "), # Ligne 2
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Ligne 6
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Ligne 9
            list("   +-----------------+BAR+-------------------+  "), # Ligne 10
            list("   |                 |   |                   |  "), # Ligne 11
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "),
            list("   |                 |   |                   |  "), # Ligne 18
            list("   +-----------------+---+-------------------+  "), # Ligne 19
            list("   12 11 10 09 08 07 |BAR| 06 05 04 03 02 01    ")  # Ligne 20
        ]
        num_board_rows = len(board_template)
        base_width = max(len(line) for line in board_template)
        col_map = { 1: 43,  2: 40,  3: 37,  4: 34,  5: 31,  6: 28, 19: 28, 20: 31, 21: 34, 22: 37, 23: 40, 24: 43, 7: 19,  8: 16,  9: 13, 10: 10, 11:  7, 12:  4, 13:  4, 14:  7, 15: 10, 16: 13, 17: 16, 18: 19 }

        opponent = 'b' if self.current_player == 'w' else 'w'
        last_completed_sequence = self.black_last_turn_sequence if opponent == 'b' else self.white_last_turn_sequence
        HIGHLIGHT_POINTS = {dst for (_, dst) in last_completed_sequence if isinstance(dst, int)}

        bar_col = 23; off_marker_col = 1; max_checkers_display = 5
        board_chars = [list(line) for line in board_template] # Copie modifiable

        # --- Pions sur le plateau ---
        for pos in range(1, 25):
            count = self.board[pos - 1];
            if count == 0: continue
            symbol = 'O' if count > 0 else 'X'; abs_count = abs(count); col = col_map.get(pos)
            start_row = 2 if pos >= 13 else 18; direction = 1 if pos >= 13 else -1

            for i in range(min(abs_count, max_checkers_display)):
                row_idx = start_row + (i * direction)
                if 0 <= row_idx < num_board_rows and 0 <= col < len(board_chars[row_idx]) and board_template[row_idx][col] == ' ':
                       char_to_put = yellow(symbol) if pos in HIGHLIGHT_POINTS else symbol
                       board_chars[row_idx][col] = char_to_put
            if abs_count > max_checkers_display:
                row_idx = start_row + ((max_checkers_display - 1) * direction)
                if 0 <= row_idx < num_board_rows and 0 <= col < len(board_chars[row_idx]):
                    count_char = str(abs_count % 10)
                    char_to_put = yellow(count_char) if pos in HIGHLIGHT_POINTS else count_char
                    board_chars[row_idx][col] = char_to_put

        # --- Pions sur la Barre ---
        for i in range(min(self.white_bar, max_checkers_display)):
            row_idx = 2 + i; board_chars[row_idx][bar_col] = 'O'
        if self.white_bar > max_checkers_display: board_chars[2 + max_checkers_display - 1][bar_col] = str(self.white_bar % 10)
        for i in range(min(self.black_bar, max_checkers_display)):
            row_idx = 18 - i; board_chars[row_idx][bar_col] = 'X'
        if self.black_bar > max_checkers_display: board_chars[18 - (max_checkers_display - 1)][bar_col] = str(self.black_bar % 10)

        # --- Pions Sortis (Off) ---
        for r_clear in range(2, 19): board_chars[r_clear][off_marker_col] = ' '
        for i in range(min(self.white_off, max_checkers_display)): board_chars[2 + i][off_marker_col] = 'O'
        if self.white_off > max_checkers_display: board_chars[2 + max_checkers_display - 1][off_marker_col] = str(self.white_off % 10)
        for i in range(min(self.black_off, max_checkers_display)): board_chars[18 - i][off_marker_col] = 'X'
        if self.black_off > max_checkers_display: board_chars[18 - (max_checkers_display - 1)][off_marker_col] = str(self.black_off % 10)

        # --- Infos sur le côté ---
        dice_faces_large = { 1: ["+-------+", "|       |", "|   o   |", "|       |", "+-------+"], 2: ["+-------+", "| o     |", "|       |", "|     o |", "+-------+"], 3: ["+-------+", "| o     |", "|   o   |", "|     o |", "+-------+"], 4: ["+-------+", "| o   o |", "|       |", "| o   o |", "+-------+"], 5: ["+-------+", "| o   o |", "|   o   |", "| o   o |", "+-------+"], 6: ["+-------+", "| o o o |", "|       |", "| o o o |", "+-------+"] }
        dice_height = 5; dice_width = 9; side_info_col = base_width + 1

        def write_text(text, row_idx, col, max_width=None):
             if not (0 <= row_idx < num_board_rows): return
             if max_width: text = text[:max_width]
             needed_len = col + len(text); current_len = len(board_chars[row_idx])
             if current_len < needed_len: board_chars[row_idx].extend([' '] * (needed_len - current_len))
             for i, char in enumerate(text):
                 if 0 <= col + i < len(board_chars[row_idx]): board_chars[row_idx][col + i] = char

        opp_last_move_str = f"{opponent.upper()} Last:"; formatted_seq = " -"
        if last_completed_sequence: formatted_seq = " " + ", ".join([f"{m[0]}/{m[1]}" for m in last_completed_sequence])
        write_text(opp_last_move_str + formatted_seq, 2, side_info_col, max_width=30)

        player_symbol = 'O' if self.current_player == 'w' else 'X'
        phase_str = f"Turn: {self.current_player.upper()}[{player_symbol}] Ph:{self.current_phase}"
        write_text(phase_str, 3, side_info_col, max_width=30)
        write_text(f"W Off [O]: {self.white_off: >2}", 4, side_info_col, max_width=15)
        write_text(f"B Off [X]: {self.black_off: >2}", 17, side_info_col, max_width=15)

        die1_base_row, die2_base_row = 5, 11; dice_to_draw = list(self.dice); drawn_count = 0
        dice_values = [None, None]
        if dice_to_draw: dice_values[0] = dice_to_draw.pop(0); drawn_count += 1
        if dice_to_draw: dice_values[1] = dice_to_draw.pop(0); drawn_count += 1
        for idx, base_row in enumerate([die1_base_row, die2_base_row]):
            die_val = dice_values[idx]
            face = dice_faces_large.get(die_val) if die_val else [" " * dice_width] * dice_height
            for i in range(dice_height): write_text(face[i], base_row + i, side_info_col, max_width=dice_width)

        more_dice_row = 18; write_text(" " * 20, more_dice_row, side_info_col, max_width=20) # Clear
        if len(self.dice) > drawn_count:
            remaining = ', '.join(map(str, self.dice[drawn_count:])); extra_dice_str = f"({remaining} left)"
            write_text(extra_dice_str, more_dice_row, side_info_col, max_width=20)

        # --- Assemblage final ---
        board_lines = [''.join(row_chars).rstrip() for row_chars in board_chars]
        return '\n'.join(board_lines)


    def evaluate_position_heuristic(self, player_to_evaluate: str, weights: HeuristicWeights) -> float:
        """
        Évalue la position actuelle du jeu avec une heuristique améliorée,
        intégrant plusieurs nouvelles fonctionnalités.
        """
        opp = 'b' if player_to_evaluate == 'w' else 'w'
        p_sign = 1 if player_to_evaluate == 'w' else -1
        o_sign = -p_sign
        board = self.board  # Accès direct

        # --- 1. Calculs de base indépendants du scan détaillé ---
        p_pip = self.calculate_pip(player_to_evaluate)
        o_pip = self.calculate_pip(opp)
        pip_score = (o_pip - p_pip) * weights.PIP_SCORE_FACTOR

        p_off = self.white_off if player_to_evaluate == 'w' else self.black_off
        o_off = self.black_off if player_to_evaluate == 'w' else self.white_off
        off_score = (p_off - o_off) * weights.OFF_SCORE_FACTOR

        p_bar = self.white_bar if player_to_evaluate == 'w' else self.black_bar
        o_bar = self.black_bar if player_to_evaluate == 'w' else self.white_bar
        bar_penalty = p_bar * weights.BAR_PENALTY
        hit_bonus = o_bar * weights.HIT_BONUS

        # --- 2. Initialisation des accumulateurs pour le scan ---
        point_bonus_total = 0.0  # Bonus pour chaque point fait
        home_point_bonus_total = 0.0  # Bonus points dans son jan
        inner_home_bonus_total = 0.0  # Bonus points dans jan profond
        anchor_bonus_total = 0.0  # Bonus points dans jan adverse
        builder_bonus_total = 0.0  # Bonus points faits sur cases constructeur
        five_point_bonus_val = 0.0  # Bonus spécifique pour le point 5 / 20
        stacking_penalty_total = 0.0  # Pénalité pour empilement excessif
        made_points_mask = [0] * 24  # Pour calcul des primes plus tard
        player_blot_indices = []  # Pour calcul pénalité blots plus tard

        # Définir les points stratégiques (Sets pour recherche rapide O(1))
        builder_points_w = {7, 8, 9, 10, 11, 13}
        builder_points_b = {18, 17, 16, 15, 14, 12}
        player_builder_points = builder_points_w if player_to_evaluate == 'w' else builder_points_b
        anchor_points_w = {1, 2, 3, 4, 5, 6}  # Jan Noir (ancres potentielles pour Blanc)
        anchor_points_b = {19, 20, 21, 22, 23, 24}  # Jan Blanc (ancres potentielles pour Noir)
        player_potential_anchor_points = anchor_points_w if player_to_evaluate == 'w' else anchor_points_b

        # --- 3. Scan unique du plateau (indices 0-23) ---
        for i in range(24):
            pos = i + 1  # Position 1-24
            count = board[i]
            player_checker_count = count * p_sign  # Positif si joueur, négatif si opp, 0 si vide

            # --- A) Point fait par le joueur (>= 2 pions) ---
            if player_checker_count >= 2:
                made_points_mask[i] = 1  # Marque le point comme fait

                point_bonus_total += weights.POINT_BONUS  # Bonus de base

                # Vérifications basées sur la localisation du point
                is_player_home_board = (player_to_evaluate == 'w' and 19 <= pos <= 24) or \
                                       (player_to_evaluate == 'b' and 1 <= pos <= 6)
                is_opponent_home_board = (player_to_evaluate == 'w' and 1 <= pos <= 6) or \
                                         (player_to_evaluate == 'b' and 19 <= pos <= 24)

                if is_player_home_board:
                    home_point_bonus_total += weights.HOME_BOARD_POINT_BONUS
                    is_player_inner_home = (player_to_evaluate == 'w' and 22 <= pos <= 24) or \
                                           (player_to_evaluate == 'b' and 1 <= pos <= 3)
                    if is_player_inner_home:
                        inner_home_bonus_total += weights.INNER_HOME_POINT_BONUS

                    # Bonus spécifique point 5 / 20 (si DANS le jan)
                    is_player_five_point = (player_to_evaluate == 'w' and pos == 5) or \
                                           (player_to_evaluate == 'b' and pos == 20)

                # Bonus spécifique point 5 / 20 (testé indépendamment de home board)
                is_player_five_point = (player_to_evaluate == 'w' and pos == 5) or \
                                       (player_to_evaluate == 'b' and pos == 20)
                if is_player_five_point and hasattr(weights, 'FIVE_POINT_BONUS'):
                    five_point_bonus_val += weights.FIVE_POINT_BONUS

                if is_opponent_home_board:  # Ancre (point dans jan adverse)
                    anchor_bonus_total += weights.ANCHOR_BONUS

                if pos in player_builder_points and hasattr(weights, 'BUILDER_BONUS'):  # Point constructeur
                    builder_bonus_total += weights.BUILDER_BONUS

                # Pénalité d'empilement
                if player_checker_count > 5 and hasattr(weights,'STACKING_PENALTY_FACTOR') and weights.STACKING_PENALTY_FACTOR != 0.0:
                    stacking_penalty_total += (player_checker_count - 5) * weights.STACKING_PENALTY_FACTOR

            # --- B) Blot du joueur (exactement 1 pion) ---
            elif player_checker_count == 1:
                player_blot_indices.append(i)  # Ajoute l'INDEX (0-23)

        # --- 4. Calcul des pénalités pour Blots (post-scan) ---
        blot_penalty_total = 0.0
        if player_blot_indices:
            opponent_checker_indices = {idx for idx, count in enumerate(board) if count * o_sign > 0}
            opp_bar_count = o_bar  # Réutilise la variable

            for blot_idx in player_blot_indices:
                blot_pos = blot_idx + 1
                direct_shots_on_blot = 0

                # Calcul tirs directs (plateau + barre)
                for shot_dist in range(1, 7):
                    shooter_pos = blot_pos + (shot_dist * p_sign)
                    shooter_idx = shooter_pos - 1
                    if 0 <= shooter_idx < 24 and shooter_idx in opponent_checker_indices:
                        direct_shots_on_blot += abs(board[shooter_idx])
                if opp_bar_count > 0:
                    entry_die_needed = blot_pos if player_to_evaluate == 'b' else (25 - blot_pos)
                    if 1 <= entry_die_needed <= 6:
                        direct_shots_on_blot += opp_bar_count

                # Pénalité de base
                blot_penalty = direct_shots_on_blot * weights.DIRECT_SHOT_PENALTY_FACTOR

                # Majoration si blot "loin" (dans son propre jan)
                is_far_blot = (player_to_evaluate == 'w' and 1 <= blot_pos <= 6) or \
                              (player_to_evaluate == 'b' and 19 <= blot_pos <= 24)
                if is_far_blot and hasattr(weights,
                                           'HIT_FAR_BLOT_PENALTY_MULTIPLIER') and weights.HIT_FAR_BLOT_PENALTY_MULTIPLIER != 1.0:
                    blot_penalty *= weights.HIT_FAR_BLOT_PENALTY_MULTIPLIER

                # Réduction si blot "stratégique" (Constructeur OU Ancre potentielle)
                is_builder_blot = blot_pos in player_builder_points
                is_potential_anchor_blot = blot_pos in player_potential_anchor_points
                if (is_builder_blot or is_potential_anchor_blot) and \
                        hasattr(weights, 'STRATEGIC_BLOT_PENALTY_REDUCTION') and \
                        weights.STRATEGIC_BLOT_PENALTY_REDUCTION != 1.0:
                    blot_penalty *= weights.STRATEGIC_BLOT_PENALTY_REDUCTION

                # Réduction générale si adversaire sur barre
                if opp_bar_count > 0:
                    blot_penalty *= weights.BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR

                blot_penalty_total += blot_penalty

        # --- 5. Calcul des Primes et bonus associés (post-scan) ---
        prime_bonus_total = 0.0
        six_prime_bonus_val = 0.0
        trapped_checker_bonus_total = 0.0
        prime_segments = []
        current_prime_len = 0
        for i in range(24):
            if made_points_mask[i] == 1:
                current_prime_len += 1
            else:
                if current_prime_len >= 4:
                    prime_bonus_total += (current_prime_len - 3) * weights.PRIME_BASE_BONUS
                    if current_prime_len == 6 and hasattr(weights, 'SIX_PRIME_BONUS'):
                        six_prime_bonus_val += weights.SIX_PRIME_BONUS
                    prime_segments.append({'start': i - current_prime_len, 'end': i - 1, 'len': current_prime_len})
                current_prime_len = 0
        # Gérer prime finissant en 24
        if current_prime_len >= 4:
            prime_bonus_total += (current_prime_len - 3) * weights.PRIME_BASE_BONUS
            if current_prime_len == 6 and hasattr(weights, 'SIX_PRIME_BONUS'):
                six_prime_bonus_val += weights.SIX_PRIME_BONUS
            prime_segments.append({'start': 24 - current_prime_len, 'end': 23, 'len': current_prime_len})

        # Bonus pions piégés derrière primes de 5+
        if hasattr(weights, 'TRAPPED_CHECKER_BONUS') and weights.TRAPPED_CHECKER_BONUS != 0 and prime_segments:
            for prime in prime_segments:
                if prime['len'] >= 5:
                    trapped_count = 0
                    trap_zone_indices = range(prime['start']) if player_to_evaluate == 'w' else range(prime['end'] + 1,
                                                                                                      24)
                    for trap_idx in trap_zone_indices:
                        if board[trap_idx] * o_sign > 0: trapped_count += abs(board[trap_idx])
                    trapped_checker_bonus_total += trapped_count * weights.TRAPPED_CHECKER_BONUS

        # --- 6. Bonus (Prison, Closeout) (post-scan) ---
        midgame_prison_bonus = 0.0
        if hasattr(weights, 'MIDGAME_HOME_PRISON_BONUS') and weights.MIDGAME_HOME_PRISON_BONUS != 0.0:
            home_indices = range(18, 24) if player_to_evaluate == 'w' else range(6)
            home_points_made = sum(made_points_mask[i] for i in home_indices)
            if home_points_made >= 3 and o_bar > 0:
                midgame_prison_bonus = weights.MIDGAME_HOME_PRISON_BONUS * o_bar

        closeout_bonus_total = 0.0
        if o_bar > 0 and hasattr(weights, 'CLOSEOUT_BONUS') and weights.CLOSEOUT_BONUS != 0.0:
            opp_entry_indices = range(6) if opp == 'b' else range(18, 24)
            is_closed_out = all(board[idx] * p_sign >= 2 for idx in opp_entry_indices)
            if is_closed_out:
                closeout_bonus_total = weights.CLOSEOUT_BONUS * o_bar

        # --- 7. Pénalités spécifiques de phase (post-scan) ---
        back_checker_penalty_midgame = 0.0
        if hasattr(weights,
                   'FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR') and weights.FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR != 0.0:
            is_far_behind = p_pip > 0 and o_pip > 0 and p_pip >= 1.5 * o_pip
            if is_far_behind:
                back_checker_pip_sum = 0
                back_zone_indices = range(6) if player_to_evaluate == 'w' else range(18, 24)
                for i in back_zone_indices:
                    if board[i] * p_sign > 0:
                        distance = (25 - (i + 1)) if player_to_evaluate == 'w' else (i + 1)
                        back_checker_pip_sum += distance * abs(board[i])
                back_checker_penalty_midgame = back_checker_pip_sum * weights.FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR

        endgame_back_checker_penalty = 0.0
        if hasattr(weights,
                   'ENDGAME_BACK_CHECKER_PENALTY_FACTOR') and weights.ENDGAME_BACK_CHECKER_PENALTY_FACTOR != 0.0:
            total_back_pips = 0
            opp_home_indices = range(6) if opp == 'w' else range(18, 24)  # Jan adverse
            pip_calc_func = (lambda idx: 25 - (idx + 1)) if player_to_evaluate == 'w' else (lambda idx: idx + 1)
            for i in opp_home_indices:
                if board[i] * p_sign > 0:  # Pion du joueur dans jan adverse
                    total_back_pips += abs(board[i]) * pip_calc_func(i)
            endgame_back_checker_penalty = total_back_pips * weights.ENDGAME_BACK_CHECKER_PENALTY_FACTOR

            # --- 7bis. Détection et application du Mode Course ---
            is_race = False
            # Condition simple : Aucun pion sur la barre ET le pion le plus arriéré du joueur
            # est strictement devant le pion le plus arriéré de l'adversaire.
            if p_bar == 0 and o_bar == 0:
                # Trouver l'index du pion le plus arriéré du joueur (plus petit index pour W, plus grand pour B)
                p_last_idx = -1
                if player_to_evaluate == 'w':
                    for i in range(24):
                        if board[i] * p_sign > 0: p_last_idx = i; break
                else:  # player_to_evaluate == 'b':
                    for i in range(23, -1, -1):
                        if board[i] * p_sign > 0: p_last_idx = i; break

                # Trouver l'index du pion le plus arriéré de l'adversaire
                o_last_idx = -1
                if opp == 'w':
                    for i in range(24):
                        if board[i] * o_sign > 0: o_last_idx = i; break
                else:  # opp == 'b':
                    for i in range(23, -1, -1):
                        if board[i] * o_sign > 0: o_last_idx = i; break

                # Vérifier si les deux joueurs ont des pions sur le plateau
                if p_last_idx != -1 and o_last_idx != -1:
                    if player_to_evaluate == 'w':
                        if p_last_idx > o_last_idx:  # Index du blanc > Index du noir => Blanc est devant
                            is_race = True
                    else:  # player_to_evaluate == 'b'
                        if p_last_idx < o_last_idx:  # Index du noir < Index du blanc => Noir est devant
                            is_race = True

            # Ajustement des scores si en mode course
            if is_race and hasattr(weights, 'RACE_MODE_FACTOR') and hasattr(weights,
                                                                            'RACE_MODE_OTHER_FACTOR_REDUCTION'):
                pip_score *= weights.RACE_MODE_FACTOR  # Augmente l'importance du pip

                # Réduit l'importance des autres facteurs
                reduction_factor = weights.RACE_MODE_OTHER_FACTOR_REDUCTION
                point_bonus_total *= reduction_factor
                home_point_bonus_total *= reduction_factor
                inner_home_bonus_total *= reduction_factor
                anchor_bonus_total *= reduction_factor  # Les ancres n'ont plus d'utilité
                builder_bonus_total *= reduction_factor
                five_point_bonus_val *= reduction_factor
                stacking_penalty_total *= reduction_factor  # Moins grave en course pure
                blot_penalty_total *= reduction_factor  # Moins de risque si pas de retour
                prime_bonus_total *= reduction_factor  # Les primes n'ont plus d'utilité de blocage
                six_prime_bonus_val *= reduction_factor
                trapped_checker_bonus_total *= reduction_factor
                midgame_prison_bonus = 0.0
                closeout_bonus_total = 0.0
                back_checker_penalty_midgame *= reduction_factor
                endgame_back_checker_penalty *= reduction_factor

            # --- 7ter. Pénalité pour Traînards en Endgame ---
            endgame_straggler_penalty = 0.0
            current_phase_for_eval = self.determine_game_phase()
            if current_phase_for_eval == 'ENDGAME' and hasattr(weights,
                                                               'ENDGAME_STRAGGLER_PENALTY_FACTOR') and weights.ENDGAME_STRAGGLER_PENALTY_FACTOR != 0.0:
                # Zone "hors du jan intérieur" pour le joueur
                outside_indices = range(0, 18) if player_to_evaluate == 'w' else range(6, 24)
                home_board_entry_pos = 19 if player_to_evaluate == 'w' else 6

                for i in outside_indices:
                    if board[i] * p_sign > 0:  # Trouvé un pion traînard du joueur
                        checker_pos = i + 1
                        num_checkers = abs(board[i])
                        # Calculer la distance minimale pour rentrer dans le jan
                        distance_to_home = abs(home_board_entry_pos - checker_pos)
                        # Appliquer la pénalité (le facteur est déjà négatif)
                        penalty_for_pos = num_checkers * distance_to_home * weights.ENDGAME_STRAGGLER_PENALTY_FACTOR
                        endgame_straggler_penalty += penalty_for_pos
                        # Augmenter la pénalité si très en retard dans la course
                        pip_diff = o_pip - p_pip
                        if pip_diff < -50:
                            endgame_straggler_penalty += penalty_for_pos * 0.5

            # --- 8. Score Total Combiné ---
        total_score = (
                pip_score + off_score + bar_penalty + hit_bonus +  # Scores de base (pip ajusté si race)
                point_bonus_total + home_point_bonus_total +  # Points standard (réduits si race)
                inner_home_bonus_total + anchor_bonus_total +  # Points spécifiques (réduits si race)
                prime_bonus_total + blot_penalty_total +  # Primes et Blots (réduits si race)
                midgame_prison_bonus + trapped_checker_bonus_total +  # Contrôle / Blocage (réduits si race)
                back_checker_penalty_midgame + endgame_back_checker_penalty +  # Pénalités phase (réduites si race)
                builder_bonus_total +
                stacking_penalty_total +
                five_point_bonus_val +
                six_prime_bonus_val +
                closeout_bonus_total
        )

        return total_score

    def compute_zobrist(self) -> np.uint64:
        """ Calcule le hash Zobrist de l'état actuel (utilise Cython si possible). """
        if compute_zobrist_cy_func is not None:
            try:
                # Utilise Cython
                return compute_zobrist_cy_func(
                    self.board, self.white_bar, self.black_bar, self.white_off, self.black_off,
                    ord(self.current_player), RND64, RND_BAR, RND_OFF, RND_TURN )
            except Exception as e:
                print(f"Warning: Cython compute_zobrist_cy échoué: {e}. Fallback Python.")

        # --- Fallback Python ---
        h = np.uint64(0)
        # Pions sur le plateau
        for idx, cnt in enumerate(self.board):
            if cnt != 0:
                col = cnt if cnt > 0 else 15 - cnt # Map count -> col index (1-30)
                if 1 <= col <= 30: h ^= RND64[idx, col]
        # Compteurs Bar/Off (limités à 15 pour l'indexation)
        h ^= RND_BAR[0, min(self.white_bar, 15)]
        h ^= RND_BAR[1, min(self.black_bar, 15)]
        h ^= RND_OFF[0, min(self.white_off, 15)]
        h ^= RND_OFF[1, min(self.black_off, 15)]
        # Tour du joueur
        h ^= RND_TURN[0] if self.current_player == 'w' else RND_TURN[1]
        return h

# --- Fonctions wrappers ---

def _evaluate_position_heuristic(game_state: BackgammonGame, player_to_evaluate: str, weights: HeuristicWeights) -> float:
    """ Wrapper pour l'appel à la méthode statique d'heuristique. """
    # Assurez-vous que BackgammonGame.evaluate_position_heuristic existe bien comme méthode statique
    return BackgammonGame.evaluate_position_heuristic(game_state, player_to_evaluate, weights)

def evaluate_position_hybrid(game_state: "BackgammonGame", player_to_evaluate: str) -> float:
    """ Combine l'heuristique et le réseau neuronal (si disponible). """
    # 1. Score Heuristique
    phase = game_state.determine_game_phase()
    weights = PHASE_WEIGHTS.get(phase, MIDGAME_WEIGHTS) # Default to midgame weights
    h_val = _evaluate_position_heuristic(game_state, player_to_evaluate, weights)

    # 2. Score Réseau Neuronal (si chargé et pondéré)
    nn_val = 0.0
    net = NET # Utilise le modèle global chargé (type MiniMaxHelper)
    if net is not None and NN_WEIGHT > 0.0:
        try:
            with torch.no_grad(): # Pas de calcul de gradient pour l'inférence
                # Obtenir le tenseur du point de vue du joueur
                t = _tensor_for_player(game_state, player_to_evaluate)
                # Le forward de MiniMaxHelper gère l'ajout/retrait de dim batch si nécessaire
                nn_pred = net(t)
                nn_val = nn_pred.item() # Récupère score scalaire
        except Exception as e:
            print(f"Erreur évaluation NN: {e}")
            # Fallback heuristique si NN échoue
            return h_val

    # 3. Combinaison pondérée
    combined_score = (1.0 - NN_WEIGHT) * h_val + NN_WEIGHT * nn_val
    return combined_score

# --- Recherche Minimax ---

def _get_random_dice_samples() -> List[Tuple[int, int]]:
    """ Génère une liste de lancers de dés aléatoires pour expectiminimax. """
    return [(random.randint(1, 6), random.randint(1, 6)) for _ in range(NUM_DICE_SAMPLES)]

def _flush_batch():
    """ Traite le batch d'évaluations NN en attente. """
    global BATCH_BUF, BATCH_KEYS, TT
    if not BATCH_BUF or NET is None: return
    try:
        tensors = [_tensor_for_player(g, p) for g, p in BATCH_BUF]
        tensor_batch = torch.stack(tensors) # Crée batch tensor (N, 54)
        with torch.no_grad():
            preds = NET(tensor_batch) # Prédictions (N,) car forward squeeze la dim 1
            preds_list = preds.tolist() # Liste de scores (N,)
        # Stocke résultats dans la Table de Transposition
        for k, v in zip(BATCH_KEYS, preds_list): TT[k] = v
    except Exception as e:
        print(f"Erreur flush batch NN: {e}")
    finally:
        BATCH_BUF.clear(); BATCH_KEYS.clear() # Vide les buffers

def zobrist(g: "BackgammonGame") -> int:
    """ Calcule et retourne le hash Zobrist comme entier. """
    return int(g.compute_zobrist())


def minimax(game: "BackgammonGame", depth: int,
            player_to_move: str, maximizing_player: str,
            alpha: float, beta: float) -> float:
    """
    Recherche Expectiminimax avec élagage alpha-beta (partiel), TT, et eval hybride.
    Retourne le score espéré du point de vue de `maximizing_player`.
    """
    # --- Lookup Table de Transposition ---
    key = (zobrist(game), depth, player_to_move)
    cached_value = TT.get(key)
    if cached_value is not None:
        return cached_value # Résultat déjà calculé

    # --- État Terminal ---
    if game.is_game_over():
        if game.winner == maximizing_player: return float("inf")
        elif game.winner is not None: return float("-inf")
        else: return 0.0 # Nul?

    # --- Profondeur Max / Feuille ---
    if depth == 0:
        val = evaluate_position_hybrid(game, maximizing_player) # Évaluation hybride
        TT[key] = val # Stocke l'évaluation
        return val

    # --- Étape Récursive ---
    is_maximizing_node = (player_to_move == maximizing_player)
    next_player = 'b' if player_to_move == 'w' else 'w'

    # --- Nœud Chance (Moyenne sur lancers de dés simulés) ---
    accumulated_score = 0.0; num_samples_processed = 0
    sampled_dice_rolls = _get_random_dice_samples()

    for d1, d2 in sampled_dice_rolls:
        dice_for_turn = (d1, d1, d1, d1) if d1 == d2 else (d1, d2)
        # Génère tous les états possibles après ce lancer
        # Utiliser une copie du jeu pour générer les états sans affecter l'état actuel
        game_copy_for_generation = game.copy()
        possible_outcomes = generate_possible_next_states_with_sequences(
            game_copy_for_generation, dice_for_turn, player_to_move)

        best_eval_for_this_roll = float("-inf") if is_maximizing_node else float("inf")
        # Vérifie si aucun coup n'était possible pour CE lancer spécifique
        no_move_possible = (not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]))

        if no_move_possible: # Si aucun coup possible -> passe son tour
             # Évalue l'état actuel (game) mais du point de vue de l'adversaire (next_player)
             # et à une profondeur moindre.
             eval_score = minimax(game, depth - 1, next_player, maximizing_player, alpha, beta)
             best_eval_for_this_roll = eval_score
        else:
            # Itère sur chaque état résultant possible
            for next_state, _ in possible_outcomes:
                # Appel récursif sur l'état *après* le coup
                eval_score = minimax(next_state, depth - 1, next_player, maximizing_player, alpha, beta)

                # Mise à jour Min/Max et Alpha/Beta
                if is_maximizing_node:
                    best_eval_for_this_roll = max(best_eval_for_this_roll, eval_score)
                    alpha = max(alpha, best_eval_for_this_roll)
                else: # Minimizing node
                    best_eval_for_this_roll = min(best_eval_for_this_roll, eval_score)
                    beta = min(beta, best_eval_for_this_roll)

                # Élagage Alpha-Beta pour CE lancer de dé
                if beta <= alpha:
                    break # Arrête d'évaluer les autres séquences pour ce lancer

        # Accumule le score pour ce lancer (attention aux infinis)
        # S'assure de compter le sample même si le score est infini ou si aucun coup n'était possible
        if best_eval_for_this_roll != float("-inf") or best_eval_for_this_roll != float("inf"):
             # S'il n'y a pas eu de coup, best_eval_for_this_roll est le score du minimax(depth-1)
             accumulated_score += best_eval_for_this_roll
             num_samples_processed += 1
        # Gérer le cas où un score infini est retourné (pruning ou état terminal)
        elif best_eval_for_this_roll in [float("-inf"), float("inf")]:
             accumulated_score += best_eval_for_this_roll
             num_samples_processed += 1
        # Note : Il pourrait manquer un cas si no_move_possible=True ET le score retourné est infini,
        # mais la logique ci-dessus devrait le couvrir.

    # Calcul de la moyenne (score espéré)
    avg_score = 0.0
    if num_samples_processed > 0:
        # Gérer les cas où la somme accumulée est infinie
        if accumulated_score == float("inf"): avg_score = float("inf")
        elif accumulated_score == float("-inf"): avg_score = float("-inf")
        else: avg_score = accumulated_score / num_samples_processed
    # else: avg_score reste 0.0 (ne devrait pas arriver si NUM_DICE_SAMPLES > 0)

    TT[key] = avg_score # Stocke le résultat moyen
    return avg_score

def generate_possible_next_states_with_sequences(
        current_game_state: BackgammonGame,
        dice_tuple: tuple, # Ex: (6, 3) ou (5, 5, 5, 5)
        player: str
    ) -> List[Tuple[BackgammonGame, List[Tuple[int | str, int | str]]]]:
    """
    Explore toutes les séquences de coups possibles pour un lancer via DFS.
    Retourne [(état_final_unique, sequence_de_coups)].
    Gère les cas où tous les dés ne peuvent pas être joués et la règle du dé fort.
    """
    results: Dict[tuple, Tuple[BackgammonGame, list]] = {} # état_plateau -> (état_jeu, séquence)
    initial_state_key = current_game_state.board_tuple()
    max_moves = len(dice_tuple)

    # --- Fonction DFS récursive ---
    def find_sequences(start_state: BackgammonGame, dice_remaining: List[int], current_sequence: List[Tuple[int | str, int | str]]):
        # Recalcule les coups possibles DANS l'état actuel de la simulation
        temp_game_state_for_moves = start_state.copy()
        temp_game_state_for_moves.dice = list(dice_remaining) # Met les dés restants pour get_legal_actions
        current_legal_singles = temp_game_state_for_moves.get_legal_actions()

        # Cas de base: Plus de dés ou plus de coups légaux
        if not dice_remaining or not current_legal_singles:
            final_state_key = start_state.board_tuple()
            if final_state_key not in results:
                 final_state_copy = start_state.copy()
                 final_state_copy.dice = [] # Nettoyer pour l'état final retourné
                 final_state_copy.available_moves = []
                 results[final_state_key] = (final_state_copy, current_sequence)
            return

        # Étape récursive: Essaye chaque coup légal possible
        played_move_in_iteration = False
        processed_singles_this_level = set()
        current_dice_to_try = list(dice_remaining)

        for die in current_dice_to_try:
            moves_for_this_die = start_state._get_single_moves_for_die(player, die, start_state)
            for move in moves_for_this_die:
                move_key = (move[0], move[1], die)
                if move in current_legal_singles and move_key not in processed_singles_this_level:
                    next_state_candidate = start_state.copy()
                    if next_state_candidate.make_move_base_logic(player, move[0], move[1]):
                         played_move_in_iteration = True
                         processed_singles_this_level.add(move_key)
                         next_dice = list(dice_remaining)
                         try:
                             next_dice.remove(die)
                             find_sequences(next_state_candidate, next_dice, current_sequence + [move])
                         except ValueError:
                             print(f"Erreur interne DFS: Dé {die} non trouvé dans {dice_remaining} pour coup {move}")
                             continue
        # Gestion fin de chemin si aucun coup joué dans cet appel
        if not played_move_in_iteration and dice_remaining:
            final_state_key = start_state.board_tuple()
            if final_state_key not in results:
                 final_state_copy = start_state.copy(); final_state_copy.dice = []; final_state_copy.available_moves = []
                 results[final_state_key] = (final_state_copy, current_sequence)

    # --- Lancement du DFS ---
    sim_game_initial = current_game_state.copy()
    sim_game_initial.dice = list(dice_tuple)
    sim_game_initial.available_moves = sim_game_initial.get_legal_actions()
    if not sim_game_initial.available_moves:
        no_move_state = current_game_state.copy(); no_move_state.dice = []; no_move_state.available_moves = []
        return [(no_move_state, [])]
    find_sequences(sim_game_initial, list(dice_tuple), [])

    # --- Filtrage des Résultats ---
    if not results: no_move_state = current_game_state.copy(); no_move_state.dice = []; no_move_state.available_moves = []; return [(no_move_state, [])]
    max_moves_made = max(len(seq) for _, seq in results.values()) if results else 0
    if max_moves_made == 0 and len(results) == 1 and initial_state_key in results: no_move_state = current_game_state.copy(); no_move_state.dice = []; no_move_state.available_moves = []; return [(no_move_state, [])]
    required_moves = max_moves_made
    is_non_double = len(dice_tuple) == 2 and dice_tuple[0] != dice_tuple[1]
    if is_non_double:
         d1, d2 = dice_tuple[0], dice_tuple[1]; larger_die, smaller_die = max(d1, d2), min(d1, d2)
         larger_moves = sim_game_initial._get_single_moves_for_die(player, larger_die, sim_game_initial) # Check initial state
         smaller_moves = sim_game_initial._get_single_moves_for_die(player, smaller_die, sim_game_initial) # Check initial state
         if larger_moves and not smaller_moves: required_moves = 1
         elif not larger_moves and smaller_moves: required_moves = 1
         elif larger_moves and smaller_moves:
              can_play_both = False; temp_g = sim_game_initial.copy() # Check from initial state
              # Test Grand puis Petit
              for m_lg in larger_moves:
                  temp_g1 = temp_g.copy()
                  if temp_g1.make_move_base_logic(player, m_lg[0], m_lg[1]):
                      # Check moves possible AFTER the first move
                      temp_g1.dice=[smaller_die] # Temporarily set dice for check
                      if temp_g1._get_single_moves_for_die(player, smaller_die, temp_g1):
                          can_play_both = True
                          break # *** BREAK ICI EST CORRECT ***
              # Test Petit puis Grand (si besoin)
              if not can_play_both:
                  for m_sm in smaller_moves:
                      temp_g1 = temp_g.copy()
                      if temp_g1.make_move_base_logic(player, m_sm[0], m_sm[1]):
                           # Check moves possible AFTER the first move
                           temp_g1.dice=[larger_die] # Temporarily set dice for check
                           if temp_g1._get_single_moves_for_die(player, larger_die, temp_g1):
                               can_play_both = True
                               break # *** BREAK ICI EST CORRECT ***
              # Si impossible de jouer les deux -> doit jouer le plus grand (seq de 1 coup)
              if not can_play_both:
                   required_moves = 1
                   specific_filtered_outcomes = []
                   for state, seq in results.values():
                       if len(seq) == 1:
                            move_src, move_dst = seq[0]; nominal = sim_game_initial._get_die_for_move(move_src, move_dst, player); die_used = nominal
                            if move_dst == 'off' and nominal is None: # Overshoot logic
                                needed = (25 - move_src) if player == 'w' else move_src; possible_d = sorted([d for d in dice_tuple if d >= needed]); die_used = possible_d[0] if possible_d else None
                            # Vérifie si le dé utilisé pour ce coup unique était le plus grand
                            if die_used == larger_die:
                                specific_filtered_outcomes.append((state, seq))
                   # Si on a trouvé des coups uniques avec le dé fort, on retourne ça
                   # Sinon (très rare?), on laisse le filtre général s'appliquer (qui prendra len=1 aussi)
                   if specific_filtered_outcomes:
                       return specific_filtered_outcomes
                   # else: fallback to general filter below

    # Filtre général: garde les séquences ayant joué le nombre requis de coups
    final_valid_outcomes = [(state, seq) for state, seq in results.values() if len(seq) == required_moves]
    if not final_valid_outcomes and initial_state_key in results and len(results[initial_state_key][1]) == 0: no_move_state = current_game_state.copy(); no_move_state.dice = []; no_move_state.available_moves = []; return [(no_move_state, [])]
    return final_valid_outcomes

# ---------------------------------------------------------------------------
# 9) CHOIX DU COUP IA (Mode Minimax)
# ---------------------------------------------------------------------------
def select_ai_move(game: "BackgammonGame", dice: Tuple[int, ...], ai_player: str):
    """ Sélectionne la meilleure séquence de coups pour l'IA via Minimax hybride. """
    # 1. Génère tous les états finaux valides et séquences pour ce lancer
    possible_outcomes = generate_possible_next_states_with_sequences(game, dice, ai_player)

    # Cas: aucun coup possible
    if not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]):
        no_move_state = game.copy(); no_move_state.dice = []; no_move_state.available_moves = []
        return [], no_move_state # Retourne état inchangé, séquence vide

    # 2. Évalue chaque état résultant via Minimax
    best_score = float("-inf"); optimal_resulting_state = None; optimal_sequence = []
    opponent_player = 'b' if ai_player == 'w' else 'w'
    alpha_init, beta_init = float("-inf"), float("inf")

    for next_state, sequence in possible_outcomes:
        # Évalue l'état APRÈS la séquence IA, du point de vue de l'adversaire (depth-1)
        score_for_state = minimax(
            next_state, MAX_DEPTH - 1, opponent_player, # Adversaire joue ensuite
            ai_player, alpha_init, beta_init # Mais l'IA maximise toujours
        )

        # 3. Choisit l'issue avec le meilleur score
        if score_for_state > best_score:
            best_score = score_for_state; optimal_resulting_state = next_state; optimal_sequence = sequence

    # Fallback si aucune issue choisie
    if optimal_resulting_state is None:
        if possible_outcomes:
             optimal_resulting_state = possible_outcomes[0][0]; optimal_sequence = possible_outcomes[0][1]
        else: # Devrait être capturé plus tôt
             optimal_resulting_state = game.copy(); optimal_sequence = []

    # Flush batch NN final (peut être optionnel si batching non utilisé intensivement)
    _flush_batch()

    # Retourne la meilleure séquence et l'état résultant (dés/coups vidés)
    final_state = optimal_resulting_state.copy()
    final_state.dice = []; final_state.available_moves = []
    return optimal_sequence, final_state

# ---------------------------------------------------------------------------
# 9b) CHOIX DU COUP IA (Mode NN Seulement)
# ---------------------------------------------------------------------------
def select_ai_move_nn_only(game: "BackgammonGame", dice: Tuple[int, ...], ai_player: str):
    """ Sélectionne la meilleure séquence via évaluation NN directe (sans Minimax). """
    # 1. Génère états finaux valides et séquences
    possible_outcomes = generate_possible_next_states_with_sequences(game, dice, ai_player)

    # Cas: aucun coup possible
    if not possible_outcomes or (len(possible_outcomes) == 1 and not possible_outcomes[0][1]):
        no_move_state = game.copy(); no_move_state.dice = []; no_move_state.available_moves = []
        return [], no_move_state

    # 2. Évalue chaque état résultant avec le NN
    best_score = float("-inf"); optimal_resulting_state = None; optimal_sequence = []

    if NET is None: # Vérifie si NN chargé
        print("AI ERREUR: Mode NN_ONLY sans modèle NN chargé.")
        no_move_state = game.copy(); no_move_state.dice = []; no_move_state.available_moves = []
        return [], no_move_state

    for next_state, sequence in possible_outcomes:
        try:
            with torch.no_grad():
                t = _tensor_for_player(next_state, ai_player) # Tenseur pour l'IA
                score_for_state = NET(t).item() # Score NN direct

            # 3. Choisit l'issue avec le meilleur score NN
            if score_for_state > best_score:
                best_score = score_for_state; optimal_resulting_state = next_state; optimal_sequence = sequence
        except Exception as e:
            print(f"AI ERREUR: Échec éval NN pour séquence {sequence}: {e}")
            continue # Ignore cette issue

    # Fallback
    if optimal_resulting_state is None:
        if possible_outcomes:
             optimal_resulting_state = possible_outcomes[0][0]; optimal_sequence = possible_outcomes[0][1]
        else:
             optimal_resulting_state = game.copy(); optimal_sequence = []

    # Retourne meilleure séquence et état résultant (dés/coups vidés)
    final_state = optimal_resulting_state.copy()
    final_state.dice = []; final_state.available_moves = []
    return optimal_sequence, final_state

# ---------------------------------------------------------------------------
# 10) BOUCLE PRINCIPALE DU JEU
# ---------------------------------------------------------------------------
def main_play_vs_ai():
    """ Boucle principale du jeu Humain vs IA hybride. """
    print("\n" + "=" * 30 + " Backgammon vs IA Hybride " + "=" * 30)
    print(f" NN Checkpoint: {NN_CHECKPOINT}, Poids NN: {NN_WEIGHT:.2f}, Mode IA: {AI_MODE}")

    human_player = None
    while human_player not in ['w', 'b']:
        human_player = input("Jouer Blanc ('w'=O, commence) ou Noir ('b'=X)? ").lower().strip()
    ai_player = 'b' if human_player == 'w' else 'w'
    print(f"\nVous êtes Joueur {human_player.upper()} ({'O' if human_player == 'w' else 'X'}). L'IA est {ai_player.upper()}.")
    if AI_MODE == "MINIMAX": print(f"Profondeur IA: {MAX_DEPTH}, Samples Dés: {NUM_DICE_SAMPLES}")
    if CYTHON_OK: print("Optimisations Cython activées.")
    if NET: print("Évaluation Réseau Neuronal activée.")
    else: print("Évaluation Réseau Neuronal désactivée (modèle absent/erreur).")
    print("Entrez coups comme 'src/dst' (ex: 13/7, bar/5, 24/off).")
    input("Appuyez sur Entrée pour commencer...")

    game = BackgammonGame(human_player=human_player)
    pm = PhraseManager("sentences.csv") # Pour les phrases d'ambiance
    game.current_player = 'w' # Blanc commence
    turn_count = 0

    while game.winner is None:
        turn_count += 1
        is_human_turn = (game.current_player == human_player)
        player_name = "Votre Tour" if is_human_turn else "Tour de l'IA"
        player_symbol = 'O' if game.current_player == 'w' else 'X'

        current_phase = game.determine_game_phase()
        os.system('cls' if os.name == 'nt' else 'clear')

        message = f" Tour {turn_count}: {player_name} ({player_symbol}) "
        total_length = 46
        stars = (total_length - len(message)) // 2
        print("\n" + "=" * stars + message + "=" * (total_length - stars - len(message)))

        # --- Lancer de Dés et Affichage ---
        current_dice_roll = game.roll_dice() # Lance et met à jour game.dice/available_moves

        print(game.draw_board()) # Affiche plateau (utilise game.dice et surligne dernier coup adverse)
        pip_h = game.calculate_pip(human_player); pip_ai = game.calculate_pip(ai_player)
        pip_diff = pip_h - pip_ai
        print(f"\n   Phase: {current_phase}")
        print(f"   Vos Pips: {pip_h} | Pips IA: {pip_ai} | Diff: {pip_diff:+} (inférieur = mieux)")
        print(f"   Bar: B={game.white_bar} N={game.black_bar} | Off: B={game.white_off} N={game.black_off}")
        print(f"   Lancer: {current_dice_roll}")

        played_sequence_this_turn = [] # Séquence jouée PENDANT ce tour

        # --- Vérifie si coups possibles ---
        if not game.available_moves:
            print("\n   Aucun coup légal ! Tour passé.")
            time.sleep(1.5)
            # Enregistre séquence vide pour le joueur qui passe
            if game.current_player == 'w': game.white_last_turn_sequence = []
            else: game.black_last_turn_sequence = []
        else:
            # --- Tour du Joueur Humain ---
            if is_human_turn:
                moves_made_count = 0
                original_dice_for_turn = list(game.dice)
                max_moves_possible = len(original_dice_for_turn)

                while game.dice and game.available_moves: # Tant que dés et coups possibles
                    print("\n" + '=' * 46)
                    print(f"   Dés restants: {game.dice}")
                    sorted_available = sorted(game.available_moves, key=lambda m: (str(m[0]), str(m[1])))
                    available_str = ", ".join([f"{s}/{d}" for s, d in sorted_available])
                    print(f"   Coups possibles: {available_str}")
                    move_prompt = f"   Entrez coup {moves_made_count + 1}/{max_moves_possible} (ex: 6/1, bar/5, 23/off) ou 'p' pour passer: "
                    move_input = input(move_prompt).lower().strip()

                    if move_input == 'p': print("   Passe les coups restants..."); time.sleep(1); break

                    _, src, dst = game.parse_move(move_input, game.current_player)
                    if src is None or dst is None: print("   Format invalide."); continue

                    current_move_tuple = (src, dst)
                    if current_move_tuple in game.available_moves:
                        move_successful = game.make_move(src, dst) # Met à jour état, dés, coups

                        if move_successful:
                            played_sequence_this_turn.append(current_move_tuple)
                            moves_made_count += 1
                            # --- Réaffiche après coup réussi ---
                            current_phase = game.determine_game_phase()
                            os.system('cls' if os.name == 'nt' else 'clear')

                            message = f" Tour {turn_count}: Votre Tour ({player_symbol}) - Coup {moves_made_count}/{max_moves_possible} "
                            total_length = 46
                            stars = (total_length - len(message)) // 2
                            print("\n" + "=" * stars + message + "=" * (total_length - stars - len(message)))

                            # Affiche avec la séquence partielle en cours (pour surlignage)
                            temp_display_game = game.copy()
                            if temp_display_game.current_player == 'w': temp_display_game.white_last_turn_sequence = played_sequence_this_turn
                            else: temp_display_game.black_last_turn_sequence = played_sequence_this_turn
                            print(temp_display_game.draw_board())

                            pip_h = game.calculate_pip(human_player); pip_ai = game.calculate_pip(ai_player); pip_diff = pip_h - pip_ai
                            print(f"\n   Phase: {current_phase}"); print(f"   Vos Pips: {pip_h} | Pips IA: {pip_ai} | Diff: {pip_diff:+}")
                            print(f"   Bar: B={game.white_bar} N={game.black_bar} | Off: B={game.white_off} N={game.black_off}")
                            print(f"   Joué: {', '.join([f'{m[0]}/{m[1]}' for m in played_sequence_this_turn])}")

                            if game.winner: break # Fin partie en plein tour ?
                        else:
                            print(f"   ERREUR: Coup {src}/{dst} invalide ou échec exécution."); time.sleep(2); break
                    else:
                        print(f"   Coup '{move_input}' non légal actuellement."); time.sleep(1.5)

                # --- Fin boucle saisie humaine ---
                if not game.winner:
                    if game.dice and not game.available_moves: print("\n   Plus de coups légaux avec les dés restants."); time.sleep(1.5)
                    print("\n   Fin de votre tour.")

            # --- Tour de l'IA ---
            else:
                profiler = cProfile.Profile()
                profiler.enable()

                # Catégorie de phrase basée sur état
                category = "relative égalité"; phase = game.current_phase
                if phase == "OPENING": category = "début de partie"
                elif phase == "ENDGAME": category = "course vers la sortie"
                else: # Midgame
                    pip_ai = game.calculate_pip(ai_player); pip_h = game.calculate_pip(human_player)
                    pip_diff_ai = pip_h - pip_ai # Vue de l'IA
                    ADVANTAGE_THRESHOLD = 20; DISADVANTAGE_THRESHOLD = -20
                    if pip_diff_ai > ADVANTAGE_THRESHOLD: category = "l'IA a l'avantage"
                    elif pip_diff_ai < DISADVANTAGE_THRESHOLD: category = "l'IA est en difficulté"

                thinking_phrase = pm.get(category) # Phrase d'ambiance
                print(f"\n   {thinking_phrase}")

                details_msg = f"   (IA {player_symbol} Mode={AI_MODE}"
                if AI_MODE == "MINIMAX": details_msg += f" Prof={MAX_DEPTH}, Samples={NUM_DICE_SAMPLES}"
                details_msg += ")"
                #print(details_msg)

                ai_start_time = time.time()
                dice_for_ai = tuple(game.dice) # Dés du lancer pour l'IA

                # Choisit fonction IA selon mode
                if AI_MODE == "MINIMAX":
                    chosen_sequence, best_resulting_state = select_ai_move(game, dice_for_ai, ai_player)
                    profiler.disable()
                    if SHOW_STATS:
                        print("-" * 40 + " STATS PERFORMANCE " + "-" * 40)
                        pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)
                        print("-" * 40 + " FIN STATS " + "-" * 40)
                        input(">>> Appuyez sur Entrée...")
                elif AI_MODE == "NN_ONLY":
                    if NET is None:
                        print("AI ERREUR: Mode NN_ONLY mais modèle .pt non chargé!")
                        chosen_sequence, best_resulting_state = [], game.copy()
                        best_resulting_state.dice = []; best_resulting_state.available_moves = []
                    else:
                        chosen_sequence, best_resulting_state = select_ai_move_nn_only(game, dice_for_ai, ai_player)

                ai_end_time = time.time()
                game = best_resulting_state # Applique le résultat du coup IA
                played_sequence_this_turn = chosen_sequence

                # --- Affiche résultat tour IA ---
                current_phase = game.determine_game_phase()
                os.system('cls' if os.name == 'nt' else 'clear')

                message = f" Tour {turn_count}: Tour de l'IA ({player_symbol}) - Terminé "
                total_length = 46
                stars = (total_length - len(message)) // 2
                print("\n" + "=" * stars + message + "=" * (total_length - stars - len(message)))


                # Met à jour séquence pour surlignage au tour suivant
                if ai_player == 'w': game.white_last_turn_sequence = played_sequence_this_turn; game.black_last_turn_sequence = []
                else: game.black_last_turn_sequence = played_sequence_this_turn; game.white_last_turn_sequence = []

                print(game.draw_board()) # Affiche plateau APRES coup IA (surligne son coup)

                pip_h = game.calculate_pip(human_player); pip_ai = game.calculate_pip(ai_player); pip_diff = pip_h - pip_ai
                print(f"\n   Phase: {current_phase}"); print(f"   Vos Pips: {pip_h} | Pips IA: {pip_ai} | Diff: {pip_diff:+}")
                print(f"   Bar: B={game.white_bar} N={game.black_bar} | Off: B={game.white_off} N={game.black_off}")
                print(f"\n   Lancer initial: {dice_for_ai}")
                print(f"   Temps calcul IA: {ai_end_time - ai_start_time:.2f}s")
                if played_sequence_this_turn:
                    seq_str = ", ".join(color(f"{a}/{b}", YEL) for a, b in played_sequence_this_turn)
                    print(f"   L'IA a joué: {seq_str}")
                else: print("   L'IA n'a joué aucun coup (ou passé).")

                if not game.winner: print("\n   Fin du tour de l'IA."); time.sleep(1.5 if played_sequence_this_turn else 1.0)

        # --- Logique Fin de Tour ---
        if game.winner: break # Sort de la boucle principale si gagnant

        # Met à jour séquence officielle pour l'humain (si c'était son tour)
        if is_human_turn:
            if human_player == 'w': game.white_last_turn_sequence = played_sequence_this_turn; game.black_last_turn_sequence = []
            else: game.black_last_turn_sequence = played_sequence_this_turn; game.white_last_turn_sequence = []

        game.switch_player() # Passe au joueur suivant

    # --- Affichage Fin de Partie ---
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "=" * 35 + " FIN DE PARTIE " + "=" * 35)
    print(f"   Durée: {turn_count} tours.")
    print("\n--- État Final du Plateau ---")
    print(game.draw_board())
    pip_h = game.calculate_pip(human_player); pip_ai = game.calculate_pip(ai_player); pip_diff = pip_h - pip_ai
    print(f"\n   Bar Final: B={game.white_bar} N={game.black_bar}")
    print(f"   Off Final: B={game.white_off} N={game.black_off}")
    print(f"   Pips Finaux: Vous={pip_h}  IA={pip_ai}  (Diff: {pip_diff:+})")
    print("\n" + "-" * 80)

    if game.winner == human_player: print(f"   🎉 BRAVO ! Vous ({game.winner.upper()}) avez gagné !")
    elif game.winner == ai_player: print(f"   Dommage ! L'IA ({game.winner.upper()}) a gagné.")
    elif game.winner == "DRAW": print("   Égalité !")
    elif game.winner == "ERROR": print("   Partie terminée sur une ERREUR interne.")
    else: print(f"   Fin de partie inattendue: {game.winner}")

    print("-" * 80 + "\n")
    input("Appuyez sur Entrée pour quitter.")
    _flush_batch() # Dernier flush au cas où

if __name__ == "__main__":
    main_play_vs_ai()
