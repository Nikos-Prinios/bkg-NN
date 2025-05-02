# speedups.pyx
# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

# --- Imports C Standard ---
from libc.stdlib cimport abs
from libc.stdint cimport uint64_t # Pour les types entiers de taille fixe

# --- Imports Cython & NumPy ---
import cython
import numpy as np
cimport numpy as cnp

# --- Initialisation NumPy C-API ---
# Doit être appelé une fois lors de l'import du module
# Mieux vaut le faire ici plutôt que d'espérer qu'il soit fait ailleurs
_ = cnp.import_array() # Assignation à '_' pour éviter warning "unused"

# --- Typedefs pour la lisibilité ---
ctypedef signed char INT8_t      # Pour le board
ctypedef uint64_t    UINT64_t    # Pour les hashs Zobrist


# ==================================================
#   calculate_pip
# ==================================================
@cython.ccall # Annotation optionnelle, mais peut aider certains outils
cpdef int calculate_pip(char player,          # 'w' or 'b'
                        const INT8_t[:] board, # Read-only memory view
                        int white_bar, int black_bar) except? -1:
    """Calculates pip count for a player using Cython."""
    cdef int pip = 0
    cdef int p_bar
    cdef int p_sign
    cdef int pos, idx, cnt, dist

    if player == b'w'[0]:
        p_bar = white_bar
        p_sign = 1
    elif player == b'b'[0]:
        p_bar = black_bar
        p_sign = -1
    else:
        return -1 # Indicate error

    for idx in range(24): # Indices 0..23
        pos = idx + 1
        cnt = board[idx]
        if cnt * p_sign > 0:
            dist = (25 - pos) if p_sign == 1 else pos
            pip += dist * abs(cnt)

    pip += p_bar * 25
    return pip


# ==================================================
#   _check_can_bear_off_cy (Internal helper)
# ==================================================
# PAS de nogil ici, car elle utilise des opérations sur objets Python (bytes)
cdef bint _check_can_bear_off_cy(char player, int checker_pos, int die_value,
                                 const INT8_t[:] board):
    """Cython internal helper to check bear-off validity. Assumes all_home is true."""
    cdef int p_sign, home_start, home_end, target_point_exact, required_dist
    cdef int check_pos, behind_idx
    cdef bint is_overshoot

    if player == b'w'[0]:
        p_sign = 1; home_start = 19; home_end = 24; target_point_exact = 25
    elif player == b'b'[0]: # Utiliser elif pour être propre
        p_sign = -1; home_start = 1; home_end = 6; target_point_exact = 0
    else:
        return False # Invalid player

    if not (home_start <= checker_pos <= home_end):
        return False

    required_dist = abs(target_point_exact - checker_pos)

    if die_value == required_dist:
        return True

    is_overshoot = (p_sign == 1 and checker_pos + die_value > 24) or \
                   (p_sign == -1 and checker_pos - die_value < 1)

    if is_overshoot and die_value > required_dist:
        if p_sign == 1: # White
            for check_pos in range(home_start, checker_pos):
                behind_idx = check_pos - 1
                if board[behind_idx] * p_sign > 0: return False
        else: # Black
             for check_pos in range(checker_pos + 1, home_end + 1):
                behind_idx = check_pos - 1
                if board[behind_idx] * p_sign > 0: return False
        return True

    return False


# ==================================================
#   get_single_moves_for_die
# ==================================================
# PAS de nogil, car retourne une liste Python
cpdef list get_single_moves_for_die(char player,
                                    int die_value,
                                    const INT8_t[:] board,
                                    bint all_home,
                                    int p_bar):
    """Generates list of single legal moves for a player and die."""
    cdef list moves = []
    cdef int p_sign, entry_point, idx, dest_cnt
    cdef int src, dst, src_idx, dst_idx

    if player == b'w'[0]: p_sign = 1
    elif player == b'b'[0]: p_sign = -1
    else: return moves # Return empty list for invalid player

    if p_bar > 0:
        entry_point = die_value if p_sign == 1 else (25 - die_value)
        if 1 <= entry_point <= 24:
            idx = entry_point - 1
            dest_cnt = board[idx]
            if dest_cnt * p_sign >= 0 or abs(dest_cnt) == 1:
                moves.append(('bar', entry_point))
        return moves # Must return immediately if on bar

    for src in range(1, 25): # Points 1..24
        src_idx = src - 1
        if board[src_idx] * p_sign > 0: # Player has checker here
            dst = src + die_value if p_sign == 1 else src - die_value

            if 1 <= dst <= 24: # Destination on board
                dst_idx = dst - 1
                dest_cnt = board[dst_idx]
                if dest_cnt * p_sign >= 0 or abs(dest_cnt) == 1:
                    moves.append((src, dst))
            elif all_home: # Bear-off attempt
                # Appel de l'aide interne (qui a maintenant le GIL)
                if _check_can_bear_off_cy(player, src, die_value, board):
                    moves.append((src, 'off'))
    return moves


# ==================================================
#   make_move_base_logic
# ==================================================
# PAS de nogil, car retourne un tuple Python
cpdef tuple make_move_base_logic(char player,
                                 int src_in, # 0 for 'bar'
                                 object dst_in, # int or string 'off'
                                 INT8_t[:] board, # Modified in place!
                                 int white_bar_in, int black_bar_in,
                                 int white_off_in, int black_off_in):
    """Applies core move logic. Returns tuple (success, wb, bb, wo, bo)."""
    cdef int p_sign, src_idx, dst_idx, dest_cnt, dst
    cdef int wb = white_bar_in, bb = black_bar_in
    cdef int wo = white_off_in, bo = black_off_in
    cdef INT8_t original_src_val = 0
    cdef INT8_t original_dst_val = 0
    cdef bint hit_occurred = False

    if player == b'w'[0]: p_sign = 1
    elif player == b'b'[0]: p_sign = -1
    else: return False, wb, bb, wo, bo

    if src_in == 0: # From bar
        if p_sign == 1:
            if wb == 0: return False, wb, bb, wo, bo
            wb -= 1
        else:
            if bb == 0: return False, wb, bb, wo, bo
            bb -= 1
    else: # From board
        src_idx = src_in - 1
        if not (0 <= src_idx < 24) or board[src_idx] * p_sign <= 0:
            return False, wb, bb, wo, bo
        original_src_val = board[src_idx]
        board[src_idx] -= p_sign

    if isinstance(dst_in, str) and dst_in == 'off': # Bearing off
        if p_sign == 1: wo += 1
        else: bo += 1
        return True, wb, bb, wo, bo
    elif isinstance(dst_in, int): # Moving to board point
        dst = <int>dst_in
        if not (1 <= dst <= 24): # Invalid destination? Rollback source.
            if src_in == 0:
                if p_sign == 1: wb += 1
                else: bb += 1
            else: board[src_idx] = original_src_val
            return False, wb, bb, wo, bo

        dst_idx = dst - 1
        dest_cnt = board[dst_idx]
        original_dst_val = dest_cnt

        if dest_cnt * p_sign < 0 and abs(dest_cnt) >= 2: # Blocked? Rollback source.
            if src_in == 0:
                if p_sign == 1: wb += 1
                else: bb += 1
            else: board[src_idx] = original_src_val
            return False, wb, bb, wo, bo

        if dest_cnt * p_sign < 0: # Hit? (abs(dest_cnt) == 1 implied)
            hit_occurred = True
            board[dst_idx] = 0
            if p_sign == 1: bb += 1
            else: wb += 1

        board[dst_idx] += p_sign
        return True, wb, bb, wo, bo
    else: # Invalid destination type? Rollback source.
        if src_in == 0:
            if p_sign == 1: wb += 1
            else: bb += 1
        else: board[src_idx] = original_src_val
        return False, wb, bb, wo, bo


# ==================================================
#   check_all_pieces_home
# ==================================================
@cython.ccall
# PAS de nogil ici non plus à cause de l'indexation bytes b'w'[0]
cpdef bint check_all_pieces_home(char player,
                                 const INT8_t[:] board,
                                 int white_bar,
                                 int black_bar):
    """Cython version: Checks if all player's pieces are home or off."""
    cdef int p_sign, i
    cdef int bar_count

    if player == b'w'[0]:
        p_sign = 1; bar_count = white_bar
    elif player == b'b'[0]:
        p_sign = -1; bar_count = black_bar
    else:
        return False # Invalid player

    if bar_count > 0: return False

    if p_sign == 1: # White - check points 1 to 18 (indices 0 to 17)
        for i in range(18):
            if board[i] > 0: return False # Check directly p_sign > 0
    else: # Black - check points 7 to 24 (indices 6 to 23)
        for i in range(6, 24):
             if board[i] < 0: return False # Check directly p_sign < 0

    return True


# ==================================================
#   compute_zobrist_cy
# ==================================================
@cython.ccall
# Retirer nogil ici aussi
cpdef UINT64_t compute_zobrist_cy(const INT8_t[:] board,
                                  int white_bar, int black_bar,
                                  int white_off, int black_off,
                                  char player,
                                  const UINT64_t[:,:] rnd64_table,
                                  const UINT64_t[:,:] rnd_bar_table,
                                  const UINT64_t[:,:] rnd_off_table,
                                  const UINT64_t[:]  rnd_turn_table): # <- Plus de nogil
    """Cython version: Computes Zobrist hash. Assumes tables RND sont valides."""
    cdef UINT64_t h = 0
    cdef int idx, cnt, col
    cdef int wb_idx = white_bar, bb_idx = black_bar
    cdef int wo_idx = white_off, bo_idx = black_off

    # Board checkers
    for idx in range(24):
        cnt = board[idx]
        if cnt == 0: continue
        if cnt > 0: col = cnt
        else: col = 15 - cnt # Map -1..-15 to 16..30
        if 1 <= col <= 30:
             # Accès mémoire direct avec les vues mémoire typées
             h ^= rnd64_table[idx, col]

    # Bar counts (clamp index pour sécurité)
    if wb_idx > 15: wb_idx = 15
    elif wb_idx < 0: wb_idx = 0 # Sécurité ajoutée
    if bb_idx > 15: bb_idx = 15
    elif bb_idx < 0: bb_idx = 0
    h ^= rnd_bar_table[0, wb_idx] # Index 0 pour White
    h ^= rnd_bar_table[1, bb_idx] # Index 1 pour Black

    # Off counts (clamp index pour sécurité)
    if wo_idx > 15: wo_idx = 15
    elif wo_idx < 0: wo_idx = 0
    if bo_idx > 15: bo_idx = 15
    elif bo_idx < 0: bo_idx = 0
    h ^= rnd_off_table[0, wo_idx] # Index 0 pour White
    h ^= rnd_off_table[1, bo_idx] # Index 1 pour Black

    # Player turn
    if player == b'w'[0]: # Comparaison OK ici car la fonction entière est nogil
        h ^= rnd_turn_table[0]
    elif player == b'b'[0]:
        h ^= rnd_turn_table[1]
    # else: ignorer si joueur invalide? h reste tel quel.

    return h