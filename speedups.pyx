# distutils: language = c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

# --- Imports C Standard ---
from libc.stdlib cimport abs
from libc.stdint cimport uint64_t # Pour les types entiers de taille fixe
from libc.math cimport fabs # Utiliser fabs pour abs sur les doubles

# --- Imports Cython & NumPy ---
import cython
import numpy as np
cimport numpy as cnp

# --- Initialisation NumPy C-API ---
_ = cnp.import_array()

# --- Typedefs pour la lisibilité ---
ctypedef cnp.int8_t INT8_t      # Pour le board
ctypedef cnp.uint64_t UINT64_t  # Pour les hashs Zobrist
ctypedef cnp.int64_t INT64_t    # Pour Pip count


# ==================================================
#   calculate_pip
# ==================================================

@cython.ccall
cpdef INT64_t calculate_pip(char player,          # 'w' or 'b'
                            const INT8_t[:] board, # Read-only memory view
                            int white_bar, int black_bar) except? -1:
    """Calculates pip count for a player using Cython."""
    cdef INT64_t pip_count = 0 # Utiliser INT64_t pour le type de retour
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

    # Pions sur le plateau
    for idx in range(24): # Indices 0..23
        cnt = board[idx]
        if cnt * p_sign > 0: # Pion du joueur
             # Calcul distance (pips)
            if p_sign == 1: # Blanc
                dist = 24 - idx
            else: # Noir
                dist = idx + 1
            pip_count += dist * abs(cnt)

    # Pions sur la barre
    pip_count += p_bar * 25 # Chaque pion sur barre coûte 25 pips (pour entrer sur 24/1 et sortir)

    return pip_count


# ==================================================
#   _check_can_bear_off_cy (Internal helper)
# ==================================================
cdef bint _check_can_bear_off_cy(char player, int checker_pos, int die_value,
                                 const INT8_t[:] board):
    """Cython internal helper to check bear-off validity. Assumes all_home is true."""
    cdef int p_sign, home_start, home_end, target_point_exact, required_dist
    cdef int check_pos, behind_idx
    cdef bint is_overshoot

    if player == b'w'[0]:
        p_sign = 1; home_start = 19; home_end = 24; target_point_exact = 25
    elif player == b'b'[0]:
        p_sign = -1; home_start = 1; home_end = 6; target_point_exact = 0
    else:
        return False

    if not (home_start <= checker_pos <= home_end):
        return False

    required_dist = abs(target_point_exact - checker_pos)

    # Sortie exacte
    if die_value == required_dist:
        return True

    # Overshoot?
    is_overshoot = (p_sign == 1 and checker_pos + die_value > 24) or \
                   (p_sign == -1 and checker_pos - die_value < 1)

    if is_overshoot and die_value > required_dist:
        # Vérifier s'il y a des pions derrière
        if p_sign == 1: 
            for check_pos in range(home_start, checker_pos):
                behind_idx = check_pos - 1
                if board[behind_idx] > 0: return False # Pion ami derrière
        else: # 
             for check_pos in range(checker_pos + 1, home_end + 1):
                behind_idx = check_pos - 1
                if board[behind_idx] < 0: return False # Pion ami derrière
        return True # Overshoot légal

    return False # Ni exact ni overshoot légal


# ==================================================
#   get_single_moves_for_die
# ==================================================
cpdef list get_single_moves_for_die(char player,
                                    int die_value,
                                    const INT8_t[:] board,
                                    bint all_home,
                                    int p_bar):
    """Generates list of single legal moves for a player and die."""
    cdef list moves = []
    cdef int p_sign, entry_point, idx, dest_cnt_int
    cdef int src, dst, src_idx, dst_idx
    cdef INT8_t dest_cnt # Type du board

    if player == b'w'[0]: p_sign = 1
    elif player == b'b'[0]: p_sign = -1
    else: return moves

    # --- 1. Depuis la barre ---
    if p_bar > 0:
        entry_point = die_value if p_sign == 1 else (25 - die_value)
        if 1 <= entry_point <= 24:
            idx = entry_point - 1
            dest_cnt = board[idx] # Lire la valeur du board
            # Vérifier si case libre, amie, ou blot adverse
            if dest_cnt * p_sign >= 0 or abs(dest_cnt) == 1:
                moves.append(('bar', entry_point))
        return moves # Doit retourner immédiatement si sur la barre

    # --- 2. Depuis le plateau ---
    for src in range(1, 25): # Points 1..24
        src_idx = src - 1
        if board[src_idx] * p_sign > 0: # Le joueur a un pion ici
            # Calculer destination
            dst = src + die_value if p_sign == 1 else src - die_value

            if 1 <= dst <= 24: # Destination sur le plateau
                dst_idx = dst - 1
                dest_cnt = board[dst_idx]
                # Vérifier si case libre, amie, ou blot adverse
                if dest_cnt * p_sign >= 0 or abs(dest_cnt) == 1:
                    moves.append((src, dst))
            elif all_home: # Tentative de sortie (dst est hors plateau)
                # Appel de l'aide interne
                if _check_can_bear_off_cy(player, src, die_value, board):
                    moves.append((src, 'off')) # Utiliser string 'off' pour cohérence Python
    return moves


# ==================================================
#   make_move_base_logic
# ==================================================
cpdef tuple make_move_base_logic(char player,
                                 int src_in, # 0 for 'bar'
                                 object dst_in, # int or string 'off'
                                 INT8_t[:] board, # Modified in place!
                                 int white_bar_in, int black_bar_in,
                                 int white_off_in, int black_off_in):
    """Applies core move logic. Returns tuple (success, wb, bb, wo, bo)."""
    cdef int p_sign, src_idx, dst_idx, dst
    cdef INT8_t dest_cnt # type du board
    cdef int wb = white_bar_in, bb = black_bar_in
    cdef int wo = white_off_in, bo = black_off_in
    cdef INT8_t original_src_val = 0 # Pour rollback

    if player == b'w'[0]: p_sign = 1
    elif player == b'b'[0]: p_sign = -1
    else: return False, wb, bb, wo, bo # Retourner état inchangé

    # --- 1. Retirer de la source ---
    if src_in == 0: # Depuis la barre
        if p_sign == 1: # Blanc
            if wb == 0: return False, wb, bb, wo, bo
            wb -= 1
        else: # Noir
            if bb == 0: return False, wb, bb, wo, bo
            bb -= 1
    else: # Depuis le plateau
        src_idx = src_in - 1
        if not (0 <= src_idx < 24) or board[src_idx] * p_sign <= 0:
            return False, wb, bb, wo, bo
        original_src_val = board[src_idx]
        board[src_idx] -= p_sign

    # --- 2. Placer sur la destination ---
    if isinstance(dst_in, str) and dst_in == 'off': # Sortie
        if p_sign == 1: wo += 1
        else: bo += 1
        return True, wb, bb, wo, bo
    elif isinstance(dst_in, int): # Vers un point du plateau
        dst = <int>dst_in
        if not (1 <= dst <= 24): # Destination invalide ? Rollback source.
            if src_in == 0:
                if p_sign == 1:
                    wb += 1
                else:
                    bb += 1
            else: board[src_idx] = original_src_val
            return False, wb, bb, wo, bo

        dst_idx = dst - 1
        dest_cnt = board[dst_idx]

        # Vérifier si bloqué
        if dest_cnt * p_sign < 0 and abs(dest_cnt) >= 2:
             # Rollback source
            if src_in == 0:
                if p_sign == 1:
                    wb += 1
                else:
                    bb += 1
            else: board[src_idx] = original_src_val
            return False, wb, bb, wo, bo

        # Gérer frappe (Hit)
        if dest_cnt * p_sign < 0:
            board[dst_idx] = 0
            if p_sign == 1: bb += 1
            else: wb += 1

        # Placer le pion du joueur
        board[dst_idx] += p_sign
        return True, wb, bb, wo, bo
    else: # Type de destination invalide ? Rollback source.
        if src_in == 0:
            if p_sign == 1:
                wb += 1
            else:
                bb += 1
        else: board[src_idx] = original_src_val
        return False, wb, bb, wo, bo


# ==================================================
#   check_all_pieces_home
# ==================================================
@cython.ccall
cpdef bint check_all_pieces_home(char player,
                                 const INT8_t[:] board,
                                 int white_bar,
                                 int black_bar):
    """Cython version: Checks if all player's pieces are home or off."""
    cdef int p_sign, i, check_start, check_end
    cdef int bar_count

    if player == b'w'[0]:
        p_sign = 1; bar_count = white_bar
        check_start = 0; check_end = 18 # Vérifier indices 0..17 (points 1..18)
    elif player == b'b'[0]:
        p_sign = -1; bar_count = black_bar
        check_start = 6; check_end = 24 # Vérifier indices 6..23 (points 7..24)
    else:
        return False

    if bar_count > 0: return False

    # Parcourir la zone hors du jan intérieur
    for i in range(check_start, check_end):
        if board[i] * p_sign > 0: # Trouvé un pion du joueur hors zone
            return False

    return True # Aucun pion trouvé hors zone


# ==================================================
#   compute_zobrist_cy
# ==================================================
@cython.ccall
cpdef UINT64_t compute_zobrist_cy(const INT8_t[:] board,
                                  int white_bar, int black_bar,
                                  int white_off, int black_off,
                                  char player,
                                  const UINT64_t[:,:] rnd64_table,
                                  const UINT64_t[:,:] rnd_bar_table,
                                  const UINT64_t[:,:] rnd_off_table,
                                  const UINT64_t[:]  rnd_turn_table):
    """Cython version: Computes Zobrist hash. Assumes tables RND sont valides."""
    cdef UINT64_t h = 0
    cdef int idx, col
    cdef INT8_t cnt # Type du board
    # Clamper les indices pour accéder aux tables RND (taille 16 -> indices 0-15)
    cdef int wb_idx = max(0, min(white_bar, 15))
    cdef int bb_idx = max(0, min(black_bar, 15))
    cdef int wo_idx = max(0, min(white_off, 15))
    cdef int bo_idx = max(0, min(black_off, 15))

    # Pions sur le plateau
    for idx in range(24):
        cnt = board[idx]
        if cnt != 0:
            col = cnt if cnt > 0 else 15 - cnt
            if 1 <= col <= 30:
                h ^= rnd64_table[idx, col]

    # Compteurs Bar/Off (indices déjà clampés)
    h ^= rnd_bar_table[0, wb_idx] # Index 0 pour White Bar
    h ^= rnd_bar_table[1, bb_idx] # Index 1 pour Black Bar
    h ^= rnd_off_table[0, wo_idx] # Index 0 pour White Off
    h ^= rnd_off_table[1, bo_idx] # Index 1 pour Black Off

    # Tour du joueur
    if player == b'w'[0]:
        h ^= rnd_turn_table[0]
    elif player == b'b'[0]:
        h ^= rnd_turn_table[1]

    return h


# ===========================================================================
#  Fonction d'Évaluation Heuristique Cython
# ===========================================================================
@cython.ccall # Export C-callable function
cpdef double calculate_heuristic_score_cy(
    # Données du jeu
    const INT8_t[:] board,    # Read-only memory view
    int white_bar,
    int black_bar,
    int white_off,
    int black_off,
    char player_char, # 'w' or 'b' comme caractère C
    # --- Poids (passés individuellement) ---
    double W_PIP_SCORE_FACTOR, double W_OFF_SCORE_FACTOR, double W_HIT_BONUS,
    double W_BAR_PENALTY, double W_POINT_BONUS, double W_HOME_BOARD_POINT_BONUS,
    double W_INNER_HOME_POINT_BONUS, double W_ANCHOR_BONUS, double W_PRIME_BASE_BONUS,
    double W_DIRECT_SHOT_PENALTY_FACTOR, double W_BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR,
    double W_MIDGAME_HOME_PRISON_BONUS, double W_FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR,
    double W_TRAPPED_CHECKER_BONUS, double W_ENDGAME_BACK_CHECKER_PENALTY_FACTOR
    ) except? -10000.0:

    # ================= Déclarations CDEF =====================
    # --- Variables générales ---
    cdef int p_sign, o_sign
    cdef int p_bar, o_bar, p_off, o_off
    cdef double total_score = 0.0
    cdef int i, pos, player_checker_count
    cdef INT8_t count # type du board
    cdef bint is_player_home_board, is_player_inner_home, is_opponent_home_board
    cdef int opponent_home_start_idx, opponent_home_end_idx
    cdef int player_back_zone_start_idx, player_back_zone_end_idx
    cdef int back_checker_count = 0 # Pour la pénalité endgame simplifiée

    # --- Pip ---
    cdef INT64_t p_pip = 0, o_pip = 0 # Initialiser
    cdef double pip_score = 0.0

    # --- Bar/Hit ---
    cdef double bar_penalty, hit_bonus

    # --- Structure (Points, Blots, Primes, etc.) ---
    cdef double point_bonus_total = 0.0
    cdef double home_point_bonus_total = 0.0
    cdef double inner_home_bonus_total = 0.0
    cdef double anchor_bonus_total = 0.0
    cdef double blot_penalty_total = 0.0
    cdef double trapped_checker_bonus_total = 0.0
    cdef double prime_bonus_total = 0.0
    cdef double midgame_prison_bonus = 0.0
    cdef double back_checker_penalty_midgame = 0.0
    cdef double endgame_back_checker_penalty = 0.0

    # --- Variables pour boucles internes ---
    # Blot calc
    cdef int blot_idx, blot_pos, direct_shots_on_blot, shot_dist, shooter_pos, shooter_idx, entry_die_needed
    cdef double blot_penalty
    cdef bint is_opponent_checker[24] # Tableau C booléen
    cdef int opp_idx

    # Prime calc
    cdef int current_prime_len = 0
    cdef int prime_start_idx = 0, prime_end_idx = 0 # Initialiser
    cdef bint made_points_mask[24] # Tableau C booléen

    # Trapped calc
    cdef int trapped_opp_checker_count
    cdef int trap_idx, trap_zone_start, trap_zone_end

    # Prison calc
    cdef int home_board_start_idx, home_board_end_idx
    cdef int home_points_made_count

    # Far behind calc
    cdef bint is_far_behind
    cdef INT64_t back_checker_pip_sum

    # Endgame penalty calc
    cdef INT64_t total_back_checker_pips_in_opp_home = 0
    cdef int pips_for_this_checker_pos, num_checkers_here
    cdef int distance # Réutiliser pour calcul pip distance

    # ================= Initialisation =====================
    if player_char == b'w'[0]:
        p_sign = 1; o_sign = -1
        p_bar = white_bar; o_bar = black_bar
        p_off = white_off; o_off = black_off
        home_board_start_idx = 18; home_board_end_idx = 24
        opponent_home_start_idx = 0; opponent_home_end_idx = 6
        player_back_zone_start_idx = 0; player_back_zone_end_idx = 6
    elif player_char == b'b'[0]:
        p_sign = -1; o_sign = 1
        p_bar = black_bar; o_bar = white_bar
        p_off = black_off; o_off = white_off
        home_board_start_idx = 0; home_board_end_idx = 6
        opponent_home_start_idx = 18; opponent_home_end_idx = 24
        player_back_zone_start_idx = 18; player_back_zone_end_idx = 24
    else:
        return -10001.0 # Code d'erreur

    # Précalculer positions adverses pour accès rapide dans boucle blot
    for opp_idx in range(24):
        is_opponent_checker[opp_idx] = (board[opp_idx] * o_sign > 0)

    # Initialiser masque points faits
    for i in range(24): made_points_mask[i] = False

    # ================= Calcul des Composantes du Score =====================

    # --- 1. Score Pip (Appel Cython->Cython) ---
    # Utiliser bloc try/except? car calculate_pip a except? -1... bof
    try:
        p_pip = calculate_pip(player_char, board, white_bar, black_bar)
        o_pip = calculate_pip(b'w'[0] if player_char == b'b'[0] else b'b'[0], board, white_bar, black_bar)
        if p_pip == -1 or o_pip == -1: # Gérer erreur de calculate_pip
             pip_score = 0.0 # Ou autre valeur d'erreur
             # Peut-être retourner une erreur ici ?
        else:
            pip_score = (o_pip - p_pip) * W_PIP_SCORE_FACTOR
    except:
         pip_score = 0.0 # Ou retourner erreur
         # return -10002.0


    # --- 2. Score Off ---
    off_score = (p_off - o_off) * W_OFF_SCORE_FACTOR

    # --- 3. Score Bar/Hit ---
    bar_penalty = p_bar * W_BAR_PENALTY # W_BAR_PENALTY est négatif
    hit_bonus = o_bar * W_HIT_BONUS     # W_HIT_BONUS est positif

    # --- 4. Scan Principal du Plateau (Points, Blots, Ancres, Masque Primes) ---
    point_bonus_total = 0.0
    home_point_bonus_total = 0.0
    inner_home_bonus_total = 0.0
    anchor_bonus_total = 0.0
    blot_penalty_total = 0.0

    for i in range(24):
        pos = i + 1
        count = board[i]
        player_checker_count = count * p_sign

        if player_checker_count >= 2: # Point fait
            made_points_mask[i] = True
            point_bonus_total += W_POINT_BONUS

            # Home Board Bonus
            is_player_home_board = (home_board_start_idx <= i < home_board_end_idx)
            if is_player_home_board:
                home_point_bonus_total += W_HOME_BOARD_POINT_BONUS
                # Inner Home Bonus
                is_player_inner_home = (player_char == b'w'[0] and i >= 21) or \
                                       (player_char == b'b'[0] and i <= 2)
                if is_player_inner_home:
                    inner_home_bonus_total += W_INNER_HOME_POINT_BONUS

            # Anchor Bonus
            is_opponent_home_board = (opponent_home_start_idx <= i < opponent_home_end_idx)
            if is_opponent_home_board:
                anchor_bonus_total += W_ANCHOR_BONUS

        elif player_checker_count == 1: # Blot trouvé -> Calculer pénalité
            blot_pos = pos
            direct_shots_on_blot = 0

            # Tirs depuis plateau adverse (utilisation du tableau booléen)
            for shot_dist in range(1, 7):
                shooter_pos = blot_pos + (shot_dist * p_sign) # Position adv.
                shooter_idx = shooter_pos - 1
                if 0 <= shooter_idx < 24 and is_opponent_checker[shooter_idx]:
                     direct_shots_on_blot += abs(board[shooter_idx])

            # Tirs depuis barre adverse
            if o_bar > 0:
                entry_die_needed = blot_pos if player_char == b'b'[0] else (25 - blot_pos)
                if 1 <= entry_die_needed <= 6:
                    direct_shots_on_blot += o_bar

            blot_penalty = direct_shots_on_blot * W_DIRECT_SHOT_PENALTY_FACTOR # Facteur est négatif
            if o_bar > 0:
                blot_penalty *= W_BLOT_PENALTY_REDUCTION_IF_OPP_ON_BAR
            blot_penalty_total += blot_penalty

    # --- 5. Calcul Primes et Pions Piégés ---
    prime_bonus_total = 0.0
    trapped_checker_bonus_total = 0.0
    current_prime_len = 0
    for i in range(24):
        if made_points_mask[i]:
            current_prime_len += 1
        else:
            if current_prime_len >= 4: # Fin d'une prime
                prime_bonus_total += (current_prime_len - 3) * W_PRIME_BASE_BONUS
                # Calcul Pions Piégés
                if current_prime_len >= 5 and W_TRAPPED_CHECKER_BONUS != 0.0:
                    prime_end_idx = i - 1 # Inclusive
                    prime_start_idx = prime_end_idx - current_prime_len + 1 # Inclusive
                    trapped_opp_checker_count = 0
                    if player_char == b'w'[0]:
                        trap_zone_start = 0; trap_zone_end = prime_start_idx
                    else:
                        trap_zone_start = prime_end_idx + 1; trap_zone_end = 24
                    for trap_idx in range(trap_zone_start, trap_zone_end):
                        if board[trap_idx] * o_sign > 0:
                            trapped_opp_checker_count += abs(board[trap_idx])
                    trapped_checker_bonus_total += trapped_opp_checker_count * W_TRAPPED_CHECKER_BONUS
            current_prime_len = 0 # Reset

    # Gérer prime finissant en pos 24
    if current_prime_len >= 4:
        prime_bonus_total += (current_prime_len - 3) * W_PRIME_BASE_BONUS
        if current_prime_len >= 5 and W_TRAPPED_CHECKER_BONUS != 0.0:
            prime_end_idx = 23
            prime_start_idx = prime_end_idx - current_prime_len + 1
            trapped_opp_checker_count = 0
            if player_char == b'w'[0]: # Prime W finissant en 24, piège B avant point 19
                trap_zone_start = 0; trap_zone_end = prime_start_idx
                for trap_idx in range(trap_zone_start, trap_zone_end):
                     if board[trap_idx] * o_sign > 0: # Si pion Noir (o_sign = -1)
                        trapped_opp_checker_count += abs(board[trap_idx])
            # else: Prime B finissant en 24 (impossible car sort par point 1) -> pas de piège
            trapped_checker_bonus_total += trapped_opp_checker_count * W_TRAPPED_CHECKER_BONUS

    # --- 6. Bonus Prison Milieu de Partie ---
    midgame_prison_bonus = 0.0
    if W_MIDGAME_HOME_PRISON_BONUS != 0.0:
         home_points_made_count = 0
         for i in range(home_board_start_idx, home_board_end_idx):
             if made_points_mask[i]:
                 home_points_made_count += 1
         if home_points_made_count >= 3 and o_bar > 0:
             midgame_prison_bonus = W_MIDGAME_HOME_PRISON_BONUS * o_bar

    # --- 7. Pénalité Milieu de Partie si Loin Derrière ---
    back_checker_penalty_midgame = 0.0
    if W_FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR != 0.0 and p_pip != -1 and o_pip != -1:
        is_far_behind = p_pip > 0 and o_pip > 0 and p_pip >= 1.5 * o_pip
        if is_far_behind:
            back_checker_pip_sum = 0
            for i in range(player_back_zone_start_idx, player_back_zone_end_idx):
                if board[i] * p_sign > 0:
                    pos = i + 1
                    distance = (25 - pos) if player_char == b'w'[0] else pos
                    back_checker_pip_sum += distance * abs(board[i])
            # Appliquer la pénalité avec le *-1.0
            back_checker_penalty_midgame = back_checker_pip_sum * W_FAR_BEHIND_BACK_CHECKER_PENALTY_FACTOR * -1.0

    # --- 8. Pénalité Fin de Partie pour Pions Arrières ---
    endgame_back_checker_penalty = 0.0
    if W_ENDGAME_BACK_CHECKER_PENALTY_FACTOR != 0.0: # Le poids lui-même DOIT être négatif

        # Compter les pions du joueur dans le jan adverse
        for i in range(opponent_home_start_idx, opponent_home_end_idx):
            if board[i] * p_sign > 0: # Pion du joueur trouvé
                back_checker_count += abs(board[i])
        endgame_back_checker_penalty = back_checker_count * W_ENDGAME_BACK_CHECKER_PENALTY_FACTOR

    # ================= Score Final =====================
    total_score = (
        pip_score + off_score + bar_penalty + hit_bonus +
        point_bonus_total + home_point_bonus_total + inner_home_bonus_total +
        anchor_bonus_total + prime_bonus_total + blot_penalty_total +
        midgame_prison_bonus + trapped_checker_bonus_total +
        back_checker_penalty_midgame + endgame_back_checker_penalty
    )

    return total_score
# ===========================================================================
