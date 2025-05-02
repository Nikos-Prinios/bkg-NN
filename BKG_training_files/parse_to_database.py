# parse_gnubg.py — full, corrected script
# Standard library imports
import re, json, sys, glob, hashlib, sqlite3
from pathlib import Path
from typing import List, Dict, Any

###############################################################################
#  Constants                                                                  #
###############################################################################
TOTAL_CHECKERS = 15
PLAYER_W, PLAYER_B = 'O', 'X'  # internal white / black symbols
_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

###############################################################################
#  Low‑level helpers                                                          #
###############################################################################

def _file_uid(path: str) -> str:
    """Return a unique id for each export file (stem if timestamped else sha1)."""
    stem = Path(path).stem
    return stem if re.match(r"^\d+_", stem) else hashlib.sha1(stem.encode()).hexdigest()[:16]

###############################################################################
#  Position ID                                                                #
###############################################################################

def _bits_from_id(pid: str) -> List[int]:
    bits = []
    for c in pid:
        v = _BASE64.index(c)
        bits.extend([(v >> i) & 1 for i in range(6)])
    return bits


def _decode_position_id(pid: str) -> Dict[str, Any]:
    """Return board_pts/bar/off decoded from a GNUbg Position ID."""
    b = _bits_from_id(pid)
    counts, i = [], 0
    while len(counts) < 52:
        n = 0
        while i < len(b) and b[i]:
            n += 1; i += 1
        counts.append(n); i += 1
    pts = [0]*24
    for p in range(24):
        pts[p] += counts[23-p]
        pts[p] -= counts[26+23-p]
    bar_b, off_b = counts[24], counts[25]
    bar_w, off_w = counts[50], counts[51]
    return {"board_pts": pts,
            "bar": {PLAYER_B: bar_b, PLAYER_W: bar_w},
            "off": {PLAYER_B: off_b, PLAYER_W: off_w}}

###############################################################################
#  ASCII board diagram                                                        #
###############################################################################

def _decode_board(lines: List[str]) -> Dict[str, Any]:
    """Parse the ASCII diagram; return {} on failure."""
    if not lines:
        return {}
    try:
        top = next(l for l in lines if l.strip().startswith('+') and '-1-' in l)
        bot = next(l for l in reversed(lines) if l.strip().startswith('+'))
    except StopIteration:
        return {}
    cols: Dict[int,int] = {}
    for m in re.finditer(r"(\d+)", top): cols[int(m.group(1))] = m.start(1)
    for m in re.finditer(r"(\d+)", bot): cols[int(m.group(1))] = m.start(1)
    cx, co = [0]*25, [0]*25
    off_b = off_w = 0
    for row in lines:
        if not row.strip().startswith('|'): continue
        for pt,col in cols.items():
            if col < len(row):
                ch = row[col]
                if ch in (PLAYER_B, PLAYER_B.lower()): cx[pt]+=1
                elif ch in (PLAYER_W, PLAYER_W.lower()): co[pt]+=1
        seg = row[row.rfind('|')+1:]
        off_b += seg.count(PLAYER_B)+seg.count(PLAYER_B.lower())
        off_w += seg.count(PLAYER_W)+seg.count(PLAYER_W.lower())
    pts = [cx[i]-co[i] for i in range(1,25)]
    bar_b = TOTAL_CHECKERS - sum(cx[1:]) - off_b
    bar_w = TOTAL_CHECKERS - sum(co[1:]) - off_w
    if (sum(max(0,x) for x in pts)+bar_b+off_b!=TOTAL_CHECKERS or
        sum(max(0,-x) for x in pts)+bar_w+off_w!=TOTAL_CHECKERS):
        return {}
    return {"board_pts": pts,
            "bar": {PLAYER_B: bar_b, PLAYER_W: bar_w},
            "off": {PLAYER_B: off_b, PLAYER_W: off_w}}

###############################################################################
#  Evaluation line                                                            #
###############################################################################

def _parse_eval(lines: List[str], i: int):
    m = re.match(r"\*?\s*(\d+)\.\s+.*?\s{2,}(.+?)\s+Eq\.:\s+([+-]?[0-9],[0-9]+)(?:\s*\(([^)]+)\))?", lines[i])
    if not m:
        return {"raw": lines[i].strip()}, i+1
    rank = int(m.group(1)); move = m.group(2)
    eq = float(m.group(3).replace(',', '.'))
    diff = float(m.group(4).replace(',', '.')) if m.group(4) else 0.0
    probs = {}
    if i+1 < len(lines):
        nums = [float(x.replace(',', '.')) for x in re.findall(r"[+-]?[0-9]+,[0-9]+", lines[i+1])]
        if len(nums) == 6:
            probs = {"player": {"win": nums[0], "gammon": nums[1], "bg": nums[2]},
                     "opponent": {"win": nums[3], "gammon": nums[4], "bg": nums[5]}}
            i += 1
    return {"rank": rank, "move": move, "equity": eq, "equity_diff": diff, "probabilities": probs}, i+1

###############################################################################
#  Parsing helpers                                                            #
###############################################################################

def _skip(lines: List[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i

###############################################################################
#  Move parser                                                                #
###############################################################################

def _parse_move(lines: List[str], i: int, game: int):
    m = re.match(r"Move number (\d+):\s+(\w+) to play (\d+)", lines[i])
    mv: Dict[str, Any] = {"move_number": int(m.group(1)), "player": m.group(2), "dice": m.group(3), "game_id": game}
    i += 1; i = _skip(lines, i)
    # IDs
    if i < len(lines) and 'Position ID' in lines[i]:
        mv['position_id'] = re.search(r"Position ID:\s+(\S+)", lines[i]).group(1)
        mv['gnubg_match_id'] = re.search(r"Match ID\s+:\s+(\S+)", lines[i+1]).group(1)
        i += 2; i = _skip(lines, i)
    # Diagram
    board_lines = []
    while i < len(lines) and re.match(r"\s*[|+^]", lines[i]):
        board_lines.append(lines[i]); i += 1
    mv.update(_decode_board(board_lines) or (_decode_position_id(mv['position_id']) if 'position_id' in mv else {}))
    # Pip counts
    i = _skip(lines, i)
    if i < len(lines) and lines[i].startswith('Pip counts'):
        p = list(map(int, re.findall(r"\d+", lines[i])))
        mv['pip_counts'] = {PLAYER_W: p[0], PLAYER_B: p[1]}
        i += 1; i = _skip(lines, i)
    # Actual move line
    if i < len(lines) and ' moves ' in lines[i]:
        mv['actual_move'] = lines[i].split(' moves ')[1].strip()
        i += 1; i = _skip(lines, i)
    # Evaluations
    evs = []
    while i < len(lines) and not lines[i].startswith('Move number') and not lines[i].startswith('Game statistics'):
        if re.match(r"\*?\s*\d+\. ", lines[i]):
            e, i = _parse_eval(lines, i); evs.append(e); continue
        i += 1
    mv['evaluations'] = evs
    i = _skip(lines, i)
    return mv, i

###############################################################################
#  Statistics parser                                                          #
###############################################################################

def _parse_stats(lines: List[str], i: int):
    key = 'game_statistics' if lines[i].startswith('Game') else 'match_statistics'
    stats: Dict[str, Any] = {}
    i += 1; i = _skip(lines, i); section = None
    while i < len(lines):
        if re.match(r"^(Game|Match) statistics", lines[i]) or lines[i].startswith('Output generated'):
            break
        s = lines[i].strip()
        if not s: i += 1; continue
        if s.endswith('statistics'):
            section = s; stats[section] = {}
        elif section:
            parts = re.split(r"\s{2,}", s)
            if len(parts) >= 3:
                stats[section][parts[0]] = {'gnubg': parts[1], 'nikos': parts[2]}
        i += 1
    return {key: stats}, i

###############################################################################
#  Parse one export file                                                      #
###############################################################################

def parse_gnubg_txt(path: str) -> Dict[str, Any]:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    data: Dict[str, Any] = {'file_uid': _file_uid(path)}
    moves: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    i = 0; game = 1
    while i < len(lines):
        l = lines[i]
        if l.startswith('The score'):
            data['score_line'] = l.strip()
        elif l.startswith('Date:'):
            data['date'] = l.partition('Date:')[2].strip()
        elif re.match(r"Game statistics for game (\d+)", l):
            game = int(re.search(r"game (\d+)", l).group(1)) + 1
        elif l.startswith('Move number'):
            mv, i = _parse_move(lines, i, game); moves.append(mv); continue
        elif re.match(r"^(Game|Match) statistics", l):
            sec, i = _parse_stats(lines, i); stats.update(sec); continue
        i += 1
    data['moves'] = moves; data['statistics'] = stats
    return data

###############################################################################
#  SQLite persistence                                                         #
###############################################################################

def _create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches (
            file_uid        TEXT PRIMARY KEY,
            gnubg_match_id  TEXT,
            score_line      TEXT,
            date            TEXT,
            raw_json        TEXT
        );

        CREATE TABLE IF NOT EXISTS moves (
            file_uid     TEXT,
            game_id      INTEGER,
            move_number  INTEGER,
            player       TEXT,
            dice         TEXT,
            board_pts    TEXT,
            bar_b        INTEGER,
            bar_w        INTEGER,
            off_b        INTEGER,
            off_w        INTEGER,
            pip_b        INTEGER,
            pip_w        INTEGER,
            actual_move  TEXT,
            position_id  TEXT,
            PRIMARY KEY (file_uid, game_id, move_number)
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            file_uid     TEXT,
            game_id      INTEGER,
            move_number  INTEGER,
            rank         INTEGER,
            move         TEXT,
            equity       REAL,
            equity_diff  REAL,
            p_win        REAL,
            p_gammon     REAL,
            p_bg         REAL,
            o_win        REAL,
            o_gammon     REAL,
            o_bg         REAL,
            PRIMARY KEY (file_uid, game_id, move_number, rank)
        );
        """
    )
    conn.commit()


def _insert_match(conn: sqlite3.Connection, parsed: Dict[str, Any]):
    uid = parsed['file_uid']
    gnubg_id = parsed['moves'][0].get('gnubg_match_id') if parsed['moves'] else None
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?)",
        (
            uid,
            gnubg_id,
            parsed.get('score_line'),
            parsed.get('date'),
            json.dumps(parsed, ensure_ascii=False)
        )
    )
    for mv in parsed['moves']:
        cur.execute(
            """INSERT OR IGNORE INTO moves VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid,
                mv['game_id'],
                mv['move_number'],
                mv['player'],
                mv['dice'],
                json.dumps(mv.get('board_pts')),
                mv.get('bar', {}).get(PLAYER_B),
                mv.get('bar', {}).get(PLAYER_W),
                mv.get('off', {}).get(PLAYER_B),
                mv.get('off', {}).get(PLAYER_W),
                mv.get('pip_counts', {}).get(PLAYER_B),
                mv.get('pip_counts', {}).get(PLAYER_W),
                mv.get('actual_move'),
                mv.get('position_id')
            )
        )
        for ev in mv.get('evaluations', []):
            pp = ev.get('probabilities', {}).get('player', {})
            po = ev.get('probabilities', {}).get('opponent', {})
            cur.execute(
                """INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    uid, mv['game_id'], mv['move_number'], ev.get('rank'), ev.get('move'),
                    ev.get('equity'), ev.get('equity_diff'),
                    pp.get('win'), pp.get('gammon'), pp.get('bg'),
                    po.get('win'), po.get('gammon'), po.get('bg')
                )
            )
    conn.commit()

###############################################################################
#  Batch ingest                                                               #
###############################################################################

def ingest_directory(folder: str, db_path: str = 'gnubg_matches.db'):
    conn = sqlite3.connect(db_path)
    _create_schema(conn)
    for txt in glob.glob(str(Path(folder) / '*.txt')):
        try:
            pdata = parse_gnubg_txt(txt)
            _insert_match(conn, pdata)
            print('✔', Path(txt).name)
        except Exception as e:
            print('✖', Path(txt).name, '→', e)
    conn.close()

###############################################################################
#  CLI                                                                        #
###############################################################################
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Parse GNU Backgammon .txt exports')
    p.add_argument('path', help='file or directory')
    p.add_argument('--db', default='gnubg_matches.db', help='SQLite DB path')
    args = p.parse_args()

    target = Path(args.path)
    if target.is_dir():
        ingest_directory(str(target), args.db)
    else:
        print(json.dumps(parse_gnubg_txt(str(target)), ensure_ascii=False, indent=2))
