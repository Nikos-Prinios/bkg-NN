import gnubg, os, time, socket, signal, sys, random

OUTPUT_DIR         = 'games'
INACTIVITY_TIMEOUT = 10.0           # s sans mouvement => bloqué
instance_id        = '%s_%s' % (socket.gethostname(), os.getpid())
DIFFICULTIES = ['beginner', 'intermediate', 'advanced', 'world_class']


# ---------- sécurité timeout ----------
class Inactivity(Exception): pass
def alarm(sig, frame): raise Inactivity
signal.signal(signal.SIGALRM, alarm)

def match_state():
    signal.alarm(int(INACTIVITY_TIMEOUT))
    try:    return gnubg.match()
    finally: signal.alarm(0)

# ---------- IA world-class ----------

def set_player_difficulty(player, difficulty):
    if difficulty == 'beginner':
        set_player_beginner(player)
    elif difficulty == 'intermediate':
        set_player_intermediate(player)
    elif difficulty == 'advanced':
        set_player_advanced(player)
    elif difficulty == 'world_class':
        set_player_world_class(player)


def set_player_beginner(player):
    gnubg.command('set player %s chequer evaluation plies 0' % player)
    gnubg.command('set player %s chequer evaluation prune off' % player)
    gnubg.command('set player %s chequer evaluation noise 0.060' % player)
    gnubg.command('set player %s cube evaluation plies 0' % player)
    gnubg.command('set player %s cube evaluation prune off' % player)
    gnubg.command('set player %s cube evaluation noise 0.060' % player)


def set_player_intermediate(player):
    gnubg.command('set player %s chequer evaluation plies 0' % player)
    gnubg.command('set player %s chequer evaluation prune off' % player)
    gnubg.command('set player %s chequer evaluation noise 0.040' % player)
    gnubg.command('set player %s cube evaluation plies 0' % player)
    gnubg.command('set player %s cube evaluation prune off' % player)
    gnubg.command('set player %s cube evaluation noise 0.040' % player)


def set_player_advanced(player):
    gnubg.command('set player %s chequer evaluation plies 0' % player)
    gnubg.command('set player %s chequer evaluation prune off' % player)
    gnubg.command('set player %s chequer evaluation noise 0.015' % player)
    gnubg.command('set player %s cube evaluation plies 0' % player)
    gnubg.command('set player %s cube evaluation prune off' % player)
    gnubg.command('set player %s cube evaluation noise 0.015' % player)


def set_player_world_class(player):
    gnubg.command('set player %s chequer evaluation plies 2' % player)
    gnubg.command('set player %s chequer evaluation prune on' % player)
    gnubg.command('set player %s chequer evaluation noise 0.000' % player)
    gnubg.command('set player %s movefilter 1 0 0 8 0.160' % player)
    gnubg.command('set player %s movefilter 2 0 0 8 0.160' % player)
    gnubg.command('set player %s movefilter 3 0 0 8 0.160' % player)
    gnubg.command('set player %s movefilter 3 2 0 2 0.040' % player)
    gnubg.command('set player %s movefilter 4 0 0 8 0.160' % player)
    gnubg.command('set player %s movefilter 4 2 0 2 0.040' % player)
    gnubg.command('set player %s cube evaluation plies 2' % player)
    gnubg.command('set player %s cube evaluation prune on' % player)
    gnubg.command('set player %s cube evaluation noise 0.000' % player)


# ---------- init GNU BG ----------
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for cmd in [
    'set matchlength 1', 'set automatic game on', 'set automatic roll on',
    'set automatic doubles 0', 'set jacoby off', 'set warning endgame off',
    'set display off', 'set player 0 gnubg', 'set player 1 gnubg'
]:
    gnubg.command(cmd)

print('[producer %s] ready' % instance_id); sys.stdout.flush()

# ---------- boucle infinie ----------
while True:
    p0_difficulty = random.choice(DIFFICULTIES)
    p1_difficulty = random.choice(DIFFICULTIES)

    set_player_difficulty(0, p0_difficulty)
    set_player_difficulty(1, p1_difficulty)

    gnubg.command('new match')                         # une seule partie
    blocked, last_mv = False, -1

    try:
        while True:
            st = match_state()
            g  = (st.get('games') or [{}])[-1]
            if g.get('info', {}).get('winner') is not None:
                break
            mv = len(g.get('moves', []))
            if mv != last_mv:
                last_mv = mv
            print('[hb %s] %d' % (instance_id, int(time.time()))); sys.stdout.flush()
            time.sleep(1)
    except Inactivity:
        blocked = True
        try: gnubg.command('stop')
        except: pass

    ts  = int(time.time()*1000)
    fn  = '%d_%s_world_class%s.txt' % (ts, instance_id, '_blocked' if blocked else '')
    gnubg.command('export match text "%s"' % os.path.join(OUTPUT_DIR, fn))
    print('[export %s] %s' % (instance_id, fn)); sys.stdout.flush()
