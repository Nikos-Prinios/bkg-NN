#!/usr/bin/env python3
import subprocess, time, os, signal, sys, shutil, re

GNUBG_BIN   = shutil.which('gnubg') or '/usr/local/bin/gnubg'   # adapte si besoin
SCRIPT_PATH = '/Users/nikos/Desktop/gnubg/collect_inf.py'
NUM_INST    = 12               # nombre d’instances GNU BG à faire tourner
HB_DEADLINE = 15              # s sans heartbeat -> redémarrage
GRACE_TERM  = 5               # délai après SIGTERM avant SIGKILL

hb_re   = re.compile(r'^\[hb ([^\]]+)\]')
exp_re  = re.compile(r'^\[export ([^\]]+)\] (.*)')

procs    = {}   # pid -> subprocess.Popen
last_hb  = {}   # pid -> timestamp dernier heartbeat

def launch():
    p = subprocess.Popen(
        [GNUBG_BIN, '-q', '-t', '-p', SCRIPT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    procs[p.pid] = p
    last_hb[p.pid] = time.time()
    print('[watchdog] launched PID', p.pid)
    return p

def main():
    # démarrage initial
    for _ in range(NUM_INST):
        launch()

    while True:
        # lecture non bloquante des sorties
        for pid, p in list(procs.items()):
            if p.stdout is None: continue
            while True:
                line = p.stdout.readline()
                if not line: break
                line = line.rstrip()
                print(line)

                m = hb_re.match(line)
                if m:
                    last_hb[pid] = time.time()
                    continue
                m = exp_re.match(line)
                if m:
                    # ici tu pourrais compter les exports,
                    # uploader les fichiers, etc.
                    continue

        # surveillance des délais
        now = time.time()
        for pid, t_last in list(last_hb.items()):
            if now - t_last > HB_DEADLINE:
                print('[watchdog] no HB -> restart PID', pid)
                p = procs.get(pid)
                if p and p.poll() is None:
                    p.send_signal(signal.SIGTERM)
                    deadline = time.time() + GRACE_TERM
                    while time.time() < deadline and p.poll() is None:
                        time.sleep(0.5)
                    if p.poll() is None:
                        p.kill()
                # nettoyage des structures
                procs.pop(pid, None)
                last_hb.pop(pid, None)
                # relance immédiate
                launch()

        # si moins d’instances que prévu (crash spontané), relancer
        while len(procs) < NUM_INST:
            launch()

        time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[watchdog] terminé')
        sys.exit(0)
