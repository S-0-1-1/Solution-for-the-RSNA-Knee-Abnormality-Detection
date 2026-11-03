import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def worker(module, *args):
    return [sys.executable, '-m', f'knee.{module}', *[str(x) for x in args]]


def env(**extra):
    e = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    e.update({k: str(v) for k, v in extra.items()})
    return e


def _start(cmd, e, log):
    return dict(proc=subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT, env=e, cwd=ROOT),
                log=log, pos=0, rest='')


def _pump(p):
    with open(p['log'], errors='replace') as fh:
        fh.seek(p['pos'])
        text = p['rest'] + fh.read()
        p['pos'] = fh.tell()
    lines = text.split('\n')
    p['rest'] = lines[-1]
    for line in lines[:-1]:
        if line.startswith('['):
            print(line, flush=True)
    if p['proc'].poll() is None:
        return None
    if p['rest'].startswith('['):
        print(p['rest'], flush=True)
    return p['proc'].returncode


def _stop(running, p):
    for q in running:
        q['proc'].kill()
    tail = open(p['log'], errors='replace').read()[-4000:]
    raise RuntimeError(f'worker stopped with exit code {p["proc"].returncode} ({p["log"]}):\n{tail}')


def run_parallel(jobs, strict=True, poll=15):
    running = [_start(cmd, e, log) for cmd, e, log in jobs]
    failed = []
    while running:
        time.sleep(poll)
        for p in list(running):
            code = _pump(p)
            if code is None:
                continue
            running.remove(p)
            if code != 0:
                if strict:
                    _stop(running, p)
                failed.append(p)
    for p in failed:
        tail = open(p['log'], errors='replace').read()[-600:].replace('\n', ' | ')
        print(f'[worker] exit code {p["proc"].returncode}: {tail}', flush=True)
    return len(failed)


def run_queue(jobs, n_slots, deadline=0, poll=15):
    queue, running = list(jobs), {}
    while queue or running:
        if queue and deadline and time.time() > deadline - 20 * 60:
            print(f'[queue] not started, too close to the time limit: {[os.path.dirname(log) for _, _, log in queue]}', flush=True)
            queue = []
        for slot in range(n_slots):
            if slot not in running and queue:
                cmd, e, log = queue.pop(0)
                running[slot] = _start(cmd, dict(e, CUDA_VISIBLE_DEVICES=str(slot)), log)
        time.sleep(poll)
        for slot, p in list(running.items()):
            code = _pump(p)
            if code is None:
                continue
            del running[slot]
            if code != 0:
                _stop(list(running.values()), p)
