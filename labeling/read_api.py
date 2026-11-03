import os
import json
import time
import random
import argparse
import threading
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from labeling.prompt import LABELS, build_messages, parse_output

QUOTA_WORDS = ('quota', 'exhausted', 'per day', 'daily', 'requests per day', 'tokens per day', 'rpd', 'tpd', 'limit reached',
               'insufficient', 'credit', 'billing', 'payment')
PRINT_LOCK = threading.Lock()
STATE = dict(deadline=0.0, no_extra=False)


def say(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def late():
    return bool(STATE['deadline']) and time.time() > STATE['deadline']


def nap(s):
    if STATE['deadline']:
        s = min(s, STATE['deadline'] + 1 - time.time())
    time.sleep(max(0.0, s))


class QuotaError(RuntimeError):
    pass


class Stopped(RuntimeError):
    pass


class RateLimiter:
    WINDOW, MIN_SEEN, SLOW_ABOVE, FAST_BELOW, HALVE_EVERY_S, UP = 20, 5, 0.5, 0.2, 30.0, 1.25

    def __init__(self, rpm, inflight):
        self.base = 60.0 / rpm
        self.gap = self.base
        self.lock = threading.Lock()
        self.next_t = time.time()
        self.pause_until, self.last_halve = 0.0, 0.0
        self.sem = threading.BoundedSemaphore(max(1, int(inflight)))
        self.last = collections.deque(maxlen=self.WINDOW)
        self.codes = collections.Counter()

    def share(self):
        return sum(self.last) / len(self.last) if len(self.last) >= self.MIN_SEEN else None

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t, self.pause_until)
            self.next_t = t + self.gap
        nap(t - now)

    def slow_down(self, retry_after=None):
        with self.lock:
            now = time.time()
            self.last.append(1)
            s = self.share()
            if s is not None and s > self.SLOW_ABOVE and now - self.last_halve >= self.HALVE_EVERY_S:
                self.gap = min(60.0, self.gap * 2)
                self.last_halve = now
                self.last.clear()
            if retry_after:
                self.pause_until = max(self.pause_until, now + min(float(retry_after), 600))

    def ok(self):
        with self.lock:
            self.last.append(0)
            s = self.share()
            if s is not None and s < self.FAST_BELOW:
                self.gap = max(self.base, self.gap / self.UP)

    def rpm(self):
        return 60.0 / self.gap

    def count(self, code):
        with self.lock:
            self.codes[str(code)] += 1


class Holes:

    def __init__(self, sids):
        self.lock, self.open, self.busy = threading.Lock(), set(sids), set()

    def take(self, sid):
        with self.lock:
            if sid not in self.open or sid in self.busy:
                return False
            self.busy.add(sid)
            return True

    def release(self, sid, answered):
        with self.lock:
            self.busy.discard(sid)
            if answered:
                self.open.discard(sid)

    def left(self):
        with self.lock:
            return len(self.open)

    def open_ids(self):
        with self.lock:
            return set(self.open)


def retry_after(r):
    v = r.headers.get('Retry-After')
    return float(v) if v is not None and v.replace('.', '', 1).isdigit() else None


def ask(api, model, messages, limiter, tries=8):
    extra = {} if STATE['no_extra'] or not api['no_thinking'] else {'chat_template_kwargs': {'enable_thinking': False, 'thinking': False}}
    err = None
    for attempt in range(tries):
        if late():
            raise Stopped('deadline')
        limiter.wait()
        body = dict(model=model, messages=messages, temperature=0.0, max_tokens=api['max_tokens'], **extra)
        with limiter.sem:
            if late():
                raise Stopped('deadline')
            try:
                r = requests.post(f'{api["base_url"]}/chat/completions', json=body, timeout=300,
                                  headers={'Authorization': f'Bearer {api["key"]}', 'Accept': 'application/json'})
            except requests.RequestException as e:
                r, err = None, repr(e)
        if r is None:
            limiter.count('none')
            nap(min(60, 2 ** attempt) + random.random())
            continue
        limiter.count(r.status_code)
        if r.status_code == 200:
            try:
                msg = r.json()['choices'][0]['message']
            except (ValueError, KeyError, IndexError):
                err = f'bad answer body: {r.text[:200]}'
                continue
            limiter.ok()
            content = msg.get('content') or ''
            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
            if '"ACL"' not in content:
                content = content + '\n' + (msg.get('reasoning_content') or msg.get('reasoning') or '')
            return content
        low, err = r.text.lower(), f'HTTP {r.status_code}: {r.text[:200]}'
        if r.status_code == 400 and extra:
            extra, STATE['no_extra'] = {}, True
            continue
        if r.status_code == 402 or (r.status_code in (403, 429) and any(w in low for w in QUOTA_WORDS) and attempt >= 1):
            raise QuotaError(err)
        if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
            if r.status_code == 429:
                limiter.slow_down(retry_after(r))
            else:
                nap(min(60, 2 ** attempt) + random.random())
            continue
        break
    raise RuntimeError(err)


def read_ids(paths):
    s = set()
    for p in paths:
        if os.path.exists(p):
            s |= {json.loads(line)['StudyInstanceUID'] for line in open(p, encoding='utf-8') if line.strip()}
    return s


def label(api, model, order, reports, out, limiter, workers, tag, also=(), holes=None, passes=4, heartbeat_min=5.0):
    done = read_ids([out, *also])
    todo = [s for s in order if s not in done]
    say(f'[{tag}] {model}: {len(todo)} reports to read ({len(done)} already read)')
    if not todo or late():
        return 0
    lock, stop, hb_stop = threading.Lock(), threading.Event(), threading.Event()
    t0, n, bad, refused, total, recent = time.time(), 0, 0, 0, len(todo), collections.deque()

    def status():
        now = time.time()
        el = max(1.0, now - t0)
        while recent and now - recent[0] > 600:
            recent.popleft()
        codes = ' '.join(f'{k}:{v}' for k, v in sorted(limiter.codes.items()))
        left = f'holes left {holes.left()}' if holes is not None else f'left {total - n}'
        return (f'[{tag}] {model}: {n} read | {el / 60:.1f} min | {60 * n / el:.1f} per minute | limiter {limiter.rpm():.1f}/min | '
                f'HTTP {codes or "none yet"} | {left} | unreadable {bad}')

    def beat():
        while not hb_stop.wait(heartbeat_min * 60):
            with lock:
                say(status())

    def one(sid):
        if stop.is_set() or late():
            return sid, '', 'stopped', False
        if holes is not None and not holes.take(sid):
            return sid, '', 'taken', False
        try:
            return sid, ask(api, model, build_messages(reports[sid]), limiter), None, False
        except Stopped:
            return sid, '', 'stopped', False
        except QuotaError as e:
            return sid, '', repr(e)[:300], True
        except (RuntimeError, requests.RequestException) as e:
            return sid, '', repr(e)[:300], False

    threading.Thread(target=beat, daemon=True).start()
    try:
        with open(out, 'a', encoding='utf-8') as f:
            for p in range(1, passes + 1):
                if not todo or stop.is_set() or late():
                    break
                if p > 1:
                    say(f'[{tag}] {model}: pass {p}, {len(todo)} reports that failed before')
                failed, shown = [], 0
                with ThreadPoolExecutor(workers) as pool:
                    for fut in as_completed([pool.submit(one, s) for s in todo]):
                        sid, txt, err, quota = fut.result()
                        with lock:
                            if holes is not None and err != 'taken':
                                holes.release(sid, err is None)
                            if err == 'taken':
                                continue
                            if err:
                                failed.append(sid)
                                if err == 'stopped':
                                    continue
                                if shown < 2:
                                    say(f'[{tag}] {model}: request failed: {err[:160]}')
                                    shown += 1
                                if quota or any(f'HTTP {c}' in err for c in (401, 403, 404)):
                                    refused += 1
                                if refused >= 5 and not stop.is_set():
                                    stop.set()
                                    say(f'[{tag}] stopping {model}: free limit reached or key refused ({err[:120]})')
                                continue
                            states, side, ok = parse_output(txt)
                            bad += int(not ok)
                            f.write(json.dumps({'StudyInstanceUID': sid, 'side': side, 'ok': ok, 'states': states,
                                                'raw': '' if ok else txt[-1500:]}, ensure_ascii=False) + '\n')
                            f.flush()
                            n += 1
                            recent.append(time.time())
                            if n % 25 == 0:
                                say(status())
                todo = failed
    finally:
        hb_stop.set()
    say(status())
    say(f'[{tag}] {model}: {n} answered | {len(todo)} still missing')
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--done', nargs='*', default=[])
    ap.add_argument('--base-url', default='https://integrate.api.nvidia.com/v1')
    ap.add_argument('--key-env', default='NVIDIA_API_KEY')
    ap.add_argument('--rpm', type=float, default=30)
    ap.add_argument('--inflight', type=int, default=6)
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--no-thinking', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--holes-first', action='store_true')
    ap.add_argument('--complete', action='store_true')
    ap.add_argument('--gold', action='store_true')
    ap.add_argument('--hours', type=float, default=0)
    a = ap.parse_args(argv)
    if a.key_env not in os.environ:
        raise RuntimeError(f'set the API key in the environment variable {a.key_env}')
    STATE['deadline'] = time.time() + a.hours * 3600 if a.hours else 0.0
    api = dict(base_url=a.base_url, key=os.environ[a.key_env], max_tokens=a.max_tokens, no_thinking=a.no_thinking)
    TR = pd.read_csv(f'{a.data}/train.csv')
    if a.gold:
        TR = TR[TR[LABELS].notna().all(axis=1)]
    ids, reports = TR.StudyInstanceUID.tolist(), dict(zip(TR.StudyInstanceUID, TR.Report))
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for model in a.models:
        name = model.replace('/', '__').replace(':', '~') + '.jsonl'
        limiter = RateLimiter(a.rpm / len(a.models), max(1, a.inflight // len(a.models)))
        jobs.append((model, limiter, [f'{a.out}/{name}'] + [f'{d}/{name}' for d in a.done]))
        say(f'{model}: at most {limiter.rpm():.0f} requests per minute')
    holes, hole_ids = None, []
    if a.holes_first:
        anyone = set().union(*[read_ids(files) for _, _, files in jobs])
        hole_ids = [s for s in ids if s not in anyone]
        holes = Holes(hole_ids)
        say(f'{len(hole_ids)} of {len(ids)} reports not read by any of these models yet: each is read once, by the first model free')
    errors = []

    def run(k, model, limiter, files):
        try:
            if holes is None:
                label(api, model, ids, reports, files[0], limiter, a.workers, 'all', files[1:])
                return
            order = hole_ids if k % 2 == 0 else hole_ids[::-1]
            label(api, model, order, reports, files[0], limiter, a.workers, 'holes', files[1:], holes)
            if a.complete and not late():
                first = holes.open_ids()
                order = [s for s in ids if s in first] + [s for s in ids if s not in first]
                label(api, model, order, reports, files[0], limiter, a.workers, 'complete', files[1:])
        except BaseException as e:
            errors.append((model, e))

    threads = [threading.Thread(target=run, args=(k, *j)) for k, j in enumerate(jobs)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    for model, limiter, files in jobs:
        say(f'{model}: {len(read_ids(files) & set(ids))} of {len(ids)} reports read | HTTP {dict(sorted(limiter.codes.items()))}')
    if errors:
        raise RuntimeError(f'{errors[0][0]} stopped with an error') from errors[0][1]


if __name__ == '__main__':
    main()
