import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time
import argparse
import torch
import pandas as pd
from labeling.prompt import LABELS, build_messages, parse_output


def make_batches(lens, order, prefill_tok, kv_tok, max_new, max_batch):
    batches, cur, longest = [], [], 0
    for i in order:
        L = max(longest, lens[i])
        n = len(cur) + 1
        if cur and (n > max_batch or n * L > prefill_tok or n * (L + max_new) > kv_tok):
            batches.append(cur)
            cur, L = [], lens[i]
        cur.append(i)
        longest = L
    if cur:
        batches.append(cur)
    return batches


def run_batch(b, generate, log):
    try:
        return generate(b)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(b) == 1:
            raise
        h = len(b) // 2
        log(f'out of memory on {len(b)} reports: {h} + {len(b) - h}')
        return run_batch(b[:h], generate, log) + run_batch(b[h:], generate, log)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--max-new', type=int, default=160)
    ap.add_argument('--max-batch', type=int, default=16)
    ap.add_argument('--gold', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(argv)
    import transformers
    import bitsandbytes as bnb
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    transformers.logging.set_verbosity_error()
    TR = pd.read_csv(f'{a.data}/train.csv')
    if a.gold:
        TR = TR[TR[LABELS].notna().all(axis=1)]
    TR = TR.iloc[a.shard::a.nshards]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)['StudyInstanceUID'] for line in open(a.out, encoding='utf-8') if line.strip()}
    TR = TR[~TR.StudyInstanceUID.isin(done)].reset_index(drop=True)
    if a.limit:
        TR = TR.head(a.limit)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tag = f'[gpu{a.shard}]'
    print(f'{tag} {len(TR)} reports to read ({len(done)} already read)', flush=True)
    if not len(TR):
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = json.load(open(os.path.join(a.model, 'config.json')))
    tcfg = cfg.get('text_config', cfg)
    kw = dict(low_cpu_mem_usage=True, device_map={'': 0})
    if 'quantization_config' not in cfg:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, **kw).eval()
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            m.compute_dtype = torch.float16
            qs = getattr(m.weight, 'quant_state', None)
            if qs is not None:
                qs.dtype = torch.float16
    free = torch.cuda.mem_get_info(0)[0]
    hd = tcfg.get('head_dim') or tcfg['hidden_size'] // tcfg['num_attention_heads']
    kv_b = tcfg['num_hidden_layers'] * tcfg.get('num_key_value_heads', tcfg['num_attention_heads']) * hd * 2 * 2
    act_b = tcfg['intermediate_size'] * 2 * 3 + tcfg['hidden_size'] * 2 * 8
    kv_tok, prefill_tok = int(0.45 * free / kv_b), int(0.30 * free / act_b)
    texts = [tok.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for r in TR.Report]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    batches = make_batches(lens, sorted(range(len(texts)), key=lambda i: lens[i]), prefill_tok, kv_tok, a.max_new, a.max_batch)
    print(f'{tag} {len(batches)} batches | longest prompt {max(lens)} tokens', flush=True)

    def generate(b):
        enc = tok([texts[i] for i in b], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        return [tok.decode(g, skip_special_tokens=True) for g in out[:, enc['input_ids'].shape[1]:]]

    t0, n, bad = time.time(), 0, 0
    with open(a.out, 'a', encoding='utf-8') as f, torch.inference_mode():
        for bi, b in enumerate(batches):
            for i, txt in zip(b, run_batch(b, generate, lambda s: print(f'{tag} {s}', flush=True))):
                states, side, ok = parse_output(txt)
                bad += int(not ok)
                f.write(json.dumps({'StudyInstanceUID': TR.StudyInstanceUID[i], 'side': side, 'ok': ok, 'states': states,
                                    'raw': '' if ok else txt[:2000]}, ensure_ascii=False) + '\n')
            f.flush()
            n += len(b)
            if bi % 5 == 0 or bi == len(batches) - 1:
                el = time.time() - t0
                print(f'{tag} {n}/{len(TR)} | {el / n:.2f} s per report | {(len(TR) - n) * el / n / 60:.0f} min left | unreadable {bad}',
                      flush=True)


if __name__ == '__main__':
    main()
