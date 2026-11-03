import re
import json

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
          "Baker's", 'Contusion', 'Fracture']
STATES = ['neg', 'none', 'eq', 'mild', 'mod', 'sev']
STATE_RANK = {s: i for i, s in enumerate(STATES)}
ALIASES = {'negative': 'neg', 'absent': 'neg', 'normal': 'neg', 'intact': 'neg', 'no': 'neg',
            'not_mentioned': 'none', 'not mentioned': 'none', 'silent': 'none', 'unknown': 'none', 'na': 'none', 'n/a': 'none',
            'equivocal': 'eq', 'possible': 'eq', 'questionable': 'eq', 'minimal': 'eq', 'trace': 'eq',
            'small': 'mild', 'low': 'mild', 'moderate': 'mod', 'definite': 'mod', 'present': 'mod',
            'severe': 'sev', 'large': 'sev', 'complete': 'sev', 'high': 'sev'}

SYSTEM_PROMPT = """You are a musculoskeletal radiologist reading knee MRI reports written in English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French. Read the report in its original language. Do not guess beyond the text.

For each of the 12 findings below, give the STATE that the report states, using exactly one code:
neg  = explicitly normal, intact, absent or negated ("no effusion", "ligaments intact")
none = the report does not mention it at all
eq   = equivocal or below threshold: possible, questionable, cannot be excluded, minimal, trace, physiological; or only a related lesser change (see definitions)
mild = present but mild, small, low-grade or partial
mod  = clearly present: moderate, definite, full-thickness, a clear tear
sev  = severe, large, high-grade, complete rupture, displaced, bone-on-bone
If the impression or conclusion disagrees with the findings section, follow the impression.

Definitions:
ACL: anterior cruciate ligament (or ACL graft) injury: sprain, partial tear (mild/mod), complete tear or rupture (sev). Mucoid degeneration or intrasubstance signal without tear = eq.
MCL: medial collateral ligament injury: grade I sprain or periligamentous edema (mild), grade II or partial tear (mod), grade III or complete tear (sev). Old thickening without acute injury = eq.
Medial Meniscus: tear of the medial meniscus of any type (horizontal, radial, complex, flap, bucket-handle, root, ramp, meniscocapsular separation, signal reaching the articular surface) = mod; displaced, complex or root tear = sev; fraying = mild. Intrasubstance or grade 1-2 degeneration, extrusion alone, or post-meniscectomy without re-tear = eq.
Lateral Meniscus: same rules for the lateral meniscus.
Medial OA: osteoarthritis of the MEDIAL femorotibial compartment: chondromalacia or cartilage loss of the medial femoral condyle or medial tibial plateau, osteophytes, joint space narrowing, subchondral sclerosis or cysts. Chondral grade 1 = eq, grade 2 = mild, grade 3 = mod, grade 4 or full-thickness loss = sev.
Lateral OA: same rules for the LATERAL femorotibial compartment.
PF OA: patellofemoral osteoarthritis: chondromalacia or cartilage loss of the patella or trochlea, patellofemoral osteophytes. Same grading as above.
Effusion: joint effusion. Physiological, minimal or trace fluid = eq; small = mild; moderate = mod; large = sev.
Synovitis: synovitis, synovial thickening, hypertrophy or proliferation (including Hoffa synovitis). Hoffa fat pad edema alone = eq.
Baker's: Baker's or popliteal cyst (fluid in the semimembranosus-gastrocnemius bursa). Minimal bursal fluid = eq; small cyst = mild; cyst = mod; large or ruptured cyst = sev.
Contusion: bone contusion or bone bruise (traumatic bone marrow edema). Subchondral edema from degeneration only = eq.
Fracture: any fracture: impaction, osteochondral, avulsion (for example Segond), tibial plateau, patella, stress or insufficiency fracture. Possible or old healed fracture = eq.

Also give "side": "R" or "L" if the report states which knee, else "?".

Answer with ONLY this JSON object, one code per finding, all 12 findings, no other text:
{"side":"?","ACL":"","MCL":"","Medial Meniscus":"","Lateral Meniscus":"","Medial OA":"","Lateral OA":"","PF OA":"","Effusion":"","Synovitis":"","Baker's":"","Contusion":"","Fracture":""}"""


def clean_report(t):
    t = str(t or '').replace('\u00b5', '\u03bc')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()[:6000]


def build_messages(report):
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': 'REPORT:\n' + clean_report(report)}]


def to_state(v):
    v = str(v or '').strip().lower().replace('-', '_')
    if v in STATE_RANK:
        return v
    return ALIASES.get(v, ALIASES.get(v.replace('_', ' ')))


def load_json(blob):
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def parse_output(text):
    t = re.sub(r'<think>.*?</think>', '', str(text), flags=re.S).strip()
    t = t.replace('```json', '').replace('```', '')
    a, b = t.find('{'), t.rfind('}')
    obj = None
    for blob in reversed(re.findall(r'\{[^{}]*\}', t)):
        if '"ACL"' in blob:
            obj = load_json(blob)
            if obj is not None:
                break
    if obj is None and a >= 0 and b > a:
        blob = t[a:b + 1]
        for cand in (blob, re.sub(r',\s*}', '}', blob), blob.replace("\'s\":", "\u2019s\":")):
            obj = load_json(cand)
            if obj is not None:
                break
    out, ok = {}, obj is not None
    for lab in LABELS:
        st = None
        if obj is not None:
            v = obj.get(lab, obj.get(lab.replace("'", '\u2019'), obj.get(lab.lower())))
            if isinstance(v, dict):
                st = to_state(v.get('s', v.get('state')))
            elif isinstance(v, str):
                st = to_state(v)
        if st is None and obj is None:
            m = re.search(re.escape(f'"{lab}"') + r'\s*:\s*(?:\{\s*"s"\s*:\s*)?"([A-Za-z_ ]+)"', t)
            st = to_state(m.group(1)) if m else None
        if st is None:
            st, ok = 'none', False
        out[lab] = st
    side = str((obj or {}).get('side', '?')).strip().upper()[:1]
    return out, (side if side in ('R', 'L') else '?'), ok
