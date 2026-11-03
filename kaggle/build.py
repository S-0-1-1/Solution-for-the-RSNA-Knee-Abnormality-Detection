import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = '/kaggle/tmp/knee-mri-findings'

SUBMIT_SETUP = '''import os
import sys
import glob
import subprocess
import importlib.metadata as metadata
from packaging.utils import parse_wheel_filename, canonicalize_name
from packaging.tags import sys_tags
import torch

CODE = '%s'
if torch.cuda.device_count() == 0:
    raise RuntimeError('no GPU: choose the accelerator GPU T4 x2')
comps, assets, check = [], None, None
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = sorted(d for d in dirs if d not in {'train_series', 'test_series', 'models', 'segmentation', 'wheels', 'training', 'crops'})
    if {'sample_submission.csv', 'test.csv'} & set(files) or os.path.isdir(f'{root}/test_series'):
        comps.append(root)
    if assets is None and os.path.isfile(f'{root}/models/blend.json'):
        assets = root
    if check is None and os.path.isdir(f'{root}/crops') and any(f.endswith('.npz') for f in os.listdir(f'{root}/crops')):
        check = f'{root}/crops'
comps.sort(key=lambda r: (os.path.basename(r) != 'rsna-knee-abnormality-detection', '/competitions/' not in r,
                          not os.path.isdir(f'{r}/test_series'), len(r)))
if not comps or assets is None:
    raise RuntimeError(f'attach the competition data and the knee-mri-assets output (found: data {comps[:1]}, assets {assets})')
data = comps[0]
if os.path.isdir(f'{assets}/training/crops'):
    check = f'{assets}/training/crops'
seg = [os.path.dirname(p) for p in glob.glob(f'{assets}/segmentation/*/*/plans.json')]
if len(seg) != 1:
    raise RuntimeError(f'expected one segmentation model in {assets}/segmentation, found {seg}')
tags = set(sys_tags())
wheels = {}
for w in sorted(glob.glob(f'{assets}/wheels/*.whl')):
    name, _, _, wtags = parse_wheel_filename(os.path.basename(w))
    wheels.setdefault(str(name), [])
    if wtags & tags:
        wheels[str(name)].append(w)
image = {canonicalize_name(d.metadata['Name']) for d in metadata.distributions() if d.metadata and d.metadata['Name']}
missing = [n for n, v in wheels.items() if not v and n not in image]
if missing:
    raise RuntimeError(f'Python {sys.version.split()[0]}: no wheel and not in the image: {missing}')
pick = [v[0] for v in wheels.values() if v]
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', *pick], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError('offline install failed: ' + (r.stderr or r.stdout)[-3000:])
r = subprocess.run([sys.executable, '-c', 'import nnunetv2, SimpleITK, pydicom, timm, cc3d, libjpeg, openjpeg, cv2'],
                   capture_output=True, text=True)
if r.returncode:
    raise RuntimeError('imports failed: ' + r.stderr[-3000:])
os.makedirs(f'{CODE}/knee', exist_ok=True)
os.chdir(CODE)
print(f'python {sys.version.split()[0]} | wheels {len(pick)} | GPUs {torch.cuda.device_count()} | data {data} | assets {assets} | '
      f'check crops {check}')''' % CODE

SUBMIT_RUN = '''sys.path.insert(0, CODE)
import predict
args = ['--data', data, '--models', f'{assets}/models', '--segmentation', seg[0], '--work', '/kaggle/tmp/work',
        '--out', '/kaggle/working/submission.csv', '--hours', '8.5']
predict.main(args + (['--check-crops', check] if check else []))'''

TRAIN_SETUP = '''import os
import sys
import torch

RUN = 'consensus'
CODE = '%s'
if RUN not in ('report', 'consensus', 'consensus_full'):
    raise RuntimeError(f'RUN must be report, consensus or consensus_full, not {RUN}')
if torch.cuda.device_count() == 0:
    raise RuntimeError('no GPU: choose the accelerator GPU T4 x2')
target = 'report' if RUN == 'report' else 'consensus'
crops = labels = pretrained = None
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = sorted(d for d in dirs if d not in {'train_series', 'test_series', 'crops', 'models', 'segmentation', 'wheels'})
    if crops is None and os.path.isdir(f'{root}/crops') and any(f.endswith('.npz') for f in os.listdir(f'{root}/crops')):
        crops = f'{root}/crops'
    if labels is None and os.path.isfile(f'{root}/labels/{target}.csv'):
        labels = f'{root}/labels/{target}.csv'
    if pretrained is None and 'resnet18d.pth' in files:
        pretrained = f'{root}/resnet18d.pth'
missing = [n for n, v in (('crops', crops), (f'labels/{target}.csv', labels), ('resnet18d.pth', pretrained)) if v is None]
if missing:
    raise RuntimeError(f'not found in the inputs: {missing}')
os.makedirs(f'{CODE}/knee', exist_ok=True)
os.makedirs(f'{CODE}/configs', exist_ok=True)
os.chdir(CODE)
print(f'{RUN} | GPUs {torch.cuda.device_count()} | crops {crops} | labels {labels}')''' % CODE

TRAIN_RUN = '''sys.path.insert(0, CODE)
import train
args = ['--labels', labels, '--crops', crops, '--pretrained', pretrained, '--out', f'/kaggle/working/{RUN}', '--hours', '11.3']
train.main(args + (['--full', '2'] if RUN == 'consensus_full' else []))'''


def file_cell(rel):
    src = open(os.path.join(ROOT, rel)).read()
    return f'%%writefile {rel}\n' + src


def notebook(cells):
    for c in cells:
        for line in c.splitlines():
            if line.strip().startswith('#'):
                raise RuntimeError(f'comment line in a cell: {line}')
    return dict(nbformat=4, nbformat_minor=5,
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                          'language_info': {'name': 'python'}},
                cells=[dict(cell_type='code', execution_count=None, id=f'c{i}', metadata={}, outputs=[],
                            source=c.rstrip('\n').splitlines(keepends=True)) for i, c in enumerate(cells)])


PREDICT_FILES = ['knee/__init__.py', 'knee/findings.py', 'knee/series.py', 'knee/bones.py', 'knee/segment.py', 'knee/regions.py',
                 'knee/crops.py', 'knee/cropping.py', 'knee/data.py', 'knee/model.py', 'knee/scoring.py', 'knee/launch.py', 'predict.py']
TRAIN_FILES = ['knee/__init__.py', 'knee/findings.py', 'knee/crops.py', 'knee/data.py', 'knee/model.py', 'knee/fit.py',
               'knee/launch.py', 'configs/kneenet.json', 'train.py']

if __name__ == '__main__':
    for name, setup, files, run in (('submit', SUBMIT_SETUP, PREDICT_FILES, SUBMIT_RUN), ('train', TRAIN_SETUP, TRAIN_FILES, TRAIN_RUN)):
        nb = notebook([setup] + [file_cell(f) for f in files] + [run])
        json.dump(nb, open(os.path.join(ROOT, 'kaggle', f'{name}.ipynb'), 'w'), indent=1)
        print(f'kaggle/{name}.ipynb: {len(nb["cells"])} cells')
