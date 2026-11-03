# Knee MRI findings

Code for my solution to the Kaggle competition [RSNA Knee Abnormality Detection 2026](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Each knee MRI study has to be scored for 12 findings: ACL tear, MCL tear, medial and lateral meniscus tear, medial,
lateral and patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture. The metric is
the mean AUC over the 12 findings.

Public leaderboard: **0.938**. Only the competition data is used for training, with no external MRI data. The labels of
the training studies come from their radiology reports, read by large language models.

## Method

The idea is to look where each finding lives. The bones are segmented first, eight anatomical regions are placed from
them, and the classifier only sees crops of those regions, in the series where each region is best seen. Since only 58
training studies have radiologist labels, the labels of the other studies are read from their reports.

```
images:  DICOM series -> sagittal series -> bone masks -> 8 region boxes -> 21 crops -> classifier -> 12 scores
labels:  report -> 3 language models -> 6 states per finding -> calibrated soft labels -> training targets
```

**1. Series.** The DICOM headers are read for every series (24,371 series in the 4,407 training studies): plane, fat
suppression, sequence, size. The best sagittal series is chosen for segmentation: spin echo before gradient echo, 2D
before 3D, a normal number of slices.

**2. Bones.** [MRAnnotator](https://arxiv.org/abs/2402.01031) segments the femur, patella, fibula and tibia on the
chosen sagittal series (its other labels are not used). It is an nnU-Net (knee and shoulder model, `3d_fullres`), and
the main question was how to feed it series of very different resolutions: the training set comes from 59 scanner
setups, with mostly 2D series of thick slices and fine pixels, and a few thin 3D series.

Every series is first turned into one volume in a fixed orientation and resampled, with linear interpolation, to the
same isotropic grid of 0.51 mm, whatever its own pixel size and slice thickness. nnU-Net is then told that this grid is
its own training grid (about 0.36 mm), so it does not resample again and sees the knee at 70% of its real size. The
model's own 0.36 mm grid was also tried. The 0.51 mm grid was kept for two reasons: a scale of 0.7 is still inside the
random scaling that nnU-Net applies in training (0.7 to 1.4), and the volume has about three times fewer voxels. The
tiles do not overlap (step 1.0 instead of the default 0.5), and there is no mirroring, since the model was trained
without it. The labels are resampled back to the grid of the series with nearest neighbour interpolation.

The raw labels are not always right on these series, so the bones are then identified again by anatomy:

- the femur and the tibia are the connected objects that hold most of the femur and tibia voxels; when they touch at
  the joint, the object is split by the distance to the voxels of each bone;
- the patella is the largest patella piece in front of the femoral condyles, and the fibula the largest fibula piece
  below the tibial plateau, behind and beside the tibia;
- loose pieces above the femur or below the tibia are put back on the shaft, and pieces under 0.5 ml are removed.

This changed the labels of 445 training studies by more than 0.5 ml (in 9 of them, the femur and the tibia were one
object). A mask then passes the quality checks when:

- the four bones are found where they belong: femur above the tibia, patella in front of the femur, fibula behind the
  tibia;
- the femur, the tibia and the patella are each at least 90% in one piece;
- the volumes are plausible: femur at least 40 ml, tibia at least 30 ml, patella 2 to 45 ml;
- the bone volumes near the joint and the knee width are not far from the rest of the cohort. The limits are the median
  ± 5 robust standard deviations (1.4826 × MAD, at least 10% of the median) over the training studies that pass the
  other checks.

With MRAnnotator on the first sagittal series, 4,147 of the 4,407 training studies (94.1%) passed the first version of
these checks. When a mask fails, the next spin-echo sagittal series is segmented too and the better mask is kept: 4,277
studies (97.1%). After the anatomical identification, with the final rules where the patella and the fibula must be
found where they belong, 4,385 studies (99.5%) pass, and the femur and the tibia can be used for the region boxes in
4,405 studies. nnInteractive was tried as a further repair on the 130 studies that failed on both series, but only 8 of
them passed, so the final masks come from MRAnnotator alone.

On two T4 GPUs, with two workers per GPU, a study takes about 15 seconds per worker: about 4.5 hours for the 4,407
training studies.

**3. Side and regions.** The side of the knee comes from the fibula, which lies on the lateral side of the tibia. When
the fibula is not found, the DICOM laterality tag is used (usable in 2,178 training studies), and for a training study
without the tag, the side written in the report (1,547 reports state it).

Eight boxes are built from the bones in patient coordinates: medial and lateral tibiofemoral compartments,
intercondylar notch, patellofemoral joint, MCL, whole joint, popliteal fossa and all bones. When the bones are not
usable, the boxes come from an atlas: for each box, the median offset from the knee centre and the median size over the
training studies that pass all checks. The atlas is placed at the knee centre, or at the centre of the field of view
(shifted by the median knee offset) when the knee centre itself is unknown. The atlas and the cohort limits are saved in
`anatomy.json` and used as they are for the test studies.

**4. Crops.** Each region is cropped from up to five series (sagittal PD/T1, sagittal fluid sensitive, coronal fluid
sensitive, coronal PD/T1, axial fluid sensitive), 21 crops in total. A crop is resampled in the slice planes of its own
series, so two thick slices are never mixed: 128 x 128 pixels, up to 24 slices (16 for the large regions), a 4 mm margin,
thin 3D series grouped into 3 mm slabs. Right knees are mirrored so every knee looks like a left knee. Intensities are
scaled by the 0.5 and 99.5 percentiles inside the whole-knee box.

**5. Labels from the reports.** Only 58 of the 4,407 training studies have radiologist labels, but every study has its
report, written in one of nine languages (English 1,736, Spanish 682, Turkish 546, Croatian 406, Greek 321, German 262,
Bulgarian 220, Dutch 153, French 81). The labels of the other studies are read from these reports by language models.

*The question.* The prompt (`labeling/prompt.py`) asks the model to read the report in its own language and to give,
for each of the 12 findings, the state that the report states, on an ordinal scale:

| `neg` | `none` | `eq` | `mild` | `mod` | `sev` |
|---|---|---|---|---|---|
| normal or negated | not mentioned | equivocal, below threshold | mild, small, partial | clearly present | severe, complete, displaced |

The prompt defines every finding with its grades (for example cartilage grade 1 = `eq`, 2 = `mild`, 3 = `mod`, 4 =
`sev` for the three osteoarthritis findings), says to follow the impression when it disagrees with the findings
section, and asks for one JSON object only, which also gives the side of the knee. The temperature is 0. The model is not
asked for 0 or 1: it only says what the text says, and where the radiologists put the threshold (does a small effusion
count?) is learned afterwards from the 58 labelled studies.

*The readers.* Several models were first compared on the 58 labelled reports, with the state itself as the score
(macro AUC against the radiologist labels):

| reader | runs on | macro AUC |
|---|---|---|
| Qwen3-14B, 4-bit | Kaggle, 2 x T4 | 0.884 |
| Nemotron-3 Super 120B-A12B | NVIDIA API, free | 0.878 |
| Nemotron-3 Ultra 550B-A55B | NVIDIA API, free | 0.875 |
| gpt-oss-120b | Groq API, free | 0.873 |
| Qwen3-8B, 4-bit | Kaggle, 2 x T4 | 0.831 |
| GLM-5.3 | NVIDIA API, free | 0.743 |

Three readers were kept: Qwen3-14B and the two Nemotron-3 models. gpt-oss-120b was as good, but its free daily limit
would have needed about 44 days for all the reports, and GLM-5.3 gave an unreadable answer for 22 of the 58 reports.
The average of the three readers reaches 0.896.

Qwen3-14B runs locally (`labeling/read_local.py`), one copy per GPU, with greedy decoding, thinking off and at most 160
new tokens. The reports are sorted by length and grouped into batches that fit in the free memory, and a batch that
runs out of memory is split in two. It reads a report in about 5.9 seconds per GPU, so the 4,407 reports take about 3.6
hours on two T4 GPUs.

The two Nemotron-3 models are called through NVIDIA's free OpenAI-compatible API (`labeling/read_api.py`). The free
tier is rate limited, so the client:

- paces its requests: at most 30 per minute and 6 at once, shared between the models. It halves its speed when more
  than half of the recent answers are "429 Too Many Requests", and speeds up again when fewer than a fifth are;
- retries a failed report in up to four passes, and stops a model, without ever paying, when the provider answers that
  payment is required or the quota is reached;
- writes every answer to a JSONL file as soon as it arrives, so a stopped run goes on where it stopped;
- with `--holes-first`, shares the reports that no model has read yet between the models, each read once by the first
  model that is free. With `--complete`, each model then reads the rest.

Super read about 15 reports per minute at first, but once NVIDIA throttled the key, both models read 1 to 3 reports per
minute, and the reading took several 11-hour Kaggle sessions. In the end, Ultra read 4,273 reports and Super 4,101, and
every report was read by at least one of them. On the 3,967 reports read by both, they give the same present or absent
call for 96.3% of the findings and the same state for 81.3%.

*From states to soft labels.* Each reader is calibrated on the 58 labelled studies (`labeling/calibrate.py`) with a
small model: for a finding $f$ that the reader puts in state $s$,

```math
P(y_f = 1 \mid s) = \sigma(a_s + b_f), \qquad a_{\text{neg}} \le a_{\text{none}} \le a_{\text{eq}} \le a_{\text{mild}} \le a_{\text{mod}} \le a_{\text{sev}}
```

The six values $a_s$ are shared by all findings and kept in order by writing them as a first value plus softplus steps;
$b_f$ is one shift per finding. With only 58 studies, this keeps the model at 18 parameters: a state means the same
thing for every finding, up to its shift. The parameters minimise the log loss on the 58 studies, with a ridge penalty
that shrinks the shifts toward 0:

```math
\mathcal{L} = -\sum_{i,f} \Big[ y_{if} \log p_{if} + (1 - y_{if}) \log (1 - p_{if}) \Big] + \lambda \sum_f b_f^2
```

The weight $\lambda$ is chosen among 0.3, 1, 3 and 10 by the leave-one-study-out log loss. The report label of a study is then the mean of the calibrated probabilities of the readers $R$ that read it:

```math
y^{\text{report}}_f = \operatorname{clip}\Big( \frac{1}{|R|} \sum_{r \in R} P_r(y_f = 1 \mid s_{r,f}),\ 0.02,\ 0.98 \Big)
```

The set of readers is chosen on the 58 studies too: the largest set within 0.005 of the best macro AUC, since a smaller
difference is noise with 58 studies. Here it is all three readers. 3,966 labels come from all three readers, 306 from
Ultra and Qwen3-14B, 134 from Super and Qwen3-14B, and one from the two API models.

*Consensus labels.* A report cannot tell what the radiologist did not write, but the image can still show it. A second
label set therefore mixes the report label with a teacher, the prediction of image models:

```math
y^{\text{cons}} = \operatorname{clip}\big( 0.5\, y^{\text{report}} + 0.5\, y^{\text{teacher}},\ 0.02,\ 0.98 \big)
```

The teacher is the mean of the out-of-fold predictions of two crop models trained on the report labels: the report
models of this repository and an earlier model that also looked at 256-pixel views of the knee (not part of this
repository). No teacher prediction comes from a model trained on that study. On the 58 labelled studies, as a label
source, the report labels reach a macro AUC of 0.897, the teacher 0.918 and the consensus 0.935. In both label sets, the
58 studies keep their radiologist labels and count three times in the loss.

**6. Model.** A resnet18d (ImageNet weights) reads windows of three neighbouring slices, 8 per crop in training and 12
at test time. Each window becomes a token with a crop embedding and its slice position. One transformer layer mixes all
tokens of the study, then every finding has its own attention query, with a prior that favours the regions where the
finding is seen (for example the notch for the ACL, the popliteal fossa for a Baker's cyst).

**7. Training.** Soft binary cross entropy, AdamW (2e-4 for the encoder, 5e-4 for the rest), warm-up then cosine decay,
batch 4 with 2 accumulation steps, 12 epochs, EMA of the weights, mixed precision. Augmentation runs on the GPU: rotation,
scale, shift, contrast, brightness, gamma, and dropping a whole crop now and then. The five folds are stratified on the 12
findings and on the radiologist-labelled studies. The epoch with the lowest validation loss is kept. One fold takes about
2 hours on a T4.

**8. Prediction.** Every model scores each study four times (no change, rotation of +5 and -5 degrees, zoom 1.05) and the
probabilities are averaged. Three families of models are averaged with equal weights:

| family | labels | models |
|---|---|---|
| `consensus` | consensus | 5 folds |
| `consensus_full` | consensus | 2 seeds, all studies |
| `report` | report | 5 folds |

## Results

Out-of-fold macro AUC of the 5-fold models (consensus and report, 2 to 1):

| reference | AUC |
|---|---|
| 58 studies labelled by radiologists | 0.9215 |
| report labels, all other studies | 0.9067 |
| report labels, only the clear ones (soft label below 0.2 or above 0.8) | 0.9667 |

Per finding, on the 58 radiologist-labelled studies:

| ACL | MCL | Med. meniscus | Lat. meniscus | Med. OA | Lat. OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.983 | 0.982 | 0.977 | 0.830 | 0.974 | 0.865 | 0.870 | 0.957 | 0.816 | 0.982 | 0.907 | 0.917 |

## Repository

```
prepare.py              training studies: bone masks, region boxes, atlas and crops
train.py                trains the 5 folds, or the models on all studies
predict.py              DICOM study folders to submission.csv
configs/kneenet.json    training settings
knee/
  series.py             DICOM headers, series table, choice of the sagittal series
  segment.py            bone segmentation with MRAnnotator (worker)
  bones.py              anatomical identification of the bones, quality checks, knee side, region boxes
  regions.py            final masks, cohort limits, region boxes, atlas (worker)
  crops.py              crop extraction and the crop file format
  cropping.py           crops for all studies (worker)
  data.py               slice sampling and GPU augmentation
  model.py              the classifier
  fit.py                training of one fold (worker)
  scoring.py            test-time scoring (worker)
  findings.py           the 12 findings, region priors, folds
  launch.py             runs the workers and shows their progress
labeling/
  prompt.py             the prompt, the six states and the answer parser
  read_api.py           reads the reports through an OpenAI-compatible API
  read_local.py         reads the reports with a local model, one process per GPU
  calibrate.py          calibration of each reader on the 58 studies, report labels
  consensus.py          consensus labels from the report labels and teacher predictions
  requirements.txt      the packages that only the labelling needs
kaggle/
  submit.ipynb          submission notebook (offline)
  train.ipynb           training notebook
  build.py              writes both notebooks from the code above
```

## Setup

```
pip install -r requirements.txt
pip install -r labeling/requirements.txt
```

The second line is only needed for the labelling. The bone segmentation needs the MRAnnotator weights
(`Dataset002_ShoulderKnee`, `nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres`), the local reader the weights of
`unsloth/Qwen3-14B-bnb-4bit`, and the API reader a key of the NVIDIA API catalog in the environment variable
`NVIDIA_API_KEY`. Never write the key in the code. The training needs the ImageNet weights of resnet18d as a state dict:

```python
import timm, torch
torch.save(timm.create_model('resnet18d.ra2_in1k', pretrained=True, num_classes=0).state_dict(), 'resnet18d.pth')
```

## From the DICOM files to a submission

**1. Bones, regions and crops of the training studies.**

```
python prepare.py --data <competition folder> --segmentation <MRAnnotator model folder> --out work
```

This reads `train.csv`, `train_series.csv` and `train_series/`, segments every study, identifies the bones, learns the
cohort limits and the atlas, and writes the crops to `work/crops/` (one file per study) and the atlas to
`work/anatomy.json`. A finished study is skipped when the command is run again, and with `--hours` the segmentation and
the cropping stop in time, so the work can be split over several sessions.

**2. Report labels.** The API readers, then the local reader on two GPUs, then the calibration:

```
python -m labeling.read_api --data <competition folder> --out answers --no-thinking --holes-first --complete --hours 11 \
    --models nvidia/nemotron-3-ultra-550b-a55b nvidia/nemotron-3-super-120b-a12b
CUDA_VISIBLE_DEVICES=0 python -m labeling.read_local --data <competition folder> --model <Qwen3-14B folder> \
    --out answers/local_0.jsonl --shard 0 --nshards 2 &
CUDA_VISIBLE_DEVICES=1 python -m labeling.read_local --data <competition folder> --model <Qwen3-14B folder> \
    --out answers/local_1.jsonl --shard 1 --nshards 2 &
wait
python -m labeling.calibrate --data <competition folder> --out labels/report.csv \
    --reader ultra=answers/nvidia__nemotron-3-ultra-550b-a55b.jsonl \
    --reader super=answers/nvidia__nemotron-3-super-120b-a12b.jsonl \
    --reader local=answers/local_0.jsonl,answers/local_1.jsonl
```

`--done <folder>` gives `read_api.py` the answers of an earlier session, so no report is asked twice, and `--gold`
reads only the 58 labelled reports, to compare readers quickly. Every reader must have read the 58 labelled reports
before the calibration.

**3. Report models.**

```
python train.py --labels labels/report.csv --crops work/crops --pretrained resnet18d.pth --out models/report
```

The folds run in parallel, one per GPU. A finished fold is skipped when the command is run again. After the five folds,
the out-of-fold AUC is printed and saved in `auc.csv`, and the out-of-fold predictions in `oof.csv`.

**4. Consensus labels.**

```
python -m labeling.consensus --labels models/report/table.csv --teachers models/report/oof.csv --out labels/consensus.csv
```

`models/report/table.csv` holds the report labels with their folds, so the consensus models keep the same folds. More
files can follow `--teachers`; they are averaged.

**5. Consensus models.**

```
python train.py --labels labels/consensus.csv --crops work/crops --pretrained resnet18d.pth --out models/consensus
python train.py --labels labels/consensus.csv --crops work/crops --pretrained resnet18d.pth --out models/consensus_full --full 2
```

**6. Prediction.**

```
cp work/anatomy.json models/
echo '{"consensus": 1.0, "report": 1.0, "consensus_full": 1.0}' > models/blend.json
python predict.py --data <competition folder> --models models --segmentation <MRAnnotator model folder> --out submission.csv
```

The study folders are read from `test_series/`, and the study list from `sample_submission.csv` or `test.csv`. By default
any error stops the run. On the hidden test set of the competition the run is lenient instead: a study that cannot be
read or segmented falls back to standard boxes or to the median prediction, and the run still stops if more than 1% of
the studies (at least 5) need such a fallback. `--mode strict` or `--mode lenient` forces one behaviour.

With `--check-crops <training crops folder>`, six training studies also go through the whole pipeline, and their crops
and predictions are compared with the stored training crops. The run stops if they do not match.

## Data formats

**Reader answers**, one JSONL file per reader, one line per report: `StudyInstanceUID`, `side` (`R`, `L` or `?`), `ok`
(false when the answer could not be read), `states` (the state of each finding) and `raw` (the answer text when it could
not be read).

**Labels**, one CSV per label set: `StudyInstanceUID`, `fold` (0 to 4, computed when missing), `is_gold` (1 for a
radiologist-labelled study) and one column per finding with a soft label between 0 and 1. Optional `report_<finding>`
columns are used to score a consensus model against the report labels.

**Crops**, one file per study, `<StudyInstanceUID>.npz`, written by `knee.crops.study_crops`:

| key | content |
|---|---|
| `<crop>__imgw` | the slices of the crop stacked vertically, 128 pixels wide, uint8, lossless WebP |
| `<crop>__pos` | position of each slice in mm from the centre of the region, float16 |
| `<crop>__mskz` | bone labels of each slice (1 femur, 2 patella, 3 fibula, 4 tibia), zlib |
| `meta` | JSON: for each crop the number of slices `n`, the share of the crop inside the image `fov`, the series used |

A crop is named `<region>__<series slot>`, for example `notch__sag_fs`. The 21 names are in `knee/crops.py`.
`knee.crops.read_study(path)` decodes a file.

**Models**, the folders that `train.py` writes, plus two files:

```
models/
  blend.json            {"consensus": 1.0, "report": 1.0, "consensus_full": 1.0}
  anatomy.json          atlas boxes and cohort limits, written by prepare.py
  consensus/            config.json, fold0/best.pt ... fold4/best.pt
  report/               config.json, fold0/best.pt ... fold4/best.pt
  consensus_full/       config.json, full1/best.pt, full2/best.pt
```

## Kaggle

`kaggle/submit.ipynb` runs offline (GPU T4 x2, Internet off). Attach the competition data and one dataset with:

```
models/                 as above
segmentation/           Dataset002_ShoulderKnee/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/
wheels/                 the packages of requirements.txt that the Kaggle image lacks, as .whl files
```

The notebook installs the wheels that fit the Python version of the image, writes the code and runs `predict.py`.
Attach a training crops folder too if you want the check above.

`kaggle/train.ipynb` trains one family per session: set `RUN` to `report`, `consensus` or `consensus_full` in the first
cell, and attach the crops, `labels/report.csv` or `labels/consensus.csv`, and `resnet18d.pth`.

The API reader needs the Internet. On Kaggle, save the key as a secret named `NVIDIA_API_KEY` and put it in the
environment before running `read_api.py`:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ['NVIDIA_API_KEY'] = UserSecretsClient().get_secret('NVIDIA_API_KEY')
```

## Credits

- Bone segmentation: A. Zhou et al., MRAnnotator: multi-Anatomy and many-Sequence MRI segmentation of 44 structures,
  Radiology Advances 2(1), 2025. [arXiv:2402.01031](https://arxiv.org/abs/2402.01031)
- nnU-Net: F. Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,
  Nature Methods 18, 2021.
- Report readers: NVIDIA Nemotron-3 Ultra and Super through the NVIDIA API catalog, and Qwen3-14B (Qwen team,
  [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)) in the 4-bit version of Unsloth.
- Image encoder: [timm](https://github.com/huggingface/pytorch-image-models), resnet18d with ImageNet weights.
- Data: RSNA Knee Abnormality Detection 2026, used under the competition rules.
