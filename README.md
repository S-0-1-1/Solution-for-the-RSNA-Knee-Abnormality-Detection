# Knee MRI findings

Code for my solution to the Kaggle competition [RSNA Knee Abnormality Detection 2026](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Each knee MRI study has to be scored for 12 findings: ACL tear, MCL tear, medial and lateral meniscus tear, medial,
lateral and patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture. The metric is
the mean AUC over the 12 findings.

Public leaderboard: **0.938**. Only the competition data is used for training, with no external MRI data.

The idea is to find the bones first, use them to define 8 knee regions, then train a model to detect findings in each region.

1. Series. DICOM information is used to select the best sagittal MRI scan for bone segmentation.

2. Bones. MRAnnotator is used to segment the femur, patella, fibula and tibia. The masks are cleaned, and another scan is tried if needed.

3. Side and regions. The knee side is identified using the fibula or DICOM information. Eight regions are defined around the bones. If segmentation fails, an atlas is used instead.

4. Crops. 21 crops are extracted from up to 5 MRI series. Each crop is 128 x 128 pixels, with up to 24 slices. Right knees are flipped to match left knees.

5. Model. ResNet18 processes groups of 3 neighbouring slices. A Transformer combines the information, and each finding focuses on the most relevant regions.

6. Labels. Only 58 studies have radiologist labels. For the others, language models extract labels from radiology reports. Two label sets are used: report labels and consensus labels (report labels combined with model predictions).

7. Training. The model is trained for 12 epochs using AdamW, data augmentation and mixed precision. Five-fold cross-validation is used, and the best model from each fold is kept.

8. Prediction. Each model predicts the study 4 times using small rotations and zoom changes. The predictions are averaged. Finally, three model families are combined with equal weights.

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



## Setup

```
pip install -r requirements.txt
```

The bone segmentation needs the MRAnnotator weights (`Dataset002_ShoulderKnee`, `3d_fullres`). The training needs the
ImageNet weights of resnet18d as a state dict:

```python
import timm, torch
torch.save(timm.create_model('resnet18d.ra2_in1k', pretrained=True, num_classes=0).state_dict(), 'resnet18d.pth')
```

## Data formats

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

**Models**, the folder that `train.py` writes, plus two files:

```
models/
  blend.json            {"consensus": 1.0, "report": 1.0, "consensus_full": 1.0}
  anatomy.json          atlas boxes and quality bounds, from knee.regions.build_atlas on the training studies
  consensus/            config.json, fold0/best.pt ... fold4/best.pt
  report/               config.json, fold0/best.pt ... fold4/best.pt
  consensus_full/       config.json, full1/best.pt, full2/best.pt
```

## Training

```
python train.py --labels labels/report.csv --crops crops --pretrained resnet18d.pth --out models/report
python train.py --labels labels/consensus.csv --crops crops --pretrained resnet18d.pth --out models/consensus
python train.py --labels labels/consensus.csv --crops crops --pretrained resnet18d.pth --out models/consensus_full --full 2
```

The folds run in parallel, one per GPU. A finished fold is skipped when the command is run again. After the five folds,
the out-of-fold AUC is printed and saved in `auc.csv`.

## Prediction

```
python predict.py --data <competition folder> --models models --segmentation <MRAnnotator model folder> --out submission.csv
```

The study folders are read from `test_series/`, and the study list from `sample_submission.csv` or `test.csv`. By default
any error stops the run. On the hidden test set of the competition the run is lenient instead: a study that cannot be
read or segmented falls back to standard boxes or to the median prediction, and the run still stops if more than 1% of
the studies (at least 5) need such a fallback. `--mode strict` or `--mode lenient` forces one behaviour.

With `--check-crops <training crops folder>`, six training studies also go through the whole pipeline, and their crops
and predictions are compared with the stored training crops. The run stops if they do not match.

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

## Credits

- Bone segmentation: A. Zhou et al., MRAnnotator: multi-Anatomy and many-Sequence MRI segmentation of 44 structures,
  Radiology Advances 2(1), 2025. [arXiv:2402.01031](https://arxiv.org/abs/2402.01031)
- Image encoder: [timm](https://github.com/huggingface/pytorch-image-models), resnet18d with ImageNet weights.
- Data: RSNA Knee Abnormality Detection 2026, used under the competition rules.
