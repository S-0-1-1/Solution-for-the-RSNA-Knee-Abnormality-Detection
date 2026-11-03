# Knee MRI Abonormality detection

My solution to the Kaggle competition [RSNA Knee Abnormality Detection 2026](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
Each knee MRI study gets a score for 12 findings, such as ACL tear, meniscus tears, osteoarthritis, effusion and fracture.

## Results

The public leaderboard score is 0.938 (mean AUC over the 12 findings). Out of fold, the models reach 0.92 on the 58
studies labelled by radiologists, from 0.82 for synovitis to 0.98 for the ACL, the MCL and Baker's cyst. Only the
competition data is used.

## Preprocessing

Every series is resampled to the same 0.51 mm grid, and [MRAnnotator](https://arxiv.org/abs/2402.01031) (an nnU-Net
model) segments the bones of the knee on the best sagittal series. The bones give 8 regions of the knee, for example the
medial compartment or the notch, and each region is cropped from up to 5 series: 21 crops per study.

Only 58 training studies have radiologist labels. For the others, three language models read the radiology reports:
Nemotron-3 Ultra and Nemotron-3 Super through the free NVIDIA API, and [Qwen3-14B](https://arxiv.org/abs/2505.09388) on
two Kaggle T4 GPUs. For each finding, they say what the report states, from negative to severe, and the 58 labelled
studies are used to turn these answers into soft labels.

## Model

A resnet18d reads the crops, a transformer layer combines them, and each finding has its own attention head. The
submission averages 12 such models, trained on the report labels or on a mix of report labels and model predictions.

## How to use

Install the packages, build the crops, make the labels with `labeling/read_api.py`, `read_local.py` and `calibrate.py`
(the API key goes in the `NVIDIA_API_KEY` environment variable), then train:

```
pip install -r requirements.txt
python prepare.py --data <competition folder> --segmentation <MRAnnotator model> --out work
python train.py --labels labels/report.csv --crops work/crops --pretrained resnet18d.pth --out models/report
```

## How to test

The trained weights are in the [v.1.0 release](https://github.com/S-0-1-1/Solution-for-the-RSNA-Knee-Abnormality-Detection/releases/tag/v.1.0).
Download `models.zip` and unzip it next to the code: you get the `models` folder with the 12 models, `blend.json` and
`anatomy.json`.

`predict.py` reads the test DICOM folders and writes `submission.csv`. On Kaggle, `kaggle/submit.ipynb` does the same
offline.

```
python predict.py --data <competition folder> --models models --segmentation <MRAnnotator model> --out submission.csv
```

