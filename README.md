# steel-defect-cv-classifier

A real, tested computer-vision project: a small CNN (PyTorch) trained to
classify steel-surface images into four classes (`none`, `scratch`,
`pitting`, `patches`) — built to close a specific gap for Primetals
Technologies Germany's "Werkstudent DevOps/MLOps" posting, which names
"Computer-Vision basierte Lösungen" as the concrete example of the
machine learning applications its team works on.

## What this is (read before citing anywhere)

**All images are synthetic, not a real manufacturing dataset.** There
is no network access to any real surface-defect dataset in this
environment — verified directly, not assumed: connections to
kaggle.com, github.com, and a mirror of the well-known NEU-CLS steel
surface defect benchmark all failed. Rather than skip the computer-vision
gap or imply a real dataset was used, `src/generate_data.py`
procedurally generates grayscale images with three defect patterns
deliberately modeled on the real, well-documented visual signatures of
common steel surface defects — thin linear scratches, clusters of small
dark pits, and diffuse brightness patches — plus a clean/no-defect
class, all on a textured, faintly-banded synthetic "steel" background.

**The model, training, and evaluation are all real.** `src/model.py` is
an actual PyTorch CNN (two conv+pool blocks, two FC layers), actually
trained with a real Adam optimizer and cross-entropy loss on CPU (no
GPU was available in this environment — checked, not assumed), and
evaluated with real scikit-learn metrics (accuracy, per-class
precision/recall/F1, confusion matrix) on a held-out test set the model
never saw during training.

## The honest result — not just an accuracy number

On a synthetic 100/25/25-per-class train/val/test split, the trained
model reaches **96% test accuracy**, but the more informative result is
in the confusion matrix: `none` and `patches` (the two most visually
distinct classes) are classified correctly essentially every time,
while `scratch` and `pitting` are occasionally confused with each other
(3 of 25 pitting images misclassified as scratch in the run captured in
`run_pipeline.py`'s output). That's a real, interpretable pattern, not
noise — both are localized dark features, and a small CNN without
extensive tuning can genuinely conflate them. This is reported directly
rather than only reporting the headline accuracy number, the same
disclosure standard used throughout this portfolio (e.g. the threshold
miscalibration finding in `claims-risk-classifier`, or the
generalization-gap finding in `eval-framework-golden-dataset`).

## What this doesn't demonstrate

This is not a claim of production-grade defect-detection accuracy on
real steel, and it hasn't been validated against any real manufacturing
dataset or real camera/lighting conditions. What it does demonstrate is
a complete, real, working computer-vision pipeline — synthetic data
generation with deliberate visual structure, a real CNN training loop,
honest evaluation beyond a single accuracy number, and a documented,
believable failure mode — which is the same underlying skill set a real
defect-detection project would need, applied here on data that could
actually be generated and verified in this environment.

## Project structure

```
src/
  generate_data.py   -- synthetic steel-surface-defect image generator (4 classes)
  model.py            -- DefectCNN (PyTorch), training loop, evaluation
tests/
  test_generate_data.py
  test_model.py
run_pipeline.py         -- end-to-end demo: generate -> train -> evaluate -> report
```

## Results

```
$ python3 -m pytest tests/ -v
============================== 20 passed in ~15s ==============================
```

All 20 tests pass live, including a genuinely meaningful assertion (not
a tautology) that the trained model clears random-chance accuracy
(0.25 for 4 balanced classes) by a wide margin on held-out data, and
that the two most visually distinct classes (`none`, `patches`) are
classified correctly at least 80% of the time.

## Running it yourself

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```
