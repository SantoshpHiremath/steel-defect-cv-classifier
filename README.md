# steel-defect-cv-classifier

A tested computer-vision project: a small PyTorch CNN that classifies
steel-surface images into four classes (`none`, `scratch`, `pitting`,
`patches`). It covers the full workflow end to end: image generation,
model training, and evaluation that goes beyond a single accuracy number.

## What it does

- **Data generation** (`src/generate_data.py`) procedurally creates
  grayscale steel-surface images on a textured, faintly-banded
  background. The three defect classes follow the familiar visual
  signatures of real steel defects: thin linear scratches, clusters of
  small dark pits, and diffuse brightness patches, alongside a clean
  no-defect class.
- **Model and training** (`src/model.py`) is a PyTorch CNN with two
  conv+pool blocks and two fully connected layers, trained on CPU with
  the Adam optimizer and cross-entropy loss.
- **Evaluation** uses scikit-learn metrics on a held-out test set that
  the model never sees during training: accuracy, per-class
  precision/recall/F1, and a confusion matrix.

## Scope

The images are synthetic, so the accuracy figures describe this
generated dataset. The pipeline is modular, so a real surface-defect
benchmark such as NEU-CLS can be dropped in by replacing the data
generation step.

## Results

On a synthetic 100/25/25-per-class train/val/test split, the trained
model reaches **96% test accuracy**. The confusion matrix adds useful
detail: `none` and `patches`, the two most visually distinct classes,
are classified correctly essentially every time, while `scratch` and
`pitting` are occasionally confused with each other (3 of 25 pitting
images predicted as scratch in the run captured by `run_pipeline.py`).
Both are localized dark features, so this is an interpretable pattern
that a small CNN naturally shows, and reporting it next to the headline
number gives a clearer picture of model behavior.

## Tests

```
$ python3 -m pytest tests/ -v
============================== 20 passed in ~15s ==============================
```

All 20 tests pass, including checks that the trained model clears
random-chance accuracy (0.25 for 4 balanced classes) by a wide margin on
held-out data, and that the two most distinct classes (`none`,
`patches`) are classified correctly at least 80% of the time. A GitHub
Actions workflow runs the tests and the pipeline demo on Python 3.11
and 3.12.

## Project structure

```
src/
  generate_data.py   -- synthetic steel-surface-defect image generator (4 classes)
  model.py           -- DefectCNN (PyTorch), training loop, evaluation
tests/
  test_generate_data.py
  test_model.py
run_pipeline.py      -- end-to-end demo: generate -> train -> evaluate -> report
```

## Running it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_pipeline.py
```
