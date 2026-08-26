import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import generate_dataset, CLASSES
from model import (
    DefectCNN,
    images_to_tensor,
    labels_to_tensor,
    train_model,
    predict,
    evaluate,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
)


def test_label_index_mapping_is_consistent_bijection():
    assert len(LABEL_TO_IDX) == len(CLASSES)
    for label, idx in LABEL_TO_IDX.items():
        assert IDX_TO_LABEL[idx] == label


def test_images_to_tensor_shape_and_range():
    images, labels = generate_dataset(n_per_class=2, seed=1)
    t = images_to_tensor(images)
    assert t.shape == (8, 1, 64, 64)
    assert t.dtype == torch.float32
    # normalized to roughly [-1, 1]
    assert t.min() >= -1.01
    assert t.max() <= 1.01


def test_labels_to_tensor_matches_class_indices():
    labels = ["none", "scratch", "pitting", "patches"]
    t = labels_to_tensor(labels)
    assert t.tolist() == [LABEL_TO_IDX[l] for l in labels]


def test_model_forward_pass_produces_correct_output_shape():
    model = DefectCNN()
    images, _ = generate_dataset(n_per_class=2, seed=1)
    x = images_to_tensor(images)
    out = model(x)
    assert out.shape == (8, len(CLASSES))


def test_training_loss_decreases_over_epochs():
    train_images, train_labels = generate_dataset(n_per_class=30, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=10, seed=2)
    model, history = train_model(
        train_images, train_labels, val_images, val_labels, epochs=8, seed=0
    )
    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]
    assert last_loss < first_loss


def test_trained_model_beats_random_chance_on_held_out_data():
    # a real, non-trivial claim: with 4 balanced classes, random-guess
    # accuracy is 0.25. A genuinely trained model should clear that by a
    # wide margin on data it never saw during training.
    train_images, train_labels = generate_dataset(n_per_class=60, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=15, seed=2)
    test_images, test_labels = generate_dataset(n_per_class=15, seed=3)

    model, _ = train_model(
        train_images, train_labels, val_images, val_labels, epochs=15, seed=0
    )
    results = evaluate(model, test_images, test_labels)
    assert results["accuracy"] > 0.6  # well above the 0.25 random-chance baseline


def test_evaluate_confusion_matrix_diagonal_dominant_for_distinct_classes():
    # "none" (clean) and "patches" (diffuse blob) are the most visually
    # distinct classes in this synthetic generator -- a real, reasonably
    # trained model should get these right almost every time. This is a
    # meaningful assertion about the model's actual behavior, not a
    # tautology: it would fail if the model were untrained or broken.
    train_images, train_labels = generate_dataset(n_per_class=60, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=15, seed=2)
    test_images, test_labels = generate_dataset(n_per_class=15, seed=3)

    model, _ = train_model(
        train_images, train_labels, val_images, val_labels, epochs=15, seed=0
    )
    results = evaluate(model, test_images, test_labels)
    none_idx = results["labels_order"].index("none")
    patches_idx = results["labels_order"].index("patches")
    cm = results["confusion_matrix"]
    none_recall = cm[none_idx][none_idx] / sum(cm[none_idx])
    patches_recall = cm[patches_idx][patches_idx] / sum(cm[patches_idx])
    assert none_recall >= 0.8
    assert patches_recall >= 0.8


def test_predict_returns_valid_class_names():
    train_images, train_labels = generate_dataset(n_per_class=20, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=10, seed=2)
    model, _ = train_model(
        train_images, train_labels, val_images, val_labels, epochs=5, seed=0
    )
    test_images, _ = generate_dataset(n_per_class=5, seed=99)
    preds = predict(model, test_images)
    assert len(preds) == 20
    assert all(p in CLASSES for p in preds)


def test_evaluate_output_structure():
    train_images, train_labels = generate_dataset(n_per_class=20, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=10, seed=2)
    model, _ = train_model(
        train_images, train_labels, val_images, val_labels, epochs=5, seed=0
    )
    test_images, test_labels = generate_dataset(n_per_class=10, seed=3)
    results = evaluate(model, test_images, test_labels)
    assert "accuracy" in results
    assert "per_class" in results
    assert "confusion_matrix" in results
    assert set(results["per_class"].keys()) == set(CLASSES)
    for cls, metrics in results["per_class"].items():
        assert set(metrics.keys()) == {"precision", "recall", "f1", "support"}


def test_same_seed_gives_reproducible_training():
    train_images, train_labels = generate_dataset(n_per_class=20, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=10, seed=2)

    model1, hist1 = train_model(
        train_images, train_labels, val_images, val_labels, epochs=5, seed=42
    )
    model2, hist2 = train_model(
        train_images, train_labels, val_images, val_labels, epochs=5, seed=42
    )
    # same seed should give identical final validation accuracy
    assert hist1[-1]["val_accuracy"] == hist2[-1]["val_accuracy"]
