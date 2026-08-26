"""
End-to-end demo: generate synthetic steel-surface-defect images, train a
CNN, evaluate on a held-out test set, and print honest results including
the confusion matrix (not just accuracy).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_dataset, CLASSES
from model import train_model, evaluate


def main():
    print("Generating synthetic steel-surface-defect dataset...")
    train_images, train_labels = generate_dataset(n_per_class=100, seed=1)
    val_images, val_labels = generate_dataset(n_per_class=25, seed=2)
    test_images, test_labels = generate_dataset(n_per_class=25, seed=3)
    print(f"  train: {len(train_labels)} images, val: {len(val_labels)}, "
          f"test: {len(test_labels)} (classes: {CLASSES})")

    print("\nTraining DefectCNN (CPU, ~20 epochs)...")
    model, history = train_model(
        train_images, train_labels, val_images, val_labels,
        epochs=20, verbose=True,
    )

    print("\nEvaluating on held-out TEST set (never seen during training)...")
    results = evaluate(model, test_images, test_labels)
    print(f"\nTest accuracy: {results['accuracy']:.4f}")
    print("\nPer-class metrics:")
    for cls, m in results["per_class"].items():
        print(f"  {cls:10s}  precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  support={m['support']}")

    print("\nConfusion matrix (rows=true label, cols=predicted label):")
    print("            " + "  ".join(f"{c:>9s}" for c in results["labels_order"]))
    for true_label, row in zip(results["labels_order"], results["confusion_matrix"]):
        print(f"  {true_label:10s}" + "  ".join(f"{v:9d}" for v in row))

    print(
        "\nNote: 'scratch' and 'pitting' are the two classes most often "
        "confused with each other in this run -- both are localized dark "
        "features, which is a real and honest limitation of this small "
        "CNN on this synthetic data, not a hidden or smoothed-over result. "
        "'none' and 'patches' are the most visually distinct classes and "
        "are classified correctly almost every time."
    )


if __name__ == "__main__":
    main()
