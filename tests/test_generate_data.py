import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_data import generate_image, generate_dataset, CLASSES, IMG_SIZE


def test_all_classes_generate_correct_shape():
    rng = np.random.default_rng(0)
    for c in CLASSES:
        img = generate_image(c, rng)
        assert img.shape == (IMG_SIZE, IMG_SIZE)


def test_unknown_label_raises():
    rng = np.random.default_rng(0)
    try:
        generate_image("not_a_real_class", rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pixel_values_within_valid_range():
    rng = np.random.default_rng(0)
    for c in CLASSES:
        img = generate_image(c, rng)
        assert img.min() >= 0
        assert img.max() <= 255


def test_dataset_is_class_balanced():
    images, labels = generate_dataset(n_per_class=10, seed=5)
    assert images.shape == (40, IMG_SIZE, IMG_SIZE)
    from collections import Counter
    counts = Counter(labels)
    for c in CLASSES:
        assert counts[c] == 10


def test_dataset_is_shuffled_not_grouped_by_class():
    # with n_per_class > 1 and a real shuffle, the first n_per_class labels
    # should not all be identical (this would fail if generate_dataset
    # forgot to shuffle)
    images, labels = generate_dataset(n_per_class=20, seed=7)
    first_block = labels[:20]
    assert len(set(first_block)) > 1


def test_same_seed_is_reproducible():
    images1, labels1 = generate_dataset(n_per_class=5, seed=123)
    images2, labels2 = generate_dataset(n_per_class=5, seed=123)
    assert labels1 == labels2
    assert np.allclose(images1, images2)


def test_different_seeds_produce_different_images():
    images1, _ = generate_dataset(n_per_class=5, seed=1)
    images2, _ = generate_dataset(n_per_class=5, seed=2)
    assert not np.allclose(images1, images2)


def test_scratch_images_are_darker_on_average_than_clean_surface():
    # a real sanity check on the defect-generation logic itself: a scratch
    # should measurably darken the average pixel value relative to a clean
    # surface generated with the same seed sequence up to that point
    rng_clean = np.random.default_rng(42)
    clean = generate_image("none", rng_clean)

    rng_scratch = np.random.default_rng(42)
    scratched = generate_image("scratch", rng_scratch)

    assert scratched.mean() < clean.mean()


def test_pitting_produces_localized_dark_pixels():
    rng = np.random.default_rng(9)
    img = generate_image("pitting", rng)
    # pitting should create some genuinely dark pixels (pits), not just
    # uniform mid-gray noise
    assert img.min() < 100


def test_patches_can_be_lighter_or_darker_than_background():
    # patches shift brightness in a randomly-chosen direction; across many
    # seeds we should see both directions occur (otherwise the "choice"
    # logic would be broken / always picking the same sign)
    saw_darker = False
    saw_lighter = False
    for seed in range(30):
        rng_clean = np.random.default_rng(seed)
        clean = generate_image("none", rng_clean)
        rng_patch = np.random.default_rng(seed)
        patched = generate_image("patches", rng_patch)
        diff = patched.mean() - clean.mean()
        if diff < -1:
            saw_darker = True
        if diff > 1:
            saw_lighter = True
    assert saw_darker and saw_lighter
