"""
Synthetic steel-surface-defect image generator.

There is no real dataset here — no network access to any real surface-
defect dataset (e.g. the NEU-CLS benchmark) was available in this
environment (verified: kaggle.com, github.com, and the NEU faculty
mirror all failed to connect). Rather than skip the computer-vision
gap or pretend a real dataset was used, this module procedurally
generates grayscale steel-surface-style images with three classes of
synthetic defect patterns, deliberately modeled on the real, well-known
visual signatures of common steel surface defects:

  - "scratch"  -- one or more thin, high-contrast linear streaks
  - "pitting"  -- a cluster of small dark circular pits
  - "patches"  -- a diffuse, blob-shaped patch of altered brightness
  - "none"     -- clean surface with only background texture/noise

This is synthetic data, not a real manufacturing dataset -- disclosed
directly, the same way every sandbox-constrained project in this
portfolio discloses its real data sources. The point of this project is
to demonstrate a real, working, tested computer-vision pipeline
(image generation -> CNN training -> evaluation -> honest metrics),
not to claim production-grade defect-detection accuracy on real steel.
"""

import numpy as np

IMG_SIZE = 64
CLASSES = ["none", "scratch", "pitting", "patches"]


def _base_surface(rng, size=IMG_SIZE, base_gray=150, noise_std=8.0):
    """A synthetic steel-surface background: mid-gray with fine texture noise
    plus a few faint horizontal rolling-mill-style streaks."""
    img = np.full((size, size), base_gray, dtype=np.float64)
    img += rng.normal(0, noise_std, size=(size, size))
    # faint horizontal banding, like a rolled-metal texture
    for _ in range(rng.integers(2, 5)):
        y = rng.integers(0, size)
        band_strength = rng.uniform(-6, 6)
        img[y, :] += band_strength
    return img


def _add_scratch(img, rng):
    size = img.shape[0]
    n_scratches = rng.integers(1, 3)
    for _ in range(n_scratches):
        y0 = rng.integers(5, size - 5)
        x0 = rng.integers(0, size // 3)
        x1 = rng.integers(2 * size // 3, size)
        thickness = rng.integers(1, 3)
        darkness = rng.uniform(60, 110)
        slope = rng.uniform(-0.3, 0.3)
        for x in range(x0, x1):
            y = int(y0 + slope * (x - x0))
            for t in range(-thickness, thickness + 1):
                yy = y + t
                if 0 <= yy < size:
                    img[yy, x] -= darkness * (1.0 - abs(t) / (thickness + 1))
    return img


def _add_pitting(img, rng):
    size = img.shape[0]
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    n_pits = rng.integers(8, 18)
    for _ in range(n_pits):
        px = int(np.clip(cx + rng.normal(0, size // 8), 0, size - 1))
        py = int(np.clip(cy + rng.normal(0, size // 8), 0, size - 1))
        radius = rng.integers(1, 3)
        darkness = rng.uniform(50, 100)
        yy, xx = np.ogrid[-py:size - py, -px:size - px]
        mask = xx * xx + yy * yy <= radius * radius
        img[mask] -= darkness
    return img


def _add_patch(img, rng):
    size = img.shape[0]
    cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
    radius = rng.integers(size // 6, size // 3)
    brightness_shift = rng.choice([-1, 1]) * rng.uniform(25, 55)
    yy, xx = np.ogrid[:size, :size]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    falloff = np.exp(-dist2 / (2 * (radius ** 2)))
    img += brightness_shift * falloff
    return img


_DEFECT_FN = {
    "scratch": _add_scratch,
    "pitting": _add_pitting,
    "patches": _add_patch,
}


def generate_image(label, rng):
    """Generate one synthetic (size, size) float64 grayscale image in [0, 255]
    for the given label ('none', 'scratch', 'pitting', or 'patches')."""
    if label not in CLASSES:
        raise ValueError(f"Unknown label {label!r}, expected one of {CLASSES}")
    img = _base_surface(rng)
    if label != "none":
        img = _DEFECT_FN[label](img, rng)
    return np.clip(img, 0, 255)


def generate_dataset(n_per_class, seed=0):
    """Generate a balanced synthetic dataset.

    Returns (images, labels): images is a float64 array of shape
    (N, IMG_SIZE, IMG_SIZE) with values in [0, 255]; labels is a list of
    N class name strings, class-balanced and in a fixed, deterministic
    but shuffled order (not grouped by class), given a fixed seed.
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for label in CLASSES:
        for _ in range(n_per_class):
            images.append(generate_image(label, rng))
            labels.append(label)
    images = np.stack(images, axis=0)

    # shuffle deterministically so class order isn't trivially learnable
    # from position, and train/val/test splits (done elsewhere) see a mix
    perm = rng.permutation(len(labels))
    images = images[perm]
    labels = [labels[i] for i in perm]
    return images, labels
