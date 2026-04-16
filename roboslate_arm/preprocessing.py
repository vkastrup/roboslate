"""
Image preprocessing for difficult slate frames.

Used when a first-pass extraction returns low confidence — applies contrast
enhancement, sharpening, and brightness normalisation before a retry.

Uses only Pillow (already a dependency). For more aggressive CLAHE,
opencv-python-headless can be added and will be used automatically if present.
"""

import os
import tempfile

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    Image = None


def _require_pillow():
    if Image is None:
        raise ImportError("Pillow is not installed. Run: venv/bin/pip install Pillow")


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def normalize_brightness(img):
    """
    Autolevels: stretch the histogram so the darkest pixel = 0, brightest = 255.
    Handles backlit or underexposed frames where the slate is in shadow.
    """
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=2)


def equalize_histogram(img):
    """
    Global histogram equalisation. Increases contrast in low-contrast frames.
    Less targeted than CLAHE but available without OpenCV.
    """
    return ImageOps.equalize(img.convert("RGB"))


def clahe_if_available(img):
    """
    Apply CLAHE if opencv-python-headless is installed; fall back to equalise.
    CLAHE is better for slates because it enhances locally — good for frames
    where the slate is in a bright/dark area surrounded by very different lighting.
    """
    try:
        import cv2
        import numpy as np

        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        img_lab_eq = cv2.merge((l_eq, a, b))
        img_rgb_eq = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb_eq)

    except ImportError:
        return equalize_histogram(img)


def sharpen(img, radius=1.5, percent=150, threshold=2):
    """
    Unsharp mask sharpening. Helps with soft/slightly-blurry text on slates.
    """
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def boost_contrast(img, factor=1.4):
    """Boost midtone contrast. Helps text pop against the slate background."""
    return ImageEnhance.Contrast(img).enhance(factor)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def preprocess_for_extraction(img):
    """
    Standard preprocessing pipeline for low-confidence slate frames.
    Order matters: normalise first, then enhance contrast, then sharpen.

    Returns a PIL Image.
    """
    _require_pillow()
    img = img.convert("RGB")
    img = normalize_brightness(img)
    img = clahe_if_available(img)
    img = boost_contrast(img, factor=1.3)
    img = sharpen(img)
    return img


def preprocess_frame_file(input_path, output_path=None):
    """
    Preprocess a frame file on disk and return the path to the preprocessed version.

    Args:
        input_path: Path to source JPEG frame.
        output_path: Where to write the preprocessed file. If None, writes to a
                     temp file alongside the original.

    Returns:
        Path to the preprocessed JPEG file.
    """
    _require_pillow()

    with Image.open(input_path) as img:
        processed = preprocess_for_extraction(img)

    if output_path is None:
        stem, ext = os.path.splitext(input_path)
        output_path = stem + "_pre" + ext

    processed.save(output_path, format="JPEG", quality=90)
    return output_path
