"""Stage 1: normalise the input image for the later stages."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import UNREADABLE, ChartError


@dataclass
class Prepared:
    color: np.ndarray    # BGR, upscaled, always light background
    gray: np.ndarray
    binary: np.ndarray   # ink = 255 (Otsu)
    neutral: np.ndarray  # low-saturation ink: axes, gridlines, text
    colored: np.ndarray  # saturated ink: usually the data series
    scale: float         # upscale factor applied to the original
    dark: bool           # input had a dark background (was inverted)


def load(path) -> np.ndarray:
    """Read an image file as BGR."""
    img = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise ChartError(UNREADABLE)
    return img


def prepare(img: np.ndarray, min_width: int = 1000, ink_delta: int = 25, min_sat: int = 60) -> Prepared:
    """Upscale, invert dark themes, binarise and split ink into neutral/coloured masks."""
    scale = 1.0
    if img.shape[1] < min_width:
        scale = min(3.0, min_width / img.shape[1])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = float(np.median(gray)) < 100  # the median pixel is background
    if dark:
        img, gray = 255 - img, 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturated = (hsv[..., 1] > min_sat) & (hsv[..., 2] > 40)
    darker = gray < np.median(gray) - ink_delta
    neutral = (darker & ~saturated).astype(np.uint8) * 255
    colored = saturated.astype(np.uint8) * 255

    return Prepared(img, gray, binary, neutral, colored, scale, dark)
