"""Read all chart text with EasyOCR. Roles (axis label, title, …) are decided later by axes.py."""
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1

NUMBER_CHARS = "0123456789.,-−+%$€£¥₹()kKMBbnTeE×^' °"
MODEL_DIR = Path(__file__).resolve().parent.parent / ".models"


@dataclass
class TextBox:
    text: str
    box: Box
    conf: float
    role: str = "other"  # value_label | time_label | other

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


_reader_lock = threading.Lock()  # a background warm-up and a first chart must not load the model twice


@lru_cache(maxsize=1)
def _load_reader():
    import easyocr  # heavy import (torch); only load when OCR is actually needed

    MODEL_DIR.mkdir(exist_ok=True)
    return easyocr.Reader(["en"], gpu=False, verbose=False,
                          model_storage_directory=str(MODEL_DIR), user_network_directory=str(MODEL_DIR))


def _reader():
    with _reader_lock:
        return _load_reader()


def ocr_ready() -> bool:
    """True once the EasyOCR model is loaded (it stays in memory for the life of the process)."""
    return _load_reader.cache_info().currsize > 0


def warm_up() -> None:
    """Load the model and run one tiny read, so the first real chart skips loading and warm-up."""
    img = np.full((48, 160, 3), 255, np.uint8)
    cv2.putText(img, "2024", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    _reader().readtext(img)


def _run(img: np.ndarray, offset=(0, 0), upscale: float = 1.0, allowlist: str | None = None) -> list[TextBox]:
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    out = []
    # EasyOCR's own box merging is left at its default; merge_words joins words more carefully.
    for pts, text, conf in _reader().readtext(img, allowlist=allowlist):
        pts = np.asarray(pts, dtype=float) / upscale
        x0, y0 = pts.min(0) + offset
        x1, y1 = pts.max(0) + offset
        out.append(TextBox(text.strip(), (int(x0), int(y0), int(x1), int(y1)), float(conf)))
    return out


_NUMERIC = re.compile(r"^[\s$€£+\-−(]*[\dOolISB.,]+[)kKmMbB%]?$")


def clean_numeric(text: str) -> str:
    """Fix common OCR confusions in strings that should be numbers ('1O0' -> '100')."""
    if not _NUMERIC.match(text) or not any(c.isdigit() for c in text):
        return text
    table = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "−": "-"})
    head, suffix = (text[:-1], text[-1]) if text[-1] in "kKmMbB%)" else (text, "")
    return head.translate(table).replace(" ", "") + suffix


def merge_words(texts: list[TextBox], max_gap: float = 1.0) -> list[TextBox]:
    """Join words on one line ('Sep' + '2023') when they have similar height (a rotated axis title
    beside a label does not) and the gap is under max_gap × text height (small overlaps allowed)."""
    out: list[TextBox] = []
    for t in sorted(texts, key=lambda t: t.box[0]):
        for i, p in enumerate(out):
            h = max(p.height, t.height)
            similar = min(p.height, t.height) / h > 0.6
            gap = t.box[0] - p.box[2]
            if similar and abs(p.center[1] - t.center[1]) < 0.5 * h and -0.5 * h <= gap < max_gap * h:
                box = (p.box[0], min(p.box[1], t.box[1]), t.box[2], max(p.box[3], t.box[3]))
                out[i] = TextBox(f"{p.text} {t.text}", box, min(p.conf, t.conf))
                break
        else:
            out.append(t)
    return out


def read_text(img: np.ndarray, min_conf: float = 0.4, upscale: float = 2.0) -> list[TextBox]:
    """Every text in the image with its box and confidence, words on one line merged.
    Upscaled because tick labels are often only ~10 px tall."""
    texts = [t for t in _run(img, upscale=upscale) if t.conf >= min_conf and t.text]
    for t in texts:
        t.text = clean_numeric(t.text)
    return merge_words(texts)


def reread(img: np.ndarray, region: Box, min_conf: float = 0.4, allowlist: str | None = None) -> list[TextBox]:
    """Second pass on one region (an axis label strip): crop, upscale 2×, optional character set."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, region[0]), max(0, region[1]), min(w, region[2]), min(h, region[3])
    if x1 <= x0 or y1 <= y0:
        return []
    texts = [t for t in _run(img[y0:y1, x0:x1], offset=(x0, y0), upscale=2.0, allowlist=allowlist)
             if t.conf >= min_conf and t.text]
    for t in texts:
        t.text = clean_numeric(t.text)
    return merge_words(texts)
