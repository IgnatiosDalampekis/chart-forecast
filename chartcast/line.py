"""Stage 6: isolate the plotted data line inside the plot box."""
import cv2
import numpy as np

from .axes import PlotArea
from .ocr import TextBox
from .preprocess import Prepared


def _cleanup(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Close anti-aliasing gaps and drop specks."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8) * 255


def _cluster_colors(prep: Prepared, region: np.ndarray, max_k: int, merge_de: float) -> list[np.ndarray]:
    """k-means on the Lab colour of coloured ink; clusters closer than merge_de (ΔE) are merged."""
    ys, xs = np.nonzero(region)
    lab = cv2.cvtColor(prep.color, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[ys, xs]
    k = min(max_k, len(samples))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)  # k-means++ seeding is random: fix it so the same image always gives the same mask
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()

    # Merge clusters that are really one colour (anti-aliased shades of the same line).
    group = list(range(k))
    for i in range(k):
        for j in range(i):
            if np.linalg.norm(centers[i] - centers[j]) < merge_de:
                group[i] = group[j]
                break
    masks = []
    for g in sorted(set(group)):
        m = np.zeros(region.shape, np.uint8)
        sel = np.isin(labels, [i for i in range(k) if group[i] == g])
        m[ys[sel], xs[sel]] = 255
        masks.append(m)
    return masks


def isolate_lines(prep: Prepared, area: PlotArea, texts: list[TextBox], max_k: int = 5,
                  merge_de: float = 25.0, min_cover: float = 0.3) -> list[np.ndarray]:
    """Mask text and gridlines, drop background, cluster ink in Lab space.

    Returns one binary mask per detected series, widest first.
    """
    x0, y0, x1, y1 = area.box
    inside = np.zeros(prep.gray.shape, np.uint8)
    inside[y0 + 3:y1 - 2, x0 + 3:x1 - 2] = 255  # shrink inward so the axis lines are excluded
    for t in texts:
        # OCR also "reads" letters in zigzags of the data line itself (W, N, 4): only text-sized
        # boxes with 2+ characters are real text to blank out.
        if t.height > 2 * area.text_height or len(t.text.strip()) < 2:
            continue
        a, b, c, d = t.box
        inside[max(0, b - 2):d + 2, max(0, a - 2):c + 2] = 0

    colored = cv2.bitwise_and(prep.colored, inside)
    width = x1 - x0
    candidates = _cluster_colors(prep, colored, max_k, merge_de) if colored.sum() // 255 > width // 2 else []

    if not candidates:
        # Monochrome chart: dark neutral ink minus gridlines and any long straight lines.
        neutral = cv2.bitwise_and(prep.neutral, inside)
        for r in area.h_grid:
            neutral[max(0, r - 2):r + 3] = 0
        for c in area.v_grid:
            neutral[:, max(0, c - 2):c + 3] = 0
        long_h = cv2.morphologyEx(neutral, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8))
        long_v = cv2.morphologyEx(neutral, cv2.MORPH_OPEN, np.ones((40, 1), np.uint8))
        candidates = [cv2.subtract(neutral, cv2.bitwise_or(long_h, long_v))]

    masks = [_cleanup(m) for m in candidates]
    cover = [np.count_nonzero(m.any(0)) / max(width, 1) for m in masks]
    ranked = sorted(zip(cover, range(len(masks))), reverse=True)
    return [masks[i] for c, i in ranked if c >= min_cover]
