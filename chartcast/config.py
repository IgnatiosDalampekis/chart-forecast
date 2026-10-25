"""Pipeline settings: one set, used for every chart (synthetic test charts and real screenshots)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    min_width: int = 1400        # upscale narrower images to this width (OCR needs readable text)
    min_conf: float = 0.3        # OCR confidence cut-off
    ocr_upscale: float = 2.0     # extra enlargement for the full-image OCR pass
    align_tol: float = 1.0       # label alignment tolerance, in text heights
    line_tol: float = 3.0        # max distance from a label group to its axis line, in text heights
    min_line_frac: float = 0.1   # straight lines shorter than this share of the image are ignored
    min_labels: int = 3          # preferred labels per axis (2 are accepted if nothing better exists)
    reread: bool = True          # second OCR pass, upscaled, on each axis' label strip
    min_samples: int = 10        # fewer extracted samples than this cannot be forecast


SETTINGS = Settings()
