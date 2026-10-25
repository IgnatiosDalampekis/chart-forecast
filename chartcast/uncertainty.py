"""Stage 9: empirical probability that the forecast is incorrect."""
from dataclasses import dataclass

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from .forecast import Backtest


@dataclass
class ErrorReport:
    tolerance: float
    mean_error: float          # mean relative error over all folds and steps
    p_incorrect: np.ndarray    # per horizon step
    ci_low: np.ndarray         # Wilson interval
    ci_high: np.ndarray
    folds: int


def error_report(bt: Backtest, tolerance: float = 0.10, alpha: float = 0.05) -> ErrorReport:
    """P(incorrect | h) = share of folds with relative error > tolerance at step h."""
    wrong = (bt.errors > tolerance).sum(axis=0)
    folds = bt.errors.shape[0]
    low, high = proportion_confint(wrong, folds, alpha=alpha, method="wilson")
    return ErrorReport(tolerance, bt.mean_error, wrong / folds, np.asarray(low), np.asarray(high), folds)
