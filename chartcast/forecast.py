"""Candidate models and rolling-origin validation with a fixed-size sliding training window."""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibrate import FREQS
from .errors import ChartError

MODELS = ("naive", "seasonal_naive", "holt_winters", "arima")
# Durations the user can pick for the horizon and the training window (seconds per unit).
UNITS = {"minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800, "months": 2629746, "years": 31556952}


@dataclass
class Backtest:
    model: str
    origins: np.ndarray    # index of the first forecast step per fold
    forecasts: np.ndarray  # (folds, horizon)
    actuals: np.ndarray    # (folds, horizon)
    window: int            # training samples per fold

    @property
    def errors(self) -> np.ndarray:
        """Relative absolute error per fold and horizon step."""
        return np.abs(self.actuals - self.forecasts) / np.maximum(np.abs(self.actuals), 1e-9)

    @property
    def mean_error(self) -> float:
        return float(self.errors.mean())

    def p_incorrect(self, tolerance: float) -> float:
        """Share of all fold × step forecasts that missed by more than `tolerance`."""
        return float((self.errors > tolerance).mean())


def to_steps(amount: float, unit: str, freq: str) -> int:
    """Duration -> number of samples at the series' frequency (1 day of hourly data = 24)."""
    return int(round(amount * UNITS[unit] / FREQS[freq]))


def _usable_period(period: int | None, n: int, cycles: int = 2) -> int | None:
    return period if period and period >= 2 and n >= cycles * period else None


def fit_predict(model: str, train: pd.Series, horizon: int, period: int | None) -> np.ndarray:
    """Fit one candidate model on `train` and forecast `horizon` steps."""
    y = train.to_numpy(dtype=float)
    p = _usable_period(period, len(y))
    if model == "naive":
        return np.repeat(y[-1], horizon)
    if model == "seasonal_naive":
        if p is None:
            return np.repeat(y[-1], horizon)
        return np.resize(y[-p:], horizon)
    if len(y) < 5:
        raise ChartError(f"{model} needs at least 5 training samples, got {len(y)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter from statsmodels
        if model == "holt_winters":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                       seasonal="add" if p else None, seasonal_periods=p).fit()
            return fit.forecast(horizon)
        if model == "arima":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            seasonal = (1, 0, 0, p) if p and p <= 52 and len(y) >= 3 * p else (0, 0, 0, 0)
            fit = SARIMAX(y, order=(1, 1, 1), seasonal_order=seasonal, trend="n").fit(disp=False)
            return fit.forecast(horizon)
    raise ValueError(f"unknown model: {model}")


MAX_FOLDS = 200  # a tiny step on a long series would otherwise refit the models hundreds of times


def fold_origins(n: int, horizon: int, window: int, step: int | None = None, folds: int = 20) -> np.ndarray:
    """Start of each fold's forecast. With a step, folds are laid back from the most recent data
    (so the last fold always tests the latest part of the chart); without one, ~`folds` evenly spaced."""
    last = n - horizon
    if step:
        return np.arange(last, window - 1, -step)[::-1]
    return np.unique(np.linspace(window, last, min(folds, last - window + 1)).astype(int))


def check_window(n: int, horizon: int, window: int, step: int | None = None) -> None:
    """Raise a readable error when the requested horizon/window/step do not fit the series."""
    if horizon < 1:
        raise ChartError("The forecast horizon is shorter than one sample of this chart. "
                         "Choose a longer horizon or a coarser unit.")
    if window < 2:
        raise ChartError("The training window is shorter than two samples of this chart. "
                         "Choose a longer window or a coarser unit.")
    if window + horizon > n:
        raise ChartError(f"The training window ({window} samples) plus the horizon ({horizon}) exceed the "
                         f"{n} samples extracted from the chart. Choose a shorter window or horizon.")
    if step is not None:
        if step < 1:
            raise ChartError("The step is shorter than one sample of this chart. Choose a longer step or a coarser unit.")
        k = len(fold_origins(n, horizon, window, step))
        if k < 2:
            raise ChartError(f"With this step only {k} validation fold fits. Choose a smaller step "
                             f"(at most {n - window - horizon} samples) or a shorter window or horizon.")
        if k > MAX_FOLDS:
            raise ChartError(f"This step gives {k} validation folds (at most {MAX_FOLDS}). Choose a larger step.")


def rolling_backtest(series: pd.Series, model: str, horizon: int, period: int | None,
                     window: int | None = None, folds: int = 20, step: int | None = None) -> Backtest:
    """Sliding window: train on the `window` samples before t, predict t..t+h-1, move t forward by `step`
    samples (default: ~`folds` evenly spaced folds)."""
    n = len(series)
    window = window or n // 2
    check_window(n, horizon, window, step)
    origins = fold_origins(n, horizon, window, step, folds)
    y = series.to_numpy(dtype=float)
    fc = np.array([fit_predict(model, series.iloc[t - window:t], horizon, period) for t in origins])
    act = np.array([y[t:t + horizon] for t in origins])
    return Backtest(model, origins, fc, act, window)


def future_index(series: pd.Series, horizon: int) -> pd.Index:
    freq = series.index.freq or pd.infer_freq(series.index) if isinstance(series.index, pd.DatetimeIndex) else None
    if freq is None:
        return pd.RangeIndex(len(series), len(series) + horizon)
    return pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]


def select_and_forecast(series: pd.Series, horizon: int, period: int | None, window: int | None = None,
                        models=MODELS, step: int | None = None, criterion: str = "mean_error",
                        tolerance: float = 0.10) -> tuple[pd.Series, Backtest, dict[str, Backtest]]:
    """Backtest the given models, refit the best one on the last `window` samples and forecast.

    criterion: 'mean_error' picks the lowest average error; 'p_incorrect' picks the model that most rarely
    missed by more than `tolerance` (ties broken by mean error). The two need not agree: one model can be
    usually very close but sometimes far off, another never great but never beyond the tolerance."""
    window = window or len(series) // 2
    check_window(len(series), horizon, window, step)
    results, errors = {}, {}
    for m in models:
        try:
            results[m] = rolling_backtest(series, m, horizon, period, window, step=step)
        except Exception as e:  # a model that cannot fit this window is skipped, not fatal
            errors[m] = str(e)
    if not results:
        raise ChartError("No model could be fitted with these settings ("
                         + "; ".join(f"{m}: {e}" for m, e in errors.items()) + "). Try a longer training window.")
    if criterion == "p_incorrect":
        best = min(results.values(), key=lambda b: (b.p_incorrect(tolerance), b.mean_error))
    else:
        best = min(results.values(), key=lambda b: b.mean_error)
    fc = fit_predict(best.model, series.iloc[-window:], horizon, period)
    return pd.Series(fc, index=future_index(series, horizon), name=best.model), best, results
