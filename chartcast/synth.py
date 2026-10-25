"""Synthetic line charts with known ground truth.

    python -m chartcast.synth --n 12 --out samples/synthetic

Each chart is saved as an image plus a JSON file holding the true series,
the plot box in pixels, the axis limits and the style that was used.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

KINDS = {
    "energy": dict(freq="h", n=24 * 14, period=24, unit="MW", title="Hourly electricity load", fmt="%b %d"),
    "sales": dict(freq="MS", n=60, period=12, unit="Units", title="Monthly sales", fmt="%Y-%m"),
    "stock": dict(freq="D", n=250, period=None, unit="USD", title="Daily closing price", fmt="%b %Y"),
    "temperature": dict(freq="D", n=730, period=365, unit="°C", title="Daily mean temperature", fmt="%b %Y"),
    "traffic": dict(freq="W", n=156, period=52, unit="Visits", title="Weekly website visits", fmt="%b %Y"),
    "growth": dict(freq="MS", n=180, period=None, unit="USD", title="Long-term growth", fmt="%Y",
                   log=True),
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#000000"]


def make_series(kind: str, rng: np.random.Generator) -> pd.Series:
    cfg = KINDS[kind]
    n, period = cfg["n"], cfg["period"]
    t = np.arange(n)
    if kind == "stock":
        y = rng.uniform(20, 300) * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    elif kind == "growth":  # exponential growth over decades: ~3 -> ~10,000, shown on a log axis
        y = rng.uniform(2, 5) * np.exp(np.cumsum(rng.normal(0.045, 0.06, n)))
    elif kind == "temperature":
        y = 15 + 10 * np.sin(2 * np.pi * (t - 100) / period) + rng.normal(0, 1.5, n)
    else:
        level = rng.uniform(100, 1000)
        y = level * (1 + rng.uniform(-0.2, 0.5) * t / n)
        y += 0.25 * level * np.sin(2 * np.pi * t / period)
        if kind == "energy":  # weekly cycle on top of the daily one
            y += 0.1 * level * np.sin(2 * np.pi * t / (24 * 7))
        y += rng.normal(0, 0.04 * level, n)
    idx = pd.date_range("2023-01-01", periods=n, freq=cfg["freq"])
    return pd.Series(y, index=idx, name=kind)


def random_style(rng: np.random.Generator) -> dict:
    return dict(
        color=str(rng.choice(COLORS)),
        linewidth=float(rng.uniform(1.0, 3.0)),
        grid=bool(rng.random() < 0.6),
        dark=bool(rng.random() < 0.2),
        all_spines=bool(rng.random() < 0.5),
        figsize=(float(rng.uniform(8, 12)), float(rng.uniform(4, 6))),
        jpeg=int(rng.integers(60, 95)) if rng.random() < 0.3 else None,
    )


def make_chart(kind: str, seed: int, out: Path) -> Path:
    """Render one chart (random style) to PNG/JPG and write its ground truth JSON next to it."""
    rng = np.random.default_rng(seed)
    series = make_series(kind, rng)
    style = random_style(rng)
    if style["dark"] and style["color"] == "#000000":
        style["color"] = "#ffd700"

    with plt.style.context("dark_background" if style["dark"] else "default"):
        fig, ax = plt.subplots(figsize=style["figsize"], dpi=100)
        ax.plot(series.index, series.values, color=style["color"], lw=style["linewidth"])
        ax.set_title(KINDS[kind]["title"])
        ax.set_ylabel(KINDS[kind]["unit"])
        if KINDS[kind].get("log"):  # log value axis with plain labels: 10, 100, 1000, …
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(KINDS[kind]["fmt"]))
        if style["grid"]:
            ax.grid(True, alpha=0.5)
        if not style["all_spines"]:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()

        H = fig.canvas.get_width_height()[1]
        bb = ax.get_window_extent()
        box = [round(bb.x0), round(H - bb.y1), round(bb.x1), round(H - bb.y0)]
        xlim = [mdates.num2date(v).replace(tzinfo=None).isoformat() for v in ax.get_xlim()]

        name = f"{kind}_{seed:03d}"
        img_path = out / f"{name}.{'jpg' if style['jpeg'] else 'png'}"
        save_kw = {"pil_kwargs": {"quality": style["jpeg"]}} if style["jpeg"] else {}
        fig.savefig(img_path, dpi=100, **save_kw)
        plt.close(fig)

    truth = dict(
        kind=kind, freq=KINDS[kind]["freq"], period=KINDS[kind]["period"], log=bool(KINDS[kind].get("log")),
        unit=KINDS[kind]["unit"], title=KINDS[kind]["title"],
        plot_box=box, ylim=list(ax.get_ylim()), xlim=xlim, style=style,
        timestamps=[t.isoformat() for t in series.index], values=series.round(4).tolist(),
    )
    img_path.with_suffix(".json").write_text(json.dumps(truth))
    return img_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("samples/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kinds = list(KINDS)
    for i in range(args.n):
        print(make_chart(kinds[i % len(kinds)], args.seed + i, args.out))


if __name__ == "__main__":
    main()
