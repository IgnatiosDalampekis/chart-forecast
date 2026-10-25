# chartcast

Turn a picture of a time series chart back into data, forecast it, and estimate how likely the forecast is to be
wrong.

You give it a screenshot of a line chart (a stock price, energy load, sales, anything with a value axis and a time
axis). It reads the axes with OCR, traces the line, rebuilds the series, and runs a rolling validation on the
chart's own history to measure how well each forecasting model would have done.

## Setup

Tested with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On first run EasyOCR downloads its models (~100 MB) into `.models/`.

## Using the app

1. Pick one of the test charts or upload your own.
2. Check the reconstruction: the original next to a redraw of what was read (axes, labels, line). For the
   synthetic samples the true data is drawn on top of the extracted series, so you can see the error.
3. Look at the overview: sampling, number of points, time range, value range, change, scale.
4. Choose a model, how far ahead to forecast, how much history to train on and how far the training window moves
   between validation runs. Units adapt to the chart (no minutes for a monthly chart).
5. Run it. You get the forecast as a continuation of the line, the mean error baseline, and for each future step
   the probability that the forecast misses by more than your tolerance.

## How it works

**Reading the chart.** The axes are found from their labels rather than from drawn lines, because labels are the
one thing every readable chart has. OCR reads all the text in the image; the value axis is the column of numbers
that increases upwards and sits on a single scale, the time axis is the row of dates that increases to the right.
Stray text (titles, prices, buttons, tables) fails these checks and is ignored. The label strips are then read a
second time, enlarged, to recover small or misread labels.

**Calibration.** A robust line fit through the labels maps every pixel to a value and a timestamp, so points
between labels are interpolated exactly. Axes are linear unless the labels fit a log scale clearly better (e.g. 10,
100, 1000 evenly spaced) or the chart says "log".

**Tracing.** The line is separated from gridlines, text and shading by colour, thinned to one pixel and read column
by column. The sampling frequency comes from the spacing of the line's corners (a line chart bends at every data
point), and the trace is resampled to one value per time step.

**Forecasting.** Four models: naive, seasonal naive, Holt-Winters and ARIMA. Each is validated by sliding a fixed
training window back through the history; at every position it forecasts the horizon and is compared with what the
chart actually shows. *Auto* picks the model with the lowest mean error; you can also pick the one with the lowest
probability of being wrong, or fix a model.

With `e(k, h) = |actual − forecast| / |actual|` for validation run `k` at `h` steps ahead:

```
mean error baseline = average of e(k, h) over all runs and steps
P(incorrect | h)    = share of runs with e(k, h) > tolerance
```

P(incorrect) is shown with a 95% Wilson interval, since it is estimated from a limited number of runs.

The whole pipeline is deterministic: the same image and settings always give the same result.

## Project layout

```
app.py               Streamlit app
chartcast/
  preprocess.py      image normalisation and ink masks
  ocr.py             text reading (EasyOCR)
  parse.py           label text -> numbers and dates
  axes.py            axis detection and plot area
  calibrate.py       pixel -> value / time
  line.py, trace.py  line isolation, tracing, resampling
  pipeline.py        image -> series, plus the drawings shown in the app
  forecast.py        models and rolling validation
  uncertainty.py     P(incorrect) and intervals
  synth.py           synthetic test charts with known data
samples/             real screenshots and synthetic charts (with their true data)
tests/
```

## Tests

```bash
pytest               # everything, ~1 min
pytest -m "not ocr"  # unit tests only, a few seconds
```

The end-to-end tests generate synthetic charts with known data and check the extraction against it. On those charts
the series comes out within about 1.5% of the value range (median), axes and plot area are found on all of them, and
the sampling frequency is right on 18 of 20.

New synthetic charts can be generated with `python -m chartcast.synth --n 6 --out samples/synthetic`.

## Limitations

- Line charts only; no bars, candlesticks or overlapping lines of the same colour.
- The chart needs readable tick labels on both axes.
- Smooth monthly data is sometimes read as weekly.
- Labels without a year (`Jan 05`) or with clock times only get a placeholder date.
- The relative error is unstable for values close to zero.
- Validation runs overlap, so the intervals on P(incorrect) are somewhat optimistic.
