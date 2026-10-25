"""Errors caused by the input (not by a bug), with messages written for the user."""


class ChartError(ValueError):
    """The image or the requested forecast cannot be processed; str(e) is shown to the user."""


UNREADABLE = "This file couldn't be read as an image. Please upload a PNG or JPG chart."
TOO_SMALL = "The image is too small to read. Please upload a larger image of the chart."
NO_TEXT = "No text was found in this image. Please upload a chart with readable axis labels."
NO_VALUES = ("No value axis was found. Please upload an image with value data: "
             "numbers along the vertical axis (at least 2).")
NO_TIMES = ("No time axis was found. Please upload an image with time data: "
            "dates or times along the horizontal axis (at least 2).")
NO_LINE = "No data line was found inside the chart. Please upload a line chart."
TOO_FEW = ("Only {n} data points were extracted, too few to forecast. "
           "Please upload a chart covering a longer period.")
