"""Turn tick-label text into numbers and timestamps.

Without a VLM the format coverage has to be explicit, so both parsers accept a broad set of
notations. Anything they do not recognise returns None and is simply not an axis candidate.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "‒": "-"})

# --------------------------------------------------------------------------- numbers

_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_CURRENCY = re.compile(r"[$€£¥₹₩₽]|\b(?:USD|EUR|GBP|JPY|CHF|CNY)\b", re.I)
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")                  # 3, -3.5, 1e5, 1.2E+03
_POW10 = re.compile(r"(?:(\d+(?:\.\d+)?)\s*[x×·*]\s*)?10\s*(?:\^\s*([+-]?\d+)|([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))")  # 10^3, 2×10³
_SCALE = {"k": 1e3, "K": 1e3, "M": 1e6, "mn": 1e6, "MM": 1e6, "B": 1e9, "b": 1e9, "bn": 1e9,
          "G": 1e9, "T": 1e12, "tn": 1e12}
_NUMBER = re.compile(r"""
    (?P<num>[+-]?(?:\d[\d.,'   ]*\d|\d))       # digits with optional separators
    \s*(?P<scale>mn|MM|bn|tn|[kKMBbGT])?(?![A-Za-z])     # 1.2k, 3M, 2bn (not the k of kWh)
    \s*(?P<unit>%|‰|°\s*[CF]?|[A-Za-z°/µ²³]{1,6})?       # 20%, 15°C, 300 MW, 5 kWh
    """, re.X)


def _to_float(num: str) -> float:
    """Resolve thousands vs decimal separators: '1,234' '1.234.567' '1 234' '1,5' '1,234.5'."""
    sign = -1.0 if num.startswith("-") else 1.0
    num = re.sub(r"['   ]", "", num.lstrip("+-"))
    if "," in num and "." in num:          # the last separator is the decimal one
        dec = max(num.rfind(","), num.rfind("."))
        num = re.sub(r"[.,]", "", num[:dec]) + "." + num[dec + 1:]
    elif num.count(",") > 1 or re.fullmatch(r"\d{1,3},\d{3}", num):
        num = num.replace(",", "")
    elif "," in num:
        num = num.replace(",", ".")
    elif num.count(".") > 1:
        num = num.replace(".", "")
    return sign * float(num)


def parse_number(text: str) -> float | None:
    """'1,200' '1.2k' '$50' '50 €' '20%' '−3' '(5)' '1e5' '10³' '2×10^3' '15°C' '300 MW' -> float."""
    t = text.strip().translate(_DASHES)
    if not t:
        return None
    if m := re.fullmatch(r"\((.+)\)", t):                  # accounting negative: (5) = -5
        v = parse_number(m.group(1))
        return -v if v is not None else None
    if m := _POW10.fullmatch(t):
        mant = float(m.group(1)) if m.group(1) else 1.0
        exp = int(m.group(2) if m.group(2) is not None else m.group(3).translate(_SUPERSCRIPT))
        return mant * 10.0 ** exp
    t = _CURRENCY.sub("", t).strip()
    if _FLOAT.fullmatch(t):
        return float(t)
    m = _NUMBER.fullmatch(t)
    if not m:
        return None
    try:
        return _to_float(m["num"]) * _SCALE.get(m["scale"] or "", 1.0)
    except ValueError:
        return None


# --------------------------------------------------------------------------- dates and times

@dataclass
class ParsedTime:
    dt: datetime          # UTC; fields missing from the label are filled with placeholders
    unit: str             # finest field in the label: second | minute | hour | day | week | month | quarter | year
    wrap: str | None      # field the label lacks, so a sequence may wrap: 'day' | 'week' | 'year'


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PLACEHOLDER_YEAR = 2000  # leap year, so 29 Feb is valid


def _name_index(token: str, names: list[str]) -> int | None:
    """'Jan' 'jan.' 'Sept' 'September' -> index; needs at least 3 letters."""
    token = token.rstrip(".")
    if len(token) < 3:
        return None
    for i, name in enumerate(names):
        if name.startswith(token):
            return i
    return None


def _year(token: str) -> int:
    y = int(token.lstrip("'"))
    if y < 100:
        y += 2000 if y < 70 else 1900
    return y


def _hour(h: int, ampm: str | None) -> int:
    if ampm == "a":
        return 0 if h == 12 else h
    if ampm == "p":
        return h if h == 12 else h + 12
    return h


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


_TIME = r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?: ?([ap])\.?m\.?)?"
# OCR often reads the colon as a dot; accept '12.00 pm' only with am/pm, so '12.00' stays a number.
_TIME_DOT = r"(\d{1,2})\.(\d{2})()(?: ?([ap])\.?m\.?)"


def _build_patterns():
    pats = []

    def add(rx, fn):
        pats.append((re.compile(rx), fn))

    def hms(m, i):
        h, mi, s, ap = m.group(i), m.group(i + 1), m.group(i + 2), m.group(i + 3)
        return _hour(int(h), ap), int(mi), int(s or 0), ("second" if s else "minute")

    def iso_dt(m):
        h, mi, s, unit = hms(m, 4)
        return ParsedTime(_utc(int(m[1]), int(m[2]), int(m[3]), h, mi, s), unit, None)

    def dmy(m):
        a, sep, b, y = int(m[1]), m[2], int(m[3]), _year(m[4])
        d, mo = (b, a) if sep == "/" and a <= 12 else (a, b)  # '/' is month-first unless impossible
        return ParsedTime(_utc(y, mo, d), "day", None)

    def md(m):
        a, b = int(m[1]), int(m[2])
        d, mo = (a, b) if a > 12 else (b, a)
        return ParsedTime(_utc(_PLACEHOLDER_YEAR, mo, d), "day", "year")

    def time_only(m):
        h, mi, s, unit = hms(m, 1)
        return ParsedTime(_utc(_PLACEHOLDER_YEAR, 1, 1, h, mi, s), unit, "day")

    def quarter(q, y):
        return ParsedTime(_utc(_year(y) if y else _PLACEHOLDER_YEAR, 3 * int(q) - 2, 1), "quarter",
                          None if y else "year")

    def half(hf, y):
        return ParsedTime(_utc(_year(y) if y else _PLACEHOLDER_YEAR, 6 * int(hf) - 5, 1), "quarter",
                          None if y else "year")

    def week(w, y):
        d = date.fromisocalendar(_year(y) if y else _PLACEHOLDER_YEAR, int(w), 1)
        return ParsedTime(_utc(d.year, d.month, d.day), "week", None if y else "year")

    add(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})[ t]" + _TIME, iso_dt)                       # 2024-01-05 14:00
    add(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", lambda m: ParsedTime(_utc(int(m[1]), int(m[2]), int(m[3])), "day", None))
    add(r"(\d{4})[-/.](\d{1,2})", lambda m: ParsedTime(_utc(int(m[1]), int(m[2]), 1), "month", None))  # 2024-01
    add(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})", dmy)                                 # 05/01/2024, 5.1.24
    add(r"(\d{1,2})/(\d{1,2})", md)                                                       # 1/5
    add(_TIME, time_only)                                                                 # 14:00, 2:30 pm
    add(_TIME_DOT, time_only)                                                             # 12.00 p.m. (OCR)
    add(r"(\d{1,2}) ?([ap])\.?m\.?", lambda m: ParsedTime(_utc(_PLACEHOLDER_YEAR, 1, 1, _hour(int(m[1]), m[2])), "hour", "day"))
    add(r"(\d{1,2})h", lambda m: ParsedTime(_utc(_PLACEHOLDER_YEAR, 1, 1, int(m[1])), "hour", "day"))  # 14h
    add(r"q([1-4])(?:[ '-]*(\d{4}|\d{2}))?", lambda m: quarter(m[1], m[2]))               # Q1, Q1 2024, Q1 '24
    add(r"(\d{4})[ -]?q([1-4])", lambda m: quarter(m[2], m[1]))                           # 2024Q1
    add(r"h([12])(?:[ '-]*(\d{4}|\d{2}))?", lambda m: half(m[1], m[2]))                   # H1 2024
    add(r"(\d{4})[ -]?h([12])", lambda m: half(m[2], m[1]))
    add(r"fy ?'?(\d{4}|\d{2})", lambda m: ParsedTime(_utc(_year(m[1]), 1, 1), "year", None))  # FY24
    add(r"(?:w|wk|week) ?(\d{1,2})(?:[ ,'-]+(\d{4}|\d{2}))?", lambda m: week(m[1], m[2]))  # W12, Week 12 2024
    add(r"(1[89]\d{2}|2[01]\d{2})", lambda m: ParsedTime(_utc(int(m[1]), 1, 1), "year", None))  # 2024
    add(r"'(\d{2})", lambda m: ParsedTime(_utc(_year(m[1]), 1, 1), "year", None))          # '24
    return pats


_PATTERNS = _build_patterns()


def _free_form(t: str) -> ParsedTime | None:
    """Month / weekday names with optional day, year and time: 'Jan', 'Jan 2024', '5 Jan',
    'January 5, 2024', 'Jan 5 14:00', 'Mon', 'Mon 14:00'."""
    clock = re.search(_TIME, t)
    h = mi = s = 0
    unit_time = None
    if clock:
        h, mi, s = _hour(int(clock[1]), clock[4]), int(clock[2]), int(clock[3] or 0)
        unit_time = "second" if clock[3] else "minute"
        t = t[:clock.start()] + " " + t[clock.end():]
    month = weekday = day = year = None
    for tok in re.split(r"[ ,/.-]+", t.strip()):
        if not tok:
            continue
        tok = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", tok)  # 5th -> 5
        if tok.isalpha():
            if (i := _name_index(tok, _MONTHS)) is not None and month is None:
                month = i + 1
            elif (i := _name_index(tok, _WEEKDAYS)) is not None:
                weekday = i
            else:
                return None  # some other word: not a date label
        elif re.fullmatch(r"'\d{2}|\d{4}", tok):
            year = _year(tok)
        elif re.fullmatch(r"\d{1,2}", tok) and day is None and 1 <= int(tok) <= 31:
            day = int(tok)
        else:
            return None
    try:
        if month is not None:
            dt = _utc(year or _PLACEHOLDER_YEAR, month, day or 1, h, mi, s)
            unit = unit_time or ("day" if day else "month")
            return ParsedTime(dt, unit, None if year else "year")
        if weekday is not None and day is None and year is None:
            dt = _utc(_PLACEHOLDER_YEAR, 1, 3, h, mi, s) + timedelta(days=weekday)  # 3 Jan 2000 = Monday
            return ParsedTime(dt, unit_time or "day", "week")
    except ValueError:  # e.g. 31 Feb
        return None
    return None


def parse_date(text: str) -> ParsedTime | None:
    """Dates, times and periods in common notations (ISO, numeric, month/weekday names,
    quarters, halves, fiscal years, weeks, clock times with or without am/pm)."""
    # Stray trailing punctuation is a common OCR artefact ('4:00 p.m_', '12:00 p.m:').
    t = re.sub(r"\s+", " ", text.strip().translate(_DASHES)).strip(" ,;:_").lower()
    if not t:
        return None
    for rx, build in _PATTERNS:
        if m := rx.fullmatch(t):
            try:
                return build(m)
            except ValueError:  # out-of-range field, e.g. month 13
                return None
    return _free_form(t)


def unwrap(times: list[ParsedTime]) -> list[datetime]:
    """Make a left-to-right label sequence increasing where labels lack a field
    (Dec -> Jan without a year, Sun -> Mon, 23:00 -> 00:00)."""
    out: list[datetime] = []
    for p in times:
        dt = p.dt
        while out and dt <= out[-1] and p.wrap:
            if p.wrap == "year":
                try:
                    dt = dt.replace(year=dt.year + 1)
                except ValueError:  # 29 Feb
                    break
            else:
                dt += timedelta(days=7 if p.wrap == "week" else 1)
        out.append(dt)
    return out
