import json
from pathlib import Path

import pytest

from chartcast.synth import KINDS, make_chart


@pytest.fixture(scope="session")
def charts(tmp_path_factory) -> list[tuple[Path, dict]]:
    """Ten synthetic charts (two per kind) with their ground truth."""
    out = tmp_path_factory.mktemp("synth")
    kinds = list(KINDS)
    paths = [make_chart(kinds[i % len(kinds)], 200 + i, out) for i in range(10)]
    return [(p, json.loads(p.with_suffix(".json").read_text())) for p in paths]
