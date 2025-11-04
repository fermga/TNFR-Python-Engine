import pytest

from tnfr.metrics.common import _coerce_jobs
from tnfr.utils import normalize_optional_int

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, 1),
        ("4", 4),
        (5.0, 5),
    ],
)
def test_coerce_jobs_accepts_positive_values(raw, expected):
    assert _coerce_jobs(raw) == expected
    assert _coerce_jobs(raw) == normalize_optional_int(
        raw, allow_non_positive=False, sentinels=None
    )

@pytest.mark.parametrize("raw", [0, -1, "-5"])
def test_coerce_jobs_rejects_non_positive(raw):
    assert _coerce_jobs(raw) is None
    assert _coerce_jobs(raw) == normalize_optional_int(
        raw, allow_non_positive=False, sentinels=None
    )

@pytest.mark.parametrize("raw", ["invalid", object()])
def test_coerce_jobs_rejects_non_numeric(raw):
    assert _coerce_jobs(raw) is None
    assert _coerce_jobs(raw) == normalize_optional_int(
        raw, allow_non_positive=False, sentinels=None
    )
