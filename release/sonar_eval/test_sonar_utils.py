"""Unit tests for sonar_utils.py.

Run with:  python -m pytest test_sonar_utils.py  (or plain: python test_sonar_utils.py)
No GPU required.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from sonar_utils import nearest_ping, fit_scale_and_score


# ---------------------------------------------------------------------------
# nearest_ping
# ---------------------------------------------------------------------------

def _make_entries(depths, start_iso="2020-01-01T00:00:00", dt_s=1.0):
    t0 = datetime.fromisoformat(start_iso)
    entries = [(t0 + timedelta(seconds=i * dt_s), d) for i, d in enumerate(depths)]
    timestamps = [e[0] for e in entries]
    return entries, timestamps


def test_nearest_ping_exact_match():
    entries, timestamps = _make_entries([1.0, 2.0, 3.0])
    depth, gap = nearest_ping(entries, timestamps, timestamps[1])
    assert depth == pytest.approx(2.0)
    assert gap == pytest.approx(0.0)


def test_nearest_ping_between_two():
    entries, timestamps = _make_entries([1.0, 3.0], dt_s=1.0)
    # 0.4 s after first ping -> closer to first
    ts = timestamps[0] + timedelta(seconds=0.4)
    depth, gap = nearest_ping(entries, timestamps, ts)
    assert depth == pytest.approx(1.0)
    assert gap == pytest.approx(0.4)


def test_nearest_ping_beyond_max_gap():
    entries, timestamps = _make_entries([1.0, 2.0], dt_s=1.0)
    ts = timestamps[0] + timedelta(seconds=5.0)
    depth, gap = nearest_ping(entries, timestamps, ts, max_gap=1.0)
    assert depth is None
    assert gap is None


def test_nearest_ping_at_max_gap_boundary():
    entries, timestamps = _make_entries([7.0])
    ts = timestamps[0] + timedelta(seconds=1.0)
    depth, gap = nearest_ping(entries, timestamps, ts, max_gap=1.0)
    assert depth == pytest.approx(7.0)
    assert gap == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fit_scale_and_score
# ---------------------------------------------------------------------------

def test_fit_scale_known_scale():
    """Synthetic trace with scale=2: ours predicts half the real range."""
    rng = np.random.default_rng(0)
    true_range = rng.uniform(0.5, 3.0, 40)
    predicted = true_range / 2.0  # scale factor is 2.0
    split = 30
    result = fit_scale_and_score(true_range[:split], predicted[:split],
                                 true_range[split:], predicted[split:])
    assert result is not None
    assert result["scale"] == pytest.approx(2.0, rel=1e-6)
    assert result["test"]["r"] == pytest.approx(1.0, abs=1e-6)
    assert result["test"]["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_fit_scale_sign_inverted():
    """Sign-inverted predictions (r == -1) should give negative r."""
    sonar = np.linspace(0.5, 3.0, 40)
    pred = -sonar + 4.0  # monotonically decreasing while sonar increases
    result = fit_scale_and_score(sonar[:30], pred[:30], sonar[30:], pred[30:])
    assert result is not None
    assert result["test"]["r"] < 0


def test_fit_scale_degenerate_constant_pred():
    """Constant predicted depth -> return None (no std)."""
    sonar = np.ones(10) * 1.5
    pred = np.ones(10) * 0.5
    result = fit_scale_and_score(sonar[:8], pred[:8], sonar[8:], pred[8:])
    assert result is None


def test_fit_scale_rmse_unscaled_present():
    """rmse_unscaled should be non-negative and present in the result."""
    rng = np.random.default_rng(1)
    sonar = rng.uniform(0.5, 3.0, 40)
    pred = sonar * 1.5 + rng.normal(0, 0.05, 40)
    result = fit_scale_and_score(sonar[:30], pred[:30], sonar[30:], pred[30:])
    assert result is not None
    assert "rmse_unscaled" in result["test"]
    assert result["test"]["rmse_unscaled"] >= 0


if __name__ == "__main__":
    import sys
    # simple runner without pytest
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
