"""Closed domains apply to supplied scalars, before their rounding to a boundary."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr._exact_time import bounded_represented_real
from tnfr.constants import inject_defaults
from tnfr.dynamics import runtime

_BEYOND_ONE = Fraction(1) + Fraction(1, 2**100)


@pytest.mark.parametrize(
    "value", [False, np.bool_(True), "1", None, complex(1), np.inf]
)
def test_closed_bounds_retain_shared_scalar_admission(value):
    with pytest.raises((TypeError, ValueError)):
        bounded_represented_real(value, "value", minimum=0, maximum=1)


@pytest.mark.parametrize(
    "value,minimum,maximum",
    [
        (_BEYOND_ONE, 0, 1),
        (Fraction(1) - Fraction(1, 2**100), 1, 2),
        (Fraction(-1) - Fraction(1, 2**100), -1, 0),
        (Fraction(1, 2**2000), 0, 1),
        (0, 1, -1),
        (0, False, 1),
        (0, 0, np.inf),
    ],
)
def test_invalid_original_domain_cannot_be_repaired_by_materialization(
    value, minimum, maximum
):
    with pytest.raises((TypeError, ValueError)):
        bounded_represented_real(value, "value", minimum=minimum, maximum=maximum)


@pytest.mark.parametrize("value", [0, 1, Fraction(1, 3), np.float64(0.5)])
def test_valid_closed_bound_values_keep_ordinary_represented_arithmetic(value):
    actual, exact = bounded_represented_real(value, "value", minimum=0, maximum=1)
    assert type(actual) is float
    assert actual == float(value)
    assert exact == Fraction(actual)


def test_fixed_width_fraction_components_do_not_overflow_original_domain_checks():
    value = Fraction(np.int64(2**62), np.int64(2**62 - 1))
    bound = Fraction(np.int64(2**62 - 1), np.int64(2**62 - 2))
    assert float(value) == float(bound) == 1.0
    # These cross products overflow int64 if the source ratios are not normalized.
    with np.errstate(over="raise"):
        actual, exact = bounded_represented_real(
            value, "ratio", minimum=0, maximum=bound
        )
        assert (actual, exact) == (1.0, Fraction(1))
        with pytest.raises(ValueError):
            bounded_represented_real(bound, "ratio", minimum=0, maximum=value)
        with pytest.raises(ValueError, match="lower bound"):
            bounded_represented_real(value, "ratio", minimum=bound, maximum=value)


@pytest.mark.parametrize(
    "value,minimum,maximum",
    [
        (np.longdouble(0), Fraction(0), Fraction(1)),
        (np.longdouble(1), Fraction(0), Fraction(1)),
        (np.longdouble(0.5), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), np.longdouble(0.5), np.longdouble(0.5)),
        (np.longdouble(0.5), np.longdouble(0), Fraction(1)),
        (np.longdouble(0.5), Fraction(0), np.longdouble(1)),
        (0.5, np.longdouble(0.5), Fraction(1, 2)),
    ],
)
def test_numpy_real_and_rational_bounds_admit_exact_endpoints(value, minimum, maximum):
    actual, exact = bounded_represented_real(
        value, "ratio", minimum=minimum, maximum=maximum
    )
    assert actual == float(value)
    assert exact == Fraction.from_float(actual)


@pytest.mark.parametrize(
    "value,minimum,maximum",
    [
        (np.longdouble(1), 0, Fraction(1) - Fraction(1, 2**100)),
        (np.longdouble(1), _BEYOND_ONE, 2),
        (_BEYOND_ONE, 0, np.longdouble(1)),
        (Fraction(1) - Fraction(1, 2**100), np.longdouble(1), 2),
        (np.longdouble(-1), Fraction(-1) + Fraction(1, 2**100), 0),
    ],
)
def test_numpy_real_and_rational_bounds_reject_exact_outside_values(
    value, minimum, maximum
):
    # The source or bound differs from its adjacent endpoint below binary64
    # resolution, including on platforms where longdouble itself is binary64.
    with pytest.raises(ValueError, match="ratio must be"):
        bounded_represented_real(value, "ratio", minimum=minimum, maximum=maximum)


@pytest.mark.parametrize(
    "minimum,maximum",
    [
        (np.longdouble(1), Fraction(1) - Fraction(1, 2**100)),
        (_BEYOND_ONE, np.longdouble(1)),
    ],
)
def test_numpy_real_and_rational_bounds_reject_exact_reversed_interval(
    minimum, maximum
):
    with pytest.raises(ValueError, match="lower bound"):
        bounded_represented_real(1, "ratio", minimum=minimum, maximum=maximum)


def test_higher_precision_real_is_not_rounded_into_the_unit_interval():
    source = np.nextafter(np.longdouble(1), np.longdouble(2))
    if source == np.longdouble(float(source)):
        pytest.skip("platform longdouble has no precision beyond binary64")
    assert source > 1 and float(source) == 1.0
    with pytest.raises(ValueError):
        bounded_represented_real(source, "ratio", minimum=0, maximum=1)


@pytest.mark.parametrize(
    "override",
    [
        {"VF_ADAPT_MU": _BEYOND_ONE},
        {"SELECTOR_THRESHOLDS": {"si_hi": _BEYOND_ONE}},
        {"PHASE_ADAPT": {"enabled": True, "up": _BEYOND_ONE}},
        {"PHASE_ADAPT": {"enabled": True, "R_lo": _BEYOND_ONE, "R_hi": 1}},
        {"PHASE_ADAPT": {"enabled": True, "kG_min": _BEYOND_ONE, "kG_max": 1}},
        {"VF_MIN": _BEYOND_ONE, "VF_MAX": 1},
    ],
)
def test_runtime_policy_preflight_checks_supplied_bounds_before_any_progress(override):
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.nodes[0].update(EPI=0.0, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    graph.graph.update(override)
    initial_graph = deepcopy(graph.graph)
    initial_nodes = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((TypeError, ValueError)):
        runtime.step(graph, dt=0.125, use_Si=False, apply_glyphs=False)
    assert graph.graph == initial_graph
    assert dict(graph.nodes(data=True)) == initial_nodes


def test_invalid_original_stability_fraction_never_satisfies_early_stop():
    assert float(_BEYOND_ONE) == 1.0
    assert not runtime._stable_tail([_BEYOND_ONE], 1, 0.9)
    assert runtime._stable_tail([Fraction(1)], 1, 0.9)
