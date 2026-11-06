"""Independent arithmetic and domain controls without derivative replay."""

import itertools
from fractions import Fraction as Q

import pytest

from tnfr.mathematics import _validated_taylor as owner
from tnfr.mathematics._rational_interval import I


def _arguments():
    return dict(
        initial_box=(I(-7, 9),),
        tube=(I(-8, 10),),
        series=((I(-7, 9), I(-2, 1), I(3), I(-1, 2)),),
        local_remainder_bounds=(I(Q(-1, 16), Q(1, 8)),),
        duration=Q(1, 2),
        order=3,
    )


def test_polynomial_corner_arithmetic_and_endpoint_only_intersection():
    arguments = _arguments()
    increment, endpoint = owner.reconstruct_box_taylor_arithmetic(**arguments)
    changes = tuple(
        c1 / 2 + Q(3, 4) + c3 / 8 + tail
        for c1, c3, tail in itertools.product(
            (Q(-2), Q(1)), (Q(-1), Q(2)), (Q(-1, 16), Q(1, 8))
        )
    )
    assert increment == (I(min(changes), max(changes)),)
    assert increment == (I(Q(-7, 16), Q(13, 8)),)
    assert endpoint == (I(Q(-119, 16), 10),)
    # Intersection must not silently replace a correlated increment with a
    # difference of widened endpoints or force every polynomial corner to fit.
    assert increment[0] != endpoint[0] - arguments["initial_box"][0]
    assert arguments["initial_box"][0].hi + increment[0].hi > endpoint[0].hi


def test_uncertain_constant_source_cancels_before_polynomial_evaluation():
    increment, endpoint = owner.reconstruct_box_taylor_arithmetic(
        (I(-100, 100),),
        (I(-101, 102),),
        ((I(-100, 100), I(3)),),
        (I(0),),
        Q(1, 4),
        order=1,
    )
    assert increment == (I(Q(3, 4)),)
    assert endpoint == (I(Q(-397, 4), Q(403, 4)),)


@pytest.mark.parametrize(
    "changes",
    (
        {"initial_box": ()},
        {"initial_box": (I(0),) * 65},
        {"initial_box": (True,)},
        {"tube": ()},
        {"tube": (I(-6, 10),)},
        {"tube": (True,)},
        {"series": ()},
        {"series": ((I(-7, 9), I(0)),)},
        {"series": ((I(-7, 8), I(0), I(0), I(0)),)},
        {"series": ((I(-7, 9), I(0), True, I(0)),)},
        {"local_remainder_bounds": ()},
        {"local_remainder_bounds": (False,)},
        {"duration": 0},
        {"duration": -1},
        {"duration": True},
        {"duration": 0.5},
        {"order": False},
        {"order": 0},
        {"order": 17},
    ),
)
def test_inconsistent_or_unadmitted_evidence_rejects(changes):
    arguments = _arguments()
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        owner.reconstruct_box_taylor_arithmetic(**arguments)


def test_disjoint_endpoint_cannot_be_repaired_by_intersection():
    with pytest.raises(ArithmeticError, match="disjoint endpoint"):
        owner.reconstruct_box_taylor_arithmetic(
            (I(0),), (I(-1, 1),), ((I(0), I(5)),), (I(0),), Q(1), order=1
        )


def test_solver_uses_same_arithmetic_without_changing_step_evidence(monkeypatch):
    calls = []
    original = owner.reconstruct_box_taylor_arithmetic

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(owner, "reconstruct_box_taylor_arithmetic", record)
    step, failed, reason = owner.validated_box_taylor_step(
        (I(1, 2), I(-1, 0)),
        Q(1, 8),
        lambda state: (-state[0], state[0] - state[1]),
        lambda _: (Q(1),),
        order=4,
    )
    assert failed is reason is None and len(calls) == 1
    args, kwargs, result = calls[0]
    assert args == (
        step.initial_box,
        step.tube,
        step.series,
        step.local_remainder_bounds,
        step.duration,
    )
    assert kwargs == {"order": 4}
    assert result == (step.increment, step.endpoint)
