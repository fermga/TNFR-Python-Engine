"""Synthetic whole-bin classification; no source or solver response is acquired.

These records exercise the summary's conditional observation contract only.
Their fabricated boxes are not asserted to be Taylor/Picard certificates.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.research import weighted_star_finite_episode as owner

H, WIDTH = Q(1, 16), Q(1, 256)
ZERO = (Q(0),) * 10
INSIDE = tuple({"lo": value, "hi": value} for value in ZERO)
OUTSIDE = tuple({"lo": Q(index == 1), "hi": Q(index == 1)} for index in range(10))
UNKNOWN = tuple({"lo": Q(0), "hi": Q(index == 1)} for index in range(10))


@pytest.fixture(autouse=True)
def no_preparation_or_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic summary controls cannot acquire a source or response")

    for name in (
        "prepare_source",
        "admit_preparation",
        "evaluate_protocol",
        "prepare_fields",
        "validated_metric_taylor_step",
        "rk4_coupled_step",
    ):
        monkeypatch.setattr(owner, name, forbidden)


def _cell(index, boxes=None):
    boxes = (INSIDE,) * 16 if boxes is None else boxes
    return {
        "index": index,
        "elapsed_start": index * H,
        "elapsed_end": (index + 1) * H,
        "structural_start": 7 + index * H,
        "initial_center": ZERO,
        "initial_radius": Q(0),
        "endpoint_center": ZERO,
        "endpoint_radius": Q(0),
        "bins": tuple(
            {
                "start": index * H + offset * WIDTH,
                "end": index * H + (offset + 1) * WIDTH,
                "box": deepcopy(box),
                "registration": {"lo": Q(-100), "hi": Q(-100)},
            }
            for offset, box in enumerate(boxes)
        ),
    }


def _response(cells):
    return {
        "schema": owner.frozen_protocol()["schema"],
        "initial_center": ZERO,
        "branches": (
            {"name": "nonlinear", "records": tuple(cells), "stop_reason": "synthetic"},
            {"name": "tangent", "records": (), "stop_reason": "synthetic"},
        ),
    }


def _summary(response):
    return owner.summarize_response(response, owner.frozen_protocol())["branches"][0]


def test_sixteen_whole_bins_supply_exactly_the_frozen_residence():
    protocol = owner.frozen_protocol()
    assert protocol["certificate_bins"] == 16
    assert protocol["certificate_step"] == protocol["minimum_residence"] == H
    result = _summary(_response((_cell(0),)))
    assert result["status"] == "finite_episode_certified"
    assert result["first_certified_run"] == {"start": Q(0), "end": H}
    assert result["validated_duration"] == H


def test_fifteen_qualifying_bins_do_not_supply_the_minimum_residence():
    result = _summary(_response((_cell(0, (INSIDE,) * 15 + (OUTSIDE,)),)))
    assert result["status"] == "episode_unavailable_within_frozen_budget"
    assert result["first_certified_run"] is None
    assert result["validated_duration"] == H


def test_residence_can_cross_a_cell_boundary_without_bridging_unknown_bins():
    response = _response(
        (
            _cell(0, (UNKNOWN,) * 8 + (INSIDE,) * 8),
            _cell(1, (INSIDE,) * 8 + (OUTSIDE,) * 8),
        )
    )
    result = _summary(response)
    assert result["first_certified_run"] == {"start": Q(1, 32), "end": Q(3, 32)}
    # Splitting the run by one unresolved bin removes the only qualifying
    # continuous interval; favorable bins on either side cannot be summed.
    response["branches"][0]["records"][1]["bins"][0]["box"] = deepcopy(UNKNOWN)
    assert _summary(response)["first_certified_run"] is None


def test_missing_sixteenth_bin_rejects_an_incomplete_cell():
    with pytest.raises(ValueError, match="cell association"):
        _summary(_response((_cell(0, (INSIDE,) * 15),)))


def test_gap_between_whole_bins_is_rejected():
    response = _response((_cell(0),))
    response["branches"][0]["records"][0]["bins"][8]["start"] += WIDTH
    with pytest.raises(ValueError, match="contiguous"):
        _summary(response)


def test_closed_threshold_accepts_an_exact_upper_bound_at_equality(monkeypatch):
    coordinates = (Q(0), Q(1, 4), Q(-1, 4), Q(0), Q(0)) * 2
    box = tuple({"lo": value, "hi": value} for value in coordinates)

    def exact_point_observation(values):
        # An independent rational point calculation isolates the classifier's
        # <= policy from outward rounding in the pairwise interval owner.
        assert all(value.lo == value.hi for value in values)
        cost = Q(0)
        for channel in range(2):
            fine = tuple(
                values[5 * channel + index].lo
                for index, count in enumerate((4, 1, 1, 3, 1))
                for _ in range(count)
            )
            mean = sum(fine) / 10
            cost += sum((value - mean) ** 2 for value in fine)
        assert cost == Q(1, 4)
        return I(cost)

    monkeypatch.setattr(owner, "registration_bounds", exact_point_observation)
    result = _summary(_response((_cell(0, (box,) * 16),)))
    assert result["status"] == "finite_episode_certified"
    assert result["minimum_bin_upper_bound"] == Q(1, 4)


def test_upper_bound_crossing_threshold_is_unavailable_not_inside(monkeypatch):
    monkeypatch.setattr(
        owner, "registration_bounds", lambda values: I(Q(1, 4), Q(1, 4) + Q(1, 2**80))
    )
    result = _summary(_response((_cell(0),)))
    assert result["status"] == "episode_unavailable_within_frozen_budget"
    assert result["first_certified_run"] is None


def test_favorable_cached_registration_is_ignored():
    response = _response((_cell(0, (OUTSIDE,) * 16),))
    response["initial_registration"] = {"lo": Q(0), "hi": Q(0)}
    response["episode_certified"] = True
    result = _summary(response)
    assert result["minimum_registration_lower_bound"] > Q(1, 4)
    assert result["first_certified_run"] is None
    assert result["status"] == "episode_unavailable_within_frozen_budget"


@pytest.mark.parametrize("change", ("order", "name"))
def test_branch_order_and_model_name_cannot_be_crossed(change):
    response = _response((_cell(0),))
    if change == "order":
        response["branches"] = tuple(reversed(response["branches"]))
    else:
        response["branches"][0]["name"] = "tangent"
    with pytest.raises(ValueError, match="law order"):
        _summary(response)


@pytest.mark.parametrize(
    "key,value",
    (
        ("structural_start", Q(0)),
        ("elapsed_start", WIDTH),
        ("initial_center", (Q(1),) + ZERO[1:]),
        ("initial_radius", Q(1, 2**20)),
    ),
)
def test_wrong_clock_or_state_carry_is_rejected(key, value):
    response = _response((_cell(0), _cell(1)))
    response["branches"][0]["records"][1][key] = value
    with pytest.raises(ValueError, match="clock/state carry"):
        _summary(response)


@pytest.mark.parametrize("count", (127, 128))
def test_no_entry_requires_full_outside_horizon(count):
    result = _summary(_response(_cell(i, (OUTSIDE,) * 16) for i in range(count)))
    assert result["validated_duration"] == count * H
    assert result["first_certified_run"] is None
    assert result["status"] == (
        "finite_horizon_no_entry_certified"
        if count == 128
        else "episode_unavailable_within_frozen_budget"
    )


def test_one_unresolved_bin_prevents_full_horizon_no_entry_claim():
    response = _response(_cell(i, (OUTSIDE,) * 16) for i in range(128))
    response["branches"][0]["records"][0]["bins"][0]["box"] = deepcopy(UNKNOWN)
    result = _summary(response)
    assert result["validated_duration"] == 8
    assert result["first_certified_run"] is None
    assert result["status"] == "episode_unavailable_within_frozen_budget"
