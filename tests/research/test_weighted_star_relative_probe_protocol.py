"""Prospective analytical margins, without a finite response evaluation."""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics import _validated_taylor
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    read_bytes_bounded,
)
from tnfr.research.weighted_star_relative_probe import (
    prepare_weighted_star_relative_probe,
)
from tnfr.research.weighted_star_relative_probe_protocol import (
    assess_relative_probe_protocol,
    frozen_protocol,
)
from tnfr.utils.io import json_dumps, json_loads


@pytest.fixture(scope="module", autouse=True)
def no_reserved_response():
    def forbidden(*args, **kwargs):
        pytest.fail("protocol admission must not evaluate a finite response")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_validated_taylor, "validated_box_taylor_step", forbidden)
        patch.setattr(_validated_taylor, "validated_taylor_step", forbidden)
        yield


@pytest.fixture(scope="module")
def prepared():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/weighted_star_registration_episode/preparation.json"
    )
    preparation = decode_exact_tree(
        json_loads(read_bytes_bounded(path, max_bytes=256 * 1024))
    )
    report = prepare_weighted_star_relative_probe(preparation)
    return preparation, report.record


def test_full_source_admission_and_independent_rational_error_budget(prepared):
    result = assess_relative_probe_protocol(*prepared, frozen_protocol())
    budget = result["error_budget"]
    assert budget["comparison_denominator"] == Q(1279, 1280)
    assert budget["matched_cubic_remainder"] == Q(1, 209551360)
    assert budget["per_branch_phase_source_error"] == Q(43, 43946105372672)
    assert budget["per_pair_complete_reading_error"] == Q(67283469, 14062753719255040)
    assert budget["reading_denominator_lower_bound"] == Q(
        151813271921, 154690290911805440
    )
    assert budget["ratio_error_upper_bound"] == Q(2220354477, 303626543842)
    assert budget["ratio_error_upper_bound"] < Q(1, 128)
    assert budget["receiver_lift_absolute_upper_bound"] < Q(1, 128)
    assert Q(9, 20) < result["ideal_relative_conductance"] < Q(1, 2)
    for branch in result["branch_bounds"]:
        assert branch["initial_storage_upper_bound"] < Q(29, 64)
        assert branch["identity_storage_margin"] > Q(3, 32)
        assert branch["event_storage_jumps"][0] > 0
        assert branch["event_storage_jumps"][1] == 0
        assert branch["phase_speed_upper_bound"] < Q(3, 4)
        assert (
            branch["source_phase_error_upper_bound"]
            <= budget["per_branch_phase_source_error"]
        )
    assert result["reserved_response_evaluated"] is False
    assert result["protocol"]["comparison_scope"].startswith("point_reference_only")


def test_increment_reading_counts_source_and_four_reading_errors(prepared):
    protocol = frozen_protocol()
    result = assess_relative_probe_protocol(*prepared, protocol)
    budget = result["error_budget"]
    remainder = budget["matched_cubic_remainder"]
    source_error = budget["per_branch_phase_source_error"]
    one_error = Q(1, 2**40)
    # Two independently prepared branch increments, each with an initial and
    # final reading; the clock error affects the final reading of each branch.
    expected_extra = (
        2 * (source_error + one_error) + Q(3, 2) * one_error + 8 * one_error
    )
    assert budget["per_pair_complete_reading_error"] - remainder == expected_extra
    lower, upper = protocol["predicted_ratio_interval"]
    assert lower == Q(9, 20) - Q(1, 128)
    assert upper == Q(1, 2) + Q(1, 128) < protocol["comparison_reference"]


@pytest.mark.parametrize(
    "field,value",
    (
        ("receiver", True),
        ("duration", True),
        ("duration", Q(1, 128)),
        ("duration", 1 / 256),
        ("per_readout_error", 0),
        ("post_word_phase_error", float("nan")),
        ("maximum_ratio_error", Q(1, 32)),
        ("comparison_reference", True),
    ),
)
def test_protocol_cannot_be_retuned_or_admitted_by_python_equality(
    prepared, field, value
):
    protocol = frozen_protocol()
    protocol[field] = value
    with pytest.raises((TypeError, ValueError)):
        assess_relative_probe_protocol(*prepared, protocol)


def test_complete_tagged_roundtrip_and_independent_declarations(prepared):
    protocol = frozen_protocol()
    roundtrip = decode_exact_tree(json_loads(json_dumps(encode_exact_tree(protocol))))
    result = assess_relative_probe_protocol(*prepared, roundtrip)
    assert (
        decode_exact_tree(json_loads(json_dumps(encode_exact_tree(result)))) == result
    )
    protocol["numerical_protocol"]["order"] = 1
    assert frozen_protocol()["numerical_protocol"]["order"] == 12
    assert result["protocol"]["numerical_protocol"]["order"] == 12


def test_declared_source_association_is_readmitted(prepared):
    preparation, record = prepared
    altered = deepcopy(record)
    altered["source_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source association"):
        assess_relative_probe_protocol(preparation, altered, frozen_protocol())
