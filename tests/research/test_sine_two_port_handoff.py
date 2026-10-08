"""Source-to-probe admission without rerunning the retained capture producer."""

import copy
import hashlib
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.research import sine_two_port_handoff as owner
from tnfr.utils.io import json_loads

DIRECTORY = Path(__file__).parents[2] / "docs/assets/sine_formed_classes"


@pytest.fixture(scope="module", autouse=True)
def no_scientific_producers():
    from tnfr.physics import (
        relational_sine_two_port_capture,
        relational_sine_two_port_compatibility,
        relational_sine_two_port_transit,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("handoff audit must not execute an earlier scientific producer")

    with pytest.MonkeyPatch.context() as patch:
        for module in (
            relational_sine_two_port_capture,
            relational_sine_two_port_compatibility,
            relational_sine_two_port_transit,
        ):
            for name in tuple(vars(module)):
                if name.startswith("assess_") or name in (
                    "validated_metric_taylor_step",
                    "_root_enclosures",
                ):
                    patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    record = json_loads((DIRECTORY / "two-port-capture-v1.json").read_bytes())
    protocol = json_loads(
        (DIRECTORY / "two-port-capture-v1.protocol.json").read_bytes()
    )
    return record, protocol


def _replace(value, path, replacement):
    """Copy only changed containers, retaining the large immutable test source."""
    result = copy.copy(value)
    key, *rest = path
    result[key] = _replace(value[key], rest, replacement) if rest else replacement
    return result


def test_fixed_artifact_handoff_rebuilds_bounds_without_execution(retained):
    record, _ = retained
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in DIRECTORY.glob("two-port-capture-v1.*")
    }
    result = owner.audit_sine_two_port_capture_handoff(DIRECTORY)
    assert result.endpoint_form_radius < result.admitted_probe_form_radius == Q(1, 8192)
    assert (
        result.endpoint_phase_radius < result.admitted_probe_phase_radius == Q(1, 1024)
    )
    assert result.reference_target_distance_upper_bound < Q(1, 10**8)
    assert result.reference_minimum_acute_margin > Q(1, 25)
    assert result.target_acute_margin_lower_bound > Q(1, 8)
    assert (
        result.endpoint_excess_storage_upper_bound
        == result.endpoint_form_radius**2 + result.endpoint_phase_radius**2
    )
    assert (
        result.capture_storage_margin
        == Q(1, 648000) - result.endpoint_excess_storage_upper_bound
        > 0
    )
    assert result.full_slow_handoff_time == 1025
    assert result.original_handoff_time_pi_squared_coefficient == 1025 * 1023 * 1024
    assert result.reference_steps_checked == 4096
    assert (
        not result.numerical_execution_replayed and not result.provenance_authenticated
    )
    assert result.conditional_premises
    assert result.to_dict()["schema"] == "tnfr.sine-two-port-handoff-audit.v1"
    # Independent retained-field comparison supplements the rebuilt calculation;
    # these fields did not serve as its input premises.
    assert (
        owner._exact(record["report"]["endpoint_relative_form_norm_upper_bound"])
        == result.endpoint_form_radius
    )
    assert (
        owner._exact(record["report"]["endpoint_phase_distance_upper_bound"])
        == result.endpoint_phase_radius
    )
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in DIRECTORY.glob("two-port-capture-v1.*")
    }


def test_cached_verdicts_and_target_boxes_do_not_supply_handoff_bounds(retained):
    record, protocol = retained
    changed = _replace(record, ("report", "capture_certified"), False)
    for path, replacement in (
        (("report", "status"), "unavailable"),
        (
            ("report", "endpoint_relative_form_norm_upper_bound"),
            {"numerator": 10**30, "denominator": 1},
        ),
        (
            ("report", "endpoint_phase_distance_upper_bound"),
            {"numerator": 0, "denominator": 1},
        ),
        (("report", "target", "local_attraction_certified"), False),
        (("report", "target", "target_phase_bounds"), None),
        (("report", "target", "acute_margin_turns_bounds"), None),
    ):
        changed = _replace(changed, path, replacement)
    result = owner.audit_sine_two_port_capture_record(changed, protocol=protocol)
    assert result.endpoint_form_radius < Q(1, 8192)
    assert result.endpoint_phase_radius < Q(1, 1024)
    assert result.capture_storage_margin > 0


@pytest.mark.parametrize(
    "path,replacement,message",
    (
        (
            ("report", "preparation", "form_error_radius"),
            {"numerator": True, "denominator": 65536},
            "fraction fields",
        ),
        (("report", "preparation", "nominal_epi", 0), False, "exact record"),
        (
            ("report", "preparation", "capacity", 0),
            {"numerator": 0, "denominator": 1},
            "capacity",
        ),
        (
            ("report", "preparation", "reference_model", "epi_weight"),
            1.0,
            "model epi_weight",
        ),
        (("report", "preparation", "geometry", "nodes", 1), True, "source nodes"),
        (("report", "preparation", "geometry", "edges", 0), [0, 2], "support edge"),
        (
            ("report", "preparation", "nominal_edge_turns", 0),
            {"numerator": 0, "denominator": 1},
            "edge gaps",
        ),
        (
            ("report", "target", "canonical_root_turn_bracket", "lower"),
            {"numerator": 0, "denominator": 1},
            "outer bracket",
        ),
        (
            ("report", "target", "inner_root_brackets_at_outer_endpoints", 0, "lower"),
            {"numerator": 0, "denominator": 1},
            "inner bracket",
        ),
        (
            ("report", "folded_reference", "metric", 0, 0),
            {"numerator": 4, "denominator": 1},
            "inherited metric",
        ),
        (("report", "reference_steps"), [], "complete reference"),
        (
            ("report", "reference_steps", 0, "time"),
            {"numerator": 1, "denominator": 1},
            "time chain",
        ),
        (
            ("report", "reference_steps", 0, "initial_radius"),
            {"numerator": 1, "denominator": 1},
            "state chain",
        ),
        (
            ("report", "reference_steps", 0, "picard_interior_margin"),
            {"numerator": 0, "denominator": 1},
            "Picard",
        ),
        (
            ("report", "reference_steps", 0, "domain_lower_bounds", 0),
            {"numerator": 1, "denominator": 1},
            "edge margin",
        ),
        (
            ("report", "reference_steps", 0, "local_metric_error_upper_bound"),
            {"numerator": -1, "denominator": 1},
            "negative retained",
        ),
        (
            ("report", "reference_steps", 0, "endpoint_radius"),
            {"numerator": 0, "denominator": 1},
            "radius recurrence",
        ),
    ),
)
def test_changed_primitives_cannot_inherit_cached_success(
    retained, path, replacement, message
):
    record, protocol = retained
    changed = _replace(record, path, replacement)
    assert changed["report"]["capture_certified"] is True
    with pytest.raises(ValueError, match=message):
        owner.audit_sine_two_port_capture_record(changed, protocol=protocol)


def test_protocol_boolean_order_is_not_equal_integer_admission(retained):
    record, protocol = retained
    changed = _replace(protocol, ("inputs", "order"), True)
    with pytest.raises(ValueError, match="order"):
        owner.audit_sine_two_port_capture_record(record, protocol=changed)


@pytest.mark.parametrize(
    "path,replacement,message",
    (
        (("complete_model", "forcing"), "supplied external source", "forcing"),
        (("complete_model", "events"), "reset at endpoint", "events"),
        (("complete_model", "clocks"), "sigma=t", "clocks"),
        (("complete_model", "beta"), True, "beta/mass"),
        (("support", "weights"), "weighted", "support weights"),
        (("handoff", "additional_slow_duration"), True, "handoff clock"),
    ),
)
def test_changed_complete_protocol_cannot_inherit_the_capture_handoff(
    retained, path, replacement, message
):
    record, protocol = retained
    with pytest.raises(ValueError, match=message):
        owner.audit_sine_two_port_capture_record(
            record, protocol=_replace(protocol, path, replacement)
        )


@pytest.mark.parametrize(
    "name,value", (("law", "native_argument_pressure"), ("clock", "t=sigma"))
)
def test_authoritative_source_law_and_clock_are_admitted(retained, name, value):
    record, protocol = retained
    with pytest.raises(ValueError, match=f"source {name}"):
        owner.audit_sine_two_port_capture_record(
            _replace(record, ("report", "preparation", name), value), protocol=protocol
        )


def test_artifact_digest_failure_stops_before_mapping_admission(
    retained, tmp_path, monkeypatch
):
    manifest = json_loads(
        (DIRECTORY / "two-port-capture-v1.manifest.json").read_bytes()
    )
    (tmp_path / "two-port-capture-v1.manifest.json").write_bytes(
        (DIRECTORY / "two-port-capture-v1.manifest.json").read_bytes()
    )
    first = manifest["artifacts"][0]["file"]
    (tmp_path / first).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        owner,
        "audit_sine_two_port_capture_record",
        lambda *a, **k: pytest.fail("bad digest reached record admission"),
    )
    with pytest.raises(ValueError, match="size or digest"):
        owner.audit_sine_two_port_capture_handoff(tmp_path)
