"""Finite prescribed-input evidence for the carried C6 nodal encoding."""

import hashlib
import json
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_nodal_remainder as campaign

INPUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/research/c6_winding_phase_kernel.json"
)


@pytest.fixture(scope="module")
def parent():
    if not INPUT.is_file():
        pytest.skip("the retained local B20 phase-kernel comparison is unavailable")
    return json.loads(INPUT.read_bytes())


@pytest.fixture(scope="module")
def report(parent):
    return campaign.analyze_c6_nodal_remainder(parent)


def _all_sequences(report):
    return (
        sequence for case in report["cases"] for sequence in case["sequences"].values()
    )


def test_validation_runs_once_without_graph_execution_or_parent_mutation(
    parent, monkeypatch
):
    from benchmarks import c6_winding_joint_domain as graph_campaign
    from benchmarks import c6_winding_phase_kernel as comparison
    from tnfr.dynamics.integrators import DefaultIntegrator

    def forbidden(*args, **kwargs):
        raise AssertionError("B22 must not execute a new graph trajectory")

    monkeypatch.setattr(graph_campaign, "run_c6_joint_case", forbidden)
    monkeypatch.setattr(comparison, "compare_c6_phase_kernel", forbidden)
    monkeypatch.setattr(DefaultIntegrator, "integrate", forbidden)
    original, calls = deepcopy(parent), []
    validate = campaign.analyze_c6_pressure_cells

    def tracked(value):
        calls.append(value)
        return validate(value)

    monkeypatch.setattr(campaign, "analyze_c6_pressure_cells", tracked)
    campaign.analyze_c6_nodal_remainder(parent)
    assert len(calls) == 1 and calls[0] is parent
    assert parent == original


def test_declared_matrix_contains_only_nine_eight_step_arithmetic_sequences(report):
    assert [(case["mode"], case["epsilon"]) for case in report["cases"]] == [
        ("null", 0),
        ("k1", F(1, 4096)),
        ("k3", F(1, 4096)),
    ]
    for case in report["cases"]:
        assert tuple(case["sequences"]) == campaign.SEQUENCES
    for sequence in _all_sequences(report):
        assert sequence["status"] == "observed"
        observed = sequence["observation"]
        assert len(observed["steps"]) == len(observed["prefixes"]) == 8
        assert len(sequence["pressure_schedule"]) == len(sequence["cycles"]) == 2
        assert all(
            step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
            for step in observed["steps"]
        )


def test_every_endpoint_equals_the_closed_form_exact_nodal_area(report):
    for sequence in _all_sequences(report):
        p, q = sequence["pressure_schedule"]
        expected = tuple(
            F(1, 2) + (F(first) + F(second)) / 4
            for first, second in zip(p, q, strict=True)
        )
        endpoint = sequence["observation"]["endpoint"]
        reconstructed = tuple(
            F(x) + r
            for x, r in zip(endpoint["epi"], endpoint["remainder"], strict=True)
        )
        assert reconstructed == expected
        assert endpoint["epi"] == tuple(float(value) for value in expected)
        assert sequence["summary"]["accumulated_nodal_mean_area"] == sum(
            expected
        ) / 6 - F(1, 2)
        assert sequence["summary"]["reconstructed_mean_change"] == sum(
            expected
        ) / 6 - F(1, 2)


def test_every_prefix_has_exact_vector_balance_and_independent_cell_membership(report):
    for sequence in _all_sequences(report):
        observation = sequence["observation"]
        p, q = sequence["pressure_schedule"]
        for ordinal, (step, prefix) in enumerate(
            zip(observation["steps"], observation["prefixes"], strict=True), 1
        ):
            expected_area = tuple(
                (min(ordinal, 4) * F(first) + max(ordinal - 4, 0) * F(second)) / 16
                for first, second in zip(p, q, strict=True)
            )
            assert prefix["ordinal"] == ordinal
            assert prefix["cumulative_nodal_area"] == expected_area
            assert prefix["reconstructed_change"] == expected_area
            assert prefix["identity_residual"] == (0,) * 6
            assert step["nodal_balance_residual"] == (0,) * 6
            assert prefix["mean_identity_residual"] == 0
            endpoint = step["after"]
            for index, cell in enumerate(prefix["output_cells"]):
                exact = F(endpoint["epi"][index]) + endpoint["remainder"][index]
                assert cell["exact_input"] == exact
                assert cell["lower"] <= exact <= cell["upper"]
                if exact in (cell["lower"], cell["upper"]):
                    assert cell["even_significand"]
                assert cell["contains_exact_input"]
                assert (
                    F(endpoint["epi"][index]) - F(1, 2)
                    == expected_area[index] - endpoint["remainder"][index]
                )
            defect = prefix["mean_visible_change"] - sum(expected_area) / 6
            assert (
                prefix["mean_rounding_lower_bound"]
                <= defect
                <= prefix["mean_rounding_upper_bound"]
            )
            assert (
                abs(defect) <= observation["uniform_mean_rounding_bound"] == F(1, 2**53)
            )


@pytest.mark.parametrize(
    "index,ordinary,visible,shadow,gap",
    (
        (0, F(0), -F(1, 6 * 2**54), F(19, 2**115), F(1, 2**54)),
        (1, F(1, 3 * 2**52), -F(1, 6 * 2**54), F(1, 2**72), F(1, 2**52)),
        (2, F(0), F(0), F(1, 2**73), F(1, 2**51)),
    ),
)
def test_recorded_pressure_results_keep_source_and_rounding_drift_distinct(
    report, index, ordinary, visible, shadow, gap
):
    summary = report["cases"][index]["sequences"]["original"]["summary"]
    assert summary["ordinary_mean_change"] == ordinary
    assert summary["visible_mean_change"] == visible
    assert (
        summary["reconstructed_mean_change"]
        == summary["accumulated_nodal_mean_area"]
        == shadow
    )
    assert summary["mean_endpoint_remainder"] == shadow - visible
    assert summary["max_endpoint_gap"] == gap
    assert summary["exact_nodal_mean_source_is_zero"] is False


def test_zero_sum_inputs_preserve_reconstructed_mean_without_forcing_visible_mean(
    report,
):
    single_visible = (-F(1, 6 * 2**54), -F(1, 6 * 2**54), 0)
    opposite_visible = (0, -F(1, 6 * 2**54), -F(1, 3 * 2**54))
    for index, case in enumerate(report["cases"]):
        for name, expected in (
            ("zero_sum_witness", single_visible[index]),
            ("opposite_pair_witness", opposite_visible[index]),
        ):
            sequence = case["sequences"][name]
            assert all(
                sum(map(F, pressure)) == 0 for pressure in sequence["pressure_schedule"]
            )
            assert all(
                prefix["mean_nodal_area"] == prefix["mean_reconstructed_change"] == 0
                for prefix in sequence["observation"]["prefixes"]
            )
            summary = sequence["summary"]
            assert summary["exact_nodal_mean_source_is_zero"]
            assert summary["visible_mean_change"] == expected
            assert summary["mean_endpoint_remainder"] == -expected


def test_carry_is_not_reset_at_the_two_pressure_tuple_boundary(report):
    for sequence in _all_sequences(report):
        observed = sequence["observation"]
        assert observed["initial"]["remainder"] == (0,) * 6
        for before, after in zip(observed["steps"], observed["steps"][1:]):
            assert before["after"] == after["before"]
        boundary = observed["steps"][3]["after"]
        assert any(boundary["remainder"])
        assert sequence["cycles"][1]["carried_source"] == boundary
        assert sequence["carry_reset_between_cycles"] is False
    # A still-frozen displayed null tuple already carries a nonzero exact state.
    null = report["cases"][0]["sequences"]["original"]["cycles"][1]
    assert null["carried_source_matches_recorded_visible_epi"] is True
    assert null["carried_source_matches_recorded_exact_epi"] is False


def test_null_ordinary_stasis_is_not_a_fixed_point_of_the_carried_reference(report):
    sequence = report["cases"][0]["sequences"]["original"]
    assert all(row == (0.5,) * 6 for row in sequence["ordinary_trace"])
    endpoint = sequence["observation"]["endpoint"]["epi"]
    assert endpoint[:5] == (0.5,) * 5
    assert endpoint[5] == float.fromhex("0x1.fffffffffffffp-2")


def test_ordinary_traces_remain_bound_to_retained_data_but_carry_pressures_are_prescribed(
    parent, report
):
    for source, case in zip(parent["current_cases"], report["cases"], strict=True):
        for sequence in case["sequences"].values():
            for index in range(2):
                endpoint = tuple(
                    map(
                        float,
                        map(
                            F,
                            source["cycles"][index]["flow"]["raw_after_integrator"][
                                "epi"
                            ],
                        ),
                    )
                )
                assert sequence["ordinary_trace"][4 * index + 3] == endpoint
                assert sequence["cycles"][index]["pressure_refresh_executed"] is False
            assert sequence["summary"]["ordinary_matches_retained_trace"]
            assert sequence["pressure_recomputed_from_carried_state"] is False
            assert sequence["canonical_pressure_generated_for_carried_state"] is False


def test_summary_flags_keep_band_evidence_distinct_from_runtime_or_future_claims(
    report,
):
    for sequence in _all_sequences(report):
        assert sequence["summary"]["all_states_in_band"]
        assert sequence["summary"]["prefix_identities_hold"]
        for step in sequence["observation"]["steps"]:
            state = step["after"]
            for x, r in zip(state["epi"], state["remainder"], strict=True):
                assert F(state["epi_lower"]) <= F(x) + r <= F(state["epi_upper"])
                assert state["epi_lower"] <= x <= state["epi_upper"]
    for flag in (
        "runtime_executed",
        "graph_events_executed",
        "live_provenance_certified",
        "future_bounds_verified",
        "production_integrator_modified",
        "empirical_correspondence_tested",
    ):
        assert report[flag] is False
    assert report["detached_record_validation"] is True
    json.dumps(campaign._payload(report), allow_nan=False)


@pytest.mark.parametrize(
    "change",
    ("claim", "seed", "horizon", "order", "controls", "pressure", "state", "history"),
)
def test_corrupt_parent_is_rejected_by_shared_validation(parent, change):
    changed = deepcopy(parent)
    if change == "claim":
        changed["manifest"]["claim_id"] = "different-experiment"
    elif change == "seed":
        changed["manifest"]["seed"] = 18
    elif change == "horizon":
        changed["cycle_count_per_case"] = 3
    elif change == "order":
        changed["current_cases"].reverse()
    elif change == "controls":
        changed["current_cases"][1]["initial"]["configured_controls"]["DT_MIN"] = 0.125
    elif change == "pressure":
        changed["current_cases"][1]["cycles"][0]["flow"]["before"]["pressure"][
            0
        ] = "1/3"
    elif change == "state":
        changed["current_cases"][1]["cycles"][0]["after_capture"]["snapshot"]["epi"][
            0
        ] = "3/4"
    else:
        changed["current_cases"][1]["cycles"][0]["il"]["before"]["state"][
            "glyph_history"
        ]["0"] = []
    with pytest.raises(ValueError):
        campaign.analyze_c6_nodal_remainder(changed)


def test_manifest_binds_historical_input_and_distinct_analysis_source(
    parent, report, tmp_path, monkeypatch
):
    target = tmp_path / "remainder.json"
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        ["remainder", "--input", str(INPUT), "--output", str(target)],
    )
    monkeypatch.setattr(
        campaign, "analyze_c6_nodal_remainder", lambda value: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: ("a" * 40, False, None)
    )
    campaign.main()
    output = json.loads(target.read_bytes())
    assert output["manifest"]["claim_id"] == "O3.a-C6-prescribed-nodal-remainder"
    assert output["manifest"]["git_sha"] == "a" * 40
    evidence = output["input_evidence"]
    assert evidence["sha256"] == hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert evidence["producer_manifest"] == parent["manifest"]
    assert evidence["producer_input_evidence"] == parent["input_evidence"]
    assert evidence["producer_manifest"]["claim_id"] != output["manifest"]["claim_id"]
    assert evidence["pressure_witness_source"].endswith("analyze_c6_pressure_cells")


@pytest.mark.parametrize("change", ("source", "input", "same_output"))
def test_cli_source_and_input_guards_prevent_invalid_publication(
    parent, report, tmp_path, monkeypatch, change
):
    source, target = tmp_path / "parent.json", tmp_path / "result.json"
    source.write_text(json.dumps(parent), encoding="utf-8")
    original = source.read_bytes()
    if change == "same_output":
        target = source
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        ["remainder", "--input", str(source), "--output", str(target)],
    )
    calls = []

    def provenance(*args):
        calls.append(None)
        return (
            ("b" if change == "source" and len(calls) > 1 else "a") * 40,
            False,
            None,
        )

    def analyze(value):
        if change == "input":
            source.write_bytes(original + b"\n")
        return deepcopy(report)

    monkeypatch.setattr(campaign, "current_git_source_provenance", provenance)
    monkeypatch.setattr(campaign, "analyze_c6_nodal_remainder", analyze)
    with pytest.raises(ValueError if change == "same_output" else RuntimeError):
        campaign.main()
    if change == "same_output":
        assert source.read_bytes() == original
    else:
        assert not target.exists()
