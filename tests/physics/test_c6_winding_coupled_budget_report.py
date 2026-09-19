"""Five coupled certificates bound one retained prefix without extending it."""

import hashlib
import json
import sys
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path

import pytest

from benchmarks import c6_winding_coupled_budget as campaign

DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts/research"
INPUTS = tuple(DIRECTORY / name for name in campaign.INPUT_NAMES)


@pytest.fixture(scope="module")
def parents():
    if any(not path.is_file() for path in INPUTS):
        pytest.skip("the retained B31 through B26 evidence chain is unavailable")
    return tuple(json.loads(path.read_bytes()) for path in INPUTS)


@pytest.fixture(scope="module")
def report(parents):
    return campaign.analyze_c6_winding_coupled_budget(*parents)


def _laplacian(values):
    return tuple(
        x - (values[i - 1] + values[(i + 1) % 6]) / 2 for i, x in enumerate(values)
    )


def _center(values):
    mean = sum(values) / 6
    return tuple(x - mean for x in values)


def _exact(state):
    return tuple(
        F(x) + r for x, r in zip(state["epi"], state["remainder"], strict=True)
    )


def test_five_sections_keep_current_profile_and_retained_prefix_origins_distinct(
    report, parents
):
    assert all(
        name in report
        for name in (
            "B32_profile",
            "B33_retained_steps",
            "B34_contraction",
            "B35_uniform_tube",
            "B36_prefix_and_mean",
        )
    )
    source = report["source"]
    assert (
        source["retained_B31_report_replayed"] is True
        and source["retained_step_count"] == 18
    )
    assert (
        campaign._payload(source["initial_state"])
        == parents[0]["source"]["inherited_state"]
    )
    assert (
        campaign._payload(source["endpoint_state"])
        == parents[0]["continuation"]["endpoint"]
    )
    assert source["initial_state"] != source["endpoint_state"]
    assert report["B35_uniform_tube"]["reference"]["state"] == source["initial_state"]
    assert report["B32_profile"]["current_reconstructed_pattern"]["epi"] == _exact(
        source["endpoint_state"]
    )


def test_B32_reuses_the_exact_Poisson_profile_with_its_nonzero_mean_source(report):
    section = report["B32_profile"]
    balance = section["reference"]["forced_balance"]
    source, profile, weight = (
        balance["forcing"],
        balance["relative_profile"],
        balance["epi_weight"],
    )
    assert weight == F(3299399280161697, 2**54)
    assert tuple(x * 2**112 for x in source) == (
        -50713790850736016,
        18184265779157704,
        139188182064155136,
        -278376364128310272,
        556752728256620544,
        -385035021120887104,
    )
    mean = sum(source) / 6
    assert mean == balance["mean_drift"] == -F(1, 6 * 2**109)
    assert sum(profile) == 0 and balance["metric_weights"] == (2,) * 6
    assert tuple(weight * x for x in _laplacian(profile)) == tuple(
        x - mean for x in source
    )
    assert (
        balance["profile_residual"] == (0,) * 6
        and balance["profile_center_residual"] == 0
    )
    assert balance["has_zero_pressure_equilibrium"] is False
    denominator = 4279441930180617987241864442413056
    assert tuple(x * denominator for x in profile) == (
        -920914671527074529,
        -109391571287326151,
        538473136940002879,
        -66355793410064327,
        1834202553394660903,
        -1276013654110198775,
    )


def test_B32_current_cut_uses_visible_shape_but_reconstructed_error_is_separate(report):
    section = report["B32_profile"]
    z = section["reference"]["forced_balance"]["relative_profile"]
    state = report["source"]["endpoint_state"]
    visible, exact = tuple(map(F, state["epi"])), _exact(state)
    for key, values in (
        ("current_visible_pattern", visible),
        ("current_reconstructed_pattern", exact),
    ):
        pattern = section[key]
        expected = tuple(x - p for x, p in zip(_center(values), z, strict=True))
        assert pattern["relative_error"] == expected
        assert pattern["error_variance"] == sum(x * x for x in expected)
    assert (
        section["current_visible_pattern"]["relative_error"]
        != section["current_reconstructed_pattern"]["relative_error"]
    )
    cut = section["node4_nonpositive_cut"]
    assert (
        cut["current_gradient_index"],
        cut["nonpositive_max_index"],
        cut["necessary_gradient_change_upper_bound"],
    ) == (-14, -22, -8)
    assert cut["required_laplacian_error_increase"] == F(1, 2**52)
    assert (
        cut["current_visible_laplacian_error"]
        == _laplacian(section["current_visible_pattern"]["relative_error"])[4]
    )
    assert cut["required_visible_laplacian_error"] == F(22, 2**55) - _laplacian(z)[4]


def test_B33_each_retained_step_is_a_canonical_pressure_and_exact_carried_decomposition(
    report, parents
):
    section = report["B33_retained_steps"]
    records = section["observations"]
    assert len(records) == len(parents[0]["continuation"]["steps"]) == 18
    balance = report["B32_profile"]["reference"]["forced_balance"]
    source, weight, z = (
        balance["forcing"],
        balance["epi_weight"],
        balance["relative_profile"],
    )
    previous = report["source"]["initial_state"]
    for record, retained in zip(
        records, parents[0]["continuation"]["steps"], strict=True
    ):
        step = record["step"]
        assert campaign._payload(step) == retained and step["before"] == previous
        assert step["timestep"] == 1 / 16 and step["capacity"] == (1.0,) * 6
        before, after = _exact(step["before"]), _exact(step["after"])
        x, remainder, pressure = (
            tuple(map(F, step["before"]["epi"])),
            step["before"]["remainder"],
            tuple(map(F, step["pressure"])),
        )
        assert tuple(b - a for a, b in zip(before, after, strict=True)) == tuple(
            p / 16 for p in pressure
        )
        carry = tuple(weight * value for value in _laplacian(remainder))
        rounding = tuple(
            p - a + weight * gradient
            for p, a, gradient in zip(pressure, source, _laplacian(x), strict=True)
        )
        assert (
            record["carry_feedback"] == carry and record["rounding_defect"] == rounding
        )
        assert record["forcing_defect"] == tuple(
            a + b for a, b in zip(carry, rounding, strict=True)
        )
        assert record["centered_forcing_defect"] == _center(record["forcing_defect"])
        assert record["centered_before"] == _center(before) and record[
            "centered_after"
        ] == _center(after)
        assert record["error_before"] == tuple(
            a - b for a, b in zip(_center(before), z, strict=True)
        )
        expected = tuple(
            e - weight * gradient / 16 + defect / 16
            for e, gradient, defect in zip(
                record["error_before"],
                _laplacian(record["error_before"]),
                record["centered_forcing_defect"],
                strict=True,
            )
        )
        assert record["error_after"] == record["modeled_error_after"] == expected
        assert record["recurrence_residual"] == (0,) * 6
        assert (
            record["mean_carry_contribution"] == record["mean_identity_residual"] == 0
        )
        assert record["mean_rounding_contribution"] == sum(rounding) / 96
        assert record["readout"]["readout_shift"] == carry
        assert record["readout"]["pressure_identity_residual"] == (0,) * 6
        assert record["graph_provenance_certified"] is False
        previous = step["after"]
    assert previous == report["source"]["endpoint_state"]
    assert (
        section["all_recurrence_residuals_zero"]
        is section["all_mean_residuals_zero"]
        is section["all_carry_means_zero"]
        is True
    )


def test_B34_spatial_contraction_keeps_the_uniform_mode_unchanged(report):
    section = report["B34_contraction"]
    s = F(3299399280161697, 2**58)
    assert section["step_factor"] == s
    assert section["nonuniform_eigenvalues"] == (F(1, 2), F(1, 2), F(3, 2), F(3, 2), 2)
    assert section["norm_factor"] == 1 - s / 2 == F(573161353023261791, 2**59)
    assert section["energy_factor"] == section["norm_factor"] ** 2 < 1
    assert all(sum(row) == 1 for row in section["transition"])
    assert all(sum(row) == 0 for row in section["centering"])
    vectors = section["eigenvectors"]
    for i, (value, vector) in enumerate(
        zip(section["nonuniform_eigenvalues"], vectors, strict=True)
    ):
        assert sum(vector) == 0
        assert _laplacian(vector) == tuple(value * x for x in vector)
        assert tuple(
            sum(a * b for a, b in zip(row, vector, strict=True))
            for row in section["transition"]
        ) == tuple((1 - s * value) * x for x in vector)
        assert all(
            sum(a * b for a, b in zip(vector, other, strict=True)) == 0
            for other in vectors[i + 1 :]
        )


def test_B35_uniform_error_tube_comes_from_arithmetic_bounds_and_not_observed_maxima(
    report,
):
    tube = report["B35_uniform_tube"]["reference"]
    source = report["B32_profile"]["reference"]["forced_balance"]
    weight, h, unit, tail = source["epi_weight"], F(1, 16), F(1, 2**53), F(1, 2**1075)
    product_error = unit * weight * F(1, 4) + tail
    assembly_error = (
        unit * (max(map(abs, source["forcing"])) + weight * F(1, 4) + product_error)
        + tail
    )
    assert tube["carry_bound"] == F(1, 2**54)
    assert tube["product_error_bound"] == product_error
    assert tube["assembly_error_bound"] == assembly_error
    assert tube["rounding_bound"] == product_error + assembly_error
    assert (
        tube["forcing_component_bound"]
        == 2 * weight * tube["carry_bound"] + product_error + assembly_error
    )
    assert (
        tube["centered_forcing_norm_squared_bound"]
        == 6 * tube["forcing_component_bound"] ** 2
    )
    assert tube["initial_energy"] == sum(x * x for x in tube["initial_error"])
    assert (
        tube["initial_error"]
        == report["B33_retained_steps"]["observations"][0]["error_before"]
    )
    q = report["B34_contraction"]["norm_factor"]
    assert (
        tube["energy_floor"]
        == h * h * tube["centered_forcing_norm_squared_bound"] / (1 - q) ** 2
    )
    assert (
        tube["energy_bound"]
        == max(tube["initial_energy"], tube["energy_floor"])
        == tube["energy_floor"]
    )
    assert tube["mean_increment_bound"] == h * (
        abs(source["mean_drift"]) + tube["rounding_bound"]
    )
    assert (
        tube["infinite_mean_control_certified"]
        is tube["future_runtime_certified"]
        is False
    )


def test_B35_tube_leaves_node_four_sign_cut_inconclusive(report):
    tube = report["B35_uniform_tube"]["reference"]
    cut = report["B35_uniform_tube"]["node4_cut_assessment"]
    assert (cut["node"], cut["nonpositive_cut"]) == (4, -22)
    assert cut["centered_laplacian_squared_bound"] == F(3, 2) * tube["energy_bound"]
    required = report["B32_profile"]["node4_nonpositive_cut"][
        "required_visible_laplacian_error"
    ]
    assert cut["remaining_laplacian_distance"] == required - 2 * tube["carry_bound"] < 0
    assert cut["cut_excluded"] is cut["cut_reachability_certified"] is False
    assert cut["conclusion"] == "inconclusive"


def test_B36_all_eighteen_prefixes_satisfy_the_uniform_energy_and_separate_signed_mean_budgets(
    report, parents
):
    section = report["B36_prefix_and_mean"]
    tube = report["B35_uniform_tube"]["reference"]
    q = report["B34_contraction"]["norm_factor"]
    for row in section["prefixes"]:
        k = row["ordinal"]
        assert all(row["uniform_checks"].values())
        assert (
            row["finite_energy_envelope"]
            == q**k * tube["initial_energy"] + (1 - q**k) * tube["energy_floor"]
        )
        assert (
            row["energy_after"] <= row["finite_energy_envelope"] <= tube["energy_bound"]
        )
        assert row["homogeneous_energy"] <= row["homogeneous_energy_bound"]
        assert row["mean_carry_prefix"] == row["mean_identity_residual"] == 0
        assert (
            row["mean_change"]
            == row["mean_source_prefix"] + row["mean_rounding_prefix"]
        )
        assert (
            abs(row["mean_change"])
            <= row["absolute_mean_prefix_bound"]
            == k * tube["mean_increment_bound"]
        )
    assert section["all_uniform_checks_pass"] is True
    assert section["mean_source_sum"] == -F(3, 2**113)
    assert section["mean_rounding_sum"] == -F(113, 3 * 2**114)
    assert section["mean_carry_sum"] == section["mean_identity_residual"] == 0
    assert section["mean_change"] == -F(131, 3 * 2**114)
    assert campaign._canonical(section["mean_change"]) == campaign._canonical(
        parents[0]["continuation"]["local_mean_nodal_area"]
    )


def test_B36_exact_finite_band_gate_is_maximal_for_its_sufficient_inequality_not_a_run(
    report,
):
    tube = report["B35_uniform_tube"]["reference"]
    band = report["B36_prefix_and_mean"]["finite_band_horizon"]
    n = band["maximum_steps"]
    assert n == 196713720348826219 > report["source"]["retained_step_count"]
    assert (
        band["tube_initially_admitted"] is True
        and band["unbounded_conditional_prefix"] is False
    )
    assert band["centered_coordinate_squared_bound"] == F(5, 6) * tube["energy_bound"]
    admitted_margin = band["minimum_initial_margin"] - n * tube["mean_increment_bound"]
    assert (
        admitted_margin >= 0
        and admitted_margin**2 >= band["centered_coordinate_squared_bound"]
    )
    assert (
        band["next_step_margin"]
        == band["minimum_initial_margin"] - (n + 1) * tube["mean_increment_bound"]
    )
    assert (
        band["next_step_margin"] < 0
        or band["next_step_margin"] ** 2 < band["centered_coordinate_squared_bound"]
    )
    assert band["next_step_passes"] is False
    assert (
        band["actual_band_exit_certified"] is band["future_runtime_certified"] is False
    )
    assert report["B36_prefix_and_mean"]["retained_prefix_within_band_horizon"] is True
    assert report["new_trajectory_steps"] == 0


@pytest.mark.parametrize(
    "change",
    (
        "claim",
        "scope",
        "carry",
        "area",
        "baseline",
        "step_pressure",
        "step_cache",
        "prefix",
        "cached_bound",
        "flag",
        "lineage",
        "weight",
    ),
)
def test_source_corruption_is_rejected_before_scientific_promotion(parents, change):
    values = list(deepcopy(parents))
    parent, source = values[0], values[-1]
    if change == "claim":
        parent["manifest"]["claim_id"] = "unrelated"
    elif change == "scope":
        parent["source_scope"] = ["benchmarks"]
    elif change == "carry":
        parent["continuation"]["endpoint"]["remainder"] = ["0"] * 6
    elif change == "area":
        parent["continuation"]["B27_nodal_area"][4] = "0"
    elif change == "baseline":
        parent["continuation"]["net_area_reference"] = "B31"
    elif change == "step_pressure":
        parent["continuation"]["steps"][0]["pressure"][4] = 0.0
    elif change == "step_cache":
        parent["continuation"]["steps"][0]["exact_increment"][4] = "0"
    elif change == "prefix":
        parent["continuation"]["prefix_balances"][0]["mean_nodal_area"] = "1"
    elif change == "cached_bound":
        parent["initial_frozen_stencil"]["initial_first_exit_bound"] = 1
    elif change == "flag":
        parent["live_provenance_certified"] = True
    elif change == "lineage":
        parent["earlier_input_evidence"]["producer_manifest"]["claim_id"] = "unrelated"
    else:
        source["lattice_reference"]["source"]["epi_weight"] = "1/2"
    with pytest.raises(ValueError):
        campaign.analyze_c6_winding_coupled_budget(*values)


def test_rebinding_uses_only_the_eighteen_retained_steps_and_no_graph_word(
    parents, monkeypatch
):
    from tnfr.operators import event_runtime, nodal_remainder_runtime
    from tnfr.physics import c6_carried_profile

    def forbidden(*args, **kwargs):
        raise AssertionError("the coupled audit must not execute a graph operator word")

    monkeypatch.setattr(event_runtime, "execute_operator_event_schedule", forbidden)
    monkeypatch.setattr(
        nodal_remainder_runtime, "execute_nodal_remainder_event_schedule", forbidden
    )
    owner, rebound = c6_carried_profile.advance_nodal_remainder, []

    def tracked(*args, **kwargs):
        result = owner(*args, **kwargs)
        rebound.append(result)
        return result

    monkeypatch.setattr(c6_carried_profile, "advance_nodal_remainder", tracked)
    before = deepcopy(parents)
    result = campaign.analyze_c6_winding_coupled_budget(*parents)
    assert parents == before and len(rebound) == 18
    assert tuple(campaign._payload(campaign.asdict(step)) for step in rebound) == tuple(
        parents[0]["continuation"]["steps"]
    )
    assert result["new_graph_trajectories"] == result["new_trajectory_steps"] == 0


def test_report_scope_keeps_finite_numeric_admission_separate_from_future_runtime(
    report,
):
    for flag in (
        "runtime_executed",
        "live_provenance_certified",
        "original_tail_reachability_certified",
        "infinite_band_invariance_certified",
        "future_pressure_sign_exit_certified",
        "full_runtime_stability_certified",
    ):
        assert report[flag] is False
    assert all(
        "profile" not in row for row in report["B33_retained_steps"]["observations"]
    )
    assert "contraction" not in report["B35_uniform_tube"]["reference"]
    assert "tube" not in report["B36_prefix_and_mean"]["finite_band_horizon"]
    json.dumps(campaign._payload(report), allow_nan=False)


def _cli(monkeypatch, paths, output):
    names = (
        "input",
        "parent-input",
        "previous-input",
        "earlier-input",
        "ancestor-input",
        "source-input",
    )
    argv = ["coupled"]
    for name, path in zip(names, paths, strict=True):
        argv.extend(("--" + name, str(path)))
    monkeypatch.setattr(sys, "argv", argv + ["--output", str(output)])


def test_cli_binds_six_input_hashes_and_output_source_identity(
    tmp_path, monkeypatch, report
):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(6))
    for path, source in zip(paths, INPUTS, strict=True):
        path.write_bytes(source.read_bytes())
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_coupled_budget", lambda *args: deepcopy(report)
    )
    monkeypatch.setattr(
        campaign,
        "current_git_source_provenance",
        lambda *args: ("a" * 40, True, "sha256:" + "b" * 64),
    )
    campaign.main()
    result = json.loads(output.read_bytes())
    assert result["manifest"]["claim_id"] == "O3.a-C6-coupled-carried-budget"
    for key, source in zip(campaign.INPUT_KEYS, INPUTS, strict=True):
        assert result[key]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result[key]["producer_manifest"] != result["manifest"]
    assert tuple(path.read_bytes() for path in paths) == tuple(
        source.read_bytes() for source in INPUTS
    )


@pytest.mark.parametrize("changed_index", (1, 2, 3, 4, 5))
def test_cli_validates_all_historical_byte_links_before_analysis(
    tmp_path, monkeypatch, changed_index
):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(6))
    for i, (path, source) in enumerate(zip(paths, INPUTS, strict=True)):
        path.write_bytes(source.read_bytes() + (b"\n" if i == changed_index else b""))
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)

    def forbidden(*args):
        raise AssertionError("invalid historical hashes must fail before analysis")

    monkeypatch.setattr(campaign, "analyze_c6_winding_coupled_budget", forbidden)
    with pytest.raises(ValueError, match="lineage hashes"):
        campaign.main()
    assert not output.exists()


def test_cli_refuses_a_source_change_during_analysis(tmp_path, monkeypatch, report):
    paths = tuple(tmp_path / f"source_{i}.json" for i in range(6))
    for path, source in zip(paths, INPUTS, strict=True):
        path.write_bytes(source.read_bytes())
    output = tmp_path / "report.json"
    _cli(monkeypatch, paths, output)
    monkeypatch.setattr(
        campaign, "analyze_c6_winding_coupled_budget", lambda *args: deepcopy(report)
    )
    snapshots = iter(
        (("a" * 40, True, "sha256:" + "b" * 64), ("a" * 40, True, "sha256:" + "c" * 64))
    )
    monkeypatch.setattr(
        campaign, "current_git_source_provenance", lambda *args: next(snapshots)
    )
    with pytest.raises(RuntimeError, match="changed during"):
        campaign.main()
    assert not output.exists()
