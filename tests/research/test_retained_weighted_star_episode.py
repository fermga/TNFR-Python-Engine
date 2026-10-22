"""Read-only audit of the single frozen finite-registration evaluation.

Source preparation, numerical integration and interval producers are forbidden.
The retained solver provenance supplies the Taylor/Picard premise; this audit
independently checks source association, metric arithmetic, observation ranges
and classification. Receipts establish byte association, not authentication.
"""

from datetime import datetime
from fractions import Fraction as Q
from pathlib import Path
from zipfile import ZipFile

import pytest

from tnfr.dynamics import integrators
from tnfr.mathematics import _validated_metric, _validated_taylor
from tnfr.mathematics._rational_interval import I, cos
from tnfr.research import weighted_star_contact
from tnfr.research import weighted_star_finite_episode as owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
    validate_artifact_path,
    verify_archive_members,
)
from tnfr.research.retained_graph_state import restore_recorded_graph_state
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/weighted_star_registration_episode"
MAX_BYTES = 256 * 1024**2
BLOCKS = ((0, 1, 2, 3), (4,), (5,), (6, 7, 8), (9,))
NODE_BLOCKS = tuple(
    next(k for k, block in enumerate(BLOCKS) if i in block) for i in range(10)
)
FROZEN = {"protocol.json", "preparation.json", "source-manifest.json", "source.zip"}


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=MAX_BYTES))
    )


def _receipt(row):
    assert type(row) is dict and set(row) == {"path", "bytes", "sha256"}
    name = validate_artifact_path(row["path"])
    path = (EVIDENCE / name).resolve()
    assert path.is_relative_to(EVIDENCE.resolve())
    assert type(row["bytes"]) is int and row["bytes"] >= 0
    assert file_receipt(path, label=name, max_bytes=MAX_BYTES) == row


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    return I(exact_record(value["lo"]), exact_record(value["hi"]))


def _pair_square_bounds(left, right):
    lo, hi = left.lo - right.hi, left.hi - right.lo
    return (
        Q(0) if lo <= 0 <= hi else min(lo * lo, hi * hi),
        max(lo * lo, hi * hi),
    )


def _fine_registration_range(box):
    """Exact rational endpoint arithmetic on all 45 fine pairs per channel."""
    lower, upper = Q(0), Q(0)
    for channel in range(2):
        fine = tuple(box[5 * channel + NODE_BLOCKS[i]] for i in range(10))
        for i in range(10):
            for j in range(i + 1, 10):
                # The admitted nominal chart makes these the same variable,
                # not independent uncertain coordinates with overlapping bounds.
                if NODE_BLOCKS[i] == NODE_BLOCKS[j]:
                    continue
                lo, hi = _pair_square_bounds(fine[i], fine[j])
                lower += lo / 10
                upper += hi / 10
    return lower, upper


def _independent_metric(source):
    mean_weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    masses = tuple(sum(mean_weights[i] for i in block) for block in BLOCKS)
    matrix = [[Q(0) for _ in range(10)] for _ in range(10)]
    for channel in range(2):
        shift = 5 * channel
        for (left, right), conductance in zip(source.edges, source.conductances):
            i, j = shift + NODE_BLOCKS[left], shift + NODE_BLOCKS[right]
            weight = conductance if channel == 0 else Q(1)
            matrix[i][i] += weight
            matrix[j][j] += weight
            matrix[i][j] -= weight
            matrix[j][i] -= weight
        for i in range(5):
            for j in range(5):
                matrix[shift + i][shift + j] += masses[i] * masses[j] / sum(masses)
    return tuple(map(tuple, matrix))


@pytest.fixture(scope="module", autouse=True)
def no_source_event_solver_or_trajectory():
    def forbidden(*args, **kwargs):
        pytest.fail("retained evidence audit cannot replay a scientific producer")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                owner,
                (
                    "prepare_source",
                    "evaluate_protocol",
                    "prepare_fields",
                    "execute_contact_word",
                    "contact_source",
                    "rk4_coupled_step",
                    "validated_metric_taylor_step",
                    "metric_taylor_observation_boxes",
                ),
            ),
            (weighted_star_contact, ("execute_contact_word", "contact_source")),
            (integrators, ("rk4_coupled_step",)),
            (
                _validated_metric,
                ("validated_metric_taylor_step", "metric_taylor_observation_boxes"),
            ),
            (
                _validated_taylor,
                ("validated_taylor_step", "validated_box_taylor_step", "flow_jets"),
            ),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    paths = tuple(
        sorted(path for path in EVIDENCE.rglob("*") if path.suffix in (".json", ".zip"))
    )
    receipts = tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)
    documents = {
        name: _read(name + ".json")
        for name in (
            "protocol",
            "preparation",
            "source-manifest",
            "freeze",
            "attempt",
            "result",
        )
    }
    documents["admitted"] = owner.admit_preparation(documents["preparation"])
    if (EVIDENCE / "response.json").exists():
        documents["response"] = _read("response.json")
    yield documents
    assert paths == tuple(
        sorted(path for path in EVIDENCE.rglob("*") if path.suffix in (".json", ".zip"))
    )
    assert receipts == tuple(file_receipt(path, max_bytes=MAX_BYTES) for path in paths)


@pytest.fixture(scope="module")
def response(retained):
    if "response" not in retained:
        pytest.skip(
            "failed frozen attempt retained no complete response; provenance audit remains active"
        )
    return retained["response"]


def test_immutable_archive_and_single_attempt_preserve_exact_byte_associations(
    retained,
):
    freeze, attempt, result = (
        retained[name] for name in ("freeze", "attempt", "result")
    )
    assert freeze["schema"] == "tnfr.weighted-star-episode-freeze.v1"
    assert freeze["evaluation_status_at_freeze"] == "not_evaluated"
    assert {row["path"] for row in freeze["artifacts"]} == FROZEN
    assert len(freeze["artifacts"]) == len(FROZEN)
    for row in freeze["artifacts"]:
        _receipt(row)
    manifest = retained["source-manifest"]["files"]
    assert len(manifest) == len({row["path"].casefold() for row in manifest})
    expected = {row["path"]: row["sha256"] for row in manifest}
    assert {
        "src/tnfr/research/weighted_star_contact.py",
        "src/tnfr/research/weighted_star_finite_episode.py",
        "src/tnfr/mathematics/_validated_metric.py",
        "src/tnfr/dynamics/integrators.py",
        "docs/assets/weighted_star_registration_episode/produce.py",
        "tests/research/test_weighted_star_episode_summary.py",
        "theory/nodal/SINE_WEIGHTED_STAR_FINITE_IDENTITY.md",
        "AGENTS.md",
        "pyproject.toml",
    } <= set(expected)
    verify_archive_members(EVIDENCE / "source.zip", expected, max_bytes=MAX_BYTES)
    with ZipFile(EVIDENCE / "source.zip") as archive:
        assert {info.filename: info.file_size for info in archive.infolist()} == {
            row["path"]: row["bytes"] for row in manifest
        }
    # Live files were checked at evaluation. Routine retained-evidence checks
    # preserve the archived producer, not an indefinite lock on current code.
    assert type(attempt["attempt_number"]) is int and attempt["attempt_number"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["runtime"] == freeze["runtime"]
    assert attempt["freeze"]["path"] == "freeze.json"
    assert result["attempt"]["path"] == "attempt.json"
    _receipt(attempt["freeze"])
    _receipt(result["attempt"])
    assert (
        datetime.fromisoformat(freeze["created_utc"])
        <= datetime.fromisoformat(attempt["created_utc"])
        <= datetime.fromisoformat(result["created_utc"])
    )
    assert len(result["artifacts"]) == len({row["path"] for row in result["artifacts"]})
    for row in result["artifacts"]:
        _receipt(row)
    assert result["status"] in {"evaluated", "failed"}
    assert (result["status"] == "evaluated") == (result["failure"] is None)
    if result["status"] == "failed":
        assert result["scientific_summary"] is None


def test_primitive_preparation_and_protocol_retain_the_actual_model(retained):
    protocol = retained["protocol"]
    owner._admit_protocol(protocol)
    assert protocol["horizon"] == 8
    assert protocol["integration_step"] == Q(1, 1024)
    assert protocol["certificate_step"] == protocol["minimum_residence"] == Q(1, 16)
    assert protocol["certificate_order"] == 12 and protocol["certificate_bins"] == 16
    assert protocol["registration_threshold"] == Q(1, 4)
    assert (
        protocol["maximum_metric_radius"]
        == protocol["maximum_rk4_metric_error"]
        == Q(1, 4096)
    )
    assert protocol["branches"] == (
        {"name": "nonlinear", "gamma": Q(1)},
        {"name": "tangent", "gamma": Q(0)},
    )
    source, partition, state = retained["admitted"]
    assert source.nodes == tuple(range(10)) and len(source.edges) == 19
    assert source.epi == (Q(5, 16),) * 4 + (Q(257, 1024),) + (Q(5, 16),) * 4 + (
        Q(1, 4),
    )
    assert source.phase == (Q(0),) * 5 + (Q(201, 512),) * 5
    assert source.capacity[4] == Q(117, 32)
    assert source.storage.hi < Q(1, 2)
    assert partition.blocks == BLOCKS
    assert state.source_membership_certified and state.full_row_equality_certified
    graph = restore_recorded_graph_state(
        retained["preparation"]["static_states"][-1]
    ).graph
    assert graph.graph["_t"] == 7
    assert tuple(graph.nodes[4]["glyph_history"]) == ("AL", "UM", "SHA")


def test_response_source_progress_prefix_and_metric_are_independently_associated(
    retained, response
):
    source, _, state = retained["admitted"]
    assert response["primitive_source"] == retained["preparation"]["primitive"]
    assert (
        response["initial_center"]
        == state.collective_form + state.collective_phase_radians
    )
    assert response["multiplicities"] == (4, 1, 1, 3, 1)
    assert response["metric"] == _independent_metric(source)
    assert response["physical_identification_claimed"] is False
    assert response["new_collective_NFR_claimed"] is False
    names = {row["path"] for row in retained["result"]["artifacts"]}
    expected = {"response.json"}
    for branch, name in zip(
        response["branches"], ("nonlinear", "tangent"), strict=True
    ):
        assert branch["name"] == name
        assert len(branch["records"]) <= 128
        for index, record in enumerate(branch["records"]):
            path = f"records/{name}-{index:04d}.json"
            expected.add(path)
            assert _read(path) == record
    assert names == expected


def test_retained_tubes_and_rk4_endpoints_obey_their_separate_metric_budgets(
    retained, response
):
    source, _, _ = retained["admitted"]
    metric = _independent_metric(source)
    for branch in response["branches"]:
        for record in branch["records"]:
            tube = tuple(_interval(value) for value in record["tube"])
            assert len(tube) == 10
            assert record["picard_margin"] > 0
            assert record["local_metric_error"] >= 0
            propagated = I(record["growth_factor"] * record["initial_radius"]).hi
            assert (
                record["endpoint_radius"]
                == I(propagated + record["local_metric_error"]).hi
            )
            if branch["name"] == "nonlinear":
                energy = I(0)
                for (left, right), weight in zip(source.edges, source.conductances):
                    i, j = NODE_BLOCKS[left], NODE_BLOCKS[right]
                    energy += weight * (tube[i] - tube[j]) ** 2 / 2
                    energy += 1 - cos(tube[5 + i] - tube[5 + j])
                assert energy.hi < 1
                assert record["domain_lower_bounds"] == (1 - energy.hi,)
            else:
                assert record["domain_lower_bounds"] == (Q(1),)
            difference = tuple(
                exact_record(numeric) - exact_record(center)
                for numeric, center in zip(
                    record["rk4_endpoint"], record["endpoint_center"], strict=True
                )
            )
            assert all(Q(float(value)) == value for value in record["rk4_endpoint"])
            square = sum(
                difference[i] * metric[i][j] * difference[j]
                for i in range(10)
                for j in range(10)
            )
            excess = record["rk4_metric_error_upper"] - record["endpoint_radius"]
            assert square >= 0 and excess >= 0 and excess * excess >= square
            # The squared norm is outward materialized before its square
            # root; bound both roundings rather than imposing a false
            # absolute error on sqrt near zero.
            rounding = Q(1, 2**120)
            assert max(Q(0), excess - rounding) ** 2 <= square + rounding
            assert (
                record["rk4_metric_error_upper"]
                <= retained["protocol"]["maximum_rk4_metric_error"]
            )


def test_whole_bin_registration_and_conditional_summary_are_rebuilt(retained, response):
    if retained["result"]["scientific_summary"] is None:
        pytest.skip("failed orchestration retained no interpreted scientific verdict")
    protocol = retained["protocol"]
    fresh = owner.summarize_response(response, protocol)
    assert fresh == retained["result"]["scientific_summary"]
    for branch, summary in zip(response["branches"], fresh["branches"], strict=True):
        end, run, episode = Q(0), None, None
        all_outside = True
        previous, radius = response["initial_center"], Q(0)
        for index, record in enumerate(branch["records"]):
            assert record["index"] == index
            assert record["elapsed_start"] == end
            assert record["structural_start"] == 7 + end
            assert record["initial_center"] == previous
            assert record["initial_radius"] == radius
            assert len(record["bins"]) == 16
            for item in record["bins"]:
                assert item["start"] == end
                assert item["end"] - item["start"] == Q(1, 256)
                box = tuple(_interval(value) for value in item["box"])
                rebuilt = owner.registration_bounds(box)
                assert _interval(item["registration"]) == rebuilt
                lower, upper = _fine_registration_range(box)
                assert rebuilt.lo <= lower <= upper <= rebuilt.hi
                assert lower - rebuilt.lo < Q(1, 2**110)
                assert rebuilt.hi - upper < Q(1, 2**110)
                all_outside &= rebuilt.lo > Q(1, 4)
                if rebuilt.hi <= Q(1, 4):
                    run = item["start"] if run is None else run
                    if episode is None and item["end"] - run >= Q(1, 16):
                        episode = {"start": run, "end": item["end"]}
                else:
                    run = None
                end = item["end"]
            assert record["elapsed_end"] == end
            previous, radius = record["endpoint_center"], record["endpoint_radius"]
            assert 0 <= radius <= Q(1, 4096)
        assert summary["validated_duration"] == end
        assert summary["first_certified_run"] == episode
        expected_status = (
            "finite_episode_certified"
            if episode is not None
            else (
                "finite_horizon_no_entry_certified"
                if end == 8 and all_outside
                else "episode_unavailable_within_frozen_budget"
            )
        )
        assert summary["status"] == expected_status
