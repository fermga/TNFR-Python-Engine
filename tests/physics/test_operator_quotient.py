"""Operator quotient closure and its fixed-dimensional boundary."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import certify_operator_quotient
from tnfr.physics.structural_morphism import certify_epi_coarse_graining


def _pair_quotient():
    projection = np.array(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]], dtype=float
    )
    lift = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float
    )
    return projection, lift


def _epi_graph():
    graph = nx.path_graph(4)
    for node, epi in enumerate([2.0, -1.0, 3.0, 0.5]):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, theta=0.0)
    return graph


def test_uniform_linear_operator_has_zero_global_quotient_residuals():
    projection, lift = _pair_quotient()
    micro = 0.7 * np.eye(4)  # representative uniform pressure contraction
    macro = 0.7 * np.eye(2)

    result = certify_operator_quotient(
        projection,
        lift,
        micro,
        macro,
        operation_kind="state_map",
        operator_name="Coherence",
    )

    assert result.support_status == "global_linear"
    assert result.global_projected_closure_within_tolerance
    assert result.global_lift_invariance_within_tolerance
    assert result.global_strong_closure_within_tolerance
    assert result.global_projection_residual == pytest.approx(0.0)
    assert result.global_lift_residual == pytest.approx(0.0)
    assert result.information_loss_dimension == 2
    assert result.has_information_loss


def test_exact_epi_diffusion_quotient_satisfies_generic_operator_certificate():
    diffusion = certify_epi_coarse_graining(
        _epi_graph(), [(0, 3), (1, 2)]
    )

    result = certify_operator_quotient(
        diffusion.projection,
        diffusion.lift,
        -diffusion.micro_generator,
        -diffusion.macro_generator,
        operation_kind="vector_field",
        operator_name="pure EPI nodal flow",
    )

    assert result.global_strong_closure_within_tolerance
    assert result.information_loss_dimension == 2


def test_projected_closure_does_not_imply_lift_invariance():
    projection = np.array([[0.5, 0.5]])
    lift = np.ones((2, 1))
    micro = np.array([[1.0, 0.0], [-1.0, 0.0]])
    macro = np.zeros((1, 1))

    result = certify_operator_quotient(projection, lift, micro, macro)

    assert result.global_projected_closure_within_tolerance
    assert not result.global_lift_invariance_within_tolerance
    assert not result.global_strong_closure_within_tolerance
    assert result.global_projection_residual == pytest.approx(0.0)
    assert result.global_lift_residual > 1.0
    assert "COUNTEREXAMPLE" in result.claim_status
    assert "VERIFIED" not in result.claim_status


def test_nonzero_global_residual_is_only_within_declared_tolerance():
    projection, lift = _pair_quotient()
    micro = np.eye(4)
    micro[0, 0] += 1e-3
    macro = np.eye(2)

    result = certify_operator_quotient(
        projection, lift, micro, macro, tolerance=1e-2
    )

    assert result.global_projection_residual > 0.0
    assert result.global_projected_closure_within_tolerance
    assert "within the declared tolerance" in result.claim_status
    assert result.claim_status.startswith("PASSED")
    assert not any(name.startswith("exact_global") for name in result.__dataclass_fields__)


def test_nonlinear_lifted_closure_can_fail_on_unresolved_fibers():
    projection, lift = _pair_quotient()

    result = certify_operator_quotient(
        projection,
        lift,
        lambda state: state**3,
        lambda state: state**3,
        macro_probes=[[0.0, 0.0], [1.0, -2.0], [-0.5, 0.25]],
        micro_probes=[[0.0, 2.0, -1.0, 1.0], [3.0, -1.0, 2.0, 0.0]],
        operation_kind="state_map",
    )

    assert result.support_status == "sampled_callable"
    assert result.sampled_lifted_projected_closure
    assert result.sampled_lift_invariance
    assert not result.sampled_arbitrary_projected_closure
    assert not result.sampled_fiber_independence
    assert not result.sampled_strong_closure
    assert result.global_strong_closure_within_tolerance is None
    assert "not a global callable-map" in result.claim_status


def test_lifted_only_nonlinear_probes_do_not_claim_strong_closure():
    projection, lift = _pair_quotient()

    result = certify_operator_quotient(
        projection,
        lift,
        lambda state: state**3,
        lambda state: state**3,
        macro_probes=[[1.0, -2.0]],
        operation_kind="state_map",
    )

    assert result.sampled_lifted_projected_closure
    assert result.sampled_lift_invariance
    assert result.sampled_arbitrary_projected_closure is None
    assert result.sampled_fiber_independence is None
    assert result.sampled_strong_closure is None


def test_sampled_nonlinear_certificate_is_reproducible():
    projection, lift = _pair_quotient()
    kwargs = dict(
        macro_probes=np.array([[0.2, -0.4], [1.5, 0.5]]),
        micro_probes=np.array([[0.0, 0.4, 1.0, -0.5]]),
        operation_kind="vector_field",
    )
    first = certify_operator_quotient(
        projection,
        lift,
        lambda state: -2.0 * state,
        lambda state: -2.0 * state,
        **kwargs,
    )
    second = certify_operator_quotient(
        projection,
        lift,
        lambda state: -2.0 * state,
        lambda state: -2.0 * state,
        **kwargs,
    )

    assert first == second
    assert first.sampled_strong_closure
    assert first.sampled_fiber_independence
    assert first.sampled_callables_repeatable
    assert first.sampled_repeatability_residual == pytest.approx(0.0)
    assert first.macro_probe_count == 2
    assert first.micro_probe_count == 1


def test_stateful_callable_cannot_receive_sampled_strong_closure():
    projection, lift = _pair_quotient()
    micro_calls = {"count": 0}
    macro_calls = {"count": 0}

    def stateful_micro(state):
        micro_calls["count"] += 1
        return state + 1e-3 * micro_calls["count"]

    def stateful_macro(state):
        macro_calls["count"] += 1
        return state + 1e-3 * macro_calls["count"]

    result = certify_operator_quotient(
        projection,
        lift,
        stateful_micro,
        stateful_macro,
        macro_probes=[[0.0, 0.0]],
        micro_probes=[[0.0, 0.0, 0.0, 0.0]],
    )

    assert not result.sampled_callables_repeatable
    assert result.sampled_repeatability_residual > 0.0
    assert not result.sampled_strong_closure
    assert "repeatability" in result.claim_status


def test_repeatability_check_snapshots_reused_callable_output_buffers():
    projection = np.array([[0.5, 0.5]])
    lift = np.array([[1.0], [1.0]])
    micro_buffer = np.zeros(2)
    macro_buffer = np.zeros(1)

    def stateful_micro(_state):
        micro_buffer[:] += 1.0
        return micro_buffer

    def stateful_macro(_state):
        macro_buffer[:] += 1.0
        return macro_buffer

    result = certify_operator_quotient(
        projection,
        lift,
        stateful_micro,
        stateful_macro,
        macro_probes=[[0.0]],
        micro_probes=[[0.0, 0.0]],
    )

    assert result.sampled_repeatability_residual > 0.0
    assert not result.sampled_callables_repeatable
    assert not result.sampled_strong_closure


def test_reciprocal_projection_lift_rescaling_preserves_valid_quotient_rank():
    result = certify_operator_quotient(
        np.array([[1e-12]]),
        np.array([[1e12]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        tolerance=1e-10,
    )

    assert result.projection_rank == 1
    assert result.lift_rank == 1
    assert result.right_inverse_residual == 0.0
    assert result.global_strong_closure_within_tolerance


def test_nonrepresentable_global_residual_cannot_be_promoted_to_closure():
    with pytest.raises(ValueError, match="floating-point range"):
        certify_operator_quotient(
            np.eye(2),
            np.eye(2),
            np.full((2, 2), 1e308),
            np.zeros((2, 2)),
            tolerance=0.1,
        )


def test_nonrepresentable_right_inverse_is_rejected_before_rank_certificate():
    with pytest.raises(ValueError, match="projection @ lift"):
        certify_operator_quotient(
            np.eye(2) * 1e308,
            np.eye(2) * 1e308,
            np.eye(2),
            np.eye(2),
            tolerance=0.1,
        )


def test_graph_mutation_reports_unsupported_without_invoking_callables():
    projection, lift = _pair_quotient()

    def must_not_run(_state):
        raise AssertionError("graph-mutating dynamics must not be evaluated")

    result = certify_operator_quotient(
        projection,
        lift,
        must_not_run,
        must_not_run,
        operation_kind="graph_mutation",
        operator_name="Recursivity",
    )

    assert not result.supported
    assert result.support_status == "unsupported_graph_mutation"
    assert result.global_strong_closure_within_tolerance is None
    assert result.sampled_strong_closure is None
    assert "fixed projection/lift" in result.unsupported_reason


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"tolerance": 0.0}, "finite and positive"),
        ({"tolerance": True}, "finite and positive"),
        ({"operation_kind": "topology"}, "operation_kind"),
        ({"macro_probes": None}, "macro_probes"),
    ],
)
def test_callable_protocol_validation(kwargs, message):
    projection, lift = _pair_quotient()
    base = dict(macro_probes=[[0.0, 0.0]])
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: state,
            lambda state: state,
            **base,
        )


def test_invalid_quotient_and_dimension_changing_map_are_rejected():
    projection, lift = _pair_quotient()
    bad_lift = lift.copy()
    bad_lift[0, 0] = 0.0
    with pytest.raises(ValueError, match="projection @ lift"):
        certify_operator_quotient(
            projection, bad_lift, np.eye(4), np.eye(2)
        )

    with pytest.raises(ValueError, match="dimension-changing"):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: np.append(state, 0.0),
            lambda state: state,
            macro_probes=[[0.0, 0.0]],
        )


def test_boolean_matrix_probe_and_callable_output_are_rejected():
    projection, lift = _pair_quotient()
    with pytest.raises(ValueError, match="not booleans"):
        certify_operator_quotient(
            projection.astype(bool), lift, np.eye(4), np.eye(2)
        )
    with pytest.raises(ValueError, match="not booleans"):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: state,
            lambda state: state,
            macro_probes=[[True, False]],
        )
    with pytest.raises(ValueError, match="returned booleans"):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: np.zeros(4, dtype=bool),
            lambda state: state,
            macro_probes=[[0.0, 0.0]],
        )


def test_textual_numeric_payloads_are_not_silently_coerced():
    projection, lift = _pair_quotient()
    with pytest.raises(ValueError, match="real numeric"):
        certify_operator_quotient(
            projection.astype(str), lift, np.eye(4), np.eye(2)
        )
    with pytest.raises(ValueError, match="real numeric"):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: state,
            lambda state: state,
            macro_probes=[["0.0", "0.0"]],
        )
    with pytest.raises(ValueError, match="non-real"):
        certify_operator_quotient(
            projection,
            lift,
            lambda state: state.astype(str),
            lambda state: state,
            macro_probes=[[0.0, 0.0]],
        )


@pytest.mark.parametrize("tolerance", ["1e-10", 1e-10j, np.bool_(True)])
def test_tolerance_rejects_coercible_nonreal_values(tolerance):
    projection, lift = _pair_quotient()
    with pytest.raises(ValueError, match="finite and positive"):
        certify_operator_quotient(
            projection, lift, np.eye(4), np.eye(2), tolerance=tolerance
        )
