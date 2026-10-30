"""Configured feedback controller using structural read-outs and public operators.

Targets, thresholds, selected node and invocation schedule are supplied policy
inputs. Measured coherence selects an action and adjusts a threshold; this does
not derive the policy from nodal dynamics or prove restoration of a form.
Public operator admission still applies to each requested action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph, NodeId

from .._coherence_validation import validate_structural_coherence
from .._exact_time import finite_represented_real
from ..alias import get_attr
from ..config.operator_names import (
    COHERENCE,
    DISSONANCE,
    EMISSION,
    SELF_ORGANIZATION,
    SILENCE,
)
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..constants.canonical import (
    FEEDBACK_COHERENCE_TOL_HIGH,
    FEEDBACK_COHERENCE_TOL_LOW,
    FEEDBACK_DNFR_THRESHOLD,
    FEEDBACK_EPI_THRESHOLD,
    FEEDBACK_LEARNING_RATE,
    FEEDBACK_TARGET_COHERENCE,
    FEEDBACK_TAU_ADAPTIVE,
)
from ..metrics.local_coherence import compute_radius_structural_coherence
from ..operators.registry import get_operator_class
from ..types import require_finite_real_scalar_epi
from ..validation.window import validate_window

__all__ = ["StructuralFeedbackLoop"]


def _nonnegative_policy_value(value: Any, name: str) -> float:
    """Admit a represented controller setting without lossy coercion."""
    normalized, _ = finite_represented_real(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


class StructuralFeedbackLoop:
    """Feedback loop that adapts nodal dynamics based on structural state.

    This class implements closed-loop regulation where the system measures its
    current coherence state and selects appropriate operators to maintain
    target coherence levels. The feedback loop adjusts thresholds adaptively
    based on performance.

    **Feedback Cycle:**

    1. **Measure**: Compute current coherence from ΔNFR and local state
    2. **Decide**: Select operator based on deviation from target
    3. **Act**: Apply selected operator
    4. **Learn**: Adjust thresholds based on achieved coherence

    Parameters
    ----------
    graph : TNFRGraph
        Graph containing the regulated node
    node : NodeId
        Identifier of the node to regulate
    target_coherence : float
        Supplied target in [0, 1]; defaults to FEEDBACK_TARGET_COHERENCE.
    tau_adaptive : float
        Nonnegative initial bifurcation threshold; defaults to FEEDBACK_TAU_ADAPTIVE.
    learning_rate : float
        Nonnegative adaptation gain; defaults to FEEDBACK_LEARNING_RATE.
    coherence_tolerance_low : float
        Nonnegative deviation below target; defaults to FEEDBACK_COHERENCE_TOL_LOW.
    coherence_tolerance_high : float
        Nonnegative deviation above target; defaults to FEEDBACK_COHERENCE_TOL_HIGH.
    dnfr_threshold : float
        Nonnegative pressure-magnitude threshold; defaults to FEEDBACK_DNFR_THRESHOLD.
    epi_threshold : float
        Signed scalar form threshold; defaults to FEEDBACK_EPI_THRESHOLD.

    Attributes
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    target_coherence : float
        Target C(t) for homeostasis
    tau_adaptive : float
        Adaptive bifurcation threshold
    learning_rate : float
        Threshold adjustment rate
    COHERENCE_TOL_LOW : float
        Lower tolerance for coherence regulation
    COHERENCE_TOL_HIGH : float
        Upper tolerance for coherence regulation
    DNFR_THRESHOLD : float
        Threshold for self-organization activation
    EPI_THRESHOLD : float
        Threshold for emission activation

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.dynamics.feedback import StructuralFeedbackLoop
    >>> G, node = create_nfr("test_node")
    >>> loop = StructuralFeedbackLoop(G, node, target_coherence=0.7)
    >>> operator_name = loop.regulate()
    >>> loop.homeostatic_cycle(num_steps=5)
    """

    # Regulation thresholds (operational constants)
    COHERENCE_TOL_LOW = FEEDBACK_COHERENCE_TOL_LOW  # ≈ 0.139
    COHERENCE_TOL_HIGH = FEEDBACK_COHERENCE_TOL_HIGH  # ≈ 0.099
    DNFR_THRESHOLD = FEEDBACK_DNFR_THRESHOLD  # √(tol_low × tol_high) ≈ 0.117
    EPI_THRESHOLD = FEEDBACK_EPI_THRESHOLD  # Configured policy ≈ 0.330

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        target_coherence: float = FEEDBACK_TARGET_COHERENCE,  # ≈ 0.737
        tau_adaptive: float = FEEDBACK_TAU_ADAPTIVE,  # ≈ 0.155
        learning_rate: float = FEEDBACK_LEARNING_RATE,  # ≈ 0.043 (operational)
        coherence_tolerance_low: float = COHERENCE_TOL_LOW,
        coherence_tolerance_high: float = COHERENCE_TOL_HIGH,
        dnfr_threshold: float = DNFR_THRESHOLD,
        epi_threshold: float = EPI_THRESHOLD,
    ) -> None:
        """Initialize a validated scalar feedback policy without selecting a backend."""
        self.G = graph
        self.node = node
        self.target_coherence = validate_structural_coherence(
            target_coherence, name="target_coherence"
        )
        self.tau_adaptive = _nonnegative_policy_value(tau_adaptive, "tau_adaptive")
        self.learning_rate = _nonnegative_policy_value(learning_rate, "learning_rate")
        self.COHERENCE_TOL_LOW = _nonnegative_policy_value(
            coherence_tolerance_low, "coherence_tolerance_low"
        )
        self.COHERENCE_TOL_HIGH = _nonnegative_policy_value(
            coherence_tolerance_high, "coherence_tolerance_high"
        )
        self.DNFR_THRESHOLD = _nonnegative_policy_value(
            dnfr_threshold, "dnfr_threshold"
        )
        self.EPI_THRESHOLD = finite_represented_real(epi_threshold, "epi_threshold")[0]

        # Compatibility attributes only: this controller has no backend operations.
        self.backend = None
        self._use_optimizations = False

    def regulate(self) -> str:
        """Select appropriate operator based on current structural state.

        The configured policy maps structural readings to operator choices:

        - **Low coherence**: Stabilize with IL (Coherence)
        - **High coherence**: Explore with OZ (Dissonance)
        - **High ΔNFR**: Self-organize with THOL
        - **Low EPI**: Activate with AL (Emission)
        - **Stable**: Consolidate with SHA (Silence)

        Returns
        -------
        str
            Operator name to apply

        Notes
        -----
        The target, tolerances and thresholds are supplied control-policy
        parameters. Reading nodal state and choosing registered operators
        does not derive those parameters or their occurrence law from the
        nodal equation. This is an engineering controller, not evidence of
        autonomous phase/form maintenance.
        """
        target = validate_structural_coherence(
            self.target_coherence, name="target_coherence"
        )
        tolerance_low = _nonnegative_policy_value(
            self.COHERENCE_TOL_LOW, "coherence_tolerance_low"
        )
        tolerance_high = _nonnegative_policy_value(
            self.COHERENCE_TOL_HIGH, "coherence_tolerance_high"
        )
        dnfr_threshold = _nonnegative_policy_value(
            self.DNFR_THRESHOLD, "dnfr_threshold"
        )
        epi_threshold = finite_represented_real(self.EPI_THRESHOLD, "epi_threshold")[0]
        dnfr = get_attr(
            self.G.nodes[self.node],
            ALIAS_DNFR,
            0.0,
            strict=True,
            conv=lambda value: finite_represented_real(value, "stored pressure")[0],
        )
        epi = get_attr(
            self.G.nodes[self.node],
            ALIAS_EPI,
            0.0,
            strict=True,
            conv=require_finite_real_scalar_epi,
        )

        # Compute local coherence estimate
        coherence = self._compute_local_coherence()

        # Structural decision tree
        if coherence < target - tolerance_low:
            # Very low coherence → stabilize
            return COHERENCE
        elif coherence > target + tolerance_high:
            # High coherence → explore
            return DISSONANCE
        elif abs(dnfr) > dnfr_threshold:
            # High reorganization pressure → self-organize
            return SELF_ORGANIZATION
        elif epi < epi_threshold:
            # Low activation → emit
            return EMISSION
        else:
            # Stable state → consolidate
            return SILENCE

    def _compute_local_coherence(self) -> float:
        """Return the node's canonical radius-zero structural coherence.

        The constitutive kernel includes both ``|DeltaNFR|`` and the recorded
        ``|dEPI/dt|`` channel. Missing rate telemetry has the explicit static
        interpretation ``dEPI/dt = 0`` supplied by the shared local reader.
        """

        return compute_radius_structural_coherence(self.G, self.node, radius=0)

    def adapt_thresholds(self, performance_metric: float) -> None:
        """Adapt thresholds based on achieved performance.

        Uses proportional feedback control to adjust tau_adaptive toward
        target coherence. This implements learning in the feedback loop.

        Parameters
        ----------
        performance_metric : float
            Finite signed coherence or other performance measure in the
            supplied target scale. This generic input is not restricted to [0, 1].

        Notes
        -----
        Threshold adaptation follows:

        .. math::

            \\tau_{t+1} = \\tau_t + \\alpha (C_{target} - C_{achieved})

        where α is the nonnegative learning rate. The existing policy clips
        the next threshold to [0.05, 0.25]. Invalid inputs or a nonrepresentable
        update are rejected before changing the stored threshold.
        """
        target = validate_structural_coherence(
            self.target_coherence, name="target_coherence"
        )
        tau = _nonnegative_policy_value(self.tau_adaptive, "tau_adaptive")
        learning_rate = _nonnegative_policy_value(self.learning_rate, "learning_rate")
        performance = finite_represented_real(performance_metric, "performance_metric")[
            0
        ]
        proposal = finite_represented_real(
            tau + learning_rate * (target - performance), "adapted threshold"
        )[0]
        self.tau_adaptive = max(0.05, min(0.25, proposal))

    def homeostatic_cycle(self, num_steps: int = 10) -> None:
        """Execute homeostatic regulation cycle.

        Request repeated sense-decide-act-learn steps toward a supplied target.

        Parameters
        ----------
        num_steps : int, default=10
            Number of regulation steps

        Notes
        -----
        Each step:

        1. Measures current coherence
        2. Selects operator via regulate()
        3. Applies operator
        4. Measures new coherence
        5. Adapts thresholds

        Automatic controller execution is not a proof of autonomous NFR
        formation, target attainment or stability. The caller supplies the
        node and number of invocations; operator admission may reject a step.
        """
        for _ in range(validate_window(num_steps)):
            # Admit adaptive settings before an operator can write graph state.
            tau = _nonnegative_policy_value(self.tau_adaptive, "tau_adaptive")
            _nonnegative_policy_value(self.learning_rate, "learning_rate")
            # Select and apply operator
            operator_name = self.regulate()
            operator_class = get_operator_class(operator_name)
            operator = operator_class()
            operator(self.G, self.node, tau=tau)

            # Measure state after
            coherence_after = self._compute_local_coherence()

            # Adapt thresholds based on performance
            self.adapt_thresholds(coherence_after)
