"""Conditional nodal reduction with inherited unique-neighbor multiplicities.

The instantaneous EPI observer holds phase, capacity, support and coefficients
fixed. Counts reconstruct the non-EPI channels; the existing exact affine
owner tests EPI projectability. The additional joint phase proposal composes
that row with the shared configured U3 sine law, retaining intrinsic capacity
separately from transport mobility. Neither interface selects a partition or
clock, constructs a macro graph, or certifies complete tetrad closure. The
transverse K3 observer and ideal-Euler envelopes retain omitted phase/form
coordinates and bound their effect under the same explicitly supplied laws.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np

from .._exact_time import finite_represented_real
from ..dynamics import fused_dnfr
from ..utils import angle_diff
from ._cycle_algebra import Vector, dot
from .epi_memory import ForcedSupportClosure, observe_forced_support_closure
from .forced_support import derive_forced_support_balance
from .forcing_realization import (
    NonEpiForcingObservation,
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from .quotient_structure import QuotientStructure, observe_quotient_structure

__all__ = [
    "JointNodalQuotient",
    "JointNodalPhaseStep",
    "observe_joint_nodal_quotient",
    "propose_joint_nodal_phase_step",
    "K3TransverseState",
    "K3TransverseSample",
    "K3TransverseEulerEnvelope",
    "observe_k3_transverse_state",
    "bound_k3_transverse_euler",
]


@dataclass(frozen=True)
class JointNodalQuotient:
    """Exact rational accounting of one counted-channel quotient observation.

    ``effective_rate`` uses the phase kernel evaluated on inherited counts.
    ``projected_model_rate`` instead uses the independently captured fine
    phase coefficients. Their difference is the separately retained phase
    materialization defect; pressure assembly and stored-pressure defects
    are not absorbed into either forcing. Exact identities concern these
    represented-coefficient models, not exact transcendental evaluation or
    repeated binary64 execution. Public fields are not execution proof seals.
    """

    fine_capture: NonEpiForcingObservation
    closure: ForcedSupportClosure
    structure: QuotientStructure
    block_phase: Vector
    macro_epi_gradient: Vector
    counted_phase_gradient: Vector
    counted_components: tuple[tuple[str, Vector], ...]
    inherited_components: tuple[tuple[str, Vector], ...]
    effective_forcing: Vector
    effective_pressure: Vector
    effective_rate: Vector
    projected_model_rate: Vector
    counted_phase_materialization_defect: Vector
    phase_materialization_rate_defect: Vector
    projected_kernel_rate_defect: Vector
    projected_stored_rate_residual: Vector
    projected_fresh_rate: Vector
    projected_stored_rate: Vector
    held_parameter_rows: tuple[str, ...]
    phase_geometry_scope: str
    exact_identity_checks: tuple[str, ...]
    scope: str


def _counted_phase_gradient(phase, multiplicity):
    """Evaluate inherited counts with the same non-JIT NumPy phase kernel.

    Repeated target indices represent distinct fine neighbors. Counts are
    derived by the support owner, not taken from a caller-selected phase
    weight. Grouped summation need not round like the original fine ordering.
    """
    source, target = [], []
    for i, row in enumerate(multiplicity):
        for j, count in enumerate(row):
            source.extend([i] * count)
            target.extend([j] * count)
    size = len(phase)
    values = fused_dnfr.compute_fused_gradients_symmetric(
        edge_src=np.asarray(source, dtype=int),
        edge_dst=np.asarray(target, dtype=int),
        phase=np.asarray(tuple(map(float, phase)), dtype=float),
        epi=np.zeros(size, dtype=float),
        vf=np.ones(size, dtype=float),
        weights={"w_phase": 1.0},
        accumulate_both_directions=False,
        use_jit=False,
    )
    return tuple(
        finite_represented_real(value, f"counted phase gradient[{i}]")[1]
        for i, value in enumerate(values)
    )


def observe_joint_nodal_quotient(graph, blocks) -> JointNodalQuotient:
    """Derive the held joint quotient, rejecting unsupported reduction domains.

    Require the existing bounded default NumPy pressure-capture domain,
    connected positive transport, positive block-constant capacities,
    exactly block-constant represented phase, equitable reciprocal unique
    support and exact all-state EPI projection closure. Internal and
    zero-weight support neighbors remain in the counted source channels.
    Arbitrary fine scalar EPI is permitted: it need not be block-constant.

    Primitive phase and fine block capacity remain explicit held coordinates.
    The loop-free transport quotient's effective capacity can differ from
    the latter. Its pressure source is rescaled by their derived ratio;
    effective capacity never replaces fine capacity in the original channel.
    No coefficient is renormalized or inferred from an observed EPI rate.
    """
    capture = capture_non_epi_forcing(graph)
    fine_components = decompose_non_epi_forcing(capture)
    structure = observe_quotient_structure(capture.snapshot, blocks)
    reference = derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing
    )
    closure = observe_forced_support_closure(reference, structure.blocks)
    if not closure.all_state_affine_closed:
        raise ValueError(
            "joint quotient requires exact all-state EPI projection closure"
        )
    if closure.macro_metric_weights != structure.macro_metric_weights:
        raise RuntimeError("joint quotient and support metric disagree")

    indices = {node: i for i, node in enumerate(capture.snapshot.nodes)}
    phase = []
    for block in structure.blocks:
        values = tuple(capture.phase[indices[node]] for node in block)
        if any(value != values[0] for value in values):
            raise ValueError("joint quotient requires exactly block-constant phase")
        phase.append(values[0])
    phase = tuple(phase)
    counted_phase = _counted_phase_gradient(phase, structure.multiplicity)
    weights = dict(capture.normalized_weights)
    counted_components = tuple(
        (name, tuple(weights[name] * value for value in values))
        for name, values in (
            ("phase", counted_phase),
            ("vf", structure.capacity_gradient),
            ("topo", structure.topology_gradient),
        )
    )
    inherited = tuple(
        (
            name,
            tuple(
                scale * value
                for scale, value in zip(structure.source_scale, values, strict=True)
            ),
        )
        for name, values in counted_components
    )
    m = len(structure.blocks)
    forcing = tuple(
        sum((values[i] for _, values in inherited), Fraction(0)) for i in range(m)
    )
    drift = tuple(dot(row, closure.projected_epi) for row in closure.macro_generator)
    macro_epi_gradient = tuple(
        sum(
            (
                weight * (closure.projected_epi[j] - closure.projected_epi[i])
                for j, weight in enumerate(row)
            ),
            Fraction(0),
        )
        / structure.macro_strengths[i]
        for i, row in enumerate(structure.macro_conductance)
    )
    effective_pressure = tuple(
        capture.epi_weight * gradient + force
        for gradient, force in zip(macro_epi_gradient, forcing, strict=True)
    )
    effective_rate = tuple(
        nu * value
        for nu, value in zip(
            structure.effective_capacity, effective_pressure, strict=True
        )
    )

    def projected_pressure_rate(values):
        rates = tuple(
            nu * value
            for nu, value in zip(capture.snapshot.capacity, values, strict=True)
        )
        return tuple(dot(row, rates) for row in closure.projection)

    phase_defect = tuple(
        fine - counted_phase[block]
        for fine, block in zip(
            capture.phase_gradient, structure.node_blocks, strict=True
        )
    )
    phase_rate_defect = projected_pressure_rate(
        tuple(weights["phase"] * value for value in phase_defect)
    )
    kernel_rate_defect = projected_pressure_rate(capture.kernel_pressure_defect)
    stored_rate_residual = projected_pressure_rate(capture.stored_pressure_residual)
    fresh_rate = projected_pressure_rate(capture.full_kernel_pressure)
    stored_rate = projected_pressure_rate(capture.snapshot.stored_pressure)
    counted_fine = dict(counted_components)
    fine = dict(fine_components)
    checks = {
        "macro EPI channel derived from aggregate conductance": tuple(
            -capture.epi_weight * nu * gradient
            for nu, gradient in zip(
                structure.effective_capacity, macro_epi_gradient, strict=True
            )
        )
        == drift,
        "fine and quotient metric weights agree": closure.metric_weights
        == structure.metric_weights,
        "capacity channel reconstructed from inherited counts": fine["vf"]
        == tuple(counted_fine["vf"][block] for block in structure.node_blocks),
        "topology channel reconstructed from inherited counts": fine["topo"]
        == tuple(counted_fine["topo"][block] for block in structure.node_blocks),
        "effective capacity times source scale is fine block capacity": tuple(
            a * b
            for a, b in zip(
                structure.effective_capacity, structure.source_scale, strict=True
            )
        )
        == structure.block_capacity,
        "projected exact model retains counted-phase materialization": closure.projected_nodal_rate
        == tuple(a + b for a, b in zip(effective_rate, phase_rate_defect, strict=True)),
        "projected fresh kernel retains pressure assembly defect": fresh_rate
        == tuple(
            a + b
            for a, b in zip(
                closure.projected_nodal_rate, kernel_rate_defect, strict=True
            )
        ),
        "projected stored rate retains stored pressure residual": stored_rate
        == tuple(a + b for a, b in zip(fresh_rate, stored_rate_residual, strict=True)),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"joint quotient identity failure: {failures}")
    return JointNodalQuotient(
        fine_capture=capture,
        closure=closure,
        structure=structure,
        block_phase=phase,
        macro_epi_gradient=macro_epi_gradient,
        counted_phase_gradient=counted_phase,
        counted_components=counted_components,
        inherited_components=inherited,
        effective_forcing=forcing,
        effective_pressure=effective_pressure,
        effective_rate=effective_rate,
        projected_model_rate=closure.projected_nodal_rate,
        counted_phase_materialization_defect=phase_defect,
        phase_materialization_rate_defect=phase_rate_defect,
        projected_kernel_rate_defect=kernel_rate_defect,
        projected_stored_rate_residual=stored_rate_residual,
        projected_fresh_rate=fresh_rate,
        projected_stored_rate=stored_rate,
        held_parameter_rows=(
            "primitive_phase",
            "fine_capacity",
            "unique_support",
            "conductance",
            "effective_channel_coefficients",
        ),
        phase_geometry_scope=(
            "not certified; represented NumPy phase coefficients may include "
            "atan2(0,0); nonzero geometric resultants are not established"
        ),
        exact_identity_checks=tuple(checks),
        scope=(
            "conditional fixed-support held-phase/capacity nodal model; "
            "represented phase coefficients and exact inherited algebra; "
            "no autonomous region selection, fine-law derivation, full-tetrad "
            "closure, event/history execution or physical identification"
        ),
    )


@dataclass(frozen=True)
class JointNodalPhaseStep:
    """Detached joint inputs and one counted/fine phase proposal.

    ``joint`` supplies the same-snapshot EPI pressure and effective mobility.
    Its held-phase interpretation applies to the instantaneous EPI row; this
    record additionally proposes the specified phase law. No EPI integration
    or graph write occurs. ``phase_step_defect`` is the exact signed difference
    of represented fine and lifted counted endpoints, before any circular
    comparison. Across a wrap it must not be interpreted as an angular norm.
    Public construction does not authenticate execution or future closure.
    """

    joint: JointNodalQuotient
    dt: Fraction
    coupling_strength: Fraction
    counted_neighbors: tuple[tuple[int, ...], ...]
    fine_phase_after: Vector
    block_phase_after: Vector
    phase_step_defect: Vector
    held_parameter_rows: tuple[str, ...]
    scope: str


def _require_joint_phase_chart(phases):
    """Check the represented chart used by this scoped phase proposal."""
    tau = Fraction.from_float(math.tau)
    if any(not 0 <= value < tau for value in phases):
        raise ValueError("joint phase proposal requires canonical [0,2*pi) phases")
    values = tuple(map(float, dict.fromkeys(phases)))
    if any(
        abs(angle_diff(a, b)) >= math.pi / 2
        for i, a in enumerate(values)
        for b in values[i + 1 :]
    ):
        raise ValueError("joint phase proposal requires a strict half-pi chart")


def propose_joint_nodal_phase_step(
    graph, blocks, *, dt, coupling_strength
) -> JointNodalPhaseStep:
    """Propose fine and reduced phase using the instantaneous joint quotient.

    Inherit the pressure observer's exact EPI projection and support/capacity
    gates. Additionally require canonical phases in one represented strict half-pi chart,
    every fine support edge admitted by the configured U3 gate, and both phase
    endpoints in that chart. The configured sine law retains fine block capacity
    as its free angular rate; cross-only mobility is used only for EPI transport.

    Both proposals use the same shared arithmetic, with observed multiplicities
    for the macro neighbor reader. Grouped summation and pressure materialization
    defects remain visible. A caller may pass the returned same-snapshot EPI
    pressure/capacity to the shared integrator, retaining its separate numerical
    defects. These angular checks are not transcendental interval certificates.
    No generic step stability, runtime invocation or new clock is proved.
    """
    from ..dynamics.phase_evolution import (
        _propose_u3_phase_from_neighbors,
        propose_u3_gated_phase_step,
    )
    from ..operators._phase_gate import resolve_u3_phase_limits

    h_float, h = finite_represented_real(dt, "dt")
    k_float, k = finite_represented_real(coupling_strength, "coupling_strength")
    if h <= 0 or k < 0:
        raise ValueError("dt must be positive and coupling_strength nonnegative")
    joint = observe_joint_nodal_quotient(graph, blocks)
    source, structure = joint.fine_capture, joint.structure
    _require_joint_phase_chart(joint.block_phase)
    _, limit = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    if any(
        abs(angle_diff(float(source.phase[i]), float(source.phase[j]))) > limit
        for i, row in enumerate(source.snapshot.support_neighbors)
        for j in row
    ):
        raise ValueError("joint phase proposal requires all fine support U3-admitted")
    neighbors = tuple(
        tuple(b for b, count in enumerate(row) for _ in range(count))
        for row in structure.multiplicity
    )
    fine = propose_u3_gated_phase_step(
        graph,
        source.snapshot.nodes,
        source.phase,
        source.snapshot.capacity,
        dt=h_float,
        coupling_strength=k_float,
    )
    macro = _propose_u3_phase_from_neighbors(
        graph.graph,
        tuple(range(len(neighbors))),
        neighbors.__getitem__,
        joint.block_phase,
        structure.block_capacity,
        dt=h_float,
        coupling_strength=k_float,
    )
    fine = tuple(Fraction.from_float(float(value)) for value in fine)
    macro = tuple(Fraction.from_float(float(value)) for value in macro)
    _require_joint_phase_chart(fine)
    _require_joint_phase_chart(macro)
    defect = tuple(
        value - macro[a] for value, a in zip(fine, structure.node_blocks, strict=True)
    )
    return JointNodalPhaseStep(
        joint=joint,
        dt=h,
        coupling_strength=k,
        counted_neighbors=neighbors,
        fine_phase_after=fine,
        block_phase_after=macro,
        phase_step_defect=defect,
        held_parameter_rows=tuple(
            name for name in joint.held_parameter_rows if name != "primitive_phase"
        ),
        scope=(
            "detached simultaneous joint pressure inputs and configured phase proposal; "
            "observed equitable support with full U3 admission and strict phase chart; "
            "represented endpoints with signed lift defects; no EPI execution, future "
            "binary64 closure, partition selection or microscopic clock derivation"
        ),
    )


@dataclass(frozen=True)
class K3TransverseState:
    """Exact represented coordinates of an ordered K3 state, without projection.

    The last two nodes form the selected pair. Phases use their ordinary raw
    lift, not a circular average across zero. No state or pressure is written.
    """

    capture: NonEpiForcingObservation
    capacity: Fraction
    phase_width: Fraction
    delta: Fraction
    eta: Fraction
    q: Fraction
    u: Fraction
    mean_epi: Fraction
    mean_phase: Fraction


def observe_k3_transverse_state(graph) -> K3TransverseState:
    """Retain hidden phase/form on unit K3 with one common positive capacity.

    Require unit transport and unit fine potential path lengths. Admit canonical
    phases whose raw lift has width at most one radian. This
    bounded proof domain lies strictly inside pi/2 and avoids ambiguous means
    across a wrap. One radian is an evaluator restriction, not a TNFR threshold.
    Positive effective EPI weight is required; other configured channels retain
    their actual normalized coefficients. Capacity/topology gradients vanish.
    """
    from .geometry_realization import _potential_geometry

    capture = capture_non_epi_forcing(graph)
    source = capture.snapshot
    if len(source.nodes) != 3 or dict(
        ((i, j), weight) for i, j, weight in source.conductance
    ) != {(i, j): 1 for i in range(3) for j in range(3) if i != j}:
        raise ValueError("transverse observation requires fixed unit K3 transport")
    if any(
        set(row) != set(range(3)) - {i}
        for i, row in enumerate(source.support_neighbors)
    ):
        raise ValueError("transverse observation requires exactly K3 unique support")
    geometry = _potential_geometry(graph, source.nodes)
    if geometry.kernel != tuple(
        tuple(Fraction(i != j) for j in range(3)) for i in range(3)
    ):
        raise ValueError("transverse potential bounds require unit K3 path lengths")
    capacity = source.capacity[0]
    if capacity <= 0 or any(value != capacity for value in source.capacity):
        raise ValueError("transverse observation requires common positive capacity")
    if capture.epi_weight <= 0:
        raise ValueError("transverse observation requires positive EPI weight")
    theta = capture.phase
    width = max(theta) - min(theta)
    if any(not 0 <= value < Fraction.from_float(math.tau) for value in theta):
        raise ValueError("transverse phases must be canonical [0,2*pi) coordinates")
    if width > 1:
        raise ValueError("transverse raw phase lift must have width <= 1 radian")
    from ..operators._phase_gate import resolve_u3_phase_limits

    _, gate = resolve_u3_phase_limits(graph.graph, operator_code="UM")
    if width > Fraction.from_float(gate):
        raise ValueError("transverse phase lift must be fully U3-admitted")
    x0, x1, x2 = source.epi
    t0, t1, t2 = theta
    return K3TransverseState(
        capture=capture,
        capacity=capacity,
        phase_width=width,
        delta=(t1 + t2) / 2 - t0,
        eta=t1 - t2,
        q=x0 - (x1 + x2) / 2,
        u=x1 - x2,
        mean_epi=(x0 + x1 + x2) / 3,
        mean_phase=(t0 + t1 + t2) / 3,
    )


@dataclass(frozen=True)
class K3TransverseSample:
    """Rational upper bounds for one ideal Euler index.

    Omission errors compare with the eta=u=0 lift at identical initial macro
    means, delta and q. Fine pressure/potential errors include direct hidden
    coordinates; they need not be quadratic like the macro omission error.
    """

    step: int
    delta_upper: Fraction
    eta_upper: Fraction
    u_upper: Fraction
    delta_omission_upper: Fraction
    q_omission_upper: Fraction
    fine_pressure_potential_upper: Vector


@dataclass(frozen=True)
class K3TransverseEulerEnvelope:
    """Conditional ideal-Euler comparison bounds, never runtime certificates."""

    initial: K3TransverseState
    dt: Fraction
    coupling_strength: Fraction
    phase_rate_lower: Fraction
    form_decay_rate: Fraction
    phase_to_form_gain_upper: Fraction
    samples: tuple[K3TransverseSample, ...]
    scope: str


def bound_k3_transverse_euler(
    graph, *, dt, coupling_strength, steps
) -> K3TransverseEulerEnvelope:
    """Bound attraction and hidden-state omission without running a solver.

    Reconstruct all coefficients from the current observed unit K3. Rational
    inequalities cos(rho)>=1-rho^2/2 and pi>3 replace transcendental evaluation
    on the admitted rho<=1 domain. With c=3K/2 and A=3*kappa*e/2 require
    h<=min(1/c,1/A). The recurrences propagate comparison inequalities only;
    they do not advance graph state or define another dynamical law.

    The resource limit is 256 steps; the ideal mathematical theorem has no
    such horizon limit. The model excludes Gamma, events and capacity/controller
    callbacks; their graph configuration is not admitted by this evaluator.
    Clipping, phase wrapping defects, pressure realization and binary64 solver
    errors are outside these ideal bounds and must be retained independently
    when comparing actual execution.
    """
    if (
        isinstance(steps, bool)
        or not isinstance(steps, Integral)
        or not 0 <= steps <= 256
    ):
        raise ValueError("steps must be an integer in [0,256]")
    _, h = finite_represented_real(dt, "dt")
    _, k = finite_represented_real(coupling_strength, "coupling_strength")
    if h <= 0 or k <= 0:
        raise ValueError("dt and coupling_strength must be positive")
    initial = observe_k3_transverse_state(graph)
    e = initial.capture.epi_weight
    w = dict(initial.capture.normalized_weights)["phase"]
    c = 3 * k / 2
    decay = 3 * initial.capacity * e / 2
    if h * c > 1 or h * decay > 1:
        raise ValueError("dt exceeds the ideal monotone Euler ceiling")
    lower = c * (1 - initial.phase_width**2 / 2)
    gain = initial.capacity * w / 2
    a, b = 1 - h * lower, 1 - h * decay
    delta, eta, u = abs(initial.delta), abs(initial.eta), abs(initial.u)
    error_delta = error_q = Fraction(0)
    samples = []
    for n in range(int(steps) + 1):
        macro_pressure = e * error_q + (w / 3) * error_delta
        hidden_pressure = e * u + (w / 3) * eta
        pair_pressure = macro_pressure / 2 + 3 * hidden_pressure / 4
        samples.append(
            K3TransverseSample(
                step=n,
                delta_upper=delta,
                eta_upper=eta,
                u_upper=u,
                delta_omission_upper=error_delta,
                q_omission_upper=error_q,
                fine_pressure_potential_upper=(
                    macro_pressure,
                    pair_pressure,
                    pair_pressure,
                ),
            )
        )
        if n == steps:
            break
        delta, eta, u, error_delta, error_q = (
            a * delta,
            a * eta,
            b * u + h * gain * eta,
            a * error_delta + h * c * delta * eta**2 / 8,
            b * error_q + h * gain * error_delta,
        )
    return K3TransverseEulerEnvelope(
        initial=initial,
        dt=h,
        coupling_strength=k,
        phase_rate_lower=lower,
        form_decay_rate=decay,
        phase_to_form_gain_upper=gain,
        samples=tuple(samples),
        scope=(
            "exact rational upper envelopes for the unclipped ideal simultaneous Euler "
            "model on unit K3, common fixed capacity and supplied sine phase law; "
            "excludes Gamma/events/controllers, whose runtime configuration is not "
            "validated; no binary64 trajectory, physical identification or partition selection"
        ),
    )
