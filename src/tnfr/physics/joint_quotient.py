"""Conditional nodal reduction with inherited unique-neighbor multiplicities.

The fine model holds phase, capacity, support and coefficients fixed while
EPI follows the declared multichannel pressure. Counts reconstruct the
non-EPI channels; the existing exact affine owner tests EPI projectability.
This observation neither chooses those held laws nor creates a macro graph,
operator, trajectory or complete tetrad-closure certificate.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .._exact_time import finite_represented_real
from ..dynamics import fused_dnfr
from ._cycle_algebra import Vector, dot
from .epi_memory import ForcedSupportClosure, observe_forced_support_closure
from .forced_support import derive_forced_support_balance
from .forcing_realization import (
    NonEpiForcingObservation,
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from .quotient_structure import QuotientStructure, observe_quotient_structure

__all__ = ["JointNodalQuotient", "observe_joint_nodal_quotient"]


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
