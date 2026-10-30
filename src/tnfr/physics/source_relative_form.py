"""Derived form coordinates relative to an independently supplied held source.

The full cross matrix ``W=z*c.conjugate().T`` retains source-relative
orientation, including unforced regions when another region has contrast
source. This read-out adds no state, controller or constitutive evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._cycle_algebra import ordered_vector
from .form_geometry import RegionalFormObservation, observe_regional_form

__all__ = ["SourceRelativeFormObservation", "observe_source_relative_form"]


@dataclass(frozen=True)
class SourceRelativeFormObservation:
    """Exact source-relative observation and its held-source nodal pushforward.

    ``relative_real + i*relative_imag_numerator/sqrt(12)`` represents W.
    Rates have the same scaling and hold the supplied source fixed; they
    exclude ``z*c_dot.conjugate().T``. ``form`` retains scalar admission,
    node order, stored-pressure provenance and nodal-rate rounding defects.
    Source means need not vanish. Only source contrasts define c.
    """

    form: RegionalFormObservation
    held_source_rate: tuple[Fraction, ...]
    source_mean: tuple[Fraction, ...]
    source_contrast_a: tuple[Fraction, ...]
    source_contrast_b: tuple[Fraction, ...]
    source_contrast_norm_squared: Fraction
    relative_real: tuple[tuple[Fraction, ...], ...]
    relative_imag_numerator: tuple[tuple[Fraction, ...], ...]
    relative_rate_real: tuple[tuple[Fraction, ...], ...]
    relative_rate_imag_numerator: tuple[tuple[Fraction, ...], ...]
    scope: tuple[str, ...] = (
        "source_is_independently_supplied_form_rate_in_report_node_order",
        "exact_rational_source_or_materialized_binary64_coefficients",
        "full_cross_matrix_z_c_adjoint_not_only_local_source_products",
        "held_source_pushforward_excludes_source_motion_and_runtime_events",
        "nonzero_source_contrast_recovers_form_contrast_not_complete_nodal_state",
        "no_source_authentication_pressure_reconstruction_or_closed_evolution_claim",
    )

    @property
    def contrast_reconstructible(self) -> bool:
        """Whether known c and full W recover z via ``z=W*c/(c†c)``."""
        return bool(self.source_contrast_norm_squared)


def observe_source_relative_form(
    graph, regions, *, held_source_rate
) -> SourceRelativeFormObservation:
    """Observe W and its derivative for an explicitly declared held rate source.

    Supply b in graph node order with form/time units (``b=nu*F`` for a
    pressure source F). It is neither the full instantaneous nodal rate nor
    a pressure reconstructed from a measured response. This function cannot
    authenticate the declaration. Exact rational source coefficients stay
    exact; other reals share the affine-law coefficient admission policy.

    Form and stored unforced rates use ``observe_regional_form`` unchanged.
    No source angle, division or nonzero amplitude is required. A constant
    source within every region has c=0 and therefore W=0 for every form.
    For varying sources the reported rate is only the held-source part.
    """
    source = ordered_vector(held_source_rate, "held form rate source")
    form = observe_regional_form(graph, regions)
    if len(source) != len(form.nodes):
        raise ValueError("held source rate must match the graph node order")
    positions = {node: index for index, node in enumerate(form.nodes)}
    triples = tuple(
        tuple(source[positions[node]] for node in region.nodes)
        for region in form.regions
    )
    ca = tuple(x - y for x, y, _ in triples)
    cb = tuple(x + y - 2 * z for x, y, z in triples)

    def cross(*, rate=False, imaginary=False):
        result = []
        for region in form.regions:
            a = region.contrast_a_rate if rate else region.contrast_a
            b = region.contrast_b_rate if rate else region.contrast_b
            result.append(
                tuple(
                    b * u - a * v if imaginary else a * u / 2 + b * v / 6
                    for u, v in zip(ca, cb, strict=True)
                )
            )
        return tuple(result)

    return SourceRelativeFormObservation(
        form=form,
        held_source_rate=source,
        source_mean=tuple(sum(triple, Fraction(0)) / 3 for triple in triples),
        source_contrast_a=ca,
        source_contrast_b=cb,
        source_contrast_norm_squared=sum(
            (a * a / 2 + b * b / 6 for a, b in zip(ca, cb, strict=True)),
            Fraction(0),
        ),
        relative_real=cross(),
        relative_imag_numerator=cross(imaginary=True),
        relative_rate_real=cross(rate=True),
        relative_rate_imag_numerator=cross(rate=True, imaginary=True),
    )
