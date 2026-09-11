"""Finite-probe temporal identifiability for TNFR operators.

The categorical operator contracts do not identify every canonical operator:
Silence and Contraction share the same declared contract signature.  This
module studies the next inverse-problem layer without adding identifiers to the
observations.  It exposes:

* an algebraic certificate for any supplied quantitative signature matrix; and
* a finite-hypothesis noise margin for nearest-signature identification; and
* a deterministic experiment that applies every canonical operator to fresh
  graph probes and observes only changes in EPI, structural frequency, pressure,
  phase, nodal velocity, and graph size.

The experiment is deliberately finite.  Distinct rows establish separation on
the declared probes only; they do not prove global operator or word
identifiability.  Caller-owned graphs are never accepted or mutated.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Sequence

from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..mathematics.unified_numerical import np
from ..rng import validate_seed
from ._helpers import wrap_angle

__all__ = [
    "SignatureMatrixCertificate",
    "SignatureNoiseMarginCertificate",
    "NearestSignatureIdentification",
    "TemporalOperatorIdentifiabilityCertificate",
    "certify_temporal_signature_matrix",
    "certify_signature_noise_margin",
    "identify_nearest_signature",
    "probe_canonical_operator_identifiability",
]


_CHANNELS = ("epi", "nu_f", "delta_nfr", "phase")


@dataclass(frozen=True, slots=True)
class SignatureMatrixCertificate:
    """Algebraic partition of labelled rows in a signature matrix.

    Labels identify the hypotheses being compared; they are not matrix
    features.  When ``quantization_decimals`` is not ``None``, equivalence is
    exact equality after decimal rounding.  Matrix ranks are numerical
    diagnostics of the supplied floating-point matrix and are not themselves
    identifiability criteria.
    """

    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    signatures: tuple[tuple[float, ...], ...]
    matrix_rank: int
    affine_rank: int
    equivalence_classes: tuple[tuple[str, ...], ...]
    uniquely_identifiable: tuple[str, ...]
    ambiguous_groups: tuple[tuple[str, ...], ...]
    all_rows_identifiable: bool
    quantization_decimals: int | None
    rank_tolerance: float | None
    claim_status: str


@dataclass(frozen=True, slots=True)
class SignatureNoiseMarginCertificate:
    """Finite-prototype robustness certificate in a declared scaled norm.

    For prototype rows ``s_i`` with minimum pairwise distance ``delta``, every
    observation ``y=s_i+e`` satisfying ``||e|| < delta/2`` has ``s_i`` as its
    unique nearest prototype.  The statement is exact for the supplied finite
    rows, feature scales and norm.  It does not model how likely the error is or
    extend the prototypes to unseen states. The minimum distance is rounded
    downward from exact arithmetic on the supplied binary64 inputs, so the
    reported half-distance remains conservative. Ill-conditioned scales that
    erase a nonzero separation are rejected.
    """

    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    signatures: tuple[tuple[float, ...], ...]
    feature_scales: tuple[float, ...]
    norm: str
    minimum_pairwise_distance: float
    certified_noise_radius: float
    closest_pairs: tuple[tuple[str, str], ...]
    all_prototypes_distinct: bool
    strict_bound_required: bool
    claim_status: str

    @property
    def minimum_distance_is_conservative_lower_bound(self) -> bool:
        """Whether the reported binary64 separation is rounded downward."""

        return True


@dataclass(frozen=True, slots=True)
class NearestSignatureIdentification:
    """Nearest-prototype result evaluated against a noise-margin certificate."""

    observation: tuple[float, ...]
    nearest_labels: tuple[str, ...]
    nearest_distance: float
    second_nearest_distance: float
    within_certified_radius: bool
    robustly_identified: bool
    identified_label: str | None
    claim_status: str


@dataclass(frozen=True, slots=True)
class TemporalOperatorIdentifiabilityCertificate:
    """Measured finite-probe signature certificate for canonical operators."""

    probe_seeds: tuple[int, ...]
    n_nodes: int
    target_node: int
    frames_per_probe: int
    algebra: SignatureMatrixCertificate
    requested_operators: tuple[str, ...]
    executed_operators: tuple[str, ...]
    silence_contraction_separated: bool
    unresolved_groups: tuple[tuple[str, ...], ...]
    claim_status: str
    noise_margin: SignatureNoiseMarginCertificate | None = None


def _validate_decimals(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("quantization_decimals must be an integer or None")
    value = int(value)
    if value < 0 or value > 15:
        raise ValueError("quantization_decimals must be between 0 and 15")
    return value


def _validate_rank_tolerance(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("rank_tolerance must be a finite non-negative number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("rank_tolerance must be a finite non-negative number")
    return result


def _validated_names(
    values: Sequence[str], expected: int, *, parameter: str
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{parameter} must be a sequence of strings")
    names = tuple(values)
    if len(names) != expected:
        raise ValueError(f"{parameter} length must match the matrix dimension")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{parameter} entries must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{parameter} entries must be unique")
    return names


def _validated_real_matrix(signatures: Sequence[Sequence[float]]) -> Any:
    """Materialize a rectangular real matrix without string/bool coercion."""

    if isinstance(signatures, (str, bytes)):
        raise TypeError("signatures must be a sequence of real rows")
    try:
        raw_rows = tuple(signatures)
    except TypeError as exc:
        raise TypeError("signatures must be a sequence of real rows") from exc
    rows: list[tuple[float, ...]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, (str, bytes)):
            raise TypeError("signature rows must contain real numbers")
        try:
            row = tuple(raw_row)
        except TypeError as exc:
            raise TypeError("each signature must be an iterable row") from exc
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in row):
            raise TypeError("signature rows must contain real numbers")
        rows.append(tuple(float(value) for value in row))
    try:
        return np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("signatures must be a rectangular real matrix") from exc


def certify_temporal_signature_matrix(
    signatures: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    feature_names: Sequence[str] | None = None,
    quantization_decimals: int | None = None,
    rank_tolerance: float | None = None,
) -> SignatureMatrixCertificate:
    """Partition hypotheses by a supplied quantitative temporal signature.

    Parameters
    ----------
    signatures:
        Two-dimensional matrix with one hypothesis per row.  Entries must be
        finite real numbers.
    labels:
        Unique row labels.  Labels are used only to report the partition and
        are never included as observed features.
    feature_names:
        Optional unique names for the matrix columns.
    quantization_decimals:
        If supplied, rows are compared after rounding each entry to this many
        decimal places.  ``None`` uses exact floating-point row equality.
    rank_tolerance:
        Optional tolerance passed to the numerical matrix-rank calculation.

    Returns
    -------
    SignatureMatrixCertificate
        Exact row-equivalence classes for the supplied matrix (after any
        declared decimal quantization), plus numerical matrix and affine ranks.

    Notes
    -----
    Distinct rows are necessary and sufficient to identify a label *within the
    supplied finite hypothesis set*.  They make no statement about unobserved
    states, noise models, or arbitrary operator words.
    """

    decimals = _validate_decimals(quantization_decimals)
    rank_tol = _validate_rank_tolerance(rank_tolerance)
    matrix = _validated_real_matrix(signatures)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("signatures must be a nonempty two-dimensional matrix")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("signatures must contain only finite values")

    row_labels = _validated_names(labels, int(matrix.shape[0]), parameter="labels")
    if feature_names is None:
        columns = tuple(f"feature_{index}" for index in range(int(matrix.shape[1])))
    else:
        columns = _validated_names(
            feature_names,
            int(matrix.shape[1]),
            parameter="feature_names",
        )

    grouped: dict[tuple[float, ...], list[str]] = {}
    rows: list[tuple[float, ...]] = []
    for label, raw_row in zip(row_labels, matrix, strict=True):
        if decimals is None:
            key = tuple(float(value) for value in raw_row)
        else:
            key = tuple(round(float(value), decimals) for value in raw_row)
        rows.append(tuple(float(value) for value in raw_row))
        grouped.setdefault(key, []).append(label)

    classes = tuple(tuple(group) for group in grouped.values())
    unique = tuple(group[0] for group in classes if len(group) == 1)
    ambiguous = tuple(group for group in classes if len(group) > 1)
    matrix_rank = _stable_numerical_rank(matrix, rank_tol, centred=False)
    affine_rank = _stable_numerical_rank(matrix, rank_tol, centred=True)

    return SignatureMatrixCertificate(
        labels=row_labels,
        feature_names=columns,
        signatures=tuple(rows),
        matrix_rank=matrix_rank,
        affine_rank=affine_rank,
        equivalence_classes=classes,
        uniquely_identifiable=unique,
        ambiguous_groups=ambiguous,
        all_rows_identifiable=not ambiguous,
        quantization_decimals=decimals,
        rank_tolerance=rank_tol,
        claim_status=(
            "EXACT row partition for the supplied matrix after the declared "
            "decimal quantization; matrix ranks are numerical diagnostics"
        ),
    )


def _stable_numerical_rank(
    matrix: Any,
    tolerance: float | None,
    *,
    centred: bool,
) -> int:
    """Compute a scale-equivalent rank without overflowing centring.

    Scaling the whole matrix by a positive scalar leaves rank unchanged.  It
    also lets the affine calculation subtract rows in ``[-1, 1]`` rather than
    forming an overflowing difference such as ``1e308 - (-1e308)``.  An
    explicitly supplied absolute tolerance is scaled by the same factor.
    """

    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return 0
    scaled = matrix / scale
    candidate = scaled - scaled[0] if centred else scaled
    rank_kwargs: dict[str, float] = {}
    if tolerance is not None:
        rank_kwargs["tol"] = float(tolerance / scale)
    return int(np.linalg.matrix_rank(candidate, **rank_kwargs))


def _validated_feature_scales(
    feature_scales: Sequence[float] | None,
    n_features: int,
) -> Any:
    if feature_scales is None:
        return np.ones(n_features, dtype=float)
    if isinstance(feature_scales, (str, bytes)):
        raise TypeError("feature_scales must be a sequence of positive numbers")
    try:
        raw_scales = tuple(feature_scales)
    except TypeError as exc:
        raise TypeError(
            "feature_scales must be a sequence of positive numbers"
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in raw_scales
    ):
        raise TypeError("feature_scales must contain real numbers")
    scales = np.asarray(raw_scales, dtype=float)
    if scales.ndim != 1 or int(scales.size) != n_features:
        raise ValueError("feature_scales length must match the signature width")
    if not bool(np.all(np.isfinite(scales))) or bool(np.any(scales <= 0.0)):
        raise ValueError("feature_scales must contain only finite positive values")
    return scales


def _scaled_distance(left: Any, right: Any, scales: Any, norm: str) -> float:
    """Return a stable scaled distance or reject an unrepresentable one."""

    components: list[float] = []
    for left_value, right_value, scale in zip(left, right, scales, strict=True):
        left_float = float(left_value)
        right_float = float(right_value)
        scale_float = float(scale)
        raw_difference = left_float - right_float
        if math.isfinite(raw_difference):
            component = raw_difference / scale_float
        else:
            # Dividing first avoids a false overflow when the declared scale
            # is large enough to make the mathematical quotient finite.
            component = left_float / scale_float - right_float / scale_float
        if not math.isfinite(component):
            raise ValueError(
                "scaled signature differences exceed floating-point range"
            )
        if component == 0.0 and left_float != right_float:
            raise ValueError(
                "scaled signature separation is below floating-point range; "
                "use better-conditioned feature scales"
            )
        components.append(abs(component))
    if norm == "linf":
        distance = max(components, default=0.0)
    else:
        # ``hypot`` scales internally, unlike a direct sum of squares.
        distance = float(math.hypot(*components))
    if not math.isfinite(distance):
        raise ValueError("signature distance exceeds floating-point range")
    return distance


def _exact_scaled_metric(left: Any, right: Any, scales: Any, norm: str) -> Fraction:
    """Return the exact binary-input distance, squared for the L2 norm."""

    components = tuple(
        abs(
            Fraction.from_float(float(left_value))
            - Fraction.from_float(float(right_value))
        )
        / Fraction.from_float(float(scale))
        for left_value, right_value, scale in zip(left, right, scales, strict=True)
    )
    if norm == "linf":
        return max(components, default=Fraction(0))
    return sum((component * component for component in components), Fraction(0))


def _distance_lower_float(
    metric: Fraction,
    norm: str,
    candidate: float,
) -> float:
    """Round an exact distance downward to a finite binary64 value."""

    if metric == 0:
        return 0.0
    result = float(candidate)

    def is_above_exact(value: float) -> bool:
        as_fraction = Fraction.from_float(value)
        return (
            as_fraction > metric
            if norm == "linf"
            else as_fraction * as_fraction > metric
        )

    def is_at_or_below_exact(value: float) -> bool:
        as_fraction = Fraction.from_float(value)
        return (
            as_fraction <= metric
            if norm == "linf"
            else as_fraction * as_fraction <= metric
        )

    while is_above_exact(result):
        result = math.nextafter(result, 0.0)
    while True:
        successor = math.nextafter(result, math.inf)
        if not math.isfinite(successor) or not is_at_or_below_exact(successor):
            break
        result = successor
    if result == 0.0:
        raise ValueError(
            "signature separation is below floating-point range; use "
            "better-conditioned feature scales"
        )
    return result


def _metric_is_inside_radius(metric: Fraction, norm: str, radius: float) -> bool:
    """Compare an exact binary-input distance with an open float radius."""

    radius_fraction = Fraction.from_float(radius)
    if norm == "linf":
        return metric < radius_fraction
    return metric < radius_fraction * radius_fraction


def certify_signature_noise_margin(
    signatures: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    feature_names: Sequence[str] | None = None,
    feature_scales: Sequence[float] | None = None,
    norm: str = "linf",
) -> SignatureNoiseMarginCertificate:
    """Certify bounded-noise nearest-prototype identification.

    Distances are computed after dividing each feature by its declared positive
    scale.  ``norm`` is either ``"linf"`` or ``"l2"``.  With minimum
    pairwise prototype distance ``delta``, the open ball ``||e|| < delta/2``
    is a sharp uniform sufficient condition for unique nearest-prototype
    recovery over this finite hypothesis set. Equality is excluded because a
    midpoint can tie two prototypes. The theorem is analytic; this routine
    reports a conservative binary64 lower bound computed from exact rational
    arithmetic on the supplied floats, and rejects a nonzero separation when
    the selected scales make that bound unrepresentable.

    Unit scales are used only when the caller omits ``feature_scales``; the
    returned certificate records that choice explicitly.
    """

    if not isinstance(norm, str):
        raise TypeError("norm must be 'linf' or 'l2'")
    normalized_norm = norm.lower()
    if normalized_norm not in {"linf", "l2"}:
        raise ValueError("norm must be 'linf' or 'l2'")

    algebra = certify_temporal_signature_matrix(
        signatures,
        labels,
        feature_names=feature_names,
    )
    if len(algebra.labels) < 2:
        raise ValueError("at least two labelled signatures are required")
    matrix = np.asarray(algebra.signatures, dtype=float)
    scales = _validated_feature_scales(feature_scales, int(matrix.shape[1]))

    pair_distances: list[tuple[int, int, Fraction, float]] = []
    for left_index in range(int(matrix.shape[0])):
        for right_index in range(left_index + 1, int(matrix.shape[0])):
            floating_distance = _scaled_distance(
                matrix[left_index],
                matrix[right_index],
                scales,
                normalized_norm,
            )
            pair_distances.append(
                (
                    left_index,
                    right_index,
                    _exact_scaled_metric(
                        matrix[left_index],
                        matrix[right_index],
                        scales,
                        normalized_norm,
                    ),
                    floating_distance,
                )
            )
    minimum_metric = min(metric for _, _, metric, _ in pair_distances)
    representative = next(
        floating
        for _, _, metric, floating in pair_distances
        if metric == minimum_metric
    )
    minimum = _distance_lower_float(
        minimum_metric,
        normalized_norm,
        representative,
    )
    closest = tuple(
        (algebra.labels[left], algebra.labels[right])
        for left, right, metric, _ in pair_distances
        if metric == minimum_metric
    )
    distinct = algebra.all_rows_identifiable
    radius = float(0.5 * minimum)
    if distinct and radius == 0.0:
        raise ValueError(
            "half the minimum signature separation is below floating-point "
            "range; use better-conditioned feature scales"
        )
    return SignatureNoiseMarginCertificate(
        labels=algebra.labels,
        feature_names=algebra.feature_names,
        signatures=algebra.signatures,
        feature_scales=tuple(float(value) for value in scales),
        norm=normalized_norm,
        minimum_pairwise_distance=float(minimum),
        certified_noise_radius=radius,
        closest_pairs=closest,
        all_prototypes_distinct=distinct,
        strict_bound_required=True,
        claim_status=(
            "ANALYTIC half-separation theorem instantiated on the supplied "
            "finite binary64 signature set in the declared scaled norm; the "
            "reported minimum is rounded downward and observations require "
            "strict error below the reported radius"
        ),
    )


def identify_nearest_signature(
    observation: Sequence[float],
    certificate: SignatureNoiseMarginCertificate,
) -> NearestSignatureIdentification:
    """Identify the nearest supplied prototype and report theorem applicability.

    ``robustly_identified`` is true only when the nearest row is unique and the
    observation lies in its certified open noise ball.  A result outside that
    ball remains a nearest-prototype decision but has no robustness guarantee.
    """

    if not isinstance(certificate, SignatureNoiseMarginCertificate):
        raise TypeError("certificate must be a SignatureNoiseMarginCertificate")
    if isinstance(observation, (str, bytes)):
        raise TypeError("observation must be a sequence of finite real numbers")
    try:
        raw_observation = tuple(observation)
    except TypeError as exc:
        raise TypeError(
            "observation must be a sequence of finite real numbers"
        ) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        for value in raw_observation
    ):
        raise TypeError("observation must contain real numbers")
    observed = np.asarray(raw_observation, dtype=float)
    n_features = len(certificate.feature_names)
    if observed.ndim != 1 or int(observed.size) != n_features:
        raise ValueError("observation length must match the signature width")
    if not bool(np.all(np.isfinite(observed))):
        raise ValueError("observation must contain only finite values")

    matrix = np.asarray(certificate.signatures, dtype=float)
    scales = np.asarray(certificate.feature_scales, dtype=float)
    distance_pairs = tuple(
        (
            _exact_scaled_metric(observed, row, scales, certificate.norm),
            _scaled_distance(observed, row, scales, certificate.norm),
        )
        for row in matrix
    )
    nearest_metric = min(metric for metric, _ in distance_pairs)
    nearest_indices = tuple(
        index
        for index, (metric, _) in enumerate(distance_pairs)
        if metric == nearest_metric
    )
    nearest_labels = tuple(certificate.labels[index] for index in nearest_indices)
    nearest_distance = min(distance_pairs[index][1] for index in nearest_indices)
    ordered_metrics = sorted(metric for metric, _ in distance_pairs)
    second_metric = ordered_metrics[1]
    second_distance = min(
        floating
        for metric, floating in distance_pairs
        if metric == second_metric
    )
    within_radius = _metric_is_inside_radius(
        nearest_metric,
        certificate.norm,
        certificate.certified_noise_radius,
    )
    robust = (
        certificate.all_prototypes_distinct
        and len(nearest_indices) == 1
        and within_radius
    )
    return NearestSignatureIdentification(
        observation=tuple(float(value) for value in observed),
        nearest_labels=nearest_labels,
        nearest_distance=float(nearest_distance),
        second_nearest_distance=second_distance,
        within_certified_radius=within_radius,
        robustly_identified=robust,
        identified_label=nearest_labels[0] if len(nearest_labels) == 1 else None,
        claim_status=(
            "CERTIFIED within the supplied finite prototype family"
            if robust
            else (
                "UNCERTIFIED nearest-prototype result outside the strict margin "
                "or with a tie"
            )
        ),
    )


def _prepare_probe_context(graph: Any, target: Any, operator_name: str) -> None:
    """Install grammar history without altering the quantitative probe state."""

    from ..operators.grammar_debt import reset_grammar_state_from_history

    if operator_name in {"mutation", "self_organization"}:
        history = ["AL", "IL", "OZ"]
    else:
        history = ["AL", "IL"]
    graph.nodes[target]["glyph_history"] = history
    reset_grammar_state_from_history(graph.nodes[target])


def _build_probe(n_nodes: int, seed: int, probe_index: int) -> tuple[Any, int]:
    """Create one connected, phase-compatible deterministic graph probe."""

    import networkx as nx

    rng = random.Random(seed)
    graph = nx.path_graph(n_nodes)
    target = n_nodes // 2
    pressure_sign = -1.0 if probe_index % 2 else 1.0
    reverse_form = bool(probe_index % 3)
    for node in graph.nodes():
        position = (n_nodes - 1 - node) if reverse_form else node
        epi = 0.24 + 0.075 * position + rng.uniform(-0.012, 0.012)
        vf = 0.66 + 0.09 * ((2 * node + probe_index) % n_nodes)
        dnfr = pressure_sign * (0.18 + 0.055 * ((node + 1) % 4))
        if node % 2:
            dnfr *= -0.72
        theta = 0.18 + 0.11 * node + rng.uniform(-0.008, 0.008)
        history = [max(-1.0, epi - 0.35), epi - 0.28, epi]
        d2_epi = history[-1] - 2.0 * history[-2] + history[-3]
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: epi,
                ALIAS_VF[0]: vf,
                ALIAS_DNFR[0]: dnfr,
                ALIAS_THETA[0]: theta,
                ALIAS_SI[0]: 0.55 + 0.04 * (node % 3),
                ALIAS_EPI_KIND[0]: "probe",
                ALIAS_D2EPI[0]: d2_epi,
                "epi_history": history,
                "glyph_history": ["AL", "IL"],
            }
        )
    graph.graph.update(
        {
            "RANDOM_SEED": seed,
            "NAV_RANDOM": False,
            "UM_FUNCTIONAL_LINKS": False,
            "THOL_METABOLIC_ENABLED": False,
            "THOL_PROPAGATION_ENABLED": False,
        }
    )
    return graph, target


def _state_matrix(graph: Any, nodes: Sequence[Any]) -> Any:
    return np.asarray(
        [
            [
                float(get_attr(graph.nodes[node], ALIAS_EPI, 0.0)),
                float(get_attr(graph.nodes[node], ALIAS_VF, 0.0)),
                float(get_attr(graph.nodes[node], ALIAS_DNFR, 0.0)),
                float(get_attr(graph.nodes[node], ALIAS_THETA, 0.0)),
            ]
            for node in nodes
        ],
        dtype=float,
    )


def _wrapped_phase_delta(after: Any, before: Any) -> Any:
    """Return elementwise phase changes in the canonical half-open interval."""

    raw = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    wrapped = np.fromiter(
        (wrap_angle(float(value)) for value in raw.flat),
        dtype=float,
        count=int(raw.size),
    )
    return wrapped.reshape(raw.shape)


def _frame_signature(
    graph: Any,
    original_nodes: Sequence[Any],
    target: Any,
    before: Any,
    before_node_count: int,
    before_edge_count: int,
) -> tuple[float, ...]:
    after = _state_matrix(graph, original_nodes)
    changes = after - before
    changes[:, 3] = _wrapped_phase_delta(after[:, 3], before[:, 3])
    target_index = tuple(original_nodes).index(target)
    target_delta = changes[target_index]
    mean_delta = np.mean(changes, axis=0)
    rms_delta = np.sqrt(np.mean(changes * changes, axis=0))

    velocity_before = before[:, 1] * before[:, 2]
    velocity_after = after[:, 1] * after[:, 2]
    velocity_delta = velocity_after - velocity_before
    velocity_features = (
        float(velocity_delta[target_index]),
        float(np.mean(velocity_delta)),
        float(np.sqrt(np.mean(velocity_delta * velocity_delta))),
    )
    graph_size = (
        float(graph.number_of_nodes() - before_node_count),
        float(graph.number_of_edges() - before_edge_count),
    )
    return tuple(float(value) for value in target_delta) + tuple(
        float(value) for value in mean_delta
    ) + tuple(float(value) for value in rms_delta) + velocity_features + graph_size


def _frame_feature_names(prefix: str) -> tuple[str, ...]:
    names: list[str] = []
    for statistic in ("target_delta", "mean_node_delta", "rms_node_delta"):
        for channel in _CHANNELS:
            names.append(f"{prefix}_{statistic}_{channel}")
    names.extend(
        (
            f"{prefix}_target_delta_nodal_velocity",
            f"{prefix}_mean_delta_nodal_velocity",
            f"{prefix}_rms_delta_nodal_velocity",
            f"{prefix}_delta_node_count",
            f"{prefix}_delta_edge_count",
        )
    )
    return tuple(names)


def _group_contains_pair(
    groups: Sequence[Sequence[str]], first: str, second: str
) -> bool:
    return any(first in group and second in group for group in groups)


def probe_canonical_operator_identifiability(
    *,
    probe_seeds: Sequence[int] = (7, 29, 101),
    n_nodes: int = 6,
    quantization_decimals: int | None = 12,
    rank_tolerance: float | None = None,
) -> TemporalOperatorIdentifiabilityCertificate:
    """Measure one-step temporal signatures for all 13 canonical operators.

    Every operator is applied to a fresh graph for every seed.  The graph
    family consists of connected path graphs with heterogeneous EPI, ``νf``,
    ``ΔNFR`` and compatible phases.  Context histories satisfy incremental
    grammar without changing the numerical initial condition.  The executed
    glyph is checked so a grammar fallback cannot be mislabelled as the
    requested operator.

    Observed features contain no names, glyphs, contract categories, or stored
    metadata.  They are one-step changes in the four nodal channels, the nodal
    velocity ``νf·ΔNFR``, and node/edge counts.  The experiment therefore tests
    finite-probe separation, including the categorical Silence/Contraction
    collision, rather than global inversion.
    """

    if isinstance(n_nodes, bool) or not isinstance(n_nodes, Integral):
        raise TypeError("n_nodes must be an integer")
    n_nodes = int(n_nodes)
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least 3")
    if isinstance(probe_seeds, (str, bytes)):
        raise TypeError("probe_seeds must be a sequence of integers")
    seeds = tuple(validate_seed(seed, allow_none=False) for seed in probe_seeds)
    if not seeds:
        raise ValueError("probe_seeds must be nonempty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("probe_seeds must not contain duplicates")
    decimals = _validate_decimals(quantization_decimals)
    rank_tol = _validate_rank_tolerance(rank_tolerance)

    from ..operators.grammar_types import glyph_function_name
    from ..operators.operator_contracts import iter_contracts
    from ..operators.registry import get_operator_class

    contracts = tuple(iter_contracts())
    requested = tuple(contract.name for contract in contracts)
    display_labels = tuple(contract.english_name for contract in contracts)
    expected_glyphs = {contract.name: contract.glyph for contract in contracts}
    signatures_by_operator: dict[str, list[float]] = {
        name: [] for name in requested
    }
    feature_names: list[str] = []
    executed: list[str] = []

    for probe_index, seed in enumerate(seeds):
        feature_names.extend(_frame_feature_names(f"probe_{probe_index}"))
        for operator_name in requested:
            graph, target = _build_probe(n_nodes, seed, probe_index)
            _prepare_probe_context(graph, target, operator_name)
            original_nodes = tuple(graph.nodes())
            before = _state_matrix(graph, original_nodes)
            before_node_count = graph.number_of_nodes()
            before_edge_count = graph.number_of_edges()
            operator = get_operator_class(operator_name)()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                operator(
                    graph,
                    target,
                    validate_preconditions=False,
                    collect_metrics=False,
                )
            history = graph.nodes[target].get("glyph_history", ())
            actual = glyph_function_name(history[-1]) if history else ""
            if actual != operator_name:
                raise RuntimeError(
                    f"probe requested {operator_name!r} but executed {actual!r}"
                )
            if probe_index == 0:
                executed.append(actual)
            signatures_by_operator[operator_name].extend(
                _frame_signature(
                    graph,
                    original_nodes,
                    target,
                    before,
                    before_node_count,
                    before_edge_count,
                )
            )

    signature_rows = tuple(
        tuple(signatures_by_operator[name]) for name in requested
    )
    algebra = certify_temporal_signature_matrix(
        signature_rows,
        display_labels,
        feature_names=tuple(feature_names),
        quantization_decimals=decimals,
        rank_tolerance=rank_tol,
    )
    margin_rows = (
        signature_rows
        if decimals is None
        else tuple(
            tuple(round(value, decimals) for value in row)
            for row in signature_rows
        )
    )
    noise_margin = certify_signature_noise_margin(
        margin_rows,
        display_labels,
        feature_names=tuple(feature_names),
        feature_scales=tuple(1.0 for _ in feature_names),
        norm="linf",
    )
    # Execution is reported separately from the quantitative feature matrix.
    # This validation metadata prevents fallback mislabelling but cannot make
    # two equal observed rows appear distinct.
    if tuple(executed) != requested:
        raise RuntimeError("canonical operator execution order drifted")
    if set(expected_glyphs) != set(requested):
        raise RuntimeError("canonical operator contract coverage drifted")

    collision_remains = _group_contains_pair(
        algebra.equivalence_classes, "Silence", "Contraction"
    )
    return TemporalOperatorIdentifiabilityCertificate(
        probe_seeds=seeds,
        n_nodes=n_nodes,
        target_node=n_nodes // 2,
        frames_per_probe=1,
        algebra=algebra,
        requested_operators=requested,
        executed_operators=tuple(executed),
        silence_contraction_separated=not collision_remains,
        unresolved_groups=algebra.ambiguous_groups,
        claim_status=(
            "MEASURED on deterministic one-step heterogeneous path-graph probes; "
            "the attached unit-scaled L-infinity margin uses the same declared "
            "quantization and proves robustness only inside this finite "
            "prototype family; no claim for unseen states, "
            "stochastic observation laws, compositions, or arbitrary operator words"
        ),
        noise_margin=noise_margin,
    )
