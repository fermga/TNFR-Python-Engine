"""One fresh sparse conductance snapshot for nodal transport read-outs.

This is the weighted adjacency convention of the EPI diffusion channel:
parallel edges aggregate, loops count once per row, and zero-strength rows
have no transport. No graph state or persistent cache is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np
from ..mathematics._weight_normalization import normalize_weights


@dataclass(frozen=True)
class ConductanceSnapshot:
    """Detached O(V+E) adjacency data in the requested node order."""

    nodes: list
    source: Any
    target: Any
    weight: Any

    @property
    def strength(self) -> Any:
        """Materialize raw row strengths only when the requested output needs them.

        A normalized walk can exist even when a finite row's sum exceeds the
        floating-point range. Raw strengths have a narrower numerical domain.
        """
        strength = np.asarray(np.bincount(
            self.source, weights=self.weight, minlength=len(self.nodes),
        ), dtype=float)
        if not np.all(np.isfinite(strength)):
            raise ValueError("Diffusion row strength exceeds finite floating-point range")
        return strength

    def normalization(self) -> tuple[Any, Any, Any]:
        """Return edge probabilities and a scaled representation of each strength."""
        return normalize_weights(self.weight, source=self.source, node_count=len(self.nodes))

    def weighted_total(self, field: Any) -> float:
        """Sum W_ij*field_i without requiring representable intermediate degrees.

        One-sign sums use compensated floating-point arithmetic. Mixed signs
        and range-limited intermediates use the shared exact binary-ratio
        reducer before rounding the final scalar; compensation of already
        rounded products cannot recover every cancellation residual.
        """
        import math

        values = field[self.source]
        mixed_signs = np.any(values > 0.0) and np.any(values < 0.0)
        if not mixed_signs:
            try:
                with np.errstate(over="raise", invalid="raise", under="raise"):
                    products = self.weight * values
                return math.fsum(products)
            except (FloatingPointError, OverflowError):
                pass
        from fractions import Fraction
        from ..mathematics._exact_weighted import exact_weighted_sum_ratio

        numerator, denominator = exact_weighted_sum_ratio(self.weight, values)
        try:
            total = float(Fraction(numerator, denominator))
        except OverflowError as exc:
            raise ValueError("Degree-weighted total exceeds finite floating-point range") from exc
        if numerator and total == 0.0:
            raise ValueError("Degree-weighted total is below floating-point range")
        return total

    def transition(self) -> Any:
        """Materialize the walk, with an absorbing diagonal at zero-strength rows."""
        probability, scale, _ = self.normalization()
        transition = self.dense(probability)
        sinks = np.flatnonzero(scale == 0.0)
        transition[sinks, sinks] = 1.0
        return transition

    def symmetric_normalized_weights(self) -> tuple[Any, Any]:
        """Return W_ij/sqrt(d_i*d_j) without raw degrees or reciprocal overflow.

        The caller must require symmetric adjacency. Taking square roots before
        ratios retains representable coefficients even when W_ij/d_i alone
        would round to zero.
        """
        _, scale, total = self.normalization()
        root_weight = np.sqrt(self.weight)
        root_scale, root_total = np.sqrt(scale), np.sqrt(total)
        with np.errstate(under="ignore"):
            left = root_weight / root_scale[self.source] / root_total[self.source]
            right = root_weight / root_scale[self.target] / root_total[self.target]
            normalized = left * right
        return normalized, scale > 0.0

    def relative_strength(self) -> Any:
        """Return strengths in a common finite scale for a stationary measure."""
        _, scale, total = self.normalization()
        scale_mantissa, scale_exponent = np.frexp(scale)
        total_mantissa, total_exponent = np.frexp(total)
        mantissa, adjustment = np.frexp(scale_mantissa * total_mantissa)
        exponent = scale_exponent + total_exponent + adjustment
        positive = scale > 0.0
        if not np.any(positive):
            return np.zeros(len(self.nodes), dtype=float)
        with np.errstate(under="ignore"):
            return np.ldexp(mantissa, exponent - np.max(exponent[positive]))

    def divide_by_strength(self, numerator: Any) -> Any:
        """Read numerator/d_i, zero at zero strength, without overflowing d_i.

        The ordinary path retains its direct division. The scaled fallback
        combines binary exponents so neither division order introduces an
        avoidable overflow or underflow. Unrepresentable positive results
        remain explicit errors, as required by the mobility read-out.
        """
        degree = np.bincount(self.source, weights=self.weight, minlength=len(self.nodes))
        result = np.zeros(len(self.nodes), dtype=float)
        positive = degree > 0.0
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                if np.all(np.isfinite(degree)):
                    np.divide(numerator, degree, out=result, where=positive)
                else:
                    _, scale, total = self.normalization()
                    number_m, number_e = np.frexp(numerator[positive])
                    scale_m, scale_e = np.frexp(scale[positive])
                    total_m, total_e = np.frexp(total[positive])
                    result[positive] = np.ldexp(
                        number_m / scale_m / total_m, number_e - scale_e - total_e,
                    )
        except FloatingPointError as exc:
            raise ValueError("Diffusion mobility exceeds finite floating-point range") from exc
        if np.any(positive & (numerator > 0.0) & (result == 0.0)):
            raise ValueError("Positive diffusion mobility is below floating-point range")
        return result

    def dense(self, values: Any = None) -> Any:
        """Materialize a matrix only for an API that actually returns one."""
        matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=float)
        matrix[self.source, self.target] = self.weight if values is None else values
        return matrix

    def divergence(self, flux: Any) -> Any:
        """Sum outgoing edge flux in O(V+E), without a current matrix."""
        result = np.bincount(self.source, weights=flux, minlength=len(self.nodes))
        if not np.all(np.isfinite(result)):
            raise ValueError("Current divergence exceeds finite floating-point range")
        return np.asarray(result, dtype=float)


def read_conductance(
    G: Any, nodes: list | None = None, *, symmetric: bool = False,
) -> ConductanceSnapshot:
    """Read effective finite nonnegative weights, optionally requiring symmetry.

    The explicit node list describes an induced adjacency, like NetworkX's
    matrix interface. Validation applies to aggregated parallel conductance;
    a zero-weight one-way arc does not break effective matrix symmetry.
    """
    import networkx as nx

    nodes = list(G) if nodes is None else list(nodes)
    indices = {node: index for index, node in enumerate(nodes)}
    if len(indices) != len(nodes):
        raise nx.NetworkXError("nodelist contains duplicates")
    if any(node not in G for node in nodes):
        raise nx.NetworkXError("nodelist contains nodes not in the graph")
    multiple = G.is_multigraph()
    entries = {}
    try:
        for node, source in indices.items():
            for neighbor, attributes in G.adj[node].items():
                if neighbor not in indices:
                    continue
                weight = (float(sum(data.get("weight", 1.0) for data in attributes.values()))
                          if multiple else float(attributes.get("weight", 1.0)))
                if not np.isfinite(weight) or weight < 0.0:
                    raise ValueError("Diffusion requires finite nonnegative edge weights")
                if weight:
                    entries[source, indices[neighbor]] = weight
    except (TypeError, OverflowError) as exc:
        raise ValueError("Diffusion requires finite nonnegative edge weights") from exc
    if symmetric and any(entries.get((target, source)) != weight
                         for (source, target), weight in entries.items()):
        raise ValueError("This transport formula requires symmetric adjacency")
    source = np.fromiter((pair[0] for pair in entries), dtype=np.intp, count=len(entries))
    target = np.fromiter((pair[1] for pair in entries), dtype=np.intp, count=len(entries))
    weight = np.fromiter(entries.values(), dtype=float, count=len(entries))
    return ConductanceSnapshot(nodes, source, target, weight)
