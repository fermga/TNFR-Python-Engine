"""Finite nonnegative row normalization for scalar and graph transport kernels."""

from __future__ import annotations

from .unified_numerical import np


def normalize_weights(weights, *, source=None, node_count=None):
    """Return probabilities, row maxima and row totals in scaled coordinates.

    ``source=None`` accepts a vector (one row) or a dense matrix. Otherwise
    ``weights`` and integer ``source`` are edge vectors and ``node_count`` is
    the number of rows, including empty ones. No dense matrix is formed for
    edge input. Zero rows stay zero; a walk's absorbing diagonal is a caller
    convention. The input is never changed.

    For a positive row, ``scale=max(weights)`` and
    ``total=sum(weights/scale)`` represent its strength as ``scale*total``
    without forming that possibly overflowing product. Probabilities smaller
    than the floating-point range round to zero; no positive raw strength is
    represented by infinity. Weights must already be effective conductances
    (parallel-edge aggregation belongs to the graph reader).
    """
    try:
        if np.iscomplexobj(weights):
            raise ValueError("Weights must be real")
        values = np.asarray(weights, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Weights require finite nonnegative real values") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Weights require finite nonnegative real values")

    if source is None:
        if values.ndim not in (1, 2):
            raise ValueError("Weights must be a vector or matrix")
        rows = values.reshape(1, -1) if values.ndim == 1 else values
        scale = np.max(rows, axis=1, initial=0.0)
        positive = scale > 0.0
        normalized = np.zeros_like(rows)
        with np.errstate(under="ignore"):
            np.divide(rows, scale[:, None], out=normalized, where=positive[:, None])
            total = normalized.sum(axis=1)
            np.divide(normalized, total[:, None], out=normalized, where=positive[:, None])
        return normalized.reshape(values.shape), scale, total

    indices = np.asarray(source)
    if (values.ndim != 1 or indices.ndim != 1 or indices.shape != values.shape
            or not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("Sparse weights require one integer source index per edge")
    if (isinstance(node_count, bool) or not isinstance(node_count, (int, np.integer))
            or node_count < 0 or np.any(indices < 0) or np.any(indices >= node_count)):
        raise ValueError("Sparse source indices must lie within node_count")
    scale = np.zeros(node_count, dtype=float)
    np.maximum.at(scale, indices, values)
    normalized = np.zeros_like(values)
    edge_scale = scale[indices]
    with np.errstate(under="ignore"):
        np.divide(values, edge_scale, out=normalized, where=edge_scale > 0.0)
        total = np.bincount(indices, weights=normalized, minlength=node_count)
        edge_total = total[indices]
        np.divide(normalized, edge_total, out=normalized, where=edge_total > 0.0)
    return normalized, scale, total
