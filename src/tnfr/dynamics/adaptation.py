"""νf adaptation routines for TNFR dynamics."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

from ..alias import collect_attr, set_vf
from ..constants import get_graph_param
from ..helpers.numeric import clamp
from ..metrics.common import ensure_neighbors_map
from ..types import CoherenceMetric, DeltaNFR, NodeId, TNFRGraph
from ..utils import get_numpy
from .aliases import ALIAS_DNFR, ALIAS_SI, ALIAS_VF

__all__ = ("adapt_vf_by_coherence",)


def _vf_adapt_chunk(
    args: tuple[list[tuple[Any, int, tuple[int, ...]]], tuple[float, ...], float]
) -> list[tuple[Any, float]]:
    """Return proposed νf updates for ``chunk`` of stable nodes."""

    chunk, vf_values, mu = args
    updates: list[tuple[Any, float]] = []
    for node, idx, neighbor_idx in chunk:
        vf = vf_values[idx]
        if neighbor_idx:
            mean = math.fsum(vf_values[j] for j in neighbor_idx) / len(neighbor_idx)
        else:
            mean = vf
        updates.append((node, vf + mu * (mean - vf)))
    return updates


def adapt_vf_by_coherence(G: TNFRGraph, n_jobs: int | None = None) -> None:
    """Adjust νf toward neighbour mean in nodes with sustained stability."""

    tau = get_graph_param(G, "VF_ADAPT_TAU", int)
    mu = float(get_graph_param(G, "VF_ADAPT_MU"))
    eps_dnfr = cast(DeltaNFR, get_graph_param(G, "EPS_DNFR_STABLE"))
    thr_sel = get_graph_param(G, "SELECTOR_THRESHOLDS", dict)
    thr_def = get_graph_param(G, "GLYPH_THRESHOLDS", dict)
    si_hi = cast(
        CoherenceMetric,
        float(thr_sel.get("si_hi", thr_def.get("hi", 0.66))),
    )
    vf_min = float(get_graph_param(G, "VF_MIN"))
    vf_max = float(get_graph_param(G, "VF_MAX"))

    nodes = list(G.nodes)
    if not nodes:
        return

    neighbors_map = ensure_neighbors_map(G)
    node_count = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    jobs: int | None
    if n_jobs is None:
        jobs = None
    else:
        try:
            jobs = int(n_jobs)
        except (TypeError, ValueError):
            jobs = None
        else:
            if jobs <= 1:
                jobs = None

    np_mod = get_numpy()
    use_np = np_mod is not None

    si_values = collect_attr(G, nodes, ALIAS_SI, 0.0, np=np_mod if use_np else None)
    dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np_mod if use_np else None)
    vf_values = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np_mod if use_np else None)

    if use_np:
        np = np_mod  # type: ignore[assignment]
        assert np is not None
        si_arr = si_values.astype(float, copy=False)
        dnfr_arr = np.abs(dnfr_values.astype(float, copy=False))
        vf_arr = vf_values.astype(float, copy=False)

        prev_counts = np.fromiter(
            (int(G.nodes[node].get("stable_count", 0)) for node in nodes),
            dtype=int,
            count=node_count,
        )
        stable_mask = (si_arr >= si_hi) & (dnfr_arr <= eps_dnfr)
        new_counts = np.where(stable_mask, prev_counts + 1, 0)

        for node, count in zip(nodes, new_counts.tolist()):
            G.nodes[node]["stable_count"] = int(count)

        eligible_mask = new_counts >= tau
        if not bool(eligible_mask.any()):
            return

        max_degree = 0
        if node_count:
            degree_counts = np.fromiter(
                (len(neighbors_map.get(node, ())) for node in nodes),
                dtype=int,
                count=node_count,
            )
            if degree_counts.size:
                max_degree = int(degree_counts.max())

        if max_degree > 0:
            neighbor_indices = np.zeros((node_count, max_degree), dtype=int)
            mask = np.zeros((node_count, max_degree), dtype=bool)
            for idx, node in enumerate(nodes):
                neigh = neighbors_map.get(node, ())
                if not neigh:
                    continue
                idxs = [node_index[nbr] for nbr in neigh if nbr in node_index]
                if not idxs:
                    continue
                length = len(idxs)
                neighbor_indices[idx, :length] = idxs
                mask[idx, :length] = True
            neighbor_values = vf_arr[neighbor_indices]
            sums = (neighbor_values * mask).sum(axis=1)
            counts = mask.sum(axis=1)
            neighbor_means = np.where(counts > 0, sums / counts, vf_arr)
        else:
            neighbor_means = vf_arr

        vf_updates = vf_arr + mu * (neighbor_means - vf_arr)
        for idx in np.nonzero(eligible_mask)[0]:
            node = nodes[int(idx)]
            vf_new = clamp(float(vf_updates[int(idx)]), vf_min, vf_max)
            set_vf(G, node, vf_new)
        return

    si_list = [float(val) for val in si_values]
    dnfr_list = [abs(float(val)) for val in dnfr_values]
    vf_list = [float(val) for val in vf_values]

    prev_counts = [int(G.nodes[node].get("stable_count", 0)) for node in nodes]
    stable_flags = [
        si >= si_hi and dnfr <= eps_dnfr
        for si, dnfr in zip(si_list, dnfr_list)
    ]
    new_counts = [prev + 1 if flag else 0 for prev, flag in zip(prev_counts, stable_flags)]

    for node, count in zip(nodes, new_counts):
        G.nodes[node]["stable_count"] = int(count)

    eligible_nodes = [node for node, count in zip(nodes, new_counts) if count >= tau]
    if not eligible_nodes:
        return

    if jobs is None:
        for node in eligible_nodes:
            idx = node_index[node]
            neigh_indices = [
                node_index[nbr]
                for nbr in neighbors_map.get(node, ())
                if nbr in node_index
            ]
            if neigh_indices:
                total = math.fsum(vf_list[i] for i in neigh_indices)
                mean = total / len(neigh_indices)
            else:
                mean = vf_list[idx]
            vf_new = vf_list[idx] + mu * (mean - vf_list[idx])
            set_vf(G, node, clamp(float(vf_new), vf_min, vf_max))
        return

    work_items: list[tuple[Any, int, tuple[int, ...]]] = []
    for node in eligible_nodes:
        idx = node_index[node]
        neigh_indices = tuple(
            node_index[nbr]
            for nbr in neighbors_map.get(node, ())
            if nbr in node_index
        )
        work_items.append((node, idx, neigh_indices))

    chunk_size = max(1, math.ceil(len(work_items) / jobs))
    chunks = [
        work_items[i : i + chunk_size]
        for i in range(0, len(work_items), chunk_size)
    ]
    vf_tuple = tuple(vf_list)
    updates: dict[Any, float] = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        args = ((chunk, vf_tuple, mu) for chunk in chunks)
        for chunk_updates in executor.map(_vf_adapt_chunk, args):
            for node, value in chunk_updates:
                updates[node] = float(value)

    for node in eligible_nodes:
        vf_new = updates.get(node)
        if vf_new is None:
            continue
        set_vf(G, node, clamp(float(vf_new), vf_min, vf_max))

