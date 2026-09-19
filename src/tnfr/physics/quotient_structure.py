"""Exact inherited support metadata for a declared equitable partition.

Unique-neighbor multiplicities and conductance are separate structures. Their
quotients need not be the adjacency and degree of a new simple macro graph.
This observer supplies no phase law, weighted EPI closure or causal certificate.
"""

from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real
from ._cycle_algebra import Matrix, Vector, ordered_vector
from .epi_memory import _exact_partition
from .support_transport import SupportTransportSnapshot, _rebuild

__all__ = ["QuotientStructure", "observe_quotient_structure"]


@dataclass(frozen=True)
class QuotientStructure:
    """Detached metadata; ``node_blocks`` follows ``source.nodes`` order.

    ``aggregated_conductance`` includes its internal diagonal; the ordinary
    ``macro_conductance`` removes it. ``block_capacity`` is retained fine
    capacity, whereas ``effective_capacity`` is cross strength divided by the
    full inherited metric. ``source_scale`` converts a block-constant fine
    pressure into pressure paired with that effective capacity. It is not a
    fitted gain. Weighted EPI closure and source block constancy are separate
    obligations. Public construction does not authenticate any execution.
    """

    source: SupportTransportSnapshot
    blocks: tuple[tuple, ...]
    node_blocks: tuple[int, ...]
    multiplicity: tuple[tuple[int, ...], ...]
    block_degree: tuple[int, ...]
    block_capacity: Vector
    aggregated_conductance: Matrix
    macro_conductance: Matrix
    fine_strengths: Vector
    metric_weights: Vector
    macro_metric_weights: Vector
    macro_strengths: Vector
    effective_capacity: Vector
    source_scale: Vector
    capacity_gradient: Vector
    topology_gradient: Vector
    scope: str


def _checked_snapshot(snapshot):
    """Reuse primitive validation and reject inconsistent derived caches."""
    source = _rebuild(snapshot)
    for field in (
        "epi_gradient",
        "capacity_gradient",
        "topology_gradient",
        "dirichlet_gradient",
        "rate",
    ):
        supplied = ordered_vector(getattr(snapshot, field), field)
        if supplied != getattr(source, field):
            raise ValueError(f"snapshot {field} differs from rebuilt support data")
    for field in ("dirichlet_energy", "energy_rate"):
        supplied = exact_or_represented_real(getattr(snapshot, field), field)
        if supplied != getattr(source, field):
            raise ValueError(f"snapshot {field} differs from rebuilt support data")
    return source


def observe_quotient_structure(snapshot, blocks) -> QuotientStructure:
    r"""Derive exact support counts, conductance and capacity normalization.

    The ordered partition is a strict reduction to at least two blocks.
    Positive symmetric transport must be connected, all capacities positive,
    and unique support reciprocal. Zero-conductance support edges still count;
    self-neighbors count once. Each fine node in a block must have exactly the
    same neighbor counts into every block and exactly the same capacity.

    N_ab counts unique support neighbors, including within-block neighbors;
    d_a=sum_b N_ab. Wbar=P^T W P includes its internal diagonal, while the
    cross-only version yields strengths sbar. With h_i=s_i/nu_i,
    hbar=P^T h, effective capacity=sbar/hbar. The ratio nu_a/nu_eff,a is the
    source rescaling; retaining nu_a and nu_eff,a separately is essential.

    The capacity and topology channels are respectively
    sum_b N_ab*nu_b/d_a-nu_a and sum_b N_ab*d_b/d_a-d_a.
    They reconstruct their fine block-constant gradients without another
    pressure kernel. Phase data, phase-resultant availability, weighted EPI
    equitability, field inheritance and evolution remain separate gates.
    No reference solve, graph construction, event or trajectory is executed.
    """
    source = _checked_snapshot(snapshot)
    partition = _exact_partition(source.nodes, blocks)
    size, count = len(source.nodes), len(partition)
    zero = Fraction(0)
    position = {node: i for i, node in enumerate(source.nodes)}
    indices = tuple(tuple(position[node] for node in block) for block in partition)
    node_blocks = [0] * size
    for a, row in enumerate(indices):
        for i in row:
            node_blocks[i] = a
    node_blocks = tuple(node_blocks)
    for i, row in enumerate(source.support_neighbors):
        if any(i not in source.support_neighbors[j] for j in row):
            raise ValueError("quotient structure requires reciprocal unique support")
    strengths, adjacency = [zero] * size, [set() for _ in source.nodes]
    aggregate = [[zero] * count for _ in partition]
    for i, j, weight in source.conductance:
        strengths[i] += weight
        adjacency[i].add(j)
        aggregate[node_blocks[i]][node_blocks[j]] += weight
    if any(s <= 0 for s in strengths) or any(nu <= 0 for nu in source.capacity):
        raise ValueError(
            "quotient structure requires positive strengths and capacities"
        )
    reached, pending = {0}, [0]
    while pending:
        new = adjacency[pending.pop()] - reached
        reached.update(new)
        pending.extend(new)
    if len(reached) != size:
        raise ValueError("quotient structure requires connected positive transport")

    profiles = []
    for row in source.support_neighbors:
        profile = [0] * count
        for j in row:
            profile[node_blocks[j]] += 1
        profiles.append(tuple(profile))
    multiplicity, capacities = [], []
    for row in indices:
        profile, capacity = profiles[row[0]], source.capacity[row[0]]
        if any(profiles[i] != profile for i in row):
            raise ValueError(
                "partition must have exact equitable support multiplicities"
            )
        if any(source.capacity[i] != capacity for i in row):
            raise ValueError("partition requires exactly block-constant capacity")
        multiplicity.append(profile)
        capacities.append(capacity)
    multiplicity, capacities = tuple(multiplicity), tuple(capacities)
    degree = tuple(sum(row) for row in multiplicity)
    metric = tuple(s / nu for s, nu in zip(strengths, source.capacity, strict=True))
    macro_metric = tuple(sum((metric[i] for i in row), zero) for row in indices)
    aggregate = tuple(tuple(row) for row in aggregate)
    cross = tuple(
        tuple(value if a != b else zero for b, value in enumerate(row))
        for a, row in enumerate(aggregate)
    )
    cross_strengths = tuple(sum(row, zero) for row in cross)
    # Connected positive transport and a proper partition imply positive cuts.
    if any(value <= 0 for value in cross_strengths):
        raise RuntimeError("connected quotient unexpectedly has a zero block cut")
    effective = tuple(s / h for s, h in zip(cross_strengths, macro_metric, strict=True))
    scales = tuple(nu / eff for nu, eff in zip(capacities, effective, strict=True))

    def gradient(values):
        return tuple(
            sum((n * values[b] for b, n in enumerate(row)), zero) / degree[a]
            - values[a]
            for a, row in enumerate(multiplicity)
        )

    capacity_gradient, topology_gradient = gradient(capacities), gradient(degree)
    for i, a in enumerate(node_blocks):
        if (
            source.capacity_gradient[i] != capacity_gradient[a]
            or source.topology_gradient[i] != topology_gradient[a]
        ):
            raise RuntimeError("inherited support-gradient identity failed")
    return QuotientStructure(
        source=source,
        blocks=partition,
        node_blocks=node_blocks,
        multiplicity=multiplicity,
        block_degree=degree,
        block_capacity=capacities,
        aggregated_conductance=aggregate,
        macro_conductance=cross,
        fine_strengths=tuple(strengths),
        metric_weights=metric,
        macro_metric_weights=macro_metric,
        macro_strengths=cross_strengths,
        effective_capacity=effective,
        source_scale=scales,
        capacity_gradient=capacity_gradient,
        topology_gradient=topology_gradient,
        scope=(
            "Exact inherited metadata on a supplied equitable reciprocal support partition "
            "with positive block-constant capacity and connected symmetric transport. "
            "Unique neighbors, conductance, retained capacity and effective capacity remain "
            "distinct. No weighted EPI closure, phase law, complete joint dynamics, field "
            "inheritance, partition selection, causal execution or NFR formation is certified."
        ),
    )
