"""Meet an organized TNFR pattern, certify its recovery and advance its state.

The prepared object is a five-node ring with one phase turn and a small signed
form contrast. Winding identifies its recovery class; the limiting pattern is
defined modulo common form/phase offsets. The native relational law evolves
form and phase on held support and positive held capacities, with no forcing
or operator events. Recovery is
conditional on that complete law; the supplied pattern's formation is a
separate question. Run after installing the repository.
"""

from __future__ import annotations

from fractions import Fraction as Q

import networkx as nx

from tnfr.sdk import Network, RelationalExchangeModel


def hello_world() -> dict[str, object]:
    """Return the endpoint network, both capture reports and the complete step."""
    graph = nx.cycle_graph(5)
    cycle = tuple(range(5))
    for node, contrast in enumerate((1, -1, 0, 0, 0)):
        graph.nodes[node].update(EPI=Q(contrast, 1024), theta=Q(5 * node, 4), nu_f=1)
    nx.set_edge_attributes(graph, 1, "weight")
    graph.graph["GAMMA"] = {"type": "none"}
    network = Network(graph)
    model = RelationalExchangeModel(
        storage_scale=1, epi_weight=0.5, phase_weight=0.5, phase_domain="acute"
    )

    # The shared certificate reads the whole prepared state and its law.
    initial = network.relational_cycle_capture(model, cycle=cycle)
    print("A prepared TNFR object: a five-node winding pattern")
    print(
        f"Initial winding: {initial.cycle_winding} "
        "(one complete phase turn around the ordered ring)"
    )
    print(f"Ideal-law recovery certificate: {initial.status}")
    print(f"Certified storage margin: {float(initial.storage_margin_lower_bound):.6g}")

    # This is structural time, not an operator count or laboratory seconds.
    step = network.step_relational(model, dt=Q(1, 1024), t=0)
    final = network.relational_cycle_capture(model, cycle=cycle)
    print(f"One joint form/phase step to structural time {step.t_after:.8f}")
    print(f"Form at node 0: {step.before.epi[0]:.8f} -> {step.after.epi[0]:.8f}")
    print(f"Phase at node 0: {step.before.phase[0]:.8f} -> {step.after.phase[0]:.8f}")
    print(f"Initial continuous storage loss: {float(step.before.continuous_loss):.6g}")
    print(f"Euler storage-balance defect: {float(step.energy_step_defect):.6g}")
    print(
        f"Endpoint winding: {final.cycle_winding}; "
        f"ideal-law recovery certificate: {final.status}"
    )
    print(
        "Each certificate concerns ideal continuation from its own snapshot. "
        "The finite step retains its numerical defects separately."
    )
    print("Next: docs/guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md")
    return {
        "network": network,
        "initial_capture": initial,
        "step": step,
        "final_capture": final,
        "network_size": network.G.number_of_nodes(),
    }


if __name__ == "__main__":
    hello_world()
