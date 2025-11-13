"""Quick test to verify that operators actually change node attributes."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "benchmarks"))
sys.path.insert(0, str(_ROOT / "src"))

from benchmark_utils import (
    create_tnfr_topology,
    initialize_tnfr_nodes,
    generate_grammar_valid_sequence,
)
from tnfr.structural import run_sequence
from tnfr.physics.fields import compute_phase_gradient, compute_structural_potential
from tnfr.config import THETA_PRIMARY, DNFR_PRIMARY, EPI_PRIMARY

def _safe_numeric_diff(a, b):
    try:
        return b - a  # works for floats
    except Exception:
        return None


def _delta_epi(a, b):
    d = _safe_numeric_diff(a, b)
    if d is not None:
        return d
    # Non-numeric or structural EPI; report boolean change
    return f"changed={a != b} (types: {type(a).__name__}->{type(b).__name__})"


def run_demo():
    # Create test graph
    G = create_tnfr_topology('ring', 6, seed=42)
    initialize_tnfr_nodes(G, nu_f=1.0, seed=43)

    print("=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    node_id = list(G.nodes())[0]
    epi_before = G.nodes[node_id].get(EPI_PRIMARY, 'N/A')
    theta_before = G.nodes[node_id].get(THETA_PRIMARY, 'N/A')
    dnfr_before = G.nodes[node_id].get(DNFR_PRIMARY, 'N/A')
    print(f"Node {node_id}:")
    print(f"  EPI: {epi_before}")
    print(f"  theta: {theta_before}")
    print(f"  ΔNFR: {dnfr_before}")

    grad_phi_before = compute_phase_gradient(G)
    phi_s_before = compute_structural_potential(G)
    print(
        f"\nPhase gradient (node {node_id}): "
        f"{grad_phi_before.get(node_id, 0.0)}"
    )
    print(
        f"Structural potential (node {node_id}): "
        f"{phi_s_before.get(node_id, 0.0)}"
    )

    # Generate and run sequence
    sequence = generate_grammar_valid_sequence('balanced', intensity=2.0)
    print(f"\n{'=' * 60}")
    print(f"RUNNING SEQUENCE: {[op.__class__.__name__ for op in sequence]}")
    print(f"{'=' * 60}")

    run_sequence(G, node_id, sequence)

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    epi_after = G.nodes[node_id].get(EPI_PRIMARY, 'N/A')
    theta_after = G.nodes[node_id].get(THETA_PRIMARY, 'N/A')
    dnfr_after = G.nodes[node_id].get(DNFR_PRIMARY, 'N/A')
    print(f"Node {node_id}:")
    print(f"  EPI: {epi_after}")
    print(f"  theta: {theta_after}")
    print(f"  ΔNFR: {dnfr_after}")

    grad_phi_after = compute_phase_gradient(G)
    phi_s_after = compute_structural_potential(G)
    print(
        f"\nPhase gradient (node {node_id}): "
        f"{grad_phi_after.get(node_id, 0.0)}"
    )
    print(
        f"Structural potential (node {node_id}): "
        f"{phi_s_after.get(node_id, 0.0)}"
    )

    print(f"\n{'=' * 60}")
    print("CHANGES")
    print(f"{'=' * 60}")
    print(f"ΔEPI: {_delta_epi(epi_before, epi_after)}")
    print(f"Δtheta: {_safe_numeric_diff(theta_before, theta_after)}")
    delta_grad = (
        grad_phi_after.get(node_id, 0.0) -
        grad_phi_before.get(node_id, 0.0)
    )
    print(f"Δgrad_phi: {delta_grad}")
    print(
        "Δphi_s: "
        f"{phi_s_after.get(node_id, 0.0) - phi_s_before.get(node_id, 0.0)}"
    )


if __name__ == "__main__":
    run_demo()
