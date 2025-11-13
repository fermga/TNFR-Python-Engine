from tnfr.examples_utils import build_radial_atom_graph, apply_synthetic_activation_sequence
from tnfr.physics.fields import compute_structural_potential
from tnfr.operators.grammar import validate_structural_potential_confinement
from tnfr.telemetry.constants import STRUCTURAL_POTENTIAL_DELTA_THRESHOLD


def test_atom_like_graph_build_and_roles():
    G = build_radial_atom_graph(n_shell=12, seed=7)
    roles = {G.nodes[n].get("role", "") for n in G.nodes()}
    assert "nucleus" in roles and "shell" in roles
    # sanity: graph is connected
    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        return
    assert nx.is_connected(G)


def test_u6_sequential_delta_phi_s_safe():
    G = build_radial_atom_graph(n_shell=16, seed=13)
    phi_before = compute_structural_potential(G)
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)
    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_POTENTIAL_DELTA_THRESHOLD, strict=False
    )
    assert ok, f"Expected ΔΦ_s to be confined; got {msg} with drift={drift}"
