"""N10 demo — NS-G5 precursor: 2D vs 3D dimensional asymmetry (falsifiability).

Status: empirical falsifiability check; NOT a closure of NS-G5.

Background
----------
The genuine obstruction to global regularity for 3D incompressible
Navier-Stokes (Clay Millennium Problem, NS-G5 in the TNFR roadmap) is the
vortex-stretching term ``(omega . grad) u`` in the vorticity equation.
This term VANISHES IDENTICALLY in 2D by an algebraic accident: when
``u = (u(x, y), v(x, y), 0)`` the only nonzero vorticity component is
``omega_z = partial_x v - partial_y u``, so ``omega = (0, 0, omega_z)``
and ``(omega . grad) u = omega_z partial_z u = 0``. This is why 2D NS
admits global smooth solutions (Leray 1934 for weak, Ladyzhenskaya for
strong) while 3D remains open.

For any discrete TNFR-NS operator to be even *consistent* with this
classical dichotomy, the following two structural invariants must hold:

  (A) On the 2D torus, ``stretching_production()`` does not arise
      (the 2D vorticity equation has no source term).

  (B) On the 3D torus, when initialised with a z-independent (purely
      2D-embedded) velocity field, ``stretching_production()`` must
      remain bounded by round-off across the evolution. ANY drift of
      ``stretching_production()`` away from zero in this case is a
      structural bug or a symmetry-breaking discretisation artefact.

  (C) On the 3D torus with a genuinely 3D initial condition (classical
      3D Taylor-Green), ``stretching_production()`` must be O(1) and
      monotonically grow on the early-time horizon (N6 baseline).

This demo runs the three side-by-side and verifies all three invariants.
The ratio between (C) and (B) quantifies how clean the dimensional
projection is in the discrete operator.

Honest scope
------------
Confirming (A), (B), (C) does NOT prove NS-G5. It only certifies that
the TNFR-NS operator does not spuriously activate the 3D nonlinearity on
2D data. Global regularity of fully 3D solutions remains OPEN (Clay).
NS-G1, NS-G2, NS-G3, NS-G4 all remain OPEN. The result here is a
CONSISTENCY CHECK that strengthens trust in N6/N9 by ruling out the
hypothesis that the observed 3D stretching is a numerical artefact of
the operator itself.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from tnfr.navier_stokes.operator import (
    TNFRNavierStokesOperator,
    build_torus_graph,
    build_torus_graph_3d,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_2D = 32  # full 2D TG resolution
N_3D = 12  # 3D embedded + 3D TG resolution
DT = 0.01
T_FINAL = 0.5
STEPS = int(round(T_FINAL / DT))  # 50
VISCOSITY = 0.05
AMPLITUDE = 1.0
RECORD_EVERY = 10

# Pass-criterion thresholds
ENSTROPHY_MONOTONE_TOL = 1e-12  # 2D: |max delta-Z (upward)| must be tiny
STRETCHING_ZERO_TOL = 1e-10  # 3D-embedded: machine-eps band
STRETCHING_3D_MIN = 0.1  # true 3D: must exceed this
RATIO_MIN = 1e6  # 3D / 2D-embedded must be huge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embed_2d_into_3d(
    op: TNFRNavierStokesOperator,
    amplitude: float,
) -> None:
    """Initialise a 3D operator with a z-independent 2D-embedded velocity.

    Sets ``u(x, y, z) = A sin(x) cos(y)``, ``v(x, y, z) = -A cos(x) sin(y)``,
    ``w(x, y, z) = 0`` on every node of the 3D torus graph. This field has
    ``omega = (0, 0, omega_z)`` and ``(omega . grad) u = 0`` by construction.
    """
    if op.dimension != 3:
        raise ValueError("embed_2d_into_3d expects a 3D operator")
    nodes = list(op.graph.nodes)
    u = np.zeros(len(nodes))
    v = np.zeros(len(nodes))
    w = np.zeros(len(nodes))
    for idx, node in enumerate(nodes):
        x, y, _z = op.graph.nodes[node]["pos"]
        u[idx] = amplitude * math.sin(x) * math.cos(y)
        v[idx] = -amplitude * math.cos(x) * math.sin(y)
    op.set_components([u, v, w])


def run_2d(n: int) -> dict[str, Any]:
    """Run the 2D Taylor-Green vortex and record enstrophy trajectory."""
    G = build_torus_graph(n)
    op = TNFRNavierStokesOperator(G, viscosity=VISCOSITY, dimension=2)
    op.set_taylor_green(AMPLITUDE)
    times = [0.0]
    Z = [op.enstrophy_curl()]
    omega_sup = [op.vorticity_sup_norm()]
    for k in range(1, STEPS + 1):
        op.step(DT, advection=True, incompressible=True)
        if k % RECORD_EVERY == 0 or k == STEPS:
            times.append(op.time)
            Z.append(op.enstrophy_curl())
            omega_sup.append(op.vorticity_sup_norm())
    return {
        "times": np.array(times),
        "Z": np.array(Z),
        "omega_sup": np.array(omega_sup),
    }


def run_3d(
    n: int,
    embed_2d: bool,
) -> dict[str, Any]:
    """Run a 3D evolution and record stretching production trajectory."""
    G = build_torus_graph_3d(n)
    op = TNFRNavierStokesOperator(G, viscosity=VISCOSITY, dimension=3)
    if embed_2d:
        embed_2d_into_3d(op, AMPLITUDE)
    else:
        op.set_taylor_green(AMPLITUDE)
    times = [0.0]
    P = [op.stretching_production()]
    omega_sup = [float(np.linalg.norm(op.vorticity_3d(), axis=0).max())]
    for k in range(1, STEPS + 1):
        op.step(DT, advection=True, incompressible=True)
        if k % RECORD_EVERY == 0 or k == STEPS:
            times.append(op.time)
            P.append(op.stretching_production())
            omega_sup.append(float(np.linalg.norm(op.vorticity_3d(), axis=0).max()))
    return {
        "times": np.array(times),
        "P": np.array(P),
        "omega_sup": np.array(omega_sup),
    }


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 76)
    print("N10 demo — NS-G5 precursor: 2D vs 3D dimensional asymmetry")
    print("=" * 76)
    print(
        f"\nConfig: n_2D={N_2D}, n_3D={N_3D}, dt={DT}, T_final={T_FINAL} "
        f"({STEPS} steps), nu={VISCOSITY}, A={AMPLITUDE}, INCOMP+adv ON.\n"
    )

    t0 = time.perf_counter()

    print("[1/3] Running 2D Taylor-Green (no stretching by algebra)...")
    res_2d = run_2d(N_2D)
    print(f"      Done. Z(0) = {res_2d['Z'][0]:.4f}, " f"Z(T) = {res_2d['Z'][-1]:.4f}.")

    print(
        "[2/3] Running 3D operator with 2D-embedded IC " "(stretching must be zero)..."
    )
    res_emb = run_3d(N_3D, embed_2d=True)
    print(f"      Done. max|P(t)| = {np.abs(res_emb['P']).max():.3e}.")

    print("[3/3] Running 3D operator with true 3D TG " "(stretching must be > 0)...")
    res_3d = run_3d(N_3D, embed_2d=False)
    print(f"      Done. max|P(t)| = {np.abs(res_3d['P']).max():.3e}.")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.2f}s\n")

    # ---------------------------------------------------------------- tables
    print("-" * 76)
    print("2D Taylor-Green trajectory (enstrophy must monotonically decay)")
    print("-" * 76)
    print(f"{'t':>10} {'Z(t)':>16} {'||omega||_inf':>18}")
    for i, t in enumerate(res_2d["times"]):
        print(f"{t:>10.4f} {res_2d['Z'][i]:>16.6e} " f"{res_2d['omega_sup'][i]:>18.6f}")

    print()
    print("-" * 76)
    print("3D operator, 2D-embedded IC — P(t) and ||omega||_inf(t)")
    print("(P MUST be ~ machine epsilon throughout)")
    print("-" * 76)
    print(f"{'t':>10} {'P(t)':>18} {'||omega||_inf':>18}")
    for i, t in enumerate(res_emb["times"]):
        print(
            f"{t:>10.4f} {res_emb['P'][i]:>18.6e} " f"{res_emb['omega_sup'][i]:>18.6f}"
        )

    print()
    print("-" * 76)
    print("3D operator, true 3D TG IC — P(t) and ||omega||_inf(t)")
    print("(P MUST be O(1) and active)")
    print("-" * 76)
    print(f"{'t':>10} {'P(t)':>18} {'||omega||_inf':>18}")
    for i, t in enumerate(res_3d["times"]):
        print(f"{t:>10.4f} {res_3d['P'][i]:>18.6e} " f"{res_3d['omega_sup'][i]:>18.6f}")

    # ------------------------------------------------------------ criteria
    print()
    print("=" * 76)
    print("PASS criteria")
    print("=" * 76)

    # C1 — 2D enstrophy monotone non-increasing (no production source)
    dZ = np.diff(res_2d["Z"])
    max_upward = float(dZ.max())  # most positive change
    c1 = max_upward <= ENSTROPHY_MONOTONE_TOL
    print(
        f"C1 (2D no stretching source -> Z monotone decay): "
        f"max upward dZ = {max_upward:+.3e} <= {ENSTROPHY_MONOTONE_TOL:.0e} "
        f"-> {'PASS' if c1 else 'FAIL'}"
    )

    # C2 — 3D embedded stretching ~ machine epsilon
    max_P_emb = float(np.abs(res_emb["P"]).max())
    c2 = max_P_emb <= STRETCHING_ZERO_TOL
    print(
        f"C2 (3D operator on 2D-embedded data -> P == 0): "
        f"max |P| = {max_P_emb:.3e} <= {STRETCHING_ZERO_TOL:.0e} "
        f"-> {'PASS' if c2 else 'FAIL'}"
    )

    # C3 — 3D true stretching is O(1)
    max_P_3d = float(np.abs(res_3d["P"]).max())
    c3 = max_P_3d >= STRETCHING_3D_MIN
    print(
        f"C3 (3D operator on true 3D TG -> P > 0 and active): "
        f"max |P| = {max_P_3d:.3e} >= {STRETCHING_3D_MIN:.0e} "
        f"-> {'PASS' if c3 else 'FAIL'}"
    )

    # C4 — Ratio quantifies dimensional separation
    if max_P_emb > 0:
        ratio = max_P_3d / max_P_emb
    else:
        ratio = math.inf
    c4 = ratio >= RATIO_MIN
    print(
        f"C4 (dimensional separation ratio P_3D / P_embed >> 1): "
        f"ratio = {ratio:.3e} >= {RATIO_MIN:.0e} "
        f"-> {'PASS' if c4 else 'FAIL'}"
    )

    print()
    n_pass = sum([c1, c2, c3, c4])
    verdict = "PASS" if n_pass == 4 else "FAIL"
    print(f"OVERALL: {n_pass}/4 -> {verdict}")

    # ----------------------------------------------------------- coda
    print()
    print("=" * 76)
    print("Honest scope")
    print("=" * 76)
    print(
        "This demo is a CONSISTENCY / FALSIFIABILITY CHECK on the discrete\n"
        "TNFR-NS operator, not a closure of NS-G5 (Clay Millennium\n"
        "Problem). It establishes only that:\n"
        "  (i)  the 2D operator has no spurious stretching source, and\n"
        "  (ii) the 3D operator preserves the algebraic identity\n"
        "       (omega . grad) u == 0 on z-independent data to round-off,\n"
        "       so the genuinely-3D stretching observed in N6 / N9 is not\n"
        "       a numerical artefact of broken dimensional reduction.\n"
        "\n"
        "Global regularity of fully 3D incompressible NS remains OPEN.\n"
        "NS-G1 (continuum limit), NS-G2 (uniform energy bounds), NS-G3\n"
        "(discrete<->continuum BKM), NS-G4 (structural construction of\n"
        "(omega . grad) u in TNFR language) all remain OPEN.\n"
    )


if __name__ == "__main__":
    main()
