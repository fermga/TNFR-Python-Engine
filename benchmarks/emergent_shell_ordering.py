"""Finite shell-group comparisons on explicitly embedded graph geometries.

The constructor supplies Fibonacci-sphere coordinates, a radial path or a
three-dimensional ball, radial weights, graph neighbors and a center point.
These choices are geometric inputs, not generated THOL or a derived nucleus.
The Cartesian-product spectrum adds for the combinatorial Laplacian D-A;
it does not establish the same additive identity for L_rw or the full pressure.
Finite sphere clusters approximate selected angular-mode multiplicities; they
are not exact continuum rotational representations. Shell grouping thresholds,
factor-two mode capacities and comparison sequences are supplied choices.
A post hoc best sequence match cannot identify a physical atom or show that
screening is the only absent ingredient. No autonomous maintenance, radiation,
spin, many-body dynamics or physical units are derived here.
Current scope: theory/EMERGENT_ONTOLOGY.md and theory/PHYSICAL_REGIME_CORRESPONDENCES.md.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np

# Use the in-repo canonical package, not a possibly-stale site-packages copy.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_chemistry import (  # noqa: E402
    fibonacci_sphere_graph,
    structural_eigenmodes,
)
from tnfr.physics.fields import classify_nodal_topology  # noqa: E402

SPDF = {0: "s", 1: "p", 2: "d", 3: "f"}

# Observed reference sequences (for POST-HOC identification only -- never used
# to build anything). Each is a list of closed-shell cumulative counts.
ATOMIC_NOBLE = [2, 10, 18, 36, 54, 86]  # Madelung (n+l) + screening
HARMONIC_OSC = [2, 8, 20, 40, 70, 112]  # 3D isotropic oscillator E ~ 2n_r + l
SPHERICAL_WELL = [2, 8, 18, 20, 34, 40, 58, 68]  # 3D infinite well (Bessel)
NUCLEAR = [2, 8, 20, 28, 50, 82, 126]  # oscillator + spin-orbit


# ---------------------------------------------------------------------------
# STEP 1 -- the rigorous emergent part: angular degeneracies (2l+1)
# ---------------------------------------------------------------------------


def angular_modes(
    n_points: int = 162, k_neighbors: int = 6, max_l: int = 3
) -> dict[int, float]:
    """Angular eigenvalues lambda_ang(l) of one structural sphere.

    Returns {l: eigenvalue} for l = 0..max_l, read from the degenerate
    (2l+1) clusters of the sphere Laplacian (the Laplace-Beltrami spectrum).
    This is the numerically-emergent, NOT postulated, angular structure.
    """
    G = fibonacci_sphere_graph(n_points, k_neighbors)
    shells = structural_eigenmodes(G, max_modes=(max_l + 1) ** 2)
    out: dict[int, float] = {}
    for sh in shells:
        ell = (sh.multiplicity - 1) // 2
        if ell <= max_l and ell not in out:
            out[ell] = sh.eigenvalue
    return out


def angular_multiplicities(
    n_points: int = 162, k_neighbors: int = 6, n_shells: int = 4
) -> list[int]:
    """Count selected finite angular spectral groups; compare with 1,3,5,7."""
    G = fibonacci_sphere_graph(n_points, k_neighbors)
    shells = structural_eigenmodes(G, max_modes=n_shells**2)
    return [sh.multiplicity for sh in shells[:n_shells]]


# ---------------------------------------------------------------------------
# STEP 2 -- the radial structure and the Cartesian-product '+'
# ---------------------------------------------------------------------------


def radial_modes(n_shells: int = 7) -> list[float]:
    """Radial eigenvalues lambda_rad(nu) of the concentric-shell path P_M.

    The radial coupling of M concentric shells is a path graph; its structural
    Laplacian eigenvalues are the radial standing waves nu = 1..M.
    """
    L = nx.laplacian_matrix(nx.path_graph(n_shells)).toarray().astype(float)
    return sorted(float(v) for v in np.linalg.eigvalsh(L))


def verify_cartesian_sum_is_plus(
    n_points: int = 42, k_neighbors: int = 6, n_shells: int = 4
) -> float:
    """Check finite combinatorial-Laplacian product additivity numerically.

    This identity is not the row-normalized nodal generator identity."""
    sphere = fibonacci_sphere_graph(n_points, k_neighbors)
    path = nx.path_graph(n_shells)
    prod = nx.cartesian_product(sphere, path)

    def lap_spec(G: nx.Graph) -> np.ndarray:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        return np.sort(np.linalg.eigvalsh(L))

    s_ang = lap_spec(sphere)
    s_rad = lap_spec(path)
    outer = np.sort((s_ang[:, None] + s_rad[None, :]).ravel())
    direct = lap_spec(prod)
    return float(np.max(np.abs(outer - direct)))


def concentric_shell_graph(
    n_points: int = 80, k_neighbors: int = 6, n_shells: int = 7
) -> nx.Graph:
    """The atom manifold sphere [] path as an explicit graph."""
    return nx.cartesian_product(
        fibonacci_sphere_graph(n_points, k_neighbors),
        nx.path_graph(n_shells),
    )


def solid_ball_graph(
    n_shells: int = 4, base_points: int = 16, k_neighbors: int = 8
) -> nx.Graph:
    """Construct a supplied 3D ball with a center and Fibonacci shells.

    Every node uses the same k-nearest-neighbor rule, but the coordinates, radii
    and central point are explicit geometric inputs."""
    pts: list[np.ndarray] = [np.zeros(3)]  # geometric center
    for s in range(1, n_shells + 1):
        r = float(s)
        npts = base_points * s * s  # ~constant areal density
        idx = np.arange(npts, dtype=float)
        golden = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - 2.0 * (idx + 0.5) / npts
        ring = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        th = golden * idx
        shell = np.stack([r * ring * np.cos(th), r * ring * np.sin(th), r * z], axis=1)
        pts.extend(shell)
    P = np.asarray(pts)
    G = nx.Graph()
    G.add_nodes_from(range(len(P)))
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k_neighbors]:
            G.add_edge(i, int(j))
    return G


def ball_closed_shells(
    G: nx.Graph, *, max_modes: int = 40
) -> tuple[list[int], list[int]]:
    """Group the supplied ball spectrum and accumulate assigned mode capacities."""
    shells = structural_eigenmodes(G, max_modes=max_modes, gap_factor=4.0)
    mults = [sh.multiplicity for sh in shells]
    cum: list[int] = []
    total = 0
    for sh in shells:
        total += 2 * sh.multiplicity
        cum.append(total)
    return mults, cum


# ---------------------------------------------------------------------------
# STEP 3 -- let the shells, filling order, and magic numbers emerge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subshell:
    nu: int  # radial index (1..M)
    ell: int  # angular index (0..3)
    energy: float  # lambda_ang(l) + w_r * lambda_rad(nu)
    capacity: int  # 2*(2l+1) structural modes

    @property
    def label(self) -> str:
        # Atomic spectroscopic convention n = nu + l (so the lowest l=1 radial
        # mode is 2p, not 1p) -- purely a display name, not used in any sort.
        return f"{self.nu + self.ell}{SPDF[self.ell]}"


def emergent_subshells(
    ang: dict[int, float], rad: list[float], w_r: float
) -> list[Subshell]:
    """All (nu, l) subshells ordered by emergent structural excitation."""
    out: list[Subshell] = []
    for ell, lam_a in ang.items():
        for nu, lam_r in enumerate(rad, start=1):
            energy = lam_a + w_r * lam_r
            out.append(Subshell(nu, ell, energy, 2 * (2 * ell + 1)))
    out.sort(key=lambda s: s.energy)
    return out


def magic_numbers(subshells: list[Subshell], k_closures: int = 6) -> list[int]:
    """Cumulative mode counts at the dominant closed shells.

    A closed shell is a large gap in the resonant spectrum. We take the
    ``k_closures`` largest energy gaps as the dominant shell boundaries -- a
    rank-based cut with no magic threshold -- and report the cumulative mode
    capacity at each (the emergent closed-shell counts).
    """
    energies = [s.energy for s in subshells]
    gaps = np.diff(energies)
    if gaps.size == 0:
        return []
    k = min(k_closures, gaps.size)
    cuts = {int(i) for i in np.argsort(gaps)[-k:]}
    magic: list[int] = []
    cumulative = 0
    for i, s in enumerate(subshells):
        cumulative += s.capacity
        if i in cuts:
            magic.append(cumulative)
    return magic


def capacity_order(subshells: list[Subshell]) -> list[int]:
    """The emergent filling order expressed as a capacity sequence."""
    return [s.capacity for s in subshells]


# ---------------------------------------------------------------------------
# Madelung reference (the postulated rule), for contrast only
# ---------------------------------------------------------------------------


def madelung_capacity_order(max_n: int = 7) -> list[int]:
    """Capacity sequence of the postulated aufbau (n+l, n) order."""
    pairs = [(n, ell) for n in range(1, max_n + 1) for ell in range(0, min(n, 4))]
    pairs.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return [2 * (2 * ell + 1) for _n, ell in pairs]


def leading_overlap(seq: list[int], ref: list[int]) -> int:
    """How many leading entries of ref appear, in order, as a prefix of seq."""
    count = 0
    for a, b in zip(seq, ref):
        if a != b:
            break
        count += 1
    return count


def best_resonator_match(magic: list[int]) -> tuple[str, int]:
    """Identify which observed family the emergent magic numbers match best."""
    refs = {
        "atomic noble gases (Madelung n+l)": ATOMIC_NOBLE,
        "3D harmonic oscillator": HARMONIC_OSC,
        "infinite spherical well": SPHERICAL_WELL,
        "nuclear shell model": NUCLEAR,
    }
    best_name, best_n = "(none)", 0
    for name, ref in refs.items():
        n = leading_overlap(magic, ref)
        if n > best_n:
            best_name, best_n = name, n
    return best_name, best_n


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("PRESCRIBED GEOMETRY AND FINITE SHELL-ORDER COMPARISON")
    print("=" * 70)

    # -- M1: angular degeneracies emerge (the rigorous part) -----------------
    mult = angular_multiplicities()
    print("\n[M1] Angular degeneracies from the sphere Laplacian:")
    print(f"     emergent multiplicities (low modes): {mult}")
    print("     expected (2l+1) for l=0,1,2,3       : [1, 3, 5, 7]")
    assert mult == [1, 3, 5, 7], f"angular (2l+1) did not emerge: {mult}"
    print("     -> PASS: (2l+1) angular shells emerge numerically.")

    # -- M2: atom manifold = sphere [] path; spectrum = '+' ------------------
    mismatch = verify_cartesian_sum_is_plus()
    print("\n[M2] Atom manifold = sphere [] radial-path (Cartesian product):")
    print(f"     max | spec(prod) - outer_sum(spec) | = {mismatch:.2e}")
    assert mismatch < 1e-9, "Cartesian-product '+' failed"
    print("     -> PASS: energies are sums lambda_ang(l)+lambda_rad(nu)")
    print("        (combinatorial-Laplacian product additivity).")

    ang = angular_modes()
    rad = radial_modes(n_shells=7)
    a1 = ang[1] - ang[0]  # first angular gap
    r1 = rad[1] - rad[0]  # first radial gap

    # -- M3: emergent order at the balanced manifold (rho = 1) ---------------
    w_balanced = a1 / r1  # first radial gap == first angular gap
    sub = emergent_subshells(ang, rad, w_balanced)
    emergent_caps = capacity_order(sub)
    madelung_caps = madelung_capacity_order()
    div = leading_overlap(emergent_caps, madelung_caps)
    magic = magic_numbers(sub)
    print("\n[M3] Emergent filling order at the balanced manifold (rho=1):")
    print("     first 12 subshells (emergent):", " ".join(s.label for s in sub[:12]))
    print(f"     emergent capacity order : {emergent_caps[:12]}")
    print(f"     Madelung capacity order : {madelung_caps[:12]}")
    print(f"     orders agree for only the first {div} subshell(s)")
    print(f"     dominant closures (rho=1): {magic[:8]} (no standard table)")
    print(f"     atomic noble gases       : {ATOMIC_NOBLE}")
    assert emergent_caps != madelung_caps, "emergent order == Madelung (!?)"
    assert magic[:6] != ATOMIC_NOBLE, "noble gases without screening?!"
    print("     -> PASS: the aufbau (n+l) order does NOT emerge; the atomic")
    print("        noble-gas numbers are NOT reproduced.")

    # -- M4: scan the manifold's radial:angular stiffness rho ----------------
    print("\n[M4] Scan radial:angular stiffness rho (no rho gives Madelung):")
    print("     rho   dominant closures                 best standard table")
    best_noble = 0
    for rho in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        w = rho * a1 / r1
        m = magic_numbers(emergent_subshells(ang, rad, w))
        name, n_match = best_resonator_match(m)
        best_noble = max(best_noble, leading_overlap(m, ATOMIC_NOBLE))
        tag = f"{name} ({n_match})" if n_match else "no standard table (0)"
        print(f"     {rho:<5} {str(m[:7]):<34} {tag}")
    print(f"\n     max leading noble-gas match over all rho: {best_noble}/6")
    assert best_noble < len(ATOMIC_NOBLE), "a rho gave all noble gases"
    print("     -> PASS: no stiffness gives the noble-gas sequence,")
    print("        nor any standard table. The free Laplacian's convex")
    print("        spectra weight angular l too weakly for Madelung (n+l).")

    # -- M5: does a confining NUCLEUS emerge? (canonical classifier) ----------
    topo = {
        "single sphere": classify_nodal_topology(fibonacci_sphere_graph(120, 6)),
        "sphere [] path": classify_nodal_topology(concentric_shell_graph(80, 6, 7)),
        "solid ball": classify_nodal_topology(solid_ball_graph(4, 16, 8)),
        "star (calib)": classify_nodal_topology(nx.star_graph(60)),
    }
    print("\n[M5] Classify the supplied geometry: classify_nodal_topology")
    print("     (radial = one central nucleus; reads c(i) = sum 1/d^2):")
    for name, t in topo.items():
        print(
            f"     {name:16s} -> {t['topology']:11s} "
            f"(conc {t['concentration']:.2f}, centers {len(t['centers'])})"
        )
    assert topo["solid ball"]["topology"] == "radial", "ball not radial"
    assert topo["sphere [] path"]["topology"] != "radial", "shells radial?!"
    assert topo["star (calib)"]["topology"] == "radial", "star not radial"
    print("     -> PASS: the supplied ball has a radial read-out (its geometric")
    print("        center, same wiring rule -- not a postulated hub);")
    print("        sphere and shell-stack have none (annular / multinodal).")

    # -- M6: with the emergent nucleus, do shell CLOSURES emerge? ------------
    mults, ball_cum = ball_closed_shells(solid_ball_graph(4, 16, 8))
    sw = leading_overlap(ball_cum, SPHERICAL_WELL)
    at = leading_overlap(ball_cum, ATOMIC_NOBLE)
    print("\n[M6] Group the supplied ball's low spectrum:")
    print(f"     emergent multiplicities : {mults[:6]}  (= 2l+1)")
    print(f"     closed-shell counts     : {ball_cum[:7]}")
    print(f"     infinite spherical well : {SPHERICAL_WELL[:7]}")
    print(f"     atomic noble gases      : {ATOMIC_NOBLE}")
    print(f"     leading match: spherical-well {sw} vs atomic {at}")
    assert mults[:3] == [1, 3, 5], f"angular shells not (2l+1): {mults[:3]}"
    assert ball_cum[:3] == [2, 8, 18], f"closures not 2,8,18: {ball_cum[:3]}"
    assert sw > at, "ball matches atomic better than spherical well?!"
    print("     -> PASS: CLOSURES emerge (2, 8, 18, ...) = the spherical-well")
    print("        / independent-particle shell model, NOT the atomic table.")

    # -- Verdict --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINITE MODEL-COMPARISON SCOPE")
    print("=" * 70)
    print(
        "FINITE GEOMETRY COMPARISON:\n  Supplied sphere/path/ball graphs determine the measured spectra.\n  Product additivity applies to the combinatorial Laplacian used here.\n  Approximate angular groups and factor-two capacities are assigned shell comparisons.\n  A central read-out maximum on an explicitly embedded ball is not autonomous nucleus formation.\n  Agreement or disagreement with a finite shell sequence does not identify an atom or its forces.\n  Neither physical dimensions nor a unique missing screening law are derived."
    )


if __name__ == "__main__":
    main()
