# TNFR Examples

128 runnable examples organized by theme. Every example derives from the
nodal equation `∂EPI/∂t = νf · ΔNFR(t)`, the 13 canonical operators, grammar
U1–U6, and the structural field tetrad `(Φ_s, |∇φ|, K_φ, ξ_C)`.

Each file keeps a **stable global number** as its identifier (the number does
not change when an example moves); the **folder gives the theme**. Run any
example directly, e.g.:

```bash
python examples/01_foundations/01_hello_world.py
python examples/09_millennium/109_p_vs_np_coherence_synthesis.py
```

`tnfr` resolves from the editable install (`pip install -e .`), so examples run
from any location.

---

## 01_foundations — first contact with TNFR

Basic tutorials: nodes, operators, grammar, coherence, topologies, the SDK.

- `01_hello_world.py` — your first nodal network
- `02_musical_resonance.py` — phase synchronization as resonance
- `03_network_formation.py` — building coupled networks
- `04_operator_sequences.py` — grammar U1–U6 in action
- `05_coherence_evolution.py` — C(t) dynamics
- `06_network_topologies.py` — TNFR across graph structures
- `07_phase_transitions.py` — bifurcation dynamics (U4)
- `08_emergent_phenomena.py` — collective behaviors
- `09_visualization_suite.py` — dynamic plotting
- `10_simplified_sdk_showcase.py` — the Simple SDK

## 02_physics_regimes — regimes and structure of the nodal dynamics

Classical/quantum correspondences, conservation, gauge, variational, tetrad,
and operator–tetrad synergies.

- `11`–`15` — classical limit, mechanics, quantum mechanics, uncertainty, kinematics
- `17` conservation law · `26` gauge structure · `27` variational principle
- `28` dissipative systems · `29` Lyapunov stability · `30` self-optimization
- `31` constants basis · `32` spiral attractors · `33` complex-field unification
- `34` conservation protocol · `35` tetrad irreducibility · `36` grammar violations
- `37` operator–tetrad synergy · `38` grammar-energy landscape · `39` nodal decomposition
- `115` operator-contract fidelity audit (measured, not asserted)

## 03_riemann_zeta — TNFR–Riemann ζ-track (P1–P31)

Discrete prime-path operators, von Mangoldt prime ladder, Weil formula,
Li–Keiper, and the ζ-track attack surface. Program **open** (paused at T-HP).

- `16, 18–25` — operator, convergence, topology, eigenmodes, spectral zeta, RMT
- `41–58` — von Mangoldt → oscillatory correction (P12–P31)

## 04_riemann_L_twisted — TNFR–Riemann χ-twisted L-track (P32–P49)

Dirichlet L-functions and the χ-twisted parity layer (GL(1)).

- `59–63` — Dirichlet L: construction, continuation, Hamiltonian, Weil, Li–Keiper
- `64–76` — twisted positivity → twisted oscillatory correction

## 05_type_hygiene — catalog type-hygiene programme (B0–B11)

REMESH-∞ residue split + the twelve type-signature / closure-discipline demos.

- `77_remesh_infinity_residue_split_demo.py`
- `78–89` — νf / EPI / φ / ΔNFR / REMESH-window / Δφ_max / coupling / tetrad / currents / aggregates / U-rules / catalog signatures

## 06_navier_stokes — TNFR–Navier–Stokes programme (N1–N17)

3D Taylor–Green, Leray/BKM, incompressibility, geometric depletion, Reynolds
sweeps. NS-G5 closed at the discrete level; Clay **open**.

- `77–86` — Taylor–Green → Reynolds sweep

## 07_number_theory — primality and arithmetic as structural equilibrium

Primality ⟺ ΔNFR = 0, Goldbach, prime families/orbits, numbers as a coupled
network, and emergent chemistry/particles from the same criterion.

- `40` arithmetic number theory · `94–97` generative / spectral / Goldbach
- `100–102` prime families, numbers-as-network, nodal flow on numbers
- `116` νf-embedded prime visibility (arithmetic via νf only; diffusion echoes ANY νf carrier — prime ≈ arbitrary set ≈ Ω(n)/log n — substrate blind)
- `emergent_chemistry_particles_demo.py` — chemistry/particles from ΔNFR = 0

## 08_emergent_geometry — the geometry the dynamics generates from itself

Emergent symplectic substrate, structural diffusion/transport, polarization,
Helmholtz–Hodge orthogonality, generating structure, flow prediction.

- `98` symplectic substrate · `99` structural diffusion
- `103–105` substrate↔Riemann, NS-is-not-Riemann, NS enstrophy
- `106` polarization · `107` orthogonal structure · `108` generating structure
- `112` structure predicts the coherence flow · `113` overdamped projection bridge · `114` substrate conserved quantities · `unified_fields_showcase.py`
- `117` emergent geometry on the residue graph (Paley factorization, honest: diffusion spectrum carries the factor cosets, symplectic substrate is blind; unifies factorization-lab ↔ emergent geometry)
- `118` where the emergent operator diverges from the classical Laplacian (residue graphs are regular Cayley → identical; on irregular graphs L_rw IS the Shi–Malik degree-aware Ncut → more balanced cuts)
- `119` the phase sector — directed residue operator (n≡3 mod4 → Paley tournament → complex spectrum; "3 distinct eigenvalues ⟺ odd prime" 58/58, resolves prime powers, phase encodes √n; extends Reading B to all odd primes, still e–π/Fix(G)^⊥ bounded)
- `120` the symmetry wall (vertex-transitivity of the residue Cayley digraph confines arithmetic to the spectrum; double dissociation — spectrum sees the QR arithmetic, per-node symplectic substrate is blind; same Fix(G)^⊥ wall as the paused Riemann program, explained not crossed)
- `121` can a canonical symmetry-break cross the wall? (B2-P2 lever, measured NEGATIVE: the nodal equation has no per-node weight slot; structure-derived νf is uniform on the vertex-transitive graph; arithmetic-injected νf is circular echo (shuffled control identical) — confirms the analytical B0★-β-P2 closure at the NT level)

## 09_millennium — Millennium-problem reformulations (honest scope)

TNFR-native reformulations that **localise the obstruction**, not solutions.

- `109_p_vs_np_coherence_synthesis.py` — synthesis vs verification (Branch B)
- `110_bsd_rank_structural_pressure.py` — rank as structural pressure (Branch B)
- `111_hodge_discrete_and_honest_gap.py` — discrete Hodge, blindness (Branch B3-leaning)

## 10_applications — applied phase-gated coupling and infrastructure

- `90` phase-gate monitor · `91` breast-cancer · `92` wine-quality · `93` structural interface
- `pytorch_cuda_demo.py` — GPU backend

---

**Note on the two `77–86` series**: `05_type_hygiene/` and `06_navier_stokes/`
were two programmes developed in parallel that previously shared the numbers
77–86. The thematic folders resolve that collision; the global numbers are
preserved as stable identifiers.
