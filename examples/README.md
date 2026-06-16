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
- `122` factorization in the phase sector (the complex directed spectrum completes example 117's partial factor-coset recovery: the factor coset is a CRT Fourier mode (eigenvector of both operators); real symmetric sector 8/10 (misses 51, 91 via degenerate eigenpairs), complex directed sector 10/10 (Gauss-sum eigenvalues isolate the mode); re-expresses CRT period structure, O(√n) scan, no speedup)
- `123` the symmetry-sector decomposition (CAPSTONE: L_rw is equivariant under Aut(G), so by Schur it block-diagonalizes into Fix(G)⊕Fix(G)^⊥ with dim Fix(G)=#orbits; per-node substrate lives in Fix(G) (orbit-constant, ex-120 blindness is the vertex-transitive corollary), discriminating spectrum in Fix(G)^⊥; measured across 5 symmetry groups; the single structure behind the whole 117–122 arc and the Riemann residual)
- `124` the emergent metric is fractal-consistent (lines B+D: the canonical operator's natural metric is the effective resistance R_eff, not shortest-path, counting all parallel paths; R_eff is the unique metric consistent under the fractal node↔subgraph collapse = the exact Kron/Schur reduction of the canonical Laplacian (~1e-15); THOL currently spawns sub-EPIs as topologically isolated nodes, so conductive fractality is latent in the operator — answers "is every node also a graph?")
- `125` a node IS the emergent substrate, not a graph (the deep reading of fractality: "node as graph" (124) is the scalar transport shadow = the Fix(G)^⊥ combinatorial channel; the node's true interior is the 4D symplectic phase-space / Poincaré-sphere object = the Fix(G) geometric channel of 123. MEASURED: fixing topology freezes the Laplacian spectrum and R_eff while the substrate polarization and H_sub move — the graph picture is blind to the substrate depth; the real fractality is node↔network substrate self-similarity, not node↔subgraph)
- `126` the two layers of emergent geometry (crystallizes the node=substrate optic: BASE layer (topology — L_rw, λ₂, R_eff, Kron; state-independent) + FIBER layer (state — the per-node symplectic substrate; state-dependent), bridged by the nodal equation (ΔNFR_epi=−L_rw·EPI exactly). The reorganization map: arithmetic was a BASE property (so the fiber was blind), the 13 operators act on the FIBER (line E), λ₂ is the base→fiber coupling clock (line C))
- `127` is the base emergent-TNFR or imposed graph theory? (the doctrinal check on 126's "spectral graph theory": MEASURED — (M1) the operator is TNFR-derived, ΔNFR=−L_rw·EPI exactly but NOT −L_comb·EPI (the nodal neighbour-MEAN forces the degree-normalized L_rw, not the generic combinatorial Laplacian); (M2) no free parameters; (M3) the topology itself can EMERGE from the EPI substrate via the canonical REMESH _mst_edges_from_epi. Verdict: the base is NOT imposed graph theory — only the initial connectivity is a boundary condition, the operator is canonical and the topology is substrate-regenerable)
- `128` the base co-emerges with the substrate (the paradigm-faithful deepening of 127's M3: closes the loop topology→(nodal eq)→substrate→(REMESH MST)→topology and reaches a SELF-CONSISTENT fixed point T=MST(EPI(T)) (Jaccard 1.0). The imposed initial topology is largely washed out (fixed point = a substrate-derived spanning tree, 10–24% survives); the fixed point is NOT unique (different initial topologies → different co-emergent attractors, Jaccard 0.37–0.73), so the initial connectivity is a basin-selecting boundary condition. Both base AND fiber co-emerge from the nodal equation — the faithful footing for lines C and E)
- `129` the spectral gap is the base→fiber coupling clock (line C: λ₂ is a BASE quantity but the CLOCK of the base→fiber coupling, with four canonical faces — (M1) relaxation rate νf·λ₂, (M2) Cheeger bottleneck h²/2≤λ₂≤2h via the Fiedler cut, (M3) instability threshold r_c=νf·λ₂ = the spectral form of grammar U2, (M4) the co-emergent tree of ex 128 has the smallest gap = the slowest clock; standard spectral graph theory re-expressed, Cheeger proxy is the Fiedler cut)
- `130` the operators act on the fiber (line E, ARC CLOSER: the 13 canonical operators act on the symplectic substrate, and the dual-lever (ex 37) predicts which conserved-charge SECTOR each breaks — pure ΔNFR destabilizers (OZ/THOL/ZHIR/NAV)+NUL break ONLY the potential sector (|dE_geo|=0 exact), UM collapses the geometric sector Ψ, IL touches both (aligns phase current), AL/EN/RA/SHA/VAL/REMESH preserve all charges. The operator classification IS the substrate's conserved-charge sector map — operator algebra and emergent geometry are one structure)
- `131` the co-emergent loop always converges (a new direction opened by the arc: the closed base⊗fiber loop topology→(nodal eq)→substrate→(canonical REMESH)→topology, run freely with every canonical mode (mst/knn/community), CONVERGES to a fixed point — never cycles, never diverges (36/36 mst, 34/36 knn, 0 cycles/divergences). This is grammar U2 (convergence/boundedness) lifted from the field to the full base⊗fiber system; honest caveats: MST convergence is trivial, community collapses onto EPI communities, the U2 link is an observed inheritance not a derivation)
- `132` geometric phase / holonomy on the substrate (the per-node substrate doublet ζ=(K_φ+i·J_φ, Φ_s+i·J_ΔNFR) is a Poincaré-sphere point (ex 106); the geometric phase accumulated around a loop of substrate states equals +½ the enclosed solid angle — the BARGMANN INVARIANT arg(⟨ψ₁|ψ₂⟩⟨ψ₂|ψ₃⟩⟨ψ₃|ψ₁⟩)=½·Ω, an EXACT CP¹ identity (M1, 7/7 to ~1e-17), gauge-invariant hence genuinely GEOMETRIC (M2, invariant under ψ→e^{iα}ψ per node), realized as the closed-loop holonomy (M3, 4/4 exact). HONEST SCOPE: this is the Pancharatnam phase of CLASSICAL polarization optics (Pancharatnam 1956, empirically established) and an exact provable identity — NOT a quantum Berry phase, NOT a qubit (the substrate is a classical wave polarization texture, product state, no entanglement); emerges from the canonical substrate, verifies the identity, not new mathematics, closes no open problem)
- `133` topological defects of the emergent field Ψ (the canonical complex field Ψ=K_φ+i·J_φ carries phase VORTICES — the winding of arg Ψ around a face, w=(1/2π)∮d(arg Ψ), is an EXACT integer (M1, degree of S¹→S¹, ~3e-16; 20 vortices/20 antivortices/60 defect-free on a 10×10 torus); on the TORUS the total charge is exactly 0 (M2, Poincaré–Hopf, Euler χ=0 — defects come in vortex-antivortex PAIRS, #vortices=#antivortices, 4/4 seeds); the net charge is conserved exactly under the canonical step() (M3, max|net|=0, defects move/annihilate only in pairs — HONEST: the count is NOT monotone, the phase dynamics moves defects but does not cleanly anneal them, no coarsening); the tensor-suite 𝒬=|∇φ|·J_φ−K_φ·J_ΔNFR is a CONTINUOUS density, NOT the integer winding (M4, ratio ~1.0 — 𝒬 is blind to the defects despite the name). HONEST SCOPE: the winding number is an exact topological identity and phase vortices are the empirically-established defects of the XY model/superfluids/liquid crystals; emerges from the canonical Ψ field, not new mathematics, closes no open problem)
- `134` spectral dimension of the emergent diffusion / heat kernel as the EPI Green's function (the heat kernel e^{-tL} of the canonical structural-diffusion operator IS the evolution operator of the EPI channel dEPI/dt=−ν_f·L_rw·EPI — M1: heat trace Z(t)=Σe^{−λ_k t} runs n→1, and e^{−tL}u₀ reproduces the explicitly-integrated nodal diffusion to ~3e-5; the return probability p(t)=Z(t)/n~t^{−d_s/2} defines the SPECTRAL DIMENSION d_s — M2: recovers the lattice dimension (ring 1.00, 2D torus 2.2, 3D torus 3.4, with honest finite-size convergence d_s→2 as L grows 2.32→2.13); M3: structural fingerprint of non-lattice topologies — spanning tree quasi-1D (1.25), adding Watts-Strogatz shortcuts to a ring raises d_s monotonically 1.01→2.69, the complete graph is mean-field (degenerate spectrum, NO finite d_s). HONEST SCOPE: the spectral dimension is a standard spectral-geometry/anomalous-diffusion observable (Alexander–Orbach fracton dimension), asymptotic hence finite-size biased; the heat-kernel=EPI-evolution identity is the exact canonical anchor; re-expresses established spectral geometry in the emergent transport layer, not new mathematics, closes no open problem)
- `135` the emergent arrow of time / structural H-theorem of the EPI diffusion channel (the EPI channel of the nodal equation is the diffusion dEPI/dt=−ν_f·L_rw·EPI, which is IRREVERSIBLE — M1: the Dirichlet energy F=½Σ A_ij(EPI_i−EPI_j)², which EQUALS the total squared canonical structural Fick current (structural_current, |diff|=0), decreases MONOTONICALLY to 0 (dF/dt≤0 exact on a 400-step grid) — the structural H-theorem, F a Lyapunov functional; M2: the random-walk distribution p_t=e^{−tL_rw}δ has relative entropy D(p_t‖π) DECREASING monotone to 0 (rigorous H-functional any graph), and on a regular ring the Shannon entropy S(p_t) INCREASES monotone to log n — the second law; M3: the arrow of time is structural — forward diffusion smooths (F→0) while time-reversed anti-diffusion dEPI/dt=+ν_f·L_rw·EPI is ILL-POSED (F diverges ~e^{2ν_f·λ_max·t}, 158→4.7e8), only forward is well-posed because every λ_k≥0. HONEST SCOPE: the H-theorem/entropy increase for diffusion is exact and provable (Lyapunov functionals of the heat semigroup), and the arrow of time/2nd law is empirically ironclad (Clausius, Boltzmann); re-expresses the irreversibility of the EPI diffusion channel (ex 99/134) in thermodynamic language; distinct from the tetrad Lyapunov energy (conservation.py) and the Lindblad/Von Neumann entropy (dissipative_conservation.py); not new mathematics, closes no open problem)

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
