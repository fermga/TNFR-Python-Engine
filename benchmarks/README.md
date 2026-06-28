# Microbenchmarks

> DEPRECATION NOTICE (Docs): Benchmark documentation is developer-focused and not part of the centralized user documentation. For canonical docs, start with `docs/README.md` and `AGENTS.md`.

This directory hosts **targeted microbenchmarks** for hot paths in the TNFR
engine. Each script isolates a specific optimisation and contrasts it with a
reference implementation. Use them to validate performance regressions after
refactoring core operators.

## Usage

```bash
PYTHONPATH=src python benchmarks/<script>.py
```

* Set `PYTHONPATH=src` so the interpreter can import the in-repo `tnfr`
  package without installing it.
* Scripts emit warnings for optional dependencies (NumPy, YAML, orjson). Those
  warnings are harmless during ad-hoc benchmarks.

### Reproducibility

For deterministic benchmark execution with checksum verification:

```bash
# Run all reproducible benchmarks with default seed (42)
make reproduce

# Run specific benchmarks with custom seed
python scripts/run_reproducible_benchmarks.py \
  --benchmarks comprehensive_cache_profiler full_pipeline_profile \
  --seed 123 \
  --output-dir artifacts

# Verify checksums against manifest
make reproduce-verify
# Or manually:
python scripts/run_reproducible_benchmarks.py --verify artifacts/manifest.json
```

The reproducibility script ensures:
- Global seeds are set consistently across all benchmarks
- Output artifacts are generated with SHA256 checksums
- A manifest file tracks all benchmark runs for verification
- CI can detect when benchmark outputs change unexpectedly

## Maintained benchmarks

| Script | Focus | Notes |
| --- | --- | --- |
| `cached_abs_max.py` | Cache-aware updates for absolute maxima (`tnfr.alias.set_attr_with_max`). | Demonstrates how cached maxima avoid scanning the graph via `multi_recompute_abs_max` on every assignment. |
| `collect_attr.py` | Vectorised collection of nodal attributes (`tnfr.alias.collect_attr`). | Requires NumPy; the script exits gracefully when the module is unavailable. |
| `contractive_vs_unitary.py` | Unitary vs. Lindblad ΔNFR evolution (`tnfr.mathematics.MathematicalDynamicsEngine` vs. `ContractiveDynamicsEngine`). | Compares wall-clock timings and Frobenius contractivity after repeated semigroup steps. |
| `evolution_backend_speedup.py` | Backend comparison for evolution engines (`MathematicalDynamicsEngine`, `ContractiveDynamicsEngine`). | Measures per-backend timings, speed-up ratios, and persists JSON artefacts for reproducibility. |
| `default_compute_delta_nfr.py` | Core ΔNFR update speed (`tnfr.dynamics.default_compute_delta_nfr`). | Runs multiple passes on random graphs and reports best/median/mean/worst timings. Accepts `--profile` to dump per-function timings. |
| `compute_dnfr_benchmark.py` | `_compute_dnfr` vectorised vs. fallback execution. | Explores how graph size/density impacts the NumPy and pure-Python paths, reporting summary stats and speed-up ratios. |
| `compute_si_profile.py` | Sense Index profiling (`tnfr.metrics.sense_index.compute_Si`). | Captures cProfile stats for NumPy and pure-Python runs, exporting `.pstats` or JSON summaries. |
| `full_pipeline_profile.py` | Full telemetry + ΔNFR pipeline profiling (`compute_Si`, `_prepare_dnfr_data`, `_compute_dnfr_common`, `default_compute_delta_nfr`). | Produces paired `.pstats` and JSON reports for vectorised and fallback runs, supports multi-configuration chunk/worker sweeps, and records per-operator wall-clock summaries. |
| `neighbor_phase_mean.py` | Fast phase averaging for neighbourhoods (`tnfr.metrics.trig.neighbor_phase_mean`). | Includes a `NodeNX`-based reference to highlight the benefit of the shared `trig_cache` module. |
| `prepare_dnfr_data.py` | ΔNFR data preparation reuse (`tnfr.dynamics._prepare_dnfr_data`). | Exercises cache reuse when assembling phase/EPI/νf arrays. |
| `neighbor_accumulation_comparison.py` | Broadcast neighbour accumulation (`tnfr.dynamics.dnfr._accumulate_neighbors_numpy`). | Benchmarks the single `np.add.at` accumulator against the legacy stack kernel; on 320 random nodes (p=0.65) with Python 3.11/NumPy 2.3.4 it delivered ~1.9× lower median runtime (0.097 s vs 0.185 s). |
| `riemann_program.py` | TNFR–Riemann σ-critical regression. | Scans `H_TNFR` over a σ grid, estimates σ_c^{(k)}, and exports telemetry via `tnfr.riemann.telemetry` to populate `results/riemann_program/`; executed automatically by `make test` (target `riemann-benchmark`). |

### Evolution backend speed-ups

Use the evolution benchmark to compare how the mathematics backends handle the unitary and contractive engines. The script honours the new CLI selector (`--backends`) and the `TNFR_MATH_BACKEND` environment variable, automatically skipping adapters when the corresponding dependencies (JAX, PyTorch) are missing. Results are reproducible thanks to the explicit RNG seed and optional JSON export:

```bash
PYTHONPATH=src python benchmarks/evolution_backend_speedup.py \
  --sizes 2 4 8 --steps 16 --repeats 3 --dt 0.05 \
  --output results/evolution_backends.json
```

Sample output on Python 3.11 with NumPy 2.3.4 (JAX/PyTorch unavailable) illustrates the reported tables:

Unitary mean time (milliseconds per run)

| dim | numpy |
| --- | --- |
| 2 | 1.808 ms |
| 4 | 0.872 ms |

Contractive mean time (milliseconds per run)

| dim | numpy |
| --- | --- |
| 2 | 1.774 ms |
| 4 | 3.437 ms |

Unitary speed-up vs. NumPy baseline

| dim | numpy |
| --- | --- |
| 2 | 1.000 x |
| 4 | 1.000 x |

Contractive speed-up vs. NumPy baseline

| dim | numpy |
| --- | --- |
| 2 | 1.000 x |
| 4 | 1.000 x |

When additional backends are available, their columns appear automatically in all four tables (unitary timing, contractive timing, and their respective speed-up ratios).

### Broadcast accumulator regression check

The performance test suite now covers the vectorised accumulator that relies on
`numpy.bincount` to collapse neighbour contributions. Run the slow-marked test
to validate both speed and numerical parity against the pure-Python path:

```bash
PYTHONPATH=src pytest tests/performance/test_dynamics_performance.py \
  -k broadcast_neighbor_accumulator_stays_faster_and_correct -m slow
```

The command prints per-test timings; the assertion requires the NumPy path to
complete at least 10 % faster while matching the fallback values within
`1e-9` relative/absolute tolerance. Use it to capture before/after measurements
when iterating on `_accumulate_neighbors_broadcasted` or related kernels.

## Structural-emergence research benchmarks

These scripts are **not** performance microbenchmarks. They are falsifiable
research harnesses that probe a single question of the TNFR programme: *which
numbers and arithmetic operations **emerge** from nodal/structural dynamics
(`∂EPI/∂t = νf · ΔNFR(t)`, with `L = D − A` read as the discrete ΔNFR /
phase-curvature operator) rather than being injected by hand?* Each harness pins
its claim to an independent, classical ground-truth theorem (graph-product
spectra, the representation theory of `Aut(G)`, the field-of-fractions theorem,
Schur's lemma) so the TNFR reading can be checked against mathematics that does
does not depend on TNFR. The last meta-harnesses flip the question around: they ask
whether the *obstruction* that the others run into is a single generic fact
(`equivariance_wall.py`), whether the two ways out of it are one missing piece or
two (`missing_piece_bridge.py`), and whether that one recipe extends from two
Millennium programmes to a third — 3D Navier–Stokes
(`navier_stokes_recipe_bridge.py`). A further harness returns to the
emergence-of-numbers thread and completes the odd primes: it adds the
`≡ 3 (mod 4)` Paley-*tournament* complement (`directed_paley_bridge.py`) to
Camino 9's `≡ 1 (mod 4)` Paley-*graph* gap, and shows the mod-4 prime split is
exactly the real-vs-phase boundary of the wall. The latest harness supplies the
*dynamical* half of that thread: where Camino 9/14 read the primes off the
**static spectrum** of a fixed graph, `kuramoto_farey_bridge.py` reads the
**rationals** (and φ) off the **time evolution** of the nodal phase — the sine
circle map as the single-node Kuramoto reduction of `∂EPI/∂t = νf · ΔNFR(t)` —
where mode-locked rotation numbers `ρ = p/q` are the rationals-OUT, the
Farey/Stern-Brocot tree organises the Arnold tongues, φ emerges as the
most-irrational (last-to-lock) Fibonacci limit, and the lock/no-lock split
mirrors the same wall. The final harness ties that dynamical thread back to
the ONE canonical proven object of the frozen Riemann programme: where
`kuramoto_farey_bridge.py` only noted as a *soft analogy* that φ plays the
residue role, `golden_residue_remesh_bridge.py` feeds its orbits through the
N15 REMESH-∞ orthogonal projector `R∞` (`tnfr.riemann.split_residue_by_remesh_infinity`)
and checks where they land — the golden orbit in `ker(R∞)` (the dynamical twin
of P50's prime-ladder `S_TNFR ∈ ker`), REMESH-commensurate lockings in
`range(R∞)` — and exposes the analogy's honest limit (the projector lattice is
coarser than the Farey set, so a locked `ρ = 1/3` lands in the kernel too). The
nineteenth and latest harness closes the *methodological* gap left by P31:
where the canonical `oscillatory_correction.py` read the oscillatory observable
`S_TNFR` off the classical Riemann–Siegel template and *plugged in* the
prime-ladder spectrum, `nodal_propagator_residue_bridge.py` instead
**generates** it from the canonical structural propagator `e^{−s H_P14}`
itself — the nodal time-evolution operator — via the weighted spectral trace
`Z(s) = Tr(W e^{−s H_freq}) = Σ log(p) e^{−s k log p}`, and diagnoses it with
the two strongest post-N15 results jointly: the propagator oscillation
`Im Z(½+iT)` lands in `ker(R∞)` (T2), and its kernel residue is `S_n`-degenerate
to machine precision (T3, CCET made dynamical — shuffling the prime labels
leaves the residue invariant), so the observable sits in `ker(R∞) ∩ Fix(S_n)`
while the true `S(T)` needs the `Fix(S_n)^⊥` half; the global amplitude scale
matches `S(T)` (T4), confirming the obstruction is the missing phase/correlation
structure, not magnitude (the structural reason P31's local correction needed a
damping `d ∼ 3–5`). The twentieth harness turns from probing the wall
*sideways* to studying **coherence itself** head-on, and folds in TNFR's own
coherence metric. Where every prior harness asked *can the catalog reach the
residue*, `coherence_projector_sense_index.py` asks *what is the residue,
exactly* — and answers with the one proven N15 object: the REMESH-∞ operator
`R∞` is a bounded self-adjoint **orthogonal projection** (T1: `‖P²−P‖≈8e-18`,
`‖P−Pᴴ‖≈2e-18`, `rank = trace(P) = L = 8`, Parseval exact, `⟨range,ker⟩≈0`), so
the "closed room" is *literally* the residue of coherence: `ker(R∞) =
range(I−P)`, the orthogonal complement of the coherent/resonant subspace. That
room is **vast** (T2: `dim(range) = L = 8` constant while `dim(ker) = N−8 → ∞`,
coherent fraction `L/N → 0`) — the coherent subspace is finite-dimensional, the
residue fills everything else. The harness then locates TNFR's **Sense Index**
`Si` relative to the room: `Si` is a node-level coherence-capacity functional
(T3: mean `Si` falls monotonically `0.70 → 0.40` as phase dispersion rises,
peaking at full synchrony), and it is `S_n`-**degenerate** on the complete prime
graph (T4: sorted `Si` invariant under prime relabelling to `≈2e-16`), so — like
`C(t)` and the weighted spectral trace — `Si ∈ Fix(S_n)` and is blind to the
room's antisymmetric floor `ker(R∞) ∩ Fix(S_n)^⊥` where `S(T)` lives. Unifies
`Si` with the other symmetric diagnostics and characterises the closed room
directly; closes nothing.

All twenty are dependency-light (NetworkX + NumPy), deterministic, and lint-clean.
`phase_wall.py`, `paley_bridge.py`, `boundary_vibration.py`,
`directed_paley_bridge.py` and `nodal_propagator_residue_bridge.py` additionally
use `mpmath` (only to draw the Riemann target `ζ(½+iT)` / `γₙ` / `S(T)`, never
to derive it).
Following the adelic discipline of `src/tnfr/dynamics/adelic.py`, every harness that
touches the Riemann programme cross-checks the canonical `tnfr` engine when present
(with NumPy-only fallbacks): `equivariance_wall.py`, `commutant_bridge.py`,
`missing_piece_bridge.py` and `navier_stokes_recipe_bridge.py` ground their
`S_n`-breaking per-node diagonal in the canonical adelic carrier `νf = log p`;
`chiral_involution.py` uses `tnfr.physics.emergent_particles`;
`commutant_bridge.py`, `missing_piece_bridge.py` and
`navier_stokes_recipe_bridge.py` also use `tnfr.yang_mills`;
`navier_stokes_recipe_bridge.py` additionally cross-checks the canonical 3D engine
`tnfr.navier_stokes.operator` (the vortex-stretching field); `phase_wall.py`,
`paley_bridge.py`, `boundary_vibration.py`, `primes_as_consequence.py` and
`directed_paley_bridge.py` use `tnfr.dynamics.adelic` (the last reuses
`paley_bridge.py`'s `is_prime`/`paley_gap`/`quadratic_residues`/`riemann_s_phase`
and cross-checks the `≡ 3 (mod 4)` prime support against the canonical carrier),
and `primes_as_consequence.py` additionally cross-checks
`tnfr_primality.core` (the canonical `ΔNFR(n)` pressure) and
`tnfr.riemann.paley_gap_coercivity` (the canonical P25 scope), and
`kuramoto_farey_bridge.py` cross-checks the canonical order parameter
`tnfr.gamma.kuramoto_R_psi` (the Adler 2-oscillator lock) and the emergent
golden-ratio limit `(1+√5)/2` (the last-to-lock rotation number), and
`golden_residue_remesh_bridge.py` projects its circle-map orbits with the
canonical N15 projector `tnfr.riemann.split_residue_by_remesh_infinity` and
reconciles against the P50 prime-ladder certificate
`tnfr.riemann.compute_residue_split_certificate` (both with a NumPy
DFT-bin-mask fallback); and `nodal_propagator_residue_bridge.py` generates its
oscillatory observable from the canonical structural propagator
`tnfr.riemann.weighted_spectral_trace` over the canonical prime-ladder
Hamiltonian and spectrum (`tnfr.riemann.build_prime_ladder_hamiltonian`,
`tnfr.riemann.build_prime_ladder_spectrum`), projects it with the same N15
projector, and reconciles against the P50 certificate
`tnfr.riemann.compute_residue_split_certificate`; and
`coherence_projector_sense_index.py` builds the `R∞` projector matrix from the
canonical N15 resonant-bin mask `tnfr.riemann.build_resonant_bin_mask`, computes
the canonical Sense Index `tnfr.metrics.sense_index.compute_Si`, and draws its
primes from the canonical `tnfr.riemann.build_prime_ladder_spectrum`. Run any of
them directly:

```bash
PYTHONPATH=src python benchmarks/<script>.py
```

| Script | Question | Engine (independent ground truth) | Verdict |
| --- | --- | --- | --- |
| `emergent_integers_symmetry.py` | Do the integers appear as structural invariants? | Laplacian eigenspace multiplicities = irrep dimensions of `Aut(G)` (Platonic solids). | Cardinals emerge: `3` first at tetrahedral, `5` requires icosahedral symmetry. |
| `inverse_spectrum_to_symmetry.py` | Can a partial spectrum predict an unmeasured integer? | Character inner product `⟨χ,χ⟩` separates irreducible from reducible eigenspaces. | Out-of-sample prediction holds; `5` is irreducible in `I`, reducible (`2+3`) in `O`. |
| `composition_arithmetic.py` | Do `+` and `×` emerge from coupling systems? | Cartesian product `G □ H` → `{λ_i + μ_j}`; tensor product `G × H` → `{α_i · β_j}`. | `+`,`×` emerge on spectra; **prime ⇔ irreducible is refuted** (`4` can be irreducible). |
| `operational_irreducibility.py` | Is "factorises" a property of the integer or of the system? | Schur's lemma + the Wigner–von Neumann non-crossing rule. | Logically independent: prime `5` can split `3+2`; composite `4` can be rigid. |
| `bridge_primes_riemann.py` | Does the prime structure of ℤ link to the TNFR-Riemann programme? | Prime-relabelling symmetry `S_n ⊆ Aut(G)` of the canonical prime-ladder graph. | The link is real and runs through `S_n`; prime content is consumed as diagonal input, never generated. |
| `emergent_rationals.py` | Does ℚ (division) emerge as the field of fractions? | Bipartite chiral symmetry (`spec(A) = −spec(A)`), `K_{a,b}` integral Laplacian, `Frac(ℤ) = ℚ`. | ℚ emerges: `−n` from bipartite coupling, `÷` from eigenvalue ratios + phase-locking; field-closed. |
| `equivariance_wall.py` | Is the “wall” that blocks Riemann, Navier–Stokes and Yang–Mills a single generic fact? | Schur's lemma + the Reynolds projector `Π = (1/\|G\|)Σ P_g` onto `Fix(G)`; every `f(A,L)` is `G`-equivariant. | One mechanism, three groups (`S_n`/`K_n`, `D_n`/`C_n`, `ℤ₂`/`P_n`): the catalog cannot reach `Fix(G)^⊥`; only a non-derivable per-node diagonal breaks it — and that diagonal is exactly the canonical adelic carrier `νf = log p` (distinct per node), which the engine reads as *imposed* input (P2 = `NodeIndexedCouplingWeights`, `B0*-beta` not nodal-derivable). |
| `chiral_involution.py` | Is the additive inverse of ℤ the same thing as the antiparticle? | Bipartite chiral symmetry `Γ A Γ = −A` (`spec(A) = −spec(A)`) + winding `W` odd under `φ → −φ`. | One chiral `ℤ₂`: `Γ` (anticommuting) builds `−n`, the same conjugation `C:φ→−φ` flips `sign(W)` (matter↔antimatter); `n+(−n)=0` is the `|W|=0` vacuum. Distinct from the *commuting* Camino-5 wall. |
| `commutant_bridge.py` | Is the Yang–Mills `U(1)→`non-Abelian gap the SAME obstruction as the Riemann `S_n`-breaking gap? | Schur / double-commutant: `{I_V⊗U}'` `=` `End(V)⊗ℂI_d`; `su(2)` `[σ_a/2,σ_b/2]=iεσ_c/2` traceless; `ℂ^{d×d}=ℂI_d⊕su(d)` orthogonal. | **One shape, two groups** (honest **OPEN**): the catalog is confined to a commutant in both — `Fix(S_n)^⊥` (RH) and the traceless `su(d)` curvature (YM) are the unreachable complements; escape needs the non-derivable P2 = the imposed adelic carrier `νf=log p` (RH) / non-commuting generators (YM, Y3 `OPEN_DERIVABILITY_GAP`). Unifies, does not close. |
| `phase_wall.py` | Is the residue unreachable because the catalog is REAL/self-adjoint while `S(T)` is a continuous PHASE? | Spectral theorem (`A=Aᵀ`, `L=D−A` self-adjoint ⇒ real spectrum ⇒ `arg∈{0,π}`); Euler/Pontryagin `z↦exp(iz)` onto `S¹`; `arg ζ(½+iT)` continuous. | **The e–π wall** (honest **OPEN**): `f(A,L)` eigen-phases lock to `{0,π}`; `S(T)` is continuous on the e–π circle; the canonical adelic carrier `U=diag(exp(i·t·νf))`, `νf=log p` reaches the circle but its content is *imposed* (`FORWARD_INDEPENDENT_OF_BACKWARD`) — the e–π mirror of the YM `U(1)` gap. Locates, does not close. |
| `paley_bridge.py` | Do the zeros come from the Paley gap? (Does grounding `νf`'s prime support spectrally breach the Camino-8 wall?) | Residue-circulant `λ₂` via FFT; Gauss sum `\|Σ exp(2πix²/n)\|=√n` for prime `n`; Paley graph Laplacian spectrum `{0,(n−√n)/2,(n+√n)/2}` so `g(n)=\|λ₂−(n−√n)/2\|=0 ⇔ n` prime `≡1 (mod 4)` (Zenodo 10.5281/zenodo.17665853). | **PARTIAL concession, honest OPEN**: the prime support of `νf=log p` is **not** sieved — it emerges from `g(n)=0`, a *self-adjoint* spectral identity (primality as `ΔNFR=0`). But that mechanism is REAL/self-adjoint, so it lives in the Camino-8 scale sector: Paley zeros (integers `≡1 mod 4`) are disjoint from the Riemann ordinates `γₙ`, and no real `g(n)` produces the continuous phase `S(T)`. Grounds the support, not the phase; sharpens the wall, does not breach it. |
| `boundary_vibration.py` | Where do the zeros come from, and why must the engine use `mpmath` to place them? | von Mangoldt `−ζ'/ζ(s)=Σ(log p)p^{−ks}` (abscissa of convergence `Re=1`); `L=D−A` self-adjoint ⇒ real spectrum; a self-adjoint operator commuting with an involution `R=Rᵀ`, `R²=I` splits into real `ℤ₂`-parity sectors; Hilbert–Pólya framing. | **Source canonical, location OPEN**: TNFR derives the vibration's *source* — `νf=log p`, the self-adjoint `{k log p}` (P14, no `mpmath`), the `νf=log p` geometric-trace carrier (adelic), and *self-adjoint + reflection ⇒ real spectrum on the fixed axis* (the HP intuition, TRUE as algebra). But the TNFR-native carrier `Z_vM(s)=Σ w e^{−sμ}` stabilises only for `Re(s)>1` (drift `0.01` at `s=2`, matching classical `−ζ'/ζ`; `13.3` at `s=½`), so it cannot be evaluated where the zeros live; `{γₙ}` enter only as Ground-Truth target (`W₁(P14,T_HP)=115`, growth ratio `26`). `mpmath` draws the target, never derives it. Locates `G4` at the continuation across `Re=1`; does not close it. |
| `primes_as_consequence.py` | Are the primes a TNFR *consequence* (`ΔNFR=0` equilibria) or a primitive input? | Canonical theorem *n prime ⟺ ΔNFR(n)=0* (`ζ=φγ, η=(γ/φ)π, θ=1/φ`; theory §4); Paley Gauss-sum identity `g(n)=0` at `n≡1(mod4)`; Schur (irreducibility of `Aut(K₅)` modes). | **Optic-shift real, full emergence OPEN**: **Reading A** `ΔNFR(n)=0` reproduces primes `≤200` exactly but *consumes the factorization* (`3026` trial divisions ⇒ circular as a derivation); **Reading B** `g(n)=0` is genuine non-circular emergence (squares `x·x mod n`, never `n%k`) but partial (`≡1 (mod4)` only) and self-adjoint (real spectrum ⇒ scale sector); frontier: irreducibility ≠ primality (`K₅` dim-4 mode irreducible yet `4=2×2`). Residual = `2`, `≡3 (mod4)` primes, and the phase `S(T)` — the same e–π wall. Locates `G4`; does not close it. |
| `missing_piece_bridge.py` | Are the two B2 escapes — RH's `S_n`-breaking diagonal and YM's non-commuting generators — ONE missing canonical piece, or two? | `gl(n)=h⊕n` Cartan/root split; commutator of two real symmetric matrices is real anti-symmetric ⇒ `[A,D]∈so(n)` traceless; tensor factorisation `(D⊗I)(I⊗T)=(I⊗T)(D⊗I)`; `tnfr.yang_mills` audit. | **Strong reading REFUTED, weaker unification SURVIVES (honest OPEN)**: no single object `X` breaks both walls — the escapes act on different tensor factors (base `V=ℂⁿ` for RH, fibre `ℂ^d` for YM), `D` is Abelian-on-base (diagonal/Cartan) while `su(d)` is non-Abelian-on-fibre, and `D⊗I` commutes with `I⊗T_a` so the base ingredient cannot supply the fibre's generators. What survives: both gaps are the **same recipe** (adjoin a non-commuting traceless operator ⇒ `so(n)` base / `su(d)` fibre) sharing **one** non-derivability root (no per-node / per-fibre slot in `∂EPI/∂t=νf·ΔNFR`). Reduces *two mysteries* to *one recipe, two realisations*, **not** *one piece*; sharpens, closes nothing. |
| `navier_stokes_recipe_bridge.py` | Does the one recipe that unifies RH and Yang–Mills extend to a THIRD Millennium programme, 3D Navier–Stokes? | Helmholtz split `∂_i u_j = S_ij + Ω_ij`; strain `S` symmetric traceless (`tr S = ∇·u = 0`, incompressibility); rotation `Ω ∈ so(3) ≅ su(2)`; coupling `(ω·∇)u ≡ 0` in 2D vs `≠ 0` in 3D; `tnfr.navier_stokes.operator` vortex stretching. | **One recipe, THREE realisations (honest OPEN)**: the NS escape is the **same shape** as YM — a non-Abelian traceless generator (`so(3) ≅ su(2)`, both rank-1, struct. const. `1`) — but on a **third** tensor factor, the velocity-component fibre `ℂ³` (distinct from RH's prime base `ℂⁿ` and YM's colour fibre `ℂ^d`). Gated by non-Abelianity: `‖(ω·∇)u‖ = 0` exactly in 2D (Abelian `so(2)`, 2D NS globally regular) and `≠ 0` in 3D (non-Abelian `so(3)`) — same threshold as RH (`n ≥ 2` primes) / YM (`d ≥ 2` colours). Shared non-derivability root: NS's residue is the Cascade Development Condition, which `N17-A` records as *“the structural analogue of `S(T) = (1/π) arg ζ(½+iT)`”*. Extends *one recipe, two realisations* to **three**; closes nothing (`NS-G5`, RH `G4`, YM mass gap all OPEN). |
| `directed_paley_bridge.py` | Do the `≡ 3 (mod 4)` primes emerge too, completing Camino 9's Reading B beyond `≡ 1 (mod 4)`? | Directed (non-symmetrised) quadratic-residue circulant; for prime `q ≡ 3 (mod 4)`, `−1` is a NON-residue ⇒ Paley *tournament* `A+Aᵀ=J−I` with eigenvalues `{(q−1)/2, (−1±i√q)/2}`; the Gauss sum `g=i√q` makes the secondary part PURELY IMAGINARY `±√q/2`, so `h(n)=‖dev from {(n−1)/2,(−1±i√n)/2}‖=0 ⇔ n` prime `≡3 (mod 4)`. | **Real/phase split = mod-4 split (honest OPEN)**: the SAME residue-QR construction is symmetric/REAL for `≡1 (mod 4)` (`−1` a QR, Camino 9 scale sector) and skew/IMAGINARY for `≡3 (mod 4)` (`−1` not a QR, phase sector) — the mod-4 prime classes split *exactly* along the wall's real-vs-phase boundary (the arithmetic of `−1` being a QR). `h(n)=0` reads the `≡3 (mod 4)` primes OUT from squares alone (`24/24` up to 200, primes-out), so `≡1` real ⊕ `≡3` imaginary = ALL odd primes (canonical adelic cross-check ✓); the only residual prime is `2`. But both sectors are NORMAL operators with DISCRETE spectra — "`i` times self-adjoint" is still not the continuous phase `S(T)`, which stays RH-equivalent and unreachable. Extends Reading B to all odd primes and connects the mod-4 split to the wall; closes nothing (`2` and `S(T)` remain, `G4 = RH` OPEN). |
| `kuramoto_farey_bridge.py` | Do the rationals (and φ) emerge from the TIME DYNAMICS, not just the static spectrum? | Sine circle map `θ_{n+1}=θ_n+Ω−(K/2π)sin(2πθ_n)` = single-node Kuramoto reduction of `∂EPI/∂t=νf·ΔNFR` (`Ω=νf`, `−(K/2π)sin=ΔNFR`); rotation number `ρ=lim(θ_N−θ_0)/N`; mode-locking ⇒ Arnold tongues / devil's staircase; Farey neighbours `\|p₁q₂−p₂q₁\|=1` ⇒ widest in-between plateau at the mediant `(p₁+p₂)/(q₁+q₂)` (Stern-Brocot); `F_n/F_{n+1}→1/φ=[0;1,1,1,…]` saturates Hurwitz `√5·q²·err→1`; Adler `\|Δω\|≤K` lock criterion. | **Lock/no-lock split = wall residue split (honest OPEN)**: the rationals emerge BLIND as the staircase plateaus (`41` harvested, `ρ→p/q` recovered by `limit_denominator` alone, the dynamical twin of Camino 9/14's primes-OUT spectral emergence); the Farey/Stern-Brocot tree organises the tongues (mediant law + widths strictly shrinking `1/2,2/3,3/5,5/8→0`); φ emerges as the most-irrational, LAST-to-lock limit (Hurwitz `0.9999`; the golden ratio `(1+√5)/2` recovered numerically ✓). At sub-critical `K` φ does NOT lock (staircase incomplete, locked measure `0.184<1`) — locked rationals = the reachable half (`range R∞`), the un-locked irrationals (φ foremost) play the residue role (`ker R∞ = Fix(G)^⊥`), the dynamical analogue of `S(T)=(1/π)arg ζ(½+iT)` (Adler winding `W` confirms the split, canonical `kuramoto_R_psi` confirms the locked pair coheres `R=0.95`). But φ is only the un-lockable LIMIT of reachable rationals (a SOFT residue), not the hard orthogonal `S(T)`; the dynamics yields discrete rationals + one φ, not the continuum. Extends emergence to the dynamical side and connects the split to the wall; closes nothing (`G4 = RH` OPEN; `ℝ` is the assumed continuum and **π** the one assumed structural scale). |
| `golden_residue_remesh_bridge.py` | Does the soft φ-residue of Camino 15 actually sit in `ker(R∞)` of the N15 theorem — and do the lockings sit in `range(R∞)`? | N15 REMESH-∞ Branch A: `R∞` is a self-adjoint ORTHOGONAL PROJECTION onto the resonant DFT lattice `{2πm/L}`, `L=lcm(4,8)=8` (canonical `tnfr.riemann.split_residue_by_remesh_infinity`); Parseval squared-norm energy fractions; a period-`q` circle-map orbit `cos(2πθ_n)` is range-supported ⇔ `q \| L`. | **Golden∈ker confirmed, map PARTIAL (honest OPEN)**: the golden quasi-periodic orbit `ρ=1/φ` lands in `ker(R∞)` (range `0.15%` — the dynamical twin of P50's prime-ladder `S_TNFR∈ker`); REMESH-commensurate lockings `ρ=1/2,1/4` (periods `2,4 \| 8`) land in `range(R∞)` (`100%`). But a genuinely locked `ρ=1/3` (period `3 ∤ 8`) ALSO lands in `ker` (`100%`) — so Camino 15's lock/no-lock split does NOT map 1-1 onto `range/ker`: `R∞`'s lattice is COARSER than the Farey set (`range R∞` = the period-divides-`L` sub-lattice only). Canonical controls reproduce (`sin(2πT/8)→range 100%`, `sin(γT)→ker 99.99%`); the SAME kernel holds the arithmetic carrier `log p` (Baker, P50 `RESIDUE_IN_KER_ONLY`) and the golden orbit — two incommensurate carriers of ONE residue subspace. Membership LOCATES the residue; it is NOT a route to RH. Sharpens AND limits the C15 analogy; closes nothing (`G4 = RH` OPEN; `ℝ` is the assumed continuum and **π** the one assumed structural scale). |
| `nodal_propagator_residue_bridge.py` | Can the oscillatory residue `S(T)` be **generated** by the canonical nodal propagator itself (not read off Riemann's template, the P31 gap) — and where does it land? | Canonical structural propagator `e^{−s H_P14}` via weighted spectral trace `Z(s)=Tr(W e^{−sH_freq})=Σ log(p) e^{−s k log p}` (P14/P12); on `Re(s)=½`, `Im Z(½+iT)=−Σ log(p) p^{−k/2} sin(T k log p)` is the von Mangoldt oscillation EMITTED by the propagator; N15 `R∞` projector + CCET `S_n`-equivariance. | **Propagator-generated, doubly-walled (honest OPEN)**: the observable is now produced BY the nodal time-evolution `e^{−iTH}` (T1: `weighted_spectral_trace` reproduces `−ζ'/ζ(2)` to truncation, machine-identical to the vectorised sum), not injected from Riemann–Siegel (the P31 methodological gap, CLOSED). It lands in `ker(R∞)` (`range 0.01%`, T2, the dynamical twin of P50/C16) AND its kernel residue is `S_n`-degenerate to machine precision (`max\|ker_can−ker_shuf\|≈6e-14`, T3 = CCET dynamical: shuffling prime labels leaves the residue invariant), so it sits in `ker(R∞) ∩ Fix(S_n)` while true `S(T)` needs `Fix(S_n)^⊥`. Global amplitude SCALE matches `S(T)` (`0.84×`, T4) — the obstruction is the missing phase/correlation structure, not magnitude (the structural reason P31 needed damping `d∼3–5`). Replaces P31's template read-off with a genuine nodal derivation and sharpens the wall to `ker(R∞)∩Fix(S_n)`; closes nothing (`G4=RH` OPEN; `ℝ` is the assumed continuum and **π** the one assumed structural scale; strengthens branch-B2). |
| `coherence_projector_sense_index.py` | What is the "closed room" *exactly*, and where does TNFR's own coherence metric `Si` sit relative to it? | N15 REMESH-∞ Branch A: `R∞` is a bounded self-adjoint ORTHOGONAL projection on `L²` (idempotent + self-adjoint ⇒ `L² = range ⊕ ker`, canonical resonant lattice `tnfr.riemann.build_resonant_bin_mask`, `L=lcm(4,8)=8`); the Sense Index `Si = α·νf + β(1−disp_θ) + γ(1−\|ΔNFR\|)` (canonical `tnfr.metrics.sense_index.compute_Si`); `S_n`-invariance of the complete prime graph `K_n`. | **Coherence = projection, room = its complement, `Si` symmetric (honest OPEN)**: `R∞` is an exact orthogonal projection (T1: `‖P²−P‖,‖P−Pᴴ‖≈1e-17`, `rank=trace(P)=L=8`, Parseval exact, `⟨range,ker⟩≈0`), so the closed room is *literally* the residue of coherence `ker(R∞)=range(I−P)`; it is vast (T2: `dim(range)=L=8` constant, `dim(ker)=N−8→∞`, coherent fraction `L/N→0`). `Si` is a coherence-capacity functional (T3: mean `Si` `0.70→0.40` monotone in phase dispersion, peaks at full synchrony) and `S_n`-degenerate (T4: sorted `Si` invariant under prime relabelling `≈2e-16`), so `Si ∈ Fix(S_n)` like `C(t)`/spectral trace — blind to `ker(R∞)∩Fix(S_n)^⊥` where `S(T)` lives. Characterises the room directly and unifies `Si` with the symmetric diagnostics; closes nothing (`G4=RH` OPEN; `ℝ` is the assumed continuum and **π** the one assumed structural scale; branch-B2). |

The harnesses build on one another: `composition_arithmetic.py` exports the
shared spectral helpers (`lap_spectrum`, `adj_spectrum`, `outer_sum`,
`outer_prod`, `character_norm`, `eigenspaces`, `automorphism_matrices`) reused by
`operational_irreducibility.py`, `bridge_primes_riemann.py`,
`emergent_rationals.py`, `equivariance_wall.py` (which also cross-checks the
canonical `tnfr.dynamics.adelic` carrier `νf = log p` as its `S_n`-breaking
per-node diagonal), `chiral_involution.py`
(which also reuses `integer_spectrum`, `is_pm_symmetric` from
`emergent_rationals.py` and `winding_ring`, `winding_number`,
`classify_particle` from `tnfr.physics.emergent_particles`), and
`commutant_bridge.py` (which reuses `automorphism_matrices`, cross-checks its
YM side against the canonical `tnfr.yang_mills.audit_nonabelian_derivability`, and
its RH side against the canonical `tnfr.dynamics.adelic` carrier `νf = log p` as
its `S_n`-breaking escape diagonal), and
`phase_wall.py` (which reuses `adj_spectrum` and cross-checks its phase carrier
against the canonical `tnfr.dynamics.adelic` engine — `νf = log p` — `mpmath`'s
`ζ(½+iT)`, and the same `tnfr.yang_mills.audit_nonabelian_derivability` `U(1)`
verdict), and `paley_bridge.py` (which reuses `adj_spectrum` to cross-check the
residue-circulant adjacency spectrum against the closed-form Paley eigenvalues
`(−1±√n)/2`, and cross-references `tnfr.dynamics.adelic` for the `νf` prime
support, `tnfr.riemann.paley_gap_coercivity` for the canonical P25 “does not close
G4” scope, and `mpmath`'s `ζ(½+iT)` for the continuous phase `S(T)`).
`boundary_vibration.py` is self-contained on the algebra side (a NetworkX path
graph and its `ℤ₂` reflection) but cross-checks the canonical TNFR–Riemann stack
end-to-end: `tnfr.riemann.von_mangoldt` (the `−ζ'/ζ` convergence barrier),
`tnfr.riemann.prime_ladder_hamiltonian` (the self-adjoint `{k log p}` source, P14),
`tnfr.riemann.hilbert_polya` (the imported `γₙ` target and the `W₁` gap, P27),
`tnfr.dynamics.adelic` (the `νf = log p` geometric-trace carrier), and `mpmath`'s
`ζ(½+iT)` to confirm the target ordinates are genuine zeros.
`missing_piece_bridge.py` builds directly on `commutant_bridge.py`: it imports its
`adjacency_laplacian`, `canonical_per_node_diagonal`, `su2_generators`,
`commutator_norm` and `symmetric_projector`, reuses `composition_arithmetic.py`'s
`automorphism_matrices`, and cross-checks the canonical
`tnfr.yang_mills.audit_nonabelian_derivability` (YM side) together with the adelic
carrier `νf = log p` (RH side).
`navier_stokes_recipe_bridge.py` extends that bridge to a third programme: it reuses
`commutant_bridge.py`'s `adjacency_laplacian`, `canonical_per_node_diagonal`,
`catalog_operators`, `commutator_norm` and `su2_generators`, builds the strain /
rotation split of a canonical Taylor–Green velocity field, and cross-checks the 3D
vortex-stretching field of `tnfr.navier_stokes.operator` (exactly zero on a
2D-embedded field, nonzero in 3D) together with the same
`tnfr.yang_mills.audit_nonabelian_derivability` `U(1)` verdict.
`directed_paley_bridge.py` extends `paley_bridge.py` to the second odd prime
class: it reuses that harness's `is_prime`, `quadratic_residues`, `paley_gap`
and `riemann_s_phase`, builds the *directed* (non-symmetrised) residue circulant
whose tournament spectrum `(−1±i√q)/2` exposes the `≡ 3 (mod 4)` primes through
the imaginary Gauss-sum signature `√q/2`, and cross-checks the union of both
odd prime classes against the canonical `tnfr.dynamics.adelic` carrier's prime
support.
`kuramoto_farey_bridge.py` is self-contained on the dynamics side (it iterates
the sine circle map and harvests the devil's-staircase plateaus with NumPy and
`fractions.Fraction` only), but cross-checks the canonical engine end-to-end:
the emergent golden-ratio limit `(1+√5)/2` (the Fibonacci-Farey
limit) and the canonical Kuramoto order parameter `tnfr.gamma.kuramoto_R_psi`
(the Adler 2-oscillator lock that confirms a commensurate detuning coheres while
the golden detuning winds), turning Camino 9/14's *static-spectrum* emergence of
the primes into the *time-evolution* emergence of the rationals and φ.
`golden_residue_remesh_bridge.py` builds directly on `kuramoto_farey_bridge.py`:
it reuses that harness's `circle_map_rho`, `invert_rho`, `sweep_rho`, `PHI_INV`
and `K_SUB` to generate the circle-map orbits, then projects each demeaned
`cos(2πθ_n)` signal with the canonical N15 projector
`tnfr.riemann.split_residue_by_remesh_infinity` (NumPy DFT-bin-mask fallback)
and reconciles the golden orbit's kernel membership against the canonical P50
prime-ladder certificate `tnfr.riemann.compute_residue_split_certificate`
(`RESIDUE_IN_KER_ONLY`), turning Camino 15's *soft* φ-residue analogy into a
precise statement about the kernel of an actual proven orthogonal projection.
or physical realisation in coupled TNFR systems rather than being postulated.
They do **not** derive the multiplicative *prime* fine structure of ℤ from pure
dynamics — that residue coincides with the `S_n`-unreachable oscillatory term
`S(T) = (1/π) arg ζ(½ + iT)` of the (frozen) TNFR-Riemann programme and is
RH-equivalent. `equivariance_wall.py` makes this last point *generic*: the same
`Fix(G)^⊥` obstruction (prime individuation for `S_n`, degeneracy-lifting for
`D_n`, the chiral sign for `ℤ₂`) is one group-theoretic shape shared by the
Riemann, Navier–Stokes and Yang–Mills walls — it **unifies the obstruction, it
does not remove it**. `chiral_involution.py` does the complementary, *constructive*
move: the additive inverse `−n` of ℤ and the antiparticle are one chiral `ℤ₂`
(the *anticommuting* `Γ A Γ = −A`, distinct from the *commuting* Camino-5
automorphism wall), with `n+(−n)=0` and the `|W|=0` vacuum as the same neutral
element — a precise structural analogy, **not** a derivation of CPT, antimatter,
or the Standard Model. The real continuum ℝ is the assumed continuum substrate,
and **π** is the one assumed structural scale; everything else emerges from the
nodal dynamics. Nothing here advances or closes
`G4 = RH`, Navier–Stokes regularity, or the Yang–Mills mass gap.

`commutant_bridge.py` is the deepest path and its thesis verdict is, by design,
**OPEN**. Its *structural* checks pass at machine precision: in both the Riemann
and the Yang–Mills programmes the reachable set is the **commutant** of a group
acting on the (colour-lifted) graph, and each open target lives in the orthogonal
complement that the commutant cannot reach — `Fix(S_n)^⊥` (the RH residue `S(T)`)
for the `S_n` permutation rep, and the traceless `su(d)` part of the non-Abelian
curvature `[A_μ,A_ν]` for the `U(d)` gauge action (`ℂ^{d×d}=ℂI_d⊕su(d)`, since
`{I_V⊗U}'=End(V)⊗ℂI_d` by the double-commutant theorem). The catalog only ever
builds `f(A,L)`, which is colour-blind (`f(A,L)⊗I_d`) and therefore trapped in
that commutant. This **unifies the two Millennium obstructions as one shape**; it
does **not** close either. The escape in each programme is exactly the ingredient
that is *not* nodal-equation-derivable: the per-node diagonal P2 for RH, and
non-commuting `su(d)` generators for YM — the latter confirmed by the canonical
`tnfr.yang_mills.audit_nonabelian_derivability` returning `OPEN_DERIVABILITY_GAP`
(`canonical_gauge_group = U(1)`, `has_noncommuting_generators = False`). Finite
toy-graph plus `su(2)` linear algebra: it proves the obstructions *coincide in
shape*, never that TNFR proves Yang–Mills, RH, or a mass gap.

`phase_wall.py` is the e–π companion to that deepest path and its thesis verdict
is likewise, by design, **OPEN**. It asks *why* the open target is a phase. The
four fields of the tetrad are the four orders of the derivative tower over the
graph (only **π** is a genuine structural scale — the phase-wrap bound; the other
field scales are heuristic or set by the spectral gap, `ξ_C ∝ 1/√λ₂`), and the catalog is
built from the symmetric coupling `A = Aᵀ` and the self-adjoint `L = D − A`. Its
structural checks pass at machine precision: every `f(A,L)` is self-adjoint, so its
spectrum is real and its eigen-phases are locked to `arg ∈ {0, π}` (a sign), while
the Riemann residue `S(T) = (1/π) arg ζ(½ + iT)` is a **continuous** phase on the
`e–π` circle `S¹` — the two sets are disjoint. The only map from the real axis to a
continuous phase is the complexification `z ↦ exp(i z)` (Euler: `exp(iπ) = −1`); the
canonical engine owns exactly one such carrier — the adelic unitary
`U(t) = diag(exp(i·t·νf))` with `νf = log p` (`tnfr.dynamics.adelic`) — and it does
reach the circle, but its per-node arithmetic content `νf = log p` is **imposed**
(a prime sieve), not produced by `∂EPI/∂t = νf·ΔNFR`, which reads `νf` as input.
Promoting `νf` to a circle-valued / Pontryagin-dual object is the non-derivable step
(`FORWARD_INDEPENDENT_OF_BACKWARD`). This is the **exact e–π mirror** of the
Yang–Mills `U(1)` gap of `commutant_bridge.py`: the canonical gauge is the same
`U(1)` circle (a scalar phase `exp(i φ)`), and the missing ingredient — non-commuting
generators (YM) / derived prime frequencies (RH) — is not nodal-derivable. The
harness **locates** the residue as a real-vs-phase wall; reaching `S(T)` is
RH-equivalent and remains **OPEN**.

`paley_bridge.py` answers a direct objection to `phase_wall.py`: *the zeros do come
from somewhere — the Paley gap*. The objection is **correct**, and the harness
concedes it with running code. The residue-circulant Laplacian `λ₂` (computed by
FFT) and the Paley gap `g(n) = |λ₂ − (n−√n)/2|` vanish **exactly** at the primes
`n ≡ 1 (mod 4)` (reproduced to `n ≤ 200` here; `≤ 2601` in the source note,
Zenodo 10.5281/zenodo.17665853) — so the prime support of the carrier frequency
`νf = log p` is **not sieved**, it emerges from a spectral **identity** that
realises primality as a `ΔNFR = 0` structural equilibrium. The adelic engine's
`≡ 1 (mod 4)` primes match the Paley-derived set exactly (`21 = 21` up to 200).
**But this does not breach the Camino-8 wall — it sharpens it.** Two facts pin the
limit: (i) the Paley mechanism is **real / self-adjoint** (the residue circulant is
symmetric ⇒ real `λ₂` ⇒ real `g(n)`, eigen-phases in `{0,π}`, adjacency spectrum
matching the closed form `(−1±√n)/2` to `5e-15`), so it lives entirely in the
real/scale sector; (ii) there are **two different “zeros”** — the Paley-gap zeros
are *real integers* (primes `≡ 1 mod 4`) and are **disjoint** from the Riemann
ordinates `γₙ` (min distance `0.59`), while `S(T) = (1/π) arg ζ(½+iT)` is a
*continuous phase* that no real `g(n)` can produce. The source note states
*“reproducible; not a primality proof”*; the canonical P25 module
(`tnfr.riemann.paley_gap_coercivity`) states it *“does not close G4”*. So the Paley
gap grounds the prime **support** (the real “where”), not the **phase** residue (the
continuous “argument”): it demonstrates the real sector reaches even prime
individuation, while the oscillatory `S(T)` stays in the orthogonal phase sector.
Reaching `S(T)` remains RH-equivalent and **OPEN**. (Honest limit: the gap covers
the `≡ 1 (mod 4)` class only; `≡ 3 (mod 4)` primes and `2` need a complementary
construction.)

`boundary_vibration.py` follows the directive that every harness should derive
whatever it can from TNFR structure and dynamics, exactly as
`src/tnfr/dynamics/adelic.py` does (`νf = log p`, the nodal equation
`∂EPI/∂t = νf·ΔNFR`, the zeros held only as Ground-Truth target). It asks the
sharpest form of the Hilbert–Pólya question — *why can't the engine derive the
zeros' location canonically, without `mpmath`?* — and answers it with running code.
Its structural checks pass at machine precision: (1) the TNFR-native von Mangoldt
carrier `Z_vM(s) = Σ w e^{−sμ}` stabilises for `Re(s) > 1` (matches the classical
`−ζ'/ζ(2) = 0.567` to `5e-3`) but **diverges** as the truncation grows at
`Re(s) = ½` (drift `0.01` vs `13.3`), so the object that sees the primes literally
cannot be evaluated where the zeros live — the abscissa `Re = 1` *is* the barrier;
(2) the canonical self-adjoint Hamiltonian P14 reproduces the source spectrum
`{k log p}` exactly with **no `mpmath`**; (3) the adelic geometric-trace carrier is
built purely from `νf = log p`, while the nodal pressure `ΔNFR = −∇V` that lands the
flow on the zeros is defined *from* the imported `known_zeros` — making visible that
`{γₙ}` enter only as the resonance target; (4) **self-adjoint + the `ℤ₂` reflection
`R` ⇒ real spectrum on the fixed axis** (eigen-parities `[+,−,+,−,…]`), the rigorous
core of the Hilbert–Pólya intuition: *“self-adjoint ⇒ real ⇒ on the line”* is TRUE
as algebra. The honest residual (5): `{k log p}` grows like `log n` while `γₙ` grows
like `2πn/log n` (`W₁(P14,T_HP) = 115`, growth ratio `26`), so no smooth structural
map carries the source to the target. This **locates `G4 = RH`** precisely as the
canonical analytic continuation of `Z_vM` across `Re = 1` / the map `{k log p} →
{γₙ}`; `mpmath` (P13/P27) only marks that barrier — it draws the target, it never
derives it. Exhibiting *the* self-adjoint operator whose spectrum is `{γₙ}` from
TNFR structure alone is the open piece. `ℝ` is the assumed continuum and **π** the
one assumed structural scale; nothing here closes `G4`.

`primes_as_consequence.py` is the thirteenth harness and returns to the very first
question of the map — *do the primes themselves emerge, or are they fed in?* — now
that Caminos 5–10 have located the wall. Its thesis verdict is, by design, **OPEN**,
and its structural checks pass at machine precision. It reuses
`composition_arithmetic.py`'s `automorphism_matrices`/`character_norm`/`eigenspaces`
and `paley_bridge.py`'s `is_prime`/`paley_gap`, and cross-checks the canonical
`tnfr_primality.core` pressure, `tnfr.dynamics.adelic`, and
`tnfr.riemann.paley_gap_coercivity`. It separates two readings of the canonical
theorem *“n prime ⟺ ΔNFR(n) = 0”* (`theory/TNFR_NUMBER_THEORY.md` §4): **Reading A**
evaluates `ΔNFR(n) = ζ(Ω−1) + η(τ−2) + θ(σ/n − (1+1/n))` (the structural-pressure coefficients are canonical units; the
prime ⟺ ΔNFR = 0 criterion is coefficient-independent, theory §4.2) and
reproduces the primes `n ≤ 200` *exactly*, but it **consumes the factorization** —
`Ω, τ, σ` are obtained by `n % d`, so as a *derivation* it is circular (`3026` trial
divisions consumed; primes go IN and come back re-labelled `ΔNFR = 0`). **Reading B**
is a genuine non-circular emergence: the Paley gap `g(n) = 0` reads the primes
`≡ 1 (mod 4)` *out* of a self-adjoint residue spectrum built only from squares
`x·x mod n` — it never computes `n % k` (`21 = 21` primes-out up to 200). The
**frontier** confirms the limit of pure emergence: the dim-4 mode of `K₅` is
irreducible (`⟨χ,χ⟩ = 1.00`) yet `4 = 2 × 2` arithmetically, so *irreducibility ≠
primality* — representation theory alone cannot reproduce unique factorisation. The
**bridge** then pins the residual: the adelic carrier reads every prime IN
(`νf = log p`), Reading B reads only the `≡ 1 (mod 4)` class OUT, leaving `2`, the
`≡ 3 (mod 4)` primes, and the continuous phase `S(T) = (1/π) arg ζ(½+iT)` as the
**same real-vs-phase wall** as Caminos 8–10. So the optic-shift is **real and
clarifying** — a TNFR prime *is* a zero-pressure structural equilibrium, partially
spectrally emergent — but it **locates** the residual; it does not close `G4 = RH`.
`ℝ` is the assumed continuum and **π** the one assumed structural scale.

`missing_piece_bridge.py` is the fourteenth harness and closes the conceptual arc
of Caminos 5–11 by asking the sharpest cross-program question directly: *are the
two ways out of the wall — the RH `S_n`-breaking diagonal and the Yang–Mills
non-commuting generators — one missing canonical piece, or two?* Its thesis
verdict is, by design, **OPEN**, and its four structural checks pass at machine
precision. The strong unifying conjecture (recorded in repo memory: *one absent
canonical piece; closing one gives the other*) is **REFUTED**: the two escapes act
on different tensor factors — the base `V = ℂⁿ` for RH, the fibre `ℂ^d` for YM —
`D` is Abelian-on-base (diagonal/Cartan) while `su(d)` is non-Abelian-on-fibre, and
`D ⊗ I` commutes with `I ⊗ T_a`, so the base ingredient cannot manufacture the
fibre's generators. What **survives** is a precise weaker unification: both gaps are
the **same recipe** (break a commutant by adjoining a non-commuting, traceless
operator — `so(n)` on the base, `su(d)` on the fibre) sharing **one**
non-derivability root (no per-node / per-fibre slot in `∂EPI/∂t = νf · ΔNFR`). It
reduces *two mysteries* to *one recipe with two independent realisations*, **not**
to *one piece*; it sharpens the conjecture and **closes nothing**. `ℝ` is the assumed continuum
and **π** the one assumed structural scale; nothing here proves RH or the
Yang–Mills
mass gap.

`navier_stokes_recipe_bridge.py` is the fifteenth harness and asks whether the
weaker unification that `missing_piece_bridge.py` left standing — *one recipe, two
realisations* — reaches a **third** Millennium programme, the global-regularity
problem for the 3D incompressible Navier–Stokes equations. Its thesis verdict is, by
design, **OPEN**, and its four structural checks pass at machine precision. The
reachable / smooth half is the linear viscous flow — the heat semigroup
`exp(−ν t L)`, self-adjoint, equivariant, scale-translation commuting — exactly the
NS analogue of Riemann's `range(R∞)` and Yang–Mills' colour-scalar part. The
obstruction is the **non-linear vortex-stretching term** `(ω·∇)u`, and the canonical
engine's own `vortex_stretching_field()` docstring states the gating verbatim:
*“in 2D it is identically zero … so 2D NS is globally regular; in 3D it can in
principle amplify enstrophy without bound … the Clay Millennium Problem NS-G5.”*
Structurally the stretching is carried by the velocity-gradient split
`∂_i u_j = S_ij + Ω_ij` into the **strain** `S` (symmetric, traceless because
`tr S = ∇·u = 0` by incompressibility) and the **rotation** `Ω` (antisymmetric,
`∈ so(3) ≅ su(2)`). This is the **same recipe** as RH (`[A,D] ∈ so(n)`, traceless
by anti-symmetry) and YM (`su(d)`, traceless by definition) — adjoin a non-commuting
traceless generator — now on a **third** tensor factor: the velocity-component fibre
`ℂ³`, distinct from RH's prime base and YM's colour fibre. The wall is gated by
non-Abelianity exactly as before: `‖(ω·∇)u‖ = 0` to machine precision for a
2D-embedded field (rotation lives in the Abelian `so(2)`, one generator, no wall —
reproduced against the canonical operator), and `≠ 0` for the genuine 3D
Taylor–Green field (non-Abelian `so(3)`, wall present) — the same threshold as RH
needing `n ≥ 2` primes and YM needing `d ≥ 2` colours. The non-derivability roots
are one family on distinct slots: RH's `νf = log p` is imposed, YM's non-Abelian
multiplet is an audited derivability gap, and NS's residue is the **Cascade
Development Condition**, which `CHANGELOG` `N17-A` records as *“not derivable from
U3, U5, or the nodal equation … the structural analogue of
`S(T) = (1/π) arg ζ(½+iT)` in the Riemann programme.”* The harness therefore
**extends** the weaker unification from two Millennium programmes to three — *one
recipe, three realisations* (`so(n)` prime base / `su(d)` colour fibre / `so(3)`
velocity fibre) — and **closes nothing**: `NS-G5`, the Clay 3D Navier–Stokes
problem, RH (`G4`), and the Yang–Mills mass gap all remain **OPEN**. The recipe
unifies the obstructions; it does not remove them. `ℝ` is the assumed continuum and
**π** the one assumed structural scale.

`directed_paley_bridge.py` is the sixteenth harness and returns to the
emergence-of-numbers thread (Caminos 1–9) to settle Camino 9's one explicit
open gap: *do the `≡ 3 (mod 4)` primes emerge too, or only the `≡ 1 (mod 4)`
class?* Its thesis verdict is, by design, **OPEN**, and its four structural
checks pass at machine precision. Camino 9 grounded the `≡ 1 (mod 4)` primes in
a REAL/self-adjoint Paley *graph* gap (`−1` is a quadratic residue there, so the
residue circulant is symmetric — the scale sector). This harness grounds the
`≡ 3 (mod 4)` primes in the IMAGINARY Gauss-sum signature of the Paley
*tournament* (`−1` is a non-residue, so `A + Aᵀ = J − I` and the secondary
eigenvalues are `(−1 ± i√q)/2`, real part `−½`, imaginary part `±√q/2` — the
phase sector). The new detector `h(n) = 0` reads the `≡ 3 (mod 4)` primes *out*
of squares alone (`24/24` up to 200, never `n % k`), so Camino 9's real gap
`g(n)` and this imaginary gap `h(n)` **together make every odd prime emerge**
(`21 + 24 = 45` odd primes `≤ 200`, cross-checked against the canonical adelic
carrier), split **exactly** by whether `−1` is a quadratic residue — which is
precisely the real-vs-phase boundary of the Equivariance Wall. But both sectors
are NORMAL operators with DISCRETE point spectra: "`i` times self-adjoint" is
still not the continuous phase `S(T) = (1/π) arg ζ(½ + iT)`, which remains
RH-equivalent and unreachable, and the even prime `2` (`≡ 2 mod 4`) sits outside
both classes. So the harness **extends** the emergence-of-numbers line to all
odd primes and **connects** the mod-4 prime split to the wall — but it **closes
nothing**: `2`, the continuous phase `S(T)`, and `G4 = RH` all remain **OPEN**.
`ℝ` is the assumed continuum and **π** the one assumed structural scale.

`kuramoto_farey_bridge.py` is the seventeenth harness and supplies the
*dynamical* half of the emergence-of-numbers thread (Caminos 1–9, 14): where
Camino 9/14 read the primes off the **static spectrum** of a fixed graph, this
harness reads the **rationals** (and φ) off the **time evolution** of the nodal
phase. Its thesis verdict is, by design, **OPEN**, and its four structural
checks pass. The carrier is the sine circle map
`θ_{n+1} = θ_n + Ω − (K/2π) sin(2π θ_n)`, which is exactly the single-node
Kuramoto reduction of `∂EPI/∂t = νf · ΔNFR(t)` (`Ω = νf` the bare structural
frequency, `−(K/2π) sin(2π θ) = ΔNFR` the coupling pressure), and the emergent
quantity is the rotation number `ρ = lim (θ_N − θ_0)/N`. **(1)** The rationals
emerge *blind* as the mode-locked plateaus of the devil's staircase: harvesting
the flat runs at criticality recovers `41` distinct rationals via
`limit_denominator` alone — the dynamical twin of Camino 9/14's primes-OUT
spectral emergence. **(2)** The Farey/Stern-Brocot tree organises the Arnold
tongues: between Farey neighbours the widest plateau is the mediant, and tongue
width strictly shrinks along the Fibonacci path `1/2, 2/3, 3/5, 5/8 → 0`.
**(3)** φ emerges as the canonical, *most-irrational* number: the Fibonacci
ratios `F_n/F_{n+1} → 1/φ = [0; 1, 1, 1, …]` saturate the Hurwitz bound
(`√5·q²·err → 0.9999`) and the limit is the golden ratio `(1+√5)/2`, recovered numerically to `6e-13`.
**(4)** At sub-critical coupling φ does **not** lock (the staircase is incomplete,
locked measure `0.184 < 1`): the locked rationals are the *reachable* half
(`range R∞`), while the un-locked irrationals — φ foremost, the LAST to lock —
play the *residue* role (`ker R∞ = Fix(G)^⊥`), the dynamical analogue of the
oscillatory `S(T) = (1/π) arg ζ(½ + iT)` (the canonical `tnfr.gamma.kuramoto_R_psi`
confirms the Adler-locked commensurate pair coheres at `R = 0.95` while the
golden detuning winds). But the analogy is **honest and soft**: φ is only the
un-lockable *limit* of reachable rationals (an accumulation boundary), **not**
the hard orthogonal residue `S(T)`; the dynamics yields a discrete set of
rationals plus one distinguished φ, never the continuum. So the harness
**extends** the emergence-of-numbers line from the spectral to the dynamical
side and **connects** the lock/no-lock split to the wall — but it **closes
nothing**: `G4 = RH` remains **OPEN**, and `ℝ` is the assumed continuum and **π**
the one assumed structural scale.

`golden_residue_remesh_bridge.py` is the eighteenth harness and is the capstone
of the dynamical thread: it ties Camino 15 to the ONE canonical proven object of
the (frozen) TNFR-Riemann programme — the N15 REMESH-∞ orthogonal projector
`R∞` (`theory/REMESH_INFINITY_DERIVATION.md`, Branch A). Its thesis verdict is,
by design, **OPEN**, and its four structural checks pass at machine precision
using the *canonical* projector `tnfr.riemann.split_residue_by_remesh_infinity`.
Camino 15 only *noted, as a soft analogy*, that φ (the last to lock) plays the
residue role; this harness makes it **precise**. **(1)** The golden
quasi-periodic orbit `ρ = 1/φ` lands in `ker(R∞)` (range fraction `0.15%`): its
incommensurate frequency `2π/φ` misses the resonant lattice `{2πm/L}`,
`L = lcm(4,8) = 8` — the **dynamical twin** of P50's prime-ladder
`S_TNFR ∈ ker(R∞)` (whose carrier `log p` is incommensurate by Baker's theorem).
**(2)** REMESH-commensurate lockings `ρ = 1/2, 1/4` (periods `2, 4 | 8`) land in
`range(R∞)` (`100%`): every harmonic sits on the resonant lattice. **(3)** The
**honest limit**: a genuinely locked `ρ = 1/3` (period `3 ∤ 8`) *also* lands in
`ker(R∞)` (`100%`), so Camino 15's lock/no-lock dichotomy does **not** map
one-to-one onto N15's `range`/`ker` split — `R∞`'s lattice is **coarser** than
the full Farey set of lockings, and `range(R∞)` is the period-divides-`L`
sub-lattice only. **(4)** The engine's own controls reproduce
(`sin(2πT/8) → range`, `sin(γT) → ker 99.99%`), and the canonical P50
certificate confirms the prime-ladder `S_TNFR ∈ ker` — the **same** kernel holds
both the arithmetic residue carrier (`log p`, Baker) and the golden orbit, two
incommensurate carriers of one large residue subspace (its complement, the
resonant lattice, is measure-zero among all frequencies). Membership in
`ker(R∞)` **LOCATES** the residue; it is **not** a route to RH. The harness
**sharpens** the Camino-15 analogy (golden ∈ ker, now precise) and **limits** it
(not all lockings reach range) at once — and **closes nothing**: `G4 = RH`
remains **OPEN**, and `ℝ` is the assumed continuum and **π** the one assumed structural scale.

## Structural-emergence benchmarks — chemistry, geometry & dimension

A second structural-emergence arc applies the same *let-it-emerge* discipline to
**shell structure, symmetry cardinals and spatial dimension**, under one
canonical gate: *everything must emerge from TNFR structure and dynamics — no
imported quantum mechanics, no Coulomb law, no postulated geometry*. Each harness
models a system as a pure structural manifold (or reads the substrate's own
emergent symplectic geometry), lets the structure/dynamics produce what it
produces, and only then identifies the emergent ontology against observed
phenomena. As with the numbers thread, the comparison framework (Laplace–Beltrami
`(2l+1)` degeneracies, the representation theory of the dynamical-symmetry groups
SO(3)/SO(4)/U(3), the `K_m`-simplex Laplacian spectra, the spectral dimension
estimator) is **standard external mathematics**; the TNFR contribution is the
emergent reading and the **honest boundary** where a pure single-coherence
manifold stops and a two-body / imported-geometry ingredient would be required.
All seven are dependency-light (NetworkX + NumPy), deterministic (no RNG — the
sphere/ball graphs are Fibonacci-deterministic), and lint-clean. Several reuse the
canonical `tnfr.physics.emergent_chemistry` primitives (`fibonacci_sphere_graph`,
`structural_eigenmodes`), the canonical nodal-topology read-out
`tnfr.physics.fields.classify_nodal_topology` (the radial/annular/multinodal
emergent geometry), the substrate certificate
`tnfr.physics.symplectic_substrate.verify_polarization_symmetry` (the U(2)
polarization symmetry), and the shared fixed-point kernel
`tnfr.metrics.common.is_structural_equilibrium` (closed shell = prime = relaxed
node = one `ΔNFR = 0` predicate). Run any of them directly:

```bash
PYTHONPATH=src python benchmarks/<script>.py
```

| Script | Question | Engine (independent ground truth) | Verdict |
| --- | --- | --- | --- |
| `emergent_shell_ordering.py` | Does atomic shell structure emerge from a pure TNFR structural manifold (no Coulomb, no QM)? | Sphere `□` radial-path Cartesian product; Laplace–Beltrami `(2l+1)` degeneracy; `classify_nodal_topology`; infinite-spherical-well closures. | The independent-particle **SKELETON** emerges: `(2l+1)` angular degeneracy, the canonical `+` sum-ordering of radial⊕angular modes, an **emergent radial nucleus** (the bounded ball's geometric centre), and spherical-well closures `2,8,18,20`. aufbau `(n+l)` does **not** emerge — it encodes the missing many-body screening, so its postulate in `emergent_chemistry` is **justified**, not a defect. |
| `emergent_screening.py` | Does electron screening emerge from a self-consistent `Φ_s` back-reaction among co-resident sub-EPIs? | SCF loop: occupied sub-EPIs (U5) → canonical `Φ_s` field `Σ ρ/d²` (U6) → shift operator → re-diagonalise; **no** Hartree–Coulomb injected. | A screening-**like** degeneracy-lifting reorganisation genuinely emerges — but it is **repulsive** (the emergent nucleus is a `Φ_s` **maximum**, not an attractive sink), so no coupling reproduces the atomic table. Both atomic ingredients (an attractive nucleus **and** correctly-signed screening) are measured **non-emergent** from one relaxing manifold (`ΔNFR → uniform` forbids a sustained sink). |
| `emergent_shell_cardinals.py` | Are the magic numbers spatial counts or dynamical-symmetry **irrep cardinals**? | Cumulative `2×(irrep dim)` of a dynamical-symmetry chain: SO(3) `(2l+1)` → `[2,8,18,32]`; SO(4) `n²` → `[2,10,28,60]`; U(3) `(N+1)(N+2)/2` → `[2,8,20,40]`. | Magic numbers are **symmetry cardinals**, not spatial counts: the atomic "10" is the **SO(4) Coulomb cardinal** (the `n=2` shell, `2s+2p` degenerate). The spatial ball **broke** SO(4)→SO(3) (split `2s` from `2p`) → `2,8` not `2,10`. Lesson: map a TNFR layer to the cardinals of its emergent dynamical symmetry, **not** to an imported spatial box. |
| `emergent_substrate_symmetry.py` | What is the substrate's **own** emergent symmetry, with nothing imported? | `verify_polarization_symmetry` (su(2) closes, charges conserved); the two conjugate sectors `K_φ+iJ_φ`, `Φ_s+iJ_ΔNFR`; U(2) isotropic-oscillator cardinals `2(N+1)`. | The substrate is **structurally locked** to U(2) / 2 sectors (`CONJUGATE_PAIR_LABELS` = 2, `BLOCK_SYMPLECTIC_FORM` is 4×4, the 13 operators are symplectomorphisms ⇒ **no third sector possible**). Its cardinals `[2,6,12,20]` = the observed **2D quantum-dot** magic numbers (Tarucha 1996) — derived with nothing imported. The substrate fibre is intrinsically 2D. |
| `emergent_base_dimension.py` | Does the network's spatial/spectral dimension emerge, or is it a free input? | Spectral dimension `N(λ) ~ λ^{d_s/2}` (calibrated: ring `1.05` < grid2D `1.85` < grid3D `2.62`). | The base spectral dimension is a **FREE topology input**: THOL tree `d_s ≈ 1.6`, U3 resonant coupling `d_s` tunable by the phase gate (`π/2 → 6.95`, `π/6 → 2.48`). No TNFR structure-builder pins it to 3. REMESH/RA both **preserve topology** (temporal recursion / propagation), so neither adds a spatial dimension — the `(2+1)` enrichment is temporal, not a third spatial sector. |
| `emergent_simplex_dimension.py` | Is the emergent **integer** the same thing as a **dimension**? | `L(K_{n+1})` spectrum `{0, (n+1)^{×n}}`; the multiplicity `n` = standard-irrep dim of `S_{n+1}` = the `n`-simplex dimension. | **number = dimension = SIMPLEX GRADE** of a coupled-NFR form (EPI): edge `K_2` → `1` (1D), triangle `K_3` → `2` (2D), tetrahedron `K_4` → `3` (3D). The **fractal-resonant lift** (one apex resonantly coupled to all, the cone `K_3 → K_4`) raises grade `2 → 3` = 2D → 3D. EPI form-complexity **is** the dimension; this unifies the emergent-integers and emergent-dimension threads. |
| `emergent_dimension_dynamics.py` | Does the **dynamics** build these coherent simplices, climbing grade/dimension by itself? | The U3 resonance gate `\|φ_i − φ_j\| ≤ Δφ_max` makes a mutually-compatible cluster a clique = `K_k` = a coherent simplex; max-clique grade = dimension. | **Yes, coherence-gated**: Emission (AL) + U3 Coupling/Resonance (UM/RA) accretes coherent NFRs, lifting the grade one step (point → edge → triangle → tetra = 0D → 1D → 2D → 3D); an **incoherent** emission does **not** lift it (only resonant degrees count); synchronisation grows the simplex. Dimension is **dynamically generated** by resonant coherence, one fractal-resonant degree at a time — **not pinned at 3**. |
| `emergent_fractal_simplex_dimension.py` | The simplex grade *climbs* and the spectral `d_s` is *free* — so what **pins** the dimension? | THOL/U5 "preserve global form + create sub-EPIs" + the Kron/`R_eff` fractal-consistency (node **is** a subgraph) ⇒ recursing `K_m` into `m` corner-glued copies = the **Sierpinski gasket of `K_m`**; a self-similar set has a definite dimension. | **Self-similar THOL nesting PINS it.** Exact similarity dimension `d = log(m)/log(2)` is **set by the local simplex grade** `m−1` (`K_3 → 1.585`, **`K_4` tetrahedron → EXACTLY 2.000**, `K_5 → 2.322`). The previously **FREE** spectral `d_s` becomes **DEFINITE** — it converges to the self-similar `2log(m)/log(m+2)` (vs a random tree's free `d_s`). The grade-3 tetrahedron (3D EPI form) nests to dimension **exactly 2 = the locked U(2) fibre** (a numerical convergence, honestly flagged — Sierpinski-tetrahedron Hausdorff dim). The **form generates** the dimension; the self-similar (fractal) lift **fixes** it. |
| `emergent_atomic_shells.py` | If dimension emerges from the EPI form, do the atom's **shells** come from that same form (not an imported spatial ball)? | A single simplex `K_{d+1}` is one shell (standard irrep of `S_{d+1}`, degeneracy `d`); THOL/U5 self-similar nesting (Sierpinski gasket of `K_m`) makes it a **tower**; shells = `L = D − A` degeneracies = irrep cardinals. | **Yes — the shell tower goes through the grade.** Shell degeneracy **= simplex grade = emergent dimension** (M1, exact: first-excited and modal degeneracy = `m−1`); the first closure **reads the dimension** as `2(grade+1)` (M2: grade 2 → 6, grade 3 → 8, grade 4 → **10** = atomic Ne / SO(4)); the **U(grade)** oscillator magic numbers appear among the closures (grade 2 → `{6,12,…}` = U(2) = 2D quantum dots, matching the substrate's own locked U(2); grade 3 → `{8,20,40}` = U(3) = 3D / nuclear), **mixed with** genuine Sierpinski localized-mode closures (a co-occurrence, not an exclusive tower). The chemical table (`2,10,18,…` = SO(4,2)) still needs two-body screening on this **independent-particle** skeleton. |
| `emergent_atom_dynamics.py` | The atom-forms are *static* spectra — but observable phenomena come from the **nodal dynamics**, and is **coherence C** (canonical, resonant, fractal) being tracked at all? | The nodal equation `∂EPI/∂t = νf·ΔNFR` on the EPI channel (`ΔNFR_epi = −L_rw·EPI`, `structural_diffusion_operator`); the conservative/wave face `u'' = −L·u`; `C = 1/(1 + mean\|ΔNFR\| + mean\|dEPI\|)`. | **The atom is the coherent form *evolving*, not the static spectrum.** **M1** an excited form relaxes, `C(t)` rises 0.44 → 1.0 to the resonant attractor (de-excitation), and the decay rate recovers the shell `λ₂` *exactly* — the static eigenvalue is observable only through the dynamics. **M2** the wave face oscillates (energy conserved) at terms `√λ_k`; the observable lines are their differences (**Rydberg–Ritz** combination). **M3** coupling two forms splits the ground mode into bonding (stays 0) + antibonding (grows with coupling) = the molecular bond. **M4** `C` is defined at every scale (molecule + atoms) — **resonant** (→ ΔNFR=0) and **fractal** (U5 multi-scale). |
| `emergent_nfr_geometry.py` | What determines whether a point is *free of structural pressure* (`ΔNFR=0`), and where do those equilibria fall? | `ΔNFR(i) = neighbour-mean(EPI) − EPI(i) = −(L_rw·EPI)(i)` = the **discrete curvature** of EPI; the canonical NFR predicates `is_structural_equilibrium` / `structural_coherence`; `classify_nodal_topology`. | **`ΔNFR=0` is a geometric condition — and those flat points ARE NFRs.** **M1** `ΔNFR` = curvature *exactly*; the nodal set (`v=0`) registers `is_structural_equilibrium=True`, `C=1.000` (NFRs), antinodes `C=0.81` (pressured). **M2** the **NFR lattice = the Chladni nodal pattern** — mode `k` has `2k` nodal NFRs, ordered by the spectral index (**Courant**): more pressure → more NFRs, geometrically spaced. **M3** the nodal NFRs are **resonant** (stationary fixed points of the standing wave, amplitude `1e-15` ∀t) and **fractal** (THOL nest → multinodal NFR topology, self-similar). **M4** the **combat** (curvature minimisation) collapses the curvature energy; the survivor is the Fiedler mode. For atoms the modes are the shells; for primes "where they fall" = the spectral (Hilbert–Pólya) form of the RH wall, stated geometrically. |
| `emergent_nfr_where.py` | How far does the emergent nodal/spectral order carry the equilibrium *locations* (atom shells, primes) before the `S_n` wall? | The nodal NFRs of a *symmetric* operator are a **regular** lattice (Courant + symmetry); the prime number theorem `π(n) ~ n/log n`; prime-gap irregularity; the `Fix(S_n)^⊥` wall. | **It carries the symmetric/smooth part fully; the wall is the prime fine structure.** **M1** on the ring every mode's nodal NFRs are evenly spaced (constant gap) — symmetry forces a regular lattice (atom shells carried). **M2** the prime **density** `π(n) ~ n/log n` is carried (ratio ≈ constant, 1.12–1.16). **M3** the **individual** prime locations are irregular (gap std/mean ≈ 0.74, gaps 1→52); no constant-gap (symmetric) operator produces this, so placing nodal NFRs at the primes needs **breaking `S_n`** = `Fix(S_n)^⊥` = the Riemann residue `S(T)`. The support partially emerges from the residue spectral-gap geometry; the fine distribution is the wall. |
| `emergent_rhythm.py` | TNFR's essence — *everything vibrates and keeps a rhythm.* Are the equilibria a fixed lattice, or the **beats** of an evolving rhythm? | The dissipative vs conservative faces of the nodal equation; `ω_k = √λ_k`; beat frequencies `ω_j − ω_k`; energy conservation; THOL spectral decimation. | **The equilibria are the beats of a sustained vibration — measured by *evolving*, not by a fixed point.** **M1** the dissipative face decays to silence; the **conservative face vibrates** (energy conserved to `3e-15`, pressure sustained). **M2** a quadratic detector of the vibration **beats at `ω_b − ω_a`** — the rhythm is the interference of the resonances. **M3** the structural pressure pulses and the system **passes through near-flat `ΔNFR~0` (NFR-coherence) states periodically** — the equilibria are the beats, set by structure + dynamics *together*. **M4** the rhythm is **resonant** (`ω_k = √λ_k`) and **fractal** (THOL nest → one frequency repeated 66×, self-similar decimation). Music bridge: atoms → shell modes; primes → the explicit-formula rhythm = RH. |
| `emergent_fractal_pulse.py` | The per-NFR pulse syncs — but on a *nested* network, does resonance lock as one global beat, or **scale by scale**? | The resonant phase channel (random-walk Kuramoto `θ̇_i = νf·⟨sin(θ_j−θ_i)⟩`, small-angle limit `−νf·L_rw·θ`); per-scale order `R = \|⟨e^{iθ}⟩\|`; relaxation rate `νf·λ`; a 3-level ultrametric (self-similar) coupling; the read-outs `net.resonance()` / `net.pulse_trajectory()`. | **Resonance locks fine → coarse, self-similarly.** **M1** the cascade `R_leaf > R_group > R_whole` holds at **every** step — local before global. **M2** the `L_sym` spectrum splits into **3 exact bands**, one per scale (degeneracies `2,6,36` = inter-group / intra-group / intra-block modes). **M3** the per-scale sync time orders as `1/(νf·λ_band)` (leaf `λ≈1.15` locks first, whole `λ≈0.19` last) — the timescales **are** the spectral bands. **M4** the 3 bands are cleanly separated by the geometric coupling ratio `r` — the fractal signature of the nested structure. The collective pulse emerges as the coarsest band finally synchronizes. |
| `emergent_arithmetic_pulse.py` | Introduce the pulse into number theory — what is the *arithmetic* NFR's pulse? | The conservative pulse `ω_k = √λ_k` of the canonical `L_rw` on the residue Cayley network `Cay(ℤ/n, R_k)`; `structural_frequency_rank` (distinct eigenvalues = resonant tones); the **proved** cyclotomy law `s_k(p) = gcd(k, p−1) + 1` (theory §9.11); the `Fix(G)/Fix(G)^⊥` wall (§9.7). | **The pulse tone-count IS the cyclotomy law — a prime is its most degenerate chord.** **M1** the distinct resonant tones = `gcd(k,p−1)+1` exactly (k=2,3,4,5, all primes, 0 mismatches). **M2** the Paley-NFR (`p≡1 mod 4`) chord = the silent mode + **two** tones (`ω_-,ω_+`), each multiplicity `(p−1)/2` (the pulse's `spectral_multiplicity` field). **M3** composites split the chord **multiplicatively** (`15→9=3×3`, `45→12=4×3`) = the factorization type (the §9.8 ladder). **M4** a prime stays minimal (3 tones at any size). The per-NFR pulse is blind (`Fix(G)`), the collective pulse carries the cyclotomy (`Fix(G)^⊥`); it sees the **type**, not the prime identities (the wall persists). |

The rigorous emergent facts are pinned as engine tests in
[`tests/physics/test_emergent_chemistry.py`](../tests/physics/test_emergent_chemistry.py)
(14 tests: the `(2l+1)` angular degeneracy, the emergent radial nucleus, the
`2,8,18` spherical-well closures, the `ΔNFR = 0` closed-shell predicate, and the
non-spectral boundary of the aufbau `(n+l)` postulate).

**Honest synthesis.** Pure single-manifold TNFR structure reproduces the
**independent-particle (mean-field) skeleton** that atoms and nuclei *share* —
the `(2l+1)` multiplets, the central-field shells, the independent-particle magic
`2, 8, 20` — and the substrate's own emergent symmetry is **U(2)** (2D
quantum-dot cardinals), structurally locked. The domain-specific corrections are
the known **two-body** physics that a single coherence manifold cannot carry:
**screening** (→ atomic `2,10,18,…`) and **spin-orbit** (→ nuclear `28,50,82`).
So spatial dimensionality and 3D rotational symmetry are **inputs**, not
predictions — TNFR's emergent geometric substrate is intrinsically a **2D
resonant theory** (the U(2) fibre). The deeper resolution of the dimension
question is that the dimension which truly emerges from a *form* is its **simplex
grade** (= the cardinal it carries): the coherent coupled-NFR cluster (EPI) is an
`n`-simplex of dimension `n`, the dynamics **builds** it by resonant coherent
accretion gated by U3, and that form-grade dimension is **unbounded and
dynamically generated** — a notion distinct from the locked U(2) fibre symmetry.
Closed shell = prime = relaxed node is one `ΔNFR = 0` fixed point across all three
domains. **Honest scope**: these are emergent-ontology falsifiers, not derivations
of the periodic table, the nuclear shell model, or 3D space; the standard
representation theory and spectral geometry are the comparison framework, and
`ℝ` is the assumed continuum and **π** the one assumed structural scale.

## Retired scripts

Older benchmarks covering glyph history trimming, usage counters, or glyph
timing updates have been removed because the optimised paths now mirror the
reference implementations. Keeping them would create maintenance noise without
providing actionable performance signals.

## Profiling workflows

### ΔNFR default pipeline

Run the benchmark with profiling enabled to capture cumulative timings for the
entire ΔNFR pipeline. Choose the output format that best fits your tooling:

```bash
PYTHONPATH=src python benchmarks/default_compute_delta_nfr.py \
  --nodes 320 --edge-probability 0.22 --repeats 3 \
  --profile profiles/dnfr_default.pstats
```

* Use `--profile-format json` to export an ordered JSON array with the
  `totaltime` and `inlinetime` for each function.
* Inspect `.pstats` files via `python -m pstats profiles/dnfr_default.pstats` and
  sort on `cumtime` (cumulative time) or `tottime` (self time) to spot hotspots.

### Sense Index vectorised vs. fallback paths

The profiling script mirrors the setup from `tests/performance/test_sense_performance.py`
to compare vectorised and pure-Python Si computations:

```bash
PYTHONPATH=src python benchmarks/compute_si_profile.py \
  --nodes 512 --loops 8 --format json --output-dir profiles
```

The command writes two files:

* `compute_Si_numpy.*` – profile captured when NumPy is available.
* `compute_Si_python.*` – profile captured with NumPy disabled, exercising the
  fallback path.

Inspect the top entries sorted by `cumtime` (cumulative time per function) to
spot the phases consuming most wall-clock time. Compare both outputs to confirm
that vectorisation shifts time into array primitives rather than Python loops.

### Full pipeline profiling (Si + ΔNFR)

```bash
PYTHONPATH=src python benchmarks/full_pipeline_profile.py \
  --nodes 384 --edge-probability 0.28 --loops 6 --output-dir profiles \
  --si-chunk-sizes auto 2048 --dnfr-chunk-sizes auto 4096
```

The profiler iterates over the Cartesian product of the requested Si and ΔNFR
chunk sizes (and, when provided, worker counts via `--si-workers` and
`--dnfr-workers`). Each combination is tagged as `cfgXX` and encoded in the
artefact name, for example
`full_pipeline_vectorized_cfg01_si_auto_dn_auto_siw_auto_dnw_auto.json`.

For every configuration + execution-mode pair the script writes matching
`.pstats` and `.json` files. The JSON schema now includes:

* `configuration` – the label, ordinal, textual description, and raw knob
  values (`SI_CHUNK_SIZE`, `DNFR_CHUNK_SIZE`, `SI_N_JOBS`, `DNFR_N_JOBS`).
* `metadata` – runtime context covering vectorisation, graph size, and the
  requested/resolved chunk sizes and worker counts applied to the seeded graph.
* `operator_totals` – raw wall-clock totals for each explicit operator call in
  the benchmark loop.
* `operator_timings` – totals and per-loop averages per operator (mirrored under
  `manual_timings` for backwards compatibility).
* `compute_Si_breakdown` – wall-clock totals, per-loop averages, and execution
  path counts for the Sense Index sub-stages (cache rebuilds, vectorised
  neighbour aggregation, normalisation/clamp, and in-place writes). Use the
  `path_counts` entry to verify whether the NumPy kernels or the pure-Python
  fallback handled the run.
* `target_functions` – cumulative profiler statistics (`cumtime`, `totaltime`)
  for the four canonical operators. Use this section to compare how much time
  each function spends (including callees) in vectorised vs. fallback modes.
* `rows` – the complete, sorted profiler table, matching the `.pstats` export.
  Inspect it when a hotspot needs deeper call-tree analysis.

Contrast the vectorised and fallback JSON summaries to confirm that NumPy shifts
most cumulative time from `_compute_dnfr_common` into array-based kernels and
that `compute_Si` benefits from the same optimisation. The console output now
prints the same Sense Index breakdown, making it easier to spot whether cache
reconstruction or neighbour aggregation dominates the runtime. Cross-check the
`path_counts` field—vectorised runs should report NumPy activity while fallback
profiles stay in the Python bucket. Significant regressions should surface as
higher `cumtime` values or inflated per-loop wall-clock figures across both the
top-level operator timings and the Si sub-stages.

### `_compute_dnfr` vectorisation checks

Use the microbenchmark to compare the NumPy and pure-Python branches directly
across multiple graph sizes and densities:

```bash
PYTHONPATH=src python benchmarks/compute_dnfr_benchmark.py \
  --nodes 192 384 768 --edge-probabilities 0.05 0.12 0.3 --repeats 8
```

The output prints a compact table per configuration with:

* **Vectorised best/median/mean/worst** – summary timings in seconds when
  NumPy is available.
* **Fallback best/median/mean/worst** – the pure-Python execution statistics.
* **Ratio (fallback ÷ vectorized)** – the average slow-down when vectorisation
  is disabled. Ratios significantly above `1.00×` confirm that dense kernels are
  exercised and provide a guardrail for regressions.

Pass `--force-dense` to ensure the dense accumulator path is exercised when the
automatic heuristics might prefer sparse accumulation. The script gracefully
degrades to reporting only the fallback timings whenever NumPy is unavailable.

### Chunked execution switches

Both benchmarks honour the new batching knobs exposed by the engine:

* Set `graph.graph["SI_CHUNK_SIZE"] = 2048` (or pass
  `chunk_size=2048` when calling `compute_Si`) inside
  `compute_si_profile.py` to process nodes in deterministic batches. This is
  helpful when profiling large (>10k nodes) graphs on memory-constrained
  machines.
* Set `graph.graph["DNFR_CHUNK_SIZE"] = 4096` before invoking
  `default_compute_delta_nfr` to bound the accumulator size in the ΔNFR
  benchmark. Larger chunks favour throughput, while smaller ones keep the
  temporary buffers inside stricter memory budgets.

Leave both settings unset for medium-sized runs; the automatic heuristics use
CPU availability and a conservative memory estimate to choose a balanced chunk
size.

## Cache Profiling

### Comprehensive Cache Analysis

The `comprehensive_cache_profiler.py` tool tracks buffer allocation effectiveness across all TNFR hot paths:

```bash
PYTHONPATH=src python benchmarks/comprehensive_cache_profiler.py \
  --nodes 200 --steps 50 --buffer-cache-size 256 \
  --output cache_report.json
```

**Key Metrics Reported:**

* **Buffer Reuse Rate**: Should remain near 100% (indicates effective buffer caching)
* **Edge Cache Hit Rate**: Per-hot-path buffer allocation cache effectiveness
* **TNFR Cache Hit Rate**: DNFR preparation state and structural cache hits
* **Cache Entry Count**: Memory usage tracking

**Sample Results** (100 nodes, 20 steps):

* `coherence_matrix`: 97.5% hit rate, 100% buffer reuse ⭐
* `default_compute_delta_nfr`: 96.7% hit rate, 100% buffer reuse ⭐
* `sense_index`: 0.7% hit rate, 100% buffer reuse (expected - creates new structural arrays)
* `dnfr_laplacian`: 0.0% hit rate, 100% buffer reuse (by design - stateless gradients)

For detailed analysis see `ARCHITECTURE.md`.

**Usage Examples:**

```bash
# Basic profiling
python benchmarks/comprehensive_cache_profiler.py --nodes 100 --steps 20

# Detailed per-step metrics
python benchmarks/comprehensive_cache_profiler.py --nodes 200 --steps 50 --verbose

# Export JSON report
python benchmarks/comprehensive_cache_profiler.py \
  --nodes 150 --steps 30 --buffer-cache-size 256 \
  --output cache_analysis.json

# Test different cache sizes
python benchmarks/comprehensive_cache_profiler.py \
  --nodes 500 --steps 100 --buffer-cache-size 512
```

**Interpreting Results:**

1. **Buffer Reuse Rate = 100%** ✅ Optimal - buffers are being reused perfectly
2. **Buffer Reuse Rate < 95%** ⚠️ Investigation needed - possible cache thrashing
3. **High Edge Cache Misses + 100% Buffer Reuse** ✅ Normal for Si/Laplacian (creates new entries but reuses buffers)
4. **High Eviction Rate** ⚠️ Consider increasing `--buffer-cache-size`

The comprehensive profiler supersedes the basic `cache_hot_path_profiler.py` by tracking all cache layers.

