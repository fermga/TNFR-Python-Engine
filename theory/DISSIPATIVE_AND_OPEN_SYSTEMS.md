# Dissipative and Open-System Diagnostics

**Status:** scoped auxiliary model with exact finite-dimensional identities and
numerical trajectory diagnostics.

This note describes the density-operator tools in
`src/tnfr/physics/dissipative_conservation.py`. The matrix density operator
`rho` used by these tools is distinct from the structural density
`rho_s = Phi_s + K_phi` used by the graph-field conservation diagnostics. No
implemented map identifies the two objects.

The module therefore does not extend the TNFR structural balance to all open
systems as a theorem. It supplies a conventional finite-dimensional GKSL model
that can be studied alongside TNFR telemetry. Likewise, a collapse operator is
an environmental-model input; it is not evidence of a U1--U6 grammar violation.

---

## 1. GKSL generator

For a density operator `rho`, Hamiltonian `H`, and collapse operators `L_k`,

$$
\dot\rho=-i[H,\rho]+D[\rho],
\qquad
D[\rho]=\sum_k\left(
L_k\rho L_k^\dagger-
\frac12\{L_k^\dagger L_k,\rho\}
\right).
$$

The dissipator is trace preserving because

$$
\operatorname{Tr}D[\rho]=0.
$$

When the generator is constructed by `build_lindblad_delta_nfr`, exponentiating
it gives a completely positive trace-preserving semigroup, up to numerical
error. Spectral non-expansion of an arbitrary matrix is weaker than the GKSL
conditions and does not independently prove complete positivity.

### Unitality

The dissipative part is unital precisely when

$$
D[I]=\sum_k(L_kL_k^\dagger-L_k^\dagger L_k)=0.
$$

`is_unital_dissipator` checks this residual numerically. Pure dephasing is
unital. Zero-temperature amplitude damping is not: it drives states toward the
pure ground state.

All functions that accept collapse operators interpret them as the effective
operators of the modeled generator. If a Liouville generator is multiplied by
a positive factor `a = nu_f * scale`, the equivalent dissipative inputs are
`sqrt(a) L_k`; passing the unscaled `L_k` would under-report instantaneous rates.

---

## 2. Universal norm bounds

For the Frobenius norm on states and spectral norm on collapse operators,
submultiplicativity gives, term by term,

$$
\|D[\rho]\|_F
\le 2\sum_k\|L_k\|_2^2\,\|\rho\|_F
=2\sum_k\|L_k\|_2^2\sqrt{P},
\qquad P=\operatorname{Tr}(\rho^2).
$$

This is the bound returned by `compute_dissipation_bound`. It is universal but
usually loose. It does not vanish for every pure state. For example, with
`L=sqrt(gamma)|0><1|`,

$$
D[|1\rangle\langle1|]
=\gamma\left(|0\rangle\langle0|-|1\rangle\langle1|\right)\ne0.
$$

Combining the first bound with Cauchy--Schwarz yields

$$
\left|\frac{dP}{dt}\right|
=\left|2\operatorname{Tr}(\rho D[\rho])\right|
\le4\sum_k\|L_k\|_2^2P.
$$

`compute_instantaneous_purity_rate` evaluates the exact signed dissipative
contribution. `compute_purity_decay_bound` retains its historical public name,
but now returns the valid bound on the absolute purity-change rate.

---

## 3. Purity and entropy require channel hypotheses

The Hamiltonian commutator does not change purity, so

$$
\frac{dP}{dt}=2\operatorname{Tr}(\rho D[\rho]).
$$

Its sign is not fixed for a general GKSL generator. For a unital semigroup,
purity is non-increasing and von Neumann entropy

$$
S(\rho)=-\operatorname{Tr}(\rho\log\rho)
$$

is non-decreasing. Without unitality, both directions can occur.

### Amplitude damping

For a qubit with `L=sqrt(gamma)|0><1|` and
`eta=exp(-gamma t)`, write the initial state as

$$
\rho(0)=\begin{pmatrix}1-p&c\\c^*&p\end{pmatrix}.
$$

Then

$$
\rho_{11}(t)=\eta p,\qquad
\rho_{01}(t)=\sqrt\eta\,c,
$$

and

$$
P(t)=(1-\eta p)^2+(\eta p)^2+2\eta|c|^2.
$$

An initially excited pure state first loses purity, reaches `P=1/2`, and then
regains purity as it approaches the pure ground state. Initial purity alone is
insufficient to predict this curve: two states with equal purity can have
different `p` and `c`. `predict_amplitude_damping_purity` therefore gives an
exact result only when passed the full `2 x 2` density matrix. Scalar input is a
deprecated compatibility interpolation and emits `DeprecationWarning`.

### Pure dephasing

When `gamma` denotes the off-diagonal amplitude decay rate,

$$
\rho_{ij}(t)=e^{-\gamma t}\rho_{ij}(0)\quad(i\ne j),
$$

while populations remain fixed. Consequently,

$$
P(t)=\sum_i|\rho_{ii}(0)|^2
+e^{-2\gamma t}\sum_{i\ne j}|\rho_{ij}(0)|^2.
$$

This is the convention used by `predict_dephasing_purity`. It matches both the
qubit operator `sqrt(gamma/2) sigma_z` and the complete projector family
`sqrt(gamma)|k><k|`.

---

## 4. Contractivity and stationary states

A CPTP map contracts trace distance between two states:

$$
T(\mathcal E(\rho),\mathcal E(\sigma))
\le T(\rho,\sigma),
\qquad
T(\rho,\sigma)=\frac12\|\rho-\sigma\|_1.
$$

If `rho_ss` is fixed by the same map, its trace distance from a trajectory is
therefore non-increasing. `verify_dissipative_balance` reports the ratio of
these trace distances when a reference state is supplied through the historical
`steady_state` parameter. The standalone balance function has no generator and
cannot verify that the reference is actually stationary. Without a reference,
the result is explicitly unevaluated; the function does not silently certify
contractivity. Arbitrary pairs of snapshots also do not prove that an
underlying map is CPTP. The tracker, which does own a generator, rejects a
supplied steady state with a nonzero stationarity residual.

`steady_state_from_generator` solves the stationary and trace-one constraints
by least squares and verifies positivity and the generator residual. A
generator can have a stationary manifold. Pure dephasing, for example, fixes
all density matrices diagonal in the dephasing basis. The returned state is one
minimum-norm representative, not a uniqueness certificate.

`analyze_dissipation_rates` reports the stationary-mode count, stable decay
rates, positive-real-part modes, and the trace-preservation residual. A positive
spectral gap measures relaxation toward the stationary subspace. Convergence
to one unique state additionally requires a one-dimensional stationary space.

---

## 5. What each diagnostic means

| Diagnostic | Implemented quantity | Scope |
|---|---|---|
| Purity change rate | `(P_after-P_before)/dt` | Signed; either sign is possible |
| Entropy change rate | `(S_after-S_before)/dt` | Signed; monotone for unital channels |
| State change rate | `||rho_after-rho_before||_F/dt` | Includes coherent and dissipative motion |
| Dissipator action norm | `||D[rho_before]||_F` | Instantaneous environmental term |
| Dissipation bound | `2 sum ||L_k||_2^2 sqrt(P)` | Universal Frobenius bound |
| Fixed-point contraction ratio | `T_after/T_before` | Requires a supplied stationary state |
| Frobenius-norm loss rate | `(||rho_before||_F-||rho_after||_F)/dt` | Purity proxy; not a Noether charge |
| Change tier | Maximum absolute purity/entropy rate | Empirical reporting bin |

The compatibility names `purity_decay_rate`, `entropy_production_rate`,
`actual_dissipation`, and `charge_leak_rate` remain accessible. Their canonical
counterparts state the measured quantity without assuming a sign or a Noether
interpretation.

The four change tiers (`weak`, `moderate`, `strong`, `large`) use configured
rate thresholds `0.001`, `0.05`, and `0.2`. They are descriptive bins rather
than physical phase boundaries. `classify_dissipative_regime` retains legacy
keys for compatibility and always reports `grammar_status_inferred=False`.

---

## 6. Validation and reproducibility

Public calculations reject density matrices that are non-square,
non-Hermitian, non-finite, non-positive, or not trace one. Collapse operators
must be finite square matrices of the same dimension. Time steps, rates, and
times must be finite and non-negative where their definitions require it.

The executable demonstration compares exact amplitude damping and dephasing
channels, including the non-monotone purity of amplitude damping and the
monotone mixing of unital dephasing:

- [28_dissipative_systems_demo.py](../examples/02_physics_regimes/28_dissipative_systems_demo.py)

Direct regression coverage:

- [test_dissipative_conservation.py](../tests/physics/test_dissipative_conservation.py)

Related implementation:

- [dissipative_conservation.py](../src/tnfr/physics/dissipative_conservation.py)
- [generators.py](../src/tnfr/mathematics/generators.py)
- [dynamics.py](../src/tnfr/mathematics/dynamics.py)

TNFR scope references:

- [DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
