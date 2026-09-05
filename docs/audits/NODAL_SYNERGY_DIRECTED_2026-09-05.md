# Nodal Synergy: Shared Directed Transition

**Scope:** The outgoing EPI transport matrix and the stationary measure now
share one validated transition in
[directed_diffusion.py](../../src/tnfr/physics/directed_diffusion.py).
This is a bounded implementation of an existing algebraic relationship:

    P_ij = W_ij / Σ_j W_ij for a positive row,
    P_ii = 1 for a zero-strength row,
    L_rw = I − P,
    EPI' = −diag(ν_f) L_rw EPI,
    π P = π,    Σ_i π_i = 1.

The nodal equation assigns frequency to the velocity, not to the geometric
transition. No EPI evolution, operator set, grammar constant, or generalized
U2 theorem is changed here.

## Problem and correction

The two public matrix readers previously normalized independently by forming
reciprocal row strengths. Subnormal positive conductance can overflow that
reciprocal even when its transition probabilities are well-defined. The raw
Laplacian API also lacked the graph API's finite/nonnegative validation.
For W=[[0,-1],[1,0]] it returned [[1,0],[-1,1]], including a spurious nonzero
row sum rather than rejecting negative conductance.

The stationary reader separately used D⁻¹W with zero sink rows, whereas the
Laplacian treated sinks as frozen. On W=[[0]], it returned π=[1] although its
documented raw transition was zero, with stationary residual 1. The coherent
absorbing convention is P=I−L; it gives L=[[0]], P=[[1]], and π=[1].

The shared private helper now:

- Validates a square real matrix with finite nonnegative conductance.
- Normalizes by direct row division, first scaling each nonzero row by its
  largest entry. This avoids both reciprocal overflow for tiny conductance
  and overflow in sums of large finite conductances.
- Sets zero-strength rows to absorbing self-loops.
- Returns detached storage without mutating the input.

The stationary API retains its strictly positive-measure requirement. It
rejects sink-supported distributions that would make its weighted norm
degenerate, accepts the singleton absorbing walk, rejects an empty state
space, and checks the selected eigenvalue and stationary residual. Invalid
stationary tolerances cannot disable the gate. It does not introduce a
general reducible-chain stationary-distribution API.

## Interpretation corrections

For symmetric conductance, L_rw is similar to a symmetric normalized
Laplacian on positive-strength nodes; L_rw itself need not be symmetric or
normal in the Euclidean metric. A sink is fixed by outgoing diffusion but
can influence a node with an arc pointing to it. For the single edge 0→1,
EPI=[0,1], and ν_f=[2,7], the derivative is [2,0].

The existing certificate's asymptotically_stable field retains its value
semantics, abscissa≤tolerance. Its documentation now identifies that legacy
flag as absence of positive spectral growth rather than full-state decay to
zero; diffusion retains stationary modes. No new transient or U2 claim is
made by this refactor.

## Verification

The parent task's pre-edit baseline passed **3,218 tests**. The focused command

    .venv312/Scripts/python.exe -m pytest tests/physics/test_directed_transition_consistency.py tests/physics/test_directed_diffusion.py tests/physics/test_directed_metrics.py tests/physics/test_structural_time.py tests/physics/test_transient_u2.py -q

passed **85 tests**, including **26 new cases**, in 6.14 seconds. Coverage
includes invalid matrices, subnormal scaling, finite inputs with overflowing
raw row sums, absorbing singleton stationarity, directed sink influence,
strict positivity, input purity, and degree similarity. Existing directed
metric, structural-time, and transient-U2 behavior remains covered.

For the exact rational fixture

    W = [[0,2,1], [1,0,1], [1,3,0]],
    π = [15,22,16]/53,

the measured errors under uniform conductance rescaling were:

| Conductance scale | Maximum transition row-sum error | Maximum stationary residual |
|-------------------|----------------------------------|-----------------------------|
| 5e-324 | 0 | 2.7755575615628914e-16 |
| 1e-310 | 0 | 2.7755575615628914e-16 |
| 1 | 0 | 2.7755575615628914e-16 |
| 1e307 | 1.1102230246251565e-16 | 2.220446049250313e-16 |

All scale checks completed with floating-point warnings promoted to errors.
The finite matrix [[1e308,1e308],[1e308,0]] also yields the independent expected
transition [[1/2,1/2],[1,0]] and stationary measure [2/3,1/3]. Whitespace checks
passed. No stochastic operator sequence or before/after coherence improvement
is claimed: these are matrix-readout consistency checks.

## Changed files

- [directed_diffusion.py](../../src/tnfr/physics/directed_diffusion.py)
- [test_directed_transition_consistency.py](../../tests/physics/test_directed_transition_consistency.py)
- This audit note.

The independent sparse conductance/current/Dirichlet refactor is documented
by its owning task. This change adds no dependency and does not alter the
canonical multi-channel pressure implementation.
