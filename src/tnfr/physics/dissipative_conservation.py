"""Diagnostics for finite-dimensional Lindblad open-system models.

This module implements the Gorini--Kossakowski--Sudarshan--Lindblad (GKSL)
dissipator and keeps three statements separate:

* ``D[rho]`` is an exact algebraic term of a specified GKSL model;
* trace-distance contraction follows for a CPTP evolution when both states are
  evolved by the same map (and hence relative to a fixed point);
* monotone purity loss and entropy growth require an additional unitality
  hypothesis. Amplitude damping is non-unital and can purify a state.

The density operator here is an auxiliary finite-dimensional model. It is not
the structural density ``Phi_s + K_phi`` from ``conservation.py``. Collapse
operators are environmental-model inputs and do not diagnose or encode TNFR
grammar violations.

For a density operator ``rho`` and collapse operators ``L_k``,

    D[rho] = sum_k (L_k rho L_k^dagger
                    - 1/2 {L_k^dagger L_k, rho}).

The universal Frobenius-norm bounds used by this module are

    ||D[rho]||_F <= 2 sum_k ||L_k||_2^2 sqrt(Tr(rho^2)),
    |d Tr(rho^2)/dt| <= 4 sum_k ||L_k||_2^2 Tr(rho^2).

They remain nonzero for a generic pure state, as required: a pure excited
state is not stationary under amplitude damping.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..mathematics.unified_numerical import np

try:
    from ..mathematics.backend import ensure_numpy
except ImportError:  # pragma: no cover - optional mathematics layer
    ensure_numpy = None  # type: ignore[assignment]


_WEAK_CHANGE_RATE_THRESHOLD = 0.001
_MODERATE_CHANGE_RATE_THRESHOLD = 0.05
_STRONG_CHANGE_RATE_THRESHOLD = 0.2
_DEFAULT_ATOL = 1e-9
_MISSING_ARGUMENT = object()


@dataclass(frozen=True)
class DissipativeSnapshot:
    """Validated density operator and its scalar state diagnostics."""

    density: Any
    trace: float
    purity: float
    von_neumann_entropy: float
    eigenvalues: Any


@dataclass(frozen=True)
class DissipativeBalance:
    """Two-snapshot diagnostics for an open-system trajectory segment.

    ``purity_change_rate`` and ``entropy_change_rate`` are signed. Their signs
    are unrestricted for a general GKSL generator. ``contractivity_gap`` is a
    ratio of trace distances to the supplied fixed point; it is NaN when no
    fixed point was supplied. ``state_change_rate`` includes every contribution
    present in the trajectory, while ``dissipator_action_norm`` is the
    instantaneous ``||D[rho_before]||_F`` for supplied collapse operators.

    The historical field names remain the stored dataclass schema so existing
    constructors, ``dataclasses.asdict`` payloads and ``dataclasses.replace``
    calls keep working.  The explicit signed/action names below are semantic
    aliases, while diagnostics added after that schema have safe defaults.
    """

    purity_before: float
    purity_after: float
    purity_decay_rate: float
    entropy_before: float
    entropy_after: float
    entropy_production_rate: float
    trace_drift: float
    dissipation_bound: float
    actual_dissipation: float
    charge_leak_rate: float
    contractivity_gap: float
    is_contractive: bool
    state_change_rate: float = math.nan
    dissipation_bound_satisfied: bool | None = None
    instantaneous_purity_change_rate: float = math.nan
    contractivity_evaluated: bool = False
    unital_dissipator: bool | None = None

    @property
    def purity_change_rate(self) -> float:
        """Signed purity change rate stored under its historical field name."""

        return self.purity_decay_rate

    @property
    def entropy_change_rate(self) -> float:
        """Signed entropy change rate stored under its historical field name."""

        return self.entropy_production_rate

    @property
    def dissipator_action_norm(self) -> float:
        """Instantaneous dissipator norm stored in the historical field."""

        return self.actual_dissipation

    @property
    def frobenius_norm_loss_rate(self) -> float:
        """Frobenius-norm loss historically labelled as charge leakage."""

        return self.charge_leak_rate


@dataclass
class DissipativeTimeSeries:
    """Time series of validated open-system state diagnostics.

    Historical rate names remain stored fields for dataclass compatibility;
    the signed ``*_change_rate`` names are read/write semantic aliases.
    """

    times: list[float] = field(default_factory=list)
    purity: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    trace_drift: list[float] = field(default_factory=list)
    purity_decay_rate: list[float] = field(default_factory=list)
    entropy_production_rate: list[float] = field(default_factory=list)
    dissipation_bound: list[float] = field(default_factory=list)
    contractivity_gap: list[float] = field(default_factory=list)

    @property
    def is_contractive(self) -> bool:
        """Whether every evaluated trace-distance ratio is at most one."""

        evaluated = [
            gap for gap in self.contractivity_gap if not math.isnan(gap)
        ]
        return bool(evaluated) and all(
            math.isfinite(gap) and gap <= 1.0 + _DEFAULT_ATOL
            for gap in evaluated
        )

    @property
    def mean_purity_change(self) -> float:
        """Average signed purity change rate across recorded steps."""

        if not self.purity_change_rate:
            return 0.0
        return float(np.mean(self.purity_change_rate))

    @property
    def total_entropy_change(self) -> float:
        """Final minus initial von Neumann entropy."""

        if len(self.entropy) < 2:
            return 0.0
        return self.entropy[-1] - self.entropy[0]

    @property
    def purity_change_rate(self) -> list[float]:
        """Signed purity-change rates stored under the historical field name."""

        return self.purity_decay_rate

    @purity_change_rate.setter
    def purity_change_rate(self, values: list[float]) -> None:
        self.purity_decay_rate = values

    @property
    def entropy_change_rate(self) -> list[float]:
        """Signed entropy-change rates stored under the historical field name."""

        return self.entropy_production_rate

    @entropy_change_rate.setter
    def entropy_change_rate(self, values: list[float]) -> None:
        self.entropy_production_rate = values

    @property
    def mean_purity_decay(self) -> float:
        """Compatibility alias for ``mean_purity_change``."""

        return self.mean_purity_change

    @property
    def total_entropy_produced(self) -> float:
        """Compatibility alias for ``total_entropy_change``."""

        return self.total_entropy_change


def _as_complex(matrix: Any) -> np.ndarray:
    """Coerce one array-like value to a finite complex128 array."""

    try:
        result = np.asarray(matrix, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a numeric matrix.") from exc
    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise ValueError("Matrices must contain only finite values.")
    return result


def _validate_nonnegative_finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value.")
    return number


def _validate_density_matrix(
    density: Any, *, atol: float = _DEFAULT_ATOL
) -> np.ndarray:
    """Return a Hermitian density matrix after strict physical validation."""

    atol = _validate_nonnegative_finite(atol, name="atol")
    rho = _as_complex(density)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] == 0:
        raise ValueError("density must be a non-empty square matrix.")
    if not np.allclose(rho, rho.conj().T, atol=atol, rtol=0.0):
        raise ValueError("density must be Hermitian within tolerance.")
    rho = 0.5 * (rho + rho.conj().T)
    trace_value = np.trace(rho)
    if not np.isfinite(trace_value):
        raise ValueError("density trace must be finite.")
    if abs(float(trace_value.imag)) > atol:
        raise ValueError("density must have a real trace.")
    if not math.isclose(
        float(trace_value.real), 1.0, abs_tol=atol, rel_tol=0.0
    ):
        raise ValueError("density must have unit trace within tolerance.")
    eigenvalues = np.linalg.eigvalsh(rho)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("density spectrum must be finite.")
    if float(np.min(eigenvalues)) < -atol:
        raise ValueError("density must be positive semidefinite within tolerance.")
    return rho


def _validate_collapse_operators(
    collapse_operators: Sequence[Any],
    *,
    expected_dim: int | None = None,
) -> list[np.ndarray]:
    """Validate finite, square collapse operators with a common dimension."""

    try:
        raw_operators = list(collapse_operators)
    except TypeError as exc:
        raise ValueError("collapse_operators must be a finite sequence.") from exc
    validated: list[np.ndarray] = []
    dimension = expected_dim
    for index, operator in enumerate(raw_operators):
        array = _as_complex(operator)
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
            raise ValueError(
                f"collapse operator[{index}] must be a non-empty square matrix."
            )
        if dimension is None:
            dimension = int(array.shape[0])
        if array.shape != (dimension, dimension):
            raise ValueError(
                f"collapse operator[{index}] has shape {array.shape}; expected "
                f"{(dimension, dimension)}."
            )
        spectral_norm = float(np.linalg.norm(array, ord=2))
        if (
            not math.isfinite(spectral_norm)
            or spectral_norm > math.sqrt(np.finfo(float).max)
        ):
            raise ValueError(
                f"collapse operator[{index}] is too large for finite products."
            )
        validated.append(array)
    return validated


def _collapse_operator_norms_sq(collapse_operators: Sequence[Any]) -> float:
    """Return ``sum_k ||L_k||_2^2`` using spectral operator norms."""

    operators = _validate_collapse_operators(collapse_operators)
    return sum(
        float(np.linalg.norm(operator, ord=2) ** 2) for operator in operators
    )


def capture_dissipative_snapshot(
    density: Any,
    *,
    atol: float = _DEFAULT_ATOL,
) -> DissipativeSnapshot:
    """Validate a density operator and capture trace, purity and entropy."""

    rho = _validate_density_matrix(density, atol=atol)
    eigenvalues = np.linalg.eigvalsh(rho)
    probabilities = np.clip(eigenvalues.real, 0.0, None)
    positive = probabilities[probabilities > 0.0]
    entropy = max(0.0, -float(np.sum(positive * np.log(positive))))
    return DissipativeSnapshot(
        density=rho,
        trace=float(np.trace(rho).real),
        purity=float(np.trace(rho @ rho).real),
        von_neumann_entropy=entropy,
        eigenvalues=eigenvalues,
    )


def compute_dissipator_action(
    density: Any,
    collapse_operators: Sequence[Any],
) -> np.ndarray:
    r"""Compute ``D[rho]`` for validated density and collapse operators."""

    rho = _validate_density_matrix(density)
    operators = _validate_collapse_operators(
        collapse_operators, expected_dim=int(rho.shape[0])
    )
    result = np.zeros_like(rho)
    for operator in operators:
        adjoint_product = operator.conj().T @ operator
        result += operator @ rho @ operator.conj().T
        result -= 0.5 * (adjoint_product @ rho + rho @ adjoint_product)
    return result


def compute_dissipation_bound(
    collapse_operators: Sequence[Any],
    purity: float,
) -> float:
    r"""Bound the Frobenius norm of a GKSL dissipator action.

    ``||D[rho]||_F <= 2 sum_k ||L_k||_2^2 sqrt(P)`` for purity ``P``.
    The bound is universal but generally not tight and does not vanish merely
    because a state is pure.
    """

    purity_value = float(purity)
    if (
        not math.isfinite(purity_value)
        or purity_value < 0.0
        or purity_value > 1.0 + _DEFAULT_ATOL
    ):
        raise ValueError("purity must be finite and lie in [0, 1].")
    purity_value = min(1.0, max(0.0, purity_value))
    return (
        2.0
        * _collapse_operator_norms_sq(collapse_operators)
        * math.sqrt(purity_value)
    )


def compute_instantaneous_purity_rate(
    collapse_operators: Sequence[Any],
    density: Any,
) -> float:
    r"""Return the exact dissipative contribution ``2 Tr(rho D[rho])``."""

    rho = _validate_density_matrix(density)
    dissipator = compute_dissipator_action(rho, collapse_operators)
    return float(2.0 * np.trace(rho @ dissipator).real)


def compute_purity_decay_bound(
    collapse_operators: Sequence[Any],
    density: Any,
) -> float:
    r"""Bound the absolute instantaneous purity-change rate.

    The historical name is retained for compatibility. The derivative can be
    positive for a non-unital generator. The implemented universal bound is
    ``|dP/dt| <= 4 sum_k ||L_k||_2^2 P``.
    """

    snapshot = capture_dissipative_snapshot(density)
    operators = _validate_collapse_operators(
        collapse_operators, expected_dim=int(snapshot.density.shape[0])
    )
    return 4.0 * _collapse_operator_norms_sq(operators) * snapshot.purity


def is_unital_dissipator(
    collapse_operators: Sequence[Any],
    *,
    tolerance: float = _DEFAULT_ATOL,
) -> bool:
    r"""Test ``D[I] = sum_k (L_k L_k^dagger - L_k^dagger L_k) = 0``."""

    tolerance_value = _validate_nonnegative_finite(tolerance, name="tolerance")
    operators = _validate_collapse_operators(collapse_operators)
    if not operators:
        return True
    residual = np.zeros_like(operators[0])
    scale = 1.0
    for operator in operators:
        residual += operator @ operator.conj().T - operator.conj().T @ operator
        scale += float(np.linalg.norm(operator, ord=2) ** 2)
    return bool(
        np.linalg.norm(residual, ord="fro") <= tolerance_value * scale
    )


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``1/2 ||left-right||_1``."""

    return 0.5 * float(np.linalg.norm(left - right, ord="nuc"))


def verify_dissipative_balance(
    before: DissipativeSnapshot,
    after: DissipativeSnapshot,
    dt: float = 1.0,
    collapse_operators: Sequence[Any] | None = None,
    steady_state: Any | None = None,
) -> DissipativeBalance:
    """Compute scoped diagnostics between two validated snapshots.

    This does not prove that the snapshots came from a CPTP map. If a reference
    state is supplied through the historical ``steady_state`` parameter,
    ``is_contractive`` reports whether this observed segment decreased trace
    distance to it. This standalone function cannot verify that the reference
    is actually stationary because it receives no generator.
    """

    time_step = _validate_nonnegative_finite(dt, name="dt")
    if time_step == 0.0:
        raise ValueError("dt must be positive.")
    before_checked = capture_dissipative_snapshot(before.density)
    after_checked = capture_dissipative_snapshot(after.density)
    rho_before = before_checked.density
    rho_after = after_checked.density
    if rho_before.shape != rho_after.shape:
        raise ValueError("before and after density operators must have the same shape.")

    purity_rate = (after_checked.purity - before_checked.purity) / time_step
    entropy_rate = (
        after_checked.von_neumann_entropy - before_checked.von_neumann_entropy
    ) / time_step
    state_change_rate = (
        float(np.linalg.norm(rho_after - rho_before, ord="fro")) / time_step
    )
    norm_loss_rate = (
        float(np.linalg.norm(rho_before, ord="fro"))
        - float(np.linalg.norm(rho_after, ord="fro"))
    ) / time_step

    if collapse_operators is None:
        bound = float("nan")
        action_norm = float("nan")
        bound_satisfied = None
        instantaneous_purity_rate = float("nan")
        unital = None
    else:
        operators = _validate_collapse_operators(
            collapse_operators, expected_dim=int(rho_before.shape[0])
        )
        action = compute_dissipator_action(rho_before, operators)
        action_norm = float(np.linalg.norm(action, ord="fro"))
        bound = compute_dissipation_bound(operators, before_checked.purity)
        bound_satisfied = action_norm <= bound + _DEFAULT_ATOL * max(1.0, bound)
        instantaneous_purity_rate = float(
            2.0 * np.trace(rho_before @ action).real
        )
        unital = is_unital_dissipator(operators)

    if steady_state is None:
        contractivity_gap = float("nan")
        contractivity_evaluated = False
        contractive = False
    else:
        stationary = _validate_density_matrix(steady_state)
        if stationary.shape != rho_before.shape:
            raise ValueError("steady_state must match the snapshot dimension.")
        distance_before = _trace_distance(rho_before, stationary)
        distance_after = _trace_distance(rho_after, stationary)
        if distance_before > _DEFAULT_ATOL:
            contractivity_gap = distance_after / distance_before
        else:
            contractivity_gap = (
                0.0 if distance_after <= _DEFAULT_ATOL else float("inf")
            )
        contractivity_evaluated = True
        contractive = contractivity_gap <= 1.0 + _DEFAULT_ATOL

    return DissipativeBalance(
        purity_before=before_checked.purity,
        purity_after=after_checked.purity,
        purity_decay_rate=purity_rate,
        entropy_before=before_checked.von_neumann_entropy,
        entropy_after=after_checked.von_neumann_entropy,
        entropy_production_rate=entropy_rate,
        trace_drift=max(
            abs(before_checked.trace - 1.0), abs(after_checked.trace - 1.0)
        ),
        dissipation_bound=bound,
        actual_dissipation=action_norm,
        charge_leak_rate=norm_loss_rate,
        contractivity_gap=contractivity_gap,
        is_contractive=contractive,
        state_change_rate=state_change_rate,
        dissipation_bound_satisfied=bound_satisfied,
        instantaneous_purity_change_rate=instantaneous_purity_rate,
        contractivity_evaluated=contractivity_evaluated,
        unital_dissipator=unital,
    )


def _validate_generator(generator: Any, dim: int) -> np.ndarray:
    """Validate a finite Liouville-space generator."""

    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim <= 0:
        raise ValueError("dim must be a positive integer.")
    matrix = _as_complex(generator)
    expected = int(dim) ** 2
    if matrix.shape != (expected, expected):
        raise ValueError(f"generator must have shape {(expected, expected)}.")
    return matrix


def _validate_stationary_state(
    generator: np.ndarray,
    density: Any,
    *,
    atol: float = 100.0 * _DEFAULT_ATOL,
) -> np.ndarray:
    """Validate a density state and its numerical stationarity residual."""

    stationary = _validate_density_matrix(density, atol=atol)
    dimension = int(stationary.shape[0])
    if generator.shape != (dimension * dimension, dimension * dimension):
        raise ValueError("steady state must match the generator dimension.")
    vector = stationary.reshape(dimension * dimension, order="F")
    residual = float(np.linalg.norm(generator @ vector))
    scale = max(1.0, float(np.linalg.norm(generator, ord=2)))
    if residual > atol * scale:
        raise ValueError("steady state is not stationary for the supplied generator.")
    return stationary


def _steady_state_from_generator(generator: Any, dim: int) -> np.ndarray:
    """Compute one physical stationary state by trace-constrained least squares.

    A generator can have several stationary states. The least-squares solution
    selects one minimum-norm trace-one kernel representative; uniqueness is not
    claimed.
    """

    matrix = _validate_generator(generator, dim)
    trace_row = np.eye(dim, dtype=np.complex128).reshape(dim * dim, order="F")
    trace_residual = float(np.linalg.norm(trace_row.conj().T @ matrix))
    generator_scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    if trace_residual > 100.0 * _DEFAULT_ATOL * generator_scale:
        raise ValueError("generator is not trace preserving within tolerance.")
    augmented = np.vstack([matrix, trace_row[np.newaxis, :]])
    target = np.zeros(dim * dim + 1, dtype=np.complex128)
    target[-1] = 1.0
    vector, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    rho = vector.reshape((dim, dim), order="F")
    rho = 0.5 * (rho + rho.conj().T)
    trace_value = np.trace(rho)
    if abs(trace_value) <= _DEFAULT_ATOL:
        raise ValueError("generator did not yield a trace-one stationary candidate.")
    rho = rho / trace_value
    residual = float(
        np.linalg.norm(matrix @ rho.reshape(dim * dim, order="F"))
    )
    if residual > 100.0 * _DEFAULT_ATOL * generator_scale:
        raise ValueError("generator has no numerically resolved stationary state.")
    try:
        return _validate_density_matrix(rho, atol=100.0 * _DEFAULT_ATOL)
    except ValueError as exc:
        raise ValueError(
            "stationary kernel candidate is not a physical density state."
        ) from exc


steady_state_from_generator = _steady_state_from_generator


class DissipativeConservationTracker:
    """Track density-state diagnostics along an engine trajectory."""

    def __init__(
        self,
        engine: Any,
        *,
        collapse_operators: Sequence[Any] | None = None,
        steady_state: Any | None = None,
    ) -> None:
        if not hasattr(engine, "generator") or not hasattr(engine, "hilbert_space"):
            raise ValueError("engine must expose generator and hilbert_space.")
        dimension = int(engine.hilbert_space.dimension)
        generator = _validate_generator(engine.generator, dimension)
        self._engine = engine
        self._collapse_operators = _validate_collapse_operators(
            collapse_operators or [], expected_dim=dimension
        )
        self._steady_state = (
            _validate_stationary_state(generator, steady_state)
            if steady_state is not None
            else None
        )
        if (
            self._steady_state is not None
            and self._steady_state.shape != (dimension, dimension)
        ):
            raise ValueError(
                "steady_state must match the engine Hilbert-space dimension."
            )
        self._series = DissipativeTimeSeries()
        self._snapshots: list[tuple[float, DissipativeSnapshot]] = []

    @property
    def steady_state(self) -> Any | None:
        return self._steady_state

    def set_steady_state(self, rho_ss: Any) -> None:
        dimension = int(self._engine.hilbert_space.dimension)
        generator = _validate_generator(self._engine.generator, dimension)
        stationary = _validate_stationary_state(generator, rho_ss)
        if stationary.shape != (dimension, dimension):
            raise ValueError(
                "steady state must match the engine Hilbert-space dimension."
            )
        self._steady_state = stationary

    def compute_steady_state(self) -> np.ndarray:
        dimension = int(self._engine.hilbert_space.dimension)
        stationary = _steady_state_from_generator(self._engine.generator, dimension)
        self._steady_state = stationary
        return stationary

    def record(self, density: Any, t: float = 0.0) -> DissipativeSnapshot:
        timestamp = float(t)
        if not math.isfinite(timestamp):
            raise ValueError("t must be finite.")
        if self._snapshots and timestamp <= self._snapshots[-1][0]:
            raise ValueError("snapshot times must be strictly increasing.")
        snapshot = capture_dissipative_snapshot(
            density, atol=100.0 * _DEFAULT_ATOL
        )
        expected = int(self._engine.hilbert_space.dimension)
        if snapshot.density.shape != (expected, expected):
            raise ValueError(
                "density must match the engine Hilbert-space dimension."
            )
        self._snapshots.append((timestamp, snapshot))

        if len(self._snapshots) == 1:
            trace_drift = abs(snapshot.trace - 1.0)
            purity_rate = 0.0
            entropy_rate = 0.0
            bound = (
                compute_dissipation_bound(
                    self._collapse_operators, snapshot.purity
                )
                if self._collapse_operators
                else float("nan")
            )
            contractivity_gap = float("nan")
        else:
            previous_time, previous = self._snapshots[-2]
            balance = verify_dissipative_balance(
                previous,
                snapshot,
                dt=timestamp - previous_time,
                collapse_operators=self._collapse_operators or None,
                steady_state=self._steady_state,
            )
            trace_drift = balance.trace_drift
            purity_rate = balance.purity_change_rate
            entropy_rate = balance.entropy_change_rate
            bound = balance.dissipation_bound
            contractivity_gap = balance.contractivity_gap

        series = self._series
        series.times.append(timestamp)
        series.purity.append(snapshot.purity)
        series.entropy.append(snapshot.von_neumann_entropy)
        series.trace_drift.append(trace_drift)
        series.purity_change_rate.append(purity_rate)
        series.entropy_change_rate.append(entropy_rate)
        series.dissipation_bound.append(bound)
        series.contractivity_gap.append(contractivity_gap)
        return snapshot

    def evolve_and_track(
        self,
        initial_density: Any,
        *,
        steps: int,
        dt: float = 1.0,
    ) -> DissipativeTimeSeries:
        """Reset the tracker, evolve ``steps`` times, and record the trajectory."""

        if ensure_numpy is None:
            raise ImportError("Mathematics backend required for evolve_and_track")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer.")
        time_step = _validate_nonnegative_finite(dt, name="dt")
        if time_step == 0.0:
            raise ValueError("dt must be positive.")
        self._series = DissipativeTimeSeries()
        self._snapshots = []
        current = _validate_density_matrix(initial_density)
        self.record(current, t=0.0)
        for index in range(int(steps)):
            evolved = self._engine.step(current, dt=time_step)
            current = np.asarray(ensure_numpy(evolved), dtype=np.complex128)
            self.record(current, t=(index + 1) * time_step)
        return self._series

    def report(self) -> DissipativeTimeSeries:
        return self._series

    @property
    def latest_balance(self) -> DissipativeBalance | None:
        if len(self._snapshots) < 2:
            return None
        previous_time, previous = self._snapshots[-2]
        current_time, current = self._snapshots[-1]
        return verify_dissipative_balance(
            previous,
            current,
            dt=current_time - previous_time,
            collapse_operators=self._collapse_operators or None,
            steady_state=self._steady_state,
        )


def predict_amplitude_damping_purity(
    initial_density: Any = _MISSING_ARGUMENT,
    gamma: Any = _MISSING_ARGUMENT,
    time: Any = _MISSING_ARGUMENT,
    dim: int = 2,
    *,
    initial_purity: Any = _MISSING_ARGUMENT,
) -> float:
    r"""Return exact qubit amplitude-damping purity for a full initial state.

    The convention is ``L = sqrt(gamma) |0><1|`` and
    ``eta = exp(-gamma time)``. Purity alone does not identify the result:
    equal-purity states with different excited populations evolve differently.
    Scalar input, including the historical ``initial_purity=`` keyword, retains
    the old interpolation with a deprecation warning; that compatibility result
    is not an exact channel prediction.  Matrix input must use
    ``initial_density`` because purity alone cannot identify channel evolution.
    """

    if (
        initial_density is not _MISSING_ARGUMENT
        and initial_purity is not _MISSING_ARGUMENT
    ):
        raise TypeError("specify only one of initial_density or initial_purity.")
    if initial_density is _MISSING_ARGUMENT:
        if initial_purity is _MISSING_ARGUMENT:
            raise TypeError("missing required argument: 'initial_density'.")
        if np.ndim(initial_purity) != 0:
            raise ValueError(
                "initial_purity must be scalar; pass a density matrix as "
                "initial_density instead."
            )
        initial_density = initial_purity
    if gamma is _MISSING_ARGUMENT:
        raise TypeError("missing required argument: 'gamma'.")
    if time is _MISSING_ARGUMENT:
        raise TypeError("missing required argument: 'time'.")

    decay_rate = _validate_nonnegative_finite(gamma, name="gamma")
    elapsed = _validate_nonnegative_finite(time, name="time")
    if np.ndim(initial_density) == 0:
        initial_purity = float(initial_density)
        if (
            not math.isfinite(initial_purity)
            or initial_purity < 0.0
            or initial_purity > 1.0
        ):
            raise ValueError("legacy initial purity must lie in [0, 1].")
        warnings.warn(
            "Scalar input cannot determine amplitude-damping purity; pass the "
            "full 2x2 density matrix. The scalar interpolation is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return 1.0 - (1.0 - initial_purity) * math.exp(
            -decay_rate * elapsed
        )
    if dim != 2:
        raise ValueError(
            "this analytical amplitude-damping model is defined for dim=2."
        )
    rho = _validate_density_matrix(initial_density)
    if rho.shape != (2, 2):
        raise ValueError("initial_density must be a 2x2 qubit density matrix.")
    eta = math.exp(-decay_rate * elapsed)
    excited_population = float(rho[1, 1].real)
    evolved = np.array(
        [
            [
                rho[0, 0] + (1.0 - eta) * excited_population,
                math.sqrt(eta) * rho[0, 1],
            ],
            [math.sqrt(eta) * rho[1, 0], eta * rho[1, 1]],
        ],
        dtype=np.complex128,
    )
    return float(np.trace(evolved @ evolved).real)


def predict_dephasing_purity(
    initial_density: Any,
    gamma: float,
    time: float,
) -> float:
    r"""Return exact purity for uniform basis dephasing.

    ``gamma`` is the off-diagonal amplitude decay rate, as generated by either
    ``sqrt(gamma/2) sigma_z`` for a qubit or the complete projector set
    ``sqrt(gamma) |k><k|``. Off-diagonal purity terms therefore decay as
    ``exp(-2 gamma time)``.
    """

    rho = _validate_density_matrix(initial_density)
    decay_rate = _validate_nonnegative_finite(gamma, name="gamma")
    elapsed = _validate_nonnegative_finite(time, name="time")
    diagonal_purity = float(np.sum(np.abs(np.diag(rho)) ** 2))
    total_purity = float(np.trace(rho @ rho).real)
    off_diagonal_purity = max(0.0, total_purity - diagonal_purity)
    return diagonal_purity + off_diagonal_purity * math.exp(
        -2.0 * decay_rate * elapsed
    )


def analyze_dissipation_rates(generator: Any, dim: int) -> dict[str, Any]:
    """Analyze a finite Liouville-generator spectrum with explicit scope."""

    matrix = _validate_generator(generator, dim)
    eigenvalues = np.linalg.eigvals(matrix)
    tolerance = _DEFAULT_ATOL * max(1.0, float(np.linalg.norm(matrix, ord=2)))
    steady_mask = np.abs(eigenvalues) <= tolerance
    stable_decay_mask = (~steady_mask) & (eigenvalues.real < -tolerance)
    neutral_mask = (~steady_mask) & (np.abs(eigenvalues.real) <= tolerance)
    decay_rates = np.sort(-eigenvalues.real[stable_decay_mask])
    spectral_gap = (
        float(decay_rates[0])
        if len(decay_rates) and not bool(np.any(neutral_mask))
        else 0.0
    )
    relaxation_time = (
        1.0 / spectral_gap if spectral_gap > 0.0 else float("inf")
    )
    identity_vector = np.eye(dim, dtype=np.complex128).reshape(
        dim * dim, order="F"
    )
    trace_residual = float(np.linalg.norm(identity_vector.conj().T @ matrix))
    max_real_part = float(np.max(eigenvalues.real))
    return {
        "eigenvalues": eigenvalues,
        "decay_rates": decay_rates,
        "oscillation_frequencies": np.abs(eigenvalues.imag),
        "spectral_gap": spectral_gap,
        "relaxation_time": relaxation_time,
        "n_steady_modes": int(np.sum(steady_mask)),
        "n_oscillating_modes": int(
            np.sum(np.abs(eigenvalues.imag) > 1e-6)
        ),
        "n_neutral_nonstationary_modes": int(np.sum(neutral_mask)),
        "has_neutral_nonstationary_modes": bool(np.any(neutral_mask)),
        "max_real_part": max_real_part,
        "has_unstable_modes": bool(max_real_part > tolerance),
        "trace_preservation_residual": trace_residual,
        "is_trace_preserving": bool(trace_residual <= tolerance),
        "relaxes_to_unique_state": bool(
            np.sum(steady_mask) == 1
            and max_real_part <= tolerance
            and not np.any(neutral_mask)
        ),
    }


def classify_dissipative_regime(balance: DissipativeBalance) -> dict[str, Any]:
    """Classify scalar-change magnitude without inferring grammar status.

    Empirical bins are applied to the largest absolute purity/entropy rate. They
    do not identify a generator, prove decoherence, or determine U1--U6 status.
    Legacy return keys remain available.
    """

    magnitude = max(
        abs(balance.purity_change_rate), abs(balance.entropy_change_rate)
    )
    if magnitude < _WEAK_CHANGE_RATE_THRESHOLD:
        tier, legacy_regime = "weak", "weak_dissipation"
    elif magnitude < _MODERATE_CHANGE_RATE_THRESHOLD:
        tier, legacy_regime = "moderate", "moderate_dissipation"
    elif magnitude < _STRONG_CHANGE_RATE_THRESHOLD:
        tier, legacy_regime = "strong", "strong_dissipation"
    else:
        tier, legacy_regime = "large", "decoherence"
    score = max(
        0.0,
        min(1.0, 1.0 - magnitude / _STRONG_CHANGE_RATE_THRESHOLD),
    )
    purity_direction = (
        "increasing"
        if balance.purity_change_rate > _DEFAULT_ATOL
        else "decreasing"
        if balance.purity_change_rate < -_DEFAULT_ATOL
        else "stationary"
    )
    entropy_direction = (
        "increasing"
        if balance.entropy_change_rate > _DEFAULT_ATOL
        else "decreasing"
        if balance.entropy_change_rate < -_DEFAULT_ATOL
        else "stationary"
    )
    return {
        "change_tier": tier,
        "regime": legacy_regime,
        "change_rate_magnitude": magnitude,
        "diagnostic_score": score,
        "conservation_quality": score,
        "purity_direction": purity_direction,
        "entropy_direction": entropy_direction,
        "unital_dissipator": balance.unital_dissipator,
        "grammar_status_inferred": False,
        "grammar_analog": "No U1--U6 status is inferred from collapse operators.",
        "structural_interpretation": (
            "Empirical density-state change tier for the observed interval; "
            "interpret it with the channel, time step and fixed-point data."
        ),
    }


__all__ = [
    "DissipativeSnapshot",
    "DissipativeBalance",
    "DissipativeTimeSeries",
    "capture_dissipative_snapshot",
    "compute_dissipation_bound",
    "compute_dissipator_action",
    "compute_instantaneous_purity_rate",
    "compute_purity_decay_bound",
    "is_unital_dissipator",
    "verify_dissipative_balance",
    "DissipativeConservationTracker",
    "predict_amplitude_damping_purity",
    "predict_dephasing_purity",
    "analyze_dissipation_rates",
    "classify_dissipative_regime",
    "steady_state_from_generator",
]
