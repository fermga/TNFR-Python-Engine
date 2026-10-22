"""One fixed record-only carrier proposal, with no certification authority.

Only the thirty historical sampling times are integrated. The returned exact
represented initial state defines a candidate for a separate validated solver;
optimizer status and floating residuals are never trajectory evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from typing import Any

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I
from ._organized_computable_prediction import _forecast_stages, _ForecastStage
from ._sine_admission import _sine_model_coefficients
from .relational_observations import _ordered
from .relational_sine_comparison import _sine_form_gradient, _sine_rate_evaluator

_FORM_BOUND = Q(1, 100)
_MAX_NFEV = 80
_RESIDUAL_SCALE = 1e-4
_RTOL = 1e-11
_ATOL = 1e-17
_MAX_STEP = 1 / 8
_EDGES = tuple((i, (i + 1) % 5) for i in range(5))
_NEIGHBORS = tuple(((i - 1) % 5, (i + 1) % 5) for i in range(5))


@dataclass(frozen=True)
class _ForecastProposal:
    initial_candidate: tuple[Q, ...] | None
    proposed_start: tuple[Q, ...] | None
    optimization_status: int | None
    optimization_success: bool
    nfev: int
    forward_calls: int
    maximum_proposal_residual: Q | None
    backend: str
    reason: str | None


def _sequence(value: Any, length: int, label: str) -> tuple[Any, ...]:
    result = _ordered(value, label, limit=length + 1)
    if len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} entries")
    return result


def _backend_provenance(numpy_module: Any, scipy_module: Any) -> str:
    """Retain stable build/BLAS facts without paths, commands or environment."""
    import json

    def selected(row: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: row[key] for key in keys if key in row}

    def build(module: Any) -> dict[str, Any]:
        raw = module.show_config(mode="dicts")
        return {
            "version": module.__version__,
            "compilers": {
                language: selected(row, ("name", "version", "linker"))
                for language, row in raw.get("Compilers", {}).items()
            },
            "machines": {
                kind: selected(
                    raw.get("Machine Information", {}).get(kind, {}),
                    ("cpu", "family", "endian", "system"),
                )
                for kind in ("host", "build")
            },
            "linear_algebra": {
                library: selected(
                    raw.get("Build Dependencies", {}).get(library, {}),
                    (
                        "name",
                        "found",
                        "version",
                        "detection method",
                        "openblas configuration",
                        "has ilp64",
                        "cython blas ilp64",
                    ),
                )
                for library in ("blas", "lapack")
            },
            "simd": selected(raw.get("SIMD Extensions", {}), ("baseline", "found")),
        }

    return json.dumps(
        {
            "arithmetic": "binary64",
            "numpy": build(numpy_module),
            "scipy": build(scipy_module),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reading_times() -> tuple[Q, ...]:
    return tuple(
        time
        for stage in _forecast_stages()[1:]
        if stage.name == "plateau"
        for time in (stage.start, stage.start + stage.duration)
    )


def _admit_inputs(
    times: Any, values: Any, observation_matrix: Any
) -> tuple[tuple[Q, ...], tuple[Q, ...], tuple[tuple[Q, ...], ...]]:
    admitted_times = tuple(
        exact_or_represented_real(value, f"sample time {i}")
        for i, value in enumerate(_sequence(times, 30, "sample times"))
    )
    if admitted_times != _reading_times():
        raise ValueError("sample times must equal the declared exact schedule")
    readings = tuple(
        exact_or_represented_real(value, f"reading {i}")
        for i, value in enumerate(_sequence(values, 150, "readings"))
    )
    matrix = []
    for row_index, row in enumerate(
        _sequence(observation_matrix, 150, "observation matrix")
    ):
        normalized = []
        for column, value in enumerate(_sequence(row, 15, "observation row")):
            label = f"matrix coefficient {row_index},{column}"
            if isinstance(value, I):
                lo = exact_or_represented_real(value.lo, label)
                hi = exact_or_represented_real(value.hi, label)
                if lo > hi:
                    raise ValueError("matrix interval endpoints are reversed")
                value = (lo + hi) / 2
            normalized.append(exact_or_represented_real(value, label))
        matrix.append(tuple(normalized))
    # Materialization is a separate admission: exact nonzero rationals cannot
    # silently disappear, even though this stage supplies proposals only.
    for value in readings + tuple(value for row in matrix for value in row):
        finite_represented_real(value, "proposal input")
    return admitted_times, readings, tuple(matrix)


def _exact_candidate(parameters: Any) -> tuple[Q, ...]:
    parameters = _sequence(parameters, 9, "candidate parameters")
    exact = tuple(
        finite_represented_real(value, f"candidate parameter {i}")[1]
        for i, value in enumerate(parameters)
    )
    # The float spelling of .01 is outside the exact allowed bound. The exact
    # clipped coordinates, rather than the optimizer floats, are validated.
    forms = tuple(max(-_FORM_BOUND, min(_FORM_BOUND, x)) for x in exact[:5])
    return forms + (Q(0),) + exact[5:]


@lru_cache(maxsize=1)
def _numeric_coefficients() -> tuple[float, float, float]:
    """Materialize the shared held rate coefficients in structural time."""
    model = RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1023, 1024),
        phase_weight=Q(1, 1024),
        phase_domain="regular",
    )
    loss, _, _ = _sine_model_coefficients(model, positive_loss=True)
    # Degree four is inherited from the complete C9-by-C5 law. Both rows
    # undergo the same tau=loss*t clock conversion.
    rates = _sine_rate_evaluator(model, (4,) * 5, (Q(1),) * 5)
    gradient = rates((Q(1),) * 5, (Q(0),) * 5)
    current = rates((Q(0),) * 5, (Q(1),) * 5)

    def midpoint(value: Any) -> float:
        value = value.midpoint if isinstance(value, I) else value
        return finite_represented_real(value / loss, "rate coefficient")[0]

    return (
        midpoint(gradient["form_rates"][0]),
        midpoint(current["form_rates"][0]),
        midpoint(gradient["phase_rates"][0]),
    )


def _numeric_carrier_field(stage: _ForecastStage) -> Any:
    """Binary64 field and analytic state Jacobian of the shared carrier law."""
    import numpy as np

    form_loss, exchange, phase_exchange = _numeric_coefficients()
    laplacian = np.array(
        [
            _sine_form_gradient(tuple(Q(i == j) for i in range(5)), _NEIGHBORS)
            for j in range(5)
        ],
        dtype=float,
    ).T

    def evaluate(local_time: float, state: Any) -> tuple[Any, Any]:
        # Exact represented local time avoids cancellation against tau=200.
        clock = stage.start + Q.from_float(float(local_time))
        command = np.array([float(value) for value in stage.command(clock)])
        phase = state[5:] + command
        gradient = np.array(_sine_form_gradient(state[:5], _NEIGHBORS))
        currents = np.zeros(5)
        current_jacobian = np.zeros((5, 5))
        for tail, head in _EDGES:
            angle = phase[head] - phase[tail]
            current, derivative = np.sin(angle), np.cos(angle)
            currents[tail] += current
            currents[head] -= current
            current_jacobian[tail, tail] -= derivative
            current_jacobian[tail, head] += derivative
            current_jacobian[head, tail] += derivative
            current_jacobian[head, head] -= derivative
        rate = np.concatenate(
            (form_loss * gradient + exchange * currents, phase_exchange * gradient)
        )
        jacobian = np.zeros((10, 10))
        jacobian[:5, :5] = form_loss * laplacian
        jacobian[:5, 5:] = exchange * current_jacobian
        jacobian[5:, :5] = phase_exchange * laplacian
        return rate, jacobian

    return evaluate


def _historical_response(parameters: Any) -> tuple[Any, Any]:
    """Carry one state and its nine sensitivities through historical stages."""
    import numpy as np
    from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

    state = np.concatenate((parameters[:5], [0.0], parameters[5:]))
    sensitivity = np.zeros((10, 9))
    sensitivity[:5, :5] = np.eye(5)
    sensitivity[6:, 5:] = np.eye(4)
    carried = np.concatenate((state, sensitivity.ravel()))
    samples, jacobians = [], []
    last_time = _reading_times()[-1]
    for stage in _forecast_stages()[1:]:
        if stage.start >= last_time:
            break
        if stage.start + stage.duration > last_time:
            raise ArithmeticError("historical stage exceeds the last reading")
        if stage.name == "plateau":
            samples.append(carried[:5].copy())
            jacobians.append(carried[10:].reshape(10, 9)[:5].copy())
        evaluate = _numeric_carrier_field(stage)

        def augmented(local_time: float, point: Any) -> Any:
            rate, derivative = evaluate(local_time, point[:10])
            tangent = derivative @ point[10:].reshape(10, 9)
            return np.concatenate((rate, tangent.ravel()))

        result = solve_ivp(
            augmented,
            (0.0, float(stage.duration)),
            carried,
            method="DOP853",
            rtol=_RTOL,
            atol=_ATOL,
            max_step=_MAX_STEP,
            t_eval=[float(stage.duration)],
        )
        if not result.success:
            raise ArithmeticError("historical proposal integration failed")
        carried = result.y[:, -1]
        if not np.all(np.isfinite(carried)):
            raise ArithmeticError("nonfinite historical proposal state")
        if stage.name == "plateau":
            samples.append(carried[:5].copy())
            jacobians.append(carried[10:].reshape(10, 9)[:5].copy())
    if len(samples) != 30:
        raise ArithmeticError("historical proposal omitted a sample")
    return np.array(samples).ravel(), np.concatenate(jacobians)


def propose_candidate(
    times: Any, values: Any, observation_matrix: Any
) -> _ForecastProposal:
    """Run the declared single-start fit using the complete historical record.

    Malformed input raises. Numerical proposal failure returns unavailable;
    there are no restarts, adaptive budgets or calls beyond the last reading.
    """
    _, readings, matrix = _admit_inputs(times, values, observation_matrix)
    import numpy as np
    import scipy  # type: ignore[import-untyped]
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    backend = _backend_provenance(np, scipy)
    start = None
    forward_calls = 0
    try:
        observed = np.array(readings, dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(
            np.array(matrix, dtype=float), observed, rcond=None
        )
        if rank != 15 or not np.all(np.isfinite(coefficients)):
            raise ArithmeticError("static phasor initialization unavailable")
        gaps = np.arctan2(coefficients[5:10], coefficients[10:15])
        forms = np.clip(observed[:5], -float(_FORM_BOUND), float(_FORM_BOUND))
        parameters = np.concatenate((forms, np.cumsum(gaps[:4])))
        start = _exact_candidate(parameters)
        cached_parameters = None
        cached_response = None

        def response(point: Any) -> tuple[Any, Any]:
            nonlocal cached_parameters, cached_response, forward_calls
            if cached_parameters is None or not np.array_equal(
                point, cached_parameters
            ):
                forward_calls += 1
                cached_response = _historical_response(point)
                cached_parameters = point.copy()
            return cached_response

        def residual(point: Any) -> Any:
            return (response(point)[0] - observed) / _RESIDUAL_SCALE

        def jacobian(point: Any) -> Any:
            return response(point)[1] / _RESIDUAL_SCALE

        optimum = least_squares(
            residual,
            parameters,
            jac=jacobian,
            bounds=(
                np.array([-float(_FORM_BOUND)] * 5 + [-np.inf] * 4),
                np.array([float(_FORM_BOUND)] * 5 + [np.inf] * 4),
            ),
            method="trf",
            max_nfev=_MAX_NFEV,
            ftol=None,
            gtol=None,
            xtol=1e-13,
            x_scale=np.array([0.01] * 5 + [1.0] * 4),
        )
        maximum = finite_represented_real(
            float(np.max(np.abs(optimum.fun))) * _RESIDUAL_SCALE,
            "maximum proposal residual",
        )[1]
        candidate = _exact_candidate(optimum.x) if optimum.success else None
        return _ForecastProposal(
            candidate,
            start,
            int(optimum.status),
            bool(optimum.success),
            int(optimum.nfev),
            forward_calls,
            maximum,
            backend,
            None if optimum.success else "optimizer_did_not_converge",
        )
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return _ForecastProposal(
            None, start, None, False, 0, forward_calls, None, backend, str(exc)
        )
