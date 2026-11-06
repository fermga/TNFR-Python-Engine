"""One frozen computational P2 acquisition with a continuous-model error budget.

The native relational Euler owner evolves both nodes. Independent interval
field residuals and prior-range derivative bounds enclose its readout error.
The simulator's parameter is known: this is not blind or physical calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import zipfile
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import numpy as np

from tnfr._exact_time import exact_or_represented_real
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.mathematics._rational_interval import I, INTERVAL_METHOD, pi_interval
from tnfr.physics.relational_observations import (
    bound_relational_coefficient_from_samples,
)
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_dumps, json_loads, safe_write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "artifacts/research/relational_coefficient_acquisition/result.json"
)
NODES = (0, 1)
SOURCE_BETA = Q(1)
BETA_PRIOR = (Q(1, 2), Q(2))
FORM_LIMIT, PHASE_LIMIT = Q(1, 8), Q(1, 256)
SAMPLE_STEP, EULER_STEP = Q(1, 2**10), Q(1, 2**21)
HORIZON = 2 * SAMPLE_STEP
STEP_COUNT = int(HORIZON / EULER_STEP)
SAMPLE_INDICES = (0, STEP_COUNT // 2, STEP_COUNT)
WIDTH_LIMIT = Q(1, 8)
PRESSURE_PATH = "fused_canonical"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prior_window_bounds():
    """Derive one window budget before reading any temporal response."""
    _require(0 < PHASE_LIMIT <= 1, "phase bound must lie in the sinc-series domain")
    _require(0 < BETA_PRIOR[0] <= BETA_PRIOR[1], "invalid positive beta prior")
    lower_sinc = 1 - PHASE_LIMIT**2 / 6
    q0 = 1 / lower_sinc
    q1 = PHASE_LIMIT / (3 * lower_sinc**2)
    a, b, c = Q(1), Q(1, 3), 1 / (3 * BETA_PRIOR[0])
    form_speed = a * FORM_LIMIT + b * PHASE_LIMIT
    phase_speed = c * FORM_LIMIT * q0
    form_second = a * form_speed + b * phase_speed
    phase_second = c * (form_speed * q0 + FORM_LIMIT * q1 * phase_speed)
    lipschitz = max(a + b, c * (q0 + FORM_LIMIT * q1))
    _require(HORIZON * phase_speed < PHASE_LIMIT, "phase first-exit bound failed")
    _require(lipschitz <= 2 and 2 * HORIZON < 1, "error propagation bound failed")
    return {
        "sinc_lower": lower_sinc,
        "inverse_sinc_upper": q0,
        "inverse_sinc_derivative_upper": q1,
        "form_speed_upper": form_speed,
        "phase_speed_upper": phase_speed,
        "second_derivative_upper": max(form_second, phase_second),
        "third_form_derivative_upper": a * form_second + b * phase_second,
        "lipschitz_upper": Q(2),
        "growth_factor_upper": 1 / (1 - 2 * HORIZON),
    }


def _source_files():
    paths = sorted((ROOT / "src/tnfr").rglob("*.py"))
    paths += [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    return {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}


def _verify_runtime_source():
    """Reject installed/other-checkout TNFR modules before certifying this source."""
    source = (ROOT / "src/tnfr").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                path = Path(origin).resolve()
                _require(
                    path.is_relative_to(source) and path.suffix == ".py",
                    f"executing TNFR source is outside the archived checkout: {name}",
                )


def prepare_protocol():
    """Declare the fixed experiment and precision before a trajectory exists."""
    _verify_runtime_source()
    _require(
        HORIZON == 2 * SAMPLE_STEP == STEP_COUNT * EULER_STEP
        and tuple(i * EULER_STEP for i in SAMPLE_INDICES) == (0, SAMPLE_STEP, HORIZON),
        "sample grid differs from Euler grid",
    )
    _require(
        Q(float(EULER_STEP)) == EULER_STEP
        and Q(float(FORM_LIMIT / 2)) == FORM_LIMIT / 2,
        "preparation or step is not represented exactly",
    )
    _require(BETA_PRIOR[0] <= SOURCE_BETA <= BETA_PRIOR[1], "source outside prior")
    return {
        "protocol": "relational-p2-coefficient-acquisition-v1",
        "nodes": NODES,
        "edges": ((0, 1),),
        "initial_form": (FORM_LIMIT / 2, -FORM_LIMIT / 2),
        "initial_phase": (Q(0), Q(0)),
        "capacity": (Q(1), Q(1)),
        "effective_weights": (Q(1, 2), Q(1, 2)),
        "source_beta": SOURCE_BETA,
        "prior_beta": BETA_PRIOR,
        "prior_rectangle": (FORM_LIMIT, PHASE_LIMIT),
        "window_bounds": prior_window_bounds(),
        "horizon": HORIZON,
        "euler_step": EULER_STEP,
        "step_count": STEP_COUNT,
        "sample_step": SAMPLE_STEP,
        "sample_indices": SAMPLE_INDICES,
        "observation": "exact represented EPI[0]-EPI[1]; gain one; baseline zero",
        "clock": "relative structural time k*euler_step; exact dyadic grid",
        "gamma": "absent",
        "events": "none; support and capacity held; no clipping or controller",
        "pressure_refresh": "native field at every step and endpoint",
        "precision_limit": WIDTH_LIMIT,
        "prediction": (
            "complete window; chi interval contains source beta "
            "and width <= precision_limit"
        ),
        "numerical_scope": (
            "nonlinear ideal P2 contrast error; "
            "not a tangent or energy-defect estimate"
        ),
        "scientific_scope": (
            "known-source software validation; no physical acquisition "
            "or universal coefficient selection"
        ),
        "source_archive_scope": (
            "all project Python under src/tnfr, this producer and pyproject; "
            "not installed dependency binaries or authenticated chronology"
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "precision": (
                "binary64 native states; exact Fraction defects; "
                "outward dyadic128 oracle"
            ),
            "integrator": "native simultaneous explicit Euler",
            "pressure_path": PRESSURE_PATH,
            "interval_method": INTERVAL_METHOD,
            "randomness": "none",
        },
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in _source_files().items()
        },
    }


def _json_default(value):
    if isinstance(value, Q):
        return {
            "numerator": int(value.numerator),
            "denominator": int(value.denominator),
        }
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _encoded(value):
    return json_dumps(value, default=_json_default, sort_keys=True, allow_nan=False)


def _write(path, value):
    text = _encoded(value) + "\n"
    safe_write(
        path, lambda stream: stream.write(text), mode="x", atomic=False, sync=True
    )


def _contrasts(field):
    _require(
        field.nodes == NODES
        and field.edges == ((0, 1),)
        and field.capacity == (1.0, 1.0),
        "changed support/node/capacity declaration",
    )
    _require(
        field.model.effective_weights == (0.5, 0.5)
        and Q(field.model.storage_scale) == SOURCE_BETA
        and field.model.phase_domain == "acute"
        and field.clipping == "none",
        "changed model declaration",
    )
    _require(field.pressure_path == PRESSURE_PATH, "pressure path changed")
    return Q(field.epi[0]) - Q(field.epi[1]), Q(field.phase[0]) - Q(field.phase[1])


def _in_rectangle(state):
    return abs(state[0]) <= FORM_LIMIT and abs(state[1]) <= PHASE_LIMIT


def enclose_p2_field(state, beta=SOURCE_BETA):
    """Bound the ideal RHS at a represented point, without integrating it."""
    if not isinstance(state, tuple) or len(state) != 2:
        raise TypeError("state must be an ordered (form, phase) pair")
    state = tuple(exact_or_represented_real(value, "coordinate") for value in state)
    beta = exact_or_represented_real(beta, "beta")
    _require(_in_rectangle(state), "numerical state left the proof rectangle")
    _require(BETA_PRIOR[0] <= beta <= BETA_PRIOR[1], "source beta left its prior")
    u, phase = map(I, state)
    square = phase**2
    low = 1 - square / 6
    high = low + square**2 / 120
    sinc = I(low.lo, high.hi)
    pi = pi_interval()
    return -u - phase / pi, u / (beta * pi * sinc)


def _frame(field, index, error_upper):
    return {
        "step": index,
        "time": index * EULER_STEP,
        "epi": field.epi,
        "phase": field.phase,
        "continuous_contrast_error_upper": error_upper,
        "pressure_path": field.pressure_path,
    }


def evaluate_protocol(protocol):
    """Execute only the declared native acquisition; never tune after response."""
    _require(
        _encoded(protocol) == _encoded(prepare_protocol()),
        "frozen protocol/source mismatch",
    )
    model = RelationalExchangeModel(
        float(SOURCE_BETA), epi_weight=0.5, phase_weight=0.5
    )
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    for node, form in zip(NODES, (FORM_LIMIT / 2, -FORM_LIMIT / 2), strict=True):
        graph.nodes[node].update(EPI=float(form), theta=0.0, nu_f=1.0)
    initial = evaluate_relational_exchange(graph, model=model)
    _require(_contrasts(initial) == (FORM_LIMIT, 0), "initial preparation changed")
    bounds = prior_window_bounds()
    local_truncation = bounds["second_derivative_upper"] * EULER_STEP**2 / 2
    field_defects = update_defects = Q(0)
    frames = [_frame(initial, 0, Q(0))]
    stopped = None
    for index in range(1, STEP_COUNT + 1):
        try:
            step = step_relational_exchange(graph, model=model, dt=float(EULER_STEP))
            before, after = _contrasts(step.before), _contrasts(step.after)
            _require(
                step.before.epi == tuple(frames[-1]["epi"])
                and step.before.phase == tuple(frames[-1]["phase"]),
                "step did not start at the preceding recorded state",
            )
            _require(
                _in_rectangle(after), "numerical endpoint left the proof rectangle"
            )
            _require(
                Q(graph.graph["_t"]) == Q(step.t_after) == index * EULER_STEP
                and Q(step.t_before) == (index - 1) * EULER_STEP
                and Q(step.dt) == EULER_STEP
                and step.clock_defect == 0,
                "clock differs from declared grid",
            )
            rates = (
                Q(step.before.form_rate[0]) - Q(step.before.form_rate[1]),
                Q(step.before.phase_rate[0]) - Q(step.before.phase_rate[1]),
            )
            ideal = enclose_p2_field(before)
            field_error = max(
                (I(rate) - value).abs_max
                for rate, value in zip(rates, ideal, strict=True)
            )
            rounding = tuple(
                new - old - EULER_STEP * rate
                for new, old, rate in zip(after, before, rates, strict=True)
            )
            field_defects += EULER_STEP * field_error
            update_defects += max(map(abs, rounding))
            error_upper = (
                field_defects + update_defects + index * local_truncation
            ) * bounds["growth_factor_upper"]
            frame = _frame(step.after, index, error_upper)
            frame.update(
                contrast_rates=rates,
                start_form_rate=step.before.form_rate,
                start_phase_rate=step.before.phase_rate,
                field_error_upper=field_error,
                update_defects=rounding,
                clock_defect=step.clock_defect,
                accumulated_field_error=field_defects,
                accumulated_update_error=update_defects,
                accumulated_truncation_error=index * local_truncation,
            )
            frames.append(frame)
        except Exception as error:
            stopped = {
                "attempted_step": index,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            break
    complete = len(frames) == STEP_COUNT + 1 and stopped is None
    analysis = analysis_error = None
    covered = precise = None
    if complete:
        try:
            _verify_runtime_source()
            samples = tuple(
                Q(frames[i]["epi"][0]) - Q(frames[i]["epi"][1]) for i in SAMPLE_INDICES
            )
            sample_error = max(
                frames[i]["continuous_contrast_error_upper"] for i in SAMPLE_INDICES
            )
            report = bound_relational_coefficient_from_samples(
                samples,
                sample_step=SAMPLE_STEP,
                sample_error_bound=sample_error,
                third_derivative_bound=bounds["third_form_derivative_upper"],
            )
            analysis = relational_report_to_dict(report)
            if report.jet.coefficient_bounds is not None:
                lower, upper = report.jet.coefficient_bounds
                covered = lower <= SOURCE_BETA <= upper
                precise = upper - lower <= WIDTH_LIMIT
        except Exception as error:
            analysis_error = {"error_type": type(error).__name__, "error": str(error)}
    return {
        "protocol": protocol,
        "completed": complete,
        "stopped": stopped,
        "frames": frames,
        "sample_analysis": analysis,
        "analysis_error": analysis_error,
        "contains_predeclared_truth": covered,
        "width_within_budget": precise,
        "passed": complete and covered is True and precise is True,
    }


def _archive(path):
    files = _source_files()

    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)

    safe_write(path, write, mode="xb", atomic=False, sync=True)


def _verify_archive(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(
            len(names) == len(set(names)) and set(names) == set(expected),
            "source archive inventory differs",
        )
        for name, digest in expected.items():
            _require(
                hashlib.sha256(archive.read(name)).hexdigest() == digest,
                "source archive content differs",
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output
    frozen = output.with_suffix(".protocol.json")
    archive = output.with_suffix(".source.zip")
    if output.exists():
        raise FileExistsError(f"retain existing response: {output}")
    if args.prepare:
        if frozen.exists() or archive.exists():
            raise FileExistsError("retain existing protocol/source archive")
        protocol = prepare_protocol()
        _archive(archive)
        _verify_archive(archive, protocol["source_sha256"])
        _write(frozen, protocol)
        print(f"Frozen protocol: {frozen}")
        return 0
    protocol = json_loads(frozen.read_bytes())
    _require(
        _encoded(protocol) == _encoded(prepare_protocol()),
        "frozen protocol/source mismatch",
    )
    _verify_archive(archive, protocol["source_sha256"])
    provenance = {
        "protocol_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    try:
        result = evaluate_protocol(protocol)
    except Exception as error:
        _write(
            output,
            {
                **provenance,
                "protocol": protocol,
                "completed": False,
                "stopped": {
                    "stage": "before_acquisition",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                "frames": (),
                "sample_analysis": None,
                "analysis_error": None,
                "contains_predeclared_truth": None,
                "width_within_budget": None,
                "passed": False,
            },
        )
        raise
    result.update(provenance)
    _write(output, result)
    print(
        _encoded(
            {
                key: result[key]
                for key in (
                    "completed",
                    "passed",
                    "contains_predeclared_truth",
                    "width_within_budget",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
