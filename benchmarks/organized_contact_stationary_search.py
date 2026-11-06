"""One frozen full-fine stationary search; never a trajectory integrator.

See theory/nodal/ORGANIZED_CONTACT_STATIONARY_SEARCH.md. Run only after
committing the declaration and source; existing evidence is never replaced.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from fractions import Fraction as Q
import io
import os
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, minimize, root

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics._organized_external_interaction import (
    _derive_organized_external_interaction,
)
from tnfr.physics._organized_stationary_certificate import (
    _certify_organized_stationary,
)
from tnfr.research.artifact_io import encode_exact_tree, write_json_once

_MARGIN = Q(1, 1024)
_RADIUS = Q(1, 10**7)
_AMPLITUDES = (0, 1 / 8, 1 / 4, 1 / 2, 1, 3 / 2)
_SOURCE_PATHS = (
    "benchmarks/organized_contact_stationary_search.py",
    "src/tnfr/physics/_organized_stationary_certificate.py",
    "theory/nodal/ORGANIZED_CONTACT_STATIONARY_SEARCH.md",
)


def _geometry():
    geometry = _derive_organized_external_interaction()
    edges = tuple((a, b) for a, b, _ in geometry.weighted_edges)
    incidence = np.zeros((144, 54), dtype=np.float64)
    for row, (a, b) in enumerate(edges):
        incidence[row, a], incidence[row, b] = -1, 1
    branches = np.array(
        [-int(a % 9 == 8) for a, _ in edges[:54]]
        + [-int(a // 9 == 4) for a, _ in edges[54:99]],
        dtype=np.int64,
    )
    return edges, incidence[:, 1:], branches


def _fields(phase, incidence, *, hessian=True):
    """Float screening only; acceptance rebuilds exact interval edge fields."""
    gaps = incidence @ phase
    sine, cosine = np.sin(gaps), np.cos(gaps)
    return (
        float(np.sum(1 - cosine)),
        incidence.T @ sine,
        incidence.T @ (cosine[:, None] * incidence) if hessian else None,
    )


def _starts(incidence, branches):
    generator = np.random.Generator(np.random.PCG64(20261010))
    for origin in range(4):
        base = np.array(
            [
                2 * np.pi * j / 9 + (2 * np.pi * c / 5 if c < 5 else origin * np.pi / 2)
                for c in range(6)
                for j in range(9)
            ],
            dtype=np.float64,
        )[1:]
        gaps = incidence[:99] @ base - 2 * np.pi * branches
        slack = np.pi - float(_MARGIN) - np.abs(gaps)
        if not np.all(slack > 0):
            raise ValueError("base is outside the declared chamber")
        for amplitude in _AMPLITUDES:
            perturbation = amplitude * generator.uniform(-1, 1, 53)
            changes = np.abs(incidence[:99] @ perturbation)
            ratios = slack[changes > 0] / (2 * changes[changes > 0])
            scale = min(1.0, float(np.min(ratios))) if ratios.size else 1.0
            yield origin, amplitude, scale, base + scale * perturbation


class _BudgetExhausted(RuntimeError):
    pass


class _Oracle:
    def __init__(self, incidence, limit):
        self.incidence = incidence
        self.limit = limit
        self.calls = 0
        self.jacobian_calls = 0
        self.last_phase = None

    def value_gradient(self, phase):
        if self.calls >= self.limit:
            raise _BudgetExhausted("value/residual oracle budget exhausted")
        self.calls += 1
        self.last_phase = phase.copy()
        value, gradient, _ = _fields(phase, self.incidence, hessian=False)
        return value, gradient

    def residual(self, phase):
        return self.value_gradient(phase)[1]

    def jacobian(self, phase):
        if self.jacobian_calls >= self.limit:
            raise _BudgetExhausted("Jacobian oracle budget exhausted")
        self.jacobian_calls += 1
        return _fields(phase, self.incidence)[2]


def _coordinates(phase):
    return [float(value).hex() for value in np.r_[0.0, phase]]


def _endpoint(phase, incidence, branches):
    if not np.all(np.isfinite(phase)):
        raise ValueError("nonfinite optimizer endpoint")
    value, gradient, hessian = _fields(phase, incidence)
    gaps = incidence[:99] @ phase - 2 * np.pi * branches
    return {
        "phase_hex": _coordinates(phase),
        "potential": value,
        "gradient_infinity_norm": float(np.max(np.abs(gradient))),
        "tightened_chart_slack": float(np.pi - float(_MARGIN) - np.max(np.abs(gaps))),
        "grounded_hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
    }


def _screen(endpoint):
    return (
        endpoint["gradient_infinity_norm"] < 1e-7
        and endpoint["tightened_chart_slack"] > 2e-7
    )


def _certificate(phase, incidence, branches, edges):
    center = (Q(0),) + tuple(Q.from_float(float(value)) for value in phase)
    inverse_float = np.linalg.inv(_fields(phase, incidence)[2])
    inverse = tuple(
        tuple(Q(round(float(value) * 2**40), 2**40) for value in row)
        for row in inverse_float
    )
    certificate = _certify_organized_stationary(
        phase_center=center, edges=edges, radius=_RADIUS, inverse=inverse
    )
    pi = pi_interval()
    admitted = True
    gap_bounds = []
    for (a, b), branch in zip(edges[:99], branches):
        uncertainty = int(a != 0) + int(b != 0)
        gap = (
            I(center[b] - center[a])
            + I(-uncertainty * _RADIUS, uncertainty * _RADIUS)
            - 2 * int(branch) * pi
        )
        gap_bounds.append((gap.lo, gap.hi))
        admitted = admitted and -pi.lo + _MARGIN < gap.lo <= gap.hi < pi.lo - _MARGIN
    return {
        "radius": _RADIUS,
        "inverse": inverse,
        "eta": certificate.eta,
        "q0": certificate.q0,
        "qbox": certificate.qbox,
        "root_inclusion": certificate.root_inclusion_certified,
        "positive_definite": certificate.relative_hessian_positive,
        "marked_gap_bounds": gap_bounds,
        "marked_chamber_certified": bool(admitted),
        "stable_identity_certified": bool(
            certificate.stable_stationary_certified and admitted
        ),
    }


def _attempt(start, incidence, branches, edges):
    record = {"initial": _endpoint(start, incidence, branches)}
    try:
        return _attempt_body(start, incidence, branches, edges, record)
    except Exception as error:
        error.partial_record = record
        raise


def _attempt_body(start, incidence, branches, edges, record):
    limit = np.pi - float(_MARGIN)
    shift = 2 * np.pi * branches
    constraints = LinearConstraint(incidence[:99], shift - limit, shift + limit)
    oracle = _Oracle(incidence, 5000)
    try:
        result = minimize(
            oracle.value_gradient,
            start,
            method="SLSQP",
            jac=True,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        phase = result.x
        record["optimizer"] = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
        }
    except _BudgetExhausted as error:
        phase = oracle.last_phase
        record["optimizer"] = {"success": False, "message": str(error)}
    record["value_gradient_calls"] = oracle.calls
    record["endpoint"] = _endpoint(phase, incidence, branches)
    record["polish"] = None
    record["certificate"] = None
    if not _screen(record["endpoint"]):
        return record
    polish = _Oracle(incidence, 600)
    try:
        result = root(
            polish.residual,
            phase,
            jac=polish.jacobian,
            method="hybr",
            options={"xtol": 1e-10, "maxfev": 600},
        )
        phase = result.x
        polish_status = {
            "success": bool(result.success),
            "message": str(result.message),
        }
    except _BudgetExhausted as error:
        if polish.last_phase is not None:
            phase = polish.last_phase
        polish_status = {"success": False, "message": str(error)}
    endpoint = _endpoint(phase, incidence, branches)
    record["polish"] = {
        **polish_status,
        "residual_calls": polish.calls,
        "jacobian_calls": polish.jacobian_calls,
        "endpoint": endpoint,
    }
    if _screen(endpoint) and endpoint["grounded_hessian_minimum_eigenvalue"] > 1e-7:
        record["certificate"] = _certificate(phase, incidence, branches, edges)
    return record


def _git(*arguments):
    return subprocess.check_output(["git", *arguments], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != repository:
        raise RuntimeError("run from the declared repository root")
    imported_sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tnfr" or name.startswith("tnfr."):
            filename = getattr(module, "__file__", None)
            if filename:
                source = Path(filename).resolve()
                if not source.is_relative_to(repository / "src"):
                    raise RuntimeError(
                        f"imported source is outside the frozen tree: {name}"
                    )
                imported_sources[name] = str(source.relative_to(repository))
    if _git("status", "--porcelain"):
        raise RuntimeError("freeze source in a clean commit before evaluation")
    output = arguments.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evidence directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    edges, incidence, branches = _geometry()
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    metadata = {
        "schema": "organized_contact_stationary_search_v1",
        "source_revision": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD:src"),
        "imported_sources": imported_sources,
        "source_blobs": {
            path: _git("rev-parse", f"HEAD:{path}") for path in _SOURCE_PATHS
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "numeric_backend": backend.getvalue(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "node_order": list(range(54)),
        "edges": edges,
        "marked_branches": branches.tolist(),
        "seed": 20261010,
        "maximum_attempts": 24,
        "evaluation_status_at_start": "not_evaluated",
    }
    write_json_once(output / "attempt.json", encode_exact_tree(metadata))
    completed, certified = [], False
    try:
        for number, (origin, amplitude, scale, start) in enumerate(
            _starts(incidence, branches)
        ):
            write_json_once(
                output / f"start-{number:02d}.json",
                {
                    "number": number,
                    "donor_origin_index": origin,
                    "amplitude": amplitude,
                    "perturbation_scale": scale,
                    "phase_hex": _coordinates(start),
                },
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = _attempt(start, incidence, branches, edges)
            record.update(
                number=number,
                donor_origin_index=origin,
                amplitude=amplitude,
                perturbation_scale=scale,
                warnings=[str(w.message) for w in caught],
            )
            write_json_once(
                output / f"candidate-{number:02d}.json", encode_exact_tree(record)
            )
            completed.append(number)
            certified = bool(
                record["certificate"]
                and record["certificate"]["stable_identity_certified"]
            )
            print(
                number,
                record["endpoint"]["gradient_infinity_norm"],
                record["endpoint"]["tightened_chart_slack"],
                certified,
                flush=True,
            )
            if certified:
                break
        verdict = (
            "certified_stable_stationary_identity"
            if certified
            else "finite_search_without_certificate"
        )
        write_json_once(
            output / "outcome.json", {"verdict": verdict, "completed": completed}
        )
    except Exception as error:
        write_json_once(
            output / "incomplete.json",
            encode_exact_tree(
                {
                    "completed": completed,
                    "exception": type(error).__name__,
                    "message": str(error),
                    "partial_record": getattr(error, "partial_record", None),
                }
            ),
        )
        raise


if __name__ == "__main__":
    main()
