"""One deterministic, uncertified candidate for the matched selection study.

This module must not be imported or called before its prospective source freeze.
It selects a candidate in binary64; it is not a basin or trajectory certificate.
There is no scan, retry with a different epsilon, or outcome-dependent tuning.

Coordinates are y=(a-2*pi/3,b-pi/3,A,B), with e=w=1/2,beta=kappa=1.
The capacity preparation is s=t=1+epsilon,u=1-4*epsilon. Holding it fixed
multiplies the complete reflected native field by 1+epsilon, so the held
stable boundary is exactly the unit-capacity boundary h0, not just its tangent.

Let lambda be the positive native saddle eigenvalue, ru its right eigenvector
with first component one, ell its left eigenvector with ell@ru=1, and
Ps=I-ru@ell. Q has orthonormal columns spanning Es=ker(ell), and S=Q.T@J@Q.
Writing E=diag(1,0,1,0), L=ell@E@J@Q and c_pm=(5+-sqrt(5))/4, the initial
capacity direction (1,1) gives

    (s-t)(t) / epsilon = sqrt(5)*(exp(-c_minus*t)-exp(-c_plus*t)).

Its dynamic mixed boundary row, relative to the exactly unchanged held
boundary, is

    Delta = -(5/2)*L@((lambda+c_minus)*I-S)^(-1)
                     @((lambda+c_plus)*I-S)^(-1).

The common native quadratic boundary cannot be omitted. If the homogeneous
quadratic native field is Q2 and v=Q@v_s, write ell@Q2(v)=v_s.T@Bq@v_s.
Then h20(v)=v_s.T@H@v_s, where

    S.T@H + H@S - lambda*H = Bq.

Equivalently h20 is minus the convergent integral of
exp(-lambda*t)*ell@Q2(exp(J*t)@v). The sign in the Sylvester equation below
therefore matters.

Extend Delta to a row annihilating ru by delta_full=Delta@Q.T@Ps. The stable
direction Ps@delta_full.T is normalized in the ambient Euclidean norm. This
projection is essential: the raw transpose need not lie in Es because Ps is
oblique. Its mixed value is strictly positive in exact arithmetic.

The selected native state is epsilon*v + epsilon**2*(h20+Delta(v)/2)*ru.
Its second-order prediction is held -> positive twist, dynamic -> consensus.
No remainder bound for the two boundary graphs is claimed here. In particular,
epsilon=1/32 is merely the prospectively fixed candidate amplitude. Only the
separate paired, regular interval transit can certify the actual finite seed.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
from scipy.linalg import null_space, solve, solve_sylvester
from scipy.optimize import brentq

EPSILON = Fraction(1, 32)
RESIDUAL_LIMIT = 1e-12


def _pencil(value: float) -> np.ndarray:
    """Native symmetric saddle pencil, using candidate-only binary64 pi."""
    squared = value * value
    inverse = 1.0 / (8.0 * math.pi * math.pi)
    return np.array(
        [
            [2.0 * squared / 5.0 + value / 6.0 - inverse, squared / 5.0 - inverse],
            [
                squared / 5.0 - inverse,
                3.0 * squared / 5.0 + value / 4.0 + 2.0 * inverse,
            ],
        ],
        dtype=np.float64,
    )


def _pencil_determinant(value: float) -> float:
    matrix = _pencil(value)
    return float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] ** 2)


def _jacobian() -> np.ndarray:
    rate = 1.0 / (2.0 * math.pi)
    return np.array(
        [
            [0.0, 0.0, 3.0 * rate, -rate],
            [0.0, 0.0, -rate, 2.0 * rate],
            [rate / 2.0, rate / 2.0, -0.5, 1.0 / 6.0],
            [rate / 2.0, -rate, 0.25, -0.5],
        ],
        dtype=np.float64,
    )


def _native_quadratic_form(left: np.ndarray) -> np.ndarray:
    """Symmetric B4 with y.T@B4@y == ell@Q2(y), not a finite difference.

    With y=(alpha,beta_dev,A,B), q=3*A-B and r=2*B-A, the exact
    homogeneous quadratic vector is sqrt(3)/(4*pi) times
    (-(alpha+beta_dev)*q, alpha*r, -2*alpha*(alpha+beta_dev), 0).
    """
    coefficient = math.sqrt(3.0) / (4.0 * math.pi)
    result = np.zeros((4, 4), dtype=np.float64)
    result[0, 0] = -2.0 * coefficient * left[2]
    result[0, 1] = -coefficient * left[2]
    result[0, 2] = coefficient * (-3.0 * left[0] - left[1]) / 2.0
    result[0, 3] = coefficient * (left[0] + 2.0 * left[1]) / 2.0
    result[1, 2] = -3.0 * coefficient * left[0] / 2.0
    result[1, 3] = coefficient * left[0] / 2.0
    for row in range(4):
        for column in range(row + 1, 4):
            result[column, row] = result[row, column]
    return result


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value)))


def make_candidate() -> dict:
    """Return one exact represented seed plus finite candidate diagnostics.

    ``initial`` is (q,r,a,b,s,t), all Fraction instances. Native A/B are
    materialized from binary64 exactly once, then q/r are formed in rational
    arithmetic. The matching inverse embedding therefore recovers exactly
    those same A/B; q/r are never rounded independently a second time.

    The function has no file writes. A failed arithmetic guard aborts this
    candidate and must not trigger amplitude changes or an undocumented retry.
    """
    # The pencil is negative at zero and positive definite at one. Its unique
    # positive root is established analytically in the saddle owner. The fixed
    # bracket and iteration budget below select only its binary64 approximation.
    eigenvalue, root_report = brentq(
        _pencil_determinant,
        0.0,
        1.0,
        xtol=5e-15,
        rtol=1e-14,
        maxiter=100,
        full_output=True,
        disp=True,
    )
    eigenvalue = float(eigenvalue)
    if not root_report.converged or not 0.0 < eigenvalue < 1.0:
        raise ArithmeticError("The declared positive-root solve did not resolve")

    pencil = _pencil(eigenvalue)
    slope = float(-pencil[0, 1] / pencil[1, 1])
    if not 0.0 < slope < 0.5:
        raise ArithmeticError("The declared unstable orientation did not resolve")
    factor = 2.0 * math.pi * eigenvalue / 5.0
    unstable = np.array(
        [1.0, slope, factor * (2.0 + slope), factor * (1.0 + 3.0 * slope)],
        dtype=np.float64,
    )
    left_raw = np.array(
        [
            (1.0 + slope) / (4.0 * math.pi * eigenvalue),
            (1.0 - 2.0 * slope) / (4.0 * math.pi * eigenvalue),
            1.0,
            slope,
        ],
        dtype=np.float64,
    )
    normalization = float(left_raw @ unstable)
    if not normalization > 0.0:
        raise ArithmeticError("The left/right normalization is not positive")
    left = left_raw / normalization
    jacobian = _jacobian()
    stable_projection = np.eye(4) - np.outer(unstable, left)
    basis = null_space(left.reshape(1, 4), rcond=1e-14)
    if basis.shape != (4, 3):
        raise ArithmeticError("The native stable hyperplane has wrong dimension")
    stable = basis.T @ jacobian @ basis

    port_projection = np.diag([1.0, 0.0, 1.0, 0.0])
    mixed_driver = left @ port_projection @ jacobian @ basis
    capacity_low = (5.0 - math.sqrt(5.0)) / 4.0
    capacity_high = (5.0 + math.sqrt(5.0)) / 4.0
    identity = np.eye(3)
    lower_resolvent = solve(
        (eigenvalue + capacity_low) * identity - stable,
        identity,
        assume_a="gen",
        check_finite=True,
    )
    upper_resolvent = solve(
        (eigenvalue + capacity_high) * identity - stable,
        identity,
        assume_a="gen",
        check_finite=True,
    )
    delta_row = -2.5 * mixed_driver @ lower_resolvent @ upper_resolvent
    delta_full = delta_row @ basis.T @ stable_projection
    direction_raw = stable_projection @ delta_full
    direction_norm = float(np.linalg.norm(direction_raw))
    if not math.isfinite(direction_norm) or not direction_norm > 0.0:
        raise ArithmeticError("The declared mixed stable direction vanished")
    direction = direction_raw / direction_norm
    direction_stable = basis.T @ direction
    separation = float(delta_row @ direction_stable)
    if not math.isfinite(separation) or not separation > 0.0:
        raise ArithmeticError("The declared mixed separation is not positive")

    quadratic_full = _native_quadratic_form(left)
    quadratic_stable = basis.T @ quadratic_full @ basis
    shifted = stable - (eigenvalue / 2.0) * identity
    common_matrix = solve_sylvester(shifted.T, shifted, quadratic_stable)
    common_quadratic = float(direction_stable @ common_matrix @ direction_stable)
    midpoint = common_quadratic + separation / 2.0
    amplitude = float(EPSILON)
    deviation = amplitude * direction + amplitude**2 * midpoint * unstable

    residuals = {
        "pencil_determinant": abs(_pencil_determinant(eigenvalue)),
        "right_eigenvector": _max_abs(jacobian @ unstable - eigenvalue * unstable),
        "left_eigenvector": _max_abs(left @ jacobian - eigenvalue * left),
        "normalization": abs(float(left @ unstable) - 1.0),
        "projection_idempotence": _max_abs(
            stable_projection @ stable_projection - stable_projection
        ),
        "projection_commutation": _max_abs(
            jacobian @ stable_projection - stable_projection @ jacobian
        ),
        "basis_orthonormality": _max_abs(basis.T @ basis - identity),
        "basis_in_stable_hyperplane": _max_abs(left @ basis),
        "stable_intertwining": _max_abs(jacobian @ basis - basis @ stable),
        "direction_in_stable_hyperplane": abs(float(left @ direction)),
        "direction_normalization": abs(float(direction @ direction) - 1.0),
        "separation_identity": abs(
            separation - float(delta_full @ delta_full) / direction_norm
        ),
        "common_sylvester": _max_abs(
            stable.T @ common_matrix
            + common_matrix @ stable
            - eigenvalue * common_matrix
            - quadratic_stable
        ),
        "common_symmetry": _max_abs(common_matrix - common_matrix.T),
    }
    if any(
        not math.isfinite(value) or value > RESIDUAL_LIMIT
        for value in residuals.values()
    ):
        raise ArithmeticError("A declared candidate arithmetic residual failed")

    materialized = {
        "A": float(deviation[2]),
        "B": float(deviation[3]),
        "a": float(2.0 * math.pi / 3.0 + deviation[0]),
        "b": float(math.pi / 3.0 + deviation[1]),
        "s": float(1.0 + amplitude),
        "t": float(1.0 + amplitude),
    }
    if any(not math.isfinite(value) for value in materialized.values()):
        raise ArithmeticError("A declared source coordinate is nonfinite")
    exact = {name: Fraction.from_float(value) for name, value in materialized.items()}
    q = 3 * exact["A"] - exact["B"]
    r = 2 * exact["B"] - exact["A"]
    capacity_center = Fraction(5) - 2 * exact["s"] - 2 * exact["t"]
    if (
        exact["s"] != Fraction(33, 32)
        or exact["t"] != Fraction(33, 32)
        or capacity_center != Fraction(7, 8)
    ):
        raise ArithmeticError("The declared exact mean-one capacity profile changed")

    diagnostics = {
        "arithmetic": "binary64 candidate construction only; not a rigorous manifold bound",
        "epsilon": amplitude,
        "positive_eigenvalue": eigenvalue,
        "root_iterations": int(root_report.iterations),
        "root_function_calls": int(root_report.function_calls),
        "root_bracket": [0.0, 1.0],
        "root_iteration_limit": 100,
        "unstable_phase_slope": slope,
        "right_unstable": unstable.tolist(),
        "left_unstable": left.tolist(),
        "stable_projection": stable_projection.tolist(),
        "stable_basis": basis.tolist(),
        "stable_matrix": stable.tolist(),
        "capacity_rates": [capacity_low, capacity_high],
        "mixed_driver": mixed_driver.tolist(),
        "dynamic_mixed_row": delta_row.tolist(),
        "ambient_separation_row": delta_full.tolist(),
        "stable_direction": direction.tolist(),
        "common_quadratic_matrix": common_matrix.tolist(),
        "common_quadratic_value": common_quadratic,
        "mixed_separation_value": separation,
        "midpoint_unstable_coefficient": midpoint,
        "native_deviation": deviation.tolist(),
        "materialized_native_float64": materialized,
        "residual_limit": RESIDUAL_LIMIT,
        "residuals": residuals,
        "finite_seed_admitted": False,
        "required_validation": "Both exact represented IVPs must pass their declared regular interval transit and actual capture gates.",
    }
    json.dumps(diagnostics, allow_nan=False)
    return {
        "initial": (q, r, exact["a"], exact["b"], exact["s"], exact["t"]),
        "exact_native": exact,
        "capacity_center": capacity_center,
        "epsilon": EPSILON,
        "diagnostics": diagnostics,
        "prediction": {"held": 1, "dynamic": 0},
        "prediction_status": "conditional second-order prediction; finite epsilon awaits paired certificate",
    }
