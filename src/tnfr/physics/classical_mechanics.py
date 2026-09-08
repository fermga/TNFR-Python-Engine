"""Explicit classical-mechanics adapters for TNFR-shaped state.

The canonical nodal equation is first order: ``dEPI/dt = nu_f * DeltaNFR``.
Reading an EPI coordinate as position and pressure as force therefore makes
``nu_f`` a mobility. This module additionally offers selected second-order
classical embeddings. They store coordinates and velocities in EPI-shaped
arrays, optionally record ``nu_f = 1/m`` as adapter metadata, and obtain forces
from the caller's Lagrangian or Hamiltonian.

These assignments are conventions of the adapter. They do not derive inertia,
gravity, friction or harmonic forces from the nodal equation, the tetrad or the
13 operators. The auxiliary symplectic substrate is also a separate declared
Hamiltonian model; it supplies no canonical identity ``m = 1/nu_f``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# Canonical constants and keys
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY

from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np


def _finite_real_vector(
    value: Any,
    label: str,
    *,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
) -> np.ndarray:
    """Materialize one finite real adapter vector with an optional shape."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{label} must be a finite real vector") from exc
    if raw.dtype.kind not in "iuf":
        raise TNFRValueError(f"{label} must be a finite real vector")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise TNFRValueError(f"{label} must be a nonempty one-dimensional vector")
    if shape is not None and array.shape != shape:
        raise TNFRValueError(f"{label} must have shape {shape!r}")
    if not bool(np.all(np.isfinite(array))):
        raise TNFRValueError(f"{label} must contain only finite values")
    if positive and bool(np.any(array <= 0.0)):
        raise TNFRValueError(f"{label} must contain only strictly positive values")
    return array


def _finite_real_scalar(value: Any, label: str, *, positive: bool = False) -> float:
    """Materialize one finite real scalar without lossy coercion."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TNFRValueError(f"{label} must be a finite real scalar") from exc
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TNFRValueError(f"{label} must be a finite real scalar")
    normalized = float(raw)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "positive " if positive else ""
        raise TNFRValueError(f"{label} must be a finite {qualifier}real scalar")
    return normalized


def _adapter_frequency(masses: np.ndarray) -> tuple[float, float]:
    """Return a stable mean-mass reference and its representable reciprocal."""

    scale = float(np.max(masses))
    mass_reference = scale * float(np.mean(masses / scale))
    frequency = 1.0 / mass_reference
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise TNFRValueError(
            "inverse-mean-mass adapter frequency is not representable"
        )
    return mass_reference, frequency


def _adapter_metadata(kind: str, mass_reference: float) -> dict[str, Any]:
    """Describe the classical-only interpretation of an adapter payload."""

    return {
        "kind": kind,
        "force_bridge_supplied": False,
        "nu_f_semantics": "inverse_mean_mass_adapter_only",
        "mass_reduction": "arithmetic_mean_single_node_summary",
        "mass_reference": mass_reference,
        "dnfr_semantics": "zero_placeholder_no_force_bridge",
        "canonical_coherence_available": False,
    }


@dataclass
class GeneralizedCoordinateSystem:
    """Represents a system in generalized coordinates (q, p)."""

    q: np.ndarray  # Generalized coordinates
    p: np.ndarray | None = None  # Generalized momenta (for Hamiltonian)
    q_dot: np.ndarray | None = None  # Generalized velocities (for Lagrangian)
    masses: np.ndarray | None = None  # Masses associated with coordinates

    def __post_init__(self) -> None:
        self.q = _finite_real_vector(self.q, "q")
        if self.p is not None:
            self.p = _finite_real_vector(self.p, "p", shape=self.q.shape)
        if self.q_dot is not None:
            self.q_dot = _finite_real_vector(
                self.q_dot, "q_dot", shape=self.q.shape
            )
        if self.masses is None:
            self.masses = np.ones_like(self.q, dtype=float)
        else:
            self.masses = _finite_real_vector(
                self.masses,
                "masses",
                shape=self.q.shape,
                positive=True,
            )

    @property
    def dimension(self) -> int:
        return len(self.q)


class ClassicalMechanicsMapper:
    """Build TNFR-shaped payloads from declared classical-model state."""

    @staticmethod
    def lagrangian_to_tnfr(
        L: Callable[[np.ndarray, np.ndarray, float], float],
        system: GeneralizedCoordinateSystem,
        t: float = 0.0,
    ) -> dict[str, Any]:
        """
        Maps a Lagrangian L(q, q_dot, t) to a TNFR Nodal State.

        Args:
            L: Lagrangian function L(q, q_dot, t) -> float (Energy)
            system: The generalized coordinate system state
            t: Current time

        Returns:
            dict containing TNFR nodal attributes:
            - EPI: Combined state vector [q, q_dot]
            - νf: Structural frequency assigned by the adapter convention
            - ΔNFR: zero placeholder; the caller must supply a force bridge
        """
        if system.q_dot is None:
            raise TNFRValueError(
                "Lagrangian mapping requires generalized velocities (q_dot)."
            )

        # A single node needs a scalar nu_f, so this adapter uses reciprocal
        # mean mass. Per-body mappings belong in a graph adapter.
        time = _finite_real_scalar(t, "t")
        mass_ref, nu_f = _adapter_frequency(system.masses)

        epi_vector = np.concatenate([system.q, system.q_dot])

        # The mapper evaluates L for provenance but does not differentiate it.
        # Pressure therefore remains an explicit zero placeholder. A caller that
        # needs Euler-Lagrange dynamics must provide and label that force bridge.

        lagrangian_value = _finite_real_scalar(
            L(system.q, system.q_dot, time), "Lagrangian value"
        )
        return {
            EPI_PRIMARY: epi_vector,
            VF_PRIMARY: nu_f,
            "classical_L": lagrangian_value,
            # No force law is inferred from the Lagrangian callable.
            DNFR_PRIMARY: np.zeros_like(epi_vector),
            "classical_adapter": _adapter_metadata(
                "lagrangian_state_embedding", mass_ref
            ),
        }

    @staticmethod
    def hamiltonian_to_tnfr(
        H: Callable[[np.ndarray, np.ndarray, float], float],
        system: GeneralizedCoordinateSystem,
        t: float = 0.0,
    ) -> dict[str, Any]:
        """
        Maps a Hamiltonian H(q, p, t) to a TNFR Nodal State.

        Args:
            H: Hamiltonian function H(q, p, t) -> float (Energy)
            system: The generalized coordinate system state
            t: Current time

        Returns:
            dict containing TNFR nodal attributes.
        """
        if system.p is None:
            raise TNFRValueError(
                "Hamiltonian mapping requires generalized momenta (p)."
            )

        time = _finite_real_scalar(t, "t")
        mass_ref, nu_f = _adapter_frequency(system.masses)

        # Decode p as velocity through the declared classical masses.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            q_dot = system.p / system.masses
        q_dot = _finite_real_vector(q_dot, "derived q_dot", shape=system.q.shape)
        epi_vector = np.concatenate([system.q, q_dot])

        # H is retained as adapter provenance. It is not structural C(t) or the
        # tetrad potential, and this mapping supplies no Hamiltonian force law.

        hamiltonian_value = _finite_real_scalar(
            H(system.q, system.p, time), "Hamiltonian value"
        )
        return {
            EPI_PRIMARY: epi_vector,
            VF_PRIMARY: nu_f,
            "classical_H": hamiltonian_value,
            DNFR_PRIMARY: np.zeros_like(epi_vector),
            "classical_adapter": _adapter_metadata(
                "hamiltonian_state_embedding", mass_ref
            ),
        }

    @staticmethod
    def equations_of_motion_to_operators(
        forces: np.ndarray, masses: np.ndarray
    ) -> list[str]:
        """Classify force presence into a grammar-valid illustrative word.

        The returned word records only whether the supplied finite classical
        force array is zero. It does not encode force magnitude, direction or
        a derivation of classical motion. Nonzero input selects an exploratory
        word with OZ and its required IL handler; zero input selects a closed
        source/coherence/silence word.
        """

        try:
            raw_forces = np.asarray(forces)
            raw_masses = np.asarray(masses)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TNFRValueError(
                "forces and masses must be real numeric arrays"
            ) from exc
        if (
            raw_forces.dtype.kind not in "iuf"
            or raw_masses.dtype.kind not in "iuf"
        ):
            raise TNFRValueError("forces and masses must be real numeric arrays")
        try:
            force_values = np.asarray(forces, dtype=float)
            mass_values = np.asarray(masses, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TNFRValueError(
                "forces and masses must be real numeric arrays"
            ) from exc
        if force_values.ndim == 0 or mass_values.ndim != 1:
            raise TNFRValueError(
                "forces must have a leading body axis and masses must be "
                "one-dimensional"
            )
        if mass_values.size == 0 or force_values.shape[0] != mass_values.size:
            raise TNFRValueError(
                "the leading force dimension must match the nonempty mass vector"
            )
        if not np.all(np.isfinite(force_values)) or not np.all(
            np.isfinite(mass_values)
        ):
            raise TNFRValueError("forces and masses must be finite")
        if np.any(mass_values <= 0.0):
            raise TNFRValueError("masses must be strictly positive")

        if bool(np.any(force_values != 0.0)):
            return ["emission", "dissonance", "coherence", "silence"]
        return ["emission", "coherence", "silence"]

    @staticmethod
    def state_vector_to_generalized(
        epi: np.ndarray, nu_f: float
    ) -> GeneralizedCoordinateSystem:
        """Decode an adapter EPI vector using the declared ``nu_f=1/m`` map."""

        values = _finite_real_vector(epi, "adapter EPI")
        if values.size % 2:
            raise TNFRValueError("adapter EPI must contain equally sized q and q_dot")
        frequency = _finite_real_scalar(nu_f, "nu_f", positive=True)
        n = values.size // 2
        q = values[:n]
        q_dot = values[n:]
        mass = 1.0 / frequency
        if not math.isfinite(mass) or mass <= 0.0:
            raise TNFRValueError(
                "inverse-frequency adapter mass is not representable"
            )
        p = q_dot * mass

        return GeneralizedCoordinateSystem(
            q=q, q_dot=q_dot, p=p, masses=np.full_like(q, mass)
        )


class ClassicalForceTranslator:
    """Return honest descriptions of legacy classical-adapter comparisons."""

    @staticmethod
    def gravity_to_tnfr() -> str:
        """Describe gravity as an externally supplied force law.

        Neither phase synchronization nor the structural potential derives
        Newtonian gravity in this adapter.
        """

        return "External Newtonian force adapter (no canonical TNFR gravity map)"

    @staticmethod
    def friction_to_tnfr() -> str:
        """Describe friction as a separately configured dissipative law.

        IL contracts structural pressure under its own contract; it does not
        supply a velocity-dependent classical friction law by itself.
        """

        return "Configured dissipative adapter (IL is only a comparison)"

    @staticmethod
    def harmonic_restoring_to_tnfr() -> str:
        """Describe a harmonic force as an explicit restoring-pressure law.

        A phase gradient is diagnostic and does not generate Hooke's law unless
        the adapter defines that bridge.
        """

        return "Configured harmonic adapter (phase gradient is diagnostic)"

    @staticmethod
    def compute_poisson_bracket(
        f: Callable[[GeneralizedCoordinateSystem], float],
        g: Callable[[GeneralizedCoordinateSystem], float],
        system: GeneralizedCoordinateSystem,
        epsilon: float = 1e-5,
    ) -> float:
        """
        Compute the classical Poisson bracket ``{f, g}`` by finite differences.

        {f, g} = Σ (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)

        The result belongs to the declared classical phase-space adapter. Under
        the usual Hamiltonian regularity assumptions, ``{f, H}=0`` makes ``f``
        constant along that Hamiltonian flow; it does not prove a TNFR
        structural invariant for engine operator trajectories.

        Args:
            f: First observable function.
            g: Second observable function.
            system: Current state.
            epsilon: Finite difference step size.

        Returns:
            Value of the Poisson Bracket.
        """
        epsilon = _finite_real_scalar(epsilon, "epsilon", positive=True)

        n = system.dimension
        bracket = 0.0

        # We need to perturb q and p.
        # Since GeneralizedCoordinateSystem is immutable-ish (dataclass),
        # we create copies.

        # Helper to evaluate gradient
        def gradient(func, sys, var_name, idx):
            original = getattr(sys, var_name)[idx]

            try:
                getattr(sys, var_name)[idx] = original + epsilon
                val_plus = _finite_real_scalar(
                    func(sys), f"{var_name} positive-shift observable"
                )
                getattr(sys, var_name)[idx] = original - epsilon
                val_minus = _finite_real_scalar(
                    func(sys), f"{var_name} negative-shift observable"
                )
            finally:
                getattr(sys, var_name)[idx] = original

            derivative = (val_plus - val_minus) / (2 * epsilon)
            if not math.isfinite(derivative):
                raise TNFRValueError("Poisson-bracket derivative must remain finite")
            return derivative

        # We need mutable arrays for this to work efficiently,
        # or we construct new systems.
        # The dataclass fields are numpy arrays, which are mutable.
        # So we can modify in place and restore.

        if system.p is None:
            raise TNFRValueError("Poisson Bracket requires momenta (p).")

        for i in range(n):
            df_dq = gradient(f, system, "q", i)
            dg_dp = gradient(g, system, "p", i)

            df_dp = gradient(f, system, "p", i)
            dg_dq = gradient(g, system, "q", i)

            bracket += (df_dq * dg_dp) - (df_dp * dg_dq)
            if not math.isfinite(bracket):
                raise TNFRValueError("Poisson bracket must remain finite")

        return float(bracket)
