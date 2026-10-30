"""Integer character and Gauss-sum helpers retain their optional SymPy boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import number_theory


@pytest.mark.parametrize("p", [2, 3, np.int64(5), np.uint64(7), 11])
def test_primitive_root_generates_the_prime_unit_group(p):
    root = number_theory.get_primitive_root(p)
    modulus = int(p)
    units = set(range(1, modulus))
    assert root is not None
    assert {pow(root, power, modulus) for power in range(modulus - 1)} == units
    assert all(
        {pow(candidate, power, modulus) for power in range(modulus - 1)} != units
        for candidate in range(1, root)
    )


@pytest.mark.parametrize("p", [-3, 0, 1, 4, np.int64(9), 561])
def test_nonprime_moduli_preserve_explicit_unavailability(p):
    assert number_theory.get_primitive_root(p) is None
    for function in (
        number_theory.compute_dirichlet_characters,
        number_theory.compute_gauss_sums,
        number_theory.AdelicOperator,
    ):
        with pytest.raises(TNFRValueError, match="must be prime"):
            function(p)


@pytest.mark.parametrize("p", [True, np.bool_(False), 3.0, 3.5, "3", None])
@pytest.mark.parametrize(
    "function",
    (
        number_theory.get_primitive_root,
        number_theory.compute_dirichlet_characters,
        number_theory.compute_gauss_sums,
        number_theory.AdelicOperator,
    ),
)
def test_prime_modulus_admission_precedes_coercion(function, p):
    with pytest.raises(TypeError, match="must be an integer"):
        function(p)


def test_fallback_primality_never_materializes_a_float_root(monkeypatch):
    monkeypatch.setattr(number_theory, "HAS_SYMPY", False)
    # Divisibility by three settles this integer immediately; sqrt(float(n))
    # would overflow before trial division even starts.
    assert not number_theory._integer_is_prime(9 * (2**2048 + 1))


def test_character_helpers_work_in_a_cold_process_without_sympy(
    source_tree_environment,
):
    code = """
import importlib.abc
import sys

class WithoutSympy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sympy" or fullname.startswith("sympy."):
            raise ModuleNotFoundError("SymPy intentionally unavailable", name="sympy")

sys.meta_path.insert(0, WithoutSympy())
import numpy as np
from tnfr.errors import TNFRValueError
from tnfr.mathematics import number_theory as nt

assert not nt.HAS_SYMPY
assert not any(name == "sympy" or name.startswith("sympy.") for name in sys.modules)
assert nt.get_primitive_root(2) == 1
assert nt.get_primitive_root(np.int64(3)) == 2
assert nt.get_primitive_root(np.uint64(5)) == 2
for composite in (4, np.int64(9), 561):
    assert nt.get_primitive_root(composite) is None
    for function in (nt.compute_dirichlet_characters, nt.compute_gauss_sums,
                     nt.AdelicOperator):
        try:
            function(composite)
        except TNFRValueError:
            pass
        else:
            raise AssertionError("A composite modulus must be rejected")

# On F_3^*, the two characters are 1 and the sign character. Their normalized
# additive sums are -1/sqrt(3) and i, respectively.
expected_characters = np.array([[1, 1], [1, -1]], dtype=complex)
expected_sums = np.array([-1 / np.sqrt(3), 1j])
np.testing.assert_allclose(nt.compute_dirichlet_characters(np.int64(3)),
                           expected_characters, atol=1e-14)
np.testing.assert_allclose(nt.compute_gauss_sums(np.int64(3)),
                           expected_sums, atol=1e-14)
operator = nt.AdelicOperator(np.int64(3))
assert type(operator.p) is int
state = np.array([2 - 1j, 3 + 4j])
np.testing.assert_allclose(operator.apply(state), expected_sums * state, atol=1e-14)
network = nt.ArithmeticTNFRNetwork(max_number=11)
assert {n for n in network.graph if network.graph.nodes[n]["is_prime"]} == {
    2, 3, 5, 7, 11
}
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
