"""Examples for the TNFR Math Suite modules."""

from pathlib import Path
import sys
from sympy import symbols

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnfr.types import Glyph  # noqa: E402
from tnfr.math import (  # noqa: E402
    symbolic,
    grammar_validators,
    fields_symbolic,
    optimizer,
)


def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*60)
    print(f"// {title.upper()}")
    print("="*60)


def main():
    """Main function to run all demonstrations."""

    # --- 1. Symbolic Module Demonstration ---
    print_header("1. Symbolic Module (symbolic.py)")
    
    nodal_eq = symbolic.get_nodal_equation()
    print("\n[Symbolic] Nodal Equation:")
    print(symbolic.pretty_print(nodal_eq))

    convergent, explanation, value = symbolic.check_convergence_exponential(
        -0.1,
        10.0,
    )
    print(
        "\n[Symbolic] Convergence (λ=-0.1, T=10): "
        f"{convergent}, {explanation}, integral={value:.4f}"
    )

    convergent, explanation, value = symbolic.check_convergence_exponential(
        0.1,
        10.0,
    )
    print(
        "[Symbolic] Convergence (λ=0.1, T=10): "
        f"{convergent}, {explanation}, integral={value:.4f}"
    )

    # --- 2. Grammar Validators Demonstration ---
    print_header("2. Grammar Validators (grammar_validators.py)")

    # A valid sequence
    seq_good = [Glyph.AL, Glyph.OZ, Glyph.IL, Glyph.SHA]
    print(
        "\n[Validator] Analyzing good sequence: "
        f"{[g.value for g in seq_good]}"
    )
    converges, growth, conv_exp = (
        grammar_validators.verify_convergence_for_sequence(seq_good)
    )
    print(f"  - U2 Convergence: {converges}, λ={growth:.2f}. {conv_exp}")
    is_safe, risk, risk_exp = (
        grammar_validators.verify_bifurcation_risk_for_sequence(seq_good)
    )
    print(
        "  - U4 Bifurcation: "
        f"safe={is_safe}, risk={risk:.2f}. {risk_exp}"
    )

    # An invalid sequence
    seq_bad = [Glyph.AL, Glyph.OZ, Glyph.VAL]
    print(
        "\n[Validator] Analyzing bad sequence: "
        f"{[g.value for g in seq_bad]}"
    )
    converges, growth, conv_exp = (
        grammar_validators.verify_convergence_for_sequence(seq_bad)
    )
    print(f"  - U2 Convergence: {converges}, λ={growth:.2f}. {conv_exp}")

    # --- 3. Symbolic Fields Demonstration ---
    print_header("3. Symbolic Fields (fields_symbolic.py)")
    
    # Get and print the structural potential field equation
    phi_s_eq, phi_s_interp = (
        fields_symbolic.get_structural_potential_field_symbolic()
    )
    alpha_sym = symbols('alpha', real=True, positive=True)
    phi_s_eq_alpha2 = phi_s_eq.subs(alpha_sym, 2.0)

    print("\n[Fields] Structural Potential Field (Φ_s) with α=2.0:")
    print(symbolic.pretty_print(phi_s_eq_alpha2))
    print(f"Interpretation: {phi_s_interp}")

    # --- 4. Optimizer Demonstration ---
    print_header("4. Optimizer (optimizer.py)")

    initial_seq = [Glyph.AL, Glyph.OZ]
    possible_glyphs = [Glyph.IL, Glyph.SHA, Glyph.UM, Glyph.RA]
    print(
        "\n[Optimizer] Starting with a bad sequence: "
        f"{[g.value for g in initial_seq]}"
    )
    print("[Optimizer] Goal: Find a grammatically valid and longer sequence.")

    best_seq, best_score = optimizer.find_optimal_sequence_greedy(
        initial_sequence=initial_seq,
        possible_glyphs=possible_glyphs,
        objective_fn=optimizer.sample_objective_function,
        max_iterations=10
    )

    print(f"\n[Optimizer] Found best sequence: {[g.value for g in best_seq]}")
    print(f"[Optimizer] Best score: {best_score:.2f}")

    # Verify the result is now valid
    converges, _, _ = grammar_validators.verify_convergence_for_sequence(
        best_seq
    )
    is_safe, risk, _ = (
        grammar_validators.verify_bifurcation_risk_for_sequence(best_seq)
    )
    print(
        "[Optimizer] Verification: "
        f"Converges={converges}, Safe={is_safe}, Risk={risk:.2f}"
    )


if __name__ == "__main__":
    main()
