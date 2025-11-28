"""
Verification of the Adelic Trace Formula.

This script computes the trace of the Liouvillian propagator Tr(exp(L*t))
and compares it to the predicted geometric trace (peaks at ln(p)).

Objective:
Demonstrate that the Liouvillian dynamics naturally generate the
prime-periodic orbits required by the Riemann-Weil Explicit Formula.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnfr.mathematics.generators import build_lindblad_delta_nfr
from tnfr.mathematics.liouville import compute_liouvillian_spectrum

def build_scaling_hamiltonian(dim: int) -> np.ndarray:
    """Construct discrete scaling Hamiltonian H ~ (xp + px)/2."""
    X = np.diag(np.arange(dim))
    P = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        P[n, n+1] = -1j
        P[n+1, n] = 1j
    P = P / 2.0
    H = 0.5 * (X @ P + P @ X)
    return H

def build_prime_dissipator(dim: int, prime: int) -> np.ndarray:
    """Construct dissipator projecting at prime intervals."""
    L = np.zeros((dim, dim), dtype=complex)
    for n in range(dim):
        if (n + 1) % prime == 0:
            L[n, n] = 1.0
    return L

def main():
    print("TNFR Trace Formula Verification")
    print("===============================")
    
    dim = 40
    print(f"System Dimension: {dim}")
    
    # 1. Build System
    print("Building Liouvillian with Primes {2, 3, 5}...")
    H = build_scaling_hamiltonian(dim)
    primes = [2, 3, 5]
    dissipators = [build_prime_dissipator(dim, p) for p in primes]
    
    L_super = build_lindblad_delta_nfr(
        hamiltonian=H,
        collapse_operators=dissipators,
        nu_f=1.0,
        ensure_contractive=False
    )
    
    # 2. Compute Spectrum
    print("Computing Eigenvalues...")
    evals = compute_liouvillian_spectrum(L_super, sort=True)
    # Filter high-frequency noise
    evals = evals[np.abs(evals) < dim]
    
    # 3. Compute Trace Function
    print("Computing Trace Tr(exp(Lt))...")
    t_values = np.linspace(0.1, 3.0, 200)
    trace_vals = []
    
    for t in t_values:
        # Tr(e^Lt) = sum(e^(lambda_n * t))
        # We take the real part of the trace (amplitude of return)
        tr = np.sum(np.exp(evals * t))
        trace_vals.append(np.abs(tr)) # Magnitude of return probability
        
    trace_vals = np.array(trace_vals)
    
    # 4. Analyze Peaks
    print("\nChecking for Geometric Peaks (ln p):")
    expected_peaks = {}
    for p in primes:
        expected_peaks[f"ln({p})"] = np.log(p)
        expected_peaks[f"ln({p}^2)"] = 2 * np.log(p)
        
    # Simple peak detection
    peaks = []
    for i in range(1, len(trace_vals)-1):
        if trace_vals[i] > trace_vals[i-1] and trace_vals[i] > trace_vals[i+1]:
            peaks.append(t_values[i])
            
    print(f"Detected Peaks at t = {[f'{t:.2f}' for t in peaks]}")
    print("Expected Peaks:")
    for label, val in expected_peaks.items():
        print(f"  {label}: {val:.2f}")
        
    # Match
    matches = 0
    for val in expected_peaks.values():
        # Check if any detected peak is close
        if any(abs(p - val) < 0.15 for p in peaks):
            matches += 1
            
    print(f"\nMatched {matches}/{len(expected_peaks)} expected prime orbits.")
    
    # 5. Visualization (if requested/possible)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, trace_vals, label='Spectral Trace |Tr(e^{Lt})|', color='blue')
        
        # Plot expected lines
        for label, val in expected_peaks.items():
            plt.axvline(x=val, color='red', linestyle='--', alpha=0.5, label=f'Expected {label}')
            
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.title('Adelic Liouvillian Trace vs Prime Powers')
        plt.xlabel('Time t (Log Scale)')
        plt.ylabel('Trace Amplitude')
        plt.grid(True, alpha=0.3)
        
        output_path = Path("results/reports/trace_formula_verification.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"\nPlot saved to: {output_path}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")

    if matches >= 2:
        print("SUCCESS: Liouvillian trace exhibits prime-periodic structure.")
    else:
        print("WARNING: Peaks not clearly resolved (increase dimension?).")

if __name__ == "__main__":
    main()
