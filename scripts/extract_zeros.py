"""
Extraction of Riemann Zeros from the Adelic Liouvillian.

This script attempts to extract the first few Riemann Zeros (gamma_n)
by diagonalizing the Liouvillian of the Adelic Scaling Flow.

Methodology:
1. Construct the Liouvillian L for a system with Prime Dissipators.
2. Compute the spectrum {lambda_n}.
3. Identify "Resonances" as eigenvalues with small real part (slow relaxation).
4. The imaginary parts Im(lambda_n) correspond to the Riemann Zeros.

Note:
This is a finite-dimensional approximation. The precision of the zeros
depends on the dimension 'dim' and the fidelity of the discretization.
We expect to see the *average density* of zeros follow the Riemann-von Mangoldt formula,
even if individual zeros are shifted by finite-size effects.
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
    print("TNFR Riemann Zero Extraction")
    print("============================")
    
    # Higher dimension for better spectral resolution
    dim = 60 
    print(f"System Dimension: {dim}")
    
    # 1. Build System
    print("Building Liouvillian with Primes {2, 3, 5, 7}...")
    H = build_scaling_hamiltonian(dim)
    primes = [2, 3, 5, 7]
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
    
    # 3. Filter Resonances
    # We look for eigenvalues lambda = -Gamma + i*gamma
    # where Gamma is small (long-lived resonance) and gamma > 0
    
    # Filter criteria:
    # 1. Positive imaginary part (gamma > 0)
    # 2. Small decay rate (Gamma < threshold)
    # 3. Not too close to zero (avoid trivial mode)
    
    gamma_threshold = 0.1
    decay_threshold = 2.0 
    
    resonances = [
        e for e in evals 
        if e.imag > gamma_threshold 
        and abs(e.real) < decay_threshold
    ]
    
    # Sort by imaginary part (height up the critical line)
    resonances.sort(key=lambda x: x.imag)
    
    print(f"\nFound {len(resonances)} candidate resonances.")
    
    print("\nTop 10 Extracted Zeros (gamma):")
    print(f"{'Index':<5} | {'Gamma (Decay)':<15} | {'gamma (Zero)':<15}")
    print("-" * 40)
    
    extracted_gammas = []
    for i, r in enumerate(resonances[:10]):
        print(f"{i+1:<5} | {abs(r.real):<15.4f} | {r.imag:<15.4f}")
        extracted_gammas.append(r.imag)
        
    # 4. Comparison with True Zeros
    true_zeros = [14.135, 21.022, 25.011, 30.425, 32.935]
    print("\nFirst 5 True Riemann Zeros:")
    print(true_zeros)
    
    print("\nAnalysis:")
    print("Note: In this finite-dimensional model, the 'energy scale' is arbitrary.")
    print("We need to check if the *ratios* or *spacing* resemble the Riemann zeros.")
    
    if len(extracted_gammas) >= 2:
        ratio_extracted = extracted_gammas[1] / extracted_gammas[0]
        ratio_true = true_zeros[1] / true_zeros[0]
        print(f"Ratio (2nd/1st): Extracted={ratio_extracted:.3f}, True={ratio_true:.3f}")
        
    # 5. Visualization
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot full spectrum
        real_parts = [e.real for e in evals]
        imag_parts = [e.imag for e in evals]
        plt.scatter(real_parts, imag_parts, alpha=0.3, s=10, label='Full Spectrum', color='gray')
        
        # Highlight resonances
        res_real = [r.real for r in resonances]
        res_imag = [r.imag for r in resonances]
        plt.scatter(res_real, res_imag, color='red', s=30, label='Resonances')
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.title('Liouvillian Spectrum & Riemann Resonances')
        plt.xlabel('Real Part (Decay Rate)')
        plt.ylabel('Imaginary Part (Frequency)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = Path("results/reports/riemann_zeros_spectrum.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"\nSpectrum plot saved to: {output_path}")
        
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()
