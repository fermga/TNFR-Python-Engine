"""
Computational verification of Hypothesis R (Spectral Rigidity).

Tests that composite moduli have non-uniform Fourier spectra,
violating the Conference condition.

Run: python scripts/verify_hypothesis_r.py
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def compute_qr_spectrum(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of Paley graph on Z_n.
    
    Paley graph: vertices = Z_n, edge i~j iff (i-j) is a nonzero QR.
    
    Returns:
        Tuple of (adjacency_spectrum, laplacian_spectrum)
    """
    if n % 4 != 1:
        raise ValueError(f"n={n} must be 1 mod 4")
    
    # Compute quadratic residues (nonzero)
    QR = {(x * x) % n for x in range(1, n)}
    k = len(QR)  # Number of QRs (should be (n-1)/2 for prime)
    
    # Build adjacency matrix explicitly
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and ((i - j) % n) in QR:
                A[i, j] = 1
    
    # Compute eigenvalues
    adj_spectrum = np.linalg.eigvalsh(A)
    
    # Laplacian = k*I - A
    L = k * np.eye(n) - A
    lap_spectrum = np.linalg.eigvalsh(L)
    
    return adj_spectrum, lap_spectrum


def analyze_spectrum(n: int, adj_spectrum: np.ndarray, lap_spectrum: np.ndarray) -> Dict[str, float]:
    """
    Analyze Laplacian spectrum for Conference pattern.
    
    KEY OBSERVATION: For PRIME p, the Laplacian has exactly 3 DISTINCT eigenvalues.
    For COMPOSITE n, it has MANY distinct eigenvalues.
    
    This is the essence of Hypothesis R: "Conference" = 3-valued spectrum characterizes primes.
    
    Returns:
        Dictionary with analysis metrics
    """
    # Get unique Laplacian eigenvalues (up to numerical tolerance)
    lap_real = np.real(lap_spectrum)
    lap_unique = np.unique(np.round(lap_real, 4))  # Round to 4 decimals for uniqueness
    
    n_distinct = len(lap_unique)
    
    # Conference criterion: exactly 3 distinct values
    is_conference = (n_distinct == 3)
    
    # For display: separate zero and nonzero
    lap_nonzero = lap_unique[lap_unique > 1e-6]
    
    return {
        'n_distinct': n_distinct,
        'n_nonzero': len(lap_nonzero),
        'lap_eigenvalues': lap_unique.tolist(),
        'is_conference': is_conference,
        'sqrt_n': np.sqrt(n)
    }


def test_hypothesis_r(max_n: int = 101) -> List[Tuple[int, bool, Dict]]:
    """
    Test Hypothesis R: Conference spectrum ⟺ prime.
    
    Args:
        max_n: Test all n ≡ 1 (mod 4) up to this value
        
    Returns:
        List of (n, is_prime, spectrum_analysis) tuples
    """
    results = []
    
    test_values = [n for n in range(5, max_n) if n % 4 == 1]
    
    print("Testing Hypothesis R: Conference Spectrum <=> Prime")
    print("=" * 80)
    print(f"{'n':>4} | {'Type':>10} | {'Distinct L':>11} | {'Conference?':>12}")
    print("-" * 65)
    
    for n in test_values:
        prime = is_prime(n)
        adj_spec, lap_spec = compute_qr_spectrum(n)
        analysis = analyze_spectrum(n, adj_spec, lap_spec)
        
        results.append((n, prime, analysis))
        
        type_str = "PRIME" if prime else "composite"
        conf_str = "✅ YES" if analysis['is_conference'] else "❌ NO"
        
        print(f"{n:>4} | {type_str:>10} | {analysis['n_distinct']:>11} | "
              f"{conf_str:>12}")
    
    return results


def verify_hypothesis(results: List[Tuple[int, bool, Dict]]) -> bool:
    """
    Verify that Hypothesis R holds: Conference ⟺ Prime.
    
    Returns:
        True if all tests pass, False if any counterexample found
    """
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_pass = True
    
    for n, is_prime_n, analysis in results:
        expected_conference = is_prime_n
        actual_conference = analysis['is_conference']
        
        if expected_conference != actual_conference:
            all_pass = False
            print(f"❌ COUNTEREXAMPLE: n={n}, prime={is_prime_n}, conference={actual_conference}")
    
    if all_pass:
        print("✅ All tests passed!")
        print(f"   - All {sum(1 for _, p, _ in results if p)} primes have conference spectrum")
        print(f"   - All {sum(1 for _, p, _ in results if not p)} composites lack conference spectrum")
    
    return all_pass


def main():
    """Run full verification."""
    print("Hypothesis R: Spectral Rigidity - Computational Verification")
    print("=" * 70)
    print()
    
    results = test_hypothesis_r(max_n=101)
    print()
    success = verify_hypothesis(results)
    
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if success:
        print("✅ Hypothesis R is computationally verified for n ≤ 100.")
        print("   Next step: Prove rigorously using Davenport-Hasse bounds.")
    else:
        print("⚠️  Counterexample found! Review Hypothesis R statement.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
