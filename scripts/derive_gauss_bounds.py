import numpy as np
from collections import defaultdict
import cmath

def get_quadratic_residues(n):
    """
    Returns the set of quadratic residues in Z_n^*.
    S = {x^2 mod n | x in Z_n, gcd(x, n) = 1}
    """
    residues = set()
    for x in range(1, n):
        if np.gcd(x, n) == 1:
            residues.add((x * x) % n)
    return sorted(list(residues))

def compute_gauss_sum_eigenvalues(n):
    """
    Computes the eigenvalues mu_k = sum_{s in S} e^(2*pi*i*k*s/n)
    Groups them by d = gcd(k, n).
    """
    S = get_quadratic_residues(n)
    eigenvalues_by_gcd = defaultdict(list)
    
    print(f"\nAnalysis for n = {n}:")
    print(f"|S| = {len(S)}")
    
    # Calculate Euler totient phi(n)
    # For n=p^k, phi(n) = p^k - p^(k-1)
    # |S| should be phi(n)/2 for odd n?
    
    for k in range(n):
        # Compute sum_{s in S} e^(2*pi*i*k*s/n)
        val = sum(cmath.exp(2j * cmath.pi * k * s / n) for s in S)
        
        # Round to avoid floating point noise
        val_rounded = complex(round(val.real, 5), round(val.imag, 5))
        
        d = np.gcd(k, n)
        eigenvalues_by_gcd[d].append(val_rounded)

    # Analyze results
    for d in sorted(eigenvalues_by_gcd.keys()):
        vals = eigenvalues_by_gcd[d]
        unique_vals = sorted(list(set(vals)), key=lambda x: (x.real, x.imag))
        
        # Calculate magnitude of the first unique value (assuming they share magnitude in the group)
        mag = abs(unique_vals[0])
        
        print(f"  gcd(k, n) = {d}:")
        print(f"    Count: {len(vals)}")
        print(f"    Unique values: {unique_vals}")
        print(f"    Magnitude: {mag:.5f}")
        
        # Check against conjectures
        # For n=p^2:
        # If d=1 (coprime), we expect magnitude sqrt(n) = p? Or something else?
        # If d=p (divisible by p), we expect behavior related to Z_p.
        
    return eigenvalues_by_gcd

def analyze_prime_powers():
    primes = [3, 5, 7]
    for p in primes:
        # Analyze p (Prime)
        compute_gauss_sum_eigenvalues(p)
        
        # Analyze p^2 (Prime Squared)
        compute_gauss_sum_eigenvalues(p*p)
        
        # Analyze p^3 (Prime Cubed) - if small enough
        if p*p*p < 200:
            compute_gauss_sum_eigenvalues(p*p*p)

if __name__ == "__main__":
    analyze_prime_powers()
