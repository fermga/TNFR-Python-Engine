import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
from sympy import prime, isprime
import time

def get_primes_mod_4(limit):
    """Generate primes p <= limit such that p = 1 mod 4."""
    primes = []
    # Sieve or simple iteration. For limit=10^6, iteration is fine.
    # Using sympy.prime(i) is slow for range.
    # Let's use a simple sieve.
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    all_primes = np.nonzero(sieve)[0]
    return [p for p in all_primes if p % 4 == 1]

def compute_l_chi4(sigma, terms=100000):
    """Compute Dirichlet L-function L(s, chi_4) = 1 - 1/3^s + 1/5^s - ..."""
    # chi_4(n) is 1 if n=1 mod 4, -1 if n=3 mod 4, 0 if even.
    # Sum_{n odd} (-1)^((n-1)/2) n^-sigma
    s = 0.0
    for n in range(1, terms * 2, 2):
        term = n**(-sigma)
        if (n % 4) == 1:
            s += term
        else: # n % 4 == 3
            s -= term
    return s

def verify_lemma_3_1():
    print("--- Verifying Lemma 3.1 (Analytic Bridge) ---")
    
    LIMIT = 1000000
    print(f"Generating primes up to {LIMIT}...")
    t0 = time.time()
    primes_1 = get_primes_mod_4(LIMIT)
    print(f"Found {len(primes_1)} primes p = 1 mod 4. Time: {time.time()-t0:.2f}s")
    
    sigmas = [1.1, 1.2, 1.5, 2.0, 3.0, 4.0]
    
    print(f"\n{'sigma':<6} | {'Sum P_1':<12} | {'0.5*log(zeta)':<12} | {'Diff':<12} | {'0.5*log(L)':<12} | {'Rem':<12}")
    print("-" * 80)
    
    for sigma in sigmas:
        # 1. Compute Sum_{p in P1} p^-sigma
        # We use the primes we found. For high sigma, convergence is fast.
        # For low sigma (1.1), tail is significant.
        # We can approximate tail? No, let's just see the raw sum vs zeta.
        
        sum_p1 = sum([p**(-sigma) for p in primes_1])
        
        # 2. Compute 0.5 * log(zeta(sigma))
        # zeta(sigma) diverges at 1.
        z_val = zeta(sigma)
        target = 0.5 * np.log(z_val)
        
        diff = sum_p1 - target
        
        # 3. Compute 0.5 * log(L(sigma, chi4))
        # L(1, chi4) = pi/4.
        l_val = compute_l_chi4(sigma)
        correction_l = 0.5 * np.log(l_val)
        
        # 4. Higher order corrections
        # Remove p=2 term (which is in log(zeta) but not in Sum P_1)
        # In log(zeta) + log(L), the p=2 term is:
        # log(zeta): 2^-s + 1/2 2^-2s ...
        # log(L):    0    + 1/2 0 ... (chi(2)=0)
        # So sum has 2^-s. We need to subtract 1/2 * 2^-s.
        corr_2 = 0.5 * 2**(-sigma)
        
        # Remove squares (k=2)
        # In log(zeta) + log(L), the p^-2s term is:
        # log(zeta): 1/2 p^-2s
        # log(L):    1/2 chi(p^2) p^-2s = 1/2 p^-2s
        # Sum is p^-2s.
        # We want to isolate Sum p^-s. So we subtract 1/2 * Sum p^-2s.
        # Approx Sum p^-2s as log(zeta(2s)).
        z_2s = zeta(2*sigma)
        corr_sq = 0.5 * np.log(z_2s)
        
        predicted = target + correction_l - corr_2 - corr_sq
        
        residual = sum_p1 - predicted
        
        print(f"{sigma:<6.2f} | {sum_p1:<12.6f} | {predicted:<12.6f} | {residual:<12.6f} | {target:<12.6f} | {correction_l:<12.6f} | {-corr_2:<12.6f} | {-corr_sq:<12.6f}")

    print("\nInterpretation:")
    print("Sum P_1   = Sum_{p=1(4)} p^-s")
    print("Predicted = 0.5*log(zeta) + 0.5*log(L) - 0.5*2^-s - 0.5*log(zeta(2s))")
    print("Residual  = Sum P_1 - Predicted")
    print("Target    = 0.5*log(zeta)")
    print("Corr L    = 0.5*log(L)")
    print("Corr 2    = -0.5*2^-s")
    print("Corr Sq   = -0.5*log(zeta(2s))")

if __name__ == "__main__":
    verify_lemma_3_1()
