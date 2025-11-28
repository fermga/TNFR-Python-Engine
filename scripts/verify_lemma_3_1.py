import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta
import time

def get_primes(n_max):
    """Generate primes up to n_max using sieve."""
    sieve = np.ones(n_max + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n_max**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.nonzero(sieve)[0]

def lhs_transform(s_vals, primes):
    """
    Compute LHS = Sum_p ((sqrt(p)-1)/(2*sqrt(p)))^(2s-1)
    """
    sqrt_p = np.sqrt(primes)
    base_term = 0.5 * (1.0 - 1.0/sqrt_p)
    
    results = []
    for s in s_vals:
        exponent = 2*s - 1
        terms = np.power(base_term.astype(complex), exponent)
        results.append(np.sum(terms))
    return np.array(results)

def lhs_transform_regularized(s_vals, primes):
    """
    Compute Regularized LHS.
    Subtract the first 3 terms of the Taylor expansion which cause divergence.
    Expansion of ((sqrt(p)-1)/(2*sqrt(p)))^(2s-1) = (1/2)^(2s-1) * (1 - p^-0.5)^(2s-1)
    Terms:
    k=0: 1
    k=1: -(2s-1) p^-0.5
    k=2: (2s-1)(2s-2)/2 p^-1
    """
    sqrt_p = np.sqrt(primes)
    p_inv = 1.0/primes
    
    results = []
    for s in s_vals:
        alpha = 2*s - 1
        prefactor = np.power(0.5 + 0j, alpha)
        
        term_exact = np.power(0.5 * (1.0 - 1.0/sqrt_p).astype(complex), alpha)
        
        term_approx = prefactor * (1.0 - alpha/sqrt_p + (alpha*(alpha-1)/2.0) * p_inv)
        
        term_reg = term_exact - term_approx
        
        results.append(np.sum(term_reg))
    return np.array(results)

def rhs_formula(s_vals):
    """
    Compute RHS = log(zeta(s)) / s
    """
    z_vals = zeta(s_vals)
    return np.log(z_vals) / s_vals

def main():
    print("=== Lemma 3.1 Numerical Verification ===")
    
    # Parameters
    N_MAX = 10**6
    print(f"Generating primes up to {N_MAX}...")
    primes = get_primes(N_MAX)
    print(f"Found {len(primes)} primes.")
    
    # Test range: Re(s) > 1 to ensure convergence
    sigma = 2.5
    t_vals = np.linspace(0, 50, 100)
    s_vals = sigma + 1j * t_vals
    
    print(f"Computing LHS (Regularized) for sigma={sigma}...")
    start = time.time()
    lhs = lhs_transform_regularized(s_vals, primes)
    print(f"LHS computed in {time.time() - start:.2f}s")
    
    print(f"Computing RHS for sigma={sigma}...")
    rhs = rhs_formula(s_vals)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_vals, np.abs(lhs), label='LHS (Regularized)')
    plt.plot(t_vals, np.abs(rhs), label='RHS (log(zeta)/s)', linestyle='--')
    plt.title(f'Magnitude Comparison (sigma={sigma})')
    plt.ylabel('|Value|')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t_vals, np.angle(lhs), label='LHS (Regularized)')
    plt.plot(t_vals, np.angle(rhs), label='RHS (log(zeta)/s)', linestyle='--')
    plt.title(f'Phase Comparison (sigma={sigma})')
    plt.xlabel('t (Im(s))')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.grid(True)
    
    output_file = 'results/lemma_3_1_verification.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    diff = np.abs(lhs - rhs)
    mean_diff = np.mean(diff)
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    
    print("\nCRITICAL CHECK:")
    print("Regularization applied: Subtracted terms k=0,1,2.")
    print("If Mean Diff is small, Lemma 3.1 holds with regularization.")
    
if __name__ == "__main__":
    main()
