import numpy as np
import matplotlib.pyplot as plt

def von_mangoldt(n):
    """Computes Von Mangoldt function Lambda(n)."""
    # Simple implementation
    if n < 2: return 0
    
    # Find smallest prime factor
    p = 0
    temp = n
    for i in range(2, int(n**0.5) + 1):
        if temp % i == 0:
            p = i
            break
    
    if p == 0: p = n # n is prime
    
    # Check if n is power of p
    temp = n
    while temp % p == 0:
        temp //= p
    
    if temp == 1:
        return np.log(p)
    else:
        return 0

def spectral_density(t, X):
    """
    Computes the approximate spectral density S(t) using primes up to X.
    S(t) ~ - sum_{n <= X} Lambda(n)/sqrt(n) * 2 * cos(t * log n)
    
    This is the 'oscillatory part' of the Explicit Formula.
    Peaks should correspond to Riemann zeros gamma.
    """
    sum_val = 0
    # We sum over prime powers n, not just primes, for better accuracy
    # The formula usually involves Lambda(n)
    
    # Optimization: Precompute primes/Lambda
    # For this script, we'll just loop.
    
    for n in range(2, X + 1):
        lam = von_mangoldt(n)
        if lam > 0:
            term = (lam / np.sqrt(n)) * 2 * np.cos(t * np.log(n))
            sum_val -= term # The minus sign is crucial!
            
    return sum_val

if __name__ == "__main__":
    print("=== Pillar 3: Adelic Trace Investigation ===")
    print("Simulating the emergence of Riemann Zeros from Prime Sums")
    
    X = 1000 # Number of integers to sum over
    t_values = np.linspace(0, 50, 1000)
    s_values = []
    
    print(f"Computing spectral density for X={X}...")
    
    for t in t_values:
        s = spectral_density(t, X)
        s_values.append(s)
        
    # Find peaks
    s_values = np.array(s_values)
    peaks = []
    for i in range(1, len(s_values)-1):
        if s_values[i] > s_values[i-1] and s_values[i] > s_values[i+1]:
            # Simple threshold
            if s_values[i] > 5.0: # Arbitrary threshold to filter noise
                peaks.append((t_values[i], s_values[i]))
    
    print("\nDetected Peaks (Candidate Zeros):")
    print("Expected: 14.13, 21.02, 25.01, 30.42, 32.93, 37.58, 40.91, 43.32, 48.00")
    print("-" * 40)
    for t, amp in peaks:
        print(f"t = {t:.2f}  (Amp: {amp:.2f})")
        
    # Note: With X=1000, we expect rough peaks. 
    # Convergence is slow (conditional).
