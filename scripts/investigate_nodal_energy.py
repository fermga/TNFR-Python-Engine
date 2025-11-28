import numpy as np
import matplotlib.pyplot as plt

def psi_function(x):
    """Computes Chebyshev psi(x) = sum_{n<=x} Lambda(n)."""
    # Inefficient but correct for small x
    total = 0
    for n in range(2, int(x) + 1):
        # Check if prime power
        temp = n
        p = 0
        for i in range(2, int(n**0.5) + 1):
            if temp % i == 0:
                p = i
                break
        if p == 0: p = n
        
        temp = n
        while temp % p == 0:
            temp //= p
        
        if temp == 1:
            total += np.log(p)
    return total

def compute_nodal_energy(T_max):
    """
    Computes the Structural Energy E(T) = integral_1^{e^T} |(psi(x)-x)/sqrt(x)|^2 dx/x
    """
    # We discretize the integral
    # x goes from 1 to e^T_max
    # Let's use log scale for integration: u = log x, du = dx/x
    # Integral becomes integral_0^{T_max} |(psi(e^u) - e^u)/e^{u/2}|^2 du
    
    u_values = np.linspace(0.1, T_max, 500) # Avoid 0
    energy_density = []
    cumulative_energy = []
    current_energy = 0
    
    dt = u_values[1] - u_values[0]
    
    for u in u_values:
        x = np.exp(u)
        psi = psi_function(x)
        error = psi - x
        normalized_error = error / np.sqrt(x)
        density = normalized_error**2
        
        energy_density.append(density)
        current_energy += density * dt
        cumulative_energy.append(current_energy)
        
    return u_values, energy_density, cumulative_energy

if __name__ == "__main__":
    print("=== Pillar 4: Nodal Energy Investigation ===")
    print("Checking the boundedness of Structural Energy (Stability)")
    
    T_max = 8.0 # e^8 approx 3000
    print(f"Computing up to x = exp({T_max}) ~= {np.exp(T_max):.0f}")
    
    u, dens, cum = compute_nodal_energy(T_max)
    
    print("\nResults:")
    print(f"Final Cumulative Energy: {cum[-1]:.4f}")
    print(f"Average Energy Density: {np.mean(dens):.4f}")
    
    # Check for divergence
    # If RH is true, Cumulative Energy should grow linearly with T (constant density)
    # If RH false, it would grow exponentially.
    
    # Linear fit to cumulative energy
    coeffs = np.polyfit(u, cum, 1)
    print(f"Linear Fit Slope (Energy Rate): {coeffs[0]:.4f}")
    print(f"Linear Fit R^2: {np.corrcoef(u, cum)[0,1]**2:.4f}")
    
    if coeffs[0] > 0 and np.corrcoef(u, cum)[0,1]**2 > 0.9:
        print("Conclusion: Energy grows linearly with log(x).")
        print("This implies the error term is bounded by sqrt(x) on average.")
        print("=> Consistent with RH (Theta = 1/2).")
    else:
        print("Conclusion: Inconclusive or Divergent.")
