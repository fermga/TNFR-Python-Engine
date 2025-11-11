#!/usr/bin/env python3
"""
Analyze Dynamic Critical Exponent z for TNFR Phase Transition

Physical basis:
  - Near critical point, relaxation time diverges: τ_relax ~ ξ^z ~ (I - I_c)^(-νz)
  - For mean-field: ν = 0.5, typical z = 2 → τ ~ (I - I_c)^(-1)
  - We measure τ_relax from U6 simulator and fit power-law exponent

Usage:
  python tools/analyze_dynamic_exponent.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple


def load_u6_results(jsonl_path: Path) -> List[Dict]:
    """Load JSONL results from U6 simulator."""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_mean_tau_by_sequence_type(results: List[Dict]) -> Tuple[float, float]:
    """
    Compute mean τ_relax for valid vs violation sequences.
    
    Returns:
        (mean_tau_valid, mean_tau_violate)
    """
    tau_valid = []
    tau_violate = []
    
    for r in results:
        tau = r.get('tau_relax', np.nan)
        if not np.isfinite(tau):
            continue
        
        if r['sequence_type'] == 'valid':
            tau_valid.append(tau)
        else:
            tau_violate.append(tau)
    
    return (
        float(np.mean(tau_valid)) if tau_valid else float('nan'),
        float(np.mean(tau_violate)) if tau_violate else float('nan')
    )


def power_law(x, A, z):
    """Power law: y = A * x^(-z)"""
    return A * x**(-z)


def fit_dynamic_exponent(intensities: np.ndarray, tau_values: np.ndarray, I_c: float) -> Dict:
    """
    Fit τ_relax ~ (I - I_c)^(-z) to extract dynamic critical exponent z.
    
    Args:
        intensities: Array of intensity values
        tau_values: Array of mean τ_relax values
        I_c: Critical intensity
    
    Returns:
        Dictionary with fitted parameters and goodness-of-fit
    """
    # Filter valid data (I > I_c, finite tau)
    valid_mask = (intensities > I_c) & np.isfinite(tau_values) & (tau_values > 0)
    I_valid = intensities[valid_mask]
    tau_valid = tau_values[valid_mask]
    
    if len(I_valid) < 3:
        return {
            'z': np.nan,
            'A': np.nan,
            'r_squared': np.nan,
            'n_points': len(I_valid),
            'status': 'Insufficient data for fitting'
        }
    
    # Compute reduced variable
    epsilon = I_valid - I_c
    
    # Fit power law via nonlinear least squares
    try:
        popt, pcov = curve_fit(
            power_law, 
            epsilon, 
            tau_valid,
            p0=[1.0, 1.0],  # Initial guess: A=1, z=1
            maxfev=5000
        )
        z_fit, A_fit = popt[1], popt[0]
        
        # Compute R²
        tau_pred = power_law(epsilon, A_fit, z_fit)
        ss_res = np.sum((tau_valid - tau_pred)**2)
        ss_tot = np.sum((tau_valid - np.mean(tau_valid))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return {
            'z': z_fit,
            'A': A_fit,
            'r_squared': r_squared,
            'n_points': len(I_valid),
            'status': 'Fitted successfully'
        }
    
    except Exception as e:
        return {
            'z': np.nan,
            'A': np.nan,
            'r_squared': np.nan,
            'n_points': len(I_valid),
            'status': f'Fitting failed: {str(e)}'
        }


def analyze_dynamic_exponent():
    """Main analysis function for dynamic critical exponent z."""
    
    # Data files with intensity labels
    intensity_files = [
        (1.50, 'u6_i150.jsonl'),
        (2.00, 'u6_i200.jsonl'),
        (2.03, 'u6_fine_i203.jsonl'),
        (2.05, 'u6_i205.jsonl'),
        (2.07, 'u6_fine_i207.jsonl'),
        (2.08, 'u6_fine_i208.jsonl'),
        (2.09, 'u6_fine_i209.jsonl'),
        (2.10, 'u6_i210.jsonl'),
        (2.20, 'u6_i220.jsonl'),
        (2.50, 'u6_i250.jsonl'),
    ]
    
    I_c = 2.015  # From universality analysis
    
    print("\n=== Dynamic Critical Exponent z Analysis ===\n")
    print("Theory: τ_relax ~ (I - I_c)^(-z)")
    print(f"Using I_c = {I_c:.3f}\n")
    
    # Collect τ_relax data
    intensities_valid = []
    tau_valid = []
    intensities_violate = []
    tau_violate = []
    
    print("Step 1: Extract τ_relax by intensity\n")
    print(f"{'Intensity':>10} | {'τ_valid':>10} | {'τ_violate':>10} | {'Status':>15}")
    print("-" * 60)
    
    for intensity, filename in intensity_files:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"{intensity:>10.2f} | {'N/A':>10} | {'N/A':>10} | File not found")
            continue
        
        results = load_u6_results(filepath)
        tau_v, tau_x = compute_mean_tau_by_sequence_type(results)
        
        intensities_valid.append(intensity)
        tau_valid.append(tau_v)
        intensities_violate.append(intensity)
        tau_violate.append(tau_x)
        
        status = "✓" if np.isfinite(tau_v) and np.isfinite(tau_x) else "Missing data"
        print(f"{intensity:>10.2f} | {tau_v:>10.3f} | {tau_x:>10.3f} | {status:>15}")
    
    # Convert to arrays
    I_valid = np.array(intensities_valid)
    tau_valid_arr = np.array(tau_valid)
    I_violate = np.array(intensities_violate)
    tau_violate_arr = np.array(tau_violate)
    
    # Fit dynamic exponent for violations (critical behavior expected)
    print("\n\nStep 2: Fit τ_relax ~ (I - I_c)^(-z) for violations\n")
    
    fit_result = fit_dynamic_exponent(I_violate, tau_violate_arr, I_c)
    
    print(f"{'Parameter':>15} | {'Value':>15}")
    print("-" * 35)
    print(f"{'z (fitted)':>15} | {fit_result['z']:>15.3f}")
    print(f"{'A (amplitude)':>15} | {fit_result['A']:>15.3f}")
    print(f"{'R² (fit)':>15} | {fit_result['r_squared']:>15.3f}")
    print(f"{'N points':>15} | {fit_result['n_points']:>15}")
    print(f"{'Status':>15} | {fit_result['status']:>15}")
    
    # Theoretical comparison
    print("\n\nStep 3: Theoretical comparison\n")
    print(f"{'Theory':>20} | {'Predicted z':>12}")
    print("-" * 40)
    print(f"{'Mean-field':>20} | {2.0:>12.1f}")
    print(f"{'3D Ising':>20} | {2.0:>12.1f}")
    print(f"{'2D Ising':>20} | {2.17:>12.2f}")
    print(f"{'TNFR (fitted)':>20} | {fit_result['z']:>12.3f}")
    
    # Interpretation
    print("\n\nInterpretation:")
    if not np.isfinite(fit_result['z']):
        print("  - Insufficient data for z extraction")
        print("  - Need more intensity points near I_c (e.g., 2.02, 2.03, 2.04)")
    elif abs(fit_result['z'] - 2.0) < 0.3:
        print("  - z ≈ 2.0 → Mean-field universality class (confirmed)")
        print("  - Consistent with β = 0.556 (mean-field)")
        print("  - Relaxation time: τ ~ (I - I_c)^(-2)")
    elif fit_result['z'] > 2.3:
        print(f"  - z = {fit_result['z']:.2f} > 2.3 → Non-mean-field correction")
        print("  - Possible strong fluctuation effects near criticality")
    else:
        print(f"  - z = {fit_result['z']:.2f} → Intermediate regime")
        print("  - May indicate crossover between universality classes")
    
    # Relaxation time table
    print("\n\nRelaxation Time vs Reduced Variable ε = I - I_c:")
    print(f"{'I':>8} | {'ε':>8} | {'τ_obs':>10} | {'τ_fit':>10} | {'Residual':>10}")
    print("-" * 60)
    
    epsilon_arr = I_violate - I_c
    valid_mask = (epsilon_arr > 0) & np.isfinite(tau_violate_arr) & (tau_violate_arr > 0)
    
    for i in np.where(valid_mask)[0]:
        I_val = I_violate[i]
        eps = epsilon_arr[i]
        tau_obs = tau_violate_arr[i]
        tau_fit_val = power_law(eps, fit_result['A'], fit_result['z'])
        residual = tau_obs - tau_fit_val
        print(f"{I_val:>8.2f} | {eps:>8.3f} | {tau_obs:>10.3f} | {tau_fit_val:>10.3f} | {residual:>10.3f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    analyze_dynamic_exponent()
