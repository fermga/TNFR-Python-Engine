#!/usr/bin/env python3
"""Quick analysis of multi-topology experiment results."""

import json
import numpy as np
from pathlib import Path

def analyze_latest_results():
    """Analyze the most recent multi-topology experimental results."""
    results_file = "benchmarks/results/multi_topology_critical_exponent_20251112_001348.jsonl"
    
    print("=== Multi-Topology ξ_C Experiment Analysis ===")
    print()
    
    topologies_data = {}
    
    try:
        with open(results_file, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    data = json.loads(line)
                    topology = data.get('topology', f'unknown_{i}')
                    duration = data.get('duration_seconds', 0)
                    
                    print(f"Topology: {topology}")
                    print(f"  Duration: {duration:.1f} seconds")
                    
                    # Check if we have xi_c data
                    if 'raw_data' in data and 'xi_c_data' in data['raw_data']:
                        raw_data = data['raw_data']
                        intensities = raw_data.get('intensities', [])
                        xi_values = raw_data.get('xi_c_data', [])
                        
                        print(f"  Intensities tested: {len(intensities)}")
                        print(f"  ξ_C measurements: {len(xi_values)}")
                        
                        # Analyze ξ_C values for each intensity
                        valid_intensities = []
                        all_means = []
                        
                        for xi_entry in xi_values:
                            intensity = xi_entry['intensity']
                            values = xi_entry['values']
                            mean_val = xi_entry['mean']
                            
                            # Count non-zero values
                            valid_runs = [v for v in values if v > 0]
                            valid_frac = len(valid_runs) / len(values)
                            
                            if len(valid_runs) >= 5:  # At least 5 valid runs
                                valid_intensities.append(intensity)
                                all_means.append(mean_val)
                        
                        print(f"  Valid intensities: {len(valid_intensities)}/13")
                        
                        if valid_intensities:
                            print(f"  ξ_C range: {min(all_means):.1f} - {max(all_means):.1f}")
                            print(f"  Critical region (2.0-2.03): ", end="")
                            critical_means = [m for i, m in zip(valid_intensities, all_means) if 2.0 <= i <= 2.03]
                            if critical_means:
                                print(f"{len(critical_means)} points, max ξ_C = {max(critical_means):.1f}")
                            else:
                                print("No data")
                        
                        topologies_data[topology] = {
                            'intensities': intensities,
                            'xi_c_values': xi_values,
                            'valid_count': len(valid_intensities),
                            'duration': duration
                        }
                    
                    print()
    
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return
    
    # Summary
    print("=== EXPERIMENT SUMMARY ===")
    print(f"Total topologies completed: {len(topologies_data)}")
    
    total_valid = sum(data['valid_count'] for data in topologies_data.values())
    total_measurements = sum(len(data['xi_c_values']) for data in topologies_data.values())
    
    print(f"Total ξ_C measurements: {total_measurements}")
    print(f"Valid ξ_C measurements: {total_valid} ({100*total_valid/total_measurements:.1f}%)")
    
    if total_valid > 0:
        print("\n🎉 SUCCESS: We have valid ξ_C data for critical exponent analysis!")
        
        # Check for critical point behavior
        for topology, data in topologies_data.items():
            if data['valid_count'] > 5:  # Need multiple points for analysis
                intensities = data['intensities']
                xi_values = data['xi_c_values']
                
                # Look for critical point signature (peak in ξ_C around I_c = 2.015)
                critical_region = [(i, xi) for i, xi in zip(intensities, xi_values) 
                                 if 2.0 <= i <= 2.03 and xi > 0]
                
                if critical_region:
                    max_xi_point = max(critical_region, key=lambda x: x[1])
                    print(f"  {topology}: Peak ξ_C = {max_xi_point[1]:.1f} at I = {max_xi_point[0]:.3f}")
    else:
        print("\n⚠️  No valid ξ_C measurements found")

if __name__ == "__main__":
    analyze_latest_results()