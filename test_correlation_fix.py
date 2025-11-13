#!/usr/bin/env python3
"""
Test the canonical correlation fix with a small experiment.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

# Import the fixed analysis function
from multi_topology_critical_exponent import analyze_canonical_correlations

def test_correlation_fix():
    """Test if the correlation analysis fix works."""
    print("🧪 TESTING CORRELATION ANALYSIS FIX")
    print("=" * 50)
    
    # Load actual data from the experiment
    results_file = "benchmarks/results/multi_topology_critical_exponent_20251111_233723.jsonl"
    
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'topology' in data:
                    results[data['topology']] = data
    
    print(f"Loaded {len(results)} topology results")
    
    # Test the analysis function
    try:
        correlations = analyze_canonical_correlations(results)
        print("✅ Correlation analysis succeeded!")
        print(f"Correlations: {correlations}")
        
        # Check if we got valid correlation values
        for field, corr_data in correlations.items():
            correlation = corr_data.get('correlation', 'N/A')
            print(f"  ξ_C ↔ {field}: r = {correlation}")
            
            if isinstance(correlation, (int, float)) and not str(correlation) == 'nan':
                print(f"    ✅ Valid correlation coefficient")
            else:
                print(f"    ⚠️  Invalid/NaN correlation: {correlation}")
        
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correlation_fix()