#!/usr/bin/env python3
"""
Execute Refined Zero Discriminant Analysis
==========================================

Runs comprehensive analysis to validate the refined approach that addresses
the mathematical critique of the original ΔNFR method.
"""

from refined_zero_discriminant import TNFRRefinedZeroDiscriminant
import json
import numpy as np

def main():
    """Execute comprehensive refined zero discriminant analysis."""
    print("REFINED ZERO DISCRIMINANT ANALYSIS")
    print("=" * 60)
    
    analyzer = TNFRRefinedZeroDiscriminant()
    result = analyzer.run_refined_analysis()
    
    # Print key metrics
    print(f"Zero candidates found: {len(result.zero_candidates)}")
    print(f"Critical line points scanned: {len(result.critical_line_scan)}")
    print(f"Off-line points scanned: {len(result.off_line_scan)}")
    print(f"Classical equivalence verified: {result.classical_equivalence_verified}")
    print(f"Mathematical rigor score: {result.mathematical_rigor_score}")
    
    # Create summary dict for JSON output
    summary = {
        "zero_candidates_count": len(result.zero_candidates),
        "zero_candidates": [str(z) for z in result.zero_candidates],
        "critical_line_points": len(result.critical_line_scan),
        "off_line_points": len(result.off_line_scan),
        "classical_equivalence_verified": result.classical_equivalence_verified,
        "mathematical_rigor_score": result.mathematical_rigor_score,
        "avg_critical_discriminant": float(np.mean([r.discriminant_value for r in result.critical_line_scan])),
        "avg_off_line_discriminant": float(np.mean([r.discriminant_value for r in result.off_line_scan]))
    }
    
    print("\nSummary Results:")
    print(json.dumps(summary, indent=2))
    
    # Save to file for analysis
    with open("refined_analysis_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to: refined_analysis_results.json")
    
    return result

if __name__ == "__main__":
    main()