#!/usr/bin/env python3
"""
Canonical Promotion Analysis for ξ_C Research Results

Analyzes multi-topology critical exponent results to assess ξ_C for canonical field promotion.
Implements the 10-point scoring system for comprehensive evaluation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_results(results_file: str) -> Dict[str, Any]:
    """Load experimental results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_critical_exponents(results: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Analyze critical exponents across topologies.
    
    Returns:
        Dict with topology-specific analysis including:
        - ν (critical exponent)  
        - R² (fit quality)
        - universality_class (classification)
        - dynamic_range (intensity span with clear scaling)
    """
    topology_analysis = {}
    
    for topology_name, topology_data in results.items():
        if "critical_exponent_fit" in topology_data:
            fit_data = topology_data["critical_exponent_fit"]
            
            analysis = {
                "exponent": fit_data.get("exponent", 0.0),
                "r_squared": fit_data.get("r_squared", 0.0),
                "std_error": fit_data.get("std_error", float('inf')),
                "fit_quality": classify_fit_quality(fit_data.get("r_squared", 0.0)),
                "universality_class": classify_universality_class(fit_data.get("exponent", 0.0)),
                "dynamic_range": calculate_dynamic_range(topology_data),
                "data_quality": assess_data_quality(topology_data)
            }
            
            topology_analysis[topology_name] = analysis
    
    return topology_analysis

def classify_fit_quality(r_squared: float) -> str:
    """Classify R² fit quality for critical exponent."""
    if r_squared >= 0.90:
        return "excellent"
    elif r_squared >= 0.80:
        return "good" 
    elif r_squared >= 0.70:
        return "acceptable"
    elif r_squared >= 0.50:
        return "weak"
    else:
        return "poor"

def classify_universality_class(exponent: float) -> str:
    """
    Classify universality class based on critical exponent ν.
    
    Reference values:
    - Mean field: ν = 0.5
    - 2D Ising: ν = 1.0  
    - 3D Ising: ν ≈ 0.63
    - Percolation 2D: ν = 4/3 ≈ 1.33
    - Percolation 3D: ν ≈ 0.88
    """
    tolerances = [
        (0.5, 0.05, "mean-field"),
        (0.63, 0.08, "ising-3d"),
        (1.0, 0.10, "ising-2d"),  
        (1.33, 0.10, "percolation-2d"),
        (0.88, 0.08, "percolation-3d")
    ]
    
    for ref_val, tolerance, class_name in tolerances:
        if abs(exponent - ref_val) <= tolerance:
            return class_name
            
    return "unknown"

def calculate_dynamic_range(topology_data: Dict) -> float:
    """Calculate the intensity range showing clear critical scaling."""
    if "xi_c_data" not in topology_data:
        return 0.0
        
    xi_c_data = topology_data["xi_c_data"]
    intensity_range = []
    
    # Find range where ξ_C shows clear variation
    for point in xi_c_data:
        intensity = point["intensity"]
        mean_xi = point["mean"]
        std_xi = point["std"]
        
        # Only include points with reasonable signal-to-noise
        if std_xi < mean_xi:  # SNR > 1
            intensity_range.append(intensity)
    
    if len(intensity_range) < 2:
        return 0.0
        
    return max(intensity_range) - min(intensity_range)

def assess_data_quality(topology_data: Dict) -> str:
    """Assess overall data quality for the topology."""
    if "xi_c_data" not in topology_data:
        return "poor"
        
    xi_c_data = topology_data["xi_c_data"]
    
    # Count points with good signal-to-noise ratio
    good_points = 0
    total_points = len(xi_c_data)
    
    for point in xi_c_data:
        mean_xi = point["mean"]
        std_xi = point["std"]
        
        # Good SNR and reasonable magnitude
        if std_xi < mean_xi and mean_xi > 1.0:
            good_points += 1
    
    fraction_good = good_points / total_points if total_points > 0 else 0.0
    
    if fraction_good >= 0.8:
        return "excellent"
    elif fraction_good >= 0.6:
        return "good"
    elif fraction_good >= 0.4:
        return "acceptable"
    else:
        return "poor"

def analyze_canonical_correlations(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze correlations between ξ_C and canonical fields across all topologies.
    
    Returns:
        Correlations between ξ_C and each canonical field (Φ_s, |∇φ|, K_φ).
    """
    all_xi_c = []
    all_phi_s = []
    all_grad_phi = []
    all_k_phi = []
    
    # Collect all measurements across topologies
    for topology_name, topology_data in results.items():
        if "canonical_fields_data" not in topology_data:
            continue
            
        fields_data = topology_data["canonical_fields_data"]
        xi_c_data = topology_data.get("xi_c_data", [])
        
        # Match intensity points between ξ_C and canonical fields
        for i, xi_point in enumerate(xi_c_data):
            if i < len(fields_data):
                field_point = fields_data[i]
                
                all_xi_c.append(xi_point["mean"])
                all_phi_s.append(field_point["phi_s"]["mean"])
                all_grad_phi.append(field_point["grad_phi"]["mean"])
                all_k_phi.append(field_point["k_phi"]["mean"])
    
    correlations = {}
    
    if len(all_xi_c) > 1:
        # Calculate Pearson correlations
        correlations["xi_c_vs_phi_s"] = np.corrcoef(all_xi_c, all_phi_s)[0, 1]
        correlations["xi_c_vs_grad_phi"] = np.corrcoef(all_xi_c, all_grad_phi)[0, 1]
        correlations["xi_c_vs_k_phi"] = np.corrcoef(all_xi_c, all_k_phi)[0, 1]
        
        # Check for NaN values (perfect correlations or no variance)
        for key, value in correlations.items():
            if np.isnan(value):
                correlations[key] = 0.0
    else:
        correlations = {
            "xi_c_vs_phi_s": 0.0,
            "xi_c_vs_grad_phi": 0.0,
            "xi_c_vs_k_phi": 0.0
        }
    
    return correlations

def score_canonical_promotion(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive 10-point scoring for ξ_C canonical field promotion.
    
    Scoring criteria:
    1. Multi-topology validation (0-2 points)
    2. Critical exponent quality (0-2 points)  
    3. Universality classification (0-1 point)
    4. Dynamic range (0-1 point)
    5. Cross-field correlations (0-2 points)
    6. Theoretical consistency (0-2 points)
    
    Returns:
        Detailed scoring breakdown and total score.
    """
    
    # Analyze topologies and correlations
    topology_analysis = analyze_critical_exponents(results)
    correlations = analyze_canonical_correlations(results)
    
    scoring = {
        "criteria": {},
        "total_score": 0,
        "max_score": 10,
        "recommendation": ""
    }
    
    # 1. Multi-topology validation (0-2 points)
    valid_topologies = sum(1 for analysis in topology_analysis.values() 
                          if analysis["fit_quality"] in ["acceptable", "good", "excellent"])
    
    if valid_topologies >= 3:
        topo_score = 2.0
    elif valid_topologies >= 2:
        topo_score = 1.5
    elif valid_topologies >= 1:
        topo_score = 1.0
    else:
        topo_score = 0.0
        
    scoring["criteria"]["multi_topology"] = {
        "score": topo_score,
        "max": 2.0,
        "valid_topologies": valid_topologies,
        "details": topology_analysis
    }
    
    # 2. Critical exponent quality (0-2 points) 
    quality_scores = []
    for analysis in topology_analysis.values():
        r_squared = analysis["r_squared"]
        if r_squared >= 0.90:
            quality_scores.append(2.0)
        elif r_squared >= 0.80:
            quality_scores.append(1.5)
        elif r_squared >= 0.70:
            quality_scores.append(1.0)
        elif r_squared >= 0.50:
            quality_scores.append(0.5)
        else:
            quality_scores.append(0.0)
    
    exponent_score = np.mean(quality_scores) if quality_scores else 0.0
    
    scoring["criteria"]["exponent_quality"] = {
        "score": exponent_score,
        "max": 2.0,
        "individual_scores": quality_scores,
        "mean_r_squared": np.mean([a["r_squared"] for a in topology_analysis.values()])
    }
    
    # 3. Universality classification (0-1 point)
    known_classes = sum(1 for analysis in topology_analysis.values() 
                       if analysis["universality_class"] != "unknown")
    
    if known_classes >= 2:
        universality_score = 1.0
    elif known_classes >= 1:
        universality_score = 0.5
    else:
        universality_score = 0.0
        
    scoring["criteria"]["universality"] = {
        "score": universality_score,
        "max": 1.0,
        "known_classes": known_classes,
        "classifications": {name: a["universality_class"] for name, a in topology_analysis.items()}
    }
    
    # 4. Dynamic range (0-1 point)
    dynamic_ranges = [analysis["dynamic_range"] for analysis in topology_analysis.values()]
    mean_range = np.mean(dynamic_ranges) if dynamic_ranges else 0.0
    
    if mean_range >= 1.0:  # At least 1.0 intensity units
        range_score = 1.0
    elif mean_range >= 0.5:
        range_score = 0.5
    else:
        range_score = 0.0
        
    scoring["criteria"]["dynamic_range"] = {
        "score": range_score,
        "max": 1.0,
        "mean_range": mean_range,
        "individual_ranges": dynamic_ranges
    }
    
    # 5. Cross-field correlations (0-2 points)
    # Lower correlations are better (indicates ξ_C provides unique information)
    max_correlation = max(abs(correlations["xi_c_vs_phi_s"]),
                         abs(correlations["xi_c_vs_grad_phi"]),
                         abs(correlations["xi_c_vs_k_phi"]))
    
    if max_correlation < 0.3:  # Low correlation = unique information
        correlation_score = 2.0
    elif max_correlation < 0.5:
        correlation_score = 1.5
    elif max_correlation < 0.7:
        correlation_score = 1.0
    elif max_correlation < 0.85:
        correlation_score = 0.5
    else:
        correlation_score = 0.0
        
    scoring["criteria"]["cross_correlations"] = {
        "score": correlation_score,
        "max": 2.0,
        "max_correlation": max_correlation,
        "correlations": correlations
    }
    
    # 6. Theoretical consistency (0-2 points)
    # Check if exponents are physically reasonable and consistent
    exponents = [analysis["exponent"] for analysis in topology_analysis.values()]
    
    # Physical range check (0.4 to 2.0 is reasonable for most systems)
    physical_exponents = sum(1 for nu in exponents if 0.4 <= nu <= 2.0)
    
    if physical_exponents == len(exponents) and len(exponents) >= 2:
        theory_score = 2.0
    elif physical_exponents >= len(exponents) * 0.8:
        theory_score = 1.5
    elif physical_exponents >= len(exponents) * 0.5:
        theory_score = 1.0
    else:
        theory_score = 0.0
        
    scoring["criteria"]["theoretical"] = {
        "score": theory_score,
        "max": 2.0,
        "physical_exponents": physical_exponents,
        "total_exponents": len(exponents),
        "exponent_values": exponents
    }
    
    # Calculate total score
    total = sum(criterion["score"] for criterion in scoring["criteria"].values())
    scoring["total_score"] = total
    
    # Generate recommendation
    if total >= 8.5:
        recommendation = "STRONG RECOMMENDATION for canonical promotion"
    elif total >= 7.0:
        recommendation = "RECOMMEND canonical promotion with minor notes"
    elif total >= 5.0:
        recommendation = "CONDITIONAL recommendation pending improvements"
    elif total >= 3.0:
        recommendation = "NOT RECOMMENDED at this time - needs significant work"
    else:
        recommendation = "REJECT canonical promotion - fundamental issues"
        
    scoring["recommendation"] = recommendation
    
    return scoring

def generate_safety_criteria(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate safety criteria recommendations for ξ_C based on experimental results.
    
    Analyzes typical ranges and critical thresholds across topologies.
    """
    all_xi_c_values = []
    critical_region_values = []  # Near I_c = 2.015
    
    # Collect ξ_C values across all topologies
    for topology_name, topology_data in results.items():
        if "xi_c_data" not in topology_data:
            continue
            
        for point in topology_data["xi_c_data"]:
            intensity = point["intensity"]
            mean_xi = point["mean"]
            
            all_xi_c_values.append(mean_xi)
            
            # Values near critical point (±0.2 range)
            if abs(intensity - 2.015) <= 0.2:
                critical_region_values.append(mean_xi)
    
    if not all_xi_c_values:
        return {"error": "No ξ_C data found"}
        
    # Statistical analysis
    mean_xi_c = np.mean(all_xi_c_values)
    std_xi_c = np.std(all_xi_c_values)
    median_xi_c = np.median(all_xi_c_values)
    
    # Percentile thresholds
    p95 = np.percentile(all_xi_c_values, 95)
    p90 = np.percentile(all_xi_c_values, 90)
    p10 = np.percentile(all_xi_c_values, 10)
    
    # Critical region analysis
    if critical_region_values:
        critical_mean = np.mean(critical_region_values)
        critical_std = np.std(critical_region_values)
    else:
        critical_mean = mean_xi_c
        critical_std = std_xi_c
    
    # Generate safety recommendations
    safety_criteria = {
        "statistical_summary": {
            "total_measurements": len(all_xi_c_values),
            "mean": mean_xi_c,
            "std": std_xi_c,
            "median": median_xi_c,
            "range": [float(np.min(all_xi_c_values)), float(np.max(all_xi_c_values))]
        },
        "critical_region": {
            "measurements": len(critical_region_values),
            "mean": critical_mean,
            "std": critical_std
        },
        "recommended_thresholds": {
            "normal_operation": {
                "description": "Safe range for typical TNFR operations",
                "threshold": f"ξ_C < {p90:.1f}",
                "rationale": "90th percentile - captures most stable operation"
            },
            "warning_level": {
                "description": "Elevated coherence length - monitor closely", 
                "threshold": f"{p90:.1f} ≤ ξ_C < {p95:.1f}",
                "rationale": "Between 90th-95th percentile - may indicate criticality"
            },
            "critical_level": {
                "description": "High coherence length - potential system transition",
                "threshold": f"ξ_C ≥ {p95:.1f}",
                "rationale": "Above 95th percentile - system near phase transition"
            }
        },
        "integration_notes": [
            "ξ_C provides complementary information to existing canonical fields",
            f"Typical operating range: {p10:.1f} to {p90:.1f}",
            f"Critical region signature: {critical_mean:.1f} ± {critical_std:.1f}",
            "Higher values indicate stronger long-range correlations",
            "Use alongside Φ_s, |∇φ|, and K_φ for comprehensive structural health"
        ]
    }
    
    return safety_criteria

def main():
    """Main analysis function - will process results when available."""
    
    print("🔬 ξ_C Canonical Promotion Analysis Framework")
    print("=" * 50)
    
    # Check if results file exists
    results_file = Path(__file__).parent / "multi_topology_xi_c_results.json"
    
    if not results_file.exists():
        print("⏳ Results file not found - experiment still running")
        print(f"   Looking for: {results_file}")
        print("\n📋 Analysis Framework Ready:")
        print("   • Critical exponent quality assessment")
        print("   • Universality class classification")  
        print("   • Cross-field correlation analysis")
        print("   • 10-point canonical promotion scoring")
        print("   • Safety criteria generation")
        print("\n⚠️  Run this script again after experiments complete")
        return
    
    # Load and analyze results
    print(f"📂 Loading results from: {results_file}")
    results = load_results(results_file)
    
    print("\n🔍 Analyzing Critical Exponents...")
    topology_analysis = analyze_critical_exponents(results)
    
    print("\n📊 Cross-field Correlation Analysis...")
    correlations = analyze_canonical_correlations(results)
    
    print("\n🎯 Canonical Promotion Scoring...")
    promotion_score = score_canonical_promotion(results)
    
    print("\n🛡️ Safety Criteria Generation...")
    safety_criteria = generate_safety_criteria(results)
    
    # Display results
    print("\n" + "=" * 60)
    print("📋 ξ_C CANONICAL PROMOTION ASSESSMENT")
    print("=" * 60)
    
    # Topology results
    print(f"\n🌐 Multi-Topology Results:")
    for topo_name, analysis in topology_analysis.items():
        print(f"   {topo_name}:")
        print(f"      ν = {analysis['exponent']:.3f} ± {analysis['std_error']:.3f}")
        print(f"      R² = {analysis['r_squared']:.3f} ({analysis['fit_quality']})")
        print(f"      Class: {analysis['universality_class']}")
        print(f"      Range: {analysis['dynamic_range']:.2f}")
    
    # Cross-correlations  
    print(f"\n🔗 Cross-Field Correlations:")
    print(f"   ξ_C vs Φ_s:  {correlations['xi_c_vs_phi_s']:.3f}")
    print(f"   ξ_C vs |∇φ|: {correlations['xi_c_vs_grad_phi']:.3f}")
    print(f"   ξ_C vs K_φ:  {correlations['xi_c_vs_k_phi']:.3f}")
    
    # Promotion scoring
    print(f"\n🎖️ Canonical Promotion Score: {promotion_score['total_score']:.1f}/10")
    print(f"   {promotion_score['recommendation']}")
    
    print(f"\n   Detailed Scoring:")
    for criterion, data in promotion_score["criteria"].items():
        print(f"      {criterion}: {data['score']:.1f}/{data['max']}")
    
    # Safety criteria
    if "error" not in safety_criteria:
        thresholds = safety_criteria["recommended_thresholds"]
        print(f"\n🛡️ Recommended Safety Criteria:")
        print(f"   Normal: {thresholds['normal_operation']['threshold']}")
        print(f"   Warning: {thresholds['warning_level']['threshold']}")  
        print(f"   Critical: {thresholds['critical_level']['threshold']}")
    
    # Save analysis results
    analysis_file = Path(__file__).parent / "xi_c_canonical_analysis.json"
    analysis_output = {
        "topology_analysis": topology_analysis,
        "correlations": correlations,
        "promotion_scoring": promotion_score,
        "safety_criteria": safety_criteria,
        "timestamp": str(np.datetime64('now'))
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()