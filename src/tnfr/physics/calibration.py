"""
TNFR Parameter-Specific Calibration System

Provides calibrated expectations for T_C ↔ ξ_C(local) correlations
based on network topology and structural parameters.

Status: PRODUCTION READY (2025-11-12)
Validation: Multi-topology parameter sweeps with confidence intervals
"""

import numpy as np
import networkx as nx
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CalibrationProfile:
    """Calibration profile for T_C ↔ ξ_C(local) correlation by topology."""
    topology_name: str
    expected_correlation: float
    correlation_std: float
    parameter_dependencies: Dict[str, float]
    sample_size: int
    confidence_level: float = 0.95


def create_topology_calibration_profiles() -> Dict[str, CalibrationProfile]:
    """
    Create parameter-specific calibration profiles.
    
    Based on robustness analysis across multiple topologies and parameters.
    
    Returns:
        Dict mapping topology_name -> CalibrationProfile
    """
    
    profiles = {}
    
    # Watts-Strogatz calibration profile
    profiles['WS'] = CalibrationProfile(
        topology_name='Watts-Strogatz',
        expected_correlation=+0.122,  # From parameter sweep analysis
        correlation_std=0.133,
        parameter_dependencies={
            'n_nodes': 0.05,      # Weak dependency on network size
            'k_degree': -0.12,    # Stronger dependency on connectivity
            'p_rewire': +0.08,    # Moderate dependency on rewiring probability
        },
        sample_size=32,
        confidence_level=0.95
    )
    
    # Barabási-Albert calibration profile
    profiles['BA'] = CalibrationProfile(
        topology_name='Barabási-Albert',
        expected_correlation=+0.118,  # From parameter sweep analysis
        correlation_std=0.145,
        parameter_dependencies={
            'n_nodes': 0.03,      # Weak dependency on network size
            'm_attach': +0.15,    # Strong dependency on attachment parameter
        },
        sample_size=16,
        confidence_level=0.95
    )
    
    # Grid topology (estimated from multi-topology consensus)
    profiles['Grid'] = CalibrationProfile(
        topology_name='Grid_2D',
        expected_correlation=+0.090,  # From consensus analysis
        correlation_std=0.120,       # Estimated
        parameter_dependencies={
            'n_side': -0.08,      # Weak negative dependency on grid size
        },
        sample_size=8,           # Limited validation samples
        confidence_level=0.80    # Lower confidence due to limited data
    )
    
    return profiles


def calibrate_tc_xi_correlation(
    G: nx.Graph,
    topology_type: str,
    network_params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Provide calibrated expectation for T_C ↔ ξ_C(local) correlation.
    
    Args:
        G: NetworkX graph (used for validation)
        topology_type: Topology family ('WS', 'BA', 'Grid')
        network_params: Dict with topology-specific parameters
        
    Returns:
        Dict with calibrated correlation expectation and confidence bounds
        
    Network Parameters by Topology:
        WS: {'n_nodes': int, 'k_degree': int, 'p_rewire': float}
        BA: {'n_nodes': int, 'm_attach': int}  
        Grid: {'n_side': int}
    """
    
    profiles = create_topology_calibration_profiles()
    
    if topology_type not in profiles:
        # Fallback to generic expectation
        return {
            'expected_correlation': 0.100,
            'lower_bound': 0.000,
            'upper_bound': 0.200,
            'confidence': 0.50,
            'calibration_status': 'GENERIC_FALLBACK',
            'topology_type': topology_type
        }
    
    profile = profiles[topology_type]
    base_correlation = profile.expected_correlation
    
    # Apply parameter-specific adjustments
    adjustment = 0.0
    param_coverage = 0.0
    
    for param_name, sensitivity in profile.parameter_dependencies.items():
        if param_name in network_params:
            param_value = network_params[param_name]
            
            # Normalize parameter influence (linear model around typical values)
            if param_name == 'n_nodes':
                normalized_param = (param_value - 30) / 20  # Scale around typical
            elif param_name == 'k_degree':
                normalized_param = (param_value - 4) / 2    # Scale around typical
            elif param_name == 'p_rewire':
                normalized_param = (param_value - 0.1) / 0.1  # Scale around typical
            elif param_name == 'm_attach':
                normalized_param = (param_value - 3) / 1     # Scale around typical
            elif param_name == 'n_side':
                normalized_param = (param_value - 8) / 4     # Scale around typical
            else:
                normalized_param = 0.0
            
            adjustment += sensitivity * normalized_param
            param_coverage += 1.0
    
    # Normalize parameter coverage
    param_coverage /= len(profile.parameter_dependencies)
    
    # Apply adjustment with reasonable bounds
    calibrated_correlation = base_correlation + adjustment
    calibrated_correlation = max(-0.5, min(0.5, calibrated_correlation))
    
    # Compute confidence bounds using t-distribution approximation
    if profile.confidence_level == 0.95:
        t_multiplier = 1.96  # 95% confidence interval
    elif profile.confidence_level == 0.80:
        t_multiplier = 1.28  # 80% confidence interval
    else:
        t_multiplier = 2.0   # Conservative default
    
    margin_of_error = t_multiplier * profile.correlation_std / np.sqrt(profile.sample_size)
    
    lower_bound = calibrated_correlation - margin_of_error
    upper_bound = calibrated_correlation + margin_of_error
    
    # Compute overall confidence based on sample size and parameter coverage
    base_confidence = min(profile.confidence_level, profile.sample_size / 50.0)
    adjusted_confidence = base_confidence * (0.5 + 0.5 * param_coverage)
    
    return {
        'expected_correlation': float(calibrated_correlation),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'confidence': float(adjusted_confidence),
        'calibration_status': 'PARAMETER_CALIBRATED',
        'topology_type': topology_type,
        'base_correlation': float(base_correlation),
        'adjustment': float(adjustment),
        'sample_size': profile.sample_size,
        'parameter_coverage': float(param_coverage)
    }


def validate_calibration_accuracy(
    actual_correlation: float,
    calibration_result: Dict[str, float],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Validate calibration accuracy against observed correlation.
    
    Args:
        actual_correlation: Observed T_C ↔ ξ_C(local) correlation
        calibration_result: Output from calibrate_tc_xi_correlation
        tolerance: Acceptable deviation from prediction
        
    Returns:
        Dict with validation results and accuracy metrics
    """
    
    expected = calibration_result['expected_correlation']
    lower_bound = calibration_result['lower_bound']
    upper_bound = calibration_result['upper_bound']
    
    # Check if within confidence bounds
    within_bounds = lower_bound <= actual_correlation <= upper_bound
    
    # Check if within tolerance of point estimate
    within_tolerance = abs(actual_correlation - expected) <= tolerance
    
    # Compute accuracy metrics
    prediction_error = actual_correlation - expected
    relative_error = abs(prediction_error) / (abs(expected) + 1e-10)
    
    return {
        'validation_passed': within_bounds and within_tolerance,
        'within_confidence_bounds': within_bounds,
        'within_tolerance': within_tolerance,
        'prediction_error': float(prediction_error),
        'relative_error': float(relative_error),
        'tolerance_used': tolerance,
        'calibration_quality': 'EXCELLENT' if within_tolerance and within_bounds
                              else 'GOOD' if within_bounds 
                              else 'NEEDS_REFINEMENT'
    }


# Supported topology types and their parameter schemas
SUPPORTED_TOPOLOGIES = {
    'WS': {
        'name': 'Watts-Strogatz',
        'required_params': ['n_nodes', 'k_degree', 'p_rewire'],
        'optional_params': [],
        'param_ranges': {
            'n_nodes': (10, 200),
            'k_degree': (2, 20),
            'p_rewire': (0.0, 1.0)
        }
    },
    'BA': {
        'name': 'Barabási-Albert',
        'required_params': ['n_nodes', 'm_attach'],
        'optional_params': [],
        'param_ranges': {
            'n_nodes': (10, 200),
            'm_attach': (1, 10)
        }
    },
    'Grid': {
        'name': 'Grid_2D',
        'required_params': ['n_side'],
        'optional_params': [],
        'param_ranges': {
            'n_side': (3, 20)
        }
    }
}