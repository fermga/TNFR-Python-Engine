"""
TNFR Feedback Integration Adapter

Adapter that integrates the feedback learning system with the 
existing SpectralPaleyFactorizer to enable closed-loop optimization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .feedback_integration import (
    OptimizationFeedbackLearner, 
    VerificationFeedback,
    OptimizationStrategy
)


class FeedbackIntegratedFactorizer:
    """Wrapper that adds feedback learning to SpectralPaleyFactorizer."""
    
    def __init__(self, base_factorizer, feedback_db_path: Optional[Path] = None):
        """Initialize with base factorizer and feedback system."""
        
        self.base_factorizer = base_factorizer
        self.feedback_learner = OptimizationFeedbackLearner(feedback_db_path)
        
        # Learning controls
        self.enable_adaptive_strategies = True
        self.enable_feedback_recording = True
        self.min_confidence_for_adaptation = 0.5
        
    def factor_with_feedback(self, n: int, **kwargs) -> Dict[str, Any]:
        """Factor a number with integrated feedback learning."""
        
        start_time = time.time()
        
        # Get adaptive strategy recommendation
        adaptive_strategy = None
        if self.enable_adaptive_strategies:
            try:
                adaptive_strategy = self.feedback_learner.get_adaptive_strategy_recommendation(n)
                
                # Apply adaptive strategy if confidence is high enough
                if adaptive_strategy.confidence >= self.min_confidence_for_adaptation:
                    # Override strategy parameters based on learned recommendations
                    if "optimization_budget" not in kwargs:
                        # Estimate budget based on learned runtime
                        estimated_budget = max(5.0, adaptive_strategy.avg_runtime_ms / 100)
                        kwargs["optimization_budget"] = estimated_budget
                        
                    # Could also adapt other parameters based on strategy
                    
            except Exception as e:
                print(f"Warning: Failed to get adaptive strategy for {n}: {e}")
        
        # Perform the actual factorization
        result = self.base_factorizer.factor(n, **kwargs)
        
        end_time = time.time()
        runtime_ms = (end_time - start_time) * 1000
        
        # Record feedback if enabled
        if self.enable_feedback_recording:
            try:
                self._record_feedback_from_result(n, result, runtime_ms, adaptive_strategy, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to record feedback for {n}: {e}")
        
        # Add feedback metadata to result
        if hasattr(result, "__dict__"):
            result.feedback_metadata = {
                "adaptive_strategy_used": adaptive_strategy is not None,
                "strategy_confidence": adaptive_strategy.confidence if adaptive_strategy else 0.0,
                "learning_enabled": self.enable_feedback_recording
            }
        
        return result
    
    def _record_feedback_from_result(
        self, 
        n: int, 
        result: Any, 
        runtime_ms: float,
        adaptive_strategy: Optional[OptimizationStrategy],
        **kwargs
    ):
        """Extract feedback information from factorization result."""
        
        # Extract basic information
        was_certified = bool(getattr(result, 'tnfr_certified_factors', None))
        certified_factors = getattr(result, 'tnfr_certified_factors', [])
        verification = getattr(result, 'tnfr_verification', None)
        
        # Get operator sequence info
        operator_sequence = []
        if hasattr(result, 'self_optimization_summary') and result.self_optimization_summary:
            try:
                opt_summary = result.self_optimization_summary
                if isinstance(opt_summary, dict):
                    promotable = opt_summary.get('promotable', {})
                    if promotable:
                        # Extract sequences from promotable partitions
                        for partition_data in promotable.values():
                            if isinstance(partition_data, dict):
                                engine_data = partition_data.get('engine', {})
                                if isinstance(engine_data, dict):
                                    sequence = engine_data.get('operator_sequence', [])
                                    if sequence:
                                        operator_sequence = sequence
                                        break
            except Exception:
                pass
        
        # Default sequence if not found
        if not operator_sequence:
            operator_sequence = ["emission", "coupling", "resonance", "coherence", "silence"]
        
        # Determine number type and factor pattern
        number_type = self.feedback_learner._classify_number_type(n)
        factor_pattern = self._classify_factor_pattern(certified_factors, n)
        
        # Extract verification metrics
        verification_score = 0.0
        dnfr_gain = 0.0
        coherence_ratio = 0.0
        phi_delta_parent = 0.0
        gradient_delta = 0.0
        curvature_delta = 0.0
        periodicity_confidence = 0.0
        
        if verification and isinstance(verification, dict):
            # Extract metrics from verification data
            per_factor_data = verification.get('per_factor_summary', {})
            if per_factor_data and isinstance(per_factor_data, dict):
                # Use first certified factor's metrics as representative
                for factor_data in per_factor_data.values():
                    if isinstance(factor_data, dict):
                        dnfr_gain = factor_data.get('average_dnfr_gain', 0.0)
                        coherence_ratio = factor_data.get('average_coherence_ratio', 0.0)
                        phi_delta_parent = factor_data.get('average_phi_delta', 0.0)
                        gradient_delta = factor_data.get('average_gradient_delta', 0.0)
                        curvature_delta = factor_data.get('average_curvature_delta', 0.0)
                        periodicity_confidence = factor_data.get('average_periodicity_confidence', 0.0)
                        break
            
            # Calculate overall verification score
            criteria_met = 0
            total_criteria = 6
            
            if dnfr_gain >= 0.15: criteria_met += 1
            if 0.72 <= coherence_ratio <= 1.38: criteria_met += 1
            if phi_delta_parent <= 0.35: criteria_met += 1
            if gradient_delta <= 0.40: criteria_met += 1
            if curvature_delta <= 0.45: criteria_met += 1
            if periodicity_confidence >= 0.55: criteria_met += 1
            
            verification_score = criteria_met / total_criteria
        
        # Record feedback for each candidate factor
        modulus = getattr(result, 'modulus', n)
        node_count = getattr(result, 'node_count', 0)
        optimization_budget = kwargs.get('optimization_budget', 10.0)
        
        partition_strategy = "default"
        if adaptive_strategy:
            partition_strategy = adaptive_strategy.context_pattern
        
        # Record feedback for certified factors
        for factor in certified_factors:
            if isinstance(factor, int):
                feedback = VerificationFeedback(
                    n=n,
                    modulus=modulus,
                    node_count=node_count,
                    candidate_factor=factor,
                    was_certified=True,
                    verification_score=verification_score,
                    dnfr_gain=dnfr_gain,
                    coherence_ratio=coherence_ratio,
                    phi_delta_parent=phi_delta_parent,
                    gradient_delta=gradient_delta,
                    curvature_delta=curvature_delta,
                    periodicity_confidence=periodicity_confidence,
                    partition_strategy=partition_strategy,
                    operator_sequence=operator_sequence,
                    optimization_budget=optimization_budget,
                    runtime_ms=runtime_ms,
                    convergence_iterations=0,  # Not easily available
                    number_type=number_type,
                    factor_pattern=factor_pattern,
                    timestamp=time.time()
                )
                
                self.feedback_learner.record_verification_feedback(feedback)
                
                # Update adaptive strategy
                if adaptive_strategy:
                    self.feedback_learner.update_strategy_from_feedback(feedback)
        
        # Also record one general feedback entry for the overall factorization
        if not certified_factors:
            feedback = VerificationFeedback(
                n=n,
                modulus=modulus,
                node_count=node_count,
                candidate_factor=None,
                was_certified=False,
                verification_score=verification_score,
                dnfr_gain=dnfr_gain,
                coherence_ratio=coherence_ratio,
                phi_delta_parent=phi_delta_parent,
                gradient_delta=gradient_delta,
                curvature_delta=curvature_delta,
                periodicity_confidence=periodicity_confidence,
                partition_strategy=partition_strategy,
                operator_sequence=operator_sequence,
                optimization_budget=optimization_budget,
                runtime_ms=runtime_ms,
                convergence_iterations=0,
                number_type=number_type,
                factor_pattern="no_factors_found",
                timestamp=time.time()
            )
            
            self.feedback_learner.record_verification_feedback(feedback)
            
            if adaptive_strategy:
                self.feedback_learner.update_strategy_from_feedback(feedback)
    
    def _classify_factor_pattern(self, factors: List[int], n: int) -> str:
        """Classify the pattern of found factors."""
        
        if not factors:
            return "no_factors"
        
        if len(factors) == 1:
            factor = factors[0]
            other_factor = n // factor
            
            # Check if factors are close
            ratio = max(factor, other_factor) / min(factor, other_factor)
            if ratio < 2.0:
                return "close_factors"
            elif ratio > 100:
                return "large_gap_factors" 
            else:
                return "moderate_gap_factors"
        
        elif len(factors) == 2:
            return "two_factors_found"
        
        elif len(factors) >= 3:
            return "multiple_factors_found"
        
        return "unknown_pattern"
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning status and performance."""
        
        analysis = self.feedback_learner.analyze_feedback_patterns()
        performance = self.feedback_learner.get_performance_summary()
        
        return {
            "learning_status": {
                "adaptive_strategies_enabled": self.enable_adaptive_strategies,
                "feedback_recording_enabled": self.enable_feedback_recording,
                "min_confidence_threshold": self.min_confidence_for_adaptation
            },
            "performance_metrics": performance,
            "feedback_analysis": {
                "success_rate_by_strategy": analysis.success_rate_by_strategy,
                "best_strategy_by_number_type": analysis.best_strategy_by_number_type,
                "confidence_score": analysis.confidence_score,
                "recommendations": analysis.optimization_recommendations
            }
        }
    
    def export_learning_report(self, output_path: Path):
        """Export comprehensive learning report."""
        
        self.feedback_learner.export_feedback_report(output_path)
        
        # Add integration-specific information
        integration_report_path = output_path.parent / f"integration_{output_path.name}"
        
        integration_data = {
            "integration_summary": self.get_learning_summary(),
            "factorizer_type": type(self.base_factorizer).__name__,
            "feedback_integration_version": "1.0"
        }
        
        with open(integration_report_path, 'w') as f:
            json.dump(integration_data, f, indent=2)
        
        print(f"Integration report exported: {integration_report_path}")


def create_feedback_integrated_factorizer(base_factorizer, feedback_db_path: Optional[Path] = None):
    """Convenience function to create a feedback-integrated factorizer."""
    
    return FeedbackIntegratedFactorizer(base_factorizer, feedback_db_path)


def enable_feedback_integration_for_factorizer(factorizer, feedback_db_path: Optional[Path] = None):
    """Enable feedback integration for an existing factorizer instance."""
    
    # Add feedback methods to the factorizer
    feedback_learner = OptimizationFeedbackLearner(feedback_db_path)
    factorizer._feedback_learner = feedback_learner
    
    # Store original factor method
    original_factor = factorizer.factor
    
    def factor_with_feedback(n: int, **kwargs):
        """Enhanced factor method with feedback integration."""
        
        start_time = time.time()
        
        # Get recommendation
        try:
            adaptive_strategy = feedback_learner.get_adaptive_strategy_recommendation(n)
            if adaptive_strategy.confidence >= 0.5 and "optimization_budget" not in kwargs:
                kwargs["optimization_budget"] = max(5.0, adaptive_strategy.avg_runtime_ms / 100)
        except Exception:
            pass
        
        # Factor with original method
        result = original_factor(n, **kwargs)
        
        # Record feedback
        try:
            runtime_ms = (time.time() - start_time) * 1000
            # Simplified feedback recording logic would go here
            pass
        except Exception:
            pass
        
        return result
    
    # Replace factor method
    factorizer.factor = factor_with_feedback
    
    return factorizer