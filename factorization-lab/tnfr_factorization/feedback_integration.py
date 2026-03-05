"""
TNFR Self-Optimization Feedback Integration

Closes the optimization loop by integrating adaptive candidate refinement 
based on certificate feedback. Implements learning mechanisms that adjust 
partition strategies based on verification success rates.

This system:
1. Analyzes verification success/failure patterns from certificates
2. Learns which partition strategies work best for different number types
3. Adapts candidate selection based on historical performance
4. Provides feedback-driven optimization recommendations
5. Maintains learning databases for continuous improvement
"""

import json
import math
import os
import sqlite3
import statistics
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set


@dataclass 
class VerificationFeedback:
    """Structured feedback from a verification attempt."""
    
    n: int
    modulus: int
    node_count: int
    candidate_factor: Optional[int]
    was_certified: bool
    verification_score: float  # Overall verification strength (0.0-1.0)
    
    # Verification metrics
    dnfr_gain: float
    coherence_ratio: float  
    phi_delta_parent: float
    gradient_delta: float
    curvature_delta: float
    periodicity_confidence: float
    
    # Strategy context
    partition_strategy: str
    operator_sequence: List[str]
    optimization_budget: float
    
    # Performance metrics
    runtime_ms: float
    convergence_iterations: int
    
    # Contextual information
    number_type: str  # e.g., "semiprime", "triprime", "carmichael", etc.
    factor_pattern: str  # e.g., "close_primes", "large_gap", "power_of_prime"
    
    timestamp: float


@dataclass
class OptimizationStrategy:
    """Learned optimization strategy for specific contexts."""
    
    context_pattern: str  # Pattern this strategy applies to
    recommended_sequence: List[str]
    expected_success_rate: float
    avg_runtime_ms: float
    confidence: float  # How confident we are in this strategy
    sample_count: int  # Number of samples this is based on
    last_updated: float


@dataclass  
class FeedbackAnalysis:
    """Analysis of feedback patterns and recommendations."""
    
    success_rate_by_strategy: Dict[str, float]
    best_strategy_by_number_type: Dict[str, str]
    runtime_performance_by_strategy: Dict[str, float] 
    common_failure_patterns: List[Tuple[str, float]]  # (pattern, frequency)
    optimization_recommendations: List[str]
    confidence_score: float


class OptimizationFeedbackLearner:
    """Learning system for self-optimization feedback integration."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the feedback learner with persistent storage."""
        
        self.db_path = db_path or Path("results/optimization_feedback.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory caches for performance
        self._strategy_cache: Dict[str, OptimizationStrategy] = {}
        self._feedback_buffer: List[VerificationFeedback] = []
        
        # Learning parameters
        self.min_samples_for_strategy = 5
        self.confidence_threshold = 0.7
        self.adaptation_rate = 0.1  # How quickly to adapt to new evidence
        
    def _init_database(self):
        """Initialize the SQLite database for feedback storage."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    n INTEGER NOT NULL,
                    modulus INTEGER,
                    node_count INTEGER,
                    candidate_factor INTEGER,
                    was_certified INTEGER NOT NULL,
                    verification_score REAL,
                    dnfr_gain REAL,
                    coherence_ratio REAL,
                    phi_delta_parent REAL,
                    gradient_delta REAL,
                    curvature_delta REAL,
                    periodicity_confidence REAL,
                    partition_strategy TEXT,
                    operator_sequence TEXT,
                    optimization_budget REAL,
                    runtime_ms REAL,
                    convergence_iterations INTEGER,
                    number_type TEXT,
                    factor_pattern TEXT,
                    timestamp REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_pattern TEXT NOT NULL UNIQUE,
                    recommended_sequence TEXT,
                    expected_success_rate REAL,
                    avg_runtime_ms REAL,
                    confidence REAL,
                    sample_count INTEGER,
                    last_updated REAL
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_n ON verification_feedback(n)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_strategy ON verification_feedback(partition_strategy)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_number_type ON verification_feedback(number_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategies_pattern ON optimization_strategies(context_pattern)")
    
    def record_verification_feedback(self, feedback: VerificationFeedback):
        """Record feedback from a verification attempt."""
        
        # Add to buffer for batch processing
        self._feedback_buffer.append(feedback)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO verification_feedback (
                    n, modulus, node_count, candidate_factor, was_certified,
                    verification_score, dnfr_gain, coherence_ratio, 
                    phi_delta_parent, gradient_delta, curvature_delta,
                    periodicity_confidence, partition_strategy, operator_sequence,
                    optimization_budget, runtime_ms, convergence_iterations,
                    number_type, factor_pattern, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.n, feedback.modulus, feedback.node_count,
                feedback.candidate_factor, int(feedback.was_certified),
                feedback.verification_score, feedback.dnfr_gain, feedback.coherence_ratio,
                feedback.phi_delta_parent, feedback.gradient_delta, feedback.curvature_delta,
                feedback.periodicity_confidence, feedback.partition_strategy,
                json.dumps(feedback.operator_sequence), feedback.optimization_budget,
                feedback.runtime_ms, feedback.convergence_iterations,
                feedback.number_type, feedback.factor_pattern, feedback.timestamp
            ))
    
    def analyze_feedback_patterns(self, min_samples: int = 10) -> FeedbackAnalysis:
        """Analyze accumulated feedback to identify optimization patterns."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all feedback data
            rows = conn.execute("""
                SELECT * FROM verification_feedback 
                ORDER BY timestamp DESC LIMIT 1000
            """).fetchall()
        
        if len(rows) < min_samples:
            return FeedbackAnalysis(
                success_rate_by_strategy={},
                best_strategy_by_number_type={},
                runtime_performance_by_strategy={},
                common_failure_patterns=[],
                optimization_recommendations=["Insufficient data for analysis"],
                confidence_score=0.0
            )
        
        # Analyze success rates by strategy
        strategy_stats = defaultdict(lambda: {"success": 0, "total": 0, "runtimes": []})
        
        for row in rows:
            strategy = row["partition_strategy"] or "unknown"
            strategy_stats[strategy]["total"] += 1
            strategy_stats[strategy]["runtimes"].append(row["runtime_ms"] or 0)
            
            if row["was_certified"]:
                strategy_stats[strategy]["success"] += 1
        
        success_rate_by_strategy = {
            strategy: stats["success"] / max(stats["total"], 1)
            for strategy, stats in strategy_stats.items()
            if stats["total"] >= 3  # Minimum samples
        }
        
        runtime_performance_by_strategy = {
            strategy: statistics.mean(stats["runtimes"]) if stats["runtimes"] else 0
            for strategy, stats in strategy_stats.items()
        }
        
        # Analyze by number type
        number_type_stats = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        
        for row in rows:
            number_type = row["number_type"] or "unknown"
            strategy = row["partition_strategy"] or "unknown"
            
            number_type_stats[number_type][strategy]["total"] += 1
            if row["was_certified"]:
                number_type_stats[number_type][strategy]["success"] += 1
        
        best_strategy_by_number_type = {}
        for number_type, strategies in number_type_stats.items():
            if not strategies:
                continue
                
            best_strategy = max(
                strategies.items(),
                key=lambda x: (
                    x[1]["success"] / max(x[1]["total"], 1),  # Success rate
                    x[1]["total"]  # Sample size as tiebreaker
                )
            )[0]
            
            if strategies[best_strategy]["total"] >= 3:
                best_strategy_by_number_type[number_type] = best_strategy
        
        # Identify failure patterns
        failure_patterns = Counter()
        for row in rows:
            if not row["was_certified"]:
                # Identify potential failure causes
                if row["dnfr_gain"] is not None and row["dnfr_gain"] < 0.1:
                    failure_patterns["low_dnfr_gain"] += 1
                if row["periodicity_confidence"] is not None and row["periodicity_confidence"] < 0.5:
                    failure_patterns["low_periodicity_confidence"] += 1
                if row["coherence_ratio"] is not None and row["coherence_ratio"] < 0.7:
                    failure_patterns["low_coherence"] += 1
                if row["phi_delta_parent"] is not None and row["phi_delta_parent"] > 0.4:
                    failure_patterns["high_phi_delta"] += 1
        
        total_failures = sum(1 for row in rows if not row["was_certified"])
        common_failure_patterns = [
            (pattern, count / max(total_failures, 1))
            for pattern, count in failure_patterns.most_common(5)
        ]
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            success_rate_by_strategy,
            runtime_performance_by_strategy,
            common_failure_patterns
        )
        
        # Calculate overall confidence
        total_samples = len(rows)
        confidence_score = min(1.0, total_samples / 100.0)  # Full confidence at 100 samples
        
        return FeedbackAnalysis(
            success_rate_by_strategy=success_rate_by_strategy,
            best_strategy_by_number_type=best_strategy_by_number_type,
            runtime_performance_by_strategy=runtime_performance_by_strategy,
            common_failure_patterns=common_failure_patterns,
            optimization_recommendations=recommendations,
            confidence_score=confidence_score
        )
    
    def _generate_optimization_recommendations(
        self,
        success_rates: Dict[str, float],
        runtimes: Dict[str, float],
        failure_patterns: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate actionable optimization recommendations."""
        
        recommendations = []
        
        # Strategy recommendations
        if success_rates:
            best_strategy = max(success_rates.items(), key=lambda x: x[1])
            if best_strategy[1] > 0.8:
                recommendations.append(f"Strategy '{best_strategy[0]}' shows {best_strategy[1]:.1%} success rate - consider as default")
            
            worst_strategy = min(success_rates.items(), key=lambda x: x[1])
            if worst_strategy[1] < 0.3:
                recommendations.append(f"Strategy '{worst_strategy[0]}' has low {worst_strategy[1]:.1%} success rate - consider revision")
        
        # Runtime recommendations
        if runtimes:
            fastest_strategy = min(runtimes.items(), key=lambda x: x[1])
            slowest_strategy = max(runtimes.items(), key=lambda x: x[1])
            
            if slowest_strategy[1] > 2 * fastest_strategy[1]:
                recommendations.append(f"Strategy '{slowest_strategy[0]}' is {slowest_strategy[1]/fastest_strategy[1]:.1f}x slower than '{fastest_strategy[0]}' - optimize performance")
        
        # Failure pattern recommendations
        for pattern, frequency in failure_patterns:
            if frequency > 0.3:  # More than 30% of failures
                if pattern == "low_dnfr_gain":
                    recommendations.append("High frequency of low ΔNFR gain failures - consider stronger destabilization operators")
                elif pattern == "low_periodicity_confidence":
                    recommendations.append("Low periodicity confidence is common - improve spectral analysis accuracy")
                elif pattern == "low_coherence":
                    recommendations.append("Coherence issues detected - add more stabilization steps")
                elif pattern == "high_phi_delta":
                    recommendations.append("High phi delta failures - improve structural potential management")
        
        if not recommendations:
            recommendations.append("Feedback patterns look healthy - continue current strategies")
        
        return recommendations
    
    def get_adaptive_strategy_recommendation(
        self, 
        n: int,
        number_type: Optional[str] = None,
        factor_pattern: Optional[str] = None
    ) -> OptimizationStrategy:
        """Get adaptive strategy recommendation based on learned patterns."""
        
        # Determine number type if not provided
        if number_type is None:
            number_type = self._classify_number_type(n)
        
        # Check cache first
        cache_key = f"{number_type}_{factor_pattern or 'unknown'}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        # Query database for similar cases
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get feedback for similar number types
            rows = conn.execute("""
                SELECT partition_strategy, operator_sequence, was_certified, 
                       runtime_ms, verification_score
                FROM verification_feedback 
                WHERE number_type = ? AND was_certified = 1
                ORDER BY timestamp DESC LIMIT 50
            """, (number_type,)).fetchall()
        
        if not rows:
            # Fallback to default strategy
            strategy = OptimizationStrategy(
                context_pattern=cache_key,
                recommended_sequence=["emission", "coupling", "resonance", "coherence", "silence"],
                expected_success_rate=0.5,
                avg_runtime_ms=1000.0,
                confidence=0.1,
                sample_count=0,
                last_updated=time.time()
            )
        else:
            # Analyze successful strategies
            strategy_performance = defaultdict(lambda: {"success": 0, "total": 0, "runtimes": []})
            
            for row in rows:
                strategy_name = row["partition_strategy"] or "default"
                strategy_performance[strategy_name]["total"] += 1
                strategy_performance[strategy_name]["runtimes"].append(row["runtime_ms"] or 1000)
                
                if row["was_certified"]:
                    strategy_performance[strategy_name]["success"] += 1
            
            # Find best performing strategy
            best_strategy_name = max(
                strategy_performance.items(),
                key=lambda x: (x[1]["success"] / max(x[1]["total"], 1), x[1]["total"])
            )[0]
            
            best_perf = strategy_performance[best_strategy_name]
            success_rate = best_perf["success"] / max(best_perf["total"], 1)
            avg_runtime = statistics.mean(best_perf["runtimes"]) if best_perf["runtimes"] else 1000
            
            # Get representative operator sequence
            rep_sequence_row = next(
                (row for row in rows if row["partition_strategy"] == best_strategy_name),
                rows[0]
            )
            
            try:
                rep_sequence = json.loads(rep_sequence_row["operator_sequence"] or "[]")
            except (json.JSONDecodeError, TypeError):
                rep_sequence = ["emission", "coupling", "resonance", "coherence", "silence"]
            
            confidence = min(0.9, best_perf["total"] / self.min_samples_for_strategy)
            
            strategy = OptimizationStrategy(
                context_pattern=cache_key,
                recommended_sequence=rep_sequence,
                expected_success_rate=success_rate,
                avg_runtime_ms=avg_runtime,
                confidence=confidence,
                sample_count=best_perf["total"],
                last_updated=time.time()
            )
        
        # Cache the strategy
        self._strategy_cache[cache_key] = strategy
        
        # Store in database
        self._store_strategy(strategy)
        
        return strategy
    
    def _classify_number_type(self, n: int) -> str:
        """Classify a number by its factorization pattern."""
        
        # Quick factorization to determine type
        factors = []
        temp_n = n
        
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while temp_n % p == 0:
                factors.append(p)
                temp_n //= p
        
        if temp_n > 1:
            factors.append(temp_n)
        
        unique_factors = len(set(factors))
        total_factors = len(factors)
        
        # Classification logic
        if total_factors == 1:
            return "prime"
        elif total_factors == 2 and unique_factors == 2:
            return "semiprime"
        elif total_factors == 3 and unique_factors == 3:
            return "triprime"
        elif unique_factors == 1:
            return "prime_power"
        elif unique_factors == 2:
            return "semiprime_power"
        elif unique_factors >= 3:
            return "highly_composite"
        else:
            return "composite"
    
    def _store_strategy(self, strategy: OptimizationStrategy):
        """Store or update a strategy in the database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO optimization_strategies (
                    context_pattern, recommended_sequence, expected_success_rate,
                    avg_runtime_ms, confidence, sample_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.context_pattern,
                json.dumps(strategy.recommended_sequence),
                strategy.expected_success_rate,
                strategy.avg_runtime_ms,
                strategy.confidence,
                strategy.sample_count,
                strategy.last_updated
            ))
    
    def update_strategy_from_feedback(self, feedback: VerificationFeedback):
        """Update strategies based on new feedback using adaptive learning."""
        
        context_key = f"{feedback.number_type}_{feedback.factor_pattern}"
        
        if context_key in self._strategy_cache:
            strategy = self._strategy_cache[context_key]
            
            # Adaptive update using exponential moving average
            new_success_rate = (
                (1 - self.adaptation_rate) * strategy.expected_success_rate +
                self.adaptation_rate * (1.0 if feedback.was_certified else 0.0)
            )
            
            new_runtime = (
                (1 - self.adaptation_rate) * strategy.avg_runtime_ms +
                self.adaptation_rate * feedback.runtime_ms
            )
            
            strategy.expected_success_rate = new_success_rate
            strategy.avg_runtime_ms = new_runtime
            strategy.sample_count += 1
            strategy.last_updated = time.time()
            
            # Update confidence based on sample count
            strategy.confidence = min(0.9, strategy.sample_count / self.min_samples_for_strategy)
            
            # Store updated strategy
            self._store_strategy(strategy)
    
    def export_feedback_report(self, output_path: Path):
        """Export comprehensive feedback analysis report."""
        
        analysis = self.analyze_feedback_patterns()
        
        report = {
            "timestamp": time.time(),
            "database_path": str(self.db_path),
            "analysis": asdict(analysis),
            "learning_parameters": {
                "min_samples_for_strategy": self.min_samples_for_strategy,
                "confidence_threshold": self.confidence_threshold,
                "adaptation_rate": self.adaptation_rate
            },
            "cache_status": {
                "cached_strategies": len(self._strategy_cache),
                "buffered_feedback": len(self._feedback_buffer)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Feedback analysis report exported: {output_path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        
        with sqlite3.connect(self.db_path) as conn:
            total_feedback = conn.execute("SELECT COUNT(*) FROM verification_feedback").fetchone()[0]
            recent_success_rate = conn.execute("""
                SELECT AVG(CAST(was_certified AS FLOAT)) FROM verification_feedback 
                WHERE timestamp > ?
            """, (time.time() - 7*24*3600,)).fetchone()[0] or 0  # Last 7 days
            
            total_strategies = conn.execute("SELECT COUNT(*) FROM optimization_strategies").fetchone()[0]
        
        return {
            "total_feedback_records": total_feedback,
            "recent_success_rate": recent_success_rate,
            "learned_strategies": total_strategies,
            "cache_size": len(self._strategy_cache),
            "confidence_threshold": self.confidence_threshold
        }