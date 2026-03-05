"""
TNFR Self-Optimization Feedback Integration Demo

Lightweight demonstration of the feedback learning system without
requiring full TNFR environment dependencies.
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path

# Mock the feedback system for demonstration
class MockVerificationFeedback:
    """Mock verification feedback for demo."""
    
    def __init__(self, n, candidate_factor, was_certified, strategy, runtime_ms, number_type):
        self.n = n
        self.candidate_factor = candidate_factor
        self.was_certified = was_certified
        self.verification_score = 0.8 if was_certified else 0.3
        self.dnfr_gain = 0.2 if was_certified else 0.05
        self.coherence_ratio = 0.9 if was_certified else 0.4
        self.phi_delta_parent = 0.1 if was_certified else 0.5
        self.gradient_delta = 0.15
        self.curvature_delta = 0.2
        self.periodicity_confidence = 0.7 if was_certified else 0.2
        self.partition_strategy = strategy
        self.operator_sequence = ["emission", "coupling", "resonance", "coherence", "silence"]
        self.optimization_budget = 10.0
        self.runtime_ms = runtime_ms
        self.convergence_iterations = 3
        self.number_type = number_type
        self.factor_pattern = "close_factors" if was_certified else "no_factors"
        self.timestamp = time.time()
        self.modulus = n
        self.node_count = 10


class FeedbackLearningDemo:
    """Demonstration of feedback learning capabilities."""
    
    def __init__(self):
        """Initialize demo with temporary database."""
        self.temp_dir = tempfile.mkdtemp(prefix="tnfr_feedback_demo_")
        self.db_path = Path(self.temp_dir) / "demo_feedback.db"
        self._init_database()
        
        print(f"Demo database created: {self.db_path}")
    
    def _init_database(self):
        """Initialize demo database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    n INTEGER NOT NULL,
                    candidate_factor INTEGER,
                    was_certified INTEGER NOT NULL,
                    verification_score REAL,
                    partition_strategy TEXT,
                    runtime_ms REAL,
                    number_type TEXT,
                    factor_pattern TEXT,
                    timestamp REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy TEXT PRIMARY KEY,
                    total_attempts INTEGER DEFAULT 0,
                    successful_attempts INTEGER DEFAULT 0,
                    avg_runtime_ms REAL DEFAULT 0,
                    success_rate REAL DEFAULT 0
                )
            """)
    
    def simulate_factorization_attempts(self, num_attempts=50):
        """Simulate a series of factorization attempts with feedback."""
        
        print(f"\nSimulating {num_attempts} factorization attempts...")
        
        # Define test strategies with different success rates
        strategies = {
            "basic_strategy": 0.3,      # 30% success rate
            "optimized_strategy": 0.7,  # 70% success rate
            "experimental_strategy": 0.5, # 50% success rate
            "adaptive_strategy": 0.8    # 80% success rate (learned)
        }
        
        # Simulate attempts
        for i in range(num_attempts):
            # Choose random number and strategy
            n = 35 + (i % 20) * 7  # Various semiprimes
            strategy = list(strategies.keys())[i % len(strategies)]
            
            # Simulate success based on strategy performance
            import random
            was_certified = random.random() < strategies[strategy]
            
            # Runtime varies by strategy and success
            base_runtime = {
                "basic_strategy": 800,
                "optimized_strategy": 600, 
                "experimental_strategy": 1000,
                "adaptive_strategy": 500
            }[strategy]
            
            runtime_ms = base_runtime + random.uniform(-200, 300)
            if not was_certified:
                runtime_ms *= 1.2  # Failed attempts take longer
            
            # Classify number type
            number_type = self._classify_number(n)
            
            # Record feedback
            feedback = MockVerificationFeedback(
                n=n,
                candidate_factor=5 if was_certified else None,
                was_certified=was_certified,
                strategy=strategy,
                runtime_ms=runtime_ms,
                number_type=number_type
            )
            
            self._record_feedback(feedback)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_attempts} attempts")
        
        print("✓ Simulation completed")
    
    def _classify_number(self, n):
        """Simple number classification."""
        # Quick check for common patterns
        if n < 100:
            return "small_composite"
        elif n < 300:
            return "medium_composite" 
        else:
            return "large_composite"
    
    def _record_feedback(self, feedback):
        """Record feedback in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO verification_feedback (
                    n, candidate_factor, was_certified, verification_score,
                    partition_strategy, runtime_ms, number_type, factor_pattern, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.n, feedback.candidate_factor, int(feedback.was_certified),
                feedback.verification_score, feedback.partition_strategy,
                feedback.runtime_ms, feedback.number_type, 
                feedback.factor_pattern, feedback.timestamp
            ))
    
    def analyze_performance(self):
        """Analyze strategy performance from recorded feedback."""
        
        print("\nAnalyzing strategy performance...")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Calculate performance by strategy
            performance_query = """
                SELECT 
                    partition_strategy,
                    COUNT(*) as total_attempts,
                    SUM(was_certified) as successful_attempts,
                    AVG(runtime_ms) as avg_runtime_ms,
                    AVG(CAST(was_certified AS FLOAT)) as success_rate
                FROM verification_feedback 
                GROUP BY partition_strategy
                ORDER BY success_rate DESC
            """
            
            results = conn.execute(performance_query).fetchall()
            
            print("\n" + "="*60)
            print("STRATEGY PERFORMANCE ANALYSIS")
            print("="*60)
            print(f"{'Strategy':<20} {'Success Rate':<12} {'Avg Runtime':<12} {'Attempts':<10}")
            print("-" * 60)
            
            for row in results:
                strategy = row['partition_strategy']
                success_rate = row['success_rate'] * 100
                avg_runtime = row['avg_runtime_ms']
                attempts = row['total_attempts']
                
                print(f"{strategy:<20} {success_rate:>8.1f}%    {avg_runtime:>8.0f}ms   {attempts:>8}")
            
            # Identify best strategy
            best_strategy = results[0] if results else None
            if best_strategy:
                print(f"\n🏆 Best performing strategy: {best_strategy['partition_strategy']}")
                print(f"   Success rate: {best_strategy['success_rate']*100:.1f}%")
                print(f"   Average runtime: {best_strategy['avg_runtime_ms']:.0f}ms")
    
    def demonstrate_adaptive_learning(self):
        """Demonstrate how the system would adapt strategies."""
        
        print("\n" + "="*60)
        print("ADAPTIVE LEARNING DEMONSTRATION")
        print("="*60)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get strategy recommendations by number type
            recommendation_query = """
                SELECT 
                    number_type,
                    partition_strategy,
                    AVG(CAST(was_certified AS FLOAT)) as success_rate,
                    COUNT(*) as sample_count
                FROM verification_feedback 
                GROUP BY number_type, partition_strategy
                HAVING sample_count >= 3
                ORDER BY number_type, success_rate DESC
            """
            
            results = conn.execute(recommendation_query).fetchall()
            
            current_number_type = None
            for row in results:
                number_type = row['number_type']
                
                if number_type != current_number_type:
                    current_number_type = number_type
                    print(f"\n📊 Recommendations for {number_type}:")
                    
                    # Show best strategy for this number type
                    best_row = next(r for r in results if r['number_type'] == number_type)
                    print(f"   ✅ Recommended: {best_row['partition_strategy']}")
                    print(f"      Success rate: {best_row['success_rate']*100:.1f}%")
                    print(f"      Based on {best_row['sample_count']} samples")
                    
                    # Show alternatives
                    alternatives = [r for r in results 
                                  if r['number_type'] == number_type and 
                                     r['partition_strategy'] != best_row['partition_strategy']][:2]
                    
                    if alternatives:
                        print("   📋 Alternatives:")
                        for alt in alternatives:
                            print(f"      - {alt['partition_strategy']}: {alt['success_rate']*100:.1f}% success")
    
    def generate_optimization_recommendations(self):
        """Generate actionable optimization recommendations."""
        
        print("\n" + "="*60) 
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        with sqlite3.connect(self.db_path) as conn:
            # Find underperforming strategies
            underperforming = conn.execute("""
                SELECT partition_strategy, AVG(CAST(was_certified AS FLOAT)) as success_rate
                FROM verification_feedback 
                GROUP BY partition_strategy
                HAVING COUNT(*) >= 5 AND success_rate < 0.4
            """).fetchall()
            
            # Find slow strategies
            slow_strategies = conn.execute("""
                SELECT partition_strategy, AVG(runtime_ms) as avg_runtime
                FROM verification_feedback
                GROUP BY partition_strategy  
                HAVING COUNT(*) >= 5
                ORDER BY avg_runtime DESC
                LIMIT 2
            """).fetchall()
            
            # Generate recommendations
            recommendations = []
            
            if underperforming:
                for strategy, success_rate in underperforming:
                    recommendations.append(
                        f"⚠️  Strategy '{strategy}' has low {success_rate*100:.1f}% success rate - consider revision"
                    )
            
            if slow_strategies:
                fastest_time = min(row[1] for row in slow_strategies)
                for strategy, avg_runtime in slow_strategies:
                    if avg_runtime > fastest_time * 1.5:
                        recommendations.append(
                            f"🐌 Strategy '{strategy}' is slow ({avg_runtime:.0f}ms avg) - optimize performance"
                        )
            
            # Success patterns
            successful_patterns = conn.execute("""
                SELECT partition_strategy, AVG(CAST(was_certified AS FLOAT)) as success_rate
                FROM verification_feedback
                GROUP BY partition_strategy
                HAVING COUNT(*) >= 5 AND success_rate > 0.7
                ORDER BY success_rate DESC
            """).fetchall()
            
            if successful_patterns:
                best_strategy, best_rate = successful_patterns[0]
                recommendations.append(
                    f"🌟 Strategy '{best_strategy}' shows excellent {best_rate*100:.1f}% success - consider as default"
                )
            
            # Print recommendations
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
            else:
                print("✅ All strategies performing within acceptable ranges")
    
    def export_feedback_report(self):
        """Export comprehensive feedback report."""
        
        report_path = Path(self.temp_dir) / "feedback_learning_report.json"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get summary statistics
            total_attempts = conn.execute("SELECT COUNT(*) FROM verification_feedback").fetchone()[0]
            overall_success = conn.execute("""
                SELECT AVG(CAST(was_certified AS FLOAT)) FROM verification_feedback
            """).fetchone()[0]
            
            # Get strategy breakdown
            strategy_stats = conn.execute("""
                SELECT 
                    partition_strategy,
                    COUNT(*) as attempts,
                    AVG(CAST(was_certified AS FLOAT)) as success_rate,
                    AVG(runtime_ms) as avg_runtime
                FROM verification_feedback
                GROUP BY partition_strategy
            """).fetchall()
            
            report = {
                "summary": {
                    "total_attempts": total_attempts,
                    "overall_success_rate": overall_success,
                    "database_path": str(self.db_path)
                },
                "strategy_performance": [
                    {
                        "strategy": row["partition_strategy"],
                        "attempts": row["attempts"],
                        "success_rate": row["success_rate"],
                        "avg_runtime_ms": row["avg_runtime"]
                    }
                    for row in strategy_stats
                ],
                "generated_at": time.time()
            }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Feedback report exported: {report_path}")
        return report_path
    
    def cleanup(self):
        """Clean up demo files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"\n🧹 Demo files cleaned up")


def run_feedback_integration_demo():
    """Run the complete feedback integration demonstration."""
    
    print("TNFR SELF-OPTIMIZATION FEEDBACK INTEGRATION DEMO")
    print("="*65)
    print("Demonstrating closed-loop optimization with adaptive learning")
    print()
    
    # Create demo instance
    demo = FeedbackLearningDemo()
    
    try:
        # Run simulation
        demo.simulate_factorization_attempts(50)
        
        # Analyze results  
        demo.analyze_performance()
        
        # Show adaptive learning
        demo.demonstrate_adaptive_learning()
        
        # Generate recommendations
        demo.generate_optimization_recommendations()
        
        # Export report
        report_path = demo.export_feedback_report()
        
        print("\n" + "="*65)
        print("✅ FEEDBACK INTEGRATION DEMO COMPLETED SUCCESSFULLY")
        print("\nKey Capabilities Demonstrated:")
        print("• Automated feedback collection from verification results")
        print("• Strategy performance analysis and ranking")
        print("• Adaptive recommendations based on number type patterns")
        print("• Continuous learning from success/failure patterns")
        print("• Optimization recommendations for underperforming strategies")
        print("• Comprehensive reporting and analytics")
        print(f"\n📊 Detailed report available at: {report_path}")
        
    finally:
        # Cleanup
        demo.cleanup()
    
    return True


if __name__ == "__main__":
    success = run_feedback_integration_demo()
    print(f"\nDemo {'completed successfully' if success else 'failed'}")
    exit(0 if success else 1)