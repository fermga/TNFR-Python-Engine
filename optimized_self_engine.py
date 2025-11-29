#!/usr/bin/env python3
"""
TNFR Self-Optimizing Engine v9.7.0 (Optimized & Refactored)
Mathematical Auto-Improvement with Clear Operational Visibility

This engine demonstrates:
1. Real TNFR physics implementation
2. Clear step-by-step optimization process
3. Meaningful metrics and improvements
4. Robust error handling
5. Physics-based insights
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import networkx as nx
import numpy as np
from typing import Dict, Tuple, Any, List

class TNFROptimizedEngine:
    """Self-Optimizing Engine v9.7.0 - Mathematical Auto-Improvement"""
    
    def __init__(self, num_nodes: int = 15, connectivity: float = 0.3):
        print(f'🏗️  Initializing TNFR network ({num_nodes} nodes, density={connectivity:.2f})...')
        
        # Create network with proper TNFR attributes
        self.G = nx.erdos_renyi_graph(num_nodes, connectivity, directed=True)
        self._initialize_tnfr_nodes()
        
        # Track optimization history
        self.history: List[Dict] = []
        
        print(f'   ✅ Network created: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges')
    
    def _initialize_tnfr_nodes(self):
        """Initialize nodes with proper TNFR physics attributes"""
        for node in self.G.nodes():
            # EPI (Primary Information Structure) - coherent form
            self.G.nodes[node]['epi'] = np.random.rand(4) * 2.0  # Vector in R^4
            
            # νf (structural frequency) - reorganization rate in Hz_str
            self.G.nodes[node]['vf'] = 0.5 + np.random.rand() * 2.0
            
            # Phase φ - network synchronization parameter [0, 2π]
            self.G.nodes[node]['phase'] = np.random.rand() * 2 * np.pi
            
            # ΔNFR (reorganization gradient) - internal pressure
            self.G.nodes[node]['delta_nfr'] = 0.1 + np.random.rand() * 0.8
    
    def compute_coherence(self) -> float:
        """Advanced coherence computation based on phase synchrony and coupling"""
        if len(self.G.nodes) == 0:
            return 1.0
            
        # Phase synchrony component
        total_sync = 0.0
        edge_count = 0
        
        for u, v in self.G.edges():
            phase_diff = abs(self.G.nodes[u]['phase'] - self.G.nodes[v]['phase'])
            # Handle phase wrapping properly
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            
            # Synchrony = cos(phase_difference) 
            sync = np.cos(phase_diff)
            
            # Weight by structural frequencies (resonance condition)
            vf_u = self.G.nodes[u]['vf']
            vf_v = self.G.nodes[v]['vf']
            vf_coupling = 1.0 / (1.0 + abs(vf_u - vf_v))
            
            total_sync += sync * vf_coupling
            edge_count += 1
        
        if edge_count == 0:
            return 0.5  # Isolated network
            
        # Average synchrony normalized to [0, 1]
        avg_sync = total_sync / edge_count
        coherence = (avg_sync + 1.0) / 2.0  # Map [-1,1] → [0,1]
        
        return np.clip(coherence, 0.0, 1.0)
    
    def compute_health_metrics(self) -> Dict[str, float]:
        """Compute TNFR structural health indicators"""
        metrics = {}
        
        # 1. Average ΔNFR (reorganization pressure)
        dnfr_values = [self.G.nodes[n]['delta_nfr'] for n in self.G.nodes()]
        metrics['avg_delta_nfr'] = np.mean(dnfr_values)
        metrics['max_delta_nfr'] = np.max(dnfr_values)
        
        # 2. Phase gradient |∇φ| (local desynchronization)
        phase_gradients = []
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                node_phase = self.G.nodes[node]['phase']
                neighbor_phases = [self.G.nodes[n]['phase'] for n in neighbors]
                
                # Compute circular mean of neighbor phases
                sin_sum = sum(np.sin(p) for p in neighbor_phases)
                cos_sum = sum(np.cos(p) for p in neighbor_phases)
                mean_neighbor_phase = np.arctan2(sin_sum, cos_sum)
                
                # Phase gradient magnitude
                phase_diff = abs(node_phase - mean_neighbor_phase)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                phase_gradients.append(phase_diff)
        
        metrics['phase_gradient'] = np.mean(phase_gradients) if phase_gradients else 0.0
        
        # 3. Structural potential Φ_s (simplified)
        phi_s_values = []
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if neighbors:
                neighbor_stress = sum(self.G.nodes[n]['delta_nfr'] for n in neighbors)
                phi_s_values.append(neighbor_stress / len(neighbors))
        
        metrics['structural_potential'] = np.mean(phi_s_values) if phi_s_values else 0.0
        
        return metrics
    
    def select_optimization_target(self) -> int:
        """Select node with highest optimization potential"""
        # Priority: highest ΔNFR (most reorganization pressure)
        candidates = []
        for node in self.G.nodes():
            dnfr = self.G.nodes[node]['delta_nfr']
            # Consider both stress and connectivity
            degree_factor = len(list(self.G.neighbors(node))) + 1
            score = dnfr * np.sqrt(degree_factor)  # Weighted by connectivity
            candidates.append((node, score, dnfr))
        
        # Select highest scoring node
        best_node, best_score, best_dnfr = max(candidates, key=lambda x: x[1])
        return best_node
    
    def apply_optimization_step(self, target_node: int) -> Dict[str, Any]:
        """Apply structural optimization to target node"""
        old_metrics = {
            'coherence': self.compute_coherence(),
            'delta_nfr': self.G.nodes[target_node]['delta_nfr'],
            'phase': self.G.nodes[target_node]['phase'],
            'vf': self.G.nodes[target_node]['vf']
        }
        
        # === OPTIMIZATION ALGORITHM ===
        
        # 1. COHERENCE OPERATOR: Reduce internal reorganization pressure
        self.G.nodes[target_node]['delta_nfr'] *= 0.75  # 25% reduction
        
        # 2. COUPLING OPERATOR: Phase synchronization with neighbors
        neighbors = list(self.G.neighbors(target_node))
        if neighbors:
            neighbor_phases = [self.G.nodes[n]['phase'] for n in neighbors]
            
            # Compute circular mean of neighbor phases
            sin_sum = sum(np.sin(p) for p in neighbor_phases)
            cos_sum = sum(np.cos(p) for p in neighbor_phases)
            target_phase = np.arctan2(sin_sum, cos_sum)
            
            # Gradual phase adjustment (30% toward neighbors)
            current_phase = self.G.nodes[target_node]['phase']
            phase_diff = target_phase - current_phase
            
            # Handle phase wrapping
            if phase_diff > np.pi:
                phase_diff -= 2*np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2*np.pi
                
            adjustment = 0.3 * phase_diff
            new_phase = (current_phase + adjustment) % (2*np.pi)
            self.G.nodes[target_node]['phase'] = new_phase
        
        # 3. RESONANCE OPERATOR: Frequency tuning for better coupling
        if neighbors:
            neighbor_vfs = [self.G.nodes[n]['vf'] for n in neighbors]
            mean_neighbor_vf = np.mean(neighbor_vfs)
            
            # Gradual frequency adjustment (20% toward neighbors)
            current_vf = self.G.nodes[target_node]['vf']
            vf_adjustment = 0.2 * (mean_neighbor_vf - current_vf)
            self.G.nodes[target_node]['vf'] += vf_adjustment
            
            # Keep frequency in valid range [0.1, 5.0] Hz_str
            self.G.nodes[target_node]['vf'] = np.clip(self.G.nodes[target_node]['vf'], 0.1, 5.0)
        
        # 4. Small EPI perturbation (exploration vs exploitation balance)
        epi_noise = np.random.randn(4) * 0.03  # Small random walk
        self.G.nodes[target_node]['epi'] += epi_noise
        
        # === MEASURE IMPROVEMENTS ===
        new_metrics = {
            'coherence': self.compute_coherence(),
            'delta_nfr': self.G.nodes[target_node]['delta_nfr'],
            'phase': self.G.nodes[target_node]['phase'],
            'vf': self.G.nodes[target_node]['vf']
        }
        
        # Compute improvements
        coherence_improvement = new_metrics['coherence'] - old_metrics['coherence']
        dnfr_reduction = old_metrics['delta_nfr'] - new_metrics['delta_nfr']
        
        return {
            'target_node': target_node,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'coherence_improvement': coherence_improvement,
            'dnfr_reduction': dnfr_reduction,
            'success': coherence_improvement >= -0.005,  # Tolerance for noise
            'neighbors': len(neighbors)
        }
    
    def run_optimization_session(self, num_steps: int = 5) -> None:
        """Run complete optimization session"""
        print(f'🎯 Starting optimization session ({num_steps} steps)...')
        
        # Initial state assessment
        initial_coherence = self.compute_coherence()
        initial_health = self.compute_health_metrics()
        
        print(f'   📊 Initial state:')
        print(f'      Coherence C(t): {initial_coherence:.4f}')
        print(f'      Avg ΔNFR: {initial_health["avg_delta_nfr"]:.4f}')
        print(f'      Phase gradient: {initial_health["phase_gradient"]:.4f}')
        print(f'      Structural Φ_s: {initial_health["structural_potential"]:.4f}')
        print()
        
        print('🔄 === OPTIMIZATION STEPS ===')
        
        successful_steps = 0
        for step in range(num_steps):
            # Select target for optimization
            target_node = self.select_optimization_target()
            
            # Apply optimization
            result = self.apply_optimization_step(target_node)
            
            # Display step results
            status = '✅' if result['success'] else '⚠️'
            coherence_change = result['coherence_improvement']
            dnfr_change = result['dnfr_reduction']
            
            print(f'   Step {step+1:2d}: Node {target_node:2d} (N={result["neighbors"]}) {status}')
            print(f'           C(t): {result["old_metrics"]["coherence"]:.4f} → {result["new_metrics"]["coherence"]:.4f} ({coherence_change:+.4f})')
            print(f'           ΔNFR: {result["old_metrics"]["delta_nfr"]:.4f} → {result["new_metrics"]["delta_nfr"]:.4f} ({-dnfr_change:+.4f})')
            
            if result['success']:
                successful_steps += 1
            
            # Store step in history
            self.history.append(result)
        
        print()
        print('📈 === OPTIMIZATION RESULTS ===')
        
        # Final state assessment
        final_coherence = self.compute_coherence()
        final_health = self.compute_health_metrics()
        
        # Calculate total improvements
        coherence_improvement = final_coherence - initial_coherence
        dnfr_improvement = initial_health['avg_delta_nfr'] - final_health['avg_delta_nfr']
        phase_grad_improvement = initial_health['phase_gradient'] - final_health['phase_gradient']
        
        print(f'   🎯 Final metrics:')
        print(f'      Coherence C(t): {final_coherence:.4f} (Δ: {coherence_improvement:+.4f})')
        print(f'      Avg ΔNFR: {final_health["avg_delta_nfr"]:.4f} (Δ: {-dnfr_improvement:+.4f})')
        print(f'      Phase gradient: {final_health["phase_gradient"]:.4f} (Δ: {-phase_grad_improvement:+.4f})')
        print(f'      Structural Φ_s: {final_health["structural_potential"]:.4f}')
        print()
        
        success_rate = (successful_steps / num_steps) * 100
        print(f'   🏆 Performance summary:')
        print(f'      Successful optimizations: {successful_steps}/{num_steps} ({success_rate:.1f}%)')
        
        # Classify optimization outcome
        if coherence_improvement > 0.02:
            outcome = '🚀 SIGNIFICANT IMPROVEMENT'
        elif coherence_improvement > 0.005:
            outcome = '📈 MODERATE IMPROVEMENT'
        elif abs(coherence_improvement) <= 0.005:
            outcome = '⚖️ STABLE OPTIMIZATION'
        else:
            outcome = '🔍 EXPLORATORY PHASE'
        
        print(f'      Outcome: {outcome}')
        
        if dnfr_improvement > 0:
            print(f'      ✅ System stress reduced (better ΔNFR management)')
        else:
            print(f'      ⚠️ System stress increased (may indicate exploration)')
        
        print()
        self._display_physics_insights()
        
        print()
        print('🎉 === SELF-OPTIMIZING ENGINE v9.7.0 COMPLETED ===')
        print('📚 Complete theory: AGENTS.md (Single Source of Truth)')
        print('🔗 https://github.com/fermga/TNFR-Python-Engine')
    
    def _display_physics_insights(self):
        """Display physics insights and what the engine actually does"""
        print('🧮 === WHAT THE ENGINE ACTUALLY DOES ===')
        print()
        print('🔬 PHYSICS BASIS:')
        print('   • Nodal equation: ∂EPI/∂t = νf · ΔNFR(t) governs all changes')
        print('   • Optimization target: Minimize |ΔNFR| while preserving coherence')
        print('   • Universal bounds: φ≈1.618, γ≈0.577, π≈3.142, e≈2.718')
        print()
        print('⚙️ OPTIMIZATION ALGORITHM:')
        print('   1. COHERENCE OPERATOR: Reduce ΔNFR (reorganization pressure)')
        print('   2. COUPLING OPERATOR: Sync phases with neighbors (resonance)')
        print('   3. RESONANCE OPERATOR: Tune frequencies for better coupling')
        print('   4. EPI PERTURBATION: Small structural exploration')
        print()
        print('📊 WHAT GETS OPTIMIZED:')
        print('   • Node selection: Highest ΔNFR × √connectivity (smart targeting)')
        print('   • Phase alignment: Reduces local desynchronization')
        print('   • Frequency tuning: Improves resonant coupling potential')
        print('   • Stress reduction: Lowers internal reorganization pressure')
        print()
        print('🎯 SUCCESS CRITERIA:')
        print('   • Coherence preservation or improvement')
        print('   • ΔNFR reduction (less internal stress)')
        print('   • Phase gradient minimization (better sync)')
        print('   • Structural potential stabilization')

def main():
    """Execute the optimized TNFR Self-Optimizing Engine"""
    print('🧮 === TNFR SELF-OPTIMIZING ENGINE v9.7.0 (OPTIMIZED) ===')
    print('🚀 Mathematical Auto-Improvement with Clear Operational Visibility')
    print('📐 Universal Tetrahedral Correspondence: φ↔Φ_s, γ↔|∇φ|, π↔K_φ, e↔ξ_C')
    print()
    
    # Set reproducible seed for consistent results
    np.random.seed(42)
    
    # Create and run the engine
    engine = TNFROptimizedEngine(num_nodes=12, connectivity=0.4)
    engine.run_optimization_session(num_steps=6)

if __name__ == "__main__":
    main()