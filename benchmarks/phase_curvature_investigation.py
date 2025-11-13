#!/usr/bin/env python3
"""
Phase Curvature K_φ Confinement Mechanism Investigation

Research pipeline to validate K_φ as canonical field through:
1. Critical threshold validation (fragmentation predictor)
2. Confinement zone mapping (strong-like interaction regime)
3. Asymptotic freedom testing (scale-dependent variance)
4. Mutation candidate prediction (ZHIR optimization)
5. Complementary safety criteria (local hotspot detection)
6. Cross-domain validation

Based on §10-11 evidence: |K_φ| > 4.88 threshold, weak correlation
but threshold behavior.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CurvatureExperiment:
    """Single K_φ threshold experiment result."""
    topology: str
    nodes: int
    edges: int
    k_phi_max: float
    k_phi_mean: float
    k_phi_var: float
    coherence: float
    fragmented: bool
    intensity: float
    sequence: str
    seed: int


class CurvatureInvestigator:
    """Phase curvature K_φ confinement mechanism research pipeline."""
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize K_φ investigator.
        
        Parameters
        ----------
        output_dir : str
            Directory for experiment results storage
        """
        # Import here to avoid module-level import errors
        from src.tnfr.physics.fields import compute_phase_curvature
        from benchmarks.benchmark_utils import (
            create_tnfr_topology,
            initialize_tnfr_nodes,
            generate_grammar_valid_sequence
        )
        
        # Store imports as class attributes
        self.compute_phase_curvature = compute_phase_curvature
        self.create_tnfr_topology = create_tnfr_topology
        self.initialize_tnfr_nodes = initialize_tnfr_nodes
        self.generate_grammar_valid_sequence = generate_grammar_valid_sequence
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Research parameters based on §11 evidence
        self.candidate_threshold = 4.88  # From fragmentation analysis
        self.threshold_range = (4.0, 6.0)  # Fine-grained sweep range
        self.critical_intensity = 2.015    # From §11 I_c value
        
        # Experiment configurations
        self.topologies = [
            'scale_free',
            'tree',
            'grid',
            'ring',
            'ws'  # Watts-Strogatz (small world)
        ]
        
        self.node_sizes = [20, 50, 100]
        self.sequences = [
            'OZ_heavy',      # High instability
            'balanced',      # Moderate dynamics
            'RA_dominated'   # Low dynamics (resonance-heavy)
        ]
    
    def run_threshold_validation(
        self, n_experiments: int = 200
    ) -> List[CurvatureExperiment]:
        """Task 1: Critical Threshold Validation.
        
        Test if |K_φ| ≈ 4.88 acts as universal fragmentation threshold
        across topologies with ≥90% classification accuracy.
        
        Parameters
        ----------
        n_experiments : int
            Number of experiments per configuration
            
        Returns
        -------
        List[CurvatureExperiment]
            Experiment results for ROC analysis
        """
        print("🔬 Task 1: Critical Threshold Validation")
        print(f"Target: |K_φ| > {self.candidate_threshold} → fragmentation")
        print(f"Testing range: {self.threshold_range}")
        print()
        
        experiments = []
        total_runs = len(self.topologies) * len(self.node_sizes) * len(self.sequences) * n_experiments
        run_count = 0
        
        for topology in self.topologies:
            for n_nodes in self.node_sizes:
                for sequence in self.sequences:
                    print(f"Testing {topology} n={n_nodes} seq={sequence}")
                    
                    for i in range(n_experiments):
                        run_count += 1
                        seed = random.randint(1000, 99999)
                        
                        try:
                            # Create graph and apply sequence
                            G = self.create_tnfr_topology(
                                topology, n_nodes, seed=seed
                            )
                            self.initialize_tnfr_nodes(G, seed=seed)
                            
                            # Apply sequence with intensity sweep around I_c
                            noise = np.random.uniform(-0.5, 0.5)
                            intensity = self.critical_intensity + noise
                            
                            # Generate and apply sequence
                            ops = self.generate_grammar_valid_sequence(
                                sequence, intensity=intensity
                            )
                            # Apply operators to graph (simplified)
                            for op in ops:
                                try:
                                    op.apply(G)
                                except Exception:
                                    pass  # Skip failed operations
                            
                            # Compute K_φ and coherence metrics
                            k_phi = self.compute_phase_curvature(G)
                            k_phi_values = list(k_phi.values())
                            
                            if not k_phi_values:
                                continue
                                
                            k_phi_max = max(abs(k) for k in k_phi_values)
                            k_phi_mean = np.mean([abs(k) for k in k_phi_values])
                            k_phi_var = np.var(k_phi_values)
                            
                            # Coherence assessment (simplified)
                            coherence = self._assess_coherence(G)
                            fragmented = coherence < 0.3
                            
                            experiment = CurvatureExperiment(
                                topology=topology.name,
                                nodes=len(G.nodes()),
                                edges=len(G.edges()),
                                k_phi_max=k_phi_max,
                                k_phi_mean=k_phi_mean,
                                k_phi_var=k_phi_var,
                                coherence=coherence,
                                fragmented=fragmented,
                                intensity=intensity,
                                sequence=sequence.name,
                                seed=seed
                            )
                            
                            experiments.append(experiment)
                            
                            if run_count % 20 == 0:
                                pct = 100 * run_count / total_runs
                                print(f"  Progress: {run_count}/{total_runs} " +
                                      f"({pct:.1f}%)")
                        
                        except Exception as e:
                            print(f"  Error in experiment {run_count}: {e}")
                            continue
        
        # Save raw results
        results_file = self.output_dir / "k_phi_threshold_validation.jsonl"
        with open(results_file, 'w') as f:
            for exp in experiments:
                f.write(json.dumps(exp.__dict__) + '\n')
        
        print(f"✅ Completed {len(experiments)} experiments")
        print(f"Results saved to: {results_file}")
        
        # ROC analysis
        self._analyze_threshold_roc(experiments)
        
        return experiments
    
    def _analyze_threshold_roc(self, experiments: List[CurvatureExperiment]) -> None:
        """Perform ROC analysis to optimize K_φ threshold."""
        if not experiments:
            return
            
        print("\n📊 ROC Analysis for K_φ Fragmentation Threshold")
        
        # Extract data for ROC
        k_phi_max_values = [exp.k_phi_max for exp in experiments]
        fragmentation_labels = [int(exp.fragmented) for exp in experiments]
        
        if len(set(fragmentation_labels)) < 2:
            print("❌ No fragmentation variation - cannot compute ROC")
            return
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(fragmentation_labels, k_phi_max_values)
        auc_score = roc_auc_score(fragmentation_labels, k_phi_max_values)
        
        # Find optimal threshold (max Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        print(f"🎯 AUC Score: {auc_score:.4f}")
        print(f"🎯 Optimal Threshold: |K_φ| > {optimal_threshold:.3f}")
        print(f"   - True Positive Rate: {optimal_tpr:.3f}")
        print(f"   - False Positive Rate: {optimal_fpr:.3f}")
        print(f"   - Accuracy: {optimal_tpr - optimal_fpr:.3f}")
        
        # Compare with candidate threshold
        candidate_predictions = [k > self.candidate_threshold for k in k_phi_max_values]
        candidate_accuracy = np.mean([pred == label for pred, label in 
                                    zip(candidate_predictions, fragmentation_labels)])
        
        print(f"📋 Candidate threshold |K_φ| > {self.candidate_threshold}:")
        print(f"   - Accuracy: {candidate_accuracy:.3f}")
        print(f"   - Target: ≥0.90 for canonical promotion")
        
        # Topology breakdown
        print("\n🗺️ Topology-wise Performance:")
        topology_stats = {}
        for topology in set(exp.topology for exp in experiments):
            topo_exps = [exp for exp in experiments if exp.topology == topology]
            topo_k_phi = [exp.k_phi_max for exp in topo_exps]
            topo_frag = [int(exp.fragmented) for exp in topo_exps]
            
            if len(set(topo_frag)) >= 2:  # Need variation
                topo_auc = roc_auc_score(topo_frag, topo_k_phi)
                topo_acc = np.mean([k > self.candidate_threshold == f 
                                  for k, f in zip(topo_k_phi, topo_frag)])
                topology_stats[topology] = {'auc': topo_auc, 'accuracy': topo_acc}
                print(f"   {topology}: AUC={topo_auc:.3f}, Acc={topo_acc:.3f}")
            else:
                print(f"   {topology}: No fragmentation variation")
        
        # Save ROC results
        roc_results = {
            'overall_auc': float(auc_score),
            'optimal_threshold': float(optimal_threshold),
            'optimal_tpr': float(optimal_tpr),
            'optimal_fpr': float(optimal_fpr),
            'candidate_threshold': self.candidate_threshold,
            'candidate_accuracy': float(candidate_accuracy),
            'topology_stats': topology_stats,
            'n_experiments': len(experiments)
        }
        
        roc_file = self.output_dir / "k_phi_roc_analysis.json"
        with open(roc_file, 'w') as f:
            json.dump(roc_results, f, indent=2)
        
        print(f"\n💾 ROC analysis saved to: {roc_file}")
    
    def identify_confinement_zones(self, G: Any, k_phi_threshold: float = 4.5) -> Tuple[Any, int, Dict]:
        """Task 2: Confinement Zone Mapping.
        
        Identify subgraphs where |K_φ| > threshold as confinement regions.
        
        Parameters
        ----------
        G : networkx.Graph
            Network to analyze
        k_phi_threshold : float
            Threshold for confinement zone identification
            
        Returns
        -------
        subgraph : networkx.Graph
            Subgraph of confined nodes
        n_components : int
            Number of connected components in confinement zones
        zone_stats : dict
            Statistics about confinement zones
        """
        k_phi = compute_phase_curvature(G)
        
        # Identify nodes with high curvature
        confined_nodes = [n for n, k in k_phi.items() if abs(k) > k_phi_threshold]
        
        if not confined_nodes:
            return G.subgraph([]), 0, {'total_nodes': 0, 'coverage': 0.0}
        
        # Create confinement subgraph
        subgraph = G.subgraph(confined_nodes).copy()
        n_components = nx.number_connected_components(subgraph)
        
        # Compute zone statistics
        zone_stats = {
            'total_nodes': len(confined_nodes),
            'coverage': len(confined_nodes) / len(G.nodes()),
            'n_components': n_components,
            'max_component_size': max(len(c) for c in nx.connected_components(subgraph)) if confined_nodes else 0,
            'avg_k_phi': np.mean([abs(k_phi[n]) for n in confined_nodes]),
            'threshold_used': k_phi_threshold
        }
        
        return subgraph, n_components, zone_stats
    
    def measure_scale_dependent_curvature(self, G: Any, scales: List[int] = None) -> Dict[int, float]:
        """Task 3: Asymptotic Freedom Investigation.
        
        Test if |K_φ| variance decreases at larger scales (asymptotic freedom).
        
        Parameters
        ----------
        G : networkx.Graph
            Network to analyze
        scales : List[int], optional
            Neighborhood radii to test. Default: [1, 2, 3, 5, 10]
            
        Returns
        -------
        Dict[int, float]
            K_φ variance at each scale
        """
        if scales is None:
            scales = [1, 2, 3, 5, 10]
        
        k_phi_base = compute_phase_curvature(G)
        scale_variances = {}
        
        for r in scales:
            k_phi_coarse = {}
            
            for node in G.nodes():
                # Get r-hop ego network
                try:
                    ego_nodes = nx.ego_graph(G, node, radius=r).nodes()
                    ego_k_phi = [k_phi_base[n] for n in ego_nodes if n in k_phi_base]
                    
                    if ego_k_phi:
                        k_phi_coarse[node] = np.mean(ego_k_phi)
                    else:
                        k_phi_coarse[node] = 0.0
                except:
                    k_phi_coarse[node] = 0.0
            
            # Compute variance at this scale
            if k_phi_coarse:
                scale_variances[r] = np.var(list(k_phi_coarse.values()))
            else:
                scale_variances[r] = 0.0
        
        return scale_variances
    
    def predict_mutation_candidates(self, G: Any, top_k: int = 5) -> List[Tuple[Any, float]]:
        """Task 4: Mutation Candidate Prediction.
        
        Identify nodes with highest |K_φ| as optimal ZHIR targets.
        
        Parameters
        ----------
        G : networkx.Graph
            Network to analyze
        top_k : int
            Number of top candidates to return
            
        Returns
        -------
        List[Tuple[Any, float]]
            List of (node, |K_φ|) for top candidates
        """
        k_phi = compute_phase_curvature(G)
        
        # Sort by absolute curvature
        sorted_candidates = sorted(
            k_phi.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def _assess_coherence(self, G: Any) -> float:
        """Simple coherence assessment for fragmentation detection."""
        try:
            # Use connected component analysis as proxy
            largest_component_size = max(len(c) for c in nx.connected_components(G))
            coherence = largest_component_size / len(G.nodes())
            return coherence
        except:
            return 0.0


def main():
    """Run K_φ confinement mechanism investigation."""
    print("🌊 Phase Curvature K_φ Confinement Mechanism Investigation")
    print("=" * 60)
    print()
    
    investigator = CurvatureInvestigator()
    
    # Task 1: Critical Threshold Validation
    experiments = investigator.run_threshold_validation(n_experiments=50)
    
    print("\n" + "=" * 60)
    print("✅ Investigation Phase 1 Complete")
    print(f"📊 {len(experiments)} experiments conducted")
    print("📈 Check results/k_phi_threshold_validation.jsonl for detailed data")
    print("📈 Check results/k_phi_roc_analysis.json for threshold optimization")


if __name__ == "__main__":
    main()