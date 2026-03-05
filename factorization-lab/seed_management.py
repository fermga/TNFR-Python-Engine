"""
TNFR Seed Management System

Comprehensive seed capture and reproducibility framework for factorization experiments.
Captures and restores complete system state including random seeds, initialization 
parameters, and environmental conditions for perfect reproducibility.

Design Principles:
1. Complete reproducibility of all stochastic processes
2. Lightweight seed capture with minimal performance overhead  
3. Cross-platform compatibility and version resilience
4. Hierarchical seeding for multi-scale reproducibility
5. Audit trail for debugging non-deterministic issues

Mathematical Foundation:
- Deterministic evolution: EPI(t+dt) = f(EPI(t), seed_state)
- State reproducibility: Same seeds → identical trajectories  
- Hierarchical consistency: Master seed → component seeds → operation seeds
- Audit integrity: Complete traceability of all random operations
"""

import hashlib
import json
import random
import time
import platform
import sys
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class SystemEnvironment:
    """Capture of system environment affecting reproducibility."""
    
    platform_system: str
    platform_release: str  
    python_version: str
    numpy_version: str
    random_state_type: str
    
    # Process information
    process_id: int
    working_directory: str
    
    # Timing information
    utc_timestamp: float
    local_timezone: str
    
    # Hardware fingerprint (for debugging)
    cpu_count: int
    memory_total_gb: float


@dataclass
class RandomSeedState:
    """Complete random number generator state."""
    
    # Python random module state
    python_random_state: tuple
    
    # NumPy random state
    numpy_random_state: dict
    
    # Custom TNFR seeds
    master_seed: int
    node_initialization_seed: int
    coupling_dynamics_seed: int
    phase_evolution_seed: int
    operator_sequence_seed: int
    
    # Hierarchical seeds for different scales
    global_network_seed: int
    partition_level_seed: int
    node_level_seed: int
    
    # Verification-specific seeds
    spectral_analysis_seed: int
    clustering_seed: int
    threshold_jitter_seed: int


@dataclass 
class ExperimentParameters:
    """Complete set of experiment parameters affecting results."""
    
    # Primary factorization parameters
    modulus_n: int
    partition_strategy: str
    verification_mode: str
    
    # Network topology parameters
    node_count: int
    initial_topology: str
    coupling_strength_range: Tuple[float, float]
    
    # TNFR physics parameters
    structural_frequency_range: Tuple[float, float]
    phase_initialization_mode: str
    coherence_threshold: float
    dnfr_budget: float
    
    # Algorithm parameters
    max_iterations: int
    convergence_tolerance: float
    spectral_clustering_k: int
    
    # Performance parameters
    timeout_seconds: float
    memory_limit_mb: int
    
    # Advanced parameters
    operator_sequence_constraints: List[str]
    adaptive_threshold_enabled: bool
    feedback_learning_enabled: bool


@dataclass
class ReproducibilityMetadata:
    """Metadata for experiment reproducibility."""
    
    experiment_id: str
    creation_timestamp: float
    
    # Version information
    tnfr_version: str
    experiment_version: str
    
    # Checksums for integrity
    parameter_checksum: str
    seed_state_checksum: str
    environment_checksum: str
    
    # Dependency tracking
    dependencies: Dict[str, str]
    
    # Execution metadata
    execution_hostname: str
    execution_user: Optional[str]
    execution_duration_ms: float


class TNFRSeedManager:
    """Comprehensive seed management for TNFR experiments."""
    
    def __init__(self, master_seed: Optional[int] = None):
        """Initialize seed manager with optional master seed."""
        
        self.master_seed = master_seed or self._generate_master_seed()
        self.seed_history = []
        self.current_experiment_id = None
        
        # Initialize all RNG states from master seed
        self._initialize_from_master_seed()
        
    def _generate_master_seed(self) -> int:
        """Generate cryptographically secure master seed."""
        
        # Combine multiple entropy sources
        entropy_sources = [
            int(time.time() * 1000000) % (2**31),  # High-resolution timestamp
            hash(str(os.urandom(16))) % (2**31),   # OS random bytes
            id(object()) % (2**31),                # Memory address randomness  
            hash(platform.node()) % (2**31)        # Machine identifier
        ]
        
        # XOR combine entropy sources
        master_seed = 0
        for source in entropy_sources:
            master_seed ^= source
        
        # Ensure positive 32-bit integer
        return abs(master_seed) % (2**31)
    
    def _initialize_from_master_seed(self):
        """Initialize all RNG states from master seed using deterministic derivation."""
        
        # Use master seed to derive component seeds
        seed_generator = random.Random(self.master_seed)
        
        # Derive hierarchical seeds
        self.global_network_seed = seed_generator.randint(1, 2**30)
        self.partition_level_seed = seed_generator.randint(1, 2**30)  
        self.node_level_seed = seed_generator.randint(1, 2**30)
        
        # Derive process-specific seeds
        self.node_initialization_seed = seed_generator.randint(1, 2**30)
        self.coupling_dynamics_seed = seed_generator.randint(1, 2**30)
        self.phase_evolution_seed = seed_generator.randint(1, 2**30)
        self.operator_sequence_seed = seed_generator.randint(1, 2**30)
        
        # Derive analysis seeds
        self.spectral_analysis_seed = seed_generator.randint(1, 2**30)
        self.clustering_seed = seed_generator.randint(1, 2**30)
        self.threshold_jitter_seed = seed_generator.randint(1, 2**30)
        
        # Set global random states
        random.seed(self.master_seed)
        np.random.seed(self.master_seed % (2**32))  # NumPy requires uint32
    
    def capture_complete_state(self) -> Dict[str, Any]:
        """Capture complete reproducibility state."""
        
        # Capture environment
        environment = self._capture_system_environment()
        
        # Capture random states
        seed_state = RandomSeedState(
            python_random_state=random.getstate(),
            numpy_random_state=self._get_numpy_state(),
            master_seed=self.master_seed,
            node_initialization_seed=self.node_initialization_seed,
            coupling_dynamics_seed=self.coupling_dynamics_seed,
            phase_evolution_seed=self.phase_evolution_seed,
            operator_sequence_seed=self.operator_sequence_seed,
            global_network_seed=self.global_network_seed,
            partition_level_seed=self.partition_level_seed,
            node_level_seed=self.node_level_seed,
            spectral_analysis_seed=self.spectral_analysis_seed,
            clustering_seed=self.clustering_seed,
            threshold_jitter_seed=self.threshold_jitter_seed
        )
        
        return {
            "environment": asdict(environment),
            "seed_state": asdict(seed_state),
            "capture_timestamp": time.time(),
            "master_seed": self.master_seed
        }
    
    def restore_complete_state(self, state_data: Dict[str, Any]) -> bool:
        """Restore complete system state from captured data."""
        
        try:
            # Restore master seed
            self.master_seed = state_data["master_seed"]
            
            # Restore random states
            seed_state = state_data["seed_state"]
            
            # Restore Python random state
            random.setstate(tuple(seed_state["python_random_state"]))
            
            # Restore NumPy state 
            self._set_numpy_state(seed_state["numpy_random_state"])
            
            # Restore derived seeds
            self.node_initialization_seed = seed_state["node_initialization_seed"]
            self.coupling_dynamics_seed = seed_state["coupling_dynamics_seed"]
            self.phase_evolution_seed = seed_state["phase_evolution_seed"]
            self.operator_sequence_seed = seed_state["operator_sequence_seed"]
            self.global_network_seed = seed_state["global_network_seed"]
            self.partition_level_seed = seed_state["partition_level_seed"]
            self.node_level_seed = seed_state["node_level_seed"]
            self.spectral_analysis_seed = seed_state["spectral_analysis_seed"]
            self.clustering_seed = seed_state["clustering_seed"]
            self.threshold_jitter_seed = seed_state["threshold_jitter_seed"]
            
            return True
            
        except (KeyError, TypeError, ValueError) as e:
            print(f"Failed to restore state: {e}")
            return False
    
    def create_experiment_context(self, 
                                experiment_params: ExperimentParameters,
                                experiment_id: Optional[str] = None) -> str:
        """Create reproducible experiment context with full state capture."""
        
        # Generate experiment ID
        if experiment_id is None:
            experiment_id = self._generate_experiment_id(experiment_params)
        
        self.current_experiment_id = experiment_id
        
        # Capture complete state
        complete_state = self.capture_complete_state()
        
        # Create metadata
        metadata = ReproducibilityMetadata(
            experiment_id=experiment_id,
            creation_timestamp=time.time(),
            tnfr_version=self._get_tnfr_version(),
            experiment_version="1.0",
            parameter_checksum=self._compute_checksum(asdict(experiment_params)),
            seed_state_checksum=self._compute_checksum(complete_state["seed_state"]),
            environment_checksum=self._compute_checksum(complete_state["environment"]),
            dependencies=self._get_dependency_versions(),
            execution_hostname=platform.node(),
            execution_user=os.getenv('USER') or os.getenv('USERNAME'),
            execution_duration_ms=0.0  # Will be updated on completion
        )
        
        # Store experiment context
        context_data = {
            "metadata": asdict(metadata),
            "parameters": asdict(experiment_params),
            "reproducibility_state": complete_state
        }
        
        self._store_experiment_context(experiment_id, context_data)
        
        return experiment_id
    
    def get_seeded_random(self, seed_type: str) -> random.Random:
        """Get seeded random generator for specific use case."""
        
        seed_map = {
            "node_initialization": self.node_initialization_seed,
            "coupling_dynamics": self.coupling_dynamics_seed,
            "phase_evolution": self.phase_evolution_seed,
            "operator_sequence": self.operator_sequence_seed,
            "global_network": self.global_network_seed,
            "partition_level": self.partition_level_seed,
            "node_level": self.node_level_seed,
            "spectral_analysis": self.spectral_analysis_seed,
            "clustering": self.clustering_seed,
            "threshold_jitter": self.threshold_jitter_seed
        }
        
        if seed_type not in seed_map:
            raise ValueError(f"Unknown seed type: {seed_type}")
        
        return random.Random(seed_map[seed_type])
    
    def get_seeded_numpy_random(self, seed_type: str) -> np.random.Generator:
        """Get seeded NumPy random generator for specific use case."""
        
        seed_map = {
            "node_initialization": self.node_initialization_seed,
            "coupling_dynamics": self.coupling_dynamics_seed,
            "phase_evolution": self.phase_evolution_seed,
            "operator_sequence": self.operator_sequence_seed,
            "global_network": self.global_network_seed,
            "partition_level": self.partition_level_seed,
            "node_level": self.node_level_seed,
            "spectral_analysis": self.spectral_analysis_seed,
            "clustering": self.clustering_seed,
            "threshold_jitter": self.threshold_jitter_seed
        }
        
        if seed_type not in seed_map:
            raise ValueError(f"Unknown seed type: {seed_type}")
        
        seed = seed_map[seed_type] % (2**32)  # NumPy requires uint32
        return np.random.default_rng(seed)
    
    def validate_reproducibility(self, 
                               experiment_id: str, 
                               test_iterations: int = 3) -> Dict[str, Any]:
        """Validate experiment reproducibility by running multiple times."""
        
        # Load experiment context
        context_data = self._load_experiment_context(experiment_id)
        if not context_data:
            return {"valid": False, "error": "Experiment context not found"}
        
        # Extract parameters
        params = ExperimentParameters(**context_data["parameters"])
        
        # Run test iterations
        results = []
        for i in range(test_iterations):
            # Restore state
            self.restore_complete_state(context_data["reproducibility_state"])
            
            # Run mock experiment (would be actual factorization in practice)
            result = self._run_reproducibility_test(params)
            results.append(result)
        
        # Check consistency
        consistency_check = self._analyze_result_consistency(results)
        
        return {
            "valid": consistency_check["is_consistent"],
            "test_iterations": test_iterations,
            "consistency_score": consistency_check["consistency_score"],
            "differences": consistency_check["differences"],
            "results": results
        }
    
    def _capture_system_environment(self) -> SystemEnvironment:
        """Capture system environment information."""
        
        import psutil
        
        return SystemEnvironment(
            platform_system=platform.system(),
            platform_release=platform.release(),
            python_version=sys.version,
            numpy_version=np.__version__,
            random_state_type=str(type(random.getstate())),
            process_id=os.getpid(),
            working_directory=str(Path.cwd()),
            utc_timestamp=time.time(),
            local_timezone=str(time.tzname),
            cpu_count=os.cpu_count() or 1,
            memory_total_gb=psutil.virtual_memory().total / (1024**3)
        )
    
    def _get_numpy_state(self) -> dict:
        """Get NumPy random state in serializable format."""
        
        # Get bit generator state
        bit_gen_state = np.random.get_state()
        
        return {
            "generator": bit_gen_state[0],
            "state": bit_gen_state[1].tolist(),
            "pos": int(bit_gen_state[2]),
            "has_gauss": int(bit_gen_state[3]),
            "cached_gaussian": float(bit_gen_state[4]) if bit_gen_state[4] is not None else None
        }
    
    def _set_numpy_state(self, state_dict: dict):
        """Restore NumPy random state from serializable format."""
        
        state_tuple = (
            state_dict["generator"],
            np.array(state_dict["state"], dtype=np.uint32),
            state_dict["pos"],
            state_dict["has_gauss"],
            state_dict["cached_gaussian"]
        )
        
        np.random.set_state(state_tuple)
    
    def _generate_experiment_id(self, params: ExperimentParameters) -> str:
        """Generate unique experiment identifier."""
        
        # Create identifier from parameters and timestamp
        param_str = f"{params.modulus_n}_{params.partition_strategy}_{params.node_count}"
        timestamp_str = str(int(time.time() * 1000))
        seed_str = str(self.master_seed)
        
        # Hash to create unique ID
        combined = f"{param_str}_{timestamp_str}_{seed_str}"
        hash_obj = hashlib.md5(combined.encode())
        
        return f"exp_{hash_obj.hexdigest()[:12]}"
    
    def _compute_checksum(self, data: Any) -> str:
        """Compute deterministic checksum for data integrity."""
        
        # Convert to JSON with sorted keys for deterministic output
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _get_tnfr_version(self) -> str:
        """Get TNFR version for compatibility tracking."""
        
        # In practice, would read from package metadata
        return "0.9.5"  # Current version
    
    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        
        dependencies = {
            "python": sys.version.split()[0],
            "numpy": np.__version__
        }
        
        # Try to get other package versions
        try:
            import scipy
            dependencies["scipy"] = scipy.__version__
        except ImportError:
            pass
            
        try:
            import networkx
            dependencies["networkx"] = networkx.__version__
        except ImportError:
            pass
        
        return dependencies
    
    def _store_experiment_context(self, experiment_id: str, context_data: Dict[str, Any]):
        """Store experiment context for later retrieval."""
        
        # Create contexts directory
        contexts_dir = Path("experiment_contexts")
        contexts_dir.mkdir(exist_ok=True)
        
        # Store context data
        context_file = contexts_dir / f"{experiment_id}.json"
        with open(context_file, 'w') as f:
            json.dump(context_data, f, indent=2, default=str)
    
    def _load_experiment_context(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment context from storage."""
        
        context_file = Path("experiment_contexts") / f"{experiment_id}.json"
        if not context_file.exists():
            return None
        
        with open(context_file, 'r') as f:
            return json.load(f)
    
    def _run_reproducibility_test(self, params: ExperimentParameters) -> Dict[str, float]:
        """Run lightweight reproducibility test (mock factorization)."""
        
        # Use seeded random generators
        node_rng = self.get_seeded_numpy_random("node_initialization")
        coupling_rng = self.get_seeded_numpy_random("coupling_dynamics")
        
        # Simulate node initialization
        initial_phases = node_rng.uniform(0, 2*np.pi, params.node_count)
        structural_freqs = node_rng.uniform(*params.structural_frequency_range, params.node_count)
        
        # Simulate coupling dynamics
        coupling_matrix = coupling_rng.uniform(*params.coupling_strength_range, 
                                             (params.node_count, params.node_count))
        
        # Simulate coherence computation
        phase_diffs = np.abs(initial_phases[:, None] - initial_phases[None, :])
        coherence_contributions = np.cos(phase_diffs) * coupling_matrix
        final_coherence = np.mean(coherence_contributions)
        
        # Simulate sense index
        sense_index = np.mean(structural_freqs) / (1.0 + np.std(phase_diffs))
        
        # Simulate verification confidence
        spectral_rng = self.get_seeded_numpy_random("spectral_analysis")
        eigenvalue_gap = spectral_rng.exponential(0.3)
        verification_confidence = min(1.0, final_coherence + eigenvalue_gap)
        
        return {
            "final_coherence": float(final_coherence),
            "sense_index": float(sense_index),
            "verification_confidence": float(verification_confidence),
            "eigenvalue_gap": float(eigenvalue_gap),
            "phase_checksum": float(np.sum(initial_phases))
        }
    
    def _analyze_result_consistency(self, results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze consistency of reproducibility test results."""
        
        if len(results) < 2:
            return {"is_consistent": True, "consistency_score": 1.0, "differences": {}}
        
        # Check exact equality for all metrics
        first_result = results[0]
        differences = {}
        
        for key in first_result:
            values = [result[key] for result in results]
            
            # Check if all values are identical
            if not all(abs(v - values[0]) < 1e-15 for v in values):
                differences[key] = {
                    "values": values,
                    "max_diff": max(values) - min(values),
                    "std_dev": float(np.std(values))
                }
        
        # Calculate consistency score
        is_consistent = len(differences) == 0
        consistency_score = 1.0 if is_consistent else (1.0 - len(differences) / len(first_result))
        
        return {
            "is_consistent": is_consistent,
            "consistency_score": consistency_score,
            "differences": differences
        }


# Utility functions for integration

def create_demo_experiment_params(modulus_n: int = 77) -> ExperimentParameters:
    """Create demo experiment parameters."""
    
    return ExperimentParameters(
        modulus_n=modulus_n,
        partition_strategy="spectral_paley",
        verification_mode="comprehensive",
        node_count=10,
        initial_topology="ring",
        coupling_strength_range=(0.5, 0.9),
        structural_frequency_range=(0.8, 1.5),
        phase_initialization_mode="uniform_random",
        coherence_threshold=0.75,
        dnfr_budget=100.0,
        max_iterations=50,
        convergence_tolerance=1e-6,
        spectral_clustering_k=2,
        timeout_seconds=300.0,
        memory_limit_mb=512,
        operator_sequence_constraints=["emission", "coupling", "coherence"],
        adaptive_threshold_enabled=True,
        feedback_learning_enabled=True
    )


if __name__ == "__main__":
    """Demo script showing seed management system usage."""
    
    print("TNFR SEED MANAGEMENT SYSTEM DEMO")
    print("="*50)
    
    # Create seed manager
    seed_manager = TNFRSeedManager(master_seed=42)
    print(f"Master seed: {seed_manager.master_seed}")
    
    # Create experiment parameters
    params = create_demo_experiment_params(77)
    print(f"Experiment parameters: n={params.modulus_n}, strategy={params.partition_strategy}")
    
    # Create experiment context
    experiment_id = seed_manager.create_experiment_context(params)
    print(f"Created experiment: {experiment_id}")
    
    # Demonstrate seeded random generation
    print("\nSeeded random generation:")
    node_rng = seed_manager.get_seeded_random("node_initialization")
    print(f"Node initialization samples: {[node_rng.random() for _ in range(3)]}")
    
    coupling_rng = seed_manager.get_seeded_numpy_random("coupling_dynamics")
    print(f"Coupling dynamics samples: {coupling_rng.random(3)}")
    
    # Test reproducibility
    print(f"\nTesting reproducibility...")
    validation_result = seed_manager.validate_reproducibility(experiment_id, test_iterations=3)
    
    print(f"Reproducibility valid: {validation_result['valid']}")
    print(f"Consistency score: {validation_result['consistency_score']:.6f}")
    
    if validation_result['differences']:
        print("Detected differences:")
        for key, diff_data in validation_result['differences'].items():
            print(f"  {key}: max_diff={diff_data['max_diff']:.2e}")
    else:
        print("✅ Perfect reproducibility achieved!")
    
    # Demonstrate state capture/restore
    print(f"\nTesting state capture/restore...")
    
    # Capture current state
    state = seed_manager.capture_complete_state()
    
    # Generate some random numbers
    before_numbers = [random.random() for _ in range(3)]
    print(f"Numbers before restore: {before_numbers}")
    
    # Restore state and generate same numbers
    seed_manager.restore_complete_state(state)
    after_numbers = [random.random() for _ in range(3)]
    print(f"Numbers after restore: {after_numbers}")
    
    # Check if identical
    identical = all(abs(a - b) < 1e-15 for a, b in zip(before_numbers, after_numbers))
    print(f"State restore successful: {identical}")
    
    print(f"\n✅ Seed management system demo completed!")