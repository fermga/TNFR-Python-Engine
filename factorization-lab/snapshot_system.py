"""
TNFR Partition Invariant Snapshot Schema

Comprehensive state capture system for partition verification processes.
Captures nodal deltas, phase relationships, and structural metrics at key 
transition points to enable detailed analysis and reproducibility.

Mathematical Foundation:
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t) 
- State evolution: EPI(t+Δt) = EPI(t) + ∫[νf(τ)·ΔNFR(τ)]dτ
- Invariants: Structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
- Partition dynamics: Coherence preservation across boundaries

Design Principles:
1. Complete reproducibility of verification trajectories
2. Minimal storage overhead with delta compression
3. Fast snapshot creation/restoration for debugging
4. Rich telemetry for analysis and optimization
"""

import json
import sqlite3
import time
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import pickle
import gzip


@dataclass
class NodalState:
    """Individual node state at snapshot point."""
    
    node_id: int
    epi_vector: List[float]  # Current EPI configuration
    structural_frequency: float  # νf value
    phase: float  # Current phase φ
    theta: float  # Alternative phase parameter
    dnfr_magnitude: float  # |ΔNFR| current value
    coupling_count: int  # Number of active couplings
    last_operator: Optional[str]  # Most recent operator applied
    operator_timestamp: float  # When last operator was applied
    stability_index: float  # Local stability measure


@dataclass 
class PartitionState:
    """State of a partition boundary region."""
    
    partition_id: str
    boundary_nodes: List[int]  # Nodes on partition boundary
    internal_nodes: List[int]  # Nodes inside partition
    coherence_ratio: float  # Partition internal coherence
    phase_gradient: float  # |∇φ| across boundary
    curvature_delta: float  # K_φ difference across boundary
    coupling_strength: float  # Average coupling across boundary
    fragmentation_risk: float  # Probability of boundary collapse
    operator_sequence: List[str]  # Operators applied to this partition


@dataclass
class StructuralFieldSnapshot:
    """Snapshot of the structural field tetrad."""
    
    phi_s_global: float  # Structural potential Φ_s
    phi_s_distribution: List[float]  # Per-node Φ_s values
    phase_gradient_field: List[float]  # |∇φ| per node
    curvature_field: List[float]  # K_φ per node  
    coherence_length: float  # ξ_C global value
    coherence_field: List[float]  # Local coherence per node
    
    # Derived invariants
    energy_density: float  # ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ²
    topological_charge: float  # 𝒬 = |∇φ|·J_φ - K_φ·J_ΔNFR
    symmetry_breaking: float  # 𝒮 = (|∇φ|² - K_φ²) + (J_φ² - J_ΔNFR²)


@dataclass
class NetworkTopologySnapshot:
    """Network structure and connectivity state."""
    
    node_count: int
    edge_count: int
    adjacency_matrix: List[List[int]]  # Connectivity matrix
    coupling_matrix: List[List[float]]  # Coupling strengths
    phase_matrix: List[List[float]]  # Phase relationships
    
    # Graph properties
    clustering_coefficient: float
    characteristic_path_length: float
    algebraic_connectivity: float  # Second smallest eigenvalue of Laplacian
    spectral_gap: float  # Gap between first two eigenvalues
    
    # TNFR-specific properties
    resonant_components: int  # Number of phase-synchronized components
    fragmented_regions: int  # Number of disconnected coherent regions
    critical_coupling_ratio: float  # Fraction of couplings near critical threshold


@dataclass
class VerificationSnapshot:
    """Complete system state snapshot during verification."""
    
    # Metadata
    snapshot_id: str
    timestamp: float
    verification_stage: str  # "initialization", "partitioning", "verification", "completion"
    modulus_n: int
    candidate_factor: Optional[int]
    partition_strategy: str
    
    # State components
    nodal_states: List[NodalState]
    partition_states: List[PartitionState]
    structural_fields: StructuralFieldSnapshot
    network_topology: NetworkTopologySnapshot
    
    # Verification metrics
    overall_coherence: float  # C(t)
    sense_index: float  # Si
    verification_confidence: float
    dnfr_budget_remaining: float
    convergence_iterations: int
    
    # Performance metrics
    elapsed_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Hash for integrity verification
    state_hash: str


class SnapshotCompressor:
    """Efficient compression for snapshot storage."""
    
    @staticmethod
    def compress_snapshot(snapshot: VerificationSnapshot) -> bytes:
        """Compress snapshot using gzip + pickle."""
        
        # Convert to dictionary for serialization
        snapshot_dict = asdict(snapshot)
        
        # Pickle and compress
        pickled_data = pickle.dumps(snapshot_dict, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = gzip.compress(pickled_data, compresslevel=6)
        
        return compressed_data
    
    @staticmethod
    def decompress_snapshot(compressed_data: bytes) -> VerificationSnapshot:
        """Decompress snapshot from storage."""
        
        # Decompress and unpickle
        pickled_data = gzip.decompress(compressed_data)
        snapshot_dict = pickle.loads(pickled_data)
        
        # Reconstruct dataclass
        # Note: This requires manual reconstruction due to nested dataclasses
        return SnapshotCompressor._reconstruct_snapshot(snapshot_dict)
    
    @staticmethod
    def _reconstruct_snapshot(data: Dict) -> VerificationSnapshot:
        """Reconstruct VerificationSnapshot from dictionary."""
        
        # Reconstruct nested dataclasses
        nodal_states = [NodalState(**ns) for ns in data['nodal_states']]
        partition_states = [PartitionState(**ps) for ps in data['partition_states']]
        structural_fields = StructuralFieldSnapshot(**data['structural_fields'])
        network_topology = NetworkTopologySnapshot(**data['network_topology'])
        
        # Reconstruct main snapshot
        return VerificationSnapshot(
            snapshot_id=data['snapshot_id'],
            timestamp=data['timestamp'],
            verification_stage=data['verification_stage'],
            modulus_n=data['modulus_n'],
            candidate_factor=data['candidate_factor'],
            partition_strategy=data['partition_strategy'],
            nodal_states=nodal_states,
            partition_states=partition_states,
            structural_fields=structural_fields,
            network_topology=network_topology,
            overall_coherence=data['overall_coherence'],
            sense_index=data['sense_index'],
            verification_confidence=data['verification_confidence'],
            dnfr_budget_remaining=data['dnfr_budget_remaining'],
            convergence_iterations=data['convergence_iterations'],
            elapsed_time_ms=data['elapsed_time_ms'],
            memory_usage_mb=data['memory_usage_mb'],
            cpu_usage_percent=data['cpu_usage_percent'],
            state_hash=data['state_hash']
        )


class PartitionSnapshotManager:
    """Manages partition state snapshots during verification."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize snapshot manager with database."""
        
        self.db_path = db_path or Path("partition_snapshots.db")
        self._init_database()
        self._snapshot_cache = {}  # In-memory cache for recent snapshots
        
    def _init_database(self):
        """Initialize snapshot database schema."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    verification_stage TEXT NOT NULL,
                    modulus_n INTEGER NOT NULL,
                    candidate_factor INTEGER,
                    partition_strategy TEXT NOT NULL,
                    overall_coherence REAL NOT NULL,
                    sense_index REAL NOT NULL,
                    verification_confidence REAL NOT NULL,
                    elapsed_time_ms REAL NOT NULL,
                    compressed_data BLOB NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_modulus 
                ON snapshots(modulus_n)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_stage
                ON snapshots(verification_stage)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON snapshots(timestamp)
            """)
    
    def create_snapshot(self, 
                       verification_stage: str,
                       modulus_n: int,
                       candidate_factor: Optional[int],
                       partition_strategy: str,
                       nodal_states: List[NodalState],
                       partition_states: List[PartitionState],
                       structural_fields: StructuralFieldSnapshot,
                       network_topology: NetworkTopologySnapshot,
                       performance_metrics: Dict[str, float]) -> str:
        """Create and store a verification snapshot."""
        
        # Generate unique snapshot ID
        snapshot_id = self._generate_snapshot_id(modulus_n, verification_stage)
        
        # Create snapshot object
        snapshot = VerificationSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            verification_stage=verification_stage,
            modulus_n=modulus_n,
            candidate_factor=candidate_factor,
            partition_strategy=partition_strategy,
            nodal_states=nodal_states,
            partition_states=partition_states,
            structural_fields=structural_fields,
            network_topology=network_topology,
            overall_coherence=performance_metrics.get('coherence', 0.0),
            sense_index=performance_metrics.get('sense_index', 0.0),
            verification_confidence=performance_metrics.get('confidence', 0.0),
            dnfr_budget_remaining=performance_metrics.get('dnfr_budget', 100.0),
            convergence_iterations=int(performance_metrics.get('iterations', 0)),
            elapsed_time_ms=performance_metrics.get('elapsed_ms', 0.0),
            memory_usage_mb=performance_metrics.get('memory_mb', 0.0),
            cpu_usage_percent=performance_metrics.get('cpu_percent', 0.0),
            state_hash=""  # Will be computed below
        )
        
        # Compute state hash for integrity
        snapshot.state_hash = self._compute_state_hash(snapshot)
        
        # Compress and store
        compressed_data = SnapshotCompressor.compress_snapshot(snapshot)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO snapshots (
                    snapshot_id, timestamp, verification_stage, modulus_n,
                    candidate_factor, partition_strategy, overall_coherence,
                    sense_index, verification_confidence, elapsed_time_ms,
                    compressed_data, state_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, snapshot.timestamp, verification_stage, modulus_n,
                candidate_factor, partition_strategy, snapshot.overall_coherence,
                snapshot.sense_index, snapshot.verification_confidence,
                snapshot.elapsed_time_ms, compressed_data, snapshot.state_hash
            ))
        
        # Cache recent snapshot
        self._snapshot_cache[snapshot_id] = snapshot
        
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> Optional[VerificationSnapshot]:
        """Load snapshot from database or cache."""
        
        # Check cache first
        if snapshot_id in self._snapshot_cache:
            return self._snapshot_cache[snapshot_id]
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT compressed_data, state_hash FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Decompress snapshot
            compressed_data = row['compressed_data']
            expected_hash = row['state_hash']
            
            snapshot = SnapshotCompressor.decompress_snapshot(compressed_data)
            
            # Verify integrity
            computed_hash = self._compute_state_hash(snapshot)
            if computed_hash != expected_hash:
                raise ValueError(f"Snapshot {snapshot_id} integrity check failed")
            
            # Cache and return
            self._snapshot_cache[snapshot_id] = snapshot
            return snapshot
    
    def list_snapshots(self, 
                      modulus_n: Optional[int] = None,
                      verification_stage: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """List available snapshots with optional filtering."""
        
        query = """
            SELECT snapshot_id, timestamp, verification_stage, modulus_n,
                   candidate_factor, partition_strategy, overall_coherence,
                   sense_index, verification_confidence, elapsed_time_ms
            FROM snapshots
            WHERE 1=1
        """
        params = []
        
        if modulus_n is not None:
            query += " AND modulus_n = ?"
            params.append(modulus_n)
        
        if verification_stage is not None:
            query += " AND verification_stage = ?"
            params.append(verification_stage)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_verification_trajectory(self, modulus_n: int, 
                                  partition_strategy: str) -> List[str]:
        """Get sequence of snapshot IDs for a complete verification."""
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT snapshot_id FROM snapshots 
                WHERE modulus_n = ? AND partition_strategy = ?
                ORDER BY timestamp ASC
            """, (modulus_n, partition_strategy)).fetchall()
            
            return [row[0] for row in rows]
    
    def analyze_trajectory_coherence(self, snapshot_ids: List[str]) -> Dict[str, Any]:
        """Analyze coherence evolution across verification trajectory."""
        
        snapshots = [self.load_snapshot(sid) for sid in snapshot_ids]
        snapshots = [s for s in snapshots if s is not None]
        
        if not snapshots:
            return {"error": "No valid snapshots found"}
        
        # Extract coherence trajectory
        coherence_values = [s.overall_coherence for s in snapshots]
        sense_index_values = [s.sense_index for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]
        
        # Compute trajectory statistics
        coherence_trend = np.polyfit(range(len(coherence_values)), coherence_values, 1)[0]
        coherence_stability = np.std(coherence_values)
        
        return {
            "trajectory_length": len(snapshots),
            "coherence_trajectory": coherence_values,
            "sense_index_trajectory": sense_index_values,
            "timestamps": timestamps,
            "coherence_trend": coherence_trend,
            "coherence_stability": coherence_stability,
            "final_coherence": coherence_values[-1] if coherence_values else 0,
            "coherence_range": (min(coherence_values), max(coherence_values)) if coherence_values else (0, 0)
        }
    
    def export_trajectory_data(self, snapshot_ids: List[str], 
                              export_path: Path) -> bool:
        """Export trajectory data for external analysis."""
        
        snapshots = [self.load_snapshot(sid) for sid in snapshot_ids]
        snapshots = [s for s in snapshots if s is not None]
        
        if not snapshots:
            return False
        
        # Prepare export data
        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "trajectory_length": len(snapshots),
                "modulus_n": snapshots[0].modulus_n,
                "partition_strategy": snapshots[0].partition_strategy
            },
            "snapshots": [asdict(snapshot) for snapshot in snapshots]
        }
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return True
    
    def _generate_snapshot_id(self, modulus_n: int, stage: str) -> str:
        """Generate unique snapshot identifier."""
        
        timestamp = time.time()
        unique_string = f"{modulus_n}_{stage}_{timestamp}"
        hash_object = hashlib.md5(unique_string.encode())
        return f"snap_{hash_object.hexdigest()[:12]}"
    
    def _compute_state_hash(self, snapshot: VerificationSnapshot) -> str:
        """Compute hash for snapshot integrity verification."""
        
        # Create deterministic representation
        hash_data = {
            "modulus_n": snapshot.modulus_n,
            "stage": snapshot.verification_stage,
            "coherence": round(snapshot.overall_coherence, 6),
            "node_count": len(snapshot.nodal_states),
            "partition_count": len(snapshot.partition_states)
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def cleanup_old_snapshots(self, max_age_hours: float = 168):
        """Remove snapshots older than specified age (default 1 week)."""
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM snapshots WHERE timestamp < ?",
                (cutoff_time,)
            )
            
            return result.rowcount


# Utility functions for snapshot creation

def create_mock_nodal_state(node_id: int, coherent: bool = True) -> NodalState:
    """Create mock nodal state for testing."""
    
    return NodalState(
        node_id=node_id,
        epi_vector=[0.5, 0.3, 0.7] if coherent else [0.1, 0.9, 0.2],
        structural_frequency=1.2 if coherent else 0.3,
        phase=0.1 * node_id,
        theta=0.2 * node_id,
        dnfr_magnitude=0.1 if coherent else 0.8,
        coupling_count=3 if coherent else 1,
        last_operator="coherence" if coherent else "dissonance",
        operator_timestamp=time.time(),
        stability_index=0.8 if coherent else 0.2
    )


def create_mock_structural_fields(node_count: int) -> StructuralFieldSnapshot:
    """Create mock structural field snapshot for testing."""
    
    return StructuralFieldSnapshot(
        phi_s_global=1.2,
        phi_s_distribution=[0.5 + 0.1 * i for i in range(node_count)],
        phase_gradient_field=[0.1 + 0.05 * i for i in range(node_count)],
        curvature_field=[0.2 + 0.03 * i for i in range(node_count)],
        coherence_length=5.0,
        coherence_field=[0.8 - 0.1 * (i % 3) for i in range(node_count)],
        energy_density=2.5,
        topological_charge=0.3,
        symmetry_breaking=0.1
    )


def create_mock_network_topology(node_count: int) -> NetworkTopologySnapshot:
    """Create mock network topology snapshot for testing."""
    
    # Simple ring topology for demonstration
    adjacency = [[0] * node_count for _ in range(node_count)]
    coupling = [[0.0] * node_count for _ in range(node_count)]
    phase = [[0.0] * node_count for _ in range(node_count)]
    
    edge_count = 0
    for i in range(node_count):
        next_node = (i + 1) % node_count
        adjacency[i][next_node] = 1
        adjacency[next_node][i] = 1
        coupling[i][next_node] = 0.8
        coupling[next_node][i] = 0.8
        phase[i][next_node] = 0.1
        phase[next_node][i] = -0.1
        edge_count += 1
    
    return NetworkTopologySnapshot(
        node_count=node_count,
        edge_count=edge_count,
        adjacency_matrix=adjacency,
        coupling_matrix=coupling,
        phase_matrix=phase,
        clustering_coefficient=0.0,  # Ring has no triangles
        characteristic_path_length=node_count / 4,
        algebraic_connectivity=0.5,
        spectral_gap=0.3,
        resonant_components=1,
        fragmented_regions=0,
        critical_coupling_ratio=0.1
    )


if __name__ == "__main__":
    """Demo script showing snapshot system usage."""
    
    # Create snapshot manager
    manager = PartitionSnapshotManager(Path("demo_snapshots.db"))
    
    # Create mock verification states
    node_count = 10
    nodal_states = [create_mock_nodal_state(i, coherent=(i % 2 == 0)) for i in range(node_count)]
    partition_states = [
        PartitionState(
            partition_id="part_1",
            boundary_nodes=[2, 3, 7],
            internal_nodes=[0, 1, 4, 5, 6, 8, 9],
            coherence_ratio=0.85,
            phase_gradient=0.15,
            curvature_delta=0.25,
            coupling_strength=0.7,
            fragmentation_risk=0.1,
            operator_sequence=["emission", "coupling", "coherence"]
        )
    ]
    structural_fields = create_mock_structural_fields(node_count)
    network_topology = create_mock_network_topology(node_count)
    
    performance_metrics = {
        "coherence": 0.82,
        "sense_index": 0.75,
        "confidence": 0.88,
        "elapsed_ms": 1250.5,
        "memory_mb": 45.2,
        "cpu_percent": 23.1
    }
    
    # Create snapshots for verification trajectory
    modulus_n = 77
    partition_strategy = "spectral_paley"
    
    print("Creating verification trajectory snapshots...")
    
    stages = ["initialization", "partitioning", "verification", "completion"]
    snapshot_ids = []
    
    for stage in stages:
        # Simulate progression
        performance_metrics["coherence"] += 0.02
        performance_metrics["elapsed_ms"] += 500
        
        snapshot_id = manager.create_snapshot(
            verification_stage=stage,
            modulus_n=modulus_n,
            candidate_factor=7 if stage == "completion" else None,
            partition_strategy=partition_strategy,
            nodal_states=nodal_states,
            partition_states=partition_states,
            structural_fields=structural_fields,
            network_topology=network_topology,
            performance_metrics=performance_metrics
        )
        
        snapshot_ids.append(snapshot_id)
        print(f"  Created snapshot: {snapshot_id} ({stage})")
    
    # Demonstrate trajectory analysis
    print("\nAnalyzing verification trajectory...")
    analysis = manager.analyze_trajectory_coherence(snapshot_ids)
    
    print(f"  Trajectory length: {analysis['trajectory_length']}")
    print(f"  Coherence trend: {analysis['coherence_trend']:.4f}")
    print(f"  Final coherence: {analysis['final_coherence']:.3f}")
    print(f"  Coherence range: {analysis['coherence_range']}")
    
    # List all snapshots
    print("\nAvailable snapshots:")
    snapshots_list = manager.list_snapshots(modulus_n=modulus_n)
    for snapshot_info in snapshots_list:
        print(f"  {snapshot_info['snapshot_id']}: {snapshot_info['verification_stage']} "
              f"(C={snapshot_info['overall_coherence']:.3f})")
    
    # Export trajectory
    export_path = Path("verification_trajectory_77.json")
    success = manager.export_trajectory_data(snapshot_ids, export_path)
    if success:
        print(f"\nTrajectory exported to: {export_path}")
    
    print("\n✅ Partition invariant snapshot system demo completed!")