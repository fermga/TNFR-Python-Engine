"""Script to split metrics.py into modular files.

This script divides src/tnfr/operators/metrics.py into 4 specialized modules
plus a facade for backward compatibility.
"""

# Read the original file
with open("src/tnfr/operators/metrics.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Define line ranges for each module (1-indexed, inclusive)
ranges = {
    "metrics_basic.py": (73, 572),          # emission, reception, coherence, dissonance
    "metrics_network.py": (573, 1093),      # coupling, resonance, silence + helpers
    "metrics_structural.py": (1094, 2146),  # expansion, contraction, self_org, mutation, transition, recursivity
}

# Extract core imports and utilities (lines 1-72)
header_lines = lines[0:72]

# Helper functions that need to be in network module (before silence_metrics)
network_helpers_start = 861  # _compute_epi_variance
network_helpers_end = 999    # _estimate_time_to_collapse

# Helper for transition (before transition_metrics)
transition_helper_line = 2058  # _detect_regime_from_state

# Create each module
for module_name, (start, end) in ranges.items():
    module_lines = []
    
    # Add header
    module_lines.append(f'"""Operator metrics: {module_name.replace("metrics_", "").replace(".py", "")} operators."""\n')
    module_lines.append("\n")
    module_lines.append("from __future__ import annotations\n")
    module_lines.append("\n")
    module_lines.append("from typing import Any\n")
    module_lines.append("\n")
    module_lines.append("from .metrics_core import (\n")
    module_lines.append("    get_node_attr as _get_node_attr,\n")
    module_lines.append("    ALIAS_D2EPI,\n")
    module_lines.append("    ALIAS_DNFR,\n")
    module_lines.append("    ALIAS_EPI,\n")
    module_lines.append("    ALIAS_THETA,\n")
    module_lines.append("    ALIAS_VF,\n")
    module_lines.append("    HAS_EMISSION_TIMESTAMP_ALIAS as _HAS_EMISSION_TIMESTAMP_ALIAS,\n")
    module_lines.append("    EMISSION_TIMESTAMP_TUPLE as _ALIAS_EMISSION_TIMESTAMP_TUPLE,\n")
    module_lines.append(")\n")
    module_lines.append("from ..alias import get_attr_str\n")
    module_lines.append("\n")
    module_lines.append("\n")
    
    # Add network helpers if this is metrics_network.py
    if module_name == "metrics_network.py":
        module_lines.extend(lines[network_helpers_start-1:network_helpers_end])
        module_lines.append("\n\n")
    
    # Add transition helper if this is metrics_structural.py
    if module_name == "metrics_structural.py":
        # Add helper before transition_metrics
        module_lines.extend(lines[transition_helper_line-1:transition_helper_line+32])  # _detect_regime_from_state
        module_lines.append("\n\n")
    
    # Add main function content
    module_lines.extend(lines[start-1:end])
    
    # Remove U6 imports from structural module (already in main facade)
    if module_name == "metrics_structural.py":
        # Remove last ~16 lines (U6 fallback imports)
        module_lines = [l for l in module_lines if "metrics_u6" not in l and "measure_tau_relax" not in l and "measure_nonlinear" not in l and "compute_bifurcation_index" not in l]
    
    # Write module
    with open(f"src/tnfr/operators/{module_name}", "w", encoding="utf-8") as f:
        f.writelines(module_lines)
    
    print(f"✅ Created {module_name}")

# Create facade metrics.py
facade_lines = []
facade_lines.append('"""Operator metrics facade for backward compatibility."""\n')
facade_lines.append("\n")
facade_lines.append("from .metrics_basic import (\n")
facade_lines.append("    emission_metrics,\n")
facade_lines.append("    reception_metrics,\n")
facade_lines.append("    coherence_metrics,\n")
facade_lines.append("    dissonance_metrics,\n")
facade_lines.append(")\n")
facade_lines.append("from .metrics_network import (\n")
facade_lines.append("    coupling_metrics,\n")
facade_lines.append("    resonance_metrics,\n")
facade_lines.append("    silence_metrics,\n")
facade_lines.append(")\n")
facade_lines.append("from .metrics_structural import (\n")
facade_lines.append("    expansion_metrics,\n")
facade_lines.append("    contraction_metrics,\n")
facade_lines.append("    self_organization_metrics,\n")
facade_lines.append("    mutation_metrics,\n")
facade_lines.append("    transition_metrics,\n")
facade_lines.append("    recursivity_metrics,\n")
facade_lines.append(")\n")
facade_lines.append("\n")
facade_lines.append("# U6 experimental telemetry\n")
facade_lines.append("try:\n")
facade_lines.append("    from .metrics_u6 import (\n")
facade_lines.append("        measure_tau_relax_observed,\n")
facade_lines.append("        measure_nonlinear_accumulation,\n")
facade_lines.append("        compute_bifurcation_index,\n")
facade_lines.append("    )\n")
facade_lines.append("except Exception:\n")
facade_lines.append("    from typing import Any\n")
facade_lines.append("    def measure_tau_relax_observed(*args: Any, **kwargs: Any) -> dict[str, Any]:\n")
facade_lines.append('        return {"error": "metrics_u6 missing", "metric_type": "u6_relaxation_time"}\n')
facade_lines.append("    def measure_nonlinear_accumulation(*args: Any, **kwargs: Any) -> dict[str, Any]:\n")
facade_lines.append('        return {"error": "metrics_u6 missing", "metric_type": "u6_nonlinear_accumulation"}\n')
facade_lines.append("    def compute_bifurcation_index(*args: Any, **kwargs: Any) -> dict[str, Any]:\n")
facade_lines.append('        return {"error": "metrics_u6 missing", "metric_type": "u6_bifurcation_index"}\n')
facade_lines.append("\n")
facade_lines.append("__all__ = [\n")
facade_lines.append('    "emission_metrics",\n')
facade_lines.append('    "reception_metrics",\n')
facade_lines.append('    "coherence_metrics",\n')
facade_lines.append('    "dissonance_metrics",\n')
facade_lines.append('    "coupling_metrics",\n')
facade_lines.append('    "resonance_metrics",\n')
facade_lines.append('    "silence_metrics",\n')
facade_lines.append('    "expansion_metrics",\n')
facade_lines.append('    "contraction_metrics",\n')
facade_lines.append('    "self_organization_metrics",\n')
facade_lines.append('    "mutation_metrics",\n')
facade_lines.append('    "transition_metrics",\n')
facade_lines.append('    "recursivity_metrics",\n')
facade_lines.append('    "measure_tau_relax_observed",\n')
facade_lines.append('    "measure_nonlinear_accumulation",\n')
facade_lines.append('    "compute_bifurcation_index",\n')
facade_lines.append("]\n")

# Rename original to .old
import os
os.rename("src/tnfr/operators/metrics.py", "src/tnfr/operators/metrics.py.old")

# Write facade
with open("src/tnfr/operators/metrics.py", "w", encoding="utf-8") as f:
    f.writelines(facade_lines)

print("✅ Created metrics.py facade")
print("✅ Renamed original to metrics.py.old")
print("\n✨ Split complete! Run tests to verify.")
