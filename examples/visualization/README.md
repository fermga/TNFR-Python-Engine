# TNFR Sequence Visualization

Advanced visualization tools for TNFR operator sequences with comprehensive structural analysis.

## Features

This module provides four main visualization types:

### 1. Sequence Flow Diagrams
Visual representation of operator sequences with compatibility-colored transitions:
- **Green arrows**: Excellent transitions (optimal structural coherence)
- **Blue arrows**: Good transitions (acceptable structural integrity)
- **Orange arrows**: Caution transitions (context-dependent, careful validation needed)
- **Red arrows**: Avoid transitions (violates structural coherence)

Node colors indicate operator categories (initiator, stabilizer, transformer, amplifier, organizer).

### 2. Health Dashboards
Comprehensive health metrics visualization with:
- **Radar chart**: All 7 health metrics (coherence, balance, sustainability, efficiency, frequency, completeness, smoothness)
- **Bar chart**: Current metrics vs. ideal benchmarks
- **Overall health gauge**: Color-coded health indicator (excellent/good/fair/needs improvement)

### 3. Pattern Analysis
Highlights structural components within detected patterns:
- Operator categorization with color coding
- Pattern signature visualization
- Component role identification

### 4. Frequency Timelines
Shows structural frequency (νf) evolution through the sequence:
- Frequency levels (high/medium/zero) displayed as a timeline
- Valid vs. invalid transitions highlighted
- Harmonic zones indicated

## Installation

```bash
# Install TNFR with visualization support
pip install tnfr[viz]

# Or install matplotlib separately
pip install matplotlib
```

## Quick Start

```python
from tnfr.visualization import SequenceVisualizer
from tnfr.operators.grammar import validate_sequence_with_health

# Define your sequence
sequence = ["emission", "reception", "coherence", "silence"]

# Validate and get health metrics
result = validate_sequence_with_health(sequence)

# Create visualizer
visualizer = SequenceVisualizer()

# Generate all visualizations
fig1, ax1 = visualizer.plot_sequence_flow(sequence, health_metrics=result.health_metrics)
fig2, axes2 = visualizer.plot_health_dashboard(result.health_metrics)
fig3, ax3 = visualizer.plot_pattern_analysis(sequence, result.health_metrics.dominant_pattern)
fig4, ax4 = visualizer.plot_frequency_timeline(sequence)

# Save visualizations
fig1.savefig("flow.png")
fig2.savefig("dashboard.png")
fig3.savefig("pattern.png")
fig4.savefig("timeline.png")
```

## Command-Line Tools

### Interactive Sequence Explorer

```bash
# Analyze a single sequence
python tools/sequence_explorer.py --sequence emission reception coherence silence

# Interactive mode
python tools/sequence_explorer.py --interactive

# Compare multiple sequences (from files)
python tools/sequence_explorer.py --compare seq1.txt seq2.txt seq3.txt

# Specify output directory
python tools/sequence_explorer.py --sequence emission reception coherence silence --output /path/to/output
```

## Examples

See `examples/visualization/` for comprehensive demos:

- **basic_plotting_demo.py**: Demonstrates all 4 visualization types with multiple sequence patterns
- **interactive_explorer_demo.py**: Shows programmatic usage of InteractiveSequenceExplorer

Run examples:
```bash
cd examples/visualization
python basic_plotting_demo.py
python interactive_explorer_demo.py
```

## API Reference

### SequenceVisualizer

Main visualization class.

#### `__init__(figsize=(12, 8), dpi=100)`
Initialize visualizer with custom figure size and resolution.

#### `plot_sequence_flow(sequence, health_metrics=None, save_path=None)`
Generate sequence flow diagram with compatibility-colored transitions.

**Parameters:**
- `sequence`: List of operator names (canonical form)
- `health_metrics`: Optional SequenceHealthMetrics to display
- `save_path`: Optional path to save figure

**Returns:** `(Figure, Axes)` tuple

#### `plot_health_dashboard(health_metrics, save_path=None)`
Generate comprehensive health metrics dashboard.

**Parameters:**
- `health_metrics`: SequenceHealthMetrics object
- `save_path`: Optional path to save figure

**Returns:** `(Figure, ndarray)` - figure and array of 3 axes (radar, bars, gauge)

#### `plot_pattern_analysis(sequence, pattern, save_path=None)`
Visualize pattern components with category highlighting.

**Parameters:**
- `sequence`: List of operator names
- `pattern`: Pattern name (e.g., "activation", "therapeutic")
- `save_path`: Optional path to save figure

**Returns:** `(Figure, Axes)` tuple

#### `plot_frequency_timeline(sequence, save_path=None)`
Show structural frequency evolution timeline.

**Parameters:**
- `sequence`: List of operator names
- `save_path`: Optional path to save figure

**Returns:** `(Figure, Axes)` tuple

### InteractiveSequenceExplorer

Command-line interface for sequence exploration.

#### `explore_sequence(sequence, output_dir="/tmp", show_plots=False)`
Comprehensive analysis of a single sequence with all visualizations.

#### `compare_sequences(sequences, labels=None, output_dir="/tmp")`
Compare multiple sequences with ranking and metrics table.

#### `interactive_mode()`
Launch interactive REPL for exploratory analysis.

## Color Scheme

### Compatibility Levels
- **Green (#2ecc71)**: EXCELLENT - optimal transitions
- **Blue (#3498db)**: GOOD - acceptable transitions
- **Orange (#f39c12)**: CAUTION - context-dependent transitions
- **Red (#e74c3c)**: AVOID - incompatible transitions

### Operator Categories
- **Purple (#9b59b6)**: Initiators (emission)
- **Green (#2ecc71)**: Stabilizers (coherence, silence)
- **Orange (#e67e22)**: Transformers (dissonance, mutation, transition)
- **Red (#e74c3c)**: Amplifiers (resonance, coupling)
- **Teal (#1abc9c)**: Organizers (self_organization, recursivity)

### Frequency Levels
- **Red (#e74c3c)**: High frequency (high energy)
- **Blue (#3498db)**: Medium frequency (moderate)
- **Gray (#95a5a6)**: Zero frequency (paused)

## Testing

Run the visualization test suite:
```bash
pytest tests/visualization/test_sequence_plotting.py -v
```

The test suite includes:
- Initialization tests
- Plot generation tests for all 4 visualization types
- Save functionality tests
- Edge case handling (empty sequences, long sequences, invalid operators)
- Performance tests

## Architecture

The visualization module integrates with:
- `tnfr.operators.health_analyzer`: Health metrics computation
- `tnfr.operators.grammar`: Sequence validation and pattern detection
- `tnfr.validation.compatibility`: Compatibility level determination
- `tnfr.operators.grammar.STRUCTURAL_FREQUENCIES`: Frequency mapping

All visualizations follow TNFR canonical principles:
- Respect operator semantics
- Display structural properties (νf, phase, compatibility)
- Maintain coherence with TNFR theory
- Use canonical operator names

## Contributing

When extending visualizations:

1. **Follow TNFR principles**: All visualizations must respect canonical grammar and structural theory
2. **Add tests**: Include comprehensive tests for new visualization types
3. **Document thoroughly**: Update this README and add docstrings
4. **Use existing utilities**: Leverage existing color schemes and helper functions
5. **Handle edge cases**: Empty sequences, long sequences, invalid operators

## License

MIT License - see LICENSE.md for details.
