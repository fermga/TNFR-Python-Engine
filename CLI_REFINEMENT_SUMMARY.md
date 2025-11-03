# CLI Refinement Summary

## Overview
This document summarizes the CLI refinement work that adds new subcommands with dotted notation (`math.run`, `epi.validate`), YAML preset templates, and comprehensive test coverage.

## Changes Made

### 1. New CLI Subcommands

#### `tnfr math.run`
- **Purpose**: Run simulations with mathematical dynamics engine validation always enabled
- **Features**:
  - Validates TNFR structural invariants on Hilbert space
  - Checks norm preservation (‖ψ‖ = 1)
  - Validates coherence bounds (C ≥ C_min)
  - Verifies structural frequency positivity (νf ≥ 0)
  - Provides detailed math engine summary output
- **Usage**:
  ```bash
  tnfr math.run --nodes 24 --steps 50
  tnfr math.run --math-dimension 32 --steps 100
  tnfr math.run --preset resonant_bootstrap
  ```

#### `tnfr epi.validate`
- **Purpose**: Validate EPI structural integrity and coherence preservation
- **Features**:
  - Checks coherence preservation (W_mean ≥ 0)
  - Validates structural frequency positivity (νf ≥ 0 for all nodes)
  - Verifies phase synchrony in couplings (phase differences bounded)
  - Configurable numerical tolerance
  - Returns exit code 0 on success, 1 on validation failure
- **Usage**:
  ```bash
  tnfr epi.validate --preset resonant_bootstrap
  tnfr epi.validate --nodes 48 --topology complete
  tnfr epi.validate --tolerance 1e-8
  ```

### 2. YAML Preset Templates

Created `presets/` directory with 5 reproducible preset templates:

1. **resonant_bootstrap.yaml**
   - Fundamental resonant initialization sequence
   - Operators: AL, EN, IL, RA, VAL, UM, SHA
   - Demonstrates basic coherence emergence

2. **contained_mutation.yaml**
   - Controlled mutations within THOL blocks
   - Operators: AL, EN, OZ, ZHIR, IL, RA, SHA
   - Shows mutation containment strategy

3. **coupling_exploration.yaml**
   - Network coupling dynamics exploration
   - Operators: AL, EN, IL, VAL, UM, OZ, NAV, RA, SHA
   - Investigates different coupling configurations

4. **fractal_expand.yaml**
   - Operational fractality via expansion
   - Operators: THOL, VAL, UM, NUL, RA
   - Demonstrates recursive expansion while preserving identity

5. **fractal_contract.yaml**
   - Operational fractality via contraction
   - Operators: THOL, NUL, UM, SHA, RA
   - Shows consolidation dynamics with structural preservation

Each preset includes:
- Metadata with name, description, and operator list
- Topology configuration (type, nodes, seed)
- Dynamics configuration (dt, integrator, steps)
- Structural sequence following TNFR canonical grammar
- Comprehensive documentation in `presets/README.md`

### 3. Enhanced CLI Help

**Main Help**:
- Now includes "Common examples" section
- Lists all subcommands with brief descriptions
- Provides quick-start examples for all major use cases

**Subcommand Help**:
- Each subcommand has detailed help with `--help`
- Includes specific usage examples
- Documents all available options
- References preset availability where applicable

Example:
```bash
$ tnfr --help
# Shows common examples for all subcommands

$ tnfr math.run --help
# Shows detailed math.run usage with examples

$ tnfr epi.validate --help
# Shows detailed epi.validate usage with examples
```

### 4. Test Coverage

Added **29 comprehensive CLI tests** in `tests/integration/test_cli_refined_subcommands.py`:

**Test Classes**:
- `TestMathRunSubcommand` (6 tests)
  - Help message validation
  - Basic execution
  - Preset handling
  - Custom Hilbert dimensions
  - Coherence spectrum configuration
  - Error handling

- `TestEpiValidateSubcommand` (7 tests)
  - Help message validation
  - Basic execution
  - Preset validation
  - Custom tolerance
  - Coherence checks
  - Frequency checks
  - Error handling

- `TestEnhancedHelp` (3 tests)
  - Main help examples
  - Subcommand listing
  - Preset references

- `TestYAMLPresets` (4 tests)
  - File existence
  - README presence
  - Valid YAML structure
  - Required metadata fields

- `TestCLIStdoutStderrCapture` (5 tests)
  - Version output
  - Error messages
  - Math.run output
  - Epi.validate output
  - Help output for all subcommands

- `TestCLIIntegrationScenarios` (4 tests)
  - YAML sequence execution
  - History export
  - Different topologies
  - Metrics generation

**Test Results**: 29/29 passing (100%)

### 5. Code Quality Improvements

- Proper EPI attribute access using `get_attr` with alias resolution
- Added `TWO_PI` constant for better readability
- Used `setattr` for safer attribute assignment
- Removed redundant assertions
- Fixed trailing whitespace in YAML files
- Passed security scan (CodeQL) with 0 alerts

## TNFR Compliance

All changes respect TNFR canonical invariants:

1. **EPI coherence preservation**: Validated by `epi.validate` command
2. **Operator closure**: All presets use canonical structural operators
3. **Operational fractality**: THOL blocks preserve functional identity
4. **Structural units**: Frequencies expressed in Hz_str
5. **Phase verification**: Couplings checked for phase synchrony
6. **Math engine validation**: Hilbert space projections verify invariants

## File Changes

### New Files
- `presets/README.md` (4004 bytes)
- `presets/resonant_bootstrap.yaml` (1020 bytes)
- `presets/contained_mutation.yaml` (775 bytes)
- `presets/coupling_exploration.yaml` (844 bytes)
- `presets/fractal_expand.yaml` (813 bytes)
- `presets/fractal_contract.yaml` (815 bytes)
- `tests/integration/test_cli_refined_subcommands.py` (13537 bytes)

### Modified Files
- `src/tnfr/cli/__init__.py`
  - Added new subcommand imports
  - Enhanced main help epilog
  - Registered new subparsers

- `src/tnfr/cli/arguments.py`
  - Added `_add_math_run_parser()`
  - Added `_add_epi_validate_parser()`
  - Exported new parser functions

- `src/tnfr/cli/execution.py`
  - Added `cmd_math_run()`
  - Added `cmd_epi_validate()`
  - Added EPI alias resolution
  - Added TWO_PI constant
  - Fixed EPI attribute access in `_log_math_engine_summary()`

## Usage Examples

### Running with Math Engine Validation
```bash
# Basic math.run
tnfr math.run --nodes 24 --steps 50

# With custom Hilbert dimension
tnfr math.run --math-dimension 32 --nodes 24 --steps 100

# With custom coherence spectrum
tnfr math.run --math-coherence-spectrum 1.0 0.8 0.6 --steps 50

# With preset
tnfr math.run --preset resonant_bootstrap --steps 100
```

### Validating EPI Integrity
```bash
# Basic validation
tnfr epi.validate --nodes 24 --steps 50

# With preset
tnfr epi.validate --preset coupling_exploration

# With custom tolerance
tnfr epi.validate --tolerance 1e-8 --nodes 48

# Different topology
tnfr epi.validate --topology complete --nodes 12
```

### Using YAML Presets
```bash
# Execute YAML preset with sequence command
tnfr sequence --sequence-file presets/resonant_bootstrap.yaml

# Override preset parameters
tnfr sequence --sequence-file presets/fractal_expand.yaml --nodes 48

# Validate a YAML preset
tnfr epi.validate --sequence-file presets/coupling_exploration.yaml
```

### Combined Workflows
```bash
# Run with math validation and export history
tnfr math.run --preset resonant_bootstrap --save-history results.json

# Validate and export metrics
tnfr epi.validate --preset contained_mutation --steps 200

# Use custom YAML with math engine
tnfr math.run --sequence-file presets/fractal_expand.yaml --math-dimension 48
```

## Testing

### Run New CLI Tests
```bash
# All new tests
pytest tests/integration/test_cli_refined_subcommands.py -v

# Specific test class
pytest tests/integration/test_cli_refined_subcommands.py::TestMathRunSubcommand -v

# Specific test
pytest tests/integration/test_cli_refined_subcommands.py::TestEpiValidateSubcommand::test_epi_validate_basic_execution -v
```

### Run All CLI Tests
```bash
# New + existing CLI tests
pytest tests/integration/test_cli_refined_subcommands.py tests/unit/cli/ -v

# Quick check
pytest tests/integration/test_cli_refined_subcommands.py -q
```

### Manual Testing
```bash
# Test help messages
tnfr --help
tnfr math.run --help
tnfr epi.validate --help

# Test commands
tnfr math.run --nodes 5 --steps 3
tnfr epi.validate --nodes 5 --steps 3

# Test presets
tnfr math.run --preset resonant_bootstrap --nodes 5 --steps 5
tnfr epi.validate --preset coupling_exploration --nodes 5 --steps 5
```

## Backward Compatibility

All existing CLI functionality is preserved:
- Original subcommands (`run`, `sequence`, `metrics`, `profile-si`, `profile-pipeline`) unchanged
- Existing command-line options work as before
- Existing tests continue to pass (46/48 passing, 2 pre-existing failures)
- No breaking changes to APIs or file formats

## Future Enhancements

Potential improvements for future PRs:
1. Add more preset templates for specific use cases
2. Support preset inheritance/composition in YAML
3. Add `tnfr preset.list` command to list available presets
4. Add `tnfr preset.validate` to validate YAML structure
5. Support loading presets from user directories
6. Add visualization subcommands (e.g., `tnfr viz.plot`)
7. Add batch processing subcommands
8. Support JSON Schema validation for YAML presets

## Documentation Updates Needed

The following documentation should be updated in a follow-up:
1. Main README.md - Add examples of new subcommands
2. docs/ - Add dedicated CLI reference page
3. CONTRIBUTING.md - Add guidelines for creating presets
4. examples/ - Add example scripts using new commands

## Performance Impact

No significant performance impact:
- New subcommands are separate code paths
- Math engine validation only runs when explicitly requested
- EPI validation is lightweight
- YAML preset loading is minimal overhead

## Security

- CodeQL scan: 0 alerts
- No new dependencies added
- YAML files use safe_load (no code execution)
- Input validation on all parameters
- No file system access beyond configured presets directory

## Conclusion

This PR successfully implements refined CLI functionality with:
- ✅ Dotted subcommands (`math.run`, `epi.validate`)
- ✅ YAML preset templates in `presets/`
- ✅ Enhanced help with examples
- ✅ Comprehensive test coverage (29 new tests, all passing)
- ✅ Full TNFR compliance
- ✅ No security issues
- ✅ Backward compatibility maintained

The CLI now provides a more structured, user-friendly interface for TNFR simulations while maintaining full fidelity to the canonical TNFR paradigm.
