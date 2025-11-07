# Theory Notebooks - Deprecation Notice

**Status**: DEPRECATED as of 2025-11-07

## What Happened

The Jupyter notebooks that previously lived in this directory have been **consolidated into the unified mathematical foundations document**:

- **New location**: [`mathematical_foundations.md`](./mathematical_foundations.md)
- **Rationale**: Maintain a single source of truth for TNFR mathematical formalization

## Removed Notebooks

The following notebooks have been deprecated and their content integrated:

| Notebook | Content Now In | Section |
|----------|---------------|---------|
| `00_overview.ipynb` | `mathematical_foundations.md` | Appendix A.1 |
| `01_hilbert_space_h_nfr.ipynb` | `mathematical_foundations.md` | Appendix A.2 |
| `03_frequency_operator_hatJ.ipynb` | `mathematical_foundations.md` | Appendix A.3 |
| `05_unitary_dynamics_and_delta_nfr.ipynb` | `mathematical_foundations.md` | Appendix A.4 |

## Remaining Notebooks

The following notebooks remain as **interactive tutorials** and are NOT deprecated:

- `01_structural_frequency_primer.ipynb` - Interactive frequency exploration
- `02_phase_synchrony_lattices.ipynb` - Phase dynamics visualization
- `03_delta_nfr_gradient_fields.ipynb` - ΔNFR field analysis
- `04_coherence_metrics_walkthrough.ipynb` - Coherence calculation walkthrough
- `04_nfr_validator_and_metrics.ipynb` - Validation utilities
- `05_sense_index_calibration.ipynb` - Si calibration guide
- `06_recursivity_cascades.ipynb` - Recursive operator exploration

These remain valuable for:
- Interactive parameter exploration
- Visualization of dynamic behaviors
- Hands-on learning experiences
- Reproducible computational experiments

## Why Consolidate?

**Benefits of single-source documentation**:

1. ✅ **No divergence**: Mathematical formulas can't become inconsistent between sources
2. ✅ **Easier maintenance**: Update equations once, not in multiple places
3. ✅ **Better searchability**: All theory in one searchable document
4. ✅ **Version control**: Markdown diffs are clearer than notebook JSON
5. ✅ **Reduced duplication**: Avoid maintaining parallel explanations

**Interactive notebooks remain for**:

- Computational demonstrations that benefit from inline execution
- Parameter sensitivity analysis requiring plots
- Step-by-step walkthroughs with intermediate visualizations
- Exploratory analysis not yet formalized

## Migration Path

If you referenced deprecated notebooks:

**Before**:
```markdown
See [Hilbert Space notebook](./01_hilbert_space_h_nfr.ipynb) for details.
```

**After**:
```markdown
See [Mathematical Foundations §2.1 and Appendix A.2](./mathematical_foundations.md#21-hilbert-space-h_nfr) for details.
```

## Questions?

- **Theory questions**: See [`mathematical_foundations.md`](./mathematical_foundations.md)
- **Implementation questions**: See docstrings in `src/tnfr/metrics/`
- **Worked examples**: See [`docs/source/examples/worked_examples.md`](../examples/worked_examples.md)
- **Interactive tutorials**: Use remaining notebooks listed above

---

**Last updated**: 2025-11-07  
**Maintained by**: TNFR Documentation Team
