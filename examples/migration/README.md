# Grammar 2.0 Migration Examples

> DEPRECATION NOTICE (Scope): Canonical examples are indexed in `docs/source/examples/README.md`. Migration examples are non-central reference material; prefer the centralized docs for current guidance.

This directory contains comprehensive examples to help you migrate from Grammar 1.0 to Grammar 2.0.

## Examples

### 1. `before_after_comparison.py`
**Side-by-side comparisons** of sequences before and after Grammar 2.0 upgrades.

```bash
python examples/migration/before_after_comparison.py
```

**Demonstrates:**
- THOL validation fixes
- Frequency transition smoothing
- Operator balance improvements
- Health optimization strategies
- Pattern upgrades
- Three adoption strategies (Conservative, Progressive, Advanced)

### 2. `health_optimization_tutorial.py`
**Interactive tutorial** for using health metrics to optimize sequences.

```bash
python examples/migration/health_optimization_tutorial.py
```

**Lessons covered:**
1. Understanding health metrics (what each metric means)
2. Iterative optimization (step-by-step improvements)
3. Automatic upgrader (using SequenceUpgrader)
4. Pattern-aware optimization (different patterns, different goals)
5. Custom optimization strategies

### 3. `pattern_upgrade_examples.py`
**Pattern evolution examples** showing how to upgrade from basic to specialized patterns.

```bash
python examples/migration/pattern_upgrade_examples.py
```

**Pattern evolutions:**
- MINIMAL → LINEAR → HIERARCHICAL
- LINEAR → THERAPEUTIC
- STABILIZE → EDUCATIONAL
- HIERARCHICAL → ORGANIZATIONAL
- CYCLIC → REGENERATIVE
- Building CREATIVE patterns from scratch

### 4. `regenerative_cycles_intro.py`
**Introduction to regenerative cycles** - self-sustaining sequences.

```bash
python examples/migration/regenerative_cycles_intro.py
```

**Topics covered:**
- Basic regenerative cycles (TRANSITION, SILENCE, RECURSIVITY)
- Transformative cycles
- Common mistakes and how to fix them
- Designing your own cycles
- Real-world applications (cell cycles, learning, sprints)

## Quick Start

Run all examples in sequence:

```bash
# 1. See before/after comparisons
python examples/migration/before_after_comparison.py

# 2. Learn health optimization
python examples/migration/health_optimization_tutorial.py

# 3. Explore pattern upgrades
python examples/migration/pattern_upgrade_examples.py

# 4. Understand regenerative cycles
python examples/migration/regenerative_cycles_intro.py
```

## Migration Tools

These examples work with the migration tools in `tools/migration/`:

### Check your code for issues:
```bash
python -m tools.migration.migration_checker your_file.py
```

### Auto-upgrade a sequence:
```bash
python -m tools.migration.sequence_upgrader emission reception self_organization
```

## See Also

- **[Migration Guide](../../docs/MIGRATION_GUIDE_2.0.md)** - Complete migration documentation
- **[Health Metrics Guide](../../docs/HEALTH_METRICS_GUIDE.md)** - Deep dive into health metrics
- **[Pattern Reference](../../docs/PATTERN_REFERENCE.md)** - Complete pattern catalog
- **[Glyph Sequences Guide](../../GLYPH_SEQUENCES_GUIDE.md)** - Grammar 2.0 full documentation

## Learning Path

**Recommended order for first-time users:**

1. ✅ **Start here:** `before_after_comparison.py` - Understand what changed
2. 📊 **Learn optimization:** `health_optimization_tutorial.py` - Use health metrics
3. 🎯 **Master patterns:** `pattern_upgrade_examples.py` - Build specialized sequences
4. 🔄 **Advanced topic:** `regenerative_cycles_intro.py` - Self-sustaining systems

## Adoption Strategies

Choose your migration path:

### 🔵 Conservative (No Code Changes)
- Keep using `validate_sequence()`
- All existing code continues working
- See: Example 1 in `before_after_comparison.py`

### 🟡 Progressive (Opt-in Features)
- Switch to `validate_sequence_with_health()`
- Get health metrics for optimization
- See: `health_optimization_tutorial.py`

### 🟢 Advanced (Full Grammar 2.0)
- Use all new capabilities
- Pattern detection, regenerative cycles, auto-upgrader
- See: All examples for comprehensive coverage

## Need Help?

- **Issues?** Open an issue on GitHub
- **Questions?** Check the [Migration Guide](../../docs/MIGRATION_GUIDE_2.0.md)
- **Examples not working?** Make sure you have TNFR installed: `pip install -e .`
