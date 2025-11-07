## Pull Request: [Brief Title]

### 🎯 Intent
<!-- Which coherence is improved or what structural capability is added? -->

### 🔧 Changes
<!-- High-level summary of changes -->

**Type of Change**:
- [ ] New feature (coherence expansion)
- [ ] Bug fix (stability improvement)
- [ ] Performance optimization
- [ ] Documentation update
- [ ] Domain extension
- [ ] Community pattern
- [ ] Infrastructure/tooling

### 🔬 Structural Impact

**Operators Involved**: 
<!-- List structural operators touched: emission, reception, coherence, etc. -->

**Affected Invariants**: 
<!-- Reference numbered canonical invariants from AGENTS.md: #1, #4, etc. -->

**Metrics Impact**:
<!-- Describe expected changes to C(t), Si, νf, or phase -->
- C(t): 
- Si: 
- νf: 
- Phase: 

### ✅ Quality Checklist

**Code Quality**:
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Type annotations complete (mypy passes)
- [ ] Docstrings follow NumPy style guide
- [ ] Code follows TNFR canonical conventions
- [ ] `.pyi` stub files generated/updated

**TNFR Canonical Requirements**:
- [ ] EPI changes only via structural operators
- [ ] Structural units (Hz_str) preserved
- [ ] ΔNFR semantics maintained
- [ ] Operator closure preserved
- [ ] Phase verification explicit in couplings
- [ ] Node birth/collapse conditions respected
- [ ] Operational fractality maintained
- [ ] Determinism/reproducibility ensured
- [ ] Structural metrics exposed in telemetry
- [ ] Domain neutrality maintained

**Testing**:
- [ ] Monotonicity tests pass (coherence doesn't decrease)
- [ ] Bifurcation tests pass (when applicable)
- [ ] Propagation tests pass (resonance)
- [ ] Multi-scale tests pass (fractality)
- [ ] Reproducibility verified (seeds work)

**Documentation**:
- [ ] CHANGELOG fragment added (`docs/changelog.d/`)
- [ ] API documentation updated (if applicable)
- [ ] Examples updated (if applicable)
- [ ] README updated (if applicable)

**Security** (if applicable):
- [ ] No vulnerabilities introduced
- [ ] Security audit passed (`make security-audit`)
- [ ] Dependency vulnerabilities addressed

### 🧪 Testing Evidence

**Test Coverage**:
```
# Paste test output showing new coverage
```

**Benchmark Results** (if performance-related):
```
# Paste benchmark comparison
```

**Health Metrics** (if applicable):
```
# Show C(t), Si measurements for new patterns/features
```

### 🔗 Related Issues
<!-- Link related issues: Closes #123, Relates to #456 -->

### 📋 Additional Context
<!-- Any other information reviewers should know -->

### 🎨 Visual Changes (if applicable)
<!-- Screenshots or diagrams showing UI/visualization changes -->

---

### For Extension Contributors

**Extension-Specific Checks** (if submitting domain extension):
- [ ] Follows `TNFRExtension` base class structure
- [ ] All patterns achieve health score > 0.75
- [ ] Minimum 3 validated use cases per pattern
- [ ] Integration tests included
- [ ] Domain documentation complete
- [ ] Real-world mapping clearly explained
- [ ] Extension validation passed

**Community Pattern Checks** (if submitting pattern):
- [ ] Pattern uses canonical English operators
- [ ] Health metrics documented
- [ ] Domain context explained
- [ ] Validation method described

---

### Reviewer Notes
<!-- Space for reviewer comments and feedback -->
