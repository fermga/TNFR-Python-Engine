# TNFR Grammar 2.0 - Interactive Web Documentation

This directory contains the interactive web documentation for TNFR Grammar 2.0, designed to provide a progressive learning experience from fundamentals to advanced applications.

## Structure

```
docs/website/
├── index.html                    # Landing page with interactive playground
├── getting-started.html          # Progressive 5-step tutorial
├── assets/
│   ├── tnfr.css                 # Complete styling system
│   ├── tnfr-validator.js        # Client-side sequence validation
│   └── tnfr-interactive.js      # Interactive tutorial system
├── patterns/
│   └── index.html               # Pattern gallery with 18+ validated patterns
├── health-metrics/
│   └── index.html               # Health metrics dashboard & calculator
├── tools/
│   └── validator.html           # Interactive validator & pattern explorer
├── api/
│   └── reference.html           # Python API documentation
└── notebooks/                   # (Future) Jupyter notebooks
```

## Features

### Landing Page (index.html)
- **Hero section** with TNFR overview
- **Interactive sequence playground** with real-time validation
- **Feature highlights**: operational fractality, traceability, reproducibility, trans-scale
- **Core concepts** overview
- **Quickstart code** example

### Getting Started Tutorial (getting-started.html)
- **5-step progressive tutorial**:
  1. Installation
  2. Core Concepts (NFR, operators, fundamental equation)
  3. First TNFR Network
  4. Health Metrics Introduction
  5. Pattern Exploration
- **Interactive validation** examples
- **Next steps** guidance

### Pattern Gallery (patterns/index.html)
- **Domain-organized** patterns:
  - 🏥 Therapeutic (Crisis Intervention, Process Therapy)
  - 🎓 Educational (Conceptual Breakthrough, Competency Development)
  - 🏢 Organizational (Team Formation, Strategic Planning)
  - 🎨 Creative (Artistic Creation)
- **Filterable** by domain
- **Live validation** for each pattern
- **Detailed metrics** (health, balance, sustainability, efficiency)
- **Use cases** and guidance

### Health Metrics Dashboard (health-metrics/index.html)
- **Interactive calculator** for any sequence
- **Comprehensive metrics explanation**:
  - Overall Health
  - Coherence Index
  - Balance Score
  - Sustainability Index
  - Complexity Efficiency
- **Example profiles** with different health scores
- **Improvement tips** for each dimension

### Interactive Tools (tools/validator.html)
- **Sequence Validator**: Real-time validation with detailed feedback
- **Pattern Explorer**: Browse and analyze pre-validated patterns
- **Operator Quick Reference**: 13 canonical operators with descriptions

### API Reference (api/reference.html)
- **Validation API**: `validate_sequence_with_health()`
- **Health Metrics**: `SequenceHealthMetrics` dataclass
- **Pattern Detection**: `AdvancedPatternDetector`
- **Operator names** and functions

## Technology Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **No dependencies**: Lightweight, fast loading
- **Responsive design**: Mobile-first approach
- **Client-side validation**: Port of Python validation logic
- **GitHub Pages**: Static site hosting

## Design Principles

Following TNFR paradigm:
- **Structural coherence**: Website design reflects TNFR principles
- **Operational focus**: All examples are executable and validated
- **Trans-scale learning**: Progressive from basic to advanced
- **Complete traceability**: Clear learning path and references
- **Reproducibility**: Consistent examples and results

## Usage

### Local Development

Simply open any HTML file in a modern browser:

```bash
cd docs/website
python -m http.server 8000
# Visit http://localhost:8000
```

### GitHub Pages Deployment

The site is automatically deployed via GitHub Pages. Configuration is in the repository root:

- `.github/workflows/` - CI/CD automation
- `docs/website/` - Published to `https://fermga.github.io/TNFR-Python-Engine/`

## Content Sources

All content is sourced from canonical TNFR documentation:
- `docs/PATTERN_COOKBOOK.md` - Pattern examples and metrics
- `docs/HEALTH_METRICS_GUIDE.md` - Health metrics explanations
- `docs/PATTERN_REFERENCE.md` - Pattern definitions
- Main documentation files for concepts and fundamentals

## Browser Compatibility

Tested and working on:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Future Enhancements

- [ ] JupyterLite notebooks for browser-based execution
- [ ] Sequence generator tool
- [ ] Pattern comparison tool
- [ ] Advanced search and filtering
- [ ] Multi-language support
- [ ] Dark mode theme
- [ ] Accessibility improvements (WCAG 2.1 AA)

## Contributing

This interactive documentation is part of the TNFR Python Engine. See main repository for contribution guidelines:
- [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)
- [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)

## License

MIT License - See [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md)
