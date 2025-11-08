# TNFR Python Engine Wiki

This directory contains the basic wiki documentation for the TNFR Python Engine repository.

## 📚 Wiki Pages

The wiki consists of the following pages:

- **[Home](Home.md)** - Main landing page with overview and navigation
- **[Getting Started](Getting-Started.md)** - Installation, quick start, and basic workflow
- **[Core Concepts](Core-Concepts.md)** - Fundamental TNFR principles and theory
- **[Examples](Examples.md)** - Domain-specific use cases and code examples

## 🎨 Visual Assets

All wiki graphics are in the [`images/`](images/) directory:

- `network_resonance.png` - Visualization of NFR nodes in a resonant network
- `structural_operators.png` - Visual reference for the 13 canonical operators
- `paradigm_comparison.png` - Traditional vs. TNFR paradigm comparison
- `coherence_metrics.png` - Time-series visualization of C(t), Si, ΔNFR, νf
- `nodal_equation.png` - Visual representation of the canonical nodal equation

## 🔧 Regenerating Graphics

To regenerate the visual assets:

```bash
cd wiki
python generate_graphics.py
```

Requirements:
- `matplotlib`
- `numpy`

The script will create all PNG images in the `images/` directory.

## 📝 Using as GitHub Wiki

These markdown files can be used in several ways:

### Option 1: GitHub Wiki (Recommended)

1. Go to your repository's Wiki tab
2. Create pages with these filenames (without .md extension):
   - Home
   - Getting-Started
   - Core-Concepts
   - Examples
3. Copy the content from each .md file
4. Upload images to each page where they're referenced

### Option 2: MkDocs Site

These pages are compatible with the existing MkDocs setup:

```bash
# Add to mkdocs.yml navigation
nav:
  - Wiki:
      - Home: wiki/Home.md
      - Getting Started: wiki/Getting-Started.md
      - Core Concepts: wiki/Core-Concepts.md
      - Examples: wiki/Examples.md
```

### Option 3: Standalone Viewing

Read directly on GitHub or locally with any markdown viewer.

## 🎯 Design Principles

The wiki follows these principles:

- ✅ **English only** - All content in English per TNFR guidelines
- ✅ **Visual-first** - Heavy use of diagrams and graphics
- ✅ **Paradigm-aligned** - Strict adherence to TNFR concepts
- ✅ **Beginner-friendly** - Progressive complexity, lots of examples
- ✅ **Code-heavy** - Executable examples throughout
- ✅ **Self-contained** - Each page can be read independently

## 📖 Content Structure

### Home Page
- Overview of TNFR paradigm
- Quick navigation
- Installation and quick example
- Key visual graphics

### Getting Started
- Step-by-step installation
- First network creation
- Core workflow patterns
- Domain examples
- Troubleshooting

### Core Concepts
- Deep dive into NFRs (EPI, νf, φ)
- Canonical nodal equation
- 13 structural operators
- Coherence metrics
- Operational fractality

### Examples
- Biology & life sciences
- Social networks
- AI & machine learning
- Distributed systems
- Finance & markets
- Physics simulation
- Creative applications

## 🤝 Contributing

To improve the wiki:

1. Edit the markdown files in this directory
2. Update `generate_graphics.py` to add new visualizations
3. Maintain consistency with TNFR paradigm (see `AGENTS.md`)
4. Test all code examples
5. Submit a PR

## 🔗 Links

- **Main Documentation**: https://tnfr.netlify.app
- **GitHub Repository**: https://github.com/fermga/TNFR-Python-Engine
- **PyPI Package**: https://pypi.org/project/tnfr/

---

**Note**: This wiki provides a basic introduction. For comprehensive documentation, see the [full documentation site](https://tnfr.netlify.app).
