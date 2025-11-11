# TNFR Documentation

## 📚 Documentation Site

The TNFR Python Engine documentation is built with **MkDocs** and automatically deployed to **GitHub Pages**.

**Live Documentation**: https://fermga.github.io/TNFR-Python-Engine/

---

## 🏗️ Building Documentation Locally

### Prerequisites

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt
```

### Build and Preview

```bash
# Build the documentation
mkdocs build

# Serve locally with live reload (recommended for development)
mkdocs serve
```

Then open http://127.0.0.1:8000/ in your browser.

---

## 🚀 Deployment

Documentation is **automatically deployed** to GitHub Pages when changes are pushed to the `main` branch:

1. **On Push to Main**: `.github/workflows/deploy-docs.yml` triggers
2. **Build Process**: MkDocs builds the site from `docs/source/`
3. **Deploy**: Built site is pushed to the `gh-pages` branch
4. **Published**: GitHub Pages serves the site at https://fermga.github.io/TNFR-Python-Engine/

### Manual Deployment

To trigger a manual deployment:

```bash
# Via GitHub Actions UI
# Go to: Actions → Deploy Documentation to GitHub Pages → Run workflow
```

---

## 📝 Documentation Structure

```
docs/
├── source/                    # MkDocs source files
│   ├── getting-started/       # Tutorials and quickstart
│   ├── api/                   # API reference
│   ├── advanced/              # Advanced topics
│   ├── theory/                # Mathematical foundations (Jupyter notebooks)
│   ├── examples/              # Example code and use cases
│   ├── security/              # Security documentation
│   └── home.md                # Homepage
├── grammar/                   # Grammar system documentation
│   ├── README.md              # Grammar navigation hub
│   ├── 01-08 *.md            # Core documentation
│   ├── examples/              # Grammar examples
│   └── schemas/               # JSON schemas
├── requirements.txt           # Documentation dependencies
└── README.md                  # This file
```

---

## 🔧 Configuration

- **MkDocs Config**: `mkdocs.yml` (root directory)
- **Theme**: Material for MkDocs
- **Plugins**: 
  - `mkdocs-jupyter` - Jupyter notebook support
  - `search` - Full-text search

---

## ✅ Validation

Documentation is validated on every pull request:

- `.github/workflows/docs.yml` runs validation checks
- Ensures documentation builds without errors
- Only PRs with valid documentation can be merged

---

## 📖 Writing Documentation

### Adding New Pages

1. Create a new markdown file in `docs/source/`
2. Add the page to the `nav` section in `mkdocs.yml`
3. Build locally to verify: `mkdocs serve`
4. Commit and push - deployment is automatic

### Jupyter Notebooks

Place Jupyter notebooks in `docs/source/theory/` or other appropriate directories. They will be automatically converted and included in the documentation.

### Markdown Extensions

Supported extensions:
- Admonitions (`!!! note`, `!!! warning`, etc.)
- Code highlighting with `pymdownx.highlight`
- Tables, footnotes, definition lists
- Table of contents with permalinks

---

## 🐛 Troubleshooting

### Build Errors

```bash
# Check for syntax errors
mkdocs build --strict

# View detailed error messages
mkdocs build --verbose
```

### Missing Pages

If a page doesn't appear:
1. Verify it's listed in `mkdocs.yml` under `nav`
2. Check the file path is correct relative to `docs/source/`
3. Ensure the file has a `.md` extension

### Broken Links

MkDocs will warn about broken links during build. Check the build output for warnings about missing targets.

---

## 📊 Migration from Netlify

**Previous Setup**: Documentation was built on Netlify  
**Current Setup**: Documentation is built and deployed via GitHub Actions to GitHub Pages

**Benefits**:
- ✅ Faster deployment (native GitHub integration)
- ✅ No external service dependencies
- ✅ Better version control (gh-pages branch)
- ✅ Automatic deployment on push to main
- ✅ Free hosting with GitHub Pages

**Netlify Configuration**: Disabled (see `netlify.toml.disabled`)

---

## 🔗 Related Documentation

- **Grammar Documentation**: [docs/grammar/README.md](grammar/README.md)
- **Main README**: [Repository README](https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md)
- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)

---

<div align="center">

**Documentation is code. Treat it with the same care.**

</div>
