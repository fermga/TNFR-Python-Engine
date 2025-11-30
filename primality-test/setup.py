from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tnfr-primality",
    version="1.0.0",
    author="F. F. Martinez Gamo",
    description="Advanced TNFR-based primality testing with full repository integration and structural coherence analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://doi.org/10.5281/zenodo.17764749",
    project_urls={
        "Repository": "https://github.com/fermga/TNFR-Python-Engine",
        "Bug Tracker": "https://github.com/fermga/TNFR-Python-Engine/issues",
        "Documentation": "https://github.com/fermga/TNFR-Python-Engine/tree/main/zenodo-package/docs",
        "Source": "https://github.com/fermga/TNFR-Python-Engine/tree/main/zenodo-package",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Minimal core - no heavy dependencies by default
    ],
    extras_require={
        "full": [
            # Full TNFR infrastructure (automatically detected)
            "numpy>=1.20", 
            "scipy>=1.8",
            "networkx>=2.8",
            "sympy>=1.10",
        ],
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pytest-benchmark>=4.0",
        ],
        "benchmark": [
            "matplotlib>=3.5",
            "numpy>=1.20",
            "pandas>=1.5",  # For advanced analytics
            "jupyter>=1.0",  # For interactive benchmarking
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.2",
            "myst-parser>=0.18",
        ],
    },
    entry_points={
        "console_scripts": [
            "tnfr-primality=tnfr_primality.cli:main",
            "tnfr-primality-advanced=tnfr_primality.advanced_cli:main",
            "tnfr-primality-legacy=tnfr_primality.__main__:main",
        ],
    },
    keywords=[
        "primality testing",
        "TNFR",
        "number theory",
        "mathematics",
        "algorithms",
        "arithmetic pressure",
        "structural coherence",
        "deterministic algorithms",
        "hierarchical caching",
        "tetrahedral correspondence",
        "canonical operators",
        "prime certificates",
        "structural fields",
        "resonant fractal dynamics"
    ],
    include_package_data=True,
    zip_safe=False,
)
