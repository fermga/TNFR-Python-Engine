import ast
import os
from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_ROOT = Path(__file__).resolve().parent
# Prevent a caller's pyproject.toml from replacing this distribution's metadata.
os.chdir(PACKAGE_ROOT)
long_description = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
module = ast.parse(
    (PACKAGE_ROOT / "tnfr_primality" / "__init__.py").read_text(encoding="utf-8")
)
version = next(
    ast.literal_eval(statement.value)
    for statement in module.body
    if isinstance(statement, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "__version__"
        for target in statement.targets
    )
)

setup(
    name="tnfr-primality",
    version=version,
    author="F. F. Martinez Gamo",
    description="Standalone arithmetic-pressure primality tests with optional TNFR adapters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    url="https://doi.org/10.5281/zenodo.17764749",
    project_urls={
        "Repository": "https://github.com/fermga/TNFR-Python-Engine",
        "Bug Tracker": "https://github.com/fermga/TNFR-Python-Engine/issues",
        "Documentation": "https://github.com/fermga/TNFR-Python-Engine/tree/main/primality-test",
        "Source": "https://github.com/fermga/TNFR-Python-Engine/tree/main/primality-test",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
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
            # Engine dependencies have their owner in the engine distribution.
            "tnfr>=0.0.3.7",
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
        # Compatibility alias: build documentation with the repository root docs extra.
        "docs": [],
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
        "structural field tetrad",
        "canonical operators",
        "prime certificates",
        "structural fields",
        "resonant fractal dynamics",
    ],
    include_package_data=True,
    zip_safe=False,
)
