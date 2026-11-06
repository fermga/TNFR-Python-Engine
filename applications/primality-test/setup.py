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
    description="Standalone arithmetic-pressure primality diagnostics and bounded caching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    url="https://doi.org/10.5281/zenodo.17764749",
    project_urls={
        "Repository": "https://github.com/fermga/TNFR-Python-Engine",
        "Bug Tracker": "https://github.com/fermga/TNFR-Python-Engine/issues",
        "Documentation": "https://github.com/fermga/TNFR-Python-Engine/tree/main/applications/primality-test",
        "Source": "https://github.com/fermga/TNFR-Python-Engine/tree/main/applications/primality-test",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
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
    packages=find_packages(include=["tnfr_primality", "tnfr_primality.*"]),
    python_requires=">=3.8",
    install_requires=[
        # Minimal core - no heavy dependencies by default
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "build>=1.0",
            "setuptools>=61",
            "wheel",
        ],
    },
    entry_points={
        "console_scripts": [
            "tnfr-primality=tnfr_primality.cli:main",
        ],
    },
    keywords=[
        "primality testing",
        "TNFR",
        "number theory",
        "mathematics",
        "algorithms",
        "arithmetic pressure",
        "deterministic algorithms",
    ],
    include_package_data=True,
    zip_safe=False,
)
