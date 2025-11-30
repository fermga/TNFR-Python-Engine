#!/usr/bin/env python3
"""
TNFR Primality Testing Package
Entry point for the TNFR primality testing system.

This module provides a command-line interface for testing primality using
the Resonant Fractal Nature Theory (TNFR) with advanced repository integration.

Author: F. F. Martinez Gamo
"""

# Import advanced CLI if available, fallback to standard CLI
try:
    from .advanced_cli import main
    print("Using advanced TNFR infrastructure")
except ImportError:
    from .cli import main
    print("Using standard TNFR implementation")

if __name__ == "__main__":
    main()
