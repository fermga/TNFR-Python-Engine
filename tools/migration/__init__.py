"""Migration tools for Grammar 2.0 adoption.

This package provides tools to help users migrate from Grammar 1.0 to Grammar 2.0:
- MigrationChecker: Scan code for potential compatibility issues
- SequenceUpgrader: Automatically improve sequence quality
"""

from .migration_checker import MigrationChecker, MigrationReport, MigrationIssue
from .sequence_upgrader import SequenceUpgrader, UpgradeResult

__all__ = [
    "MigrationChecker",
    "MigrationReport",
    "MigrationIssue",
    "SequenceUpgrader",
    "UpgradeResult",
]
