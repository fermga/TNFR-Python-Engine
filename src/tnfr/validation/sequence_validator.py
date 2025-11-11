"""Semantic validation for TNFR operator sequences.

This module validates the semantic correctness of operator sequences beyond
syntactic grammar rules. It checks for patterns that may indicate structural
instability or non-canonical reorganization.
"""

from __future__ import annotations

from .invariants import InvariantSeverity, InvariantViolation

__all__ = [
    "SequenceSemanticValidator",
]


class SequenceSemanticValidator:
    """Validates semantics of operator sequences."""

    def __init__(self) -> None:
        """Initialize semantic validator with default rules."""
        # Semantic rules between operators
        self.semantic_rules: dict[str, dict] = {
            "mutation_without_stabilization": {
                "pattern": ["mutation"],
                "not_followed_by": ["coherence", "silence"],
                "severity": InvariantSeverity.WARNING,
                "message": "Mutation should be followed by stabilization (coherence or silence)",
            },
            "excessive_dissonance": {
                "pattern": ["dissonance", "dissonance", "dissonance"],
                "severity": InvariantSeverity.ERROR,
                "message": "Excessive consecutive dissonance may destabilize structure",
            },
            "contraction_after_silence": {
                "pattern": ["silence", "contraction"],
                "severity": InvariantSeverity.WARNING,
                "message": "Contraction immediately after silence may be redundant",
            },
            "excessive_recursivity": {
                "pattern": ["recursivity", "recursivity", "recursivity"],
                "severity": InvariantSeverity.WARNING,
                "message": "Excessive consecutive recursivity may indicate inefficient sequence design",
            },
            "silence_after_mutation": {
                "pattern": ["mutation", "silence"],
                "severity": InvariantSeverity.INFO,
                "message": "Silence after mutation preserves the new phase state (valid pattern)",
            },
        }

    def validate_semantic_sequence(self, sequence: list[str]) -> list[InvariantViolation]:
        """Validates semantics of the operator sequence.

        Parameters
        ----------
        sequence : list[str]
            List of operator names in sequence.

        Returns
        -------
        list[InvariantViolation]
            List of semantic violations found.
        """
        violations: list[InvariantViolation] = []

        for rule_name, rule in self.semantic_rules.items():
            violations.extend(self._check_rule(sequence, rule_name, rule))

        return violations

    def _check_rule(
        self, sequence: list[str], rule_name: str, rule: dict
    ) -> list[InvariantViolation]:
        """Verifies a specific semantic rule.

        Parameters
        ----------
        sequence : list[str]
            Operator sequence to check.
        rule_name : str
            Name of the rule being checked.
        rule : dict
            Rule definition with pattern and constraints.

        Returns
        -------
        list[InvariantViolation]
            Violations found for this rule.
        """
        violations: list[InvariantViolation] = []
        pattern = rule["pattern"]

        # Search for patterns in the sequence
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i : i + len(pattern)] == pattern:

                # If there are no additional constraints, the pattern itself is a violation
                if "not_followed_by" not in rule and "not_preceded_by" not in rule:
                    violations.append(
                        InvariantViolation(
                            invariant_id=0,
                            severity=rule["severity"],
                            description=f"Semantic rule violation: {rule['message']}",
                            suggestion=rule.get("suggestion", "Review operator sequence"),
                        )
                    )
                    continue

                # Verificar reglas 'not_followed_by'
                if "not_followed_by" in rule:
                    next_pos = i + len(pattern)
                    if next_pos < len(sequence):
                        next_op = sequence[next_pos]
                        if next_op not in rule["not_followed_by"]:
                            violations.append(
                                InvariantViolation(
                                    invariant_id=0,  # Semántica, no invariante numérico
                                    severity=rule["severity"],
                                    description=f"Semantic rule violation: {rule['message']}",
                                    suggestion=f"Consider adding one of: {rule['not_followed_by']}",
                                )
                            )
                    # Si es el último operador y la regla requiere algo después
                    elif next_pos == len(sequence):
                        violations.append(
                            InvariantViolation(
                                invariant_id=0,
                                severity=rule["severity"],
                                description=f"Semantic rule violation: {rule['message']}",
                                suggestion=f"Sequence should end with one of: {rule['not_followed_by']}",
                            )
                        )

                # Verificar reglas 'not_preceded_by'
                if "not_preceded_by" in rule:
                    if i == 0:  # No hay operador anterior
                        violations.append(
                            InvariantViolation(
                                invariant_id=0,
                                severity=rule["severity"],
                                description=f"Semantic rule violation: {rule['message']}",
                                suggestion=f"Sequence should start with one of: {rule['not_preceded_by']}",
                            )
                        )
                    else:
                        prev_op = sequence[i - 1]
                        if prev_op not in rule["not_preceded_by"]:
                            violations.append(
                                InvariantViolation(
                                    invariant_id=0,
                                    severity=rule["severity"],
                                    description=f"Semantic rule violation: {rule['message']}",
                                    suggestion=f"'{pattern[0]}' should be preceded by one of: {rule['not_preceded_by']}",
                                )
                            )

        return violations
