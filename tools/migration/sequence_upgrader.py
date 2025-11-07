"""Sequence upgrader for automatic Grammar 2.0 optimization.

Automatically improves operator sequences by:
- Fixing THOL validation issues
- Optimizing frequency transitions
- Balancing stabilizers and destabilizers
- Improving overall health metrics
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from tnfr.operators.grammar import validate_sequence_with_health
    from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
    HAS_TNFR = True
except ImportError:
    HAS_TNFR = False


@dataclass
class UpgradeResult:
    """Result of sequence upgrade operation."""
    original_sequence: List[str]
    upgraded_sequence: List[str]
    improvements: List[str]
    original_health: Optional[float] = None
    upgraded_health: Optional[float] = None
    
    @property
    def was_upgraded(self) -> bool:
        """Check if sequence was actually changed."""
        return self.original_sequence != self.upgraded_sequence
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("Sequence Upgrade Report")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Original:  {' → '.join(self.original_sequence)}")
        lines.append(f"Upgraded:  {' → '.join(self.upgraded_sequence)}")
        lines.append("")
        
        if self.original_health is not None and self.upgraded_health is not None:
            lines.append(f"Health: {self.original_health:.2f} → {self.upgraded_health:.2f} "
                        f"({self.upgraded_health - self.original_health:+.2f})")
        
        if self.improvements:
            lines.append("")
            lines.append("Improvements:")
            for improvement in self.improvements:
                lines.append(f"  • {improvement}")
        else:
            lines.append("")
            lines.append("✓ Sequence already optimal, no changes needed")
        
        return "\n".join(lines)


class SequenceUpgrader:
    """Automatic sequence optimizer for Grammar 2.0.
    
    Applies targeted improvements to sequences:
    1. Fix THOL validation (add destabilizers)
    2. Smooth frequency transitions
    3. Balance stabilizers/destabilizers
    4. Add missing terminators
    """
    
    DESTABILIZERS = {"dissonance", "mutation", "contraction"}
    STABILIZERS = {"coherence", "silence", "resonance", "coupling"}
    HIGH_FREQ = {"emission", "dissonance", "resonance", "mutation", "contraction"}
    MEDIUM_FREQ = {"reception", "coherence", "coupling", "expansion", 
                   "self_organization", "transition", "recursivity"}
    ZERO_FREQ = {"silence"}
    
    def __init__(self, target_health: float = 0.75):
        """Initialize upgrader.
        
        Args:
            target_health: Target health score for upgrades (0.0-1.0)
        """
        self.target_health = target_health
        self.analyzer = SequenceHealthAnalyzer() if HAS_TNFR else None
    
    def upgrade_sequence(self, sequence: List[str], 
                        preserve_length: bool = False) -> UpgradeResult:
        """Upgrade a sequence to improve Grammar 2.0 compatibility.
        
        Args:
            sequence: Original operator sequence
            preserve_length: If True, avoid adding operators (only reorder/replace)
            
        Returns:
            UpgradeResult with upgraded sequence and improvements
        """
        original = list(sequence)
        upgraded = list(sequence)
        improvements = []
        
        # Get original health if available
        original_health = self._get_health(original)
        
        # Step 1: Fix THOL issues (critical)
        upgraded, thol_improvements = self._fix_thol_issues(upgraded, preserve_length)
        improvements.extend(thol_improvements)
        
        # Step 2: Fix frequency transitions
        if not preserve_length:
            upgraded, freq_improvements = self._fix_frequency_transitions(upgraded)
            improvements.extend(freq_improvements)
        
        # Step 3: Balance stabilizers and destabilizers
        if not preserve_length:
            upgraded, balance_improvements = self._balance_operators(upgraded)
            improvements.extend(balance_improvements)
        
        # Step 4: Ensure proper termination
        if not preserve_length:
            upgraded, term_improvements = self._ensure_termination(upgraded)
            improvements.extend(term_improvements)
        
        # Get upgraded health
        upgraded_health = self._get_health(upgraded)
        
        return UpgradeResult(
            original_sequence=original,
            upgraded_sequence=upgraded,
            improvements=improvements,
            original_health=original_health,
            upgraded_health=upgraded_health
        )
    
    def improve_to_target(self, sequence: List[str], 
                         max_iterations: int = 3) -> UpgradeResult:
        """Iteratively improve sequence until target health is reached.
        
        Args:
            sequence: Original operator sequence
            max_iterations: Maximum improvement iterations
            
        Returns:
            UpgradeResult with final upgraded sequence
        """
        current = list(sequence)
        all_improvements = []
        original_health = self._get_health(sequence)
        
        for iteration in range(max_iterations):
            result = self.upgrade_sequence(current, preserve_length=False)
            
            if not result.was_upgraded:
                break  # No more improvements possible
            
            current = result.upgraded_sequence
            all_improvements.extend(result.improvements)
            
            # Check if target reached
            if result.upgraded_health and result.upgraded_health >= self.target_health:
                all_improvements.append(f"Target health {self.target_health:.2f} reached")
                break
        
        return UpgradeResult(
            original_sequence=sequence,
            upgraded_sequence=current,
            improvements=all_improvements,
            original_health=original_health,
            upgraded_health=self._get_health(current)
        )
    
    def _fix_thol_issues(self, sequence: List[str], 
                        preserve_length: bool) -> Tuple[List[str], List[str]]:
        """Fix SELF_ORGANIZATION validation issues."""
        improvements = []
        result = list(sequence)
        seq_lower = [op.lower() for op in sequence]
        
        for i, op in enumerate(seq_lower):
            if op in {"self_organization", "thol"}:
                # Check 3-operator window before THOL
                window_start = max(0, i - 3)
                window = seq_lower[window_start:i]
                
                has_destabilizer = any(op in self.DESTABILIZERS for op in window)
                
                if not has_destabilizer:
                    if preserve_length and i > 0:
                        # Replace previous operator with destabilizer
                        result[i - 1] = "dissonance"
                        improvements.append(f"Replaced operator before SELF_ORGANIZATION with DISSONANCE")
                    else:
                        # Insert destabilizer before THOL
                        result.insert(i, "dissonance")
                        improvements.append(f"Added DISSONANCE before SELF_ORGANIZATION")
                    break  # Only fix first occurrence per pass
        
        return result, improvements
    
    def _fix_frequency_transitions(self, sequence: List[str]) -> Tuple[List[str], List[str]]:
        """Smooth out problematic frequency transitions."""
        improvements = []
        result = list(sequence)
        seq_lower = [op.lower() for op in sequence]
        
        i = 0
        while i < len(seq_lower) - 1:
            current = seq_lower[i]
            next_op = seq_lower[i + 1]
            
            # Check for SILENCE → HIGH frequency jump
            if current == "silence" and next_op in self.HIGH_FREQ:
                # Insert medium-frequency bridge
                result.insert(i + 1, "transition")
                seq_lower.insert(i + 1, "transition")
                improvements.append(f"Added TRANSITION to smooth SILENCE → {next_op.upper()} transition")
                i += 1  # Skip the inserted operator
            
            i += 1
        
        return result, improvements
    
    def _balance_operators(self, sequence: List[str]) -> Tuple[List[str], List[str]]:
        """Balance stabilizers and destabilizers."""
        improvements = []
        result = list(sequence)
        seq_lower = [op.lower() for op in sequence]
        
        # Count stabilizers and destabilizers
        stabilizer_count = sum(1 for op in seq_lower if op in self.STABILIZERS)
        destabilizer_count = sum(1 for op in seq_lower if op in self.DESTABILIZERS)
        
        # If too many destabilizers without stabilizers, add coherence at end
        if destabilizer_count > 0 and stabilizer_count == 0:
            result.append("coherence")
            improvements.append("Added COHERENCE to balance destabilizers")
        
        # If sequence has destabilizer but doesn't end with stabilizer
        elif destabilizer_count > 0:
            last_op = seq_lower[-1]
            if last_op not in self.STABILIZERS:
                result.append("coherence")
                improvements.append("Added COHERENCE terminator after destabilizers")
        
        return result, improvements
    
    def _ensure_termination(self, sequence: List[str]) -> Tuple[List[str], List[str]]:
        """Ensure sequence ends with appropriate terminator."""
        improvements = []
        result = list(sequence)
        
        if not result:
            return result, improvements
        
        seq_lower = [op.lower() for op in sequence]
        last_op = seq_lower[-1]
        
        # Valid terminators
        valid_terminators = {
            "coherence", "silence", "resonance", "coupling",
            "self_organization", "recursivity", "contraction", "mutation"
        }
        
        if last_op not in valid_terminators:
            # Add appropriate terminator based on sequence context
            if any(op in self.DESTABILIZERS for op in seq_lower):
                result.append("coherence")
                improvements.append("Added COHERENCE terminator for stability")
            else:
                result.append("silence")
                improvements.append("Added SILENCE terminator")
        
        return result, improvements
    
    def _get_health(self, sequence: List[str]) -> Optional[float]:
        """Get health score for sequence if TNFR available."""
        if not HAS_TNFR or not self.analyzer:
            return None
        
        try:
            # Validate and get health
            result = validate_sequence_with_health(sequence)
            if result.passed:
                return result.health_metrics.overall_health
        except Exception:
            pass
        
        return None


def main():
    """CLI entry point for sequence upgrader."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m tools.migration.sequence_upgrader <operator1> <operator2> ...")
        print("\nExample: python -m tools.migration.sequence_upgrader emission reception self_organization")
        sys.exit(1)
    
    sequence = sys.argv[1:]
    upgrader = SequenceUpgrader(target_health=0.75)
    
    result = upgrader.upgrade_sequence(sequence)
    print(result.summary())
    
    sys.exit(0 if result.upgraded_health and result.upgraded_health >= 0.65 else 1)


if __name__ == "__main__":
    main()
