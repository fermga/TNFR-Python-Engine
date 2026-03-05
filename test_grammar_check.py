
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_types import SequenceValidationResult

seq = ['emission', 'dissonance']
result = validate_sequence(seq)
print(f"Sequence: {seq}")
print(f"Passed: {result.passed}")
print(f"Message: {result.message}")
print(f"Error: {result.error}")
print(f"Error Type: {type(result.error)}")
