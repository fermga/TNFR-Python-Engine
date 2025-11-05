# SQL Injection Prevention in TNFR

## Overview

This document describes the SQL injection prevention utilities provided by the TNFR Python Engine. While the engine currently uses in-memory NetworkX graphs and file-based persistence, these utilities are provided proactively for future database functionality.

## Current Status

**No SQL databases are currently used in TNFR.** The codebase uses:
- NetworkX graphs (in-memory)
- File-based persistence (JSON, YAML, TOML)
- Optional caching (Shelve, Redis with Pickle serialization)

The SQL injection prevention utilities in `tnfr.security` are provided as a **proactive security measure** should database functionality be added in the future.

## Security Module

### Location

All security utilities are in the `tnfr.security` module:

```
src/tnfr/security/
├── __init__.py          # Module exports
├── database.py          # SQL injection prevention
└── validation.py        # TNFR structural data validation
```

### Module Imports

```python
from tnfr.security import (
    # SQL injection prevention
    SecureQueryBuilder,
    validate_identifier,
    sanitize_string_input,
    execute_parameterized_query,
    SQLInjectionError,
    
    # TNFR structural validation
    validate_nodal_input,
    validate_structural_frequency,
    validate_phase_value,
    validate_coherence_value,
    validate_sense_index,
)
```

## SQL Injection Prevention

### 1. SecureQueryBuilder

The `SecureQueryBuilder` class provides a fluent interface for constructing safe, parameterized SQL queries.

#### Basic SELECT Query

```python
from tnfr.security import SecureQueryBuilder

builder = SecureQueryBuilder()
query, params = builder.select("nfr_nodes", ["id", "nu_f", "phase"]).build()

# Result:
# query = "SELECT id, nu_f, phase FROM nfr_nodes"
# params = []
```

#### SELECT with WHERE Clause

```python
builder = SecureQueryBuilder()
query, params = builder.select("nfr_nodes", ["id", "nu_f"])\
    .where("nu_f > ?", 0.5)\
    .where("phase BETWEEN ? AND ?", 0.0, 3.14)\
    .build()

# Result:
# query = "SELECT id, nu_f FROM nfr_nodes WHERE nu_f > ? AND phase BETWEEN ? AND ?"
# params = [0.5, 0.0, 3.14]
```

#### Complex SELECT Query

```python
builder = SecureQueryBuilder()
query, params = builder.select("nfr_nodes", ["id", "nu_f", "phase"])\
    .where("nu_f > ?", 0.7)\
    .order_by("nu_f", "DESC")\
    .limit(10)\
    .build()

# Result:
# query = "SELECT id, nu_f, phase FROM nfr_nodes WHERE nu_f > ? ORDER BY nu_f DESC LIMIT 10"
# params = [0.7]
```

#### INSERT Query

```python
builder = SecureQueryBuilder()
query, params = builder.insert("nfr_nodes", ["id", "nu_f", "phase"]).build()

# Result:
# query = "INSERT INTO nfr_nodes (id, nu_f, phase) VALUES (?, ?, ?)"
# params = []
# Then bind values: cursor.execute(query, [123, 0.75, 1.57])
```

#### UPDATE Query

```python
builder = SecureQueryBuilder()
query, params = builder.update("nfr_nodes")\
    .set(nu_f=0.8, phase=1.57)\
    .where("id = ?", 123)\
    .build()

# Result:
# query = "UPDATE nfr_nodes SET nu_f = ?, phase = ? WHERE id = ?"
# params = [0.8, 1.57, 123]
```

#### DELETE Query

```python
builder = SecureQueryBuilder()
query, params = builder.delete("nfr_nodes")\
    .where("nu_f < ?", 0.1)\
    .build()

# Result:
# query = "DELETE FROM nfr_nodes WHERE nu_f < ?"
# params = [0.1]
```

### 2. Identifier Validation

Always validate table and column names to prevent SQL injection:

```python
from tnfr.security import validate_identifier, SQLInjectionError

# Valid identifiers
table = validate_identifier("nfr_nodes")  # OK
column = validate_identifier("nu_f")      # OK

# Invalid identifiers (raise SQLInjectionError)
try:
    validate_identifier("DROP")  # SQL keyword
except SQLInjectionError:
    pass

try:
    validate_identifier("table; DROP TABLE users;--")  # SQL injection attempt
except SQLInjectionError:
    pass
```

**Valid identifier rules:**
- Must contain only alphanumeric characters and underscores
- Must start with a letter or underscore
- Must be 1-64 characters long
- Cannot be a SQL keyword (unless `allow_keywords=True`)

### 3. String Input Sanitization

Sanitize string inputs before using them in queries:

```python
from tnfr.security import sanitize_string_input

# Valid strings
safe_string = sanitize_string_input("valid input")

# Reject strings that are too long
try:
    sanitize_string_input("a" * 10000, max_length=1000)
except SQLInjectionError:
    pass

# Reject strings with null bytes
try:
    sanitize_string_input("string\x00with null")
except SQLInjectionError:
    pass
```

**Important:** Sanitization is NOT a replacement for parameterized queries. Always use both.

### 4. Parameterized Query Execution

```python
from tnfr.security import execute_parameterized_query

# Safe parameterized query
execute_parameterized_query(
    "SELECT * FROM nfr_nodes WHERE nu_f > ? AND phase < ?",
    [0.5, 3.14]
)
```

**Note:** This is a demonstration function. In production, use your database library's parameterization:

```python
# SQLite example
import sqlite3
conn = sqlite3.connect("tnfr.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM nfr_nodes WHERE nu_f > ?", [0.5])

# PostgreSQL example (using psycopg2)
import psycopg2
cursor.execute("SELECT * FROM nfr_nodes WHERE nu_f > %s", [0.5])
```

## TNFR Structural Validation

### 1. Structural Frequency (νf)

```python
from tnfr.security import validate_structural_frequency

# Valid frequencies
nu_f = validate_structural_frequency(0.75)  # OK
nu_f = validate_structural_frequency(0.0)   # OK (silence operator)

# Invalid frequencies (raise ValueError)
try:
    validate_structural_frequency(-0.1)  # Negative
except ValueError:
    pass

try:
    validate_structural_frequency(float("nan"))  # NaN
except ValueError:
    pass
```

### 2. Phase (φ)

```python
from tnfr.security import validate_phase_value

# Valid phases
phase = validate_phase_value(1.57)  # π/2, OK
phase = validate_phase_value(7.0)   # Wraps to valid range [0, 2π]

# No wrapping (strict mode)
phase = validate_phase_value(1.57, allow_wrap=False)  # OK
try:
    validate_phase_value(7.0, allow_wrap=False)  # Out of range
except ValueError:
    pass
```

### 3. Coherence C(t)

```python
from tnfr.security import validate_coherence_value

# Valid coherence
coherence = validate_coherence_value(0.85)  # OK
coherence = validate_coherence_value(0.0)   # Minimum, OK

# Invalid (negative)
try:
    validate_coherence_value(-0.1)
except ValueError:
    pass
```

### 4. Sense Index (Si)

```python
from tnfr.security import validate_sense_index

# Valid sense index
si = validate_sense_index(0.7)   # OK
si = validate_sense_index(1.2)   # Can exceed 1.0 in high-coherence networks

# Invalid (negative)
try:
    validate_sense_index(-0.1)
except ValueError:
    pass
```

### 5. Complete Nodal Validation

```python
from tnfr.security import validate_nodal_input

# Validate complete node data
node_data = {
    "node_id": "nfr_001",
    "nu_f": 0.75,
    "phase": 1.57,
    "coherence": 0.85,
    "si": 0.9,
    "epi": [1.5, 2.3, 0.8],
}

validated = validate_nodal_input(node_data)
# All TNFR structural fields are validated
# Other fields are passed through unchanged
```

## Complete Workflow Example

```python
from tnfr.security import (
    SecureQueryBuilder,
    validate_identifier,
    validate_nodal_input,
)

def store_nfr_node(node_data: dict) -> None:
    """Store an NFR node in the database safely."""
    
    # Step 1: Validate TNFR structural data
    validated_data = validate_nodal_input(node_data)
    
    # Step 2: Validate database identifiers
    table = validate_identifier("nfr_nodes")
    
    # Step 3: Build parameterized query
    builder = SecureQueryBuilder()
    query, params = builder.insert(
        table,
        ["node_id", "nu_f", "phase", "coherence", "si"]
    ).build()
    
    # Step 4: Execute with parameters
    # cursor.execute(query, [
    #     validated_data["node_id"],
    #     validated_data["nu_f"],
    #     validated_data["phase"],
    #     validated_data["coherence"],
    #     validated_data["si"],
    # ])

def query_high_frequency_nodes(threshold: float) -> None:
    """Query NFR nodes with high structural frequency."""
    
    # Step 1: Validate inputs
    threshold = validate_structural_frequency(threshold)
    table = validate_identifier("nfr_nodes")
    
    # Step 2: Build secure query
    builder = SecureQueryBuilder()
    query, params = builder.select(table, ["node_id", "nu_f", "phase"])\
        .where("nu_f > ?", threshold)\
        .order_by("nu_f", "DESC")\
        .limit(100)\
        .build()
    
    # Step 3: Execute parameterized query
    # cursor.execute(query, params)
    # return cursor.fetchall()
```

## Security Best Practices

### DO ✓

1. **Always use parameterized queries**
   ```python
   builder.select("table").where("id = ?", user_input).build()
   ```

2. **Always validate identifiers**
   ```python
   table = validate_identifier(table_name)
   ```

3. **Validate TNFR structural data before persistence**
   ```python
   validated = validate_nodal_input(node_data)
   ```

4. **Use SecureQueryBuilder for query construction**
   ```python
   builder = SecureQueryBuilder()
   query, params = builder.select(...).build()
   ```

### DON'T ✗

1. **Never use string concatenation or f-strings**
   ```python
   # ❌ DANGEROUS!
   query = f"SELECT * FROM {table} WHERE id = {user_id}"
   ```

2. **Never trust user input**
   ```python
   # ❌ DANGEROUS!
   query = "SELECT * FROM " + user_provided_table
   ```

3. **Never skip identifier validation**
   ```python
   # ❌ DANGEROUS!
   query = f"SELECT * FROM {table_from_user}"
   ```

4. **Never use .format() or % for queries**
   ```python
   # ❌ DANGEROUS!
   query = "SELECT * FROM users WHERE id = {}".format(user_id)
   ```

## Testing

Comprehensive tests are provided in:
- `tests/unit/security/test_sql_injection_prevention.py` (34 tests)
- `tests/unit/security/test_validation.py` (32 tests)

Run tests:
```bash
pytest tests/unit/security/ -v
```

## TNFR Structural Fidelity

These security utilities maintain TNFR canonical invariants:

1. **EPI Coherence**: Validation ensures persisted data maintains structural coherence
2. **Structural Frequency (νf)**: Enforced as non-negative in Hz_str units
3. **Phase (φ)**: Validated and optionally wrapped to [0, 2π] range
4. **Operator Closure**: No new operators introduced, only data validation
5. **Domain Neutrality**: Security patterns are domain-agnostic

## Future Enhancements

If database functionality is added to TNFR:

1. **Database Abstraction Layer**: Create a `tnfr.database` module
2. **Connection Pooling**: Implement secure connection management
3. **Transaction Support**: Add transaction wrappers with proper error handling
4. **Migration Tools**: Provide schema migration utilities
5. **Query Logging**: Implement secure query logging for audit trails

## References

- OWASP SQL Injection Prevention Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
- TNFR Security Policy: `SECURITY.md`
- TNFR Architecture: `ARCHITECTURE.md`
- TNFR Glossary: `GLOSSARY.md`

---

**Last Updated**: November 2025  
**Module Version**: 1.0  
**Status**: Proactive Security Implementation
