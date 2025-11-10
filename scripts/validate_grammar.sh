#!/bin/bash
# Complete grammar validation suite for TNFR
# Tests all U1-U4 constraints, patterns, and anti-patterns

set -e

echo "=============================================="
echo "  TNFR Grammar Validation Suite"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 1. Run unit tests
echo -e "${BLUE}1. Running unit tests...${NC}"
pytest tests/unit/operators/test_unified_grammar.py -v \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=term-missing \
    || { echo "Unit tests failed"; exit 1; }

# 2. Run integration tests
echo ""
echo -e "${BLUE}2. Running integration tests...${NC}"
pytest tests/integration/test_grammar_2_0_integration.py -v \
    || { echo "Integration tests failed"; exit 1; }

# 3. Run property tests
echo ""
echo -e "${BLUE}3. Running property tests...${NC}"
pytest tests/property/test_grammar_invariants.py -v \
    || { echo "Property tests failed"; exit 1; }

# 4. Run performance tests
echo ""
echo -e "${BLUE}4. Running performance benchmarks...${NC}"
pytest tests/performance/test_grammar_2_0_performance.py \
    --benchmark-only --benchmark-columns=mean,stddev \
    || { echo "Performance tests failed"; exit 1; }

# 5. Coverage report
echo ""
echo -e "${BLUE}5. Generating comprehensive coverage report...${NC}"
pytest tests/unit/operators/test_unified_grammar.py \
    --cov=tnfr.operators.unified_grammar \
    --cov-report=html:htmlcov/grammar \
    --cov-report=term \
    --cov-branch \
    --cov-fail-under=95 \
    --quiet \
    || { echo "Coverage below 95%"; exit 1; }

echo ""
echo -e "${GREEN}=============================================="
echo "  Validation Complete! ✓"
echo "=============================================="
echo -e "${NC}"
echo "Coverage report: htmlcov/grammar/index.html"
echo ""
echo "Summary:"
echo "  - Unit tests: PASSED"
echo "  - Integration tests: PASSED"
echo "  - Property tests: PASSED"
echo "  - Performance tests: PASSED"
echo "  - Coverage: >= 95%"
echo ""
