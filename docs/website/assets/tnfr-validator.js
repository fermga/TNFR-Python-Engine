/**
 * TNFR Sequence Validator - Client-side validation logic
 * Port of Python validation rules for browser execution
 */

// Operator definitions (canonical names)
const OPERATORS = {
    EMISSION: 'emission',
    RECEPTION: 'reception',
    COHERENCE: 'coherence',
    DISSONANCE: 'dissonance',
    COUPLING: 'coupling',
    RESONANCE: 'resonance',
    SILENCE: 'silence',
    EXPANSION: 'expansion',
    CONTRACTION: 'contraction',
    SELF_ORGANIZATION: 'self_organization',
    MUTATION: 'mutation',
    TRANSITION: 'transition',
    RECURSIVITY: 'recursivity'
};

const VALID_START_OPERATORS = [OPERATORS.EMISSION, OPERATORS.RECEPTION];
const VALID_END_OPERATORS = [OPERATORS.SILENCE, OPERATORS.RESONANCE, OPERATORS.COHERENCE];
const DESTABILIZERS = [OPERATORS.DISSONANCE];
const TRANSFORMERS = [OPERATORS.MUTATION, OPERATORS.SELF_ORGANIZATION];

// Validation state
class SequenceValidator {
    constructor() {
        this.sequence = [];
        this.errors = [];
        this.warnings = [];
        this.patterns = [];
    }

    /**
     * Validate a sequence of operators
     * @param {string[]} operators - Array of operator names
     * @returns {Object} Validation result
     */
    validate(operators) {
        this.sequence = operators.map(op => op.toLowerCase().trim());
        this.errors = [];
        this.warnings = [];
        this.patterns = [];

        // Basic validation
        if (!this.validateBasicStructure()) {
            return this.buildResult(false);
        }

        // Grammar rules
        this.validateGrammar();

        // Pattern detection
        this.detectPatterns();

        // Calculate health metrics
        const healthMetrics = this.calculateHealthMetrics();

        return this.buildResult(this.errors.length === 0, healthMetrics);
    }

    validateBasicStructure() {
        if (this.sequence.length === 0) {
            this.errors.push('Sequence cannot be empty');
            return false;
        }

        // Check all operators are valid
        for (const op of this.sequence) {
            if (!Object.values(OPERATORS).includes(op)) {
                this.errors.push(`Unknown operator: ${op}`);
            }
        }

        if (this.errors.length > 0) {
            return false;
        }

        // Must start with valid operator
        if (!VALID_START_OPERATORS.includes(this.sequence[0])) {
            this.errors.push(`Sequence must start with ${VALID_START_OPERATORS.join(' or ')}`);
        }

        // Must end with valid operator
        const lastOp = this.sequence[this.sequence.length - 1];
        if (!VALID_END_OPERATORS.includes(lastOp)) {
            this.warnings.push(`Consider ending with ${VALID_END_OPERATORS.join(', ')}`);
        }

        return this.errors.length === 0;
    }

    validateGrammar() {
        // Check for mutation without prior dissonance
        let hasDissonance = false;
        for (let i = 0; i < this.sequence.length; i++) {
            const op = this.sequence[i];
            
            if (op === OPERATORS.DISSONANCE) {
                hasDissonance = true;
            }

            if (op === OPERATORS.MUTATION && !hasDissonance) {
                this.warnings.push('Mutation without prior dissonance (weak structural basis)');
            }

            // Check for self_organization closure
            if (op === OPERATORS.SELF_ORGANIZATION) {
                const hasCoherence = this.sequence.slice(i + 1).includes(OPERATORS.COHERENCE);
                const hasResonance = this.sequence.slice(i + 1).includes(OPERATORS.RESONANCE);
                if (!hasCoherence && !hasResonance) {
                    this.warnings.push('Self-organization should be followed by coherence or resonance');
                }
            }
        }
    }

    detectPatterns() {
        const seq = this.sequence.join(' ');

        // Basic patterns
        if (seq.includes('emission reception coherence')) {
            this.patterns.push({ type: 'BOOTSTRAP', confidence: 0.9 });
        }

        if (seq.includes('dissonance') && seq.includes('mutation')) {
            this.patterns.push({ type: 'BIFURCATED', confidence: 0.85 });
        }

        if (seq.includes('self_organization')) {
            this.patterns.push({ type: 'HIERARCHICAL', confidence: 0.8 });
        }

        if (seq.includes('resonance') && seq.includes('coupling')) {
            this.patterns.push({ type: 'RESONATE', confidence: 0.75 });
        }

        // Complex patterns
        if (seq.includes('dissonance') && seq.includes('coherence') && 
            seq.includes('self_organization')) {
            this.patterns.push({ type: 'THERAPEUTIC', confidence: 0.88 });
        }

        if (seq.includes('expansion') && seq.includes('transition') && 
            seq.includes('coherence')) {
            this.patterns.push({ type: 'EDUCATIONAL', confidence: 0.85 });
        }

        if (this.patterns.length === 0) {
            this.patterns.push({ type: 'CUSTOM', confidence: 0.5 });
        }
    }

    calculateHealthMetrics() {
        const length = this.sequence.length;
        
        // Coherence index (valid structure)
        const hasValidStart = VALID_START_OPERATORS.includes(this.sequence[0]) ? 1 : 0;
        const hasValidEnd = VALID_END_OPERATORS.includes(this.sequence[this.sequence.length - 1]) ? 1 : 0;
        const patternScore = this.patterns.length > 0 ? Math.max(...this.patterns.map(p => p.confidence)) : 0;
        const coherenceIndex = (hasValidStart + hasValidEnd + patternScore) / 3;

        // Balance score (stabilizers vs destabilizers)
        const stabilizers = this.sequence.filter(op => 
            [OPERATORS.COHERENCE, OPERATORS.SILENCE, OPERATORS.RESONANCE].includes(op)
        ).length;
        const destabilizers = this.sequence.filter(op => 
            DESTABILIZERS.includes(op)
        ).length;
        const balanceScore = length > 0 ? Math.min(1, stabilizers / Math.max(1, destabilizers + 1)) : 0;

        // Sustainability index
        const endsWithStabilizer = VALID_END_OPERATORS.includes(this.sequence[this.sequence.length - 1]);
        const sustainabilityIndex = endsWithStabilizer ? 0.8 + (balanceScore * 0.2) : balanceScore * 0.6;

        // Complexity efficiency
        const complexityEfficiency = length > 0 ? Math.min(1, patternScore / Math.log(length + 1)) : 0;

        // Overall health
        const overallHealth = (
            coherenceIndex * 0.35 +
            balanceScore * 0.25 +
            sustainabilityIndex * 0.25 +
            complexityEfficiency * 0.15
        );

        return {
            overallHealth: Math.round(overallHealth * 100) / 100,
            coherenceIndex: Math.round(coherenceIndex * 100) / 100,
            balanceScore: Math.round(balanceScore * 100) / 100,
            sustainabilityIndex: Math.round(sustainabilityIndex * 100) / 100,
            complexityEfficiency: Math.round(complexityEfficiency * 100) / 100
        };
    }

    buildResult(passed, healthMetrics = null) {
        return {
            passed,
            sequence: this.sequence,
            errors: this.errors,
            warnings: this.warnings,
            patterns: this.patterns,
            healthMetrics
        };
    }
}

// Global validator instance
const validator = new SequenceValidator();

/**
 * Validate sequence and display results
 * Called from HTML button onclick
 */
function validateSequence() {
    const input = document.getElementById('sequence-input');
    const resultDiv = document.getElementById('result-display');
    
    const sequence = input.value.trim().split(/\s+/);
    const result = validator.validate(sequence);
    
    // Build HTML for results
    let html = '<div class="validation-result">';
    
    if (result.passed) {
        html += '<div class="result-status success">✅ Valid Sequence</div>';
        resultDiv.className = 'result-display success';
    } else {
        html += '<div class="result-status error">❌ Invalid Sequence</div>';
        resultDiv.className = 'result-display error';
    }
    
    // Show errors
    if (result.errors.length > 0) {
        html += '<div class="result-section"><h4>Errors:</h4><ul>';
        result.errors.forEach(err => {
            html += `<li class="error-item">${err}</li>`;
        });
        html += '</ul></div>';
    }
    
    // Show warnings
    if (result.warnings.length > 0) {
        html += '<div class="result-section"><h4>Warnings:</h4><ul>';
        result.warnings.forEach(warn => {
            html += `<li class="warning-item">${warn}</li>`;
        });
        html += '</ul></div>';
        if (resultDiv.className === 'result-display success') {
            resultDiv.className = 'result-display warning';
        }
    }
    
    // Show detected patterns
    if (result.patterns.length > 0) {
        html += '<div class="result-section"><h4>Detected Patterns:</h4><ul>';
        result.patterns.forEach(pattern => {
            html += `<li class="pattern-item">${pattern.type} (confidence: ${Math.round(pattern.confidence * 100)}%)</li>`;
        });
        html += '</ul></div>';
    }
    
    // Show health metrics
    if (result.healthMetrics) {
        const hm = result.healthMetrics;
        html += '<div class="result-section"><h4>Health Metrics:</h4>';
        html += '<div class="health-grid">';
        html += `<div class="health-metric">
            <span class="metric-label">Overall Health:</span>
            <span class="metric-value ${getHealthClass(hm.overallHealth)}">${hm.overallHealth}</span>
        </div>`;
        html += `<div class="health-metric">
            <span class="metric-label">Coherence:</span>
            <span class="metric-value">${hm.coherenceIndex}</span>
        </div>`;
        html += `<div class="health-metric">
            <span class="metric-label">Balance:</span>
            <span class="metric-value">${hm.balanceScore}</span>
        </div>`;
        html += `<div class="health-metric">
            <span class="metric-label">Sustainability:</span>
            <span class="metric-value">${hm.sustainabilityIndex}</span>
        </div>`;
        html += '</div></div>';
    }
    
    html += '</div>';
    resultDiv.innerHTML = html;
}

/**
 * Get CSS class for health score
 */
function getHealthClass(score) {
    if (score >= 0.8) return 'health-excellent';
    if (score >= 0.6) return 'health-good';
    if (score >= 0.4) return 'health-fair';
    return 'health-poor';
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SequenceValidator, validateSequence };
}
