/**
 * TNFR Interactive Tutorial System
 * Provides step-by-step learning experience
 */

class TNFRTutorial {
    constructor() {
        this.currentStep = 0;
        this.userProgress = {
            validatedSequence: null,
            healthAnalyzed: false,
            patternExplored: false,
            toolsUsed: false
        };
        this.steps = [
            {
                id: 'welcome',
                title: 'Welcome to TNFR Grammar 2.0',
                description: 'Learn the fundamentals of structural operators'
            },
            {
                id: 'basic-validation',
                title: 'Step 1: Basic Sequence Validation',
                description: 'Validate your first operator sequence'
            },
            {
                id: 'health-analysis',
                title: 'Step 2: Health Metrics Analysis',
                description: 'Understand quantitative quality assessment'
            },
            {
                id: 'pattern-exploration',
                title: 'Step 3: Pattern Exploration',
                description: 'Discover structural patterns in sequences'
            },
            {
                id: 'domain-applications',
                title: 'Step 4: Domain Applications',
                description: 'Apply TNFR to real-world scenarios'
            },
            {
                id: 'advanced-tools',
                title: 'Step 5: Advanced Tools',
                description: 'Use sequence generator and optimizer'
            }
        ];
    }

    start() {
        this.currentStep = 0;
        this.showStep();
    }

    showStep() {
        const step = this.steps[this.currentStep];
        console.log(`Tutorial Step ${this.currentStep + 1}/${this.steps.length}: ${step.title}`);
        // In a full implementation, this would update the UI
    }

    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.showStep();
            return true;
        }
        return false;
    }

    previousStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.showStep();
            return true;
        }
        return false;
    }

    completeStep(stepId, data) {
        switch (stepId) {
            case 'basic-validation':
                this.userProgress.validatedSequence = data.sequence;
                break;
            case 'health-analysis':
                this.userProgress.healthAnalyzed = true;
                break;
            case 'pattern-exploration':
                this.userProgress.patternExplored = true;
                break;
            case 'advanced-tools':
                this.userProgress.toolsUsed = true;
                break;
        }
    }

    getProgress() {
        const completed = Object.values(this.userProgress).filter(v => v === true || v !== null).length;
        return {
            current: this.currentStep + 1,
            total: this.steps.length,
            percentComplete: Math.round((completed / 4) * 100)
        };
    }
}

/**
 * Pattern Example Generator
 */
class PatternExamples {
    static examples = {
        BOOTSTRAP: {
            sequence: ['emission', 'reception', 'coherence', 'silence'],
            description: 'Basic initialization pattern',
            useCase: 'Starting new nodes or processes'
        },
        THERAPEUTIC: {
            sequence: ['emission', 'reception', 'coherence', 'dissonance', 'self_organization', 'coherence', 'silence'],
            description: 'Complete healing cycle',
            useCase: 'Personal transformation and therapeutic processes'
        },
        EDUCATIONAL: {
            sequence: ['emission', 'reception', 'expansion', 'transition', 'coherence', 'resonance', 'silence'],
            description: 'Transformative learning pattern',
            useCase: 'Educational programs and skill development'
        },
        ORGANIZATIONAL: {
            sequence: ['emission', 'coupling', 'expansion', 'self_organization', 'coherence', 'resonance', 'silence'],
            description: 'Institutional evolution pattern',
            useCase: 'Organizational change and team development'
        },
        RESONATE: {
            sequence: ['resonance', 'coupling', 'resonance', 'silence'],
            description: 'Amplification through coupling',
            useCase: 'Spreading patterns through network'
        }
    };

    static getExample(patternType) {
        return this.examples[patternType] || null;
    }

    static getAllPatterns() {
        return Object.keys(this.examples);
    }

    static getRandomExample() {
        const patterns = this.getAllPatterns();
        const randomPattern = patterns[Math.floor(Math.random() * patterns.length)];
        return {
            type: randomPattern,
            ...this.examples[randomPattern]
        };
    }
}

/**
 * Health Metrics Explainer
 */
class HealthMetricsExplainer {
    static explainMetric(metricName, value) {
        const explanations = {
            overallHealth: {
                name: 'Overall Health',
                description: 'Composite score of sequence quality',
                interpretation: value >= 0.8 ? 'Excellent' : 
                               value >= 0.6 ? 'Good' : 
                               value >= 0.4 ? 'Fair' : 'Poor',
                recommendation: value < 0.6 ? 
                    'Consider adding stabilizing operators (coherence, silence) at the end' : 
                    'Sequence is well-structured'
            },
            coherenceIndex: {
                name: 'Coherence Index',
                description: 'Global sequential flow quality',
                interpretation: value >= 0.7 ? 'Clear pattern structure' : 
                               value >= 0.5 ? 'Recognizable structure' : 
                               'Unclear structure',
                recommendation: value < 0.7 ? 
                    'Ensure sequence starts with emission/reception and ends with valid terminator' : 
                    'Good structural flow'
            },
            balanceScore: {
                name: 'Balance Score',
                description: 'Equilibrium between stabilizers and destabilizers',
                interpretation: value >= 0.7 ? 'Well-balanced' : 
                               value >= 0.4 ? 'Moderately balanced' : 
                               'Imbalanced',
                recommendation: value < 0.5 ? 
                    'Add more stabilizing operators (coherence, resonance) to balance dissonance' : 
                    'Good force equilibrium'
            },
            sustainabilityIndex: {
                name: 'Sustainability Index',
                description: 'Long-term viability of the pattern',
                interpretation: value >= 0.7 ? 'Highly sustainable' : 
                               value >= 0.5 ? 'Moderately sustainable' : 
                               'Low sustainability',
                recommendation: value < 0.7 ? 
                    'End sequence with silence, resonance, or coherence for better sustainability' : 
                    'Pattern has good long-term stability'
            },
            complexityEfficiency: {
                name: 'Complexity Efficiency',
                description: 'Pattern quality relative to sequence length',
                interpretation: value >= 0.7 ? 'Efficient' : 
                               value >= 0.5 ? 'Moderately efficient' : 
                               'Inefficient',
                recommendation: value < 0.5 ? 
                    'Consider simplifying or using more effective operator combinations' : 
                    'Good complexity-to-quality ratio'
            }
        };

        return explanations[metricName] || {
            name: metricName,
            description: 'Unknown metric',
            interpretation: 'N/A',
            recommendation: 'N/A'
        };
    }

    static explainAll(healthMetrics) {
        const explanations = {};
        for (const [key, value] of Object.entries(healthMetrics)) {
            explanations[key] = this.explainMetric(key, value);
        }
        return explanations;
    }
}

/**
 * Sequence Comparison Tool
 */
class SequenceComparator {
    static compare(sequence1, sequence2) {
        const validator = new SequenceValidator();
        
        const result1 = validator.validate(sequence1);
        const result2 = validator.validate(sequence2);
        
        return {
            sequence1: {
                operators: sequence1,
                valid: result1.passed,
                health: result1.healthMetrics?.overallHealth || 0,
                patterns: result1.patterns
            },
            sequence2: {
                operators: sequence2,
                valid: result2.passed,
                health: result2.healthMetrics?.overallHealth || 0,
                patterns: result2.patterns
            },
            winner: result1.healthMetrics?.overallHealth > result2.healthMetrics?.overallHealth ? 
                'sequence1' : 'sequence2',
            healthDifference: Math.abs(
                (result1.healthMetrics?.overallHealth || 0) - 
                (result2.healthMetrics?.overallHealth || 0)
            )
        };
    }
}

/**
 * Interactive demonstrations
 */
function demonstratePattern(patternType) {
    const example = PatternExamples.getExample(patternType);
    if (!example) {
        console.error(`Unknown pattern: ${patternType}`);
        return null;
    }

    const validator = new SequenceValidator();
    const result = validator.validate(example.sequence);

    return {
        pattern: patternType,
        sequence: example.sequence,
        description: example.description,
        useCase: example.useCase,
        validation: result
    };
}

function explainHealthDashboard(healthMetrics) {
    const explanations = HealthMetricsExplainer.explainAll(healthMetrics);
    
    console.log('=== Health Metrics Dashboard ===');
    for (const [metric, data] of Object.entries(explanations)) {
        const value = healthMetrics[metric];
        console.log(`\n${data.name}: ${value}`);
        console.log(`  ${data.description}`);
        console.log(`  Interpretation: ${data.interpretation}`);
        console.log(`  Recommendation: ${data.recommendation}`);
    }
}

// Initialize global tutorial instance
const tutorial = new TNFRTutorial();

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        TNFRTutorial,
        PatternExamples,
        HealthMetricsExplainer,
        SequenceComparator,
        demonstratePattern,
        explainHealthDashboard
    };
}
