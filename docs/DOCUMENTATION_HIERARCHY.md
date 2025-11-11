# Documentation Hierarchy - Visual Reference

**Mermaid diagrams showing TNFR documentation structure**

See [CANONICAL_SOURCES.md](CANONICAL_SOURCES.md) for complete hierarchy specification.

---

## Canonical Documentation Hierarchy

```mermaid
graph TB
    subgraph Tier1["Tier 1: Ultimate Sources<br/>(Physics & Philosophy)"]
        AGENTS[AGENTS.md<br/>Complete TNFR guide<br/>Operators, Grammar, Invariants]
        GRAMMAR[UNIFIED_GRAMMAR_RULES.md<br/>U1-U6 Mathematical derivations]
        GLOSSARY[GLOSSARY.md<br/>Term definitions<br/>Quick reference]
    end
    
    subgraph Tier2["Tier 2: Specialized Documentation<br/>(Single Responsibility)"]
        CONSTRAINTS[02-CANONICAL-CONSTRAINTS.md<br/>U1-U6 Technical specs]
        U6[U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md<br/>U6 Complete specification]
        ARCH[ARCHITECTURE.md<br/>System design]
        TEST[TESTING.md<br/>Test strategy]
        CONTRIB[CONTRIBUTING.md<br/>Development workflow]
    end
    
    subgraph Tier3["Tier 3: Supporting Documentation<br/>(References Only)"]
        README[README.md<br/>Project overview]
        INDEX[DOCUMENTATION_INDEX.md<br/>Navigation hub]
        CANONICAL[CANONICAL_SOURCES.md<br/>THIS DOCUMENT<br/>Hierarchy rules]
        GRAMMARINDEX[docs/grammar/README.md<br/>Grammar navigation]
    end
    
    %% References from Tier 2 to Tier 1
    CONSTRAINTS -.references.-> GRAMMAR
    CONSTRAINTS -.references.-> AGENTS
    U6 -.references.-> GRAMMAR
    ARCH -.references.-> AGENTS
    TEST -.references.-> AGENTS
    CONTRIB -.references.-> AGENTS
    
    %% References from Tier 3 to Tier 1 & 2
    README -.links to.-> AGENTS
    README -.links to.-> GLOSSARY
    INDEX -.maps.-> AGENTS
    INDEX -.maps.-> GRAMMAR
    INDEX -.maps.-> ARCH
    CANONICAL -.defines hierarchy for.-> AGENTS
    CANONICAL -.defines hierarchy for.-> GRAMMAR
    CANONICAL -.defines hierarchy for.-> CONSTRAINTS
    GRAMMARINDEX -.navigates to.-> CONSTRAINTS
    GRAMMARINDEX -.navigates to.-> U6
    
    style AGENTS fill:#e1f5e1
    style GRAMMAR fill:#e1f5e1
    style GLOSSARY fill:#e1f5e1
    style CONSTRAINTS fill:#fff4e1
    style U6 fill:#fff4e1
    style CANONICAL fill:#e1e8f5
```

## Information Flow

```mermaid
graph LR
    subgraph Discovery["User Discovery"]
        USER[User/Developer]
    end
    
    subgraph Navigation["Entry Points"]
        README[README.md]
        INDEX[DOCUMENTATION_INDEX.md]
        CANONICAL[CANONICAL_SOURCES.md]
    end
    
    subgraph Sources["Canonical Sources"]
        AGENTS[AGENTS.md]
        GRAMMAR[UNIFIED_GRAMMAR_RULES.md]
        GLOSSARY[GLOSSARY.md]
    end
    
    subgraph Specialized["Specialized Docs"]
        CONSTRAINTS[02-CANONICAL-CONSTRAINTS.md]
        ARCH[ARCHITECTURE.md]
        TEST[TESTING.md]
    end
    
    USER --> README
    USER --> INDEX
    README --> CANONICAL
    INDEX --> CANONICAL
    
    CANONICAL --> AGENTS
    CANONICAL --> GRAMMAR
    CANONICAL --> GLOSSARY
    
    AGENTS --> CONSTRAINTS
    AGENTS --> ARCH
    AGENTS --> TEST
    
    GRAMMAR --> CONSTRAINTS
    
    style USER fill:#f9f9f9
    style CANONICAL fill:#e1e8f5
    style AGENTS fill:#e1f5e1
    style GRAMMAR fill:#e1f5e1
    style GLOSSARY fill:#e1f5e1
```

## Concept Ownership Map

```mermaid
graph TD
    subgraph Concepts["TNFR Concepts"]
        NODAL[Nodal Equation<br/>∂EPI/∂t = νf · ΔNFR]
        OPS[13 Canonical Operators<br/>AL, EN, IL, OZ, UM, RA,<br/>SHA, VAL, NUL, THOL,<br/>ZHIR, NAV, REMESH]
        GRAMMARRULES[Grammar Rules U1-U6<br/>Derivations & Proofs]
        SPECS[U1-U6 Technical Specs<br/>Implementation details]
        INV[10 Canonical Invariants<br/>EPI, νf, ΔNFR, etc.]
        TERMS[Terminology<br/>EPI, νf, ΔNFR, C(t), Si, etc.]
    end
    
    subgraph Owners["Canonical Owners"]
        A1[AGENTS.md]
        A2[AGENTS.md]
        G1[UNIFIED_GRAMMAR_RULES.md]
        C1[02-CANONICAL-CONSTRAINTS.md]
        A3[AGENTS.md]
        GL[GLOSSARY.md]
    end
    
    NODAL --> A1
    OPS --> A2
    GRAMMARRULES --> G1
    SPECS --> C1
    INV --> A3
    TERMS --> GL
    
    style A1 fill:#e1f5e1
    style A2 fill:#e1f5e1
    style A3 fill:#e1f5e1
    style G1 fill:#e1f5e1
    style GL fill:#e1f5e1
    style C1 fill:#fff4e1
```

## Correct vs Incorrect Patterns

```mermaid
graph TB
    subgraph Correct["✅ Correct Pattern"]
        DOC1[Tutorial.md]
        REF1["See AGENTS.md § X<br/>for complete details"]
        CANON1[AGENTS.md<br/>Complete definition]
        
        DOC1 --> REF1
        REF1 -.references.-> CANON1
    end
    
    subgraph Incorrect["❌ Incorrect Pattern"]
        DOC2[Tutorial.md]
        COPY["Complete 500-line<br/>definition copied<br/>from AGENTS.md"]
        CANON2[AGENTS.md<br/>Original definition]
        
        DOC2 --> COPY
        COPY -.duplicates.-> CANON2
    end
    
    style DOC1 fill:#e1f5e1
    style REF1 fill:#e1f5e1
    style CANON1 fill:#e1f5e1
    style DOC2 fill:#ffe1e1
    style COPY fill:#ffe1e1
    style CANON2 fill:#ffe1e1
```

---

## Usage

**For contributors**: Check concept ownership map before adding definitions  
**For maintainers**: Use hierarchy diagram in PR reviews  
**For AI agents**: Follow information flow when answering questions

**Rule**: If adding >100 words about a concept, check if it's already canonical. If yes, reference instead of replicate.

---

**Related**:
- [CANONICAL_SOURCES.md](CANONICAL_SOURCES.md) - Complete hierarchy specification
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Full documentation map
- [AGENTS.md](AGENTS.md) - Primary canonical source

