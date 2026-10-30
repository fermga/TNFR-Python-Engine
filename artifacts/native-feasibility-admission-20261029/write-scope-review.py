"""Record an independent scope audit; no scientific evaluation or approval gate."""
from pathlib import Path
from tnfr.research.artifact_io import file_receipt, write_json_once

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def receipt(path):
    return file_receipt(path, label=path.relative_to(ROOT).as_posix())

drafts = ('admission-section.md', 'program-introduction.md', 'program-decision.md')
sources = (
    'AGENTS.md',
    'theory/RESEARCH_PROGRAM.md',
    'theory/research_prerequisites.json',
    'scripts/check_research_prerequisites.py',
    'theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md',
    'theory/PHYSICAL_REGIME_CORRESPONDENCES.md',
    'artifacts/native-causal-access-20261029/operator_causality-review.json',
    'artifacts/native-causal-access-20261029/state_observation-review.json',
    'artifacts/native-causal-access-20261029/organizational_identity-review.json',
    'artifacts/native-causal-access-20261029/research_selection-review.json',
)
payload = {
    'schema': 'tnfr.native-feasibility-admission-scope-review.v1',
    'verdict': 'PASS',
    'independent': True,
    'author': '/root',
    'reviewer': '/root/admission_guard',
    'reviewed_files': [receipt(OUT / name) for name in drafts],
    'consulted_sources': [receipt(ROOT / name) for name in sources],
    'contribution_disclosure': (
        'The reviewer previously advised the root on the distinction between investigating access '
        'and assuming it, the exact readiness schema, and the fact that the checker does not enforce '
        'semantic source budgets. The reviewer did not author or edit the three submitted drafts. '
        'This receipt independently assesses their concrete final formulation, not an actual physical realization.'
    ),
    'assessments': {
        'exact_question': (
            'The proposed task asks whether independently published apparatus descriptions warrant a later '
            'bridge review. It does not consume a physical preparation, actuator or instrument dictionary as '
            'already realized and does not apply the FO threshold to physical data. This is a substantive '
            'restriction rather than a feasibility label used to bypass the gate.'
        ),
        'joint_foundations': (
            'The original prepared single-bridge P1-P4 scopes remain intact. The complete native law is a '
            'well-defined conditional comparator; sufficient state and hidden-phase ambiguity identify access '
            'obligations; prepared organization is not promoted to formation; and matched-law alternatives '
            'remain available. No additional autonomous controller mechanism is assumed in order to ask '
            'whether a source documents one. A later prediction using such a mechanism requires its own admission.'
        ),
        'activity_and_guard': (
            'The physical-mapping activity marker is appropriate because external apparatus inspection is '
            'treated as physical work, even though only documentary feasibility is permitted. The draft '
            'requires a substantive final readiness review binding exact final program bytes and all four '
            'current prerequisite reviews, then a passing unchanged strict guard before external inspection. '
            'The checker cannot interpret the documentary limits; PROGRAM and the readiness assessment must '
            'remain the exact authority. Its generic ready message does not authorize broader physics.'
        ),
        'bounded_selection': (
            'The program fixes three ordered terrestrial apparatus families and at most one concrete '
            'apparatus per family, selected for documented access rather than waveform resemblance. '
            'Six primary documents and six targeted search queries are explicit ceilings. Separate '
            'supplements/methods/acquisition documents count separately; unavailable selected documents '
            'remain counted. Identity and selection reason are recorded before detailed assessment. '
            'Query results locate documents and are not capability evidence.'
        ),
        'allowed_evidence': (
            'Primary papers, methods and authoritative acquisition descriptions may be read for support, '
            'state/dynamics, preparation, actual capacity policy, hidden memory, reactions, readout, gain, '
            'clock and error evidence. Source-backed facts must remain separate from missing information, '
            'proposed interpretation and exact-law or controlled-reduction obligations. An extracted phase, '
            'oscillator label or measured frequency is not silently equated to a native coordinate.'
        ),
        'excluded_work': (
            'Raw signal archives, plot digitization, response extraction, model-output calculation, fitting, '
            'simulation, controller installation and apparatus construction are excluded. No physical '
            'variable identification, particle/field identity, empirical validation or FO classification '
            'is admitted. Existing source exposure is not reserved validation data. Programmed equation '
            'emulators cannot be used as independent evidence for a candidate natural dynamics.'
        ),
        'outcomes_and_uncertainty': (
            'The three permitted conclusions distinguish documented incompatibility with a specific '
            'requirement, insufficient/inaccessible/not-yet-comparable evidence, and a documented candidate '
            'warranting a separate bridge review. None admits a physical realization or changes experimental '
            'admission automatically. Missing documentation is not physical absence, and a bounded search '
            'cannot exclude a platform class universally. The native 2^-166 budget cannot be compared to '
            'published precision without independently justified units, gain and clock; it is sufficient, '
            'not a demonstrated necessary resolution.'
        ),
        'closure_and_followup': (
            'The audit stops when cases are classified or either source/query ceiling is reached, even if '
            'none survives. It retains one result in the physical-assessment owner and one source ledger, '
            'while PROGRAM remains the sole queue. At most one later bridge review or concrete dependency '
            'is recommended. A changed consumer, budget, mapping derivation, simulation or empirical test '
            'requires new exact-task admission and applicable prospective evaluation controls.'
        ),
        'readiness_boundary': (
            'This PASS is a review of proposed draft scope only. It is not a tnfr.physical-mapping-readiness.v1 '
            'record, does not alter the registry and does not authorize external source inspection before '
            'the required exact final readiness review and strict guard succeed.'
        ),
    },
    'blocking_findings': [],
    'performed': {
        'source_reading_and_scope_review': True,
        'artifact_bookkeeping_only': True,
        'external_source_searches': 0,
        'scientific_evaluations': 0,
        'simulations': 0,
        'tests_executed': 0,
        'historical_producer_replays': 0,
        'draft_edits': 0,
        'canonical_edits': 0,
        'checker_edits': 0,
        'readiness_records_written': 0,
    },
    'writer': receipt(Path(__file__).resolve()),
}
write_json_once(OUT / 'scope-review.json', payload)
print(receipt(OUT / 'scope-review.json'))
