import json
from pathlib import Path
from collections import defaultdict

# Check hierarchical topologies
BASE = Path('benchmarks') / 'results'
data = [json.loads(line) for line in open(BASE / 'u6_hierarchical_i209.jsonl')]
print(f'Total records: {len(data)}')

# Group by topology
by_topo = defaultdict(list)
for r in data:
    by_topo[r['topology']].append(r)

for topo, records in by_topo.items():
    print(f'\n{topo}:')
    viols = [r for r in records if 'violate' in r['sequence_type']]
    print(f'  Violations: {len(viols)}')
    if viols:
        frags = sum(1 for r in viols if r.get('fragmentation', False))
        print(f'  Fragmented: {frags}/{len(viols)} = {frags / len(viols):.1%}')
        print(f'  C_min range: [{min(r["coherence_min"] for r in viols):.3f}, {max(r["coherence_min"] for r in viols):.3f}]')
        print(f'  Phi_s drift: {[round(r["phi_s_mean_final"] - r["phi_s_mean_initial"], 2) for r in viols[:3]]}')
