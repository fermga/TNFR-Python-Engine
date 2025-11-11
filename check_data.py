import json

# Check i207 violations
viols207 = [json.loads(l) for l in open('u6_fine_i207.jsonl') if 'violate' in json.loads(l)['sequence_type']]
print('i207 violations:')
frags207 = sum(1 for r in viols207 if r.get('fragmentation', False))
print(f'Fragmented: {frags207}/{len(viols207)} = {frags207/len(viols207):.1%}')
print(f'Coherence min range: [{min(r["coherence_min"] for r in viols207):.3f}, {max(r["coherence_min"] for r in viols207):.3f}]')
print(f'Fragmented samples (C_min): {[round(r["coherence_min"], 3) for r in viols207 if r.get("fragmentation")]}')
print(f'Not fragmented samples (C_min): {[round(r["coherence_min"], 3) for r in viols207 if not r.get("fragmentation")]}')

# Check i212 violations
viols212 = [json.loads(l) for l in open('u6_hysteresis_down_i212.jsonl') if 'violate' in json.loads(l)['sequence_type']]
print('\ni212 violations:')
frags212 = sum(1 for r in viols212 if r.get('fragmentation', False))
print(f'Fragmented: {frags212}/{len(viols212)} = {frags212/len(viols212):.1%}')
print(f'Coherence min range: [{min(r["coherence_min"] for r in viols212):.3f}, {max(r["coherence_min"] for r in viols212):.3f}]')