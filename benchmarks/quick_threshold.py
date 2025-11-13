"""Quick safety threshold analysis."""
import json
import statistics

data = [json.loads(l) for l in open('benchmarks/results/alternative_metrics_results.jsonl')]

# Define high-stress threshold
threshold = 0.01

high_stress = [r for r in data if r['max_dnfr_final'] > threshold]
stable = [r for r in data if r['max_dnfr_final'] <= threshold]

print(f'High stress runs (max_ΔNFR > {threshold}): {len(high_stress)}')
print(f'Stable runs: {len(stable)}')

if high_stress and stable:
    grad_high = statistics.mean([r['grad_phi_final'] for r in high_stress])
    grad_stable = statistics.mean([r['grad_phi_final'] for r in stable])
    
    print(f'\n|∇φ| mean (high stress): {grad_high:.4f}')
    print(f'|∇φ| mean (stable):     {grad_stable:.4f}')
    print(f'Discrimination: {(grad_high/grad_stable-1)*100:.1f}% higher')
    
    # Suggested threshold: P95 of high-stress distribution
    grad_high_vals = sorted([r['grad_phi_final'] for r in high_stress])
    p50_high = grad_high_vals[len(grad_high_vals)//2]
    
    print(f'\nSuggested safety threshold: |∇φ| < {p50_high:.4f}')
    print(f'  (P50 of high-stress distribution)')
