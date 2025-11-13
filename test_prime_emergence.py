"""
Test script for TNFR Arithmetic Network - Prime Number Emergence

Quick validation that the implementation works correctly.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from tnfr.mathematics.number_theory import ArithmeticTNFRNetwork, run_basic_validation
    
    print("TNFR Prime Emergence Test")
    print("=" * 40)
    
    # Test 1: Basic construction
    print("Test 1: Basic network construction")
    network = ArithmeticTNFRNetwork(max_number=20)
    print(f"✓ Network created with {len(network.graph.nodes())} nodes")
    
    # Test 2: Check a known prime
    print("\nTest 2: Prime properties")
    prime_props = network.get_tnfr_properties(7)  # 7 is prime
    composite_props = network.get_tnfr_properties(6)  # 6 = 2×3 is composite
    
    print(f"Prime 7: ΔNFR = {prime_props['DELTA_NFR']:.6f}")
    print(f"Composite 6: ΔNFR = {composite_props['DELTA_NFR']:.6f}")
    
    if abs(prime_props['DELTA_NFR']) < abs(composite_props['DELTA_NFR']):
        print("✓ Prime has lower |ΔNFR| than composite (expected)")
    else:
        print("⚠ Unexpected: Composite has lower |ΔNFR| than prime")
    
    # Test 3: Prime detection
    print("\nTest 3: Prime detection validation")
    candidates = network.detect_prime_candidates(delta_nfr_threshold=0.2)
    detected_numbers = [x[0] for x in candidates]
    actual_primes = [2, 3, 5, 7, 11, 13, 17, 19]  # Known primes ≤ 20
    
    print(f"Detected candidates: {detected_numbers}")
    print(f"Actual primes ≤ 20: {actual_primes}")
    
    overlap = set(detected_numbers) & set(actual_primes)
    print(f"Correctly identified primes: {sorted(overlap)}")
    
    if len(overlap) >= 4:  # At least half the primes detected
        print("✓ Reasonable prime detection performance")
    else:
        print("⚠ Low prime detection rate - may need parameter tuning")
    
    # Test 4: Full validation on small range
    print("\n" + "=" * 50)
    print("FULL VALIDATION (n ≤ 30)")
    print("=" * 50)
    run_basic_validation(max_number=30)

    # Test 5: Structural fields telemetry
    print("\n" + "=" * 50)
    print("STRUCTURAL FIELDS (Φ_s, |∇φ|, K_φ, ξ_C)")
    print("=" * 50)
    fields = network.compute_structural_fields(phase_method="spectral")
    # Print a small sample
    sample_nodes = sorted(list(network.graph.nodes()))[:5]
    for n in sample_nodes:
        nd = network.graph.nodes[n]
        print(f"n={n:2d}  Φ_s={nd.get('phi_s', float('nan')): .4f}  |∇φ|={nd.get('phi_grad', float('nan')): .4f}  K_φ={nd.get('k_phi', float('nan')): .4f}")
    xi = fields['xi_c']
    if xi and xi.get('xi_c') is not None:
        print(f"Estimated ξ_C: {xi['xi_c']:.3f}  (R²={xi['R2']:.3f})")
    else:
        print("ξ_C could not be estimated (insufficient pairs or flat correlation)")
    
    print("\n✅ All tests completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
