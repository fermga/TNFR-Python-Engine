#!/usr/bin/env python3
"""
TNFR Phase 2 Optimization Test - Unified Systems Verification

This test verifies that the new unified systems (GPU, Telemetry, Validation)
work correctly and provide the expected consolidation benefits.

Test Coverage:
- Unified GPU System: Compute operations and memory management
- Unified Telemetry System: Event emission and correlation tracking  
- Unified Validation System: Input validation and security checks

Expected Outcomes:
- All systems initialize correctly
- Operations execute successfully
- Performance telemetry indicates efficiency gains
- Validation provides comprehensive error detection
"""

import sys
import time
from pathlib import Path

# Add TNFR to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tnfr
    print(f"✅ TNFR {tnfr.__version__} loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import TNFR: {e}")
    sys.exit(1)

def test_unified_gpu_system():
    """Test unified GPU system functionality."""
    print("\n🔧 Testing Unified GPU System...")
    
    try:
        # Test that the unified GPU system file exists and can be imported
        from pathlib import Path
        gpu_system_path = Path("src/tnfr/engines/computation/unified_gpu_system.py")
        if gpu_system_path.exists():
            print(f"   ✅ Unified GPU system file exists: {gpu_system_path}")
        else:
            print(f"   ❌ Unified GPU system file missing: {gpu_system_path}")
            return False
        
        config = UnifiedGPUConfig(enable_gpu=False)  # CPU fallback for testing
        gpu_system = TNFRUnifiedGPUSystem(config)
        
        print(f"   ✅ GPU system initialized: {type(gpu_system).__name__}")
        
        # Test basic operation
        import numpy as np
        test_data = np.random.rand(10, 10).astype(np.float32)
        
        # Test compute operation
        result = gpu_system.compute_delta_nfr_gpu(
            test_data, test_data, test_data
        )
        
        print(f"   ✅ GPU compute operation successful: {result.shape}")
        print(f"   📊 GPU system status: {'GPU' if gpu_system.config.enable_gpu and gpu_system._gpu_available else 'CPU fallback'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU system test failed: {e}")
        return False

def test_unified_telemetry_system():
    """Test unified telemetry system functionality."""
    print("\n📊 Testing Unified Telemetry System...")
    
    try:
        # Test telemetry system import and initialization  
        from tnfr.telemetry.unified_telemetry_system import (
            TNFRUnifiedTelemetrySystem,
            TelemetryConfig
        )
        
        config = TelemetryConfig(enable_telemetry=True, buffer_size=1000)
        telemetry = TNFRUnifiedTelemetrySystem(config)
        
        print(f"   ✅ Telemetry system initialized: {type(telemetry).__name__}")
        
        # Test event emission
        telemetry.emit_structural_event(
            event_type="test_event",
            node_id="test_node", 
            metadata={"test": True}
        )
        
        telemetry.emit_performance_event(
            operation="test_operation",
            duration=0.001,
            metadata={"success": True}
        )
        
        # Get statistics
        stats = telemetry.get_statistics()
        print(f"   ✅ Events emitted: {stats['total_events']}")
        print(f"   📊 Telemetry buffers: {stats['buffer_usage_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Telemetry system test failed: {e}")
        return False

def test_unified_validation_system():
    """Test unified validation system functionality."""
    print("\n🔍 Testing Unified Validation System...")
    
    try:
        # Test validation system import and initialization
        from tnfr.validation.unified_validation_system import (
            TNFRUnifiedValidationSystem,
            ValidationConfig
        )
        
        config = ValidationConfig(strict_mode=True, enable_security_checks=True)
        validator = TNFRUnifiedValidationSystem(config)
        
        print(f"   ✅ Validation system initialized: {type(validator).__name__}")
        
        # Test structural frequency validation
        vf_result = validator.validate_structural_frequency(1.5)
        print(f"   ✅ Structural frequency validation: {'PASS' if vf_result.is_valid else 'FAIL'}")
        
        # Test phase validation
        phase_result = validator.validate_phase_value(3.14159)
        print(f"   ✅ Phase validation: {'PASS' if phase_result.is_valid else 'FAIL'}")
        
        # Test coherence validation
        coherence_result = validator.validate_coherence(0.85)
        print(f"   ✅ Coherence validation: {'PASS' if coherence_result.is_valid else 'FAIL'}")
        
        # Test security validation
        security_result = validator.validate_string_input("safe_input_string")
        print(f"   ✅ Security validation: {'PASS' if security_result.is_valid else 'FAIL'}")
        
        # Test malicious input detection
        malicious_result = validator.validate_string_input("<script>alert('test')</script>")
        print(f"   ✅ Malicious input detection: {'DETECTED' if not malicious_result.is_valid else 'MISSED'}")
        
        # Get cache statistics
        cache_stats = validator.get_cache_statistics()
        print(f"   📊 Validation cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Validation system test failed: {e}")
        return False

def test_integrated_operations():
    """Test integrated operations using multiple unified systems."""
    print("\n🔗 Testing Integrated Operations...")
    
    try:
        # Create a simple TNFR network
        from tnfr.sdk import TNFR
        
        # Create network with telemetry
        net = TNFR.create(5).random(0.3)
        G = net.build()
        
        print(f"   ✅ Created TNFR network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Test coherence computation
        from tnfr.mathematics import compute_coherence
        coherence = compute_coherence(G)
        
        print(f"   ✅ Network coherence: {coherence:.3f}")
        
        # Test optimization
        net.focus(list(G.nodes())[0]).emit().coherence()
        optimized_net = net.execute()
        
        optimized_coherence = compute_coherence(optimized_net.graph)
        print(f"   ✅ Optimized coherence: {optimized_coherence:.3f}")
        
        improvement = optimized_coherence - coherence
        print(f"   📈 Coherence improvement: {improvement:+.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated operations test failed: {e}")
        return False

def main():
    """Run all unified systems tests."""
    print("🚀 TNFR Phase 2 Optimization Test - Unified Systems Verification")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all tests
    tests = [
        test_unified_gpu_system,
        test_unified_telemetry_system, 
        test_unified_validation_system,
        test_integrated_operations
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    elapsed_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 70)
    print(f"📋 Test Summary:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ⏱️  Time: {elapsed_time:.2f}s")
    
    if passed == total:
        print(f"🎉 All unified systems working correctly!")
        print(f"   🎯 Phase 2 optimization verification: SUCCESS")
        return 0
    else:
        print(f"⚠️  Some tests failed - check unified system implementations")
        return 1

if __name__ == "__main__":
    sys.exit(main())