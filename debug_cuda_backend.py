
import os
import sys

# Simulate the dashboard environment setup
os.environ["TNFR_MATH_BACKEND"] = "torch"
os.environ["TNFR_CUDA_ENABLED"] = "true"

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    from tnfr.mathematics.backend import get_backend
    
    backend = get_backend()
    print(f"Backend Name: {backend.name}")
    print(f"Backend Device: {backend.get_device_name()}")
    print(f"Backend Info: {backend.get_backend_info()}")
    
    from tnfr.engines.computation.unified_gpu_system import TNFRUnifiedGPUSystem
    gpu_sys = TNFRUnifiedGPUSystem()
    print(f"GPU System Available: {gpu_sys.is_available}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
