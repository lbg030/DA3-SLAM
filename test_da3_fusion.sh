#!/bin/bash

# Test script for DA3-SLAM Fusion

echo "=========================================="
echo "DA3-SLAM Fusion Test Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python --version

# Check CUDA availability
echo ""
echo "[2/5] Checking CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None"

# Check module syntax
echo ""
echo "[3/5] Checking module syntax..."
python -m py_compile droid_slam/da3_fusion/*.py && echo "✓ All DA3 fusion modules OK"
python -m py_compile droid_slam/droid_frontend.py && echo "✓ droid_frontend.py OK"
python -m py_compile droid_slam/droid.py && echo "✓ droid.py OK"

# Test basic imports (without lietorch if not built)
echo ""
echo "[4/5] Testing basic imports..."
python << 'EOF'
import sys
sys.path.insert(0, 'droid_slam')

try:
    # Test individual components that don't need lietorch for import
    import torch
    print("✓ PyTorch import OK")

    import torch.nn.functional as F
    print("✓ torch.nn.functional import OK")

    # Check depth_anything_3
    try:
        from depth_anything_3.api import DepthAnything3
        print("✓ Depth Anything 3 import OK")
    except Exception as e:
        print(f"✗ Depth Anything 3 import failed: {e}")

    print("\nNote: Full module testing requires built lietorch and droid_backends")
    print("Run this after building:")
    print("  cd thirdparty/lietorch && python setup.py install")
    print("  cd ../.. && python setup.py install")

except Exception as e:
    print(f"✗ Import test failed: {e}")
    import traceback
    traceback.print_exc()
EOF

# Show usage
echo ""
echo "[5/5] Usage instructions:"
echo ""
echo "Without DA3 Fusion (baseline DROID-SLAM):"
echo "  python demo.py --imagedir /path/to/images --calib calib/replica.txt"
echo ""
echo "With DA3 Fusion (our method):"
echo "  python demo.py --imagedir /path/to/images --calib calib/replica.txt --use_da3_fusion"
echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
