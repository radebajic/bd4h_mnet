#!/usr/bin/env python
"""
Quick test to verify Full3D VMamba implementation works.
"""
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_module():
    """Test CB3dFull3DVMamba block."""
    print("="*60)
    print("Testing CB3dFull3DVMamba...")
    print("="*60)
    
    from nnunet.network_architecture.reproduction_mnet.basic_module_full3d import (
        CB3dFull3DVMamba, VMAMBA_3D_AVAILABLE
    )
    
    print(f"VMamba 3D backend available: {VMAMBA_3D_AVAILABLE}")
    
    # Test with small volume
    block = CB3dFull3DVMamba(
        in_channels=32,
        out_channels=32,
        hidden_ratio=1.0,
        d_state=8,  # Smaller for testing
        dropout=0.0,
        use_se=True,
        se_reduction=4
    )
    
    x = torch.randn(1, 32, 8, 16, 16)  # Small test volume
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = block(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
    print("✓ CB3dFull3DVMamba test passed!\n")
    
    return True

def test_mnet_full3d():
    """Test MNetFull3D network."""
    print("="*60)
    print("Testing MNetFull3D...")
    print("="*60)
    
    from nnunet.network_architecture.reproduction_mnet.mnet_full3d import MNetFull3D
    
    # Test without VMamba (baseline)
    print("\n1. Testing baseline (no VMamba)...")
    net_baseline = MNetFull3D(
        in_channels=1,
        num_classes=3,
        kn=(8, 12, 16, 20, 24),  # Smaller for testing
        ds=True,
        FMU='sub',
        vm_bottleneck_stages=[],  # No VMamba
    )
    
    x = torch.randn(1, 1, 8, 32, 32)  # Small test volume
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y_baseline = net_baseline(x)
    
    if isinstance(y_baseline, tuple):
        print(f"Output shapes (deep supervision): {[yi.shape for yi in y_baseline]}")
    else:
        print(f"Output shape: {y_baseline.shape}")
    
    baseline_params = sum(p.numel() for p in net_baseline.parameters())
    print(f"Baseline params: {baseline_params:,}")
    
    # Test with VMamba in bottleneck
    print("\n2. Testing with Full3D VMamba in bottleneck 5...")
    net_vmamba = MNetFull3D(
        in_channels=1,
        num_classes=3,
        kn=(8, 12, 16, 20, 24),
        ds=True,
        FMU='sub',
        vm_bottleneck_stages=[5],  # Enable VMamba in bottleneck5
        vmamba_hidden_ratio=1.0,
        vmamba_d_state=8,
        vmamba_dropout=0.0,
        vmamba_use_se=True,
        vmamba_se_reduction=4,
    )
    
    with torch.no_grad():
        y_vmamba = net_vmamba(x)
    
    if isinstance(y_vmamba, tuple):
        print(f"Output shapes (deep supervision): {[yi.shape for yi in y_vmamba]}")
    else:
        print(f"Output shape: {y_vmamba.shape}")
    
    vmamba_params = sum(p.numel() for p in net_vmamba.parameters())
    print(f"VMamba params: {vmamba_params:,}")
    print(f"Param increase: {vmamba_params - baseline_params:,} ({(vmamba_params/baseline_params - 1)*100:.2f}%)")
    
    print("✓ MNetFull3D test passed!\n")
    
    return True

def test_trainer():
    """Test that trainer can be imported."""
    print("="*60)
    print("Testing Full3D Trainer...")
    print("="*60)
    
    from nnunet.training.network_training.myTrainer_reproduction_WandB_Full3D import (
        myTrainer_reproduction_WandB_Full3D
    )
    
    print(f"Trainer class: {myTrainer_reproduction_WandB_Full3D.__name__}")
    print(f"Parent class: {myTrainer_reproduction_WandB_Full3D.__bases__[0].__name__}")
    print("✓ Trainer import successful!\n")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Full3D VMamba Implementation Test Suite")
    print("="*60 + "\n")
    
    try:
        test_basic_module()
        test_mnet_full3d()
        test_trainer()
        
        print("="*60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*60)
        print("\nFull3D VMamba implementation is ready to use!")
        print("\nTo train, run:")
        print("  python hydra_trainer.py --config-name=best_arch_3_full3d")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())








