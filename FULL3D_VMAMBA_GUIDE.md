# Full 3D VMamba Implementation Guide

## Overview

This implementation adds **Full 3D VMamba** support to MNet while keeping all existing functionality intact. The key difference:

| Approach | Sequence Length | Context | Memory |
|----------|----------------|---------|---------|
| **Original (Z-axis)** | L = D | Slice-to-slice only | Lower |
| **Full 3D (NEW)** | L = D × H × W | Full volumetric | Higher |

## What Was Created

### 1. Core Modules

#### `basic_module_full3d.py`
- **CB3dFull3DVMamba**: Drop-in replacement for CB3d/CBzMamba
- Flattens entire 3D volume to sequence L = D×H×W
- Uses VMambaBlock3D from vmamba_tri_plane.py
- Architecture: `1×1×1 Conv → VMamba3D → residual connection`

#### `mnet_full3d.py`
- **MNetFull3D**: Full network using CB3dFull3DVMamba
- Identical structure to MNet, only block type changes
- All other features preserved (deep supervision, FMU, gated fusion, etc.)

### 2. Training Infrastructure

#### `myTrainer_reproduction_WandB_Full3D.py`
- **Minimal extension** of myTrainer_reproduction_WandB
- Only overrides `initialize_network()` to use MNetFull3D
- Adds 5 new parameters (all other settings inherited):
  - `vmamba_hidden_ratio` (default: 1.0)
  - `vmamba_d_state` (default: 16)
  - `vmamba_dropout` (default: 0.0)
  - `vmamba_use_se` (default: True)
  - `vmamba_se_reduction` (default: 8)

#### `best_arch_3_full3d.yaml`
- **Identical** to best_arch_3.yaml except:
  - Uses `myTrainer_reproduction_WandB_Full3D` trainer
  - Adds Full3D VMamba parameters under `vmamba:` section
  - Updated W&B run names for tracking

### 3. Integration

#### Updated `hydra_trainer.py`
- Added 5 lines (189-194) to handle Full3D VMamba parameters
- No other changes to existing functionality
- Backward compatible with all existing configs

#### Updated `__init__.py`
- Added exports for MNetFull3D and CB3dFull3DVMamba
- All existing imports unchanged

## Configuration

### Full3D VMamba Parameters (NEW)

```yaml
vmamba:
  # Stage selection (SAME as original)
  down_stages: [3]
  up_stages: [2]
  bottleneck_stages: [4,5]
  
  # Full3D-specific (NEW - only additions)
  hidden_ratio: 1.0        # Hidden dimension ratio (1.0 = no reduction)
  d_state: 16              # SSM state dimension (controls state richness)
  dropout: 0.0             # Dropout rate for regularization
  use_se: true             # Squeeze-excitation attention
  se_reduction: 8          # SE reduction ratio
```

### What Each Parameter Does

- **hidden_ratio**: Controls VMamba internal dimension
  - 1.0 = no compression (more capacity)
  - 0.5 = half internal dimension (less memory)
  
- **d_state**: SSM state size
  - Higher = richer state representation
  - Typical: 8-32 for medical imaging
  
- **dropout**: Regularization
  - 0.0 = no dropout
  - 0.1 = 10% dropout (helps prevent overfitting)
  
- **use_se**: Whether to use channel attention
  - true = adds SE blocks (better feature selection)
  - false = skip SE (faster)
  
- **se_reduction**: SE compression ratio
  - Higher = more compression (faster, less expressive)
  - Lower = less compression (more parameters)

## Usage

### 1. Test Installation

```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1
python test_full3d_vmamba.py
```

This will verify:
- CB3dFull3DVMamba works correctly
- MNetFull3D initializes and runs
- Trainer can be imported

### 2. Train with Full3D VMamba

```bash
# Using the pre-configured Full3D config
python hydra_trainer.py --config-name=best_arch_3_full3d

# Or create your own by copying best_arch_3_full3d.yaml
```

### 3. Compare to Baseline

To compare Full3D vs Z-axis VMamba:

```bash
# Original Z-axis VMamba
python hydra_trainer.py --config-name=best_arch_3

# Full3D VMamba (new)
python hydra_trainer.py --config-name=best_arch_3_full3d
```

Both use the same stage selection `[4,5]`, so differences are purely architectural.

## Memory Considerations

Full3D VMamba has **higher memory** requirements because:
- Sequence length: L = D×H×W (vs L = D for Z-axis)
- Example: 16×64×64 = 65,536 tokens (vs 16 tokens)

**Mitigation strategies:**

1. **Reduce hidden_ratio**:
   ```yaml
   vmamba:
     hidden_ratio: 0.5  # Instead of 1.0
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   trainer:
     use_checkpoint: true
   ```

3. **Reduce batch size** if OOM occurs

4. **Use fewer stages**:
   ```yaml
   vmamba:
     bottleneck_stages: [5]  # Only deepest bottleneck
   ```

## When to Use Full3D vs Z-axis

### Use Full3D VMamba when:
- ✓ Dataset has isotropic or near-isotropic spacing
- ✓ Full 3D context is important (organs with complex 3D structure)
- ✓ You have sufficient GPU memory
- ✓ Training time is not critical

### Use Z-axis VMamba when:
- ✓ Dataset has anisotropic spacing (common in CT/MRI)
- ✓ Slice-to-slice dependencies are the main concern
- ✓ Memory is limited
- ✓ Faster training is needed

## Implementation Details

### Architecture Comparison

**Z-axis VMamba (original):**
```
x (B,C,D,H,W) → reshape → (B×H×W, D, C) → Mamba1D → reshape → (B,C,D,H,W)
```

**Full3D VMamba (new):**
```
x (B,C,D,H,W) → flatten → (B, D×H×W, C) → SelectiveScan3D (8 directions) → (B,C,D,H,W)
```

Full3D uses **8 scanning directions** (all cube corners) vs 2 directions for Z-axis.

### Code Changes Summary

**New files created:**
- `basic_module_full3d.py` (153 lines)
- `mnet_full3d.py` (609 lines)
- `myTrainer_reproduction_WandB_Full3D.py` (157 lines)
- `best_arch_3_full3d.yaml` (56 lines)
- `test_full3d_vmamba.py` (182 lines)
- `FULL3D_VMAMBA_GUIDE.md` (this file)

**Modified files:**
- `hydra_trainer.py`: +5 lines (189-194)
- `__init__.py`: +2 lines

**Total new code:** ~1,200 lines
**Modified existing code:** 7 lines

## Example Training Command

```bash
# Full training with Full3D VMamba on Task024
python hydra_trainer.py \
  --config-name=best_arch_3_full3d \
  trainer.task=Task024_Promise \
  trainer.fold=0 \
  trainer.max_epochs=150

# Override Full3D parameters
python hydra_trainer.py \
  --config-name=best_arch_3_full3d \
  trainer.vmamba.hidden_ratio=0.5 \
  trainer.vmamba.d_state=8 \
  trainer.vmamba.dropout=0.1
```

## Troubleshooting

### "VMambaBlock3D not available"
- Check that `vmamba_tri_plane.py` is in the correct location
- Ensure no import errors in that file

### Out of Memory (OOM)
- Reduce `hidden_ratio` to 0.5
- Enable `use_checkpoint: true`
- Use only bottleneck stage: `bottleneck_stages: [5]`
- Reduce batch size in plans file

### "mamba-ssm not found"
- Full3D implementation has a **fallback** to standard convolutions
- It will still run but without selective scan
- Warning will be printed at initialization

## Next Steps

1. **Test on small dataset** to verify memory requirements
2. **Compare performance** to Z-axis VMamba baseline
3. **Tune Full3D parameters** based on your data characteristics
4. **Monitor W&B** for training curves and validation metrics

## Questions?

This implementation:
- ✓ Keeps ALL original functionality
- ✓ Only adds necessary Full3D parameters
- ✓ Is backward compatible
- ✓ Has minimal code changes
- ✓ Includes comprehensive testing

For issues, check:
1. Test script passes: `python test_full3d_vmamba.py`
2. Config syntax: `trainer.trainer_name: myTrainer_reproduction_WandB_Full3D`
3. VMamba parameters under `vmamba:` section






