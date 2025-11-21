# nnUNet Results Folder Structure - CORRECTED

## Overview
Each model+spacing combination now has a **unique plans_identifier**, ensuring all 12 configurations save to separate folders in `nnUNet_results`.

## Folder Naming Convention
```
{trainer_name}__{plans_identifier}/fold_{fold}/
```

- **trainer_name**: Same for all runs (`myTrainer_reproduction_WandB`)
- **plans_identifier**: Unique per model+spacing (contains model suffix)

## All 12 Output Folders

### Baseline Model (No Gating, No VMamba)
```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_baseline/
└── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_baseline/
```

### Spatial Gate (SG) Model
```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_sg/
└── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_sg/
```

### VMamba Model (No Gating)
```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_vmamba/
└── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_vmamba/
```

### SG + VMamba Model (Combined)
```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_sg_vmamba/
└── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_sg_vmamba/
```

## Z-Spacing Values
- **z1p0**: 1.0 mm (isotropic-like)
- **z2p2**: 2.2 mm (moderate anisotropy)
- **z4p0**: 4.0 mm (high anisotropy)

## Model Configurations

| Model       | Gated Fusion | VMamba Bottlenecks |
|-------------|--------------|-------------------|
| baseline    | None         | []                |
| sg          | spatial      | []                |
| vmamba      | None         | [3, 4, 5]         |
| sg_vmamba   | spatial      | [3, 4, 5]         |

## Each Folder Contains
```
fold_0/
├── model_final_checkpoint.model
├── model_best.model
├── validation_raw/                    # Raw predictions
├── validation_raw_postprocessed/      # Post-processed + summary.json
├── training_log.txt
├── progress.png
└── arch_meta.json
```

## Corresponding Plans Files
Each combination has its own plans file in `nnUNet_preprocessed/Task029_LITS/`:
```
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline_plans_3D.pkl
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg_plans_3D.pkl
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_vmamba_plans_3D.pkl
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg_vmamba_plans_3D.pkl
... (×3 for z2p2 and z4p0)
```

## Why This Approach?

### ❌ Previous Approach (Failed)
```python
trainer_name = "myTrainer_reproduction_WandB_baseline_z1p0"  # Not a real Python class!
plans_identifier = "nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121"
```
**Problem**: nnUNet dynamically imports trainer classes by name. Custom names don't exist in the codebase.

### ✅ Correct Approach
```python
trainer_name = "myTrainer_reproduction_WandB"  # Real Python class
plans_identifier = "nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline"  # Unique per model+spacing
```
**Solution**: Same trainer class, unique plans_identifier ensures separate folders.

## Benefits
✅ No overwrites - each configuration is isolated  
✅ Easy comparison - results organized by model and spacing  
✅ Clear naming - folder names explicitly show configuration  
✅ Parallelizable - can run multiple configs simultaneously (if resources allow)  
✅ **nnUNet-compatible** - uses existing trainer classes  
