# Sweep Script Fix Summary

## Problems Identified

### 1. **Invalid Trainer Names** ❌
**Error**: `TypeError: issubclass() arg 1 must be a class`
```
My trainer class is:  None
```

**Cause**: The script was generating trainer names like:
- `myTrainer_reproduction_WandB_baseline_z1p0`
- `myTrainer_reproduction_WandB_sg_z2p2`

These are **not real Python classes**. nnUNet dynamically imports trainer classes by name, so they must exist in the codebase.

### 2. **Wrong Experiment Planner Class Name** ❌
**Error**: `ImportError: cannot import name 'ExperimentPlanner3DFabiansResUNet_v21'`

**Cause**: The planner files were trying to import a non-existent class.

**Fix**: Changed to the correct class name: `ExperimentPlanner3D_v21`

### 3. **Missing Plans Files** ❌
**Error**: `FileNotFoundError: nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_plans_3D.pkl`

**Cause**: Only 3 experiment planners existed (one per z-spacing), but we need unique plans for each model+spacing combination to ensure separate output folders.

## Solution ✅

### Key Insight
nnUNet's output folder structure is:
```
{results_base}/nnUNet/3d_fullres/{task}/{trainer_name}__{plans_identifier}/fold_{fold}/
```

**Uniqueness comes from `plans_identifier`, NOT `trainer_name`!**

### Changes Made

#### 1. **Same Trainer Name for All**
```python
trainer_name = "myTrainer_reproduction_WandB"  # Same for all 12 runs
```

#### 2. **Unique Plans Identifier per Model+Spacing**
```python
plans_identifier = f"nnUNetData_plans_v2.1_trgSp_{z}_yx0p9121_{model_key}"
```

Examples:
- `nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline`
- `nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_sg`
- `nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_vmamba`

#### 3. **12 Experiment Planners** (4 models × 3 spacings)
Created separate planner files for each combination:
```
ExperimentPlanner3D_v21_trgSp_z1p0_yx0p9121_baseline.py
ExperimentPlanner3D_v21_trgSp_z1p0_yx0p9121_sg.py
ExperimentPlanner3D_v21_trgSp_z1p0_yx0p9121_vmamba.py
ExperimentPlanner3D_v21_trgSp_z1p0_yx0p9121_sg_vmamba.py
... (×3 for z2p2 and z4p0)
```

#### 4. **12 Preprocessing Steps**
Each model+spacing combination gets its own plans file:
```
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline_plans_3D.pkl
nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg_plans_3D.pkl
... (×12 total)
```

## Result: 12 Unique Output Folders ✅

```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_baseline/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_baseline/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_sg/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_sg/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_sg_vmamba/
├── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121_sg_vmamba/
└── myTrainer_reproduction_WandB__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121_sg_vmamba/
```

## Status

✅ **Sweep is now running successfully**

- Process ID: 86171
- Log file: `sweep_output_fixed.log`
- Current stage: Preprocessing z1p0_baseline
- Estimated time: 6-8 hours (preprocessing + 12 training runs @ ~20min each)

### Monitor Progress
```bash
tail -f sweep_output_fixed.log
```

### Current Activity
The sweep is verifying the 131 LiTS training cases for the first preprocessing step. This will take 10-30 minutes per combination.

