# Two-Phase Sweep Workflow

## Overview
The sweep is now split into two phases:
1. **Phase 1**: Preprocessing on fast CPU machine
2. **Phase 2**: Training on preferred GPU machine

---

## Phase 1: Preprocessing (Fast CPU Machine)

### Run on your fast CPU machine:
```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1
nohup ./run_full_sweep_overnight.sh > sweep_preprocessing.log 2>&1 &
```

### What happens:
- ✅ Generates 12 config files
- ✅ Creates 12 experiment planners
- ✅ Preprocesses all 12 model+spacing combinations
- ⏹️  **STOPS automatically before training starts**

### Expected time:
- **Fast CPU**: ~2-3 hours for all 12 preprocessings
- **Current progress**: 3/12 done (25%)

### Output:
```
All preprocessed data saved to:
/teamspace/studios/this_studio/nnUNet_preprocessed/Task029_LITS/

Files created:
├── nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline_plans_3D.pkl
├── nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline_stage0/ (48 cases)
├── nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_baseline_stage1/ (48 cases)
└── ... (×12 combinations)
```

### When Phase 1 completes, you'll see:
```
========================================
PREPROCESSING COMPLETE!
========================================

All 12 combinations have been preprocessed.

To continue with training on your preferred GPU:
  1. Switch to the GPU machine
  2. Run: SKIP_PREPROCESSING=1 ./run_full_sweep_overnight.sh

Exiting now (training not started).
========================================
```

---

## Phase 2: Training (Preferred GPU Machine)

### Switch to your preferred GPU machine

### Run with SKIP_PREPROCESSING flag:
```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1
SKIP_PREPROCESSING=1 nohup ./run_full_sweep_overnight.sh > sweep_training.log 2>&1 &
```

### What happens:
- ⏩ Skips config generation (already done)
- ⏩ Skips planner creation (already done)  
- ⏩ Skips preprocessing (already done)
- 🚀 **Runs 12 training jobs** (fold 0 only)
- 📊 Validates all models
- 📈 Collects results

### Expected time:
- **Per training run**: ~30-40 minutes
- **Total**: ~6-8 hours for all 12 runs

### Output:
```
Results saved to:
/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task029_LITS/

12 unique output folders:
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

---

## Benefits of Two-Phase Workflow

### ✅ Flexibility
- Use fastest CPU for preprocessing
- Use preferred GPU for training
- Different machines for each phase

### ✅ Efficiency  
- Preprocessing: CPU-intensive (no GPU needed)
- Training: GPU-intensive (no need for fast CPU)

### ✅ Cost Optimization
- Don't waste expensive GPU time on CPU-only preprocessing
- Don't waste fast CPU time waiting for GPU training

### ✅ Safety
- All preprocessing saved before training starts
- Can restart training multiple times without redoing preprocessing

---

## Monitoring

### Phase 1 (Preprocessing):
```bash
tail -f sweep_preprocessing.log
```

### Phase 2 (Training):
```bash
tail -f sweep_training.log
```

---

## Current Status

**Completed preprocessing:**
- ✅ z1p0_baseline
- ✅ z2p2_baseline  
- ✅ z4p0_baseline
- ⏸️  z1p0_sg (partial - will redo)

**Remaining preprocessing:**
- ⏳ z1p0_sg, z2p2_sg, z4p0_sg
- ⏳ z1p0_vmamba, z2p2_vmamba, z4p0_vmamba
- ⏳ z1p0_sg_vmamba, z2p2_sg_vmamba, z4p0_sg_vmamba

**Ready to start:**
Phase 1 on fast CPU machine! 🚀

