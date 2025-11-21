# LiTS Anisotropy Sweep - Quick Start Guide

## What This Does

Automatically runs a complete sweep of:
- **4 model variants**: Baseline, Spatial Gate, VMamba, Spatial Gate + VMamba
- **3 z-spacings**: 1.0 mm, 2.2 mm, 4.0 mm
- **1 fold**: Fold 0 only

**Total: 12 training runs** (approximately 4-8 hours on A100)

## Key Features

✅ **No overwrites**: Each model variant uses a unique trainer name  
✅ **Fully automated**: Config generation, preprocessing, training, validation, results  
✅ **Error resilient**: Continues even if individual runs fail  
✅ **Complete logging**: All output saved to timestamped log file  

## Quick Start

### Option 1: Run in Background with nohup

```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1
nohup ./run_full_sweep_overnight.sh > sweep_output.log 2>&1 &

# Monitor progress
tail -f sweep_overnight_fold0_*.log
```

### Option 2: Run in tmux (Recommended)

```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1

# Start tmux session
tmux new -s lits_sweep

# Run the script
./run_full_sweep_overnight.sh

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t lits_sweep
```

### Option 3: Run Directly (Stays in Foreground)

```bash
cd /teamspace/studios/this_studio/bd4h_mnet-1
./run_full_sweep_overnight.sh
```

## Monitoring Progress

### Check Current Status
```bash
# Count completed runs
grep "✓ SUCCESS" sweep_overnight_fold0_*.log | wc -l

# Check for failures
grep "✗ FAILED" sweep_overnight_fold0_*.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

### View Real-Time Log
```bash
tail -f sweep_overnight_fold0_*.log
```

## Output Structure

Results will be organized as:

```
nnUNet_results/nnUNet/3d_fullres/Task029_LITS/
├── myTrainer_reproduction_WandB_baseline__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121/
│   └── fold_0/
├── myTrainer_reproduction_WandB_baseline__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p9121/
│   └── fold_0/
├── myTrainer_reproduction_WandB_baseline__nnUNetData_plans_v2.1_trgSp_z4p0_yx0p9121/
│   └── fold_0/
├── myTrainer_reproduction_WandB_sg__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121/
│   └── fold_0/
├── (... and 8 more combinations)
```

## What Gets Created

1. **Config files**: `conf/training/lits_sweep_{model}/{spacing}/`
   - 12 training YAMLs
   - 12 validation YAMLs

2. **Experiment planners**: `nnunet/experiment_planning/alternative_experiment_planning/target_spacing/`
   - ExperimentPlanner3D_v21_trgSp_z1p0_yx0p9121.py
   - ExperimentPlanner3D_v21_trgSp_z2p2_yx0p9121.py
   - ExperimentPlanner3D_v21_trgSp_z4p0_yx0p9121.py

3. **Preprocessed data**: `nnUNet_preprocessed/Task029_LITS/`
   - 3 plans files (one per spacing)
   - Preprocessed datasets

4. **Training outputs**: 12 separate result folders (no overwrites!)

5. **Log file**: `sweep_overnight_fold0_YYYYMMDD_HHMMSS.log`

## Results Summary

At the end of the run, you'll get:

```
================================================================================
Model                     z-spacing    Liver (Dice %)       Tumor (Dice %)      
================================================================================
MNet (Ours)              1.0          94.3 ± 0.0           54.6 ± 0.0     
MNet (Ours)              2.2          XX.X ± 0.0           XX.X ± 0.0     
MNet (Ours)              4.0          XX.X ± 0.0           XX.X ± 0.0     
MNet + Spatial Gate      1.0          XX.X ± 0.0           XX.X ± 0.0     
...
================================================================================
```

Plus LaTeX-ready table format at the bottom.

## Troubleshooting

### Script Won't Start
```bash
# Make sure it's executable
chmod +x run_full_sweep_overnight.sh

# Check environment
echo $nnUNet_preprocessed
echo $RESULTS_FOLDER
```

### Preprocessing Takes Too Long
The script checks if preprocessing already exists and skips it. First run will be slower.

### Training Fails for Specific Runs
The script continues even if individual runs fail. Check the log for specific error messages.

### Out of Disk Space
Each run generates ~5-10GB. Make sure you have at least 150GB free.

### CUDA Out of Memory
Reduce batch size in the config generator (line with `"batch_size": 4`) to 2.

## Stopping the Sweep

### If running in tmux:
```bash
tmux attach -t lits_sweep
# Press Ctrl+C
```

### If running with nohup:
```bash
# Find the process
ps aux | grep run_full_sweep_overnight

# Kill it
kill <PID>
```

## Running Specific Parts Only

### Only preprocessing:
Comment out the training and validation sections in the script.

### Only training (skip preprocessing):
The script already checks if preprocessing exists and skips it automatically.

### Restart failed runs:
Just re-run the script. It will skip completed runs automatically (due to checkpoint loading).

## Expected Timeline

- **Config generation**: < 1 minute
- **Planner creation**: < 1 minute
- **Preprocessing** (first time): 30-60 minutes per spacing
- **Training** (per run): 20-40 minutes @ 150 epochs
- **Validation** (per run): 2-5 minutes
- **Results collection**: < 1 minute

**Total (first run with preprocessing)**: 6-10 hours  
**Total (if preprocessed already)**: 4-8 hours

## Next Steps

After the sweep completes:
1. Check the log file for any failures
2. Review the results table
3. Analyze WandB dashboard for training curves
4. Use results for your paper's anisotropy sensitivity table!


