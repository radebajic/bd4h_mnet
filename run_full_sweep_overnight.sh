#!/bin/bash
set -e

# ============================================================
# LITS FULL ANISOTROPY + MODEL SWEEP - OVERNIGHT AUTOMATION
# ============================================================
# This script will:
# 1. Generate all config files (with unique trainer names)
# 2. Create experiment planners (A100 80GB optimized)
# 3. Preprocess all spacings (60GB memory budget)
# 4. Train all 12 model×spacing combinations (fold 0 only)
# 5. Validate all runs
# 6. Collect results
#
# A100 80GB Optimizations:
# - Preprocessing: 60GB memory budget for larger batch/patch sizes
# - Training: batch_size=4, num_threads=16
# - Preprocessing threads: 16 loading + 16 transform threads
# ============================================================

START_TIME=$(date +%s)
LOGFILE="sweep_overnight_fold0_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to log file AND terminal
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "LITS FULL SWEEP - FOLD 0 ONLY"
echo "A100 80GB Optimized (312 TFLOPs)"
echo "Started: $(date)"
echo "Log file: $LOGFILE"
echo "Total runs: 12 (4 models × 3 spacings × 1 fold)"
echo "========================================"

# Environment setup
export nnUNet_raw_data_base=/teamspace/studios/this_studio/nnUNet_raw
export nnUNet_preprocessed=/teamspace/studios/this_studio/nnUNet_preprocessed
export RESULTS_FOLDER=/teamspace/studios/this_studio/nnUNet_results
export CUDA_VISIBLE_DEVICES=0

cd /teamspace/studios/this_studio/bd4h_mnet-1

# ============================================================
# STEP 1: Generate Config Files
# ============================================================
echo ""
echo "========================================"
echo "STEP 1/5: Generating config files..."
echo "========================================"

python3 << 'PYTHON_EOF'
from pathlib import Path
import yaml

models = {
    "baseline": {
        "gated_fusion": None, 
        "bottleneck_stages": []
    },
    "sg": {
        "gated_fusion": "spatial", 
        "bottleneck_stages": []
    },
    "vmamba": {
        "gated_fusion": None, 
        "bottleneck_stages": [3, 4, 5]
    },
    "sg_vmamba": {
        "gated_fusion": "spatial", 
        "bottleneck_stages": [3, 4, 5]
    },
}

spacings = ["z1p0", "z2p2", "z4p0"]
folds = [0]  # Fold 0 only
base_dir = Path("conf/training")

# All models use the same trainer class - differentiation comes from plans_identifier
trainer_name = "myTrainer_reproduction_WandB"

for model_key, model_config in models.items():
    for z in spacings:
        config_dir = base_dir / f"lits_sweep_{model_key}" / z
        config_dir.mkdir(parents=True, exist_ok=True)
        
        for fold in folds:
            # Training config
            train_config = {
                "hydra": {
                    "job": {
                        "chdir": False,
                        "env_set": {
                            "WANDB_ENTITY": "xplstm",
                            "WANDB_PROJECT": "CSE6250_MNet_LiTS_Sweep",
                            "WANDB_GROUP": f"lits_{model_key}_{z}"
                        }
                    }
                },
                "trainer": {
                    "model": "3d_fullres",
                    "trainer_name": trainer_name,
                    "task": "Task029_LITS",
                    "fold": fold,
                    "plans_identifier": f"nnUNetData_plans_v2.1_trgSp_{z}_yx0p9121_{model_key}",  # Unique plans per model+spacing
                    "save_npz": True,
                    "validation_only": False,
                    "valbest": False,
                    "dataset_directory": "/teamspace/studios/this_studio/nnUNet_preprocessed",
                    "results_base": "/teamspace/studios/this_studio/nnUNet_results",
                    "fp16": True,
                    "deterministic": False,
                    "unpack_data": True,
                    "max_epochs": 50,
                    "gated_fusion": model_config["gated_fusion"],
                    "width_mult": 1.0,
                    "use_sep3d": False,
                    "use_checkpoint": False,
                    "cat_reduce": False,
                    "batch_size": 4,
                    "num_threads": 16,
                    "vmamba": {
                        "down_stages": [],
                        "up_stages": [],
                        "bottleneck_stages": model_config["bottleneck_stages"],
                        "axial_reduce": 0.5
                    },
                    "wandb": {
                        "enabled": True,
                        "project": "CSE6250_MNet_LiTS_Sweep",
                        "run_name": f"lits_{model_key}_{z}_fold{fold}",
                        "group": f"lits_{model_key}_{z}"
                    }
                }
            }
            
            with open(config_dir / f"best_arch_3_fold{fold}.yaml", 'w') as f:
                yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)
            
            # Validation config
            val_config = train_config.copy()
            val_config["trainer"]["validation_only"] = True
            val_config["trainer"]["valbest"] = True
            
            with open(config_dir / f"validate_best_arch3_fold{fold}.yaml", 'w') as f:
                yaml.dump(val_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created configs for: {model_key}/{z} → trainer: {trainer_name}")

print("\n✓ All config files generated!")
print("\n" + "=" * 80)
print("OUTPUT FOLDER STRUCTURE:")
print("=" * 80)
print("Each configuration will save to a unique folder in nnUNet_results:")
print()
trainer_name = "myTrainer_reproduction_WandB"
for model_key in models.keys():
    for z in spacings:
        plan_id = f"nnUNetData_plans_v2.1_trgSp_{z}_yx0p9121_{model_key}"
        folder_name = f"{trainer_name}__{plan_id}"
        print(f"  {model_key:12s} + {z:6s} → {folder_name}")
print("=" * 80)
PYTHON_EOF

echo "✓ Config generation complete"

# ============================================================
# Check if we should skip preprocessing (for GPU training machine)
# ============================================================
if [ "${SKIP_PREPROCESSING:-0}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "SKIP_PREPROCESSING=1 detected"
    echo "Jumping directly to training phase..."
    echo "========================================"
    # Skip to training section (Steps 2-3 will be skipped)
    SKIP_TO_TRAINING=1
else
    SKIP_TO_TRAINING=0
fi

# ============================================================
# STEP 2: Create Experiment Planners
# ============================================================
if [ "$SKIP_TO_TRAINING" = "0" ]; then
echo ""
echo "========================================"
echo "STEP 2/5: Creating experiment planners..."
echo "========================================"

PLANNER_DIR="nnunet/experiment_planning/alternative_experiment_planning/target_spacing"
mkdir -p "$PLANNER_DIR"

# Create 12 experiment planners (4 models × 3 spacings)
# Each model variant will have separate plans files to ensure unique output folders

python3 << 'PYTHON_PLANNER_EOF'
from pathlib import Path

models = ["baseline", "sg", "vmamba", "sg_vmamba"]
spacings = {"z1p0": 1.0, "z2p2": 2.2, "z4p0": 4.0}
planner_dir = Path("nnunet/experiment_planning/alternative_experiment_planning/target_spacing")
planner_dir.mkdir(parents=True, exist_ok=True)

template = '''from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
import numpy as np

class ExperimentPlanner3D_v21_trgSp_{z_key}_yx0p9121_{model_key}(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super().__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_trgSp_{z_key}_yx0p9121_{model_key}"
        self.plans_fname = self.data_identifier + "_plans_3D.pkl"
        
        # A100 80GB optimization: Use 60GB for batch size computation
        self.use_this_for_batch_size_computation_3D = 60 * 1024 * 1024 * 1024

    def get_target_spacing(self):
        return np.array([{z_value}, 0.9121, 0.9121])
'''

for model_key in models:
    for z_key, z_value in spacings.items():
        planner_file = planner_dir / f"ExperimentPlanner3D_v21_trgSp_{z_key}_yx0p9121_{model_key}.py"
        content = template.format(model_key=model_key, z_key=z_key, z_value=z_value)
        planner_file.write_text(content)
        print(f"✓ Created planner: {z_key}_{model_key}")

PYTHON_PLANNER_EOF

echo "✓ All 12 experiment planners created"
fi  # End of SKIP_TO_TRAINING check for Step 2

# ============================================================
# STEP 3: Preprocessing
# ============================================================
if [ "$SKIP_TO_TRAINING" = "0" ]; then
echo ""
echo "========================================"
echo "STEP 3/5: Preprocessing all combinations..."
echo "========================================"

models=("baseline" "sg" "vmamba" "sg_vmamba")
spacings=("z1p0" "z2p2" "z4p0")

for model in "${models[@]}"; do
  for z in "${spacings[@]}"; do
    # Check if preprocessing is already done by looking for stage folder with data
    STAGE_FOLDER="$nnUNet_preprocessed/Task029_LITS/nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}_stage1"
    PLANS_FILE="$nnUNet_preprocessed/Task029_LITS/nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}_plans_3D.pkl"
    
    if [ -d "$STAGE_FOLDER" ] && [ "$(ls -A $STAGE_FOLDER/*.npz 2>/dev/null | wc -l)" -gt 40 ]; then
        echo "✓ Preprocessing exists: ${z}_${model}, skipping ($(ls $STAGE_FOLDER/*.npz 2>/dev/null | wc -l) files found)"
    else
        echo "→ Preprocessing: ${z}_${model} (A100 80GB optimized)"
        # Use more threads for A100 preprocessing (16 for loading, 16 for transform)
        nnUNet_plan_and_preprocess -t 29 \
          -pl3d ExperimentPlanner3D_v21_trgSp_${z}_yx0p9121_${model} \
          -tl 16 -tf 16 --verify_dataset_integrity
        
        # Move plans file if it was saved to wrong location (nnUNet quirk with custom planners)
        if [ -f "nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}_plans_3D.pkl" ]; then
            mv "nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}_plans_3D.pkl" "$PLANS_FILE"
            echo "  → Moved plans file to correct location"
        fi
        
        echo "✓ Preprocessing complete: ${z}_${model}"
    fi
  done
done

echo "✓ All preprocessing complete"
fi  # End of SKIP_TO_TRAINING check for Step 3

# ============================================================
# STOP HERE - Switch to training GPU
# ============================================================
if [ "$SKIP_TO_TRAINING" = "0" ]; then
    echo ""
    echo "========================================"
    echo "PREPROCESSING COMPLETE!"
    echo "========================================"
    echo ""
    echo "All 12 combinations have been preprocessed."
    echo ""
    echo "To continue with training on your preferred GPU:"
    echo "  1. Switch to the GPU machine"
    echo "  2. Run: SKIP_PREPROCESSING=1 ./run_full_sweep_overnight.sh"
    echo ""
    echo "Exiting now (training not started)."
    echo "========================================"
    exit 0
fi

# ============================================================
# STEP 4: Training (12 runs)
# ============================================================
echo ""
echo "========================================"
echo "STEP 4/5: Training all combinations..."
echo "========================================"

models=("baseline" "sg" "vmamba" "sg_vmamba")
spacings=("z1p0" "z2p2" "z4p0")
folds=(0)

total_runs=$((${#models[@]} * ${#spacings[@]} * ${#folds[@]}))
current_run=0
failed_runs=()
skipped_runs=()

for model in "${models[@]}"; do
  for z in "${spacings[@]}"; do
    for fold in "${folds[@]}"; do
      current_run=$((current_run + 1))
      
      # Check if training is already complete
      PLAN_ID="nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}"
      MODEL_FINAL="$RESULTS_FOLDER/nnUNet/3d_fullres/Task029_LITS/myTrainer_reproduction_WandB__${PLAN_ID}/fold_${fold}/model_final_checkpoint.model"
      
      echo ""
      echo "========================================" 
      echo "[$current_run/$total_runs] Training: ${model} | ${z} | fold ${fold}"
      echo "Time: $(date)"
      echo "========================================"
      
      # Skip if already completed
      if [ -f "$MODEL_FINAL" ]; then
        echo "✓ ALREADY COMPLETED: ${model} ${z} fold ${fold}"
        echo "  Found: model_final_checkpoint.model"
        skipped_runs+=("${model}_${z}_fold${fold}")
        continue
      fi
      
      # Check if partial training exists (checkpoint but not final)
      MODEL_BEST="$RESULTS_FOLDER/nnUNet/3d_fullres/Task029_LITS/myTrainer_reproduction_WandB__${PLAN_ID}/fold_${fold}/model_best.model"
      CONTINUE_FLAG=""
      
      if [ -f "$MODEL_BEST" ]; then
        echo "⚠ RESUMING from checkpoint: ${model} ${z} fold ${fold}"
        echo "  Found: model_best.model (will continue training)"
        CONTINUE_FLAG="trainer.continue_training=true"
      fi
      
      # Try to train, but continue on failure
      if python hydra_trainer.py \
        --config-path=conf/training/lits_sweep_${model}/${z} \
        --config-name=best_arch_3_fold${fold} \
        ${CONTINUE_FLAG}; then
        echo "✓ SUCCESS: ${model} ${z} fold ${fold}"
      else
        echo "✗ FAILED: ${model} ${z} fold ${fold}"
        failed_runs+=("${model}_${z}_fold${fold}")
      fi
    done
  done
done

echo ""
echo "========================================"
echo "Training phase complete!"
echo "Total runs: $total_runs"
echo "Skipped (already done): ${#skipped_runs[@]}"
echo "Newly trained: $((total_runs - ${#failed_runs[@]} - ${#skipped_runs[@]}))"
echo "Failed: ${#failed_runs[@]}"
if [ ${#skipped_runs[@]} -gt 0 ]; then
    echo ""
    echo "Skipped runs: ${skipped_runs[@]}"
fi
if [ ${#failed_runs[@]} -gt 0 ]; then
    echo ""
    echo "Failed runs: ${failed_runs[@]}"
fi
echo "========================================"

# ============================================================
# STEP 5: Validation
# ============================================================
echo ""
echo "========================================"
echo "STEP 5/5: Validating all runs..."
echo "========================================"

val_skipped=0
val_completed=0
val_failed=0

for model in "${models[@]}"; do
  for z in "${spacings[@]}"; do
    for fold in "${folds[@]}"; do
      # Check if validation is already complete
      PLAN_ID="nnUNetData_plans_v2.1_trgSp_${z}_yx0p9121_${model}"
      SUMMARY_JSON="$RESULTS_FOLDER/nnUNet/3d_fullres/Task029_LITS/myTrainer_reproduction_WandB__${PLAN_ID}/fold_${fold}/validation_raw_postprocessed/summary.json"
      
      echo "→ Validating: ${model} ${z} fold ${fold}"
      
      # Skip if validation already exists
      if [ -f "$SUMMARY_JSON" ]; then
        echo "  ✓ Already validated (found summary.json)"
        val_skipped=$((val_skipped + 1))
        continue
      fi
      
      if python hydra_trainer.py \
        --config-path=conf/training/lits_sweep_${model}/${z} \
        --config-name=validate_best_arch3_fold${fold}; then
        echo "  ✓ Validation complete"
        val_completed=$((val_completed + 1))
      else
        echo "  ✗ Validation failed"
        val_failed=$((val_failed + 1))
      fi
    done
  done
done

echo ""
echo "========================================"
echo "Validation summary:"
echo "  Already validated: $val_skipped"
echo "  Newly validated: $val_completed"
echo "  Failed: $val_failed"
echo "========================================"

# ============================================================
# STEP 6: Collect Results
# ============================================================
echo ""
echo "========================================"
echo "Collecting results..."
echo "========================================"

python3 << 'PYTHON_EOF'
import json
import numpy as np
from pathlib import Path

models = {
    "baseline": "MNet (Ours)",
    "sg": "MNet + Spatial Gate",
    "vmamba": "MNet + VMamba",
    "sg_vmamba": "MNet + (SG + VM)"
}

spacings = {
    "z1p0": "1.0",
    "z2p2": "2.2",
    "z4p0": "4.0"
}

folds = [0]
results_base = Path("/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task029_LITS")

print("\n" + "=" * 100)
print(f"{'Model':<25} {'z-spacing':<12} {'Liver (Dice %)':<20} {'Tumor (Dice %)':<20}")
print("=" * 100)

results_data = []

for model_key, model_name in models.items():
    for z_key, z_value in spacings.items():
        # All use same trainer, differentiation from plans_identifier
        trainer_name = "myTrainer_reproduction_WandB"
        plan_id = f"nnUNetData_plans_v2.1_trgSp_{z_key}_yx0p9121_{model_key}"
        
        dice_liver = []
        dice_tumor = []
        
        for fold in folds:
            summary_path = (
                results_base / 
                f"{trainer_name}__{plan_id}" /
                f"fold_{fold}" /
                "validation_raw_postprocessed" /
                "summary.json"
            )
            
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    dice_liver.append(data['results']['mean']['1']['Dice'])
                    dice_tumor.append(data['results']['mean']['2']['Dice'])
            except FileNotFoundError:
                print(f"⚠ Missing: {model_key} {z_key} fold {fold}")
        
        if dice_liver:
            liver_mean = np.mean(dice_liver) * 100
            liver_std = np.std(dice_liver, ddof=1) * 100 if len(dice_liver) > 1 else 0
            tumor_mean = np.mean(dice_tumor) * 100
            tumor_std = np.std(dice_tumor, ddof=1) * 100 if len(dice_tumor) > 1 else 0
            
            print(f"{model_name:<25} {z_value:<12} {liver_mean:5.1f} ± {liver_std:4.1f}       {tumor_mean:5.1f} ± {tumor_std:4.1f}")
            
            results_data.append({
                'model': model_name,
                'z': z_value,
                'liver': f"{liver_mean:.1f}",
                'tumor': f"{tumor_mean:.1f}"
            })
        else:
            print(f"{model_name:<25} {z_value:<12} {'NO DATA':<20} {'NO DATA':<20}")

print("=" * 100)

# Export LaTeX-friendly format
print("\n" + "=" * 80)
print("LATEX TABLE FORMAT:")
print("=" * 80)
for result in results_data:
    print(f"{result['model']} & {result['z']} & {result['liver']} & {result['tumor']} \\\\")
print("=" * 80)

PYTHON_EOF

# ============================================================
# Summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================"
echo "FULL SWEEP COMPLETE!"
echo "========================================"
echo "Started:  $(date -d @$START_TIME 2>/dev/null || date -r $START_TIME)"
echo "Finished: $(date)"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Log saved to: $LOGFILE"
echo "========================================"


