#!/bin/bash
# Train 3 folds in parallel (requires 3 GPUs or 3 separate instances)

cd /teamspace/studios/this_studio/bd4h_mnet-1

echo "Starting 3-fold training..."
echo "Make sure you have sufficient GPU resources!"

# Option 1: If you have 3 GPUs on same machine
# CUDA_VISIBLE_DEVICES=0 python hydra_trainer.py --config-name best_arch_3_fold0 &
# CUDA_VISIBLE_DEVICES=1 python hydra_trainer.py --config-name best_arch_3_fold1 &
# CUDA_VISIBLE_DEVICES=2 python hydra_trainer.py --config-name best_arch_3_fold2 &
# wait

# Option 2: Sequential (use this if you have only 1 GPU)
python hydra_trainer.py --config-name best_arch_3_fold0
echo "Fold 0 complete!"

python hydra_trainer.py --config-name best_arch_3_fold1
echo "Fold 1 complete!"

python hydra_trainer.py --config-name best_arch_3_fold2
echo "Fold 2 complete!"

echo "All 3 folds trained successfully!"

