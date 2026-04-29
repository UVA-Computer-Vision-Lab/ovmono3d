#!/bin/bash

#SBATCH --job-name=bs64_roi_unidepth_omni3d
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=dedicated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=94
#SBATCH --gres=gpu:8
#SBATCH --mem=740G
#SBATCH --time=6-00:00:00
#SBATCH --mail-type=begin,end,fail


python tools/train_net.py --config-file configs/OVMono3D_dinov2_SFP.yaml  --dist-url auto --resume \
 --num-gpus 8    \
 OUTPUT_DIR  output/bs64_roi_unidepth_omni3d    \
 VIS_PERIOD 500 \
 TEST.EVAL_PERIOD 5000   \
 MODEL.STABILIZE  0.03    \
 SOLVER.BASE_LR 0.012     \
 SOLVER.CHECKPOINT_PERIOD 1000     \
 SOLVER.IMS_PER_BATCH 64 \
 INPUT.USE_DEPTH True \
 DATASETS.FOLDER_NAME "Omni3D_unidepth" \
 TEST.CAT_MODE "novel"