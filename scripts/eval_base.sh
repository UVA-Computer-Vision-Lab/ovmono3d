#!/bin/bash

# Evaluate on base categories using the model's own 2D head (no oracle 2D).
python tools/train_net.py --config-file configs/OVMono3D_dinov2_SFP.yaml --eval-only --dist-url auto --resume \
 --num-gpus 8 \
 OUTPUT_DIR output/eval/bs64_roi_unidepth_omni3d_base \
 MODEL.WEIGHTS "checkpoints/ovmono3d_lift.pth" \
 INPUT.USE_DEPTH True \
 DATASETS.FOLDER_NAME "Omni3D_unidepth" \
 TEST.ORACLE2D False \
 TEST.CAT_MODE "base"
