#!/bin/bash

# Evaluate on Cityscapes3D_test using the model's own 2D head (no oracle 2D).
# Cityscapes3D was reported with Metric3D depth, so this expects depth maps in
# datasets/Omni3D_metric3d/ (generated via: python tools/metric3d_script.py --dataset Cityscapes3D --split test).
python tools/train_net.py --config-file configs/OVMono3D_dinov2_SFP.yaml --eval-only --dist-url auto --resume \
 --num-gpus 8 \
 OUTPUT_DIR output/eval/bs64_roi_unidepth_omni3d_cityscapes \
 MODEL.WEIGHTS "checkpoints/ovmono3d_lift.pth" \
 INPUT.USE_DEPTH True \
 DATASETS.FOLDER_NAME "Omni3D_metric3d" \
 TEST.ORACLE2D False \
 TEST.CAT_MODE "base" \
 DATASETS.TEST_BASE "('Cityscapes3D_test',)"
