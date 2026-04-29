#!/bin/bash

# Evaluate on novel categories using Grounding DINO oracle 2D boxes (previous-metric mode).
python tools/train_net.py --config-file configs/OVMono3D_dinov2_SFP.yaml --eval-only --dist-url auto --resume \
 --num-gpus 8 \
 OUTPUT_DIR output/eval/bs64_roi_unidepth_omni3d_novel_prev \
 MODEL.WEIGHTS "checkpoints/ovmono3d_lift.pth" \
 INPUT.USE_DEPTH True \
 DATASETS.FOLDER_NAME "Omni3D_unidepth" \
 DATASETS.ORACLE2D_PROMPT "gdino_previous_metric" \
 TEST.CAT_MODE "novel"
