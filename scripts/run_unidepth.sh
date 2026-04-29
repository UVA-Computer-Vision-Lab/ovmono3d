# scripts/run_unidepth.sh

# Function to run commands for a single GPU serially
run_on_gpu() {
    local gpu_id=$1
    shift
    for cmd in "$@"; do
        echo "Running on GPU $gpu_id: $cmd"
        CUDA_VISIBLE_DEVICES=$gpu_id $cmd
    done
}

# Objectron on GPU 2
(
run_on_gpu 2 \
    "python tools/unidepth_script.py --dataset Objectron --split train" \
    "python tools/unidepth_script.py --dataset Objectron --split val" \
    "python tools/unidepth_script.py --dataset Objectron --split test"
) &

# KITTI on GPU 1
(
run_on_gpu 1 \
    "python tools/unidepth_script.py --dataset KITTI --split train" \
    "python tools/unidepth_script.py --dataset KITTI --split val" \
    "python tools/unidepth_script.py --dataset KITTI --split test" \
    "python tools/unidepth_script.py --dataset KITTI --split test_novel"
) &

# ARKitScenes on GPU 3
(
run_on_gpu 3 \
    "python tools/unidepth_script.py --dataset ARKitScenes --split val" \
    "python tools/unidepth_script.py --dataset ARKitScenes --split test" \
    "python tools/unidepth_script.py --dataset ARKitScenes --split test_novel"
) &

# nuScenes on GPU 4
(
run_on_gpu 4 \
    "python tools/unidepth_script.py --dataset nuScenes --split val" \
    "python tools/unidepth_script.py --dataset nuScenes --split test"
) &

# Hypersim on GPU 5
(
run_on_gpu 5 \
    "python tools/unidepth_script.py --dataset Hypersim --split train" \
    "python tools/unidepth_script.py --dataset Hypersim --split val" \
    "python tools/unidepth_script.py --dataset Hypersim --split test"
) &

# SUNRGBD on GPU 0
(
run_on_gpu 0 \
    "python tools/unidepth_script.py --dataset SUNRGBD --split train" \
    "python tools/unidepth_script.py --dataset SUNRGBD --split val" \
    "python tools/unidepth_script.py --dataset SUNRGBD --split test" \
    "python tools/unidepth_script.py --dataset SUNRGBD --split test_novel"
) &

# ARKitScenes train on GPU 6
(
run_on_gpu 6 \
    "python tools/unidepth_script.py --dataset ARKitScenes --split train"
) &

# nuScenes train on GPU 7
(
run_on_gpu 7 \
    "python tools/unidepth_script.py --dataset nuScenes --split train"
) &

# Wait for all background jobs to finish
wait
echo "All jobs finished."
