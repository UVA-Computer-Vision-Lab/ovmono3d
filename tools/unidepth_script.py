import torch
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import open3d as o3d
from datetime import datetime

from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, required=True)
args = parser.parse_args()

root = Path("./datasets")

dataset_json = root / "Omni3D" / f"{args.dataset}_{args.split}.json"

with open(dataset_json, "r") as f:
    dataset = json.load(f)

depth_save_dir = root / "pseudo_unidepth" / f"{args.dataset}"
depth_save_dir.mkdir(parents=True, exist_ok=True)

dataset_save_path = root / "Omni3D_unidepth" /f"{args.dataset}_{args.split}.json"
dataset_save_path.parent.mkdir(parents=True, exist_ok=True)

# Create visualization folder
vis_save_dir = root / "unidepth_visualizations" / f"{args.dataset}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
vis_save_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving visualizations to: {vis_save_dir}")

# Load model
type_ = "l"  # available types: s, b, l
name = f"unidepth-v2-vit{type_}14"
model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

# set interpolation mode (only V2)
model.interpolation_mode = "bilinear"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def create_rgbd_point_cloud(rgb, depth, K, save_path):
    """Create and save RGBD point cloud"""
    h, w = depth.shape
    
    # Create mesh grid for pixel coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # Get camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Convert to 3D points
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    
    # Stack to get 3D points
    points = np.stack([x, y, z], axis=-1)
    
    # Reshape to point cloud format
    points = points.reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    
    # Remove invalid points (where depth is 0 or very large)
    valid_mask = (depth.reshape(-1) > 0) & (depth.reshape(-1) < 100)
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save point cloud
    o3d.io.write_point_cloud(str(save_path), pcd)
    return pcd

im_list = dataset["images"]

for idx, img in enumerate(tqdm(im_list)):
    im_path = root / img["file_path"]
    gt_f_px = torch.tensor(img["K"][0][0])
    im_id = img["id"]
    depth_save_path = depth_save_dir / f"{im_id}.npy"
    
    # Check if we need to process this image (either depth doesn't exist or we need to save visualization)
    need_process = not depth_save_path.exists() or (idx % 50 == 0)
    
    if depth_save_path.exists() and idx % 50 != 0:
        img["depth_path"] = str(depth_save_path.relative_to(root)) 
        continue
    
    # Load and preprocess an image using PIL
    rgb = np.array(Image.open(im_path))
    
    if need_process:
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        intrinsics_torch = torch.from_numpy(np.array(img['K']))
        camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
        
        # infer method of V1 uses still the K matrix as input
        if isinstance(model, (UniDepthV2old, UniDepthV1)):
            camera = camera.K.squeeze(0)

        # Run inference
        predictions = model.infer(rgb_torch, camera)
        depth = predictions["depth"].squeeze().cpu().numpy()  # [H, W]

        depth_numpy = depth.astype(np.float32)
        
        # Save depth if not exists
        if not depth_save_path.exists():
            np.save(depth_save_path, depth_numpy)
        else:
            # Load existing depth if we're just doing visualization
            depth_numpy = np.load(depth_save_path)
        
        img["depth_path"] = str(depth_save_path.relative_to(root))
        
        # Save visualization every 50 images
        if idx % 50 == 0:
            print(f"\nSaving visualization for image {idx}: {im_id}")
            
            # Create subfolder for this checkpoint
            checkpoint_dir = vis_save_dir / f"checkpoint_{idx:06d}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save RGB image
            rgb_save_path = checkpoint_dir / f"{im_id}_rgb.png"
            Image.fromarray(rgb).save(rgb_save_path)
            
            # Save depth visualization
            depth_tensor = torch.from_numpy(depth_numpy).unsqueeze(0).unsqueeze(0)
            depth_colorized = colorize(depth_tensor)
            
            # Handle different possible output shapes from colorize
            if depth_colorized.dim() == 4:  # [B, C, H, W]
                depth_vis = (depth_colorized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            elif depth_colorized.dim() == 3:  # [C, H, W]
                depth_vis = (depth_colorized.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            elif depth_colorized.dim() == 2:  # [H, W] - grayscale
                depth_vis = (depth_colorized.cpu().numpy() * 255).astype(np.uint8)
                depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)  # Convert to RGB
            else:
                # Fallback: normalize depth directly without colorize
                depth_norm = (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min() + 1e-8)
                depth_vis = (depth_norm * 255).astype(np.uint8)
                depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)  # Convert to RGB
            
            # Ensure depth_vis has the correct shape and type
            if len(depth_vis.shape) == 2:  # If still grayscale
                depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
            
            # Ensure it's contiguous and has the right dtype
            depth_vis = np.ascontiguousarray(depth_vis, dtype=np.uint8)
            
            # If it's single channel but has shape (H, W, 1), convert to (H, W, 3)
            if len(depth_vis.shape) == 3 and depth_vis.shape[2] == 1:
                depth_vis = np.repeat(depth_vis, 3, axis=2)
            
            depth_vis_save_path = checkpoint_dir / f"{im_id}_depth.png"
            Image.fromarray(depth_vis).save(depth_vis_save_path)
            
            # Save RGBD point cloud
            pcd_save_path = checkpoint_dir / f"{im_id}_pointcloud.ply"
            K_matrix = np.array(img['K'])
            create_rgbd_point_cloud(rgb, depth_numpy, K_matrix, pcd_save_path)
            
            # Save combined visualization - create side-by-side image manually
            # Create a side-by-side visualization without using image_grid
            h, w = rgb.shape[:2]
            combined_vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
            
            # Place RGB on the left
            combined_vis[:, :w, :] = rgb
            
            # Place depth visualization on the right
            combined_vis[:, w:, :] = depth_vis
            
            combined_save_path = checkpoint_dir / f"{im_id}_combined.png"
            Image.fromarray(combined_vis).save(combined_save_path)
            
            # Save metadata
            metadata = {
                "image_id": im_id,
                "image_index": idx,
                "image_path": str(im_path.relative_to(root)),
                "K": img["K"],
                "depth_range": [float(depth_numpy.min()), float(depth_numpy.max())]
            }
            metadata_path = checkpoint_dir / f"{im_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
    else:
        img["depth_path"] = str(depth_save_path.relative_to(root)) 

dataset["images"] = im_list
with open(dataset_save_path, "w") as f:
    json.dump(dataset, f)






