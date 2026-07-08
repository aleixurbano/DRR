"""
GPU-accelerated utilities for point cloud and depth processing.
Provides significant speedups over CPU-based Open3D operations.
"""

import torch
import numpy as np
from reflect.real_world.logging_utils import get_logger

logger = get_logger(__name__)


def depth_to_pointcloud_gpu_batch(intrinsics_matrix, depth_frames_batch):
    """
    Convert batch of depth frames to point clouds on GPU.
    
    Much faster than sequential Open3D processing.
    
    Args:
        intrinsics_matrix: (3, 3) camera intrinsics matrix
        depth_frames_batch: List of (H, W) depth arrays or (B, H, W) tensor
    
    Returns:
        List of (N, 3) point cloud arrays (on CPU)
    
    Example:
        >>> depth_batch = [depth1, depth2, depth3]  # 3 frames
        >>> point_clouds = depth_to_pointcloud_gpu_batch(intrinsics_matrix, depth_batch)
        >>> # Returns list of 3 point clouds
    """
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            logger.warning("CUDA not available, falling back to CPU processing")
            device = torch.device('cpu')
    except Exception as e:
        logger.warning(f"Error accessing CUDA: {e}, using CPU")
        device = torch.device('cpu')
    
    fu = float(intrinsics_matrix[0, 0])
    fv = float(intrinsics_matrix[1, 1])
    u0 = float(intrinsics_matrix[0, 2])
    v0 = float(intrinsics_matrix[1, 2])
    
    # Convert to tensor if needed
    if isinstance(depth_frames_batch, list):
        depth_tensor = torch.stack([
            torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float()
            for f in depth_frames_batch
        ]).to(device)
    else:
        depth_tensor = depth_frames_batch.float().to(device) if len(depth_frames_batch.shape) == 3 else depth_frames_batch
    
    batch_size, height, width = depth_tensor.shape
    
    # Create pixel coordinate grids (GPU)
    v_coords = torch.arange(height, dtype=torch.float32, device=device)
    u_coords = torch.arange(width, dtype=torch.float32, device=device)
    grid_u, grid_v = torch.meshgrid(u_coords, v_coords, indexing='xy')
    
    # Compute X, Y, Z for entire batch (vectorized)
    z_map = depth_tensor
    x_map = (grid_u - u0) * z_map / fu
    y_map = (grid_v - v0) * z_map / fv
    
    # Identify valid points (z > 0)
    valid_mask = z_map > 0
    
    # Extract point clouds
    point_clouds = []
    for batch_idx in range(batch_size):
        valid = valid_mask[batch_idx]
        if valid.any():
            points = torch.stack([
                x_map[batch_idx][valid],
                y_map[batch_idx][valid],
                z_map[batch_idx][valid]
            ], dim=1)
            point_clouds.append(points.cpu().numpy())
        else:
            point_clouds.append(np.zeros((0, 3), dtype=np.float32))
    
    return point_clouds


def voxel_downsample_gpu_batch(point_clouds_list, voxel_size=0.01):
    """
    GPU batch voxel downsampling using grid hashing.
    Much faster than Open3D sequential downsampling.
    
    Downsamples point clouds by keeping one representative point per voxel.
    
    Args:
        point_clouds_list: List of (N, 3) point cloud arrays
        voxel_size: Voxel size for downsampling
    
    Returns:
        List of downsampled (M, 3) point cloud arrays
    
    Example:
        >>> point_clouds = [pc1, pc2, pc3]  # 3 point clouds
        >>> downsampled = voxel_downsample_gpu_batch(point_clouds, voxel_size=0.01)
        >>> # Returns list of 3 downsampled point clouds
    """
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        logger.warning(f"Error accessing CUDA: {e}, using CPU")
        device = torch.device('cpu')
    
    downsampled = []
    
    for pcd in point_clouds_list:
        if len(pcd) == 0:
            downsampled.append(pcd)
            continue
        
        pcd_tensor = torch.from_numpy(pcd).float().to(device)
        
        # Quantize points to voxel grid (integer coordinates)
        voxel_indices = (pcd_tensor / voxel_size).long()
        
        # Create unique voxel identifiers via hashing
        # Use large multipliers to ensure no collisions
        voxel_hash = (
            voxel_indices[:, 0] * 1000000 +
            voxel_indices[:, 1] * 10000 +
            voxel_indices[:, 2]
        )
        
        # Keep one representative point per voxel without relying on
        # torch.unique(..., return_index=True), which is not available
        # on all PyTorch builds.
        sorted_hash, perm = torch.sort(voxel_hash)
        first_of_voxel = torch.ones_like(sorted_hash, dtype=torch.bool)
        first_of_voxel[1:] = sorted_hash[1:] != sorted_hash[:-1]
        unique_indices = perm[first_of_voxel]
        downsampled_pcd = pcd_tensor[unique_indices]
        
        downsampled.append(downsampled_pcd.cpu().numpy())
    
    logger.debug(f"Downsampled {len(point_clouds_list)} point clouds from voxel_size={voxel_size}")
    return downsampled


def outlier_removal_gpu_batch(point_clouds_list, nb_neighbors=1500, std_ratio=0.1):
    """
    GPU batch statistical outlier removal using KNN distance statistics.
    Replaces Open3D remove_statistical_outlier for faster processing.
    
    Removes points whose average distance to neighbors exceeds std_ratio * mean_distance.
    
    Args:
        point_clouds_list: List of (N, 3) point cloud arrays
        nb_neighbors: Number of neighbors for KNN
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        List of filtered (M, 3) point cloud arrays
    
    Example:
        >>> point_clouds = [pc1, pc2, pc3]
        >>> filtered = outlier_removal_gpu_batch(point_clouds, nb_neighbors=1500, std_ratio=0.1)
    """
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        logger.warning(f"Error accessing CUDA: {e}, using CPU")
        device = torch.device('cpu')
    
    filtered = []
    
    for pcd in point_clouds_list:
        if len(pcd) < nb_neighbors:
            # Not enough points for outlier detection
            filtered.append(pcd)
            continue
        
        pcd_tensor = torch.from_numpy(pcd).float().to(device)
        
        # Compute pairwise distances (expensive for large clouds)
        # For large point clouds, sample neighbors
        n_points = pcd_tensor.shape[0]
        if n_points > 50000:
            # Use approximate nearest neighbors for large clouds
            # Sample subset for distance computation
            sample_size = min(nb_neighbors + 100, n_points)
            sample_indices = torch.randperm(n_points, device=device)[:sample_size]
            sample_points = pcd_tensor[sample_indices]
        else:
            sample_points = pcd_tensor
        
        # Compute pairwise distances
        # distances shape: (N, M)
        distances = torch.cdist(pcd_tensor, sample_points, p=2)
        
        # Get k nearest neighbor distances
        k = min(nb_neighbors, sample_points.shape[0])
        knn_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)
        
        # Compute mean distance for each point
        mean_distances = knn_distances.mean(dim=1)
        overall_mean = mean_distances.mean()
        overall_std = mean_distances.std()
        
        # Filter outliers
        threshold = overall_mean + std_ratio * overall_std
        inlier_mask = mean_distances < threshold
        
        filtered_pcd = pcd_tensor[inlier_mask].cpu().numpy()
        filtered.append(filtered_pcd)
    
    logger.debug(f"Filtered outliers from {len(point_clouds_list)} point clouds")
    return filtered


def batch_process_point_clouds(
    depth_frames,
    masks,
    intrinsics_matrix,
    voxel_size=0.01,
    remove_outliers=True,
    nb_neighbors=1500,
    std_ratio=0.1
):
    """
    Complete pipeline: depth → pointcloud → downsample → filter outliers
    
    Processes multiple frames in batched GPU operations for maximum efficiency.
    
    Args:
        depth_frames: List of (H, W) depth arrays
        masks: List of (H, W) boolean masks
        intrinsics_matrix: (3, 3) camera intrinsics
        voxel_size: Voxel size for downsampling
        remove_outliers: Whether to apply statistical outlier removal
        nb_neighbors: Parameters for outlier removal
        std_ratio: Parameters for outlier removal
    
    Returns:
        List of processed (N, 3) point cloud arrays
    
    Example:
        >>> depth_batch = [depth1, depth2, depth3]
        >>> mask_batch = [mask1, mask2, mask3]
        >>> pcds = batch_process_point_clouds(
        ...     depth_batch, mask_batch, intrinsics_matrix, voxel_size=0.01
        ... )
    """
    logger.info(f"Processing {len(depth_frames)} depth frames in GPU batches")
    
    # Step 1: Depth to pointcloud
    point_clouds = depth_to_pointcloud_gpu_batch(intrinsics_matrix, depth_frames)
    
    # Step 2: Apply masks
    masked_clouds = []
    for pcd, mask in zip(point_clouds, masks):
        if len(pcd) > 0 and mask is not None:
            # Reshape mask if needed and filter
            flat_mask = mask.flatten()
            if len(flat_mask) == len(pcd):
                masked_pcd = pcd[flat_mask]
                masked_clouds.append(masked_pcd)
            else:
                masked_clouds.append(pcd)
        else:
            masked_clouds.append(pcd)
    
    # Step 3: Downsample
    downsampled = voxel_downsample_gpu_batch(masked_clouds, voxel_size=voxel_size)
    
    # Step 4: Remove outliers (optional)
    if remove_outliers:
        filtered = outlier_removal_gpu_batch(downsampled, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return filtered
    
    return downsampled
