import numpy as np


_PIXEL_GRID_CACHE = {}

def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.
    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.
    Returns:
        bool: True if array is a valid transform else False.
    """
    # check shape
    if t.shape != (4,4):
        return False

    # check all elements are real
    real_check = np.all(np.isreal(t))

    # calc intermediates
    rtr = np.matmul(t[:3, :3].T, t[:3, :3])
    rrt = np.matmul(t[:3, :3], t[:3, :3].T)

    # make rtr and rrt are identity
    inverse_check = np.isclose(np.eye(3), rtr, atol=tolerance).all() and np.isclose(np.eye(3), rrt, atol=tolerance).all()

    # check det
    det_check = np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=tolerance).all()

    # make sure last row is correct
    last_row_check = np.isclose(t[3, :3], np.zeros((1, 3)), atol=tolerance).all() and np.isclose(t[3, 3], 1.0, atol=tolerance).all()

    return real_check and inverse_check and det_check and last_row_check

def transform_concat(t1, t2):
    """[summary]
    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.
    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.
    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')

    return np.matmul(t1, t2)

def transform_point3s(t, ps):
    """Transfrom 3D points from one space to another.
    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).
    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.
    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points ps')

    # convert to homogeneous
    ps_homogeneous = np.hstack([ps, np.ones((len(ps), 1), dtype=np.float32)])
    ps_transformed = np.dot(t, ps_homogeneous.T).T

    return ps_transformed[:, :3]

def transform_inverse(t):
    """Find the inverse of the transfom.
    Args:
        t (numpy.array [4, 4]): SE3 transform.
    Raises:
        ValueError: If t is not a valid transform.
    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')

    return np.linalg.inv(t)

def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.
    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.
    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    x_map, y_map, z_map, valid_mask = depth_to_point_cloud_components(intrinsics, depth_image)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack((x_map[valid_mask], y_map[valid_mask], z_map[valid_mask]), axis=1)


def get_pixel_grids(shape):
    cache_key = tuple(shape)
    cached = _PIXEL_GRID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    height, width = shape
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u_coords, v_coords)
    _PIXEL_GRID_CACHE[cache_key] = (grid_u, grid_v)
    return grid_u, grid_v


def depth_to_point_cloud_components(intrinsics, depth_image):
    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    z_map = depth_image.astype(np.float32, copy=False)
    valid_mask = z_map > 0
    if not np.any(valid_mask):
        empty = np.zeros_like(z_map, dtype=np.float32)
        return empty, empty, z_map, valid_mask

    grid_u, grid_v = get_pixel_grids(depth_image.shape)
    x_map = (grid_u - u0) * z_map / fu
    y_map = (grid_v - v0) * z_map / fv
    return x_map, y_map, z_map, valid_mask
