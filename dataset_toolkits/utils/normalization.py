import numpy as np


def rotate_vertices(vertices, degrees, axis):
    """Rotate vertices around the specified axis by the given degrees."""
    if degrees == 0:
        return vertices

    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)

    rotations = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }

    key = axis.lower()
    if key not in rotations:
        raise ValueError(f"Invalid axis: {axis}")

    return vertices @ rotations[key].T


def flip_vertices(vertices, axis):
    """Flip (mirror) vertices along the specified axis."""
    idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
    if idx is None:
        raise ValueError(f"Invalid axis: {axis}")
    flipped = vertices.copy()
    flipped[:, idx] *= -1
    return flipped


def calculate_bounds(vertices):
    """Calculate bounding box of vertices. Returns (min, max) arrays."""
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def normalize_mesh_to_unit_cube(vertices):
    """Normalize vertices to fit within a unit cube centered at origin."""
    bounds_min, bounds_max = calculate_bounds(vertices)
    center = (bounds_min + bounds_max) / 2
    scale = np.max(bounds_max - bounds_min)

    normalized = vertices - center
    if scale > 0:
        normalized /= scale
    return normalized


def calculate_uniform_scale(vertices):
    """Calculate uniform scaling factor so the mesh fits within [-0.35, 0.35]."""
    bounds_min, bounds_max = calculate_bounds(vertices)
    scale = np.max(bounds_max - bounds_min)
    target_size = 0.7
    return target_size / scale if scale > 0 else 1.0


def normalize_mesh_group_bbox(vertices_list, uniform_scale):
    """
    Normalize a group of meshes using bounding box center of the 3rd mesh.

    Returns:
        normalized_vertices: list of normalized vertex arrays
        bbox_center: center of the 3rd mesh's bounding box
    """
    ref_verts = vertices_list[2]
    bounds_min, bounds_max = calculate_bounds(ref_verts)
    bbox_center = (bounds_min + bounds_max) / 2

    normalized = []
    for vertices in vertices_list:
        normalized.append((vertices - bbox_center) * uniform_scale)

    return normalized, bbox_center


def translate_position_map(position_map, translation):
    """Translate non-zero pixels in a position map by subtracting the translation."""
    translated = position_map.copy()

    if translated.shape[-1] == 4:
        mask = np.linalg.norm(translated[:, :, :3], axis=-1) > 0
        translated[mask, :3] -= translation
    else:
        mask = np.linalg.norm(translated, axis=-1) > 0
        translated[mask] -= translation

    return translated


def check_rgb_range(position_map, name=""):
    """Assert all non-zero RGB values in the position map are within [0, 1]."""
    data = position_map[:, :, :3] if position_map.shape[-1] == 4 else position_map
    mask = np.linalg.norm(data, axis=-1) > 0
    if not np.any(mask):
        return

    valid = data[mask]
    has_negative = np.any(valid < 0)
    has_over_one = np.any(valid > 1)

    if has_negative or has_over_one:
        min_vals = np.min(valid, axis=0)
        max_vals = np.max(valid, axis=0)
        raise ValueError(
            f"{name} RGB out of range: min=[{min_vals[0]:.6f}, {min_vals[1]:.6f}, {min_vals[2]:.6f}], "
            f"max=[{max_vals[0]:.6f}, {max_vals[1]:.6f}, {max_vals[2]:.6f}]"
        )
