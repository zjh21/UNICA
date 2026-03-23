import cv2 as cv
import numpy as np


def crop_center_region(image, crop_width, crop_height=None):
    """Crop the center region from an image."""
    h, w = image.shape[:2]
    if crop_height is None:
        crop_height = h

    start_col = (w - crop_width) // 2
    start_row = (h - crop_height) // 2

    return image[start_row : start_row + crop_height, start_col : start_col + crop_width]


def rotate_image_90(image):
    """Rotate image 90 degrees clockwise."""
    return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)


def create_six_view_matrices(ref_center):
    """
    Create camera extrinsic matrices for 6 orthographic views.

    Returns dict with keys: front, back, left, right, up, down.
    """

    def make_view(rotation_vec=None):
        mv = np.identity(4, np.float32)
        if rotation_vec is not None:
            rot = cv.Rodrigues(np.array(rotation_vec, np.float32))[0]
            mv[:3, :3] = rot
            mv[:3, 3] = -rot @ ref_center + np.array([0, 0, -10], np.float32)
        else:
            mv[:3, 3] = -ref_center + np.array([0, 0, -10], np.float32)
        mv[1:3] *= -1
        return mv

    return {
        "front": make_view(),
        "back": make_view([0, np.pi, 0]),
        "left": make_view([0, -np.pi / 2, 0]),
        "right": make_view([0, np.pi / 2, 0]),
        "up": make_view([-np.pi / 2, 0, 0]),
        "down": make_view([np.pi / 2, 0, 0]),
    }


def concatenate_six_views(views_dict, render_size, final_size):
    """
    Concatenate 6 views into one position map.

    Layout:
        Top-left  (render_size x render_size): back + front side by side
        Top-right (render_size x side_size):   right view
        Bot-left  (side_size x render_size):   left view (rotated 90 deg CW)
        Bot-right (side_size x side_size):     up + down stacked
    """
    side_size = final_size - render_size
    half_render = render_size // 2
    half_side = side_size // 2

    final_map = np.zeros((final_size, final_size, 3), dtype=np.float32)

    # Front + back side by side in top-left
    front_cropped = crop_center_region(views_dict["front"], half_render, render_size)
    back_cropped = crop_center_region(views_dict["back"], half_render, render_size)
    final_map[:render_size, :render_size] = np.concatenate(
        [back_cropped, front_cropped], axis=1
    )

    # Right in top-right
    right_cropped = crop_center_region(views_dict["right"], side_size, render_size)
    final_map[:render_size, render_size:] = right_cropped

    # Left rotated 90 deg CW in bottom-left
    left_cropped = crop_center_region(views_dict["left"], side_size, render_size)
    final_map[render_size:, :render_size] = rotate_image_90(left_cropped)

    # Up + down stacked in bottom-right
    up_cropped = crop_center_region(views_dict["up"], side_size, half_side)
    down_cropped = crop_center_region(views_dict["down"], side_size, half_side)
    final_map[render_size:, render_size:] = np.concatenate(
        [up_cropped, down_cropped], axis=0
    )

    return final_map


def calculate_root_from_position_map(position_map, render_size, y_keypoint, x_ranges=None):
    """
    Calculate root position from pixels at y_keypoint in the front/back region.

    Args:
        position_map: Concatenated position map (H, W, 3 or 4)
        render_size: Width of the front/back region
        y_keypoint: Y coordinate to sample
        x_ranges: List of (start, end) tuples (inclusive), or None for full width
    """
    data = position_map[:, :, :3] if position_map.shape[-1] == 4 else position_map

    if x_ranges is None:
        row = data[y_keypoint, :render_size, :]
    else:
        segments = []
        for x_start, x_end in x_ranges:
            x_start = max(0, x_start)
            x_end = min(render_size - 1, x_end)
            if x_start <= x_end:
                segments.append(data[y_keypoint, x_start : x_end + 1, :])
        if not segments:
            return np.zeros(3, dtype=np.float32)
        row = np.concatenate(segments, axis=0)

    mask = np.linalg.norm(row, axis=-1) > 0
    if not np.any(mask):
        return np.zeros(3, dtype=np.float32)

    return np.mean(row[mask], axis=0)


def render_six_views(renderer, ref_vertices_dup, position_verts_dup, view_matrices, view_magnifications):
    """Render 6 views and return as dict of RGB numpy arrays."""
    views = {}
    for view_name, view_matrix in view_matrices.items():
        xmag, ymag = view_magnifications[view_name]
        renderer.set_camera(view_matrix, xmag=xmag, ymag=ymag)
        renderer.set_model(ref_vertices_dup, position_verts_dup)
        rendered = renderer.render()
        views[view_name] = cv.cvtColor(rendered, cv.COLOR_BGRA2RGB)
    return views
