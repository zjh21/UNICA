"""
Position Map Rendering Pipeline

Renders 6-view position maps from OBJ meshes. Optionally reorganizes the output
into a dataset-ready folder structure with motion labels (when LABEL_ROOT is set).

Usage:
    python dataset_toolkits/render_position_maps.py
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import traceback

import cv2 as cv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.renderer import PositionMapRenderer
from utils.mesh_io import load_obj_with_uv, get_consecutive_groups, get_case_folders
from utils.normalization import (
    rotate_vertices,
    flip_vertices,
    calculate_uniform_scale,
    normalize_mesh_to_unit_cube,
    normalize_mesh_group_bbox,
    translate_position_map,
    check_rgb_range,
)
from utils.six_view import (
    create_six_view_matrices,
    concatenate_six_views,
    calculate_root_from_position_map,
    render_six_views,
)
from utils.uv_matching import build_uv_correspondence
from utils.reorganize import get_motion_label_for_group


def process_group(
    group_name,
    group_paths,
    group_numbers,
    ref_data,
    uniform_scale,
    renderer,
    output_folder,
    render_size,
    final_size,
    y_keypoint,
    x_ranges=None,
    default_xmag=0.65,
    default_ymag=0.65,
    updown_xmag=1.1,
    updown_ymag=1.1,
):
    """
    Process a group of position meshes: render 6-view position maps and save as EXR.

    Files are saved as 1.exr, 2.exr, ... in output_folder.
    """
    print(f"\nProcessing group: {group_name}")

    ref_vertices, ref_uvs, ref_faces, ref_face_uvs = ref_data
    ref_num_vertices = len(ref_vertices)

    # Skip if already processed
    if os.path.exists(output_folder):
        all_exist = all(
            os.path.exists(os.path.join(output_folder, f"{i + 1}.exr"))
            for i in range(len(group_numbers))
        )
        if all_exist:
            print(f"  Already processed, skipping: {output_folder}")
            return

    os.makedirs(output_folder, exist_ok=True)

    # Load and transform position meshes
    meshes_data = []
    for path in group_paths:
        verts, uvs, faces, face_uvs = load_obj_with_uv(path)
        verts = rotate_vertices(verts, -90, "z")
        verts = rotate_vertices(verts, 90, "x")
        verts = flip_vertices(verts, "z")
        meshes_data.append((verts, uvs, faces, face_uvs))

    # Determine matching strategy
    use_direct = all(len(d[0]) == ref_num_vertices for d in meshes_data)
    if use_direct:
        print(f"  Direct vertex assignment ({ref_num_vertices} vertices)")
    else:
        print(f"  UV matching (vertex count mismatch)")

    # Normalize vertex positions using 3rd mesh's bbox center
    verts_list = [d[0] for d in meshes_data]
    normalized_verts, bbox_center = normalize_mesh_group_bbox(verts_list, uniform_scale)
    for i in range(len(meshes_data)):
        meshes_data[i] = (
            normalized_verts[i],
            meshes_data[i][1],
            meshes_data[i][2],
            meshes_data[i][3],
        )

    # Setup 6-view cameras
    ref_center = 0.5 * (ref_vertices.min(0) + ref_vertices.max(0))
    view_matrices = create_six_view_matrices(ref_center)
    view_mags = {
        "front": (default_xmag, default_ymag),
        "back": (default_xmag, default_ymag),
        "left": (default_xmag, default_ymag),
        "right": (default_xmag, default_ymag),
        "up": (updown_xmag, updown_ymag),
        "down": (updown_xmag, updown_ymag),
    }

    def get_vertex_data(mesh_idx):
        """Get duplicated vertex data for rendering (position + reference)."""
        pos_verts, pos_uvs_m, pos_faces, pos_face_uvs = meshes_data[mesh_idx]
        if use_direct:
            return (
                pos_verts[ref_faces.reshape(-1)],
                ref_vertices[ref_faces.reshape(-1)],
            )
        else:
            if pos_uvs_m is None or ref_uvs is None:
                raise RuntimeError("UV coordinates required for UV matching")
            colors, mask = build_uv_correspondence(
                ref_faces, ref_face_uvs, ref_uvs,
                pos_faces, pos_face_uvs, pos_uvs_m, pos_verts,
            )
            colors[~mask] = 0.0
            return colors, ref_vertices[ref_faces.reshape(-1)]

    # Calculate root translation from 3rd mesh
    print(f"  Computing root from 3rd mesh (y={y_keypoint})...")
    pos_dup, ref_dup = get_vertex_data(2)
    views = render_six_views(renderer, ref_dup, pos_dup, view_matrices, view_mags)
    concat_map = concatenate_six_views(views, render_size, final_size)
    root_translation = calculate_root_from_position_map(
        concat_map, render_size, y_keypoint, x_ranges
    )

    # Render all meshes and save
    print(f"  Rendering {len(group_numbers)} position maps...")
    for i in range(len(meshes_data)):
        pos_dup, ref_dup = get_vertex_data(i)
        views = render_six_views(renderer, ref_dup, pos_dup, view_matrices, view_mags)
        concat_map = concatenate_six_views(views, render_size, final_size)

        # Normalize: subtract root, shift to [0, 1]
        translated = translate_position_map(concat_map, root_translation)
        mask = np.linalg.norm(translated, axis=-1) > 0
        translated[mask] += 0.5

        check_rgb_range(translated, f"mesh {i + 1}")

        output_path = os.path.join(output_folder, f"{i + 1}.exr")
        cv.imwrite(output_path, translated)
        print(f"    Saved: {output_path}")


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    POSITION_MESHES_BASE_FOLDER = r"/media/test/sdc1/zjh_workspace/ADNAS/Dataset/berserker_s1/meshes"
    REFERENCE_MESH_PATH = r"/media/test/sdc1/zjh_workspace/ADNAS/Dataset/berserker_s1/A-pose.obj"
    OUTPUT_BASE_FOLDER = r"/media/test/sdc1/zjh_workspace/ADNAS/Dataset/berserker_s1/posmap_debug"

    # Set LABEL_ROOT to enable reorganized output structure.
    # When set, output folders are: {OUTPUT_BASE_FOLDER}/{case}_{id:05d}_{motion}/
    # When None, output folders are: {OUTPUT_BASE_FOLDER}/{case}/{group_name}/
    LABEL_ROOT = POSITION_MESHES_BASE_FOLDER

    # Rendering parameters
    RENDER_SIZE = 96  # Size of each rendered view
    FINAL_SIZE = 128  # Size of the final concatenated position map
    Y_KEYPOINT = 38  # Y coordinate for root keypoint calculation
    X_RANGES = [(18, 30), (65, 78)]  # X coordinate ranges for root calculation (inclusive)
    GROUP_SIZE = 4  # Number of consecutive frames per group

    # Camera magnification
    DEFAULT_XMAG = 0.75
    DEFAULT_YMAG = 0.75
    UPDOWN_XMAG = 1.1
    UPDOWN_YMAG = 1.1
    # =========================================================================

    print("Position map rendering pipeline")
    print(f"  Meshes:     {POSITION_MESHES_BASE_FOLDER}")
    print(f"  Reference:  {REFERENCE_MESH_PATH}")
    print(f"  Output:     {OUTPUT_BASE_FOLDER}")
    print(f"  Resolution: {RENDER_SIZE} -> {FINAL_SIZE}")
    if LABEL_ROOT:
        print(f"  Reorganize: Yes (label_root={LABEL_ROOT})")
    else:
        print(f"  Reorganize: No (set LABEL_ROOT to enable)")

    case_folders = get_case_folders(POSITION_MESHES_BASE_FOLDER)
    if not case_folders:
        print("No case folders found!")
        return

    print(f"\nFound {len(case_folders)} cases")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    renderer = PositionMapRenderer(RENDER_SIZE, RENDER_SIZE, device=device)

    # Load and prepare reference mesh
    ref_vertices, ref_uvs, ref_faces, ref_face_uvs = load_obj_with_uv(
        REFERENCE_MESH_PATH
    )
    print(f"Reference mesh: {len(ref_vertices)} verts, {len(ref_faces)} faces")

    uniform_scale = calculate_uniform_scale(ref_vertices)
    ref_vertices = normalize_mesh_to_unit_cube(ref_vertices)
    ref_data = (ref_vertices, ref_uvs, ref_faces, ref_face_uvs)

    total_processed = 0
    total_skipped = 0

    for case_name, models_path in case_folders:
        print(f"\n{'=' * 60}")
        print(f"Case: {case_name}")

        groups = get_consecutive_groups(models_path, group_size=GROUP_SIZE)
        if not groups:
            print(f"  No groups found, skipping.")
            continue

        print(f"  {len(groups)} groups")

        # Sequential group_id for reorganized naming
        group_id = 1

        for group_name, group_paths, group_numbers in groups:
            try:
                # Determine output folder
                if LABEL_ROOT:
                    motion = get_motion_label_for_group(
                        LABEL_ROOT, case_name, group_numbers
                    )
                    if motion is None:
                        print(f"  Skipping {group_name}: no motion label found")
                        total_skipped += 1
                        continue
                    folder_name = f"{case_name}_{group_id:05d}_{motion}"
                    out_folder = os.path.join(OUTPUT_BASE_FOLDER, folder_name)
                    group_id += 1
                else:
                    out_folder = os.path.join(
                        OUTPUT_BASE_FOLDER, case_name, group_name
                    )

                process_group(
                    group_name,
                    group_paths,
                    group_numbers,
                    ref_data,
                    uniform_scale,
                    renderer,
                    out_folder,
                    RENDER_SIZE,
                    FINAL_SIZE,
                    Y_KEYPOINT,
                    X_RANGES,
                    DEFAULT_XMAG,
                    DEFAULT_YMAG,
                    UPDOWN_XMAG,
                    UPDOWN_YMAG,
                )
                total_processed += 1
            except Exception as e:
                print(f"  Error in {group_name}: {e}")
                traceback.print_exc()
                raise

    print(f"\n{'=' * 60}")
    print(f"Done. Processed: {total_processed}, Skipped: {total_skipped}")
    print(f"Output: {OUTPUT_BASE_FOLDER}")


if __name__ == "__main__":
    main()
