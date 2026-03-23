import os
import re

import numpy as np


def load_obj_with_uv(filepath):
    """
    Load OBJ file and return vertices, UVs, faces, and face_uvs.

    Returns:
        vertices: (N, 3) float32 array of vertex positions
        uvs: (M, 2) float32 array of UV coordinates, or None
        faces: (F, 3) int32 array of vertex indices
        face_uvs: (F, 3) int32 array of UV indices, or None
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face_v = []
                face_vt = []
                for part in parts:
                    indices = part.split("/")
                    face_v.append(int(indices[0]) - 1)
                    if len(indices) > 1 and indices[1]:
                        face_vt.append(int(indices[1]) - 1)
                    else:
                        face_vt.append(-1)

                # Triangulate quads
                if len(face_v) == 3:
                    faces.append(face_v)
                    face_uvs.append(face_vt)
                elif len(face_v) == 4:
                    faces.append([face_v[0], face_v[1], face_v[2]])
                    faces.append([face_v[0], face_v[2], face_v[3]])
                    face_uvs.append([face_vt[0], face_vt[1], face_vt[2]])
                    face_uvs.append([face_vt[0], face_vt[2], face_vt[3]])

    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
    faces = np.array(faces, dtype=np.int32)
    face_uvs = np.array(face_uvs, dtype=np.int32) if face_uvs else None

    return vertices, uvs, faces, face_uvs


def get_numbered_obj_files(obj_folder):
    """Get all numbered OBJ files (e.g. 00001.obj) sorted numerically."""
    obj_files = []
    for filename in os.listdir(obj_folder):
        if filename.endswith(".obj"):
            match = re.match(r"(\d+)\.obj$", filename)
            if match:
                obj_files.append((int(match.group(1)), filename))
    obj_files.sort(key=lambda x: x[0])
    return obj_files


def get_consecutive_groups(obj_folder, group_size=4):
    """Get all groups of consecutive numbered meshes from the folder."""
    numbered_files = get_numbered_obj_files(obj_folder)
    if not numbered_files:
        return []

    groups = []
    for i in range(len(numbered_files) - group_size + 1):
        is_consecutive = all(
            numbered_files[i + j][0] == numbered_files[i][0] + j
            for j in range(1, group_size)
        )
        if is_consecutive:
            group_paths = [
                os.path.join(obj_folder, numbered_files[i + j][1])
                for j in range(group_size)
            ]
            group_numbers = [numbered_files[i + j][0] for j in range(group_size)]
            group_name = f"frames_{group_numbers[0]:05d}-{group_numbers[-1]:05d}"
            groups.append((group_name, group_paths, group_numbers))

    return groups


def get_case_folders(base_folder):
    """Get all case folders that contain a 'models' subfolder."""
    if not os.path.exists(base_folder):
        print(f"Error: Base folder does not exist: {base_folder}")
        return []

    case_folders = []
    for item in sorted(os.listdir(base_folder)):
        item_path = os.path.join(base_folder, item)
        models_path = os.path.join(item_path, "models")
        if os.path.isdir(item_path) and os.path.isdir(models_path):
            case_folders.append((item, models_path))
    return case_folders
