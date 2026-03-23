import numpy as np
from collections import defaultdict


def build_uv_correspondence(ref_faces, ref_face_uvs, ref_uvs, pos_faces, pos_face_uvs, pos_uvs, pos_vertices):
    """
    Build correspondence between reference mesh and position mesh using UV coordinates.

    For each reference face, find the position mesh face with matching UV triangle
    (considering all 3 cyclic rotations), then assign the position vertex coordinates.

    Returns:
        vertex_colors: (num_corners, 3) array of position coordinates per corner
        valid_mask: (num_corners,) boolean mask indicating successful matches
    """
    # Build lookup: UV triangle (with rotations) -> list of vertex index tuples
    uv_to_verts = defaultdict(list)

    for face, face_uv in zip(pos_faces, pos_face_uvs):
        if any(idx < 0 for idx in face_uv):
            continue

        uvs = [tuple(pos_uvs[face_uv[k]].round(decimals=6)) for k in range(3)]
        verts = [face[0], face[1], face[2]]

        for rot in range(3):
            key = tuple(uvs[rot:] + uvs[:rot])
            uv_to_verts[key].append(tuple(verts[rot:] + verts[:rot]))

    # Deduplicate
    for key in uv_to_verts:
        uv_to_verts[key] = list(set(uv_to_verts[key]))

    # Match reference faces to position faces
    num_corners = len(ref_faces) * 3
    vertex_colors = np.zeros((num_corners, 3), dtype=np.float32)
    valid_mask = np.zeros(num_corners, dtype=bool)
    unmatched = 0

    for face_idx, (face, face_uv) in enumerate(zip(ref_faces, ref_face_uvs)):
        if any(idx < 0 for idx in face_uv):
            unmatched += 1
            continue

        ref_key = tuple(
            tuple(ref_uvs[face_uv[k]].round(decimals=6)) for k in range(3)
        )

        if ref_key in uv_to_verts:
            pos_vi = uv_to_verts[ref_key][0]
            for c in range(3):
                idx = face_idx * 3 + c
                vertex_colors[idx] = pos_vertices[pos_vi[c]]
                valid_mask[idx] = True
        else:
            unmatched += 1

    total = len(ref_faces)
    matched = total - unmatched
    if unmatched > 0:
        print(
            f"    UV matching: {matched}/{total} faces matched "
            f"({100 * unmatched / total:.1f}% unmatched)"
        )
    else:
        print(f"    UV matching: {matched}/{total} faces matched")

    return vertex_colors, valid_mask
