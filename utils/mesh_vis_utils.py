import os
from typing import Optional

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from utils.system_utils import mkdir_p


def save_mesh_with_vertex_colors_ply(
    path: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    vertex_colors_uint8: torch.Tensor,
    face_normals: Optional[torch.Tensor] = None,
):
    """Save a triangular mesh to PLY with per-vertex RGB colors.

    Args:
        path: output .ply
        verts: (V, 3) float tensor
        faces: (F, 3) int tensor
        vertex_colors_uint8: (V, 3) uint8 tensor (RGB)
        face_normals: optional (F, 3) float tensor, written as face nx/ny/nz
    """
    mkdir_p(os.path.dirname(path))

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (V,3), got {tuple(verts.shape)}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {tuple(faces.shape)}")
    if vertex_colors_uint8.ndim != 2 or vertex_colors_uint8.shape[1] != 3:
        raise ValueError(
            f"vertex_colors_uint8 must be (V,3), got {tuple(vertex_colors_uint8.shape)}"
        )
    if vertex_colors_uint8.shape[0] != verts.shape[0]:
        raise ValueError(
            f"vertex color count mismatch: colors={vertex_colors_uint8.shape[0]} vs verts={verts.shape[0]}"
        )

    v_np = verts.detach().cpu().numpy().astype(np.float32)
    f_np = faces.detach().cpu().numpy().astype(np.int32)
    c_np = vertex_colors_uint8.detach().cpu().numpy().astype(np.uint8)

    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(v_np.shape[0], dtype=vertex_dtype)
    vertices["x"] = v_np[:, 0]
    vertices["y"] = v_np[:, 1]
    vertices["z"] = v_np[:, 2]
    vertices["red"] = c_np[:, 0]
    vertices["green"] = c_np[:, 1]
    vertices["blue"] = c_np[:, 2]

    face_elems = []

    face_dtype = [("vertex_indices", "i4", (3,))]
    if face_normals is not None:
        if face_normals.ndim != 2 or face_normals.shape[1] != 3 or face_normals.shape[0] != f_np.shape[0]:
            raise ValueError(
                f"face_normals must be (F,3), got {tuple(face_normals.shape)} vs F={f_np.shape[0]}"
            )
        face_dtype.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])

    faces_el = np.empty(f_np.shape[0], dtype=face_dtype)
    faces_el["vertex_indices"] = f_np
    if face_normals is not None:
        fn = face_normals.detach().cpu().numpy().astype(np.float32)
        faces_el["nx"] = fn[:, 0]
        faces_el["ny"] = fn[:, 1]
        faces_el["nz"] = fn[:, 2]

    ply_verts = PlyElement.describe(vertices, "vertex")
    ply_faces = PlyElement.describe(faces_el, "face")
    PlyData([ply_verts, ply_faces], text=False).write(path)


@torch.no_grad()
def vertex_colors_from_visible_faces(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_visible: torch.Tensor,
    visible_rgb=(255, 0, 0),
    invisible_rgb=(80, 80, 80),
) -> torch.Tensor:
    """Compute per-vertex colors by marking vertices belonging to visible faces.

    Returns:
        (V,3) uint8 tensor on CPU.
    """
    if face_visible.ndim != 1 or face_visible.shape[0] != faces.shape[0]:
        raise ValueError(
            f"face_visible must be (F,), got {tuple(face_visible.shape)} vs F={faces.shape[0]}"
        )

    V = int(verts.shape[0])
    colors = torch.empty((V, 3), dtype=torch.uint8, device="cpu")
    colors[:, 0] = int(invisible_rgb[0])
    colors[:, 1] = int(invisible_rgb[1])
    colors[:, 2] = int(invisible_rgb[2])

    if faces.numel() == 0:
        return colors

    vis = face_visible.to(device=faces.device)
    if vis.any():
        faces_vis = faces[vis]
        vidx = torch.unique(faces_vis.reshape(-1)).to("cpu")
        colors[vidx, 0] = int(visible_rgb[0])
        colors[vidx, 1] = int(visible_rgb[1])
        colors[vidx, 2] = int(visible_rgb[2])

    return colors
