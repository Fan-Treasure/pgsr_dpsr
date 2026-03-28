from typing import Dict, Iterable, Optional
import pymeshlab as ml
import numpy as np
import plyfile


def _apply_filter_candidates(ms, candidates: Iterable[str], kwargs: Optional[Dict] = None) -> bool:
    kwargs = kwargs or {}
    for name in candidates:
        try:
            ms.apply_filter(name, **kwargs)
            return True
        except (RuntimeError, ValueError, TypeError, AttributeError):
            continue
    return False


def clean_and_repair(
    input_mesh_path: str,
    output_mesh_path: str,
    max_hole_size: int = 100,
) -> str:
    """Fill small holes on mesh via PyMeshLab and save to output path.

    Args:
        input_mesh_path: Source mesh path (.ply/.obj/...)
        output_mesh_path: Repaired mesh path to write.
        max_hole_size: Hole-size threshold passed to MeshLab close-holes filter.
    """

    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)

    # Basic clean-up before hole closing.

    closed = _apply_filter_candidates(ms, ["meshing_close_holes", "close_holes"], kwargs={"maxholesize": int(max_hole_size)})
    if not closed:
        raise RuntimeError("Failed to apply close_holes filter with current PyMeshLab version.")

    # Recompute normals when available.
    #_apply_filter_candidates(ms, ["compute_normal_per_vertex", "meshing_re_orient_faces_coherently"])
    _apply_filter_candidates(ms, ["meshing_merge_close_vertices", "merge_close_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_vertices", "remove_duplicate_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_faces", "remove_duplicate_faces"])
    _apply_filter_candidates(ms, ["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_null_faces", "remove_degenerate_faces"])

    # Enforce coherent orientation and per-face normals for downstream anchor loading.
    # _apply_filter_candidates(ms, ["meshing_re_orient_faces_coherently", "re_orient_faces_coherently"])
    ms.compute_normal_per_face()

    mesh = ms.current_mesh()
    vertex_matrix = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    face_matrix = np.asarray(mesh.face_matrix(), dtype=np.int32)
    face_normal_matrix = np.asarray(mesh.face_normal_matrix(), dtype=np.float64)

    if vertex_matrix.ndim != 2 or vertex_matrix.shape[1] != 3:
        raise RuntimeError(f"Unexpected vertex matrix shape: {vertex_matrix.shape}")
    if face_matrix.ndim != 2 or face_matrix.shape[1] != 3:
        raise RuntimeError(f"Unexpected face matrix shape: {face_matrix.shape}")
    if face_normal_matrix.ndim != 2 or face_normal_matrix.shape[1] != 3:
        raise RuntimeError(f"Unexpected face normal matrix shape: {face_normal_matrix.shape}")

    vertex_color_matrix = None
    try:
        vertex_color_matrix = np.asarray(mesh.vertex_color_matrix(), dtype=np.float64)
        if vertex_color_matrix.ndim != 2 or vertex_color_matrix.shape[0] != vertex_matrix.shape[0]:
            vertex_color_matrix = None
    except (RuntimeError, ValueError, AttributeError):
        vertex_color_matrix = None

    if vertex_color_matrix is not None and vertex_color_matrix.shape[1] not in (3, 4):
        vertex_color_matrix = None

    vertex_elements_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    if vertex_color_matrix is not None:
        color_channels = ["red", "green", "blue"]
        if vertex_color_matrix.shape[1] == 4:
            color_channels.append("alpha")
        for channel_name in color_channels:
            vertex_elements_dtype.append((channel_name, "u1"))

    vertex_elements = np.empty(vertex_matrix.shape[0], dtype=vertex_elements_dtype)
    vertex_elements["x"] = vertex_matrix[:, 0].astype(np.float32)
    vertex_elements["y"] = vertex_matrix[:, 1].astype(np.float32)
    vertex_elements["z"] = vertex_matrix[:, 2].astype(np.float32)
    if vertex_color_matrix is not None:
        if np.nanmax(vertex_color_matrix) <= 1.0:
            vertex_colors_u8 = np.clip(np.rint(vertex_color_matrix * 255.0), 0, 255).astype(np.uint8)
        else:
            vertex_colors_u8 = np.clip(np.rint(vertex_color_matrix), 0, 255).astype(np.uint8)
        vertex_elements["red"] = vertex_colors_u8[:, 0]
        vertex_elements["green"] = vertex_colors_u8[:, 1]
        vertex_elements["blue"] = vertex_colors_u8[:, 2]
        if vertex_color_matrix.shape[1] == 4:
            vertex_elements["alpha"] = vertex_colors_u8[:, 3]

    face_dtype = [
        ("vertex_indices", "i4", (3,)),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    face_elements = np.empty(face_matrix.shape[0], dtype=face_dtype)
    face_elements["vertex_indices"] = face_matrix.astype(np.int32)
    face_elements["nx"] = face_normal_matrix[:, 0].astype(np.float32)
    face_elements["ny"] = face_normal_matrix[:, 1].astype(np.float32)
    face_elements["nz"] = face_normal_matrix[:, 2].astype(np.float32)

    vertex_el = plyfile.PlyElement.describe(vertex_elements, "vertex")
    face_el = plyfile.PlyElement.describe(face_elements, "face")
    plyfile.PlyData([vertex_el, face_el], text=False).write(output_mesh_path)
    return output_mesh_path

def compute_padded_bbox_from_mesh(mesh_path, padding=0.10):
    """Compute a fixed cubic bbox from mesh vertices with optional padding."""
    ply = plyfile.PlyData.read(mesh_path)
    if "vertex" not in ply:
        raise ValueError(f"Mesh file has no vertex element: {mesh_path}")

    verts = np.stack(
        [
            np.asarray(ply["vertex"]["x"], dtype=np.float32),
            np.asarray(ply["vertex"]["y"], dtype=np.float32),
            np.asarray(ply["vertex"]["z"], dtype=np.float32),
        ],
        axis=1,
    )
    if verts.shape[0] == 0:
        raise ValueError(f"Mesh has zero vertices: {mesh_path}")

    pmin = np.min(verts, axis=0)
    pmax = np.max(verts, axis=0)
    center = 0.5 * (pmin + pmax)
    max_extent = float(np.max(pmax - pmin))
    half_extent = max(0.5 * max_extent * (1.0 + 2.0 * float(padding)), 1e-6)
    return center.astype(np.float32), float(half_extent)