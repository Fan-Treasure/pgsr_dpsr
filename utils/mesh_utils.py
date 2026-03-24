from typing import Dict, Iterable, Optional
import pymeshlab as ml
import numpy as np
from plyfile import PlyData


def _apply_filter_candidates(ms, candidates: Iterable[str], kwargs: Optional[Dict] = None) -> bool:
    kwargs = kwargs or {}
    for name in candidates:
        try:
            ms.apply_filter(name, **kwargs)
            return True
        except Exception:
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

    closed = _apply_filter_candidates(
        ms,
        ["meshing_close_holes", "close_holes"],
        kwargs={"maxholesize": int(max_hole_size)},
    )
    if not closed:
        raise RuntimeError("Failed to apply close_holes filter with current PyMeshLab version.")

    # Recompute normals when available.
    #_apply_filter_candidates(ms, ["compute_normal_per_vertex", "meshing_re_orient_faces_coherently"])
    _apply_filter_candidates(ms, ["meshing_merge_close_vertices", "merge_close_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_vertices", "remove_duplicate_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_faces", "remove_duplicate_faces"])
    _apply_filter_candidates(ms, ["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_null_faces", "remove_degenerate_faces"])

    ms.save_current_mesh(output_mesh_path)
    return output_mesh_path

def compute_padded_bbox_from_mesh(mesh_path, padding=0.10):
    """Compute a fixed cubic bbox from mesh vertices with optional padding."""
    ply = PlyData.read(mesh_path)
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