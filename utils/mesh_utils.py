from typing import Dict, Iterable, Optional


def _apply_filter_candidates(ms, candidates: Iterable[str], kwargs: Optional[Dict] = None) -> bool:
    kwargs = kwargs or {}
    for name in candidates:
        try:
            ms.apply_filter(name, **kwargs)
            return True
        except Exception:
            continue
    return False


def fill_small_holes_with_pymeshlab(
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
    try:
        import pymeshlab as ml
    except ImportError as e:
        raise ImportError("pymeshlab is required for mesh hole filling. Please install pymeshlab.") from e

    ms = ml.MeshSet()
    ms.load_new_mesh(input_mesh_path)

    # Basic clean-up before hole closing.
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_vertices", "remove_duplicate_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_duplicate_faces", "remove_duplicate_faces"])
    _apply_filter_candidates(ms, ["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"])
    _apply_filter_candidates(ms, ["meshing_remove_null_faces", "remove_degenerate_faces"])

    closed = _apply_filter_candidates(
        ms,
        ["meshing_close_holes", "close_holes"],
        kwargs={"maxholesize": int(max_hole_size)},
    )
    if not closed:
        raise RuntimeError("Failed to apply close_holes filter with current PyMeshLab version.")

    # Recompute normals when available.
    _apply_filter_candidates(ms, ["compute_normal_per_vertex", "meshing_re_orient_faces_coherently"])

    ms.save_current_mesh(output_mesh_path)
    return output_mesh_path

