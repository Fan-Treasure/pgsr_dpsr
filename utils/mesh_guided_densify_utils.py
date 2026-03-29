from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def compute_face_geometry(verts_world: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12):
    """计算三角面的几何信息。

    返回字典，包含面中心、法线、切线基和两个切向跨度。
    """
    if faces.dtype != torch.long:
        faces = faces.to(torch.long)

    v0 = verts_world[faces[:, 0]]
    v1 = verts_world[faces[:, 1]]
    v2 = verts_world[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0

    e01 = v1 - v0
    e02 = v2 - v0
    normals = torch.cross(e01, e02, dim=-1)
    normal_norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    # valid_mask = normal_norm[:, 0] > eps
    normals = normals / torch.clamp_min(normal_norm, eps)

    t1 = e01
    t1_norm = torch.linalg.norm(t1, dim=-1, keepdim=True)
    t1 = t1 / torch.clamp_min(t1_norm, eps)
    fallback_mask = t1_norm[:, 0] <= eps
    if torch.any(fallback_mask):
        t1 = t1.clone()
        t1[fallback_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=t1.dtype, device=t1.device)

    t2 = torch.cross(normals, t1, dim=-1)
    t2 = F.normalize(t2, dim=-1)
    t1 = torch.cross(t2, normals, dim=-1)
    t1 = F.normalize(t1, dim=-1)

    p0 = v0 - centroids
    p1 = v1 - centroids
    p2 = v2 - centroids

    def _abs_dot(p, t):
        return torch.abs(torch.sum(p * t, dim=-1))

    u0 = _abs_dot(p0, t1)
    u1 = _abs_dot(p1, t1)
    u2 = _abs_dot(p2, t1)
    v0p = _abs_dot(p0, t2)
    v1p = _abs_dot(p1, t2)
    v2p = _abs_dot(p2, t2)

    span_u = torch.maximum(torch.maximum(u0, u1), u2)
    span_v = torch.maximum(torch.maximum(v0p, v1p), v2p)

    return {
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "centroids": centroids,
        "normals": normals,
        "t1": t1,
        "t2": t2,
        "span_u": span_u,
        "span_v": span_v,
    }


def compute_face_coverage_mask(
    face_centroids: torch.Tensor,
    gaussian_xyz: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """标记哪些面片至少被一个高斯选为最近邻。

    返回：
        covered_face_mask: (F,) bool
        nn_face_idx: (G,) long，每个高斯对应的最近邻面片索引
        nn_face_dist: (G,) float，每个高斯到最近邻面片中心的距离
    """
    if face_centroids.numel() == 0 or gaussian_xyz.numel() == 0:
        covered = torch.zeros((face_centroids.shape[0],), dtype=torch.bool, device=face_centroids.device)
        return covered, None, None

    d2, idx, _ = knn_points(gaussian_xyz.unsqueeze(0), face_centroids.unsqueeze(0), K=1)
    nn_face_idx = idx[0, :, 0]
    nn_face_dist = torch.sqrt(torch.clamp_min(d2[0, :, 0], 1e-12))

    covered = torch.zeros((face_centroids.shape[0],), dtype=torch.bool, device=face_centroids.device)
    if nn_face_idx.numel() > 0:
        covered[torch.unique(nn_face_idx)] = True
    return covered, nn_face_idx, nn_face_dist


def visible_uncovered_face_ids(face_visible: torch.Tensor, covered_face_mask: torch.Tensor) -> torch.Tensor:
    """返回同时满足“可见”和“未被覆盖”的面片索引。"""
    if face_visible.ndim != 1 or covered_face_mask.ndim != 1:
        raise ValueError("face_visible and covered_face_mask must be 1D boolean tensors")
    if face_visible.shape[0] != covered_face_mask.shape[0]:
        raise ValueError(
            f"shape mismatch: face_visible={tuple(face_visible.shape)} covered={tuple(covered_face_mask.shape)}"
        )
    return torch.nonzero(face_visible & (~covered_face_mask), as_tuple=False).squeeze(1)


def build_orthonormal_frame_from_normals(normals: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """根据输入法线构造右手正交基。

    返回的旋转矩阵三列依次是 [切线1, 切线2, 法线]。
    """
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must be (N,3), got {tuple(normals.shape)}")

    normal = F.normalize(normals, dim=-1)
    device = normal.device
    dtype = normal.dtype

    ref0 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(normal)
    ref1 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(normal)

    tangent1 = torch.cross(ref0, normal, dim=-1)
    tangent1_norm = torch.linalg.norm(tangent1, dim=-1, keepdim=True)
    fallback_mask = tangent1_norm[:, 0] <= eps
    if torch.any(fallback_mask):
        tangent1 = tangent1.clone()
        tangent1[fallback_mask] = torch.cross(ref1[fallback_mask], normal[fallback_mask], dim=-1)
        tangent1_norm = torch.linalg.norm(tangent1, dim=-1, keepdim=True)

    tangent1 = tangent1 / torch.clamp_min(tangent1_norm, eps)
    tangent2 = torch.cross(normal, tangent1, dim=-1)
    tangent2 = F.normalize(tangent2, dim=-1)
    tangent1 = torch.cross(tangent2, normal, dim=-1)
    tangent1 = F.normalize(tangent1, dim=-1)

    return torch.stack([tangent1, tangent2, normal], dim=-1)
