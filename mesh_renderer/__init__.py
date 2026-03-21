#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex normals (view space). verts: (N,3) view-space, faces: (M,3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    vn = F.normalize(vn + 1e-9, dim=1)
    return vn


def render(viewpoint_camera, verts_world: torch.Tensor, faces: torch.Tensor):
    """
    使用 nvdiffrast 渲染网格法线图与深度图（无需 UV/纹理）。
    输入:
        verts_world: (N, 3) 世界坐标
        faces: (M, 3) 三角形索引
    输出:
        rend_alpha: (1, H, W)
        rend_normal: (3, H, W) 世界坐标法线
        rend_depth: (H, W) 视空间深度（取 |z|）
    """
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    view = viewpoint_camera.world_view_transform.transpose(0, 1)
    proj = viewpoint_camera.projection_matrix.transpose(0, 1)
    
    # Convert camera MVP from D3D/GS NDC (z∈[0,1]) to OpenGL NDC (z∈[-1,1])
    proj_glndc = proj.clone()
    proj_glndc[2, :] = proj_glndc[2, :] * 2.0 - proj_glndc[3, :]

    # Rasterization preparation
    glctx = dr.RasterizeCudaContext()
    faces = faces.to(torch.int32).contiguous()
    ones = torch.ones((verts_world.shape[0], 1), device=verts_world.device, dtype=verts_world.dtype)
    v4 = torch.cat([verts_world, ones], dim=1)

    def rast_with_proj(proj_mat):
        mvp = proj_mat @ view
        pos_clip = (v4 @ mvp.t()).unsqueeze(0).contiguous()
        rast, triangle_id = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))
        cov = (rast[..., 3] > 0).float().mean().item()
        return pos_clip, rast, cov, triangle_id

    pos_clip, rast, cov, triangle_id = rast_with_proj(proj_glndc)
    
    # Depth and normal (normal in camera view space, will be converted to world space during training)
    # View space coordinates (using mesh vertices)
    ones = torch.ones((verts_world.shape[0], 1), device=verts_world.device, dtype=verts_world.dtype)
    v4_view = torch.cat([verts_world, ones], dim=1) @ view
    verts_view = v4_view[:, :3]
    v4_view2 = torch.cat([verts_world, ones], dim=1) @ view.t()
    verts_view2 = v4_view2[:, :3]
    dist_attr = verts_view2.norm(dim=1, keepdim=True).unsqueeze(0).contiguous() 
    phys_depth = dr.interpolate(dist_attr, rast, faces)[0][..., 0]
    
    # Vertex normals
    vnorm = _compute_vertex_normals(verts_view, faces) 
    vnorm_attr = vnorm.unsqueeze(0) 
    vnorm_attr = vnorm_attr.contiguous()
    rend_normal = dr.interpolate(vnorm_attr, rast, faces)[0]
    if rend_normal.dim() == 4:
        rend_normal = rend_normal.squeeze(0)
    rend_normal = F.normalize(rend_normal, dim=2)
    rend_normal = rend_normal.permute(2, 0, 1)
    # Convert to world coordinates
    Rvw = viewpoint_camera.world_view_transform[:3, :3]
    rend_normal = (rend_normal.permute(1, 2, 0) @ Rvw.T).permute(2, 0, 1)

    # Background composition and alpha/mask
    alpha = rast[..., 3:].clamp(0, 1)
    alpha_aa = dr.antialias(alpha, rast, pos_clip, faces)
    if alpha_aa.dim() == 4:
        alpha_aa = alpha_aa.squeeze(0).squeeze(-1)
    alpha_aa = alpha_aa.unsqueeze(0)
   
    return {
        "rend_alpha": alpha_aa,           # (1,H,W)
        "rend_normal": rend_normal,    # (3,H,W)
        "rend_depth": phys_depth,      # (H,W) 
    }
 