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


def _compute_face_normals_world(verts_world: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-face normals in world coordinates."""
    fv = verts_world[faces]  # (F, 3, 3)
    fn = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=-1)
    return F.normalize(fn, dim=-1)


def render(viewpoint_camera, verts_world: torch.Tensor, faces: torch.Tensor):
    """
    MILO 风格的 mesh rasterization:
    - 使用 camera.full_proj_transform 做裁剪空间投影
    - 深度使用 view-space z（与高斯深度约定一致）
    - 法线使用 world-space 面法线（后续可在外部转 view-space 做 loss）

    Args:
        verts_world: (N, 3) world-space vertices
        faces: (F, 3) triangle indices

    Returns:
        rend_alpha:  (1, H, W)
        rend_normal: (3, H, W) world-space normals
        rend_depth:  (H, W) view-space z depth
    """
    device = verts_world.device
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    faces = faces.to(torch.int32).contiguous()
    ones = torch.ones((verts_world.shape[0], 1), device=device, dtype=verts_world.dtype)
    verts_h = torch.cat([verts_world, ones], dim=1)  # (N, 4)

    # Match MILO: row-vector multiply with camera.full_proj_transform.
    camera_mtx = viewpoint_camera.full_proj_transform
    pos_clip = torch.matmul(verts_h, camera_mtx).unsqueeze(0).contiguous()  # (1, N, 4)

    glctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(glctx, pos=pos_clip, tri=faces, resolution=[H, W])
    rast = rast.contiguous()

    # Alpha / raster coverage
    alpha = rast[..., 3:].clamp(0.0, 1.0)
    alpha = dr.antialias(alpha, rast, pos_clip, faces)
    alpha = alpha.squeeze(0).squeeze(-1).unsqueeze(0)  # (1, H, W)

    # Depth in view space z (same convention as MILO mesh renderer)
    view_mtx = viewpoint_camera.world_view_transform
    verts_view_z = torch.matmul(verts_h, view_mtx)[:, 2:3].contiguous()  # (N, 1)
    depth_attr = verts_view_z.unsqueeze(0).contiguous()  # (1, N, 1)
    depth_img = dr.interpolate(depth_attr, rast, faces)[0]  # (1, H, W, 1)
    depth_img = depth_img.contiguous()
    depth_img = dr.antialias(depth_img, rast, pos_clip, faces)
    depth_img = depth_img.squeeze(0).squeeze(-1)  # (H, W)

    # Vectorized: compute per-face normals, transform to view-space, then index per-pixel once.
    face_normals = _compute_face_normals_world(verts_world, faces)  # (F, 3)
    # pix_to_face stored in rast[...,3] as (1,H,W) with +1 encoding for no-hit
    pix_to_face = (rast[..., 3].to(torch.int64) - 1).squeeze(0)  # (H, W)
    valid_mask = pix_to_face >= 0

    # Visible face mask (whether a face appears in rasterization)
    face_visible = torch.zeros((faces.shape[0],), dtype=torch.bool, device=device)
    if faces.shape[0] > 0:
        vis_idx = torch.unique(pix_to_face[valid_mask])
        if vis_idx.numel() > 0:
            face_visible[vis_idx] = True

    if face_normals.shape[0] > 0:
        view_R = viewpoint_camera.world_view_transform[:3, :3]    # world->view rotation
        face_normals_view = torch.matmul(face_normals, view_R)    # (F, 3)
        face_normals_view = F.normalize(face_normals_view, dim=-1)

        safe_idx = pix_to_face.clamp(min=0, max=face_normals_view.shape[0] - 1)
        normal_img_view = face_normals_view[safe_idx.view(-1)].view(H, W, 3)
        normal_img_view[~valid_mask] = 0.0
    else:
        normal_img_view = torch.zeros((H, W, 3), dtype=verts_world.dtype, device=device)

    rend_normal = normal_img_view.permute(2, 0, 1).contiguous()    # (3, H, W)
    
    return {
        "rend_alpha": alpha,
        "rend_normal": rend_normal,  # (3, H, W)
        "rend_depth": depth_img,                      # (H, W)
        "pix_to_face": pix_to_face,                   # (H, W)  -1 for background
        "face_visible": face_visible,                 # (F,)
    }
 