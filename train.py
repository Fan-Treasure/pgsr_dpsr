#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import (
    l1_loss,
    ssim,
    lncc,
    get_img_grad_weight,
    milo_mesh_depth_loss_log,
    milo_mesh_normal_loss_absdot,
)
from gaussian_renderer import render
import sys, time
from scene import Scene, GaussianModel
import cv2
import uuid
from tqdm import tqdm
from utils.general_utils import safe_state, freeze_gaussians_rotation
from utils.graphics_utils import patch_offsets, patch_warp
from utils.image_utils import psnr, erode
from utils.mesh_utils import clean_and_repair, compute_padded_bbox_from_mesh, load_mesh_vertices_faces
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
from scene.mesh_model import MeshModel
from mesh_renderer import render as render_mesh_normal_depth
from utils.mesh_guided_densify_utils import compute_face_geometry
from plyfile import PlyData
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(22)
def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    mesh_init_path = os.path.join(dataset.source_path, "mesh_init.ply")
    repaired_mesh_path = os.path.join(dataset.model_path, "mesh_init_filled.ply")
    mesh_init_ply = PlyData.read(mesh_init_path)
    initial_face_count = np.asarray(mesh_init_ply["face"].data["vertex_indices"]).shape[0]
    is_scene_level_mesh = initial_face_count > 1_000_000
    init_max_faces = 1_000_000 if is_scene_level_mesh else None
    scene_level_scale_boost = min(np.sqrt(initial_face_count / 1_000_000.0), 2.0) if is_scene_level_mesh else 1.0
    if is_scene_level_mesh:
        print(f"Scene-level mesh detected: {initial_face_count} faces. ")
    clean_and_repair(
        input_mesh_path=mesh_init_path,
        output_mesh_path=repaired_mesh_path,
    )
    mesh_for_init = repaired_mesh_path
    print(f"Using mesh-based Gaussian initialization: {mesh_for_init}")
    gaussians.create_from_mesh(
        mesh_path=mesh_for_init,
        spatial_lr_scale=scene.cameras_extent,
        flatten_ratio=0.12,
        tangent_scale=0.55,
        opacity_init=0.10,
        min_scale_ratio=1e-4,
        max_scale_ratio=0.05,
        max_init_faces=init_max_faces,
        scene_level_scale_boost=scene_level_scale_boost,
    )

    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_mesh_normal_for_log = 0.0
    ema_mesh_depth_for_log = 0.0
    normal_loss, geo_loss, ncc_loss, mesh_normal_loss, mesh_depth_loss = None, None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)
    observe_count = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
    fixed_bbox_center, fixed_bbox_half_extent = compute_padded_bbox_from_mesh(mesh_for_init, padding=0.10)
    # Pre-create mesh_model to avoid repeated instantiation inside the loop
    mesh_model = MeshModel(
        grid_res=opt.grid_res_in_the_loop,
        dpsr_sig=0.0,
        density_thres=0,
        nerf_normalization=scene.nerf_normalization,
        normalization_mode="fixed_aabb",
        aabb_padding=0,
        fixed_aabb_center=fixed_bbox_center,
        fixed_aabb_half_extent=fixed_bbox_half_extent,
        device="cuda",
        mesh_backend="diffdmc",  # "diffmc" | "diffdmc" | "mc"
    )

    if not opt.no_surface_prior:
        mesh_model.build_surface_prior_from_mesh(
            mesh_for_init,
            blur_iters=4,  # blur_iters：越大，avg_pool3d 模糊越多，先验会向外“膨胀”，有效区域更大
            prior_thresh=0.1,  # 越小，更多体素会被判为有效
            prior_temp=0.05,  # 越小，有效/无效区域的过渡越硬
            outside_value=0.5,  
        )
        mesh_model.export_surface_prior_ply(os.path.join(scene.model_path, "surface_prior_points.ply"), threshold=0.3)

    init_mesh_verts, init_mesh_faces = load_mesh_vertices_faces(
        mesh_for_init,
        max_faces=init_max_faces,
        sample_by_area=is_scene_level_mesh,
    )
    init_mesh_geometry = compute_face_geometry(init_mesh_verts, init_mesh_faces)

    def refresh_normal_flip_guidance_from_initial_mesh():
        # init_face_visible = torch.ones((init_mesh_faces.shape[0],), dtype=torch.bool, device="cuda")
        gaussians.set_mesh_guidance(
            face_centroids=init_mesh_geometry["centroids"],
            face_visible=None,
            face_normals=init_mesh_geometry["normals"],
        )
        gaussians.update_mesh_normal_cache(
            distance_ratio=opt.mesh_guidance_dist_ratio,
            normal_cos_thresh=opt.mesh_guidance_normal_cos_thresh,
        )

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # EMA decay for view-direction stats (approx. recent-window reliability)
        if opt.view_dir_decay < 1.0 and (iteration % opt.densification_interval == 1):
            gaussians.decay_view_dir_stats(opt.view_dir_decay)
        # Freeze Gaussian rotations for early iterations so they stick to mesh
        if iteration < 100:
            freeze_gaussians_rotation(gaussians, True)
        elif iteration == 101:
            freeze_gaussians_rotation(gaussians, False)
        
        if iteration == opt.mesh_from_iter:
            # 训练过程中的 mesh 还没重建出来，先用初始 mesh 初始化法线翻转缓存。
            refresh_normal_flip_guidance_from_initial_mesh()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        train_bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda").view(3, 1, 1).expand_as(gt_image)
        if viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + train_bg * (1 - gt_alpha_mask)
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 统计视角方向，用于后续修正法线
        with torch.no_grad():
            if observe_count.shape[0] != gaussians.get_xyz.shape[0]:
                observe_count = torch.zeros((gaussians.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
            gaussians.add_view_dir_stat(viewpoint_cam.camera_center, visibility_filter)
            observe_count |= visibility_filter
        
        if (iteration > opt.mesh_from_iter):
            points, normals, weights, _ = gaussians.extract_oriented_pointcloud(
                opacity_threshold=opt.mesh_opacity_threshold,
                normalize_weight=True,
                visibility_mask=observe_count > 0,
                log_weight=False,
                use_cut_points=opt.use_cut_points_for_mesh,
            )
            mesh_out = mesh_model.reconstruct(points, normals, weights)
            # 用导出的 mesh 渲染法线图和深度图
            mesh_render_pkg = render_mesh_normal_depth(viewpoint_cam, mesh_out["verts"], mesh_out["faces"])
            mesh_normal = mesh_render_pkg["rend_normal"]
            mesh_depth = mesh_render_pkg["rend_depth"].squeeze(0)
        
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss（展平3DGS）
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()

        # MILO-style mesh depth/normal losses (use gaussian rendered_normal and plane_depth)
        if ( 'mesh_depth' in locals() and 'mesh_normal' in locals() ):
            gauss_depth = render_pkg["plane_depth"].squeeze()
            gauss_normal = render_pkg["rendered_normal"]  # depth_normal or rendered_normal
            if opt.detach_gaussian_rendering:
                gauss_depth = gauss_depth.detach()
                gauss_normal = gauss_normal.detach()
            mesh_mask = (mesh_depth > 0)
            gauss_mask = (gauss_depth > 0)
            raster_mask = mesh_mask & gauss_mask
            if opt.mesh_depth_weight >= 0.0:
                mesh_depth_loss, mesh_depth_map = milo_mesh_depth_loss_log(
                    mesh_depth=mesh_depth,
                    gaussians_depth=gauss_depth,
                    spatial_lr_scale=gaussians.spatial_lr_scale,
                    raster_mask=raster_mask,
                )
                mesh_depth_loss = opt.mesh_depth_weight * mesh_depth_loss
                loss += mesh_depth_loss
            if opt.mesh_normal_weight >= 0.0:
                mesh_normal_loss, mesh_normal_map = milo_mesh_normal_loss_absdot(
                    mesh_normal_view=mesh_normal,
                    gaussians_normal_view=gauss_normal,
                    raster_mask=raster_mask,
                )
                mesh_normal_loss = opt.mesh_normal_weight * mesh_normal_loss
                loss += mesh_normal_loss
            
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                
                if (iteration % 500 == 0) or (iteration == 200):
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    # 第一行：左->右 = 真实图像（gt_img_show）、渲染结果（img_show）、渲染法线可视化（normal_show）、渲染距离热图（distance_color）
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    # 第二行：左->右 = 几何一致性/权重掩码（d_mask_show_color）、参考深度热图（depth_color）、深度法线可视化（depth_normal_show）、像素权重热图（image_weight_color）
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    # 第三行：优先显示 mesh 渲染的法线/深度（若已缓存），否则回退到多张几何损失热图（geo_loss_color）
                    # mesh normal -> same format as normal_show (H,W,3) uint8
                    
                    if ( 'mesh_depth' in locals() and 'mesh_normal' in locals() ):
                        rend_normal_t = ((mesh_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1)
                        mesh_normal_show = (rend_normal_t * 255).detach().cpu().numpy().astype(np.uint8)
                        # mesh depth -> single-channel -> apply colormap like depth_color
                        depth_arr = mesh_depth.squeeze().detach().cpu().numpy()
                        d_min, d_max = depth_arr.min(), depth_arr.max()
                        depth_norm = (depth_arr - d_min) / (d_max - d_min)
                        depth_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
                        mesh_depth_show = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                        
                        mesh_dmap = mesh_depth_map.detach().cpu().numpy()
                        # 归一化（忽略未被 raster 的像素）
                        valid = raster_mask.detach().cpu().numpy().astype(bool)
                        v = mesh_dmap[valid]
                        mn, mx = float(v.min()), float(v.max())
                        norm = (mesh_dmap - mn) / max((mx - mn), 1e-8)
                        norm_u8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
                        mesh_dmap_show = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
                        mesh_nmap = mesh_normal_map.detach().cpu().numpy()
                        valid = raster_mask.detach().cpu().numpy().astype(bool)
                        v = mesh_nmap[valid]
                        mn, mx = float(v.min()), float(v.max())
                        norm = (mesh_nmap - mn) / max((mx - mn), 1e-8)
                        norm_u8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
                        mesh_nmap_show = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
                        row2 = np.concatenate([mesh_dmap_show, mesh_depth_show, mesh_normal_show, mesh_nmap_show], axis=1)
                        image_to_show = np.concatenate([row0, row1, row2], axis=0)

                        # [NEW] 保存有向点云PLY（带权重）
                        debug_viz_path = os.path.join(scene.model_path, "debug_viz")
                        os.makedirs(debug_viz_path, exist_ok=True)
                        viz_path_oriented_pc = os.path.join(scene.model_path, "debug_viz", f"oriented_pointcloud_{iteration}.ply")
                        gaussians.save_oriented_pointcloud_ply(viz_path_oriented_pc, opacity_threshold=opt.mesh_opacity_threshold, normalize_weight=True, visibility_mask=observe_count > 0, log_weight=True, use_cut_points=opt.use_cut_points_for_mesh)
                        mesh_path = os.path.join(scene.model_path, "debug_viz", f"dpsr_diffmc_mesh_{iteration}.ply")
                        mesh_model.save_mesh_ply(mesh_path, mesh_out["verts"], mesh_out["faces"])
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
                # 在 iteration 大于 9900 且 当前相机名为 '33' 时，额外保存单张图像到 debug 目录
                '''if iteration > 5000 and viewpoint_cam.image_name == '0033':
                    special_dir = os.path.join(debug_path, "special_saves")
                    os.makedirs(special_dir, exist_ok=True)
                    # 渲染 RGB（优先使用已构造的 img_show）
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    rgb_fname = os.path.join(special_dir, f"render_rgb_iter{iteration}_{viewpoint_cam.image_name}.png")
                    cv2.imwrite(rgb_fname, img_show)
                    # 高斯法线（normal_show 已是 uint8 (H,W,3)）
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    gauss_n_fname = os.path.join(special_dir, f"gauss_normal_iter{iteration}_{viewpoint_cam.image_name}.png")
                    cv2.imwrite(gauss_n_fname, normal_show)
                    # 高斯深度（depth_color 已是 uint8 单通道/伪彩色）
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    gauss_d_fname = os.path.join(special_dir, f"gauss_depth_iter{iteration}_{viewpoint_cam.image_name}.png")
                    cv2.imwrite(gauss_d_fname, depth_color)
                    # 如果有 mesh 渲染结果，再保存 mesh 法线和 mesh 深度
                    rend_normal_t = ((mesh_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1)
                    mesh_normal_show = (rend_normal_t * 255).detach().cpu().numpy().astype(np.uint8)
                    # mesh depth -> single-channel -> apply colormap like depth_color
                    depth_arr = mesh_depth.squeeze().detach().cpu().numpy()
                    d_min, d_max = depth_arr.min(), depth_arr.max()
                    depth_norm = (depth_arr - d_min) / (d_max - d_min)
                    depth_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
                    mesh_depth_show = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                    mesh_n_fname = os.path.join(special_dir, f"mesh_normal_iter{iteration}_{viewpoint_cam.image_name}.png")
                    cv2.imwrite(mesh_n_fname, mesh_normal_show)
                    mesh_d_fname = os.path.join(special_dir, f"mesh_depth_iter{iteration}_{viewpoint_cam.image_name}.png")
                    cv2.imwrite(mesh_d_fname, mesh_depth_show)'''

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        loss.backward()
        iter_end.record()

        if iteration % 100 == 0:
            observe_count.zero_()
                
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = (0.4 * normal_loss.item() + 0.6 * ema_single_view_for_log) if normal_loss is not None else (0.0 + 0.6 * ema_single_view_for_log)
            ema_multi_view_geo_for_log = (0.4 * geo_loss.item() + 0.6 * ema_multi_view_geo_for_log) if geo_loss is not None else (0.0 + 0.6 * ema_multi_view_geo_for_log)
            ema_multi_view_pho_for_log = (0.4 * ncc_loss.item() + 0.6 * ema_multi_view_pho_for_log) if ncc_loss is not None else (0.0 + 0.6 * ema_multi_view_pho_for_log)
            ema_mesh_normal_for_log = (0.4 * mesh_normal_loss.item() + 0.6 * ema_mesh_normal_for_log) if mesh_normal_loss is not None else (0.0 + 0.6 * ema_mesh_normal_for_log)
            ema_mesh_depth_for_log = (0.4 * mesh_depth_loss.item() + 0.6 * ema_mesh_depth_for_log) if mesh_depth_loss is not None else (0.0 + 0.6 * ema_mesh_depth_for_log)
            if iteration % 10 == 0:
                loss_dict = {
                    "Img": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "G-M_Normal": f"{ema_mesh_normal_for_log:.{5}f}",
                    "G-M_Depth": f"{ema_mesh_depth_for_log:.{5}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            if (iteration > opt.mesh_from_iter) and (iteration % opt.densification_interval) == 0:
                face_visible = mesh_render_pkg["face_visible"]
                faces = mesh_out["faces"]
                verts = mesh_out["verts"]
                # Store mesh visibility and face normals for the post-clone/split guided densify pass.
                face_geometry = compute_face_geometry(verts, faces)
                gaussians.set_mesh_guidance(
                    face_centroids=face_geometry["centroids"],
                    face_visible=face_visible,
                    face_normals=face_geometry["normals"],
                )
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if (iteration > opt.densify_from_iter) and (iteration % opt.densification_interval == 0) and (iteration > opt.mesh_from_iter):                                          
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # Mesh-guided densification: always enabled when mesh outputs are available.
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_abs_grad_threshold,
                        opt.opacity_cull_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        mesh_guide=True
                    )
                elif (iteration > opt.densify_from_iter) and (iteration % opt.densification_interval) == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_abs_grad_threshold,
                        opt.opacity_cull_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        mesh_guide=False
                    )
                        
            if iteration > opt.mesh_from_iter and iteration % opt.densification_interval == 0:
                # 法向翻转缓存必须在 densify/prune 之后重建，才能覆盖新增/删除后的高斯状态。
                if opt.use_initial_mesh_for_normal_flip:
                    gaussians.clear_mesh_guidance()
                    refresh_normal_flip_guidance_from_initial_mesh()
                else:
                    gaussians.update_mesh_normal_cache(
                        distance_ratio=opt.mesh_guidance_dist_ratio,
                        normal_cos_thresh=opt.mesh_guidance_normal_cos_thresh,
                    )
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")