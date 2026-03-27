import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from nvdiffrast_utils.dpsr import DPSR
from nvdiffrast_utils.dpsr_utils import mc_from_psr, point_rasterize
from utils.system_utils import mkdir_p
from diso import DiffMC, DiffDMC

try:
    import kaolin as kal
except ImportError:
    kal = None

SMALL_NUMBER = 1e-6


class MeshModel:
    """Wrap DPSR + meshing backends for one-shot mesh extraction from oriented point clouds."""

    def __init__(
        self,
        grid_res=256,
        dpsr_sig=0.5,
        density_thres=0.0,
        nerf_normalization=None,
        normalization_mode="aabb",
        aabb_padding=0.05,
        fixed_aabb_center=None,
        fixed_aabb_half_extent=None,
        device="cuda",
        mesh_backend="diffmc",
    ):
        self.device = device
        self.grid_res = grid_res
        self.mesh_backend = str(mesh_backend).lower()
        self.dpsr = DPSR(res=(grid_res, grid_res, grid_res), sig=dpsr_sig).to(device)
        self.density_thres = torch.tensor([density_thres], dtype=torch.float32, device=device)
        self.normalization_mode = normalization_mode
        self.aabb_padding = float(aabb_padding)
        self.fixed_aabb_center = None
        self.fixed_aabb_half_extent = None
        self.surface_prior = None
        self.prior_thresh = 0.1
        self.prior_temp = 0.05
        self.prior_outside_value = 0.5

        if nerf_normalization is None:
            nerf_normalization = {"translate": np.zeros(3, dtype=np.float32), "radius": 1.0}
        self.translate = torch.tensor(
            nerf_normalization.get("translate", np.zeros(3, dtype=np.float32)),
            dtype=torch.float32,
            device=device,
        )
        self.radius = torch.tensor(
            [float(nerf_normalization.get("radius", 1.0))],
            dtype=torch.float32,
            device=device,
        )

        self.fixed_aabb_center = torch.tensor(fixed_aabb_center, dtype=torch.float32, device=device)
        self.fixed_aabb_half_extent = torch.tensor(
            [float(fixed_aabb_half_extent)], dtype=torch.float32, device=device
        )

        self.diffmc = DiffMC(dtype=torch.float32).to(device)
        self.diffdmc = DiffDMC(dtype=torch.float32).to(device)

    def _get_normalization_params(self, points_world):
        if self.normalization_mode == "fixed_aabb":
            center = self.fixed_aabb_center
            half_extent = torch.clamp(self.fixed_aabb_half_extent.squeeze(0), min=SMALL_NUMBER)
            return center, half_extent

        if self.normalization_mode == "aabb":
            pmin = torch.min(points_world, dim=0).values
            pmax = torch.max(points_world, dim=0).values
            center = 0.5 * (pmin + pmax)
            max_extent = torch.max(pmax - pmin)
            half_extent = 0.5 * max_extent * (1.0 + 2.0 * self.aabb_padding)
            half_extent = torch.clamp(half_extent, min=SMALL_NUMBER)
            return center, half_extent

        if self.normalization_mode == "nerf_normalization":
            center = -self.translate
            half_extent = torch.clamp(self.radius.squeeze(0), min=SMALL_NUMBER)
            return center, half_extent

        # Fallback: use nerf normalization if unknown mode
        center = -self.translate
        half_extent = torch.clamp(self.radius.squeeze(0), min=SMALL_NUMBER)
        return center, half_extent

    def build_surface_prior_from_mesh(
        self,
        mesh_path,
        blur_iters=3,  # 用 avg_pool3d 进行多少次模糊迭代；越大支持区越“膨胀”、越平滑
        prior_thresh=0.1,  # 后续用于计算 keep 权重的阈值
        prior_temp=0.05,
        outside_value=0.5,
    ):
        """Build a soft support volume from initial mesh vertices for PSR correction."""
        # 读取初始 mesh 的顶点坐标，作为先验的几何来源
        ply = PlyData.read(mesh_path)

        # 提取顶点的 x/y/z 坐标，组织成 (N, 3) 的世界坐标点云
        verts = np.stack(
            [
                np.asarray(ply["vertex"]["x"], dtype=np.float32),
                np.asarray(ply["vertex"]["y"], dtype=np.float32),
                np.asarray(ply["vertex"]["z"], dtype=np.float32),
            ],
            axis=1,
        )
        verts_t = torch.tensor(verts, dtype=torch.float32, device=self.device)
        # 按当前 normalization_mode 求出中心和半边长，再把顶点映射到 DPSR 的 [0,1] 坐标系
        center, half_extent = self._get_normalization_params(verts_t)
        verts_dpsr = self.normalize_points(verts_t, center, half_extent)

        # 每个顶点先赋值为 1，再 rasterize 到体素网格中，形成初始的占据/支持体素场
        vals = torch.ones((1, verts_dpsr.shape[0], 1), dtype=torch.float32, device=self.device)
        prior = point_rasterize(
            verts_dpsr.unsqueeze(0),
            vals,
            (self.grid_res, self.grid_res, self.grid_res),
        )[0, 0]

        # 用 3D 平均池化做若干次平滑，让先验区域稍微扩展一点，避免太稀疏
        prior = prior.unsqueeze(0).unsqueeze(0)
        for _ in range(max(int(blur_iters), 0)):
            prior = F.avg_pool3d(prior, kernel_size=3, stride=1, padding=1)
        prior = prior.squeeze(0).squeeze(0)

        # 再做一次最大值归一化，把先验压到 [0,1]，便于后续 sigmoid 融合
        prior_max = torch.max(prior)
        if prior_max > 0:
            prior = prior / prior_max

        # 保存先验体素场，并记录后续导出和反归一化所需的参数
        self.surface_prior = prior.detach()
        self.prior_center = center.clone().detach()
        self.prior_half_extent = half_extent.clone().detach()
        # 这些参数控制后续在 _build_psr 中如何把“有支持/无支持”的区域软融合
        self.prior_thresh = float(prior_thresh)
        self.prior_temp = max(float(prior_temp), 1e-6)
        self.prior_outside_value = float(outside_value)

    def export_surface_prior_ply(self, path, threshold=0.5):
        """Export voxels judged near-surface as a point-cloud PLY.

        - `path`: output PLY path
        - `threshold`: keep voxels with prior > threshold

        The method converts voxel indices to normalized [0,1] coordinates (voxel centers)
        then maps them back to world coordinates using the same center/half_extent used
        when building the prior, and writes a vertex-only PLY.
        """
        if self.surface_prior is None:
            raise ValueError("surface_prior is None; call build_surface_prior_from_mesh first")

        # ensure we have stored normalization params
        if not hasattr(self, "prior_center") or not hasattr(self, "prior_half_extent"):
            raise ValueError("prior center/half_extent not found; rebuild prior to record them")

        mask = (self.surface_prior > float(threshold))
        nz = torch.nonzero(mask, as_tuple=False)
        mkdir_p(os.path.dirname(path))

        vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        if nz.shape[0] == 0:
            # write empty vertex-only PLY
            verts = np.empty(0, dtype=vertex_dtype)
            ply_verts = PlyElement.describe(verts, "vertex")
            PlyData([ply_verts], text=False).write(path)
            return

        # torch.nonzero 返回的维度顺序与张量本身一致，这里就是 (x, y, z)
        dx = nz[:, 0].to(torch.float32)
        dy = nz[:, 1].to(torch.float32)
        dz = nz[:, 2].to(torch.float32)
        D = float(self.grid_res)

        # voxel center in normalized coords [0,1]
        p_norm = torch.stack([(dx + 0.5) / D, (dy + 0.5) / D, (dz + 0.5) / D], dim=1)
        p_norm = p_norm.to(self.prior_center.device)

        # convert to world coords using same transform as _to_world
        verts_world = self._to_world(p_norm, self.prior_center, self.prior_half_extent)

        v_np = verts_world.detach().cpu().numpy()
        vertices = np.empty(v_np.shape[0], dtype=vertex_dtype)
        vertices["x"] = v_np[:, 0]
        vertices["y"] = v_np[:, 1]
        vertices["z"] = v_np[:, 2]

        ply_verts = PlyElement.describe(vertices, "vertex")
        PlyData([ply_verts], text=False).write(path)

    def normalize_points(self, points_world, center, half_extent):
        """Map world points to [0, 1] coordinates used by DPSR."""
        p = (points_world - center[None]) / (2.0 * half_extent) + 0.5
        p = torch.clamp(p, SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        return p

    def _build_psr(self, points_world, normals_world, weights=None):
        center, half_extent = self._get_normalization_params(points_world)
        points_dpsr = self.normalize_points(points_world, center, half_extent)
        normals = F.normalize(normals_world, dim=-1)
        if weights is not None:
            normals = normals * weights[..., None]

        psr = self.dpsr(points_dpsr.unsqueeze(0), normals.unsqueeze(0))

        sign = psr[0, 0, 0, 0].detach()
        sign = -1.0 if sign < 0 else 1.0
        psr = psr * sign
        psr = psr - self.density_thres

        if self.surface_prior is not None:
            keep_w = torch.sigmoid((self.surface_prior - self.prior_thresh) / self.prior_temp)
            psr = keep_w.unsqueeze(0) * psr + (1.0 - keep_w.unsqueeze(0)) * self.prior_outside_value

        return psr.squeeze(0), center, half_extent

    def _extract_mesh(self, psr):
        if self.mesh_backend == "diffmc":
            verts, faces = self.diffmc(psr, deform=None, isovalue=0.0)
            verts = verts.to(torch.float32)
            faces = faces.to(torch.int32)
            return verts, faces

        if self.mesh_backend == "diffdmc":
            verts, faces = self.diffdmc(psr, deform=None, isovalue=0.0)
            verts = verts.to(torch.float32)
            faces = faces.to(torch.int32)
            return verts, faces

        if self.mesh_backend == "mc":
            verts, faces, _ = mc_from_psr(psr.unsqueeze(0), pytorchify=True, real_scale=False)
            return verts.to(torch.float32), faces.to(torch.int32)

        raise ValueError(f"Unknown mesh backend: {self.mesh_backend}. Use 'diffmc', 'diffdmc', or 'mc'.")

    def _to_world(self, verts, center, half_extent):
        verts = (verts - 0.5) * (2.0 * half_extent) + center[None]
        return verts

    def reconstruct(self, points_world, normals_world, weights=None):
        psr, center, half_extent = self._build_psr(points_world, normals_world, weights)
        verts, faces = self._extract_mesh(psr)
        verts_world = self._to_world(verts, center, half_extent)
        return {"psr": psr, "verts": verts_world, "faces": faces}

    @staticmethod
    def save_mesh_ply(path, verts, faces):
        mkdir_p(os.path.dirname(path))

        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vertices = np.empty(verts_np.shape[0], dtype=vertex_dtype)
        vertices["x"] = verts_np[:, 0]
        vertices["y"] = verts_np[:, 1]
        vertices["z"] = verts_np[:, 2]

        face_dtype = [("vertex_indices", "i4", (3,))]
        faces_el = np.empty(faces_np.shape[0], dtype=face_dtype)
        faces_el["vertex_indices"] = faces_np.astype(np.int32)

        ply_verts = PlyElement.describe(vertices, "vertex")
        ply_faces = PlyElement.describe(faces_el, "face")
        PlyData([ply_verts, ply_faces], text=False).write(path)
