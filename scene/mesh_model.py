import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from nvdiffrast_utils.dpsr import DPSR
from nvdiffrast_utils.dpsr_utils import mc_from_psr
from utils.system_utils import mkdir_p
from diso import DiffMC
from utils.mesh_utils import get_opacity_field_from_gaussians

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

        self.diffmc = DiffMC(dtype=torch.float32).to(device)
        self.flexicubes_engine = None
        self.flexi_cached_res = None
        self.flexi_cached_vertices = None
        self.flexi_cached_cube_idx = None

    def _ensure_flexicubes_grid(self, grid_dim):
        if self.flexi_cached_res == grid_dim:
            return self.flexi_cached_vertices, self.flexi_cached_cube_idx

        # Vertex coordinates on [0, 1]^3, matching DPSR normalized domain.
        axis = torch.linspace(0.0, 1.0, steps=grid_dim, device=self.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
        voxelgrid_vertices = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        vid = torch.arange(grid_dim ** 3, device=self.device, dtype=torch.long).reshape(grid_dim, grid_dim, grid_dim)
        cube_idx = torch.stack(
            [
                vid[:-1, :-1, :-1],
                vid[1:, :-1, :-1],
                vid[1:, 1:, :-1],
                vid[:-1, 1:, :-1],
                vid[:-1, :-1, 1:],
                vid[1:, :-1, 1:],
                vid[1:, 1:, 1:],
                vid[:-1, 1:, 1:],
            ],
            dim=-1,
        ).reshape(-1, 8)

        self.flexi_cached_res = grid_dim
        self.flexi_cached_vertices = voxelgrid_vertices
        self.flexi_cached_cube_idx = cube_idx
        return voxelgrid_vertices, cube_idx

    def _extract_mesh_flexicubes(self, psr, isovalue=0.0):
        if self.flexicubes_engine is None:
            self.flexicubes_engine = kal.ops.conversions.FlexiCubes(device=self.device)

        grid_dim = int(psr.shape[0])
        if psr.ndim != 3 or psr.shape[1] != grid_dim or psr.shape[2] != grid_dim:
            raise ValueError(f"FlexiCubes expects cubic [D,D,D] field, got shape={tuple(psr.shape)}")

        voxelgrid_vertices, cube_idx = self._ensure_flexicubes_grid(grid_dim)
        #scalar_field = (psr - float(isovalue)).reshape(-1).to(torch.float32)
        import torch.nn.functional as F

        D = psr.shape[0]
        # 1) 构造要在其上采样的顶点坐标（[0,1]^3 -> grid_sample expects [-1,1]）
        axis = torch.linspace(0.0, 1.0, steps=D, device=psr.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing='ij')
        grid = torch.stack([gx, gy, gz], dim=-1)  # (D,D,D,3)
        grid = 2.0 * grid - 1.0  # to [-1,1] for grid_sample

        # 2) 使用 grid_sample 做三线性重采样（输入形状 NCDHW for 3D: (1,1,D,D,D)）
        psr_in = psr[None, None]  # (1,1,D,D,D)
        vertex_psr = F.grid_sample(psr_in, grid[None], mode='bilinear', padding_mode='border', align_corners=True)
        vertex_psr = vertex_psr[0,0]  # (D,D,D)
        
        # 3) 可选：简单平滑（3D 均值）以去噪
        kernel = torch.ones((1,1,3,3,3), device=psr.device) / 27.0
        vertex_psr_pad = vertex_psr[None,None, None]  # expand dims
        # 使用 conv3d: (N,C,D,H,W)
        vertex_psr_smooth = F.conv3d(vertex_psr_pad, kernel, padding=1)[0,0]

        out = self.flexicubes_engine(
            voxelgrid_vertices=voxelgrid_vertices,
            scalar_field=scalar_field,
            cube_idx=cube_idx,
            resolution=grid_dim - 1,
        )

        if isinstance(out, tuple):
            verts, faces = out[0], out[1]
        else:
            raise RuntimeError("Unexpected FlexiCubes return type")

        return verts.to(torch.float32), faces.to(torch.int32)

    def _get_normalization_params(self, points_world):
        if self.normalization_mode == "aabb":
            pmin = torch.min(points_world, dim=0).values
            pmax = torch.max(points_world, dim=0).values
            center = 0.5 * (pmin + pmax)
            max_extent = torch.max(pmax - pmin)
            half_extent = 0.5 * max_extent * (1.0 + 2.0 * self.aabb_padding)
            half_extent = torch.clamp(half_extent, min=SMALL_NUMBER)
            return center, half_extent

        center = -self.translate
        half_extent = torch.clamp(self.radius.squeeze(0), min=SMALL_NUMBER)
        return center, half_extent

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
        return psr.squeeze(0), center, half_extent

    def _extract_mesh(self, psr):
        if self.mesh_backend == "diffmc":
            verts, faces = self.diffmc(psr, deform=None, isovalue=0.0)
            verts = verts.to(torch.float32)
            faces = faces.to(torch.int32)
            return verts, faces

        if self.mesh_backend == "flexicubes":
            return self._extract_mesh_flexicubes(psr, isovalue=0.0)

        if self.mesh_backend == "mc":
            verts, faces, _ = mc_from_psr(psr.unsqueeze(0), pytorchify=True, real_scale=False)
            return verts.to(torch.float32), faces.to(torch.int32)

        raise ValueError(f"Unknown mesh backend: {self.mesh_backend}. Use 'diffmc', 'flexicubes', or 'mc'.")

    def _to_world(self, verts, center, half_extent):
        verts = (verts - 0.5) * (2.0 * half_extent) + center[None]
        return verts

    @torch.no_grad()
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
