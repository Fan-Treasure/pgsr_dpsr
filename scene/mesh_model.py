import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from nvdiffrast_utils.dpsr import DPSR
from nvdiffrast_utils.dpsr_utils import mc_from_psr
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
        self.diffdmc = DiffDMC(dtype=torch.float32).to(device)

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
