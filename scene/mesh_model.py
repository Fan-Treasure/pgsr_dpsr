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

SMALL_NUMBER = 1e-6


class MeshModel:
    """Wrap DPSR + DiffMC for one-shot mesh extraction from oriented point clouds."""

    def __init__(
        self,
        grid_res=256,
        dpsr_sig=0.5,
        density_thres=0.0,
        nerf_normalization=None,
        normalization_mode="aabb",
        aabb_padding=0.05,
        device="cuda",
    ):
        self.device = device
        self.grid_res = grid_res
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
        if self.diffmc is not None:
            verts, faces = self.diffmc(psr, deform=None, isovalue=0.0)
            verts = verts.to(torch.float32)
            faces = faces.to(torch.int32)
            return verts, faces

        verts, faces, _ = mc_from_psr(psr.unsqueeze(0), pytorchify=True, real_scale=False)
        return verts.to(torch.float32), faces.to(torch.int32)

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

    def save_psr_slices(self, path_prefix, psr, axis='z', indices=None, cmap='bwr'):
        """Save 2D PNG slices of a PSR grid.

        - `psr`: torch.Tensor with shape [D0, D1, D2] (or [1, D0, D1, D2]).
        - `axis`: one of 'x','y','z' indicating slice orientation (z -> axis=0).
        - `indices`: list of slice indices (if None, saves the middle slice).
        """
        if torch.is_tensor(psr):
            psr_np = psr.detach().cpu().numpy().squeeze()
        else:
            psr_np = np.array(psr)

        if psr_np.ndim == 4 and psr_np.shape[0] == 1:
            psr_np = psr_np[0]

        axis_map = {'z': 0, 'y': 1, 'x': 2}
        if axis not in axis_map:
            raise ValueError("axis must be one of 'x','y','z'")
        ax = axis_map[axis]

        size = psr_np.shape[ax]
        if indices is None:
            indices = [size // 2]

        mkdir_p(os.path.dirname(path_prefix) or '.')

        vmin = float(np.nanmin(psr_np))
        vmax = float(np.nanmax(psr_np))
        # choose symmetric range around zero if that seems reasonable
        absmax = max(abs(vmin), abs(vmax))

        for idx in indices:
            if ax == 0:
                img = psr_np[idx, :, :]
            elif ax == 1:
                img = psr_np[:, idx, :]
            else:
                img = psr_np[:, :, idx]

            out_path = f"{path_prefix}_slice_{axis}{idx}.png"
            plt.imsave(out_path, img, cmap=cmap, vmin=-absmax, vmax=absmax)

    @torch.no_grad()
    def export_occupancy_init_from_gaussians(
        self,
        xyz_world: torch.Tensor,
        rotations: torch.Tensor,
        scalings: torch.Tensor,
        opacities: torch.Tensor,
        out_path: str,
        occ_res: int = 256,
        num_blocks: int = 16,
        relax_ratio: float = 0.5,
        opacity_threshold: float = 0.005,
        bbox_scale: float = 2.0,
        aabb_padding: float = None,
    ):
        """Build occupancy from Gaussians, extract mesh via DiffMC and save PLY.

        Inputs are in world coordinates. This function computes a local AABB
        normalization (center, half_extent) and maps Gaussians into the
        canonical grid used by `get_opacity_field_from_gaussians`.
        """
        device = xyz_world.device

        mask = (opacities.squeeze(-1) > opacity_threshold)
        if not mask.any():
            return None

        xyz_w = xyz_world[mask]
        rot = rotations[mask]
        scl_w = scalings[mask]
        opa = opacities[mask]

        # compute AABB normalization (reuse mesh_model padding if not provided)
        pad = self.aabb_padding if aabb_padding is None else aabb_padding
        pmin = torch.min(xyz_w, dim=0).values
        pmax = torch.max(xyz_w, dim=0).values
        center = 0.5 * (pmin + pmax)
        max_extent = torch.max(pmax - pmin)
        half_extent = torch.clamp(0.5 * max_extent * (1.0 + 2.0 * pad), min=SMALL_NUMBER)

        # normalize to DG-Mesh domain [-bbox_scale, bbox_scale] for occupancy builder
        xyz_n = (xyz_w - center[None]) / half_extent * bbox_scale
        scl_n = scl_w / half_extent * bbox_scale

        occ = get_opacity_field_from_gaussians(
            xyz_n,
            rot,
            scl_n,
            opa,
            resolution=occ_res,
            num_blocks=num_blocks,
            relax_ratio=relax_ratio,
            opacity_threshold=opacity_threshold,
            bbox_scale=bbox_scale,
        )

        # use DiffMC to extract a coarse mesh from occupancy (occ small positive -> surface)
        diffmc_occ = self.diffmc
        verts01, faces = diffmc_occ(-occ.to(device), deform=None, isovalue=-0.01)
        verts01 = verts01.to(torch.float32)
        faces = faces.to(torch.int32)

        # Map verts from [0,1] to normalized coords [-bbox_scale, bbox_scale]
        verts_n = (verts01 - 0.5) * (2.0 * bbox_scale)
        # Back to world space
        verts_w = verts_n * half_extent.to(device) + center[None].to(device)

        # save
        self.save_mesh_ply(out_path, verts_w, faces)
        return {
            "path": out_path,
            "verts": verts_w,
            "faces": faces,
            "center": center,
            "half_extent": half_extent,
        }
