import torch
import torch.nn.functional as F
from kaolin.ops.conversions import FlexiCubes as KaolinFlexiCubes


class KaolinFlexiCubesBackend:
    """Thin wrapper around Kaolin FlexiCubes for a single dense voxel grid."""

    def __init__(self, resolution, device="cuda"):
        if KaolinFlexiCubes is None:
            raise ImportError("Kaolin FlexiCubes is not available in this environment.")

        self.device = device
        self.resolution = resolution
        self.backend = KaolinFlexiCubes(device=device)
        self.voxelgrid_vertices, self.cube_idx = self.backend.construct_voxel_grid(self.resolution)

    def _sample_scalar_field(self, psr):
        if psr.ndim != 3:
            raise ValueError(f"Expected psr to have shape (D, H, W), got {tuple(psr.shape)}")

        field = psr.to(device=self.voxelgrid_vertices.device, dtype=torch.float32)
        field = field.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)

        points = self.voxelgrid_vertices.to(device=field.device, dtype=field.dtype) + 0.5
        points = torch.clamp(points, 0.0, 1.0)
        grid = points.mul(2.0).sub(1.0).view(1, -1, 1, 1, 3)

        sampled = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(-1)

    def extract(self, psr, output_tetmesh=False, training=False, grad_func=None):
        scalar_field = self._sample_scalar_field(psr)
        verts, faces, _ = self.backend(
            self.voxelgrid_vertices,
            scalar_field,
            self.cube_idx,
            self.resolution,
            output_tetmesh=output_tetmesh,
            training=training,
            grad_func=grad_func,
        )
        return verts, faces