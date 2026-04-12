#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from dsl import CandidateProgram, SearchBounds, gene_center_z

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None


KIND_TO_CODE = {"rectangle": 0, "circle": 1, "rounded_rectangle": 2}
AXIS_TO_CODE = {"x": 0, "y": 1, "z": 2}


@dataclass
class ScoreWeights:
    iou_weight: float = 1.0
    missing_weight: float = 0.72
    extra_weight: float = 0.58
    primitive_weight: float = 0.05
    empty_weight: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreWeights:
        return cls(**payload)


@dataclass
class TargetGrid:
    occupancy: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    pitch: float
    score_min_corner: np.ndarray
    score_max_corner: np.ndarray
    search_bounds: SearchBounds
    source_stl: str

    def __post_init__(self) -> None:
        self.occupancy = np.asarray(self.occupancy, dtype=bool)
        self.xs = np.asarray(self.xs, dtype=np.float32)
        self.ys = np.asarray(self.ys, dtype=np.float32)
        self.zs = np.asarray(self.zs, dtype=np.float32)
        self.score_min_corner = np.asarray(self.score_min_corner, dtype=np.float32)
        self.score_max_corner = np.asarray(self.score_max_corner, dtype=np.float32)
        self.pitch = float(self.pitch)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.occupancy.shape)

    @property
    def target_voxels(self) -> int:
        return int(self.occupancy.sum())

    def to_payload(self) -> dict[str, Any]:
        return {
            "occupancy": self.occupancy.astype(np.uint8),
            "xs": self.xs,
            "ys": self.ys,
            "zs": self.zs,
            "pitch": self.pitch,
            "score_min_corner": self.score_min_corner,
            "score_max_corner": self.score_max_corner,
            "search_bounds": self.search_bounds.to_dict(),
            "source_stl": self.source_stl,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TargetGrid:
        return cls(
            occupancy=np.asarray(payload["occupancy"], dtype=bool),
            xs=np.asarray(payload["xs"], dtype=np.float32),
            ys=np.asarray(payload["ys"], dtype=np.float32),
            zs=np.asarray(payload["zs"], dtype=np.float32),
            pitch=float(payload["pitch"]),
            score_min_corner=np.asarray(payload["score_min_corner"], dtype=np.float32),
            score_max_corner=np.asarray(payload["score_max_corner"], dtype=np.float32),
            search_bounds=SearchBounds.from_dict(payload["search_bounds"]),
            source_stl=str(payload["source_stl"]),
        )

    @classmethod
    def from_stl(cls, input_path: Path, pitch: float, search_pad_ratio: float = 0.08) -> TargetGrid:
        mesh = trimesh.load(str(input_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise SystemExit(f"Could not load a mesh from {input_path}.")

        voxels = mesh.voxelized(pitch)
        if mesh.is_watertight:
            voxels = voxels.fill()

        occupancy = np.asarray(voxels.matrix, dtype=bool)
        if occupancy.ndim != 3 or not occupancy.any():
            raise SystemExit("Voxelization produced an empty target grid.")

        x0 = float(voxels.translation[0] - pitch * 0.5)
        y0 = float(voxels.translation[1] - pitch * 0.5)
        z0 = float(voxels.translation[2] - pitch * 0.5)

        xs = np.asarray(x0 + pitch * 0.5 + np.arange(occupancy.shape[0]) * pitch, dtype=np.float32)
        ys = np.asarray(y0 + pitch * 0.5 + np.arange(occupancy.shape[1]) * pitch, dtype=np.float32)
        zs = np.asarray(z0 + pitch * 0.5 + np.arange(occupancy.shape[2]) * pitch, dtype=np.float32)

        score_min_corner = np.array([x0, y0, z0], dtype=np.float32)
        score_max_corner = score_min_corner + np.asarray(occupancy.shape, dtype=np.float32) * pitch
        pad = np.maximum((score_max_corner - score_min_corner) * search_pad_ratio, pitch * 2.0)
        search_bounds = SearchBounds(
            min_corner=score_min_corner - pad.astype(np.float32),
            max_corner=score_max_corner + pad.astype(np.float32),
            pitch=pitch,
        )
        return cls(
            occupancy=occupancy,
            xs=xs,
            ys=ys,
            zs=zs,
            pitch=float(pitch),
            score_min_corner=score_min_corner,
            score_max_corner=score_max_corner,
            search_bounds=search_bounds,
            source_stl=str(input_path),
        )


def resolve_backend(requested_device: str) -> tuple[str, str]:
    want_cuda = requested_device in {"auto", "cuda", "gpu"}
    if want_cuda and torch is not None and torch.cuda.is_available():
        return "torch", "cuda"
    return "numpy", "cpu"


class ProgramScorer:
    def __init__(self, target: TargetGrid, requested_device: str = "auto", batch_size: int = 8) -> None:
        self.target = target
        self.backend, self.device = resolve_backend(requested_device)
        self.batch_size = max(1, int(batch_size))
        self.target_count = target.target_voxels

        self._xs_np = target.xs
        self._ys_np = target.ys
        self._zs_np = target.zs
        self._target_np = target.occupancy

        if self.backend == "torch":
            assert torch is not None
            self._xs_t = torch.as_tensor(target.xs, device=self.device, dtype=torch.float32)
            self._ys_t = torch.as_tensor(target.ys, device=self.device, dtype=torch.float32)
            self._zs_t = torch.as_tensor(target.zs, device=self.device, dtype=torch.float32)
            self._target_t = torch.as_tensor(target.occupancy, device=self.device, dtype=torch.bool)

    def backend_name(self) -> str:
        return f"{self.backend}:{self.device}"

    def score_candidates(
        self,
        candidates: list[CandidateProgram],
        weights: ScoreWeights,
        primitive_budget: int,
    ) -> list[dict[str, float]]:
        if not candidates:
            return []

        results: list[dict[str, float]] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            encoded = self._encode_batch(batch, primitive_budget)
            if self.backend == "torch":
                batch_results = self._score_torch_batch(encoded, weights, primitive_budget)
            else:
                batch_results = self._score_numpy_batch(encoded, weights, primitive_budget)
            results.extend(batch_results)
        return results

    def _encode_batch(
        self,
        candidates: list[CandidateProgram],
        primitive_budget: int,
    ) -> dict[str, np.ndarray]:
        max_primitives = max(1, primitive_budget)
        batch = len(candidates)
        active = np.zeros((batch, max_primitives), dtype=bool)
        kind = np.zeros((batch, max_primitives), dtype=np.int16)
        op = np.zeros((batch, max_primitives), dtype=np.int8)
        center_x = np.zeros((batch, max_primitives), dtype=np.float32)
        center_y = np.zeros((batch, max_primitives), dtype=np.float32)
        center_z = np.zeros((batch, max_primitives), dtype=np.float32)
        height = np.zeros((batch, max_primitives), dtype=np.float32)
        size_x = np.zeros((batch, max_primitives), dtype=np.float32)
        size_y = np.zeros((batch, max_primitives), dtype=np.float32)
        aux = np.zeros((batch, max_primitives), dtype=np.float32)
        axis = np.zeros((batch, max_primitives), dtype=np.int8)
        angle_deg = np.zeros((batch, max_primitives), dtype=np.float32)
        primitive_count = np.zeros(batch, dtype=np.int16)

        for row, candidate in enumerate(candidates):
            primitive_count[row] = min(len(candidate.genes), max_primitives)
            for col, gene in enumerate(candidate.genes[:max_primitives]):
                active[row, col] = True
                kind[row, col] = KIND_TO_CODE[gene.kind]
                op[row, col] = 1 if gene.op == "add" else 0
                center_x[row, col] = gene.center_x
                center_y[row, col] = gene.center_y
                center_z[row, col] = gene_center_z(gene)
                height[row, col] = gene.height
                size_x[row, col] = gene.size_x
                size_y[row, col] = gene.size_y
                aux[row, col] = gene.aux
                axis[row, col] = AXIS_TO_CODE.get(gene.axis, AXIS_TO_CODE["z"])
                angle_deg[row, col] = gene.angle_deg

        return {
            "active": active,
            "kind": kind,
            "op": op,
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "height": height,
            "size_x": size_x,
            "size_y": size_y,
            "aux": aux,
            "axis": axis,
            "angle_deg": angle_deg,
            "primitive_count": primitive_count,
        }

    def _score_numpy_batch(
        self,
        encoded: dict[str, np.ndarray],
        weights: ScoreWeights,
        primitive_budget: int,
    ) -> list[dict[str, float]]:
        active = encoded["active"]
        kind = encoded["kind"]
        op = encoded["op"]
        center_x = encoded["center_x"]
        center_y = encoded["center_y"]
        center_z = encoded["center_z"]
        height = encoded["height"]
        size_x = encoded["size_x"]
        size_y = encoded["size_y"]
        aux = encoded["aux"]
        axis = encoded["axis"]
        angle_deg = encoded["angle_deg"]
        primitive_count = encoded["primitive_count"]
        batch = active.shape[0]

        occ = np.zeros((batch,) + self._target_np.shape, dtype=bool)
        x_grid = self._xs_np[None, :, None, None]
        y_grid = self._ys_np[None, None, :, None]
        z_grid = self._zs_np[None, None, None, :]

        for column in range(active.shape[1]):
            active_col = active[:, column]
            if not active_col.any():
                continue

            rect_idx = active_col & (kind[:, column] == KIND_TO_CODE["rectangle"])
            if rect_idx.any():
                theta = np.deg2rad(angle_deg[rect_idx, column])[:, None, None, None]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                x_rel = x_grid - center_x[rect_idx, column][:, None, None, None]
                y_rel = y_grid - center_y[rect_idx, column][:, None, None, None]
                local_x = cos_t * x_rel + sin_t * y_rel
                local_y = -sin_t * x_rel + cos_t * y_rel
                mask_xy = (np.abs(local_x) <= size_x[rect_idx, column][:, None, None, None] * 0.5) & (
                    np.abs(local_y) <= size_y[rect_idx, column][:, None, None, None] * 0.5
                )
                mask_z = np.abs(z_grid - center_z[rect_idx, column][:, None, None, None]) <= (
                    height[rect_idx, column][:, None, None, None] * 0.5
                )
                primitive = mask_xy & mask_z
                add_rect = rect_idx & (op[:, column] == 1)
                if add_rect.any():
                    occ[add_rect] |= primitive[add_rect[rect_idx]]
                cut_rect = rect_idx & (op[:, column] == 0)
                if cut_rect.any():
                    occ[cut_rect] &= ~primitive[cut_rect[rect_idx]]

            circle_idx = active_col & (kind[:, column] == KIND_TO_CODE["circle"])
            if circle_idx.any():
                axis_col = axis[circle_idx, column]
                radius_sq = size_x[circle_idx, column][:, None, None, None] ** 2
                primitive = np.zeros((circle_idx.sum(),) + self._target_np.shape, dtype=bool)
                x_sel = x_grid - center_x[circle_idx, column][:, None, None, None]
                y_sel = y_grid - center_y[circle_idx, column][:, None, None, None]
                z_sel = z_grid - center_z[circle_idx, column][:, None, None, None]
                half_len = height[circle_idx, column][:, None, None, None] * 0.5
                z_mask = axis_col == AXIS_TO_CODE["z"]
                if z_mask.any():
                    radial = (x_sel[z_mask] * x_sel[z_mask] + y_sel[z_mask] * y_sel[z_mask]) <= radius_sq[z_mask]
                    axial = np.abs(z_sel[z_mask]) <= half_len[z_mask]
                    primitive[z_mask] = radial & axial
                x_mask = axis_col == AXIS_TO_CODE["x"]
                if x_mask.any():
                    radial = (y_sel[x_mask] * y_sel[x_mask] + z_sel[x_mask] * z_sel[x_mask]) <= radius_sq[x_mask]
                    axial = np.abs(x_sel[x_mask]) <= half_len[x_mask]
                    primitive[x_mask] = radial & axial
                y_mask = axis_col == AXIS_TO_CODE["y"]
                if y_mask.any():
                    radial = (x_sel[y_mask] * x_sel[y_mask] + z_sel[y_mask] * z_sel[y_mask]) <= radius_sq[y_mask]
                    axial = np.abs(y_sel[y_mask]) <= half_len[y_mask]
                    primitive[y_mask] = radial & axial
                add_circle = circle_idx & (op[:, column] == 1)
                if add_circle.any():
                    occ[add_circle] |= primitive[add_circle[circle_idx]]
                cut_circle = circle_idx & (op[:, column] == 0)
                if cut_circle.any():
                    occ[cut_circle] &= ~primitive[cut_circle[circle_idx]]

            rounded_idx = active_col & (kind[:, column] == KIND_TO_CODE["rounded_rectangle"])
            if rounded_idx.any():
                theta = np.deg2rad(angle_deg[rounded_idx, column])[:, None, None, None]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                x_rel = x_grid - center_x[rounded_idx, column][:, None, None, None]
                y_rel = y_grid - center_y[rounded_idx, column][:, None, None, None]
                local_x = cos_t * x_rel + sin_t * y_rel
                local_y = -sin_t * x_rel + cos_t * y_rel
                width = size_x[rounded_idx, column][:, None, None, None]
                depth = size_y[rounded_idx, column][:, None, None, None]
                radius = np.minimum(aux[rounded_idx, column][:, None, None, None], np.minimum(width, depth) * 0.5)
                core_x = np.maximum(width * 0.5 - radius, 0.0)
                core_y = np.maximum(depth * 0.5 - radius, 0.0)
                qx = np.maximum(np.abs(local_x) - core_x, 0.0)
                qy = np.maximum(np.abs(local_y) - core_y, 0.0)
                mask_xy = (qx * qx + qy * qy) <= (radius * radius)
                mask_z = np.abs(z_grid - center_z[rounded_idx, column][:, None, None, None]) <= (
                    height[rounded_idx, column][:, None, None, None] * 0.5
                )
                primitive = mask_xy & mask_z
                add_rounded = rounded_idx & (op[:, column] == 1)
                if add_rounded.any():
                    occ[add_rounded] |= primitive[add_rounded[rounded_idx]]
                cut_rounded = rounded_idx & (op[:, column] == 0)
                if cut_rounded.any():
                    occ[cut_rounded] &= ~primitive[cut_rounded[rounded_idx]]

        target = self._target_np[None, ...]
        intersection = np.count_nonzero(occ & target, axis=(1, 2, 3)).astype(np.float32)
        candidate_count = np.count_nonzero(occ, axis=(1, 2, 3)).astype(np.float32)
        union = np.maximum(candidate_count + self.target_count - intersection, 1.0)
        missing = np.maximum(self.target_count - intersection, 0.0)
        extra = np.maximum(candidate_count - intersection, 0.0)
        iou = intersection / union
        missing_ratio = missing / max(self.target_count, 1)
        extra_ratio = extra / max(self.target_count, 1)
        primitive_ratio = primitive_count.astype(np.float32) / max(primitive_budget, 1)
        empty_penalty = (candidate_count <= 0).astype(np.float32)
        score = (
            weights.iou_weight * iou
            - weights.missing_weight * missing_ratio
            - weights.extra_weight * extra_ratio
            - weights.primitive_weight * primitive_ratio
            - weights.empty_weight * empty_penalty
        )

        results: list[dict[str, float]] = []
        for row in range(active.shape[0]):
            results.append(
                {
                    "score": float(score[row]),
                    "iou": float(iou[row]),
                    "missing_ratio": float(missing_ratio[row]),
                    "extra_ratio": float(extra_ratio[row]),
                    "intersection_voxels": float(intersection[row]),
                    "candidate_voxels": float(candidate_count[row]),
                    "target_voxels": float(self.target_count),
                    "primitive_count": float(primitive_count[row]),
                }
            )
        return results

    def _score_torch_batch(
        self,
        encoded: dict[str, np.ndarray],
        weights: ScoreWeights,
        primitive_budget: int,
    ) -> list[dict[str, float]]:
        assert torch is not None

        active = torch.as_tensor(encoded["active"], device=self.device, dtype=torch.bool)
        kind = torch.as_tensor(encoded["kind"], device=self.device, dtype=torch.int64)
        op = torch.as_tensor(encoded["op"], device=self.device, dtype=torch.int64)
        center_x = torch.as_tensor(encoded["center_x"], device=self.device, dtype=torch.float32)
        center_y = torch.as_tensor(encoded["center_y"], device=self.device, dtype=torch.float32)
        center_z = torch.as_tensor(encoded["center_z"], device=self.device, dtype=torch.float32)
        height = torch.as_tensor(encoded["height"], device=self.device, dtype=torch.float32)
        size_x = torch.as_tensor(encoded["size_x"], device=self.device, dtype=torch.float32)
        size_y = torch.as_tensor(encoded["size_y"], device=self.device, dtype=torch.float32)
        aux = torch.as_tensor(encoded["aux"], device=self.device, dtype=torch.float32)
        axis = torch.as_tensor(encoded["axis"], device=self.device, dtype=torch.int64)
        angle_deg = torch.as_tensor(encoded["angle_deg"], device=self.device, dtype=torch.float32)
        primitive_count = torch.as_tensor(encoded["primitive_count"], device=self.device, dtype=torch.float32)

        batch = active.shape[0]
        occ = torch.zeros((batch,) + self._target_t.shape, device=self.device, dtype=torch.bool)
        x_grid = self._xs_t.view(1, -1, 1, 1)
        y_grid = self._ys_t.view(1, 1, -1, 1)
        z_grid = self._zs_t.view(1, 1, 1, -1)

        for column in range(active.shape[1]):
            active_col = active[:, column]
            if not bool(active_col.any().item()):
                continue

            rect_idx = active_col & (kind[:, column] == KIND_TO_CODE["rectangle"])
            if bool(rect_idx.any().item()):
                theta = torch.deg2rad(angle_deg[rect_idx, column]).view(-1, 1, 1, 1)
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                x_rel = x_grid - center_x[rect_idx, column].view(-1, 1, 1, 1)
                y_rel = y_grid - center_y[rect_idx, column].view(-1, 1, 1, 1)
                local_x = cos_t * x_rel + sin_t * y_rel
                local_y = -sin_t * x_rel + cos_t * y_rel
                mask_xy = (torch.abs(local_x) <= size_x[rect_idx, column].view(-1, 1, 1, 1) * 0.5) & (
                    torch.abs(local_y) <= size_y[rect_idx, column].view(-1, 1, 1, 1) * 0.5
                )
                mask_z = torch.abs(z_grid - center_z[rect_idx, column].view(-1, 1, 1, 1)) <= (
                    height[rect_idx, column].view(-1, 1, 1, 1) * 0.5
                )
                primitive = mask_xy & mask_z
                add_rect = rect_idx & (op[:, column] == 1)
                if bool(add_rect.any().item()):
                    occ[add_rect] |= primitive[add_rect[rect_idx]]
                cut_rect = rect_idx & (op[:, column] == 0)
                if bool(cut_rect.any().item()):
                    occ[cut_rect] &= ~primitive[cut_rect[rect_idx]]

            circle_idx = active_col & (kind[:, column] == KIND_TO_CODE["circle"])
            if bool(circle_idx.any().item()):
                axis_col = axis[circle_idx, column]
                primitive = torch.zeros((int(circle_idx.sum().item()),) + self._target_t.shape, device=self.device, dtype=torch.bool)
                x_sel = x_grid - center_x[circle_idx, column].view(-1, 1, 1, 1)
                y_sel = y_grid - center_y[circle_idx, column].view(-1, 1, 1, 1)
                z_sel = z_grid - center_z[circle_idx, column].view(-1, 1, 1, 1)
                radius_sq = size_x[circle_idx, column].view(-1, 1, 1, 1) ** 2
                half_len = height[circle_idx, column].view(-1, 1, 1, 1) * 0.5
                z_mask = axis_col == AXIS_TO_CODE["z"]
                if bool(z_mask.any().item()):
                    radial = (x_sel[z_mask] * x_sel[z_mask] + y_sel[z_mask] * y_sel[z_mask]) <= radius_sq[z_mask]
                    axial = torch.abs(z_sel[z_mask]) <= half_len[z_mask]
                    primitive[z_mask] = radial & axial
                x_mask = axis_col == AXIS_TO_CODE["x"]
                if bool(x_mask.any().item()):
                    radial = (y_sel[x_mask] * y_sel[x_mask] + z_sel[x_mask] * z_sel[x_mask]) <= radius_sq[x_mask]
                    axial = torch.abs(x_sel[x_mask]) <= half_len[x_mask]
                    primitive[x_mask] = radial & axial
                y_mask = axis_col == AXIS_TO_CODE["y"]
                if bool(y_mask.any().item()):
                    radial = (x_sel[y_mask] * x_sel[y_mask] + z_sel[y_mask] * z_sel[y_mask]) <= radius_sq[y_mask]
                    axial = torch.abs(y_sel[y_mask]) <= half_len[y_mask]
                    primitive[y_mask] = radial & axial
                add_circle = circle_idx & (op[:, column] == 1)
                if bool(add_circle.any().item()):
                    occ[add_circle] |= primitive[add_circle[circle_idx]]
                cut_circle = circle_idx & (op[:, column] == 0)
                if bool(cut_circle.any().item()):
                    occ[cut_circle] &= ~primitive[cut_circle[circle_idx]]

            rounded_idx = active_col & (kind[:, column] == KIND_TO_CODE["rounded_rectangle"])
            if bool(rounded_idx.any().item()):
                theta = torch.deg2rad(angle_deg[rounded_idx, column]).view(-1, 1, 1, 1)
                cos_t = torch.cos(theta)
                sin_t = torch.sin(theta)
                x_rel = x_grid - center_x[rounded_idx, column].view(-1, 1, 1, 1)
                y_rel = y_grid - center_y[rounded_idx, column].view(-1, 1, 1, 1)
                local_x = cos_t * x_rel + sin_t * y_rel
                local_y = -sin_t * x_rel + cos_t * y_rel
                width = size_x[rounded_idx, column].view(-1, 1, 1, 1)
                depth = size_y[rounded_idx, column].view(-1, 1, 1, 1)
                radius = torch.minimum(aux[rounded_idx, column].view(-1, 1, 1, 1), torch.minimum(width, depth) * 0.5)
                core_x = torch.clamp(width * 0.5 - radius, min=0.0)
                core_y = torch.clamp(depth * 0.5 - radius, min=0.0)
                qx = torch.clamp(torch.abs(local_x) - core_x, min=0.0)
                qy = torch.clamp(torch.abs(local_y) - core_y, min=0.0)
                mask_xy = (qx * qx + qy * qy) <= (radius * radius)
                mask_z = torch.abs(z_grid - center_z[rounded_idx, column].view(-1, 1, 1, 1)) <= (
                    height[rounded_idx, column].view(-1, 1, 1, 1) * 0.5
                )
                primitive = mask_xy & mask_z
                add_rounded = rounded_idx & (op[:, column] == 1)
                if bool(add_rounded.any().item()):
                    occ[add_rounded] |= primitive[add_rounded[rounded_idx]]
                cut_rounded = rounded_idx & (op[:, column] == 0)
                if bool(cut_rounded.any().item()):
                    occ[cut_rounded] &= ~primitive[cut_rounded[rounded_idx]]

        target = self._target_t.unsqueeze(0)
        intersection = torch.count_nonzero(occ & target, dim=(1, 2, 3)).to(torch.float32)
        candidate_count = torch.count_nonzero(occ, dim=(1, 2, 3)).to(torch.float32)
        union = torch.clamp(candidate_count + self.target_count - intersection, min=1.0)
        missing = torch.clamp(self.target_count - intersection, min=0.0)
        extra = torch.clamp(candidate_count - intersection, min=0.0)
        iou = intersection / union
        missing_ratio = missing / max(self.target_count, 1)
        extra_ratio = extra / max(self.target_count, 1)
        primitive_ratio = primitive_count / max(primitive_budget, 1)
        empty_penalty = (candidate_count <= 0).to(torch.float32)
        score = (
            weights.iou_weight * iou
            - weights.missing_weight * missing_ratio
            - weights.extra_weight * extra_ratio
            - weights.primitive_weight * primitive_ratio
            - weights.empty_weight * empty_penalty
        )

        score_np = score.detach().cpu().numpy()
        iou_np = iou.detach().cpu().numpy()
        missing_np = missing_ratio.detach().cpu().numpy()
        extra_np = extra_ratio.detach().cpu().numpy()
        inter_np = intersection.detach().cpu().numpy()
        count_np = candidate_count.detach().cpu().numpy()
        prim_np = primitive_count.detach().cpu().numpy()

        results: list[dict[str, float]] = []
        for row in range(active.shape[0]):
            results.append(
                {
                    "score": float(score_np[row]),
                    "iou": float(iou_np[row]),
                    "missing_ratio": float(missing_np[row]),
                    "extra_ratio": float(extra_np[row]),
                    "intersection_voxels": float(inter_np[row]),
                    "candidate_voxels": float(count_np[row]),
                    "target_voxels": float(self.target_count),
                    "primitive_count": float(prim_np[row]),
                }
            )
        return results
