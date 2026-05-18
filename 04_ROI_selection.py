"""
ROI_selection_04.py
===================

Python port of ROI_selection_04.m.

Manual ROI/object selection for downstream signal-intensity analysis:
  1. Discover registered Reference/Target 3-D TIFF stack pairs under data/.
  2. Display each pair as a Target/Reference merged MIP.
  3. Click Reference-positive object centers, then press Enter.
  4. Crop a fixed XY box around each point across the full Z-stack.
  5. Segment Reference and Target crops with Method 3.
  6. Save crops, masks, labels, QC panels, metadata, and manifests.

The input/output layout intentionally mirrors ROI_selection_04.m.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects

_mpl_cache = Path(tempfile.gettempdir()) / "roi_selection_04_matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


@dataclass
class ROIParams:
    crop_radius_xy: int = 50
    expand_z: int = 3
    min_volume_voxels: int = 1
    display_low_pct: float = 1.0
    display_high_pct: float = 99.8
    reference_smooth_sigma: float = 0.5
    target_smooth_sigma: float = 0.5
    target_dilate_radius: int = 3
    start_at_series_index: int = 1
    batch_timestamp_override: str = ""


@dataclass
class SeriesInfo:
    sample_id: str
    condition: str
    image_id: str
    series_dir: Path
    reference_stack_path: Path
    target_stack_path: Path


MANIFEST_FIELDS = [
    "sample_id",
    "condition",
    "image_id",
    "object_id",
    "center_x",
    "center_y",
    "crop_x_start",
    "crop_x_end",
    "crop_y_start",
    "crop_y_end",
    "z_start",
    "z_end",
    "expand_z",
    "min_volume_voxels",
    "reference_threshold_m3",
    "target_threshold_m3",
    "n_reference_objects",
    "n_target_objects",
    "reference_stack_path",
    "target_stack_path",
    "reference_crop_path",
    "target_crop_path",
    "reference_object_mask_path",
    "target_object_mask_path",
    "valid_roi_mask_path",
    "qc_path",
]


def discover_registered_series(data_root: Path) -> list[SeriesInfo]:
    series_list: list[SeriesInfo] = []
    if not data_root.exists():
        return series_list

    for sample_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            registered_dir = series_dir / "registered_stacks"
            if not registered_dir.is_dir():
                continue

            reference_stack_path = registered_dir / "Reference_stack_registered.tif"
            target_stack_path = registered_dir / "Target_stack_registered.tif"
            if not reference_stack_path.exists() or not target_stack_path.exists():
                print(f"[SKIP] Missing Reference/Target pair: {registered_dir}")
                continue

            image_id = series_dir.name
            sample_id = sample_dir.name
            series_list.append(
                SeriesInfo(
                    sample_id=sample_id,
                    condition=infer_condition(image_id),
                    image_id=image_id,
                    series_dir=series_dir,
                    reference_stack_path=reference_stack_path,
                    target_stack_path=target_stack_path,
                )
            )

    series_list.sort(key=lambda s: (s.sample_id, s.image_id))
    return series_list


def read_tiff_volume(path: Path) -> np.ndarray:
    """Read a TIFF stack into MATLAB-like (Y, X, Z) order."""
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) == 0:
            raise ValueError(f"Empty TIFF stack: {path}")
        planes = [page.asarray() for page in tf.pages]

    first_shape = planes[0].shape
    if len(first_shape) != 2:
        arr = tifffile.imread(str(path))
        if arr.ndim == 3:
            return np.moveaxis(arr, 0, 2)
        raise ValueError(f"Expected 2-D pages or a 3-D TIFF stack: {path}")

    if any(p.shape != first_shape for p in planes):
        raise ValueError(f"TIFF pages have inconsistent shapes: {path}")
    return np.stack(planes, axis=2)


def write_volume_tiff(vol: np.ndarray, out_path: Path) -> None:
    """Write a (Y, X, Z) volume as a multi-page TIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vol)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected (Y, X, Z), got {arr.shape} for {out_path}")

    if arr.dtype == np.uint32:
        max_value = int(arr.max()) if arr.size else 0
        if max_value > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Cannot write {out_path} as uint16 label TIFF; max label is {max_value}."
            )
        arr = arr.astype(np.uint16)

    pages_zyx = np.moveaxis(arr, 2, 0)
    tifffile.imwrite(str(out_path), pages_zyx, photometric="minisblack")


def normalize_percentile_for_display(
    img: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float64)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def make_selection_rgb(
    reference_raw: np.ndarray,
    target_raw: np.ndarray,
    params: ROIParams,
) -> np.ndarray:
    reference_display = normalize_percentile_for_display(
        ndi.gaussian_filter(reference_raw, params.reference_smooth_sigma).max(axis=2),
        params.display_low_pct,
        params.display_high_pct,
    )
    target_display = normalize_percentile_for_display(
        ndi.gaussian_filter(target_raw, params.target_smooth_sigma).max(axis=2),
        params.display_low_pct,
        params.display_high_pct,
    )
    return np.stack(
        [target_display, reference_display, np.zeros_like(reference_display)],
        axis=2,
    )


def select_points_via_clicks(
    merged_rgb: np.ndarray,
    series: SeriesInfo,
    series_idx: int,
    n_series: int,
) -> np.ndarray:
    import matplotlib

    current_backend = matplotlib.get_backend().lower()
    if current_backend in ("agg", "module://matplotlib_inline.backend_inline"):
        for backend in ("Qt5Agg", "TkAgg", "MacOSX"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue

    import matplotlib.pyplot as plt

    height, width = merged_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=100)
    fig.canvas.manager.set_window_title(f"ROI selection {series_idx}/{n_series}")
    ax.imshow(
        merged_rgb,
        origin="upper",
        interpolation="nearest",
        extent=(1, width, height, 1),
    )
    ax.set_title(
        "\n".join(
            [
                f"[{series_idx}/{n_series}] {series.image_id}",
                "Click Reference-positive object centers. Target is red, Reference is green.",
                "Press Enter to save this image and advance. Press Enter without points to skip.",
            ]
        )
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    print("[CLICK] Click object centers; press Enter when done.")
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def expand_z_volume(vol: np.ndarray, expand_z: int) -> np.ndarray:
    if expand_z <= 1:
        return vol
    return np.repeat(vol, expand_z, axis=2)


def synapse_threshold_m3_local(arr: np.ndarray) -> tuple[np.ndarray, float]:
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool), 0.0

    a1 = arr.astype(np.float64) - np.median(arr)
    a1 = np.maximum(a1, 0)

    cross_k2d = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.float64,
    ) / 5.0

    a1s = np.zeros_like(a1, dtype=np.float64)
    for z in range(a1.shape[2]):
        a1s[:, :, z] = ndi.convolve(a1[:, :, z], cross_k2d, mode="nearest")

    coarse_gate = float(a1s.max() / 10.0) if a1s.size else 0.0
    if not np.isfinite(coarse_gate):
        return np.zeros(arr.shape, dtype=bool), 0.0

    bright_voxels = a1s[a1s > coarse_gate]
    if bright_voxels.size == 0:
        return np.zeros(arr.shape, dtype=bool), coarse_gate

    threshold = float(bright_voxels.mean())
    grid = np.arange(-2, 3)
    ky, kx, kz = np.meshgrid(grid, grid, grid, indexing="ij")
    k7_raw = (np.sqrt(kx**2 + ky**2 + kz**2) <= 2.5).astype(np.float64)
    k7 = k7_raw / k7_raw.sum()

    a2 = ndi.convolve(a1s - threshold, k7, mode="constant", cval=0.0)
    return a2 > 0, threshold


def clean_and_label_mask(
    mask: np.ndarray,
    min_volume_voxels: int,
    dilate_disk_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if min_volume_voxels > 1:
        mask = remove_small_objects(mask, min_size=min_volume_voxels, connectivity=3)

    if dilate_disk_radius > 0 and mask.size:
        footprint = disk(dilate_disk_radius)
        dilated = np.zeros_like(mask, dtype=bool)
        for z in range(mask.shape[2]):
            dilated[:, :, z] = ndi.binary_dilation(mask[:, :, z], structure=footprint)
        mask = dilated

    mask = ndi.binary_fill_holes(mask)
    labels, n_objects = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    return mask, labels.astype(np.uint32, copy=False), int(n_objects)


def save_roi_qc(
    reference_expanded: np.ndarray,
    target_expanded: np.ndarray,
    reference_mask: np.ndarray,
    target_mask: np.ndarray,
    object_id: int,
    reference_threshold: float,
    target_threshold: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ref_mip = normalize_percentile_for_display(reference_expanded.max(axis=2), 1, 99.8)
    tgt_mip = normalize_percentile_for_display(target_expanded.max(axis=2), 1, 99.8)
    ref_mask_mip = reference_mask.max(axis=2).astype(bool)
    tgt_mask_mip = target_mask.max(axis=2).astype(bool)

    overlay_ref = np.stack(
        [
            ref_mip * ~ref_mask_mip,
            np.minimum(1, ref_mip + 0.55 * ref_mask_mip),
            ref_mip * ~ref_mask_mip,
        ],
        axis=2,
    )
    overlay_tgt = np.stack(
        [
            np.minimum(1, tgt_mip + 0.55 * tgt_mask_mip),
            tgt_mip * ~tgt_mask_mip,
            tgt_mip * ~tgt_mask_mip,
        ],
        axis=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200)
    ax = axes.ravel()
    ax[0].imshow(ref_mip, cmap="gray")
    ax[0].set_title("Reference crop MIP")
    ax[1].imshow(tgt_mip, cmap="gray")
    ax[1].set_title("Target crop MIP")
    ax[2].imshow(np.stack([tgt_mip, ref_mip, np.zeros_like(ref_mip)], axis=2))
    ax[2].set_title("Target red / Reference green")
    ax[3].imshow(ref_mask_mip, cmap="gray")
    ax[3].set_title(f"Reference M3 mask (th={reference_threshold:.2f})")
    ax[4].imshow(tgt_mask_mip, cmap="gray")
    ax[4].set_title(f"Target M3 mask (th={target_threshold:.2f})")
    ax[5].imshow(np.maximum(overlay_ref, overlay_tgt))
    ax[5].set_title(f"Object {object_id:03d} masks overlay")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def process_registered_series(
    series: SeriesInfo,
    params: ROIParams,
    batch_timestamp: str,
    series_idx: int,
    n_series: int,
    points_override: Optional[np.ndarray] = None,
) -> list[dict]:
    print(f"\n[{series_idx}/{n_series}] {series.sample_id} | {series.condition} | {series.image_id}")
    print(f"Loading Reference: {series.reference_stack_path}")
    reference_raw = read_tiff_volume(series.reference_stack_path)
    print(f"Loading Target   : {series.target_stack_path}")
    target_raw = read_tiff_volume(series.target_stack_path)

    if reference_raw.shape != target_raw.shape:
        raise ValueError(f"Reference and Target stack sizes must match for {series.image_id}.")

    height, width, z_slices = reference_raw.shape
    merged = make_selection_rgb(reference_raw, target_raw, params)

    if points_override is None:
        points = select_points_via_clicks(merged, series, series_idx, n_series)
    else:
        points = np.asarray(points_override, dtype=np.float64)

    if points.size == 0:
        print(f"[SKIP] No ROI/object centers selected for {series.image_id}.")
        return []
    points = points.reshape((-1, 2))

    in_bounds = (
        (points[:, 0] >= 1)
        & (points[:, 0] <= width)
        & (points[:, 1] >= 1)
        & (points[:, 1] <= height)
    )
    if np.any(~in_bounds):
        print(f"[WARN] Ignoring {int((~in_bounds).sum())} point(s) outside {series.image_id}.")
        points = points[in_bounds]

    if points.shape[0] == 0:
        print(f"[SKIP] No in-bounds ROI/object centers selected for {series.image_id}.")
        return []
    print(f"{points.shape[0]} object center(s) selected.")

    roi_root = series.series_dir / "manual_rois" / f"roi_{batch_timestamp}"
    roi_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for object_id, (x_float, y_float) in enumerate(points, start=1):
        center_x = round_matlab(float(x_float))
        center_y = round_matlab(float(y_float))

        x_start = max(1, center_x - params.crop_radius_xy)
        x_end = min(width, center_x + params.crop_radius_xy)
        y_start = max(1, center_y - params.crop_radius_xy)
        y_end = min(height, center_y + params.crop_radius_xy)

        if x_start > x_end or y_start > y_end:
            print(f"[WARN] Skipping object {object_id:03d} for {series.image_id}; empty crop.")
            continue

        ys = slice(y_start - 1, y_end)
        xs = slice(x_start - 1, x_end)
        reference_crop = reference_raw[ys, xs, :]
        target_crop = target_raw[ys, xs, :]

        reference_expanded = expand_z_volume(reference_crop, params.expand_z)
        target_expanded = expand_z_volume(target_crop, params.expand_z)

        reference_mask_raw, reference_threshold = synapse_threshold_m3_local(reference_expanded)
        target_mask_raw, target_threshold = synapse_threshold_m3_local(target_expanded)

        reference_mask, reference_label, n_reference_objects = clean_and_label_mask(
            reference_mask_raw,
            params.min_volume_voxels,
            dilate_disk_radius=0,
        )
        target_mask, target_label, n_target_objects = clean_and_label_mask(
            target_mask_raw,
            params.min_volume_voxels,
            dilate_disk_radius=params.target_dilate_radius,
        )

        object_folder = roi_root / f"object_{object_id:03d}"
        object_folder.mkdir(parents=True, exist_ok=True)

        reference_crop_path = object_folder / "reference_crop.tif"
        target_crop_path = object_folder / "target_crop.tif"
        reference_mask_path = object_folder / "reference_object_mask.tif"
        target_mask_path = object_folder / "target_object_mask.tif"
        reference_label_path = object_folder / "reference_object_labels.tif"
        target_label_path = object_folder / "target_object_labels.tif"
        valid_roi_mask_path = object_folder / "valid_roi_mask.tif"
        qc_path = object_folder / "roi_segmentation_qc.png"

        write_volume_tiff(reference_crop, reference_crop_path)
        write_volume_tiff(target_crop, target_crop_path)
        write_volume_tiff(reference_mask.astype(np.uint8), reference_mask_path)
        write_volume_tiff(target_mask.astype(np.uint8), target_mask_path)
        write_volume_tiff(reference_label, reference_label_path)
        write_volume_tiff(target_label, target_label_path)
        write_volume_tiff(np.ones_like(target_expanded, dtype=np.uint8), valid_roi_mask_path)

        save_roi_qc(
            reference_expanded,
            target_expanded,
            reference_mask,
            target_mask,
            object_id,
            reference_threshold,
            target_threshold,
            qc_path,
        )

        metadata = {
            "sample_id": series.sample_id,
            "condition": series.condition,
            "image_id": series.image_id,
            "object_id": object_id,
            "center_x": center_x,
            "center_y": center_y,
            "crop_x_start": x_start,
            "crop_x_end": x_end,
            "crop_y_start": y_start,
            "crop_y_end": y_end,
            "z_start": 1,
            "z_end": z_slices,
            "expand_z": params.expand_z,
            "min_volume_voxels": params.min_volume_voxels,
            "reference_threshold_m3": reference_threshold,
            "target_threshold_m3": target_threshold,
            "n_reference_objects": n_reference_objects,
            "n_target_objects": n_target_objects,
            "reference_stack_path": str(series.reference_stack_path),
            "target_stack_path": str(series.target_stack_path),
            "object_folder": str(object_folder),
            "qc_path": str(qc_path),
        }
        write_json(metadata, object_folder / "object_metadata.json")

        manifest_rows.append(
            {
                "sample_id": series.sample_id,
                "condition": series.condition,
                "image_id": series.image_id,
                "object_id": object_id,
                "center_x": center_x,
                "center_y": center_y,
                "crop_x_start": x_start,
                "crop_x_end": x_end,
                "crop_y_start": y_start,
                "crop_y_end": y_end,
                "z_start": 1,
                "z_end": z_slices,
                "expand_z": params.expand_z,
                "min_volume_voxels": params.min_volume_voxels,
                "reference_threshold_m3": reference_threshold,
                "target_threshold_m3": target_threshold,
                "n_reference_objects": n_reference_objects,
                "n_target_objects": n_target_objects,
                "reference_stack_path": str(series.reference_stack_path),
                "target_stack_path": str(series.target_stack_path),
                "reference_crop_path": str(reference_crop_path),
                "target_crop_path": str(target_crop_path),
                "reference_object_mask_path": str(reference_mask_path),
                "target_object_mask_path": str(target_mask_path),
                "valid_roi_mask_path": str(valid_roi_mask_path),
                "qc_path": str(qc_path),
            }
        )

    write_manifest_csv(roi_root / "roi_manifest.csv", manifest_rows)
    params_struct = asdict(params)
    params_struct.update(
        {
            "sample_id": series.sample_id,
            "condition": series.condition,
            "image_id": series.image_id,
            "reference_stack_path": str(series.reference_stack_path),
            "target_stack_path": str(series.target_stack_path),
            "roi_root": str(roi_root),
            "batch_timestamp": batch_timestamp,
        }
    )
    write_json(params_struct, roi_root / "roi_selection_parameters.json")
    print(f"Saved: {roi_root}")
    return manifest_rows


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def infer_condition(series_name: str) -> str:
    s = series_name.lower()
    if "fix" in s:
        return "FiX-ITS"
    if "multiexr" in s or "multi" in s:
        return "multi-ExR"
    if "panexm" in s or "panex" in s:
        return "pan-ExM-t"
    return "unknown"


def round_matlab(value: float) -> int:
    """Match MATLAB round for positive image coordinates."""
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def load_points_csv(path: Path) -> dict[str, np.ndarray]:
    """Load optional noninteractive points CSV with columns image_id,x,y."""
    points_by_image: dict[str, list[tuple[float, float]]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"image_id", "x", "y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            points_by_image.setdefault(row["image_id"], []).append(
                (float(row["x"]), float(row["y"]))
            )
    return {k: np.asarray(v, dtype=np.float64) for k, v in points_by_image.items()}


def run_batch(
    script_dir: Path,
    params: ROIParams,
    max_series: Optional[int] = None,
    points_csv: Optional[Path] = None,
    list_only: bool = False,
) -> list[dict]:
    data_root = script_dir / "data"
    results_dir = script_dir / "analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if params.batch_timestamp_override:
        batch_timestamp = params.batch_timestamp_override
    else:
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    series_list = discover_registered_series(data_root)
    if not series_list:
        raise RuntimeError(f"No registered Reference/Target stack pairs were found under {data_root}.")

    print(f"\nDiscovered {len(series_list)} registered stack pair(s):")
    for idx, series in enumerate(series_list, start=1):
        print(f"  {idx:03d}: {series.sample_id} | {series.condition} | {series.image_id}")

    if list_only:
        return []

    selected = series_list[params.start_at_series_index - 1 :]
    if max_series is not None:
        selected = selected[:max_series]

    points_by_image = load_points_csv(points_csv) if points_csv else {}
    batch_manifest: list[dict] = []
    total = len(series_list)
    for offset, series in enumerate(selected, start=params.start_at_series_index):
        rows = process_registered_series(
            series,
            params,
            batch_timestamp,
            offset,
            total,
            points_override=points_by_image.get(series.image_id),
        )
        batch_manifest.extend(rows)

    batch_manifest_path = results_dir / f"roi_selection_batch_{batch_timestamp}.csv"
    if batch_manifest:
        write_manifest_csv(batch_manifest_path, batch_manifest)
    print("\nBatch ROI selection complete.")
    print(f"Batch manifest: {batch_manifest_path}")
    return batch_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python port of ROI_selection_04.m for manual ROI selection.",
    )
    parser.add_argument("--script-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--crop-radius-xy", type=int, default=ROIParams.crop_radius_xy)
    parser.add_argument("--expand-z", type=int, default=ROIParams.expand_z)
    parser.add_argument("--min-volume-voxels", type=int, default=ROIParams.min_volume_voxels)
    parser.add_argument("--display-low-pct", type=float, default=ROIParams.display_low_pct)
    parser.add_argument("--display-high-pct", type=float, default=ROIParams.display_high_pct)
    parser.add_argument("--reference-smooth-sigma", type=float, default=ROIParams.reference_smooth_sigma)
    parser.add_argument("--target-smooth-sigma", type=float, default=ROIParams.target_smooth_sigma)
    parser.add_argument("--target-dilate-radius", type=int, default=ROIParams.target_dilate_radius)
    parser.add_argument("--start-at-series-index", type=int, default=ROIParams.start_at_series_index)
    parser.add_argument("--batch-timestamp-override", type=str, default="")
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--points-csv", type=Path, default=None)
    parser.add_argument("--list-only", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    params = ROIParams(
        crop_radius_xy=args.crop_radius_xy,
        expand_z=args.expand_z,
        min_volume_voxels=args.min_volume_voxels,
        display_low_pct=args.display_low_pct,
        display_high_pct=args.display_high_pct,
        reference_smooth_sigma=args.reference_smooth_sigma,
        target_smooth_sigma=args.target_smooth_sigma,
        target_dilate_radius=args.target_dilate_radius,
        start_at_series_index=args.start_at_series_index,
        batch_timestamp_override=args.batch_timestamp_override,
    )
    run_batch(
        script_dir=args.script_dir.resolve(),
        params=params,
        max_series=args.max_series,
        points_csv=args.points_csv,
        list_only=args.list_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
