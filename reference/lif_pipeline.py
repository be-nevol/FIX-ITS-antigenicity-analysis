"""
LIF / TIFF volumetric image drift-correction pipeline
=====================================================

Pipeline overview
-----------------
1. Parse images + paired metadata from either:
     - Leica .lif files (readlif), or
     - Pre-extracted per-channel TIFF stacks following the layout
       <base_dir>/<lif_name>/<series_name>/stacks/{DAPI,Reference,Target}_stack.tif
   The user is prompted at startup to choose mode A (.lif) or mode B (.tif).

2. For each volumetric multichannel series (3 channels, configurable Z),
   perform sequential slice-by-slice 2D drift correction with the SimpleITK
   ImageRegistrationMethod framework. This mirrors the MATLAB `imregtform`
   `monomodal` workflow.
     - For each Z=k (k>=1), the *fixed* image is the already-registered
       Reference slice at k-1, and the *moving* image is the raw Reference
       slice at k. A 2D translation is estimated with mean-squares metric
       and a regular-step gradient-descent optimizer.
     - The resulting transform is applied to the raw DAPI, Reference and
       Target slices at Z=k and written into the registered stacks.
     - If the change in transform between adjacent slices exceeds
       `jump_threshold_voxels`, registration is repeated using the previous
       slice's transform as initialization (recovery, not discard).
     - Z=0 is the anchor. Drift accumulation lives implicitly in the
       registered stack itself; no cumulative-offset bookkeeping is required.

3. Both raw and registered volumes are intensity-normalized per channel,
   per 3D volume using percentiles [2, 99.9] and rescaled to uint8 for
   display / GIF / MIP rendering.

4. Side-by-side animated GIFs (unregistered | registered) are produced per
   series, animating through Z, saved to <series>/GIFS/.

5. Maximum-intensity projections (merged RGB) of the registered volume are
   saved as PNG to <series>/MIP/.

Output tree (nested in each series folder for searchability):
  <base_dir>/
    <lif_name>/
      <series_name>/
        stacks/
          <series_name>_stack.tif
        registered_stacks/
          <series_name>_stack_registered.tif
        GIFS/
          <series_name>_sidebyside.gif
        MIP/
          <series_name>_registered_MIP.png
        metadata.json

Output format (ImageJ/Fiji hyperstack TIFF)
------------------------------------------
All TIFF files are written in ImageJ hyperstack format via ``tifffile``
with ``imagej=True``.  This gives Fiji the three things it needs without
Bio-Formats:

1. **Pixel scale** — ``XResolution`` / ``YResolution`` tags hold the
   lateral pixel size as an exact rational (e.g. 1000/97 for 0.097 µm/px),
   and the ``unit`` field in the ImageJ metadata string holds ``"um"``.
   Z-spacing goes into ``spacing``.  Fiji reads all three natively and
   displays a calibrated scale bar immediately on open.

2. **Multichannel hyperstack** — channels are interleaved as a ``(Z, C, Y, X)``
   array with ``axes='ZCYX'`` and ``mode='composite'``, so Fiji opens the
   file as a composite hyperstack rather than a plain grayscale stack.
   Raw and registered outputs are each a single file (not three separate
   per-channel files).

3. **Channel LUTs** — the ``CHANNEL_LUTS`` dict maps each channel label to
   a ``(3, 256)`` uint8 LUT that Fiji applies on open.  Defaults: DAPI=blue,
   Reference=green, Target=red.  The registered-stack file additionally
   carries the full registration provenance (parameters, per-slice transform
   log) in private TIFF tag 65000 as UTF-8 JSON.

``metadata.json`` is still written for human readability and backward
compatibility.  The read path in ``load_series_from_tiff_dir`` uses
``tifffile.imread`` on the single multichannel file and splits on the
channel axis to reconstruct the per-label ``stacks`` dict.

Dependencies
------------
- numpy
- scikit-image
- matplotlib
- readlif
- SimpleITK  (the standard PyPI wheel is sufficient — we use
  `ImageRegistrationMethod`, which does not require the SimpleElastix build.)
- imageio
- scipy (for Gaussian smoothing during registration metric computation;
  benchmarked faster than sitk.SmoothingRecursiveGaussian on 2D float32
  at typical confocal slice sizes)
- tifffile  (ImageJ hyperstack output; already a transitive dependency of
  scikit-image)

See the README / install section at the bottom of this file for environment
setup instructions.
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import numpy as np
import SimpleITK as sitk
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from readlif.reader import LifFile
from scipy.ndimage import gaussian_filter, label as nd_label, median_filter as nd_median_filter
from skimage import exposure, io


# =============================================================================
# Colormaps (mirrors the notebook conventions)
# =============================================================================
CMAP_BLUE = LinearSegmentedColormap.from_list("dapi", ["black", "blue"])
CMAP_GREEN = LinearSegmentedColormap.from_list("reference", ["black", "green"])
CMAP_RED = LinearSegmentedColormap.from_list("target", ["black", "red"])


# =============================================================================
# Channel LUTs for Fiji/ImageJ composite mode
# =============================================================================
# Each LUT is a (3, 256) uint8 array in [R, G, B] row order, as required by
# tifffile's ImageJ 'LUTs' metadata field.  Default assignments:
#   DAPI      -> blue   (channel 0)
#   Reference -> green  (channel 1)
#   Target    -> red    (channel 2)
# These are embedded in both raw and registered multichannel TIFFs so that
# Fiji opens the file in composite mode with the correct colour mapping
# without any manual LUT assignment.
def _make_lut(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    lut = np.zeros((3, 256), dtype=np.uint8)
    lut[0] = r; lut[1] = g; lut[2] = b
    return lut


_ramp = np.arange(256, dtype=np.uint8)
_zero = np.zeros(256, dtype=np.uint8)

CHANNEL_LUTS: dict[str, np.ndarray] = {
    "DAPI":      _make_lut(_zero, _zero, _ramp),   # blue
    "Reference": _make_lut(_zero, _ramp, _zero),   # green
    "Target":    _make_lut(_ramp, _zero, _zero),   # red
}
"""Default LUT per channel label.  Override by updating this dict before
calling ``process_series`` if your colour assignments differ."""


# =============================================================================
# Small helpers (kept consistent with notebook)
# =============================================================================
def sanitize_name(name: str) -> str:
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = str(name)
    for ch in bad_chars:
        out = out.replace(ch, "_")
    return out.strip()


def ensure_uint16(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.uint16):
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0, np.iinfo(np.uint16).max)
        return arr.astype(np.uint16)
    return arr.astype(np.uint16)


def safe_scale_to_dict(scale) -> Optional[dict]:
    if scale is None:
        return None
    try:
        if isinstance(scale, dict):
            return scale
        if isinstance(scale, (list, tuple)):
            return {
                "x": scale[0] if len(scale) > 0 else None,
                "y": scale[1] if len(scale) > 1 else None,
                "z": scale[2] if len(scale) > 2 else None,
                "t": scale[3] if len(scale) > 3 else None,
            }
        return {"raw": str(scale)}
    except Exception:
        return {"raw": str(scale)}


def normalize_for_view(
    data: np.ndarray, p_low: float = 2.0, p_high: float = 99.9
) -> np.ndarray:
    """
    Per-volume percentile-based intensity normalization to float [0, 1].
    [2, 99.9] per the user's specification.
    """
    data = np.asarray(data)
    if data.size == 0 or np.all(data == data.flat[0]):
        return np.zeros_like(data, dtype=np.float32)
    low, high = np.percentile(data, (p_low, p_high))
    if low == high:
        return np.zeros_like(data, dtype=np.float32)
    return exposure.rescale_intensity(
        data, in_range=(low, high), out_range=(0, 1)
    ).astype(np.float32)


def to_uint8_for_display(volume: np.ndarray) -> np.ndarray:
    """Normalize a 3D volume to uint8 with [2, 99.9] saturation."""
    norm = normalize_for_view(volume, p_low=2.0, p_high=99.9)
    return (norm * 255.0).astype(np.uint8)


# =============================================================================
# Section 1 — Parsing
# =============================================================================
@dataclass
class SeriesData:
    """Holds raw 3D stacks for a single series (one ROI/series within a .lif).

    `stacks` maps channel label -> 3D ndarray (Z, Y, X), uint16 preferred.
    """
    lif_name: str
    series_name: str
    series_index: int
    stacks: dict = field(default_factory=dict)         # {label: (Z,Y,X) ndarray}
    metadata: dict = field(default_factory=dict)


def inspect_single_lif(lif_path: Path, verbose: bool = True):
    """Inspect a single .lif file and return a list of per-series info dicts."""
    lif_path = Path(lif_path)
    if not lif_path.exists():
        print(f"[ERROR] File not found: {lif_path}")
        return None
    try:
        lif = LifFile(str(lif_path))
        if verbose:
            print(f"\nTarget File: {lif_path.name}")
            print(f"Full Path  : {lif_path}")
            print(f"Total number of images (ROI/Series): {lif.num_images}")
            print("=" * 70)

        results = []
        for i in range(lif.num_images):
            img = lif.get_image(i)
            info = {
                "index": i,
                "name": img.name,
                "x_size": getattr(img.dims, "x", None),
                "y_size": getattr(img.dims, "y", None),
                "z_slices": getattr(img.dims, "z", None),
                "t_frames": getattr(img.dims, "t", None),
                "channels": getattr(img, "channels", None),
                "scale": getattr(img, "scale", None),
            }
            results.append(info)
            if verbose:
                print(f"Index [{i}]: {info['name']}")
                print(f"  - Resolution: {info['x_size']} x {info['y_size']} pixels")
                print(f"  - Z-slices  : {info['z_slices']} planes")
                print(f"  - Time      : {info['t_frames']} frames")
                print(f"  - Channels  : {info['channels']} channels")
                print(f"  - Scale     : {info['scale']}")
                print("-" * 70)
        return results
    except Exception as e:
        print(f"[ERROR] Failed to read file: {lif_path}")
        print(f"Reason: {e}")
        return None


def get_channel_stack_from_lif(img, channel_idx: int) -> np.ndarray:
    """Return (Z, Y, X) stack for a given channel from a readlif image."""
    z_slices = getattr(img.dims, "z", 1)
    frames = []
    for z in range(z_slices):
        frames.append(np.array(img.get_frame(z=z, t=0, c=channel_idx)))
    return np.stack(frames, axis=0)


def load_series_from_lif(
    lif_path: Path,
    series_info: dict,
    channel_order=(0, 1, 2),
    channel_labels=("DAPI", "Reference", "Target"),
) -> SeriesData:
    lif = LifFile(str(lif_path))
    img = lif.get_image(series_info["index"])
    stacks = {}
    for ch_idx, ch_label in zip(channel_order, channel_labels):
        stacks[ch_label] = ensure_uint16(get_channel_stack_from_lif(img, ch_idx))

    metadata = {
        "lif_name": Path(lif_path).stem,
        "series_name": series_info["name"],
        "series_index": series_info["index"],
        "x_size": series_info.get("x_size"),
        "y_size": series_info.get("y_size"),
        "z_slices": series_info.get("z_slices"),
        "t_frames": series_info.get("t_frames"),
        "channels": series_info.get("channels"),
        "scale": safe_scale_to_dict(series_info.get("scale")),
        "channel_order": list(channel_order),
        "channel_labels": list(channel_labels),
        "source": "lif",
        "source_path": str(lif_path),
    }

    return SeriesData(
        lif_name=Path(lif_path).stem,
        series_name=series_info["name"],
        series_index=series_info["index"],
        stacks=stacks,
        metadata=metadata,
    )


def load_series_from_tiff_dir(
    series_dir: Path,
    channel_labels=("DAPI", "Reference", "Target"),
) -> Optional[SeriesData]:
    """Load a SeriesData from ``<series_dir>/stacks/<series_name>_stack.tif``.

    The file is expected to be a multichannel ImageJ hyperstack with shape
    ``(Z, C, Y, X)`` written by ``save_series_outputs``.  Channels are
    split on axis 1 in the order given by ``channel_labels``.

    Falls back to the old per-channel convention
    (``<label>_stack.tif`` per channel) if the multichannel file is absent,
    so existing output trees produced by earlier pipeline versions remain
    loadable.
    """
    series_dir = Path(series_dir)
    stacks_dir = series_dir / "stacks"
    if not stacks_dir.is_dir():
        return None

    stacks = {}

    # --- Primary path: multichannel file (current convention) ---
    mc_path = stacks_dir / f"{sanitize_name(series_dir.name)}_stack.tif"
    if mc_path.exists():
        arr = tifffile.imread(str(mc_path))  # shape (Z,C,Y,X) from imagej=True
        if arr.ndim == 4 and arr.shape[1] == len(channel_labels):
            for i, lbl in enumerate(channel_labels):
                stacks[lbl] = ensure_uint16(arr[:, i, :, :])
        else:
            print(f"[WARN] Multichannel stack has unexpected shape {arr.shape}: {mc_path}")
            return None

    # --- Fallback: legacy per-channel files ---
    else:
        for ch_label in channel_labels:
            tif_path = stacks_dir / f"{sanitize_name(ch_label)}_stack.tif"
            if not tif_path.exists():
                print(f"[WARN] Missing stack: {tif_path}")
                return None
            arr = io.imread(str(tif_path))
            if arr.ndim != 3:
                print(f"[WARN] Expected 3D stack, got shape {arr.shape}: {tif_path}")
                return None
            stacks[ch_label] = ensure_uint16(arr)

    # Try to load existing metadata.json; otherwise build a minimal one.
    metadata_path = series_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        any_stack = next(iter(stacks.values()))
        z, y, x = any_stack.shape
        metadata = {
            "lif_name": series_dir.parent.name,
            "series_name": series_dir.name,
            "series_index": None,
            "x_size": x,
            "y_size": y,
            "z_slices": z,
            "t_frames": 1,
            "channels": len(channel_labels),
            "scale": None,
            "channel_order": list(range(len(channel_labels))),
            "channel_labels": list(channel_labels),
            "source": "tiff",
            "source_path": str(series_dir),
        }

    return SeriesData(
        lif_name=metadata.get("lif_name", series_dir.parent.name),
        series_name=metadata.get("series_name", series_dir.name),
        series_index=metadata.get("series_index", -1) or -1,
        stacks=stacks,
        metadata=metadata,
    )


def discover_tiff_series(base_dir: Path, channel_labels=("DAPI", "Reference", "Target")):
    """Walk ``<base_dir>`` and find every series folder containing a valid stack.

    Accepts both the current convention (single multichannel
    ``<series_name>_stack.tif``) and the legacy convention (one
    ``<label>_stack.tif`` per channel), so output trees from older pipeline
    runs remain discoverable.
    """
    base_dir = Path(base_dir)
    found = []
    if not base_dir.is_dir():
        return found

    for lif_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for series_dir in sorted(p for p in lif_dir.iterdir() if p.is_dir()):
            stacks_dir = series_dir / "stacks"
            if not stacks_dir.is_dir():
                continue
            # Current convention: single multichannel file.
            mc_path = stacks_dir / f"{sanitize_name(series_dir.name)}_stack.tif"
            if mc_path.exists():
                found.append(series_dir)
                continue
            # Legacy fallback: one file per channel.
            if all(
                (stacks_dir / f"{sanitize_name(lbl)}_stack.tif").exists()
                for lbl in channel_labels
            ):
                found.append(series_dir)
    return found


# =============================================================================
# Section 1.5 — Optional preprocessing: scanner line-noise removal
# =============================================================================
# Targets resonant-scanner line artifacts in sparse fluorescence Z-stacks.
# These appear as isolated dim pixels scattered horizontally across one or
# more Y rows in a slice. Algorithm:
#   1. For each row, count nonzero-valued pixels.
#   2. Compute a local trend as the running median over a vertical
#      neighborhood (default 31 rows).
#   3. Flag rows whose nonzero density exceeds the local trend by more
#      than `detection_threshold` × MAD of the residual.
#   4. On flagged rows, perform 8-connected component labeling on the
#      slice's binary mask. Pixels belonging to components smaller than
#      `min_punctum_size` are zeroed; larger components (real puncta) are
#      preserved. This protects rows that were flagged solely because they
#      happen to traverse puncta-rich regions.
#
# The same correction is applied independently to each Z-slice and to each
# channel (DAPI, Reference, Target) so the registered output stays denoised.
# =============================================================================
def denoise_scanner_lines_2d(
    img: np.ndarray,
    detection_neighborhood: int = 31,
    detection_threshold: float = 4.0,
    min_punctum_size: int = 5,
) -> tuple[np.ndarray, dict]:
    """Remove horizontal scanner-line noise from a single 2D slice.

    Parameters
    ----------
    img : (H, W) ndarray
        Single 2D slice. Any numeric dtype; output matches input dtype.
    detection_neighborhood : int
        Number of rows for the local-trend median filter (forced odd).
    detection_threshold : float
        Detection sensitivity in MAD units. Lower = more rows flagged.
    min_punctum_size : int
        Connected components smaller than this (in pixels) are treated as
        scanner noise on flagged rows. Larger = more aggressive removal.

    Returns
    -------
    cleaned : ndarray, same shape and dtype as input
    info : dict with diagnostic fields:
        - "n_bad_rows"   : int, number of rows flagged
        - "n_pixels_zeroed" : int, total pixels removed
        - "bad_rows"     : list of int, indices of flagged rows
    """
    src_dtype = img.dtype
    img_f = img.astype(np.float32)
    H, W = img_f.shape

    # 1. Per-row nonzero density
    row_stat = (img_f > 0).sum(axis=1).astype(np.float32)

    # 2. Local trend
    nbhd = int(detection_neighborhood)
    if nbhd % 2 == 0:
        nbhd += 1
    local_trend = nd_median_filter(row_stat, size=nbhd, mode="reflect")

    # 3. MAD-based threshold
    residual = row_stat - local_trend
    mad = float(np.median(np.abs(residual - np.median(residual)))) + 1e-9
    sigma_mad = 1.4826 * mad
    bad_rows = residual > detection_threshold * sigma_mad

    cleaned = img_f.copy()
    n_zeroed = 0

    if bad_rows.any():
        # 4. 8-connected component labeling on the full-slice nonzero mask.
        binary = img_f > 0
        structure = np.ones((3, 3), dtype=int)
        labels, _ = nd_label(binary, structure=structure)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # background label

        for y in np.where(bad_rows)[0]:
            row_labels = labels[y]
            row_component_sizes = sizes[row_labels]
            scanner_mask = (row_labels > 0) & (row_component_sizes < min_punctum_size)
            n_zeroed += int(scanner_mask.sum())
            cleaned[y, scanner_mask] = 0

    if np.issubdtype(src_dtype, np.integer):
        info_dt = np.iinfo(src_dtype)
        cleaned = np.clip(np.round(cleaned), info_dt.min, info_dt.max).astype(src_dtype)
    else:
        cleaned = cleaned.astype(src_dtype)

    info = {
        "n_bad_rows": int(bad_rows.sum()),
        "n_pixels_zeroed": n_zeroed,
        "bad_rows": np.where(bad_rows)[0].tolist(),
    }
    return cleaned, info


def denoise_scanner_lines_volume(
    stack: np.ndarray,
    detection_neighborhood: int = 31,
    detection_threshold: float = 4.0,
    min_punctum_size: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, list]:
    """Apply `denoise_scanner_lines_2d` slice-by-slice over a (Z, Y, X) volume.

    Returns
    -------
    cleaned : same shape and dtype as input
    per_slice_info : list of dicts (one per Z), each as returned by the 2D func.
    """
    out = np.empty_like(stack)
    per_slice_info = []
    for z in range(stack.shape[0]):
        cleaned_slice, info = denoise_scanner_lines_2d(
            stack[z],
            detection_neighborhood=detection_neighborhood,
            detection_threshold=detection_threshold,
            min_punctum_size=min_punctum_size,
        )
        out[z] = cleaned_slice
        per_slice_info.append(info)
        if verbose and info["n_bad_rows"] > 0:
            print(f"   [denoise] Z={z}: {info['n_bad_rows']} bad rows, "
                  f"{info['n_pixels_zeroed']} pixels zeroed")
    return out, per_slice_info


def denoise_scanner_lines_series(
    series: SeriesData,
    detection_neighborhood: int = 31,
    detection_threshold: float = 4.0,
    min_punctum_size: int = 5,
    channel_labels=("DAPI", "Reference", "Target"),
    verbose: bool = False,
) -> dict:
    """Apply line denoising to every channel of a SeriesData IN PLACE.

    The same denoising is applied to all channels independently. This
    modifies ``series.stacks`` directly and returns a diagnostics dict
    with two extra keys per channel used by the reporting layer:

    ``"worst_z"``
        Index of the Z-slice with the most pixels zeroed for this channel.
    ``"worst_raw"``
        The raw (pre-denoising) 2D slice at ``worst_z`` as a uint16 ndarray.
        Saved here because ``series.stacks`` is overwritten in place and the
        original data is no longer accessible after this function returns.
    ``"worst_cleaned"``
        The corresponding cleaned 2D slice at ``worst_z``.
    """
    diagnostics = {}
    for label in channel_labels:
        if label not in series.stacks:
            continue
        if verbose:
            print(f"   [denoise] processing channel '{label}'")

        # Keep a reference to the raw stack before in-place replacement so
        # we can capture the worst slice without duplicating the full volume.
        raw_stack = series.stacks[label]

        cleaned, per_slice = denoise_scanner_lines_volume(
            raw_stack,
            detection_neighborhood=detection_neighborhood,
            detection_threshold=detection_threshold,
            min_punctum_size=min_punctum_size,
            verbose=verbose,
        )

        # Identify worst slice (most pixels zeroed) and save 2-D pair.
        worst_z = int(np.argmax([s["n_pixels_zeroed"] for s in per_slice]))
        worst_raw = raw_stack[worst_z].copy()
        worst_cleaned = cleaned[worst_z].copy()

        series.stacks[label] = cleaned
        diagnostics[label] = {
            "total_bad_rows": sum(s["n_bad_rows"] for s in per_slice),
            "total_pixels_zeroed": sum(s["n_pixels_zeroed"] for s in per_slice),
            "per_slice": per_slice,
            "worst_z": worst_z,
            "worst_raw": worst_raw,
            "worst_cleaned": worst_cleaned,
        }
    return diagnostics


# =============================================================================
# Section 2 — Sequential slice-by-slice drift correction
# =============================================================================
# Direct port of stack_Reg.m (MATLAB `imregtform`, 'monomodal').
#
# Strategy:
#   - For each Z=k (k>=1):
#       fixed  = registered Reference slice at k-1 (NOT raw)
#       moving = raw Reference slice at k
#     Estimate a 2D translation, then apply that single transform to the
#     raw DAPI / Reference / Target slices at Z=k. Drift accumulation lives
#     implicitly in the registered stack — no cumulative-offset bookkeeping.
#   - Jump detection: if |T(k) - T(k-1)| > jump_threshold_voxels, re-register
#     using T(k-1) as initial transform. This recovers from local minima
#     instead of discarding the slice.
#   - Z=0 is the anchor; nothing is composed back to it.
# =============================================================================
def _make_registration_method() -> sitk.ImageRegistrationMethod:
    """Build a registration method tuned to noisy confocal data.

    Strategy: multi-resolution pyramid (shrink factors [4, 2, 1]) plus
    mean-squares metric, plus moderate learning rate. The pyramid is the
    critical piece — at the coarsest level (1/4 resolution) each iteration
    covers ~4x more ground in physical space, so the optimizer can converge
    on shifts of ~10+ voxels in 200 iterations. Without the pyramid, a
    single-resolution config either uses a high learning rate (which gets
    stuck at the identity on noisy/banded images) or a low learning rate
    (which can't traverse more than ~2 voxels in 200 iterations).

    MATLAB's `imregtform` uses a similar internal pyramid by default; this
    matches that behavior.
    """
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.NONE)            # use all pixels
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    # 3-level pyramid. Coarsest level is 1/4 the resolution and is itself
    # additionally smoothed by sigma=2; this lets the optimizer find the
    # global basin even on shifts that are 10+ voxels at full resolution.
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    R.SetOptimizerScalesFromPhysicalShift()
    return R


def _prep_for_registration(
    image: np.ndarray, sigma: float = 1.0,
) -> np.ndarray:
    """Pre-process a 2D slice for the registration metric ONLY.

    The recovered transform is still applied to the raw input downstream;
    this preprocessing exists purely to give the optimizer a clean signal:

      1. Gaussian smoothing (sigma=1 by default) attenuates high-frequency
         shot noise and resonant-scanner banding, both of which contribute
         large mean-squares energy that is identical between adjacent
         slices and therefore drowns out the actual biological drift.
      2. Percentile-based normalization to [0, 1] makes mean-squares
         scale-invariant and prevents a few saturated pixels from
         dominating the optimizer's gradient.
    """
    img = image.astype(np.float32)
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)
    lo, hi = np.percentile(img, (1.0, 99.0))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    return img


def _register_translation_2d(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    init_tx: float = 0.0,
    init_ty: float = 0.0,
    smoothing_sigma: float = 1.0,
) -> tuple[float, float]:
    """Estimate a 2D translation that maps `moving_np` -> `fixed_np`.

    Both inputs are smoothed + percentile-normalized for the metric only.
    Returns (tx, ty) as a SimpleITK TranslationTransform's parameters,
    suitable for direct use with `_warp_translation_2d` below on the
    *raw* moving slice.
    """
    fixed_prep = _prep_for_registration(fixed_np, sigma=smoothing_sigma)
    moving_prep = _prep_for_registration(moving_np, sigma=smoothing_sigma)

    fixed = sitk.GetImageFromArray(fixed_prep)
    moving = sitk.GetImageFromArray(moving_prep)

    transform = sitk.TranslationTransform(2)
    transform.SetParameters((init_tx, init_ty))

    R = _make_registration_method()
    R.SetInitialTransform(transform, inPlace=True)
    R.Execute(fixed, moving)
    return transform.GetParameters()


def _warp_translation_2d(image_2d: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Apply a SimpleITK TranslationTransform with parameters (tx, ty).

    The parameters are used as-is — they come straight from the registration
    method, which already returns a transform in the correct moving->fixed
    convention for the resampler.
    """
    src_dtype = image_2d.dtype
    moving = sitk.GetImageFromArray(image_2d.astype(np.float32))

    T = sitk.TranslationTransform(2)
    T.SetParameters((tx, ty))

    out = sitk.Resample(moving, moving, T, sitk.sitkLinear, 0.0)
    out = sitk.GetArrayFromImage(out)

    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        out = np.clip(np.round(out), info.min, info.max).astype(src_dtype)
    else:
        out = out.astype(src_dtype)
    return out


def register_volume_slice_by_slice(
    series: SeriesData,
    reference_label: str = "Reference",
    jump_threshold_voxels: float = 5.0,
    fatal_jump_threshold: float = 10.0,
    smoothing_sigma: float = 1.0,
    channel_labels=("DAPI", "Reference", "Target"),
    verbose: bool = True,
):
    """Sequential drift correction — direct port of stack_Reg.m.

    Parameters
    ----------
    series : SeriesData
        Input volume (one ROI/series), with per-channel (Z, Y, X) stacks.
    reference_label : str
        Channel used to estimate the per-slice transform.
    jump_threshold_voxels : float
        If the change in transform between adjacent slices exceeds this,
        registration is repeated using the previous slice's transform as
        initialization (mirrors MATLAB `jump > 5` branch).
    fatal_jump_threshold : float
        Circuit-breaker. If even the recovery attempt produces an
        inter-slice jump larger than this, the loop aborts and the
        registered volumes are clipped to slices [0..k-1]. Prevents a
        thrashing optimizer from corrupting downstream slices. Set very
        high (e.g. inf) to disable.
    smoothing_sigma : float
        Gaussian sigma applied to slices for the registration metric ONLY.
        The recovered transform is applied to the raw (unsmoothed) data.
        Set to 0 to disable. Default 1.0 — appropriate for noisy confocal
        data dominated by shot noise or scanner banding.
    channel_labels : tuple
        Channel labels (kept for API symmetry; not used directly here).
    verbose : bool
        Print per-slice diagnostics.

    Returns
    -------
    dict with keys:
      - "registered_stacks": {label: (Z, Y, X) ndarray}  (may be clipped
        if the circuit breaker fired)
      - "transform_log": list of {z, tx, ty, reason} per slice.
    """
    del channel_labels  # unused — kept for caller-API consistency

    ref_stack = series.stacks[reference_label]
    Z = ref_stack.shape[0]

    # Allocate output volumes; copy Z=0 unchanged (anchor).
    registered = {label: np.empty_like(stack) for label, stack in series.stacks.items()}
    for label in series.stacks:
        registered[label][0] = series.stacks[label][0]

    # Per-slice (tx, ty) — index k-1 holds the transform that mapped
    # raw slice k onto registered slice k-1.
    trans = np.zeros((max(Z - 1, 0), 2), dtype=np.float64)

    transform_log = [{"z": 0, "tx": 0.0, "ty": 0.0, "reason": "anchor"}]

    if verbose:
        print(f"   [INFO] Registering {Z - 1} slices sequentially "
              f"(fixed = registered[k-1], moving = raw[k])")

    # Track whether the circuit breaker fired and at what slice. If it does,
    # we abort the loop and clip arrays to the last successfully-registered
    # slice (k-1, where k is the offending index).
    clipped_at = None  # None -> not clipped; otherwise int = first INVALID Z
    fatal_reason = None

    for k in range(1, Z):
        fixed = registered[reference_label][k - 1]   # registered, not raw
        moving = ref_stack[k]

        tx, ty = _register_translation_2d(
            fixed, moving, smoothing_sigma=smoothing_sigma,
        )

        # Jump detection — same condition as MATLAB lines 77–88.
        reason = "ok"
        if k > 1:
            prev_tx, prev_ty = trans[k - 2]
            jump = float(np.hypot(tx - prev_tx, ty - prev_ty))
            if jump > jump_threshold_voxels:
                if verbose:
                    print(f"   [INFO] Z={k}: large jump ({jump:.2f} > "
                          f"{jump_threshold_voxels}) — re-registering with "
                          f"previous transform as init")
                tx, ty = _register_translation_2d(
                    fixed, moving,
                    init_tx=prev_tx, init_ty=prev_ty,
                    smoothing_sigma=smoothing_sigma,
                )
                reason = "ok_after_jump_recovery"

                # Circuit breaker: if even the recovery attempt produced a
                # massive inter-slice jump, the optimizer is thrashing.
                # Abort and clip — better a short, valid stack than a long
                # one polluted by garbage transforms downstream.
                post_jump = float(np.hypot(tx - prev_tx, ty - prev_ty))
                if post_jump > fatal_jump_threshold:
                    fatal_reason = (
                        f"recovery still produced jump={post_jump:.2f} > "
                        f"fatal_jump_threshold={fatal_jump_threshold}"
                    )
                    if verbose:
                        print(f"   [FATAL] Z={k}: {fatal_reason} — clipping "
                              f"stack at Z={k - 1}")
                    transform_log.append({
                        "z": k,
                        "tx": float(tx),
                        "ty": float(ty),
                        "reason": "fatal_jump_circuit_breaker",
                    })
                    clipped_at = k
                    break  # do NOT write registered[k]; abort the loop

        trans[k - 1] = (tx, ty)

        # Apply the single transform to all channels at Z=k.
        for label, stack in series.stacks.items():
            registered[label][k] = _warp_translation_2d(stack[k], tx, ty)

        transform_log.append({
            "z": k,
            "tx": float(tx),
            "ty": float(ty),
            "reason": reason,
        })

    # If the circuit breaker fired, clip every channel's registered stack
    # to the last good slice. The last valid index is clipped_at - 1, so
    # we keep indices [0 .. clipped_at) which has length `clipped_at`.
    if clipped_at is not None:
        for label in registered:
            registered[label] = registered[label][:clipped_at]
        if verbose:
            print(f"   [INFO] Registration ABORTED at Z={clipped_at}. "
                  f"Output volumes clipped to {clipped_at} slices.")
    elif verbose:
        print(f"   [INFO] Registration complete ({Z - 1} slices, Z=0 anchor).")

    return {"registered_stacks": registered, "transform_log": transform_log}

# =============================================================================
# Section 3 — Saving registered stacks + extended metadata
# =============================================================================
def _write_imagej_tiff(
    path: Path,
    arr_zcyx: np.ndarray,
    pixel_size_um_xy: Optional[float],
    pixel_size_um_z: Optional[float],
    channel_labels: tuple,
    extra_json: Optional[dict] = None,
) -> None:
    """Write a ``(Z, C, Y, X)`` uint16 stack as a Fiji-compatible ImageJ
    hyperstack TIFF.

    Why ``imagej=True`` over ``ome=True``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Fiji reads pixel calibration from the TIFF ``XResolution`` /
    ``YResolution`` tags together with the ``unit`` field in the ImageJ
    metadata string.  With ``ome=True`` alone, tifffile leaves
    ``XResolution=(1,1)`` and ``ResolutionUnit=1`` (no unit), so Fiji shows
    no pixel scale unless Bio-Formats is installed and invoked explicitly.
    ``imagej=True`` writes the resolution as an exact rational tag
    (e.g. 1000/97 for 0.097 µm/px) and ``unit='um'`` in the ImageJ
    description, which Fiji reads natively.

    Parameters
    ----------
    path : Path
        Output file path (created or overwritten).
    arr_zcyx : (Z, C, Y, X) uint16 ndarray
        All channels stacked on axis 1 in the same order as
        ``channel_labels``.
    pixel_size_um_xy : float or None
        Lateral pixel size in micrometres.  None → no calibration written.
    pixel_size_um_z : float or None
        Axial step size in micrometres written as ``spacing``.
    channel_labels : tuple of str
        Label for each channel; used to look up ``CHANNEL_LUTS``.  Any
        label not present in ``CHANNEL_LUTS`` gets a white (identity) LUT.
    extra_json : dict or None
        Arbitrary JSON-serialisable dict embedded in private TIFF tag 65000
        as UTF-8 JSON.  Used for registration provenance on registered-stack
        files; ``None`` for raw stacks.
    """
    ij_meta: dict = {
        "axes": "ZCYX",
        "hyperstack": True,
        "mode": "composite",
        "LUTs": [
            CHANNEL_LUTS.get(lbl, _make_lut(_ramp, _ramp, _ramp))
            for lbl in channel_labels
        ],
    }
    if pixel_size_um_z is not None:
        ij_meta["spacing"] = float(pixel_size_um_z)
    if pixel_size_um_xy is not None:
        ij_meta["unit"] = "um"

    resolution = None
    if pixel_size_um_xy is not None:
        resolution = (1.0 / pixel_size_um_xy, 1.0 / pixel_size_um_xy)

    extratags = []
    if extra_json is not None:
        payload = json.dumps(extra_json, default=str)
        extratags.append((65000, "s", 1, payload, True))

    tifffile.imwrite(
        str(path),
        ensure_uint16(arr_zcyx),
        imagej=True,
        resolution=resolution,
        metadata=ij_meta,
        software="lif_drift_correction_pipeline",
        extratags=extratags if extratags else None,
    )


def _summarize_denoise_diag(diag: Optional[dict]) -> Optional[dict]:
    """Compress per-slice denoising diagnostics into a compact per-channel
    summary suitable for metadata.json.

    Strips the ``worst_raw`` and ``worst_cleaned`` ndarray fields that
    ``denoise_scanner_lines_series`` now attaches for the reporting layer —
    those are not JSON-serialisable and are not needed in the JSON record.
    """
    if not diag:
        return None
    summary = {}
    for label, ch_diag in diag.items():
        per_slice = ch_diag.get("per_slice", [])
        slices_with_correction = [
            s["bad_rows"] for s in per_slice if s["n_bad_rows"] > 0
        ]
        summary[label] = {
            "total_bad_rows": ch_diag.get("total_bad_rows", 0),
            "total_pixels_zeroed": ch_diag.get("total_pixels_zeroed", 0),
            "n_slices_affected": len(slices_with_correction),
            "n_slices_total": len(per_slice),
            "worst_z": ch_diag.get("worst_z"),
        }
    return summary


def save_series_outputs(
    series: SeriesData,
    base_dir: Path,
    registered_stacks: dict,
    transform_log: list,
    channel_labels=("DAPI", "Reference", "Target"),
    save_raw_stacks_if_missing: bool = True,
    registration_params: Optional[dict] = None,
):
    """Save raw and registered multichannel TIFFs plus metadata.json.

    Output format
    -------------
    Each series produces **two** ImageJ hyperstack TIFFs — one for the raw
    input and one for the registered output — each containing all channels
    as a ``(Z, C, Y, X)`` array.  This replaces the previous convention of
    writing a separate ``<label>_stack.tif`` per channel.

    ``stacks/<series_name>_stack.tif``
        Raw multichannel input.  Written once; skipped on re-runs if the
        file already exists.

    ``registered_stacks/<series_name>_stack_registered.tif``
        Registered multichannel output.  Always overwritten so re-runs
        update results without accumulating stale files.

    Both files are written with ``imagej=True`` so Fiji opens them
    correctly without Bio-Formats:

    * ``XResolution`` / ``YResolution`` tags carry the lateral pixel size as
      an exact rational; ``unit='um'`` and ``spacing=<z_step>`` in the IJ
      metadata string complete the 3-D calibration.
    * ``mode='composite'`` + ``LUTs`` entries give each channel its colour
      on open (DAPI=blue, Reference=green, Target=red by default).
    * Private tag 65000 on the **registered** file carries the full
      registration provenance as UTF-8 JSON; the raw file carries no
      provenance payload.

    ``metadata.json`` is still written for human readability and backward
    compatibility.
    """
    base_dir = Path(base_dir)
    lif_dir = base_dir / sanitize_name(series.lif_name)
    series_dir = lif_dir / sanitize_name(series.series_name)
    stacks_dir = series_dir / "stacks"
    reg_dir = series_dir / "registered_stacks"

    series_dir.mkdir(parents=True, exist_ok=True)
    reg_dir.mkdir(parents=True, exist_ok=True)

    # Pixel sizes from .lif scale metadata; None if unavailable.
    scale = series.metadata.get("scale") or {}
    try:
        pixel_size_um_xy = float(scale["x"]) if scale.get("x") is not None else None
    except (TypeError, ValueError):
        pixel_size_um_xy = None
    try:
        pixel_size_um_z = float(scale["z"]) if scale.get("z") is not None else None
    except (TypeError, ValueError):
        pixel_size_um_z = None

    # Multichannel array assembly — stack channels on axis 1 (ZCYX).
    # Channel order follows channel_labels so LUT indexing stays consistent.
    def _stack_channels(src: dict) -> np.ndarray:
        return np.stack(
            [ensure_uint16(src[lbl]) for lbl in channel_labels], axis=1
        )

    series_stem = sanitize_name(series.series_name)

    # ------------------------------------------------------------------
    # Raw multichannel TIFF — written once; skipped on re-runs.
    # ------------------------------------------------------------------
    if save_raw_stacks_if_missing:
        stacks_dir.mkdir(parents=True, exist_ok=True)
        raw_path = stacks_dir / f"{series_stem}_stack.tif"
        if not raw_path.exists():
            _write_imagej_tiff(
                path=raw_path,
                arr_zcyx=_stack_channels(series.stacks),
                pixel_size_um_xy=pixel_size_um_xy,
                pixel_size_um_z=pixel_size_um_z,
                channel_labels=channel_labels,
                extra_json=None,
            )

    # ------------------------------------------------------------------
    # Registered multichannel TIFF — always written; includes provenance.
    # ------------------------------------------------------------------
    actual_z = registered_stacks[channel_labels[0]].shape[0]
    original_z = series.metadata.get("z_slices", actual_z) or actual_z
    has_fatal_break = any(
        entry.get("reason") == "fatal_jump_circuit_breaker"
        for entry in transform_log
        if isinstance(entry, dict)
    )
    is_clipped = actual_z < original_z or has_fatal_break

    rp = registration_params or {}
    reg_block = {
        "method": "SimpleITK ImageRegistrationMethod (mean-squares, "
                  "regular-step gradient descent, multi-resolution pyramid)",
        "transform": "translation (2D)",
        "reference_channel": rp.get("reference_channel", "Reference"),
        "fixed_image": "registered slice at k-1 (sequential)",
        "jump_threshold_voxels": rp.get("jump_threshold_voxels"),
        "fatal_jump_threshold": rp.get("fatal_jump_threshold"),
        "smoothing_sigma_for_metric": rp.get("smoothing_sigma"),
        "anchor_slice": 0,
        "compose_across_slices": False,
        "clipped_due_to_failure": is_clipped,
        "original_z_slices": original_z,
        "final_z_slices": actual_z,
        "transform_log": transform_log,
    }
    preprocessing_block = {
        "denoise_lines": bool(rp.get("denoise_lines", False)),
        "denoise_detection_neighborhood": rp.get("denoise_detection_neighborhood"),
        "denoise_detection_threshold": rp.get("denoise_detection_threshold"),
        "denoise_min_punctum_size": rp.get("denoise_min_punctum_size"),
        "denoise_diagnostics_summary": _summarize_denoise_diag(
            rp.get("denoise_diagnostics")
        ),
    }
    provenance = {
        "lif_name": series.lif_name,
        "series_name": series.series_name,
        "series_index": series.series_index,
        "source": series.metadata.get("source"),
        "channel_labels": list(channel_labels),
        "registration": reg_block,
        "preprocessing": preprocessing_block,
    }

    reg_path = reg_dir / f"{series_stem}_stack_registered.tif"
    _write_imagej_tiff(
        path=reg_path,
        arr_zcyx=_stack_channels(registered_stacks),
        pixel_size_um_xy=pixel_size_um_xy,
        pixel_size_um_z=pixel_size_um_z,
        channel_labels=channel_labels,
        extra_json=provenance,
    )

    # Backward-compatible per-channel registered TIFFs.  Downstream ROI and
    # intensity notebooks expect these paths, and clipped registration output
    # is valid as long as every channel is written with the surviving Z range.
    for ch_label in channel_labels:
        tif_path = reg_dir / f"{sanitize_name(ch_label)}_stack_registered.tif"
        io.imsave(
            str(tif_path),
            ensure_uint16(registered_stacks[ch_label]),
            check_contrast=False,
        )

    # ------------------------------------------------------------------
    # metadata.json — kept for human readability and backward compat.
    # ------------------------------------------------------------------
    metadata = dict(series.metadata)
    metadata["z_slices"] = actual_z
    metadata["registration"] = reg_block
    metadata["preprocessing"] = preprocessing_block
    with open(series_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    series.metadata = metadata

    return series_dir


# =============================================================================
# Section 4 — Side-by-side GIFs (unregistered | registered)
# =============================================================================
def _merge_rgb_from_stacks_at_z(
    stacks_uint8: dict, z: int,
    channel_labels=("DAPI", "Reference", "Target"),
) -> np.ndarray:
    """Build a single RGB frame at depth z from the per-channel uint8 stacks.
    Color mapping (matches notebook): Target=R, Reference=G, DAPI=B.
    """
    dapi = stacks_uint8[channel_labels[0]][z]
    ref = stacks_uint8[channel_labels[1]][z]
    tgt = stacks_uint8[channel_labels[2]][z]
    rgb = np.stack([tgt, ref, dapi], axis=-1)
    return rgb


def _normalize_stacks_to_uint8(stacks: dict) -> dict:
    """Per-channel, per-volume normalization → uint8."""
    return {label: to_uint8_for_display(stack) for label, stack in stacks.items()}


def _add_label_banner(frame: np.ndarray, text: str, banner_h: int = 28) -> np.ndarray:
    """Add a black banner with white text on top of an RGB frame.
    Pure-numpy / matplotlib-free banner using a small bitmap font would be
    overkill; instead we just leave a black band and rely on filenames /
    montage layout. Banners are drawn in the figure montage at save-time
    via matplotlib in `make_sidebyside_gif`.
    """
    return frame  # banner handled by composition function below


def make_sidebyside_gif(
    series: SeriesData,
    registered_stacks: dict,
    out_path: Path,
    channel_labels=("DAPI", "Reference", "Target"),
    duration_per_frame: float = 0.25,
    label_height: int = 30,
):
    """Build a single side-by-side GIF (unregistered | registered).

    If the registered volume was clipped by the circuit breaker, only the
    surviving slices are animated, with the raw side clipped to match.
    """
    raw_u8 = _normalize_stacks_to_uint8(series.stacks)
    reg_u8 = _normalize_stacks_to_uint8(registered_stacks)

    # Use the (possibly clipped) registered stack length as the GIF length.
    z_slices = reg_u8[channel_labels[0]].shape[0]
    raw_z = raw_u8[channel_labels[0]].shape[0]
    clipped = z_slices < raw_z

    # Use matplotlib to compose each frame so we can add titles cleanly.
    frames = []
    for z in range(z_slices):
        raw_rgb = _merge_rgb_from_stacks_at_z(raw_u8, z, channel_labels)
        reg_rgb = _merge_rgb_from_stacks_at_z(reg_u8, z, channel_labels)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), facecolor="black")
        title_suffix = f"  [CLIPPED at Z={z_slices}/{raw_z}]" if clipped else ""
        fig.suptitle(
            f"{series.series_name}  |  Z = {z + 1}/{z_slices}{title_suffix}",
            color="white", fontsize=11, y=0.98,
        )
        for ax, im, title in zip(axes, [raw_rgb, reg_rgb],
                                 ["Unregistered", "Registered"]):
            ax.imshow(im)
            ax.set_title(title, color="white", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        fig.canvas.draw()
        # Use buffer_rgba (compatible with matplotlib >= 3.8) and drop alpha.
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgba = rgba.reshape(h, w, 4)
        frame = rgba[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, duration=duration_per_frame, loop=0)
    return out_path


# =============================================================================
# Section 5 — MIP of registered volume (merged RGB)
# =============================================================================
def make_registered_mip_png(
    series: SeriesData,
    registered_stacks: dict,
    out_path: Path,
    channel_labels=("DAPI", "Reference", "Target"),
    bar_length_um: float = 20.0,
    scale_bar_thickness: int = 6,
    scale_bar_fontsize: int = 12,
    show: bool = False,
    ):
    """Save a 4-panel PNG: DAPI MIP, Reference MIP, Target MIP, merged MIP.

    Uses [2, 99.9] percentile normalization on the *MIP image* so each panel
    is well-displayed (consistent with the per-volume display normalization
    convention).
    """
    mips = {label: np.max(stack, axis=0) for label, stack in registered_stacks.items()}

    dapi_n = normalize_for_view(mips[channel_labels[0]])
    ref_n = normalize_for_view(mips[channel_labels[1]])
    tgt_n = normalize_for_view(mips[channel_labels[2]])

    merged = np.zeros((*dapi_n.shape, 3), dtype=np.float32)
    merged[..., 0] = tgt_n
    merged[..., 1] = ref_n
    merged[..., 2] = dapi_n

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="black")
    fig.suptitle(
        f"{series.lif_name}  |  {series.series_name}  |  Registered MIP",
        color="white", fontsize=18, fontweight="bold", y=0.96,
    )
    panels = [
        (dapi_n, CMAP_BLUE, channel_labels[0]),
        (ref_n, CMAP_GREEN, channel_labels[1]),
        (tgt_n, CMAP_RED, channel_labels[2]),
        (merged, None, "Merged"),
    ]

    # Scale bar from metadata if available
    pixel_size_um = None
    scale = series.metadata.get("scale")
    if isinstance(scale, dict):
        pixel_size_um = scale.get("x")

    bar_pixels = None
    if pixel_size_um not in (None, 0):
        try:
            bar_pixels = bar_length_um / float(pixel_size_um)
        except Exception:
            bar_pixels = None

    for ax, (im, cmap, title) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color="white", fontsize=14)
        ax.axis("off")
        if bar_pixels is not None:
            scalebar = AnchoredSizeBar(
                ax.transData, bar_pixels, f"{bar_length_um} µm",
                "lower right", pad=0.6, borderpad=0.8, sep=6,
                color="white", frameon=False,
                size_vertical=scale_bar_thickness,
                fontproperties={"size": scale_bar_fontsize},
            )
            ax.add_artist(scalebar)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor="black")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# =============================================================================
# Top-level orchestration
# =============================================================================
def process_series(
    series: SeriesData,
    base_dir: Path,
    channel_labels=("DAPI", "Reference", "Target"),
    jump_threshold_voxels: float = 5.0,
    fatal_jump_threshold: float = 10.0,
    smoothing_sigma: float = 1.0,
    denoise_lines: bool = False,
    denoise_detection_neighborhood: int = 31,
    denoise_detection_threshold: float = 4.0,
    denoise_min_punctum_size: int = 5,
    gif_duration_per_frame: float = 0.25,
    verbose: bool = True,
    show_mip: bool = False,
    save_raw_stacks_if_missing: bool = True,
):
    print(f"\n[PROCESS] {series.lif_name} :: {series.series_name}")

    # Step 1.5 (optional): preprocess all channels to suppress horizontal
    # scanner-line noise. Applied IN PLACE to series.stacks before
    # registration, so the registered output inherits the denoising.
    denoise_diagnostics = None
    if denoise_lines:
        if verbose:
            print(f"   [denoise] removing scanner-line noise from all channels "
                  f"(threshold={denoise_detection_threshold} MAD, "
                  f"min_punctum_size={denoise_min_punctum_size})")
        denoise_diagnostics = denoise_scanner_lines_series(
            series=series,
            detection_neighborhood=denoise_detection_neighborhood,
            detection_threshold=denoise_detection_threshold,
            min_punctum_size=denoise_min_punctum_size,
            channel_labels=channel_labels,
            verbose=verbose,
        )
        if verbose:
            for label, diag in denoise_diagnostics.items():
                print(f"   [denoise] {label:<10s}: "
                      f"{diag['total_bad_rows']} bad rows, "
                      f"{diag['total_pixels_zeroed']} pixels zeroed")

    # Step 2: drift correction
    reg_result = register_volume_slice_by_slice(
        series=series,
        reference_label=channel_labels[1],  # "Reference" channel
        jump_threshold_voxels=jump_threshold_voxels,
        fatal_jump_threshold=fatal_jump_threshold,
        smoothing_sigma=smoothing_sigma,
        channel_labels=channel_labels,
        verbose=verbose,
    )

    # Step 2 (save): registered stacks + metadata
    series_dir = save_series_outputs(
        series=series,
        base_dir=base_dir,
        registered_stacks=reg_result["registered_stacks"],
        transform_log=reg_result["transform_log"],
        channel_labels=channel_labels,
        save_raw_stacks_if_missing=save_raw_stacks_if_missing,
        registration_params={
            "reference_channel": channel_labels[1],
            "jump_threshold_voxels": jump_threshold_voxels,
            "fatal_jump_threshold": fatal_jump_threshold,
            "smoothing_sigma": smoothing_sigma,
            "denoise_lines": denoise_lines,
            "denoise_detection_neighborhood": denoise_detection_neighborhood,
            "denoise_detection_threshold": denoise_detection_threshold,
            "denoise_min_punctum_size": denoise_min_punctum_size,
            "denoise_diagnostics": denoise_diagnostics,
        },
    )

    # Step 4: side-by-side GIF
    gif_path = series_dir / "GIFS" / f"{sanitize_name(series.series_name)}_sidebyside.gif"
    make_sidebyside_gif(
        series=series,
        registered_stacks=reg_result["registered_stacks"],
        out_path=gif_path,
        channel_labels=channel_labels,
        duration_per_frame=gif_duration_per_frame,
    )
    print(f"   [SAVE] GIF -> {gif_path}")

    # Step 5: MIP
    mip_path = series_dir / "MIP" / f"{sanitize_name(series.series_name)}_registered_MIP.png"
    make_registered_mip_png(
        series=series,
        registered_stacks=reg_result["registered_stacks"],
        out_path=mip_path,
        channel_labels=channel_labels,
        show=show_mip,
    )
    print(f"   [SAVE] MIP -> {mip_path}")

    return {
        "series": series,
        "series_dir": series_dir,
        "gif_path": gif_path,
        "mip_path": mip_path,
        "transform_log": reg_result["transform_log"],
        "registered_stacks": reg_result["registered_stacks"],
        "denoise_diagnostics": denoise_diagnostics,
        "channel_labels": channel_labels,
    }


# =============================================================================
# Section 6 — Per-LIF reporting: CSV, denoise validation PNGs, PDF
# =============================================================================
_PIPELINE_VERSION = "1.3.0"


def _make_denoise_validation_png(
    series_name: str,
    channel_label: str,
    worst_z: int,
    raw_slice: np.ndarray,
    cleaned_slice: np.ndarray,
    n_pixels_zeroed: int,
    n_bad_rows: int,
    detection_neighborhood: int,
    detection_threshold: float,
    min_punctum_size: int,
    out_path: Path,
) -> Path:
    """Save a three-panel denoising validation figure matching notebook block 7.

    Layout and styling reproduce the notebook cell exactly:
      - white background, fontsize=12 suptitle
      - two-line title: series/channel/Z on line 1, parameters on line 2
      - Panel 0: Original
      - Panel 1: Cleaned (<n_pixels_zeroed> px zeroed on <n_bad_rows> rows)
      - Panel 2: Difference (cleaned - original), ±<dmax>
      - vmin/vmax from percentiles [1, 99.5] of the raw slice
      - dmax from max(percentile(|diff|, 99), 1.0)

    Parameters
    ----------
    n_bad_rows : int
        Number of rows flagged on the worst slice — shown in panel 1 title.
    """
    img_f = raw_slice.astype(np.float32)
    cleaned_f = cleaned_slice.astype(np.float32)
    diff = cleaned_f - img_f

    vmin, vmax = np.percentile(img_f, 1), np.percentile(img_f, 99.5)
    dmax = max(float(np.percentile(np.abs(diff), 99)), 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Line denoising preview — {series_name} | {channel_label} | Z={worst_z}\n"
        f"NEIGHBORHOOD={detection_neighborhood}, "
        f"THRESHOLD={detection_threshold} MAD, "
        f"MIN_PUNCTUM_SIZE={min_punctum_size}",
        fontsize=12,
    )

    axes[0].imshow(img_f, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cleaned_f, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(
        f"Cleaned ({n_pixels_zeroed} px zeroed on {n_bad_rows} rows)"
    )
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title(f"Difference (cleaned - original), ±{dmax:.1f}")
    axes[2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def generate_lif_report(
    lif_name: str,
    lif_dir: Path,
    series_results: list,
    run_timestamp: str,
    source_path: Optional[str] = None,
) -> dict:
    """Generate CSV, per-series denoising PNGs, and a PDF report for one LIF.

    Parameters
    ----------
    lif_name : str
        Stem of the .lif filename (used in output filenames and report titles).
    lif_dir : Path
        ``<base_dir>/<lif_name>/`` — CSV and PDF are written here.
    series_results : list of dict
        Each dict is the return value of ``process_series`` for one series,
        augmented with the keys added in this version: ``"series"``,
        ``"registered_stacks"``, ``"denoise_diagnostics"``,
        ``"channel_labels"``, ``"transform_log"``, ``"series_dir"``.
    run_timestamp : str
        ISO-format timestamp string for the cover page.
    source_path : str or None
        Original .lif file path for the cover page (None for TIFF-mode runs).

    Returns
    -------
    dict with keys ``"csv_path"``, ``"pdf_path"``,
    ``"denoise_png_paths"`` (list).
    """
    lif_dir = Path(lif_dir)
    lif_dir.mkdir(parents=True, exist_ok=True)

    denoise_png_paths: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Denoise validation PNGs (one per series, if denoising was run)
    # ------------------------------------------------------------------
    for res in series_results:
        diag = res.get("denoise_diagnostics")
        if not diag:
            continue
        series = res["series"]
        series_dir = res["series_dir"]
        ch_labels = res["channel_labels"]

        # Find the channel+slice with the globally highest pixel-zero count.
        best_label = max(
            (lbl for lbl in ch_labels if lbl in diag),
            key=lambda lbl: diag[lbl].get("total_pixels_zeroed", 0),
        )
        ch_diag = diag[best_label]
        worst_z = ch_diag["worst_z"]
        raw_sl = ch_diag["worst_raw"]
        cln_sl = ch_diag["worst_cleaned"]
        n_zeroed = ch_diag["per_slice"][worst_z]["n_pixels_zeroed"]

        # Recover denoising parameters from series metadata if stored there,
        # otherwise use sensible display defaults.
        reg_meta = series.metadata.get("preprocessing", {})
        nbhd = reg_meta.get("denoise_detection_neighborhood", "?")
        thr = reg_meta.get("denoise_detection_threshold", "?")
        mps = reg_meta.get("denoise_min_punctum_size", "?")

        png_path = (
            series_dir
            / f"{sanitize_name(series.series_name)}_denoise_validation.png"
        )
        _make_denoise_validation_png(
            series_name=series.series_name,
            channel_label=best_label,
            worst_z=worst_z,
            raw_slice=raw_sl,
            cleaned_slice=cln_sl,
            n_pixels_zeroed=n_zeroed,
            n_bad_rows=ch_diag["per_slice"][worst_z]["n_bad_rows"],
            detection_neighborhood=nbhd,
            detection_threshold=thr,
            min_punctum_size=mps,
            out_path=png_path,
        )
        denoise_png_paths.append(png_path)
        print(f"   [REPORT] Denoise PNG -> {png_path}")

    # ------------------------------------------------------------------
    # 2. CSV — one row per Z-slice per series
    # ------------------------------------------------------------------
    import csv
    csv_path = lif_dir / f"{sanitize_name(lif_name)}_transform_log.csv"
    csv_rows: list[dict] = []
    for res in series_results:
        series = res["series"]
        log = res["transform_log"]
        txs = [e["tx"] for e in log]
        tys = [e["ty"] for e in log]
        for i, entry in enumerate(log):
            prev_tx = txs[i - 1] if i > 0 else entry["tx"]
            prev_ty = tys[i - 1] if i > 0 else entry["ty"]
            csv_rows.append({
                "lif_name": lif_name,
                "series_name": series.series_name,
                "z": entry["z"],
                "tx": round(entry["tx"], 6),
                "ty": round(entry["ty"], 6),
                "displacement": round(
                    float(np.hypot(entry["tx"], entry["ty"])), 6
                ),
                "delta": round(
                    float(np.hypot(entry["tx"] - prev_tx, entry["ty"] - prev_ty)), 6
                ),
                "reason": entry["reason"],
            })

    fieldnames = ["lif_name", "series_name", "z", "tx", "ty",
                  "displacement", "delta", "reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"   [REPORT] CSV -> {csv_path}")

    # ------------------------------------------------------------------
    # 3. PDF — built entirely with matplotlib PDF backend
    # ------------------------------------------------------------------
    from matplotlib.backends.backend_pdf import PdfPages
    from collections import Counter

    pdf_path = lif_dir / f"{sanitize_name(lif_name)}_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:

        # ---- Cover page ----
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_facecolor("white")

        y = 0.88
        ax.text(0.5, y, "LIF Drift-Correction Pipeline — Run Report",
                ha="center", va="top", fontsize=20, fontweight="bold",
                transform=ax.transAxes)
        y -= 0.07
        ax.text(0.5, y, lif_name, ha="center", va="top", fontsize=15,
                color="#333333", transform=ax.transAxes)
        y -= 0.06

        cover_fields = [
            ("Pipeline version", _PIPELINE_VERSION),
            ("Run timestamp", run_timestamp),
            ("Source file", str(source_path) if source_path else "TIFF mode"),
            ("Series processed", str(len(series_results))),
            ("Series clipped by circuit breaker",
             str(sum(
                 1 for r in series_results
                 if r["series"].metadata.get("registration", {})
                    .get("clipped_due_to_failure", False)
             ))),
            ("Denoising applied",
             "Yes" if any(r.get("denoise_diagnostics") for r in series_results)
             else "No"),
        ]
        for label, value in cover_fields:
            ax.text(0.15, y, f"{label}:", ha="left", va="top", fontsize=11,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.42, y, value, ha="left", va="top", fontsize=11,
                    transform=ax.transAxes)
            y -= 0.055

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- Summary table page ----
        col_headers = ["Series", "Z (orig→final)", "XY µm/px", "Z µm/step",
                       "Denoised", "Clipped", "Jump recoveries"]
        table_data = []
        for res in series_results:
            s = res["series"]
            meta = s.metadata
            reg = meta.get("registration", {})
            scale = meta.get("scale") or {}
            px_xy = scale.get("x")
            px_z = scale.get("z")
            orig_z = reg.get("original_z_slices", meta.get("z_slices", "?"))
            final_z = reg.get("final_z_slices", meta.get("z_slices", "?"))
            z_str = (f"{orig_z}→{final_z}" if reg.get("clipped_due_to_failure")
                     else str(final_z))
            log = res["transform_log"]
            recoveries = sum(
                1 for e in log if e["reason"] == "ok_after_jump_recovery"
            )
            table_data.append([
                s.series_name[:30],
                z_str,
                f"{px_xy:.4f}" if px_xy else "—",
                f"{px_z:.4f}" if px_z else "—",
                "Yes" if res.get("denoise_diagnostics") else "No",
                "YES" if reg.get("clipped_due_to_failure") else "No",
                str(recoveries),
            ])

        fig, ax = plt.subplots(figsize=(14, max(4, 1 + 0.4 * len(table_data))))
        ax.axis("off")
        fig.suptitle(f"{lif_name} — Series Summary", fontsize=14,
                     fontweight="bold", y=0.98)
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_headers,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        # Header row styling
        for j in range(len(col_headers)):
            tbl[0, j].set_facecolor("#2c5f8a")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        # Alternating row shading; flag clipped rows
        for i, row in enumerate(table_data):
            is_clipped = row[5] == "YES"
            for j in range(len(col_headers)):
                cell = tbl[i + 1, j]
                if is_clipped:
                    cell.set_facecolor("#ffe0e0")
                elif i % 2 == 0:
                    cell.set_facecolor("#f0f4f8")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- Per-series pages ----
        for res in series_results:
            series = res["series"]
            log = res["transform_log"]
            diag = res.get("denoise_diagnostics")
            meta = series.metadata
            reg_meta = meta.get("registration", {})
            ch_labels = res["channel_labels"]

            zs = np.array([e["z"] for e in log])
            tx_arr = np.array([e["tx"] for e in log])
            ty_arr = np.array([e["ty"] for e in log])
            reasons = [e["reason"] for e in log]
            delta = np.zeros(len(log))
            if len(log) > 1:
                delta[1:] = np.hypot(np.diff(tx_arr), np.diff(ty_arr))

            reason_color = {
                "ok": "tab:green",
                "anchor": "tab:gray",
                "ok_after_jump_recovery": "tab:orange",
                "fatal_jump_circuit_breaker": "tab:red",
            }
            bar_colors = [reason_color.get(r, "black") for r in reasons]

            jt = reg_meta.get("jump_threshold_voxels", 5.0)

            # --- Drift plots (top half of page) ---
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(
                f"{lif_name}  |  {series.series_name}",
                fontsize=13, fontweight="bold",
            )

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.bar(zs, delta, color=bar_colors, width=0.8)
            if jt:
                ax1.axhline(jt, color="k", ls="--", lw=1,
                            label=f"jump threshold = {jt}")
                ax1.legend(fontsize=8)
            ax1.set_xlabel("Z slice")
            ax1.set_ylabel("|T(k) − T(k−1)|  (voxels)")
            ax1.set_title("Inter-slice transform change")

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(tx_arr, ty_arr, "-o", color="tab:blue", lw=1, ms=4)
            step = max(1, len(zs) // 8)
            for k, (x, y) in enumerate(zip(tx_arr, ty_arr)):
                if k % step == 0:
                    ax2.annotate(str(k), (x, y), textcoords="offset points",
                                 xytext=(4, 4), fontsize=7, color="dimgray")
            ax2.axhline(0, color="lightgray", lw=0.5)
            ax2.axvline(0, color="lightgray", lw=0.5)
            ax2.set_xlabel("tx (voxels)")
            ax2.set_ylabel("ty (voxels)")
            ax2.set_title("Drift trajectory")
            ax2.set_aspect("equal", adjustable="datalim")

            # --- Stats + params text block (bottom-left) ---
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.axis("off")

            reason_counts = Counter(reasons)
            scale = meta.get("scale") or {}
            px_xy = scale.get("x")
            px_z = scale.get("z")
            final_z = reg_meta.get("final_z_slices",
                                   meta.get("z_slices", len(log)))
            orig_z = reg_meta.get("original_z_slices", final_z)
            is_clipped = reg_meta.get("clipped_due_to_failure", False)

            disp_vals = np.hypot(tx_arr, ty_arr)
            stats_lines = [
                ("Registration parameters", ""),
                ("  Reference channel",
                 reg_meta.get("reference_channel", ch_labels[1])),
                ("  Jump threshold", f"{jt} voxels"),
                ("  Fatal threshold",
                 str(reg_meta.get("fatal_jump_threshold", "—"))),
                ("  Smoothing σ (metric only)",
                 str(reg_meta.get("smoothing_sigma_for_metric", "—"))),
                ("", ""),
                ("Transform log summary", ""),
                ("  Z slices (orig → final)",
                 f"{orig_z} → {final_z}"
                 + (" ⚠ CLIPPED" if is_clipped else "")),
                ("  Displacement min / mean / max",
                 f"{disp_vals.min():.2f} / {disp_vals.mean():.2f} / "
                 f"{disp_vals.max():.2f} voxels"),
                ("  tx range",
                 f"{tx_arr.min():.2f} → {tx_arr.max():.2f}"),
                ("  ty range",
                 f"{ty_arr.min():.2f} → {ty_arr.max():.2f}"),
                ("  Jump recoveries",
                 str(reason_counts.get("ok_after_jump_recovery", 0))),
                ("  Circuit breaker events",
                 str(reason_counts.get("fatal_jump_circuit_breaker", 0))),
                ("", ""),
                ("Physical calibration", ""),
                ("  XY pixel size",
                 f"{px_xy:.4f} µm" if px_xy else "not available"),
                ("  Z step",
                 f"{px_z:.4f} µm" if px_z else "not available"),
            ]

            y_txt = 0.97
            for k, (label, val) in enumerate(stats_lines):
                weight = "bold" if val == "" else "normal"
                color = "#cc0000" if "CLIPPED" in val else "black"
                ax3.text(0.02, y_txt, label, ha="left", va="top",
                         fontsize=9, fontweight=weight, color=color,
                         transform=ax3.transAxes)
                if val:
                    ax3.text(0.55, y_txt, val, ha="left", va="top",
                             fontsize=9, color=color,
                             transform=ax3.transAxes)
                y_txt -= 0.058

            # --- Denoising summary text (bottom-right) ---
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis("off")

            if diag:
                preprocessing = meta.get("preprocessing", {})
                den_lines = [
                    ("Denoising parameters", ""),
                    ("  Neighborhood",
                     str(preprocessing.get("denoise_detection_neighborhood",
                                           "—"))),
                    ("  Threshold (MAD)",
                     str(preprocessing.get("denoise_detection_threshold",
                                           "—"))),
                    ("  Min punctum size",
                     str(preprocessing.get("denoise_min_punctum_size", "—"))),
                    ("", ""),
                    ("Per-channel results", ""),
                ]
                for lbl in ch_labels:
                    if lbl not in diag:
                        continue
                    cd = diag[lbl]
                    den_lines += [
                        (f"  {lbl}", ""),
                        ("    Bad rows",
                         str(cd.get("total_bad_rows", 0))),
                        ("    Pixels zeroed",
                         str(cd.get("total_pixels_zeroed", 0))),
                        ("    Slices affected",
                         f"{sum(1 for s in cd['per_slice'] if s['n_bad_rows'] > 0)}"
                         f" / {len(cd['per_slice'])}"),
                        ("    Worst Z",
                         str(cd.get("worst_z", "—"))),
                    ]
            else:
                den_lines = [("Denoising", "not applied")]

            y_txt = 0.97
            for label, val in den_lines:
                weight = "bold" if val == "" else "normal"
                ax4.text(0.02, y_txt, label, ha="left", va="top",
                         fontsize=9, fontweight=weight,
                         transform=ax4.transAxes)
                if val:
                    ax4.text(0.55, y_txt, val, ha="left", va="top",
                             fontsize=9, transform=ax4.transAxes)
                y_txt -= 0.058

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # --- Denoise validation figure embedded in PDF ---
            if diag:
                best_label = max(
                    (lbl for lbl in ch_labels if lbl in diag),
                    key=lambda lbl: diag[lbl].get("total_pixels_zeroed", 0),
                )
                cd = diag[best_label]
                worst_z = cd["worst_z"]
                n_zeroed = cd["per_slice"][worst_z]["n_pixels_zeroed"]
                n_bad_rows_pdf = cd["per_slice"][worst_z]["n_bad_rows"]
                preprocessing = meta.get("preprocessing", {})
                nbhd_pdf = preprocessing.get("denoise_detection_neighborhood", "?")
                thr_pdf = preprocessing.get("denoise_detection_threshold", "?")
                mps_pdf = preprocessing.get("denoise_min_punctum_size", "?")
                fig_d, axes_d = plt.subplots(1, 3, figsize=(16, 5))
                img_f = cd["worst_raw"].astype(np.float32)
                cln_f = cd["worst_cleaned"].astype(np.float32)
                diff = cln_f - img_f
                vmin_d, vmax_d = np.percentile(img_f, 1), np.percentile(img_f, 99.5)
                dmax = max(float(np.percentile(np.abs(diff), 99)), 1.0)
                fig_d.suptitle(
                    f"Line denoising preview — {series.series_name} | "
                    f"{best_label} | Z={worst_z}\n"
                    f"NEIGHBORHOOD={nbhd_pdf}, "
                    f"THRESHOLD={thr_pdf} MAD, "
                    f"MIN_PUNCTUM_SIZE={mps_pdf}",
                    fontsize=12,
                )
                axes_d[0].imshow(img_f, cmap="gray", vmin=vmin_d, vmax=vmax_d)
                axes_d[0].set_title("Original")
                axes_d[0].axis("off")
                axes_d[1].imshow(cln_f, cmap="gray", vmin=vmin_d, vmax=vmax_d)
                axes_d[1].set_title(
                    f"Cleaned ({n_zeroed} px zeroed on {n_bad_rows_pdf} rows)"
                )
                axes_d[1].axis("off")
                axes_d[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
                axes_d[2].set_title(
                    f"Difference (cleaned - original), ±{dmax:.1f}"
                )
                axes_d[2].axis("off")
                plt.tight_layout()
                pdf.savefig(fig_d, bbox_inches="tight")
                plt.close(fig_d)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = f"{lif_name} — Drift Correction Report"
        d["Author"] = f"lif_drift_correction_pipeline v{_PIPELINE_VERSION}"
        d["Subject"] = "Confocal Z-stack drift correction QC"
        d["CreationDate"] = run_timestamp

    print(f"   [REPORT] PDF  -> {pdf_path}")
    return {
        "csv_path": csv_path,
        "pdf_path": pdf_path,
        "denoise_png_paths": denoise_png_paths,
    }

def run_pipeline_lif(
    lif_files,
    base_dir: Path,
    required_channels: int = 3,
    required_z: int = 10,
    channel_order=(0, 1, 2),
    channel_labels=("DAPI", "Reference", "Target"),
    jump_threshold_voxels: float = 5.0,
    fatal_jump_threshold: float = 10.0,
    smoothing_sigma: float = 1.0,
    denoise_lines: bool = False,
    denoise_detection_neighborhood: int = 31,
    denoise_detection_threshold: float = 4.0,
    denoise_min_punctum_size: int = 5,
    verbose: bool = True,
):
    import datetime
    base_dir = Path(base_dir)

    for lif_item in lif_files:
        lif_path = Path(lif_item)
        if not lif_path.is_absolute():
            lif_path = base_dir / lif_path

        results = inspect_single_lif(lif_path, verbose=verbose)
        if not results:
            continue

        lif_name = lif_path.stem
        lif_dir = base_dir / sanitize_name(lif_name)
        run_ts = datetime.datetime.now().isoformat(timespec="seconds")
        series_results: list[dict] = []

        for s in results:
            z = s["z_slices"] or 0
            if s["channels"] != required_channels or z < required_z:
                continue
            series = load_series_from_lif(
                lif_path=lif_path,
                series_info=s,
                channel_order=channel_order,
                channel_labels=channel_labels,
            )
            res = process_series(
                series=series,
                base_dir=base_dir,
                channel_labels=channel_labels,
                jump_threshold_voxels=jump_threshold_voxels,
                fatal_jump_threshold=fatal_jump_threshold,
                smoothing_sigma=smoothing_sigma,
                denoise_lines=denoise_lines,
                denoise_detection_neighborhood=denoise_detection_neighborhood,
                denoise_detection_threshold=denoise_detection_threshold,
                denoise_min_punctum_size=denoise_min_punctum_size,
                verbose=verbose,
            )
            series_results.append(res)

        if not series_results:
            print(
                f"[INFO] No series matched in {lif_path.name} "
                f"(required_channels={required_channels}, "
                f"required_z={required_z})."
            )
            continue

        print(f"\n[INFO] Processed {len(series_results)} series from "
              f"{lif_path.name}. Generating report...")
        generate_lif_report(
            lif_name=lif_name,
            lif_dir=lif_dir,
            series_results=series_results,
            run_timestamp=run_ts,
            source_path=str(lif_path),
        )

    print("\n[INFO] run_pipeline_lif complete.")


def run_pipeline_tiff(
    base_dir: Path,
    channel_labels=("DAPI", "Reference", "Target"),
    required_z: Optional[int] = 10,
    jump_threshold_voxels: float = 5.0,
    fatal_jump_threshold: float = 10.0,
    smoothing_sigma: float = 1.0,
    denoise_lines: bool = False,
    denoise_detection_neighborhood: int = 31,
    denoise_detection_threshold: float = 4.0,
    denoise_min_punctum_size: int = 5,
    verbose: bool = True,
):
    import datetime
    base_dir = Path(base_dir)
    series_dirs = discover_tiff_series(base_dir, channel_labels=channel_labels)
    if not series_dirs:
        print(f"[INFO] No <lif>/<series>/stacks/ folders found under {base_dir}")
        return

    run_ts = datetime.datetime.now().isoformat(timespec="seconds")
    # Group results by lif_name so one report is generated per LIF folder.
    from collections import defaultdict
    results_by_lif: dict[str, list[dict]] = defaultdict(list)

    for sdir in series_dirs:
        series = load_series_from_tiff_dir(sdir, channel_labels=channel_labels)
        if series is None:
            continue
        z = series.stacks[channel_labels[0]].shape[0]
        if required_z is not None and z < required_z:
            if verbose:
                print(f"[SKIP] {sdir} (z={z}, required={required_z})")
            continue
        res = process_series(
            series=series,
            base_dir=base_dir,
            channel_labels=channel_labels,
            jump_threshold_voxels=jump_threshold_voxels,
            fatal_jump_threshold=fatal_jump_threshold,
            smoothing_sigma=smoothing_sigma,
            denoise_lines=denoise_lines,
            denoise_detection_neighborhood=denoise_detection_neighborhood,
            denoise_detection_threshold=denoise_detection_threshold,
            denoise_min_punctum_size=denoise_min_punctum_size,
            verbose=verbose,
        )
        results_by_lif[series.lif_name].append(res)

    if not results_by_lif:
        print("[INFO] No TIFF series matched.")
        return

    for lif_name, series_results in results_by_lif.items():
        lif_dir = base_dir / sanitize_name(lif_name)
        print(f"\n[INFO] Processed {len(series_results)} series under "
              f"'{lif_name}'. Generating report...")
        generate_lif_report(
            lif_name=lif_name,
            lif_dir=lif_dir,
            series_results=series_results,
            run_timestamp=run_ts,
            source_path=None,
        )

    total = sum(len(v) for v in results_by_lif.values())
    print(f"\n[INFO] run_pipeline_tiff complete. {total} series processed.")


# =============================================================================
# Interactive CLI
# =============================================================================
def _prompt_input_mode() -> str:
    print("\nSelect input mode:")
    print("  [A] Process .lif files directly")
    print("  [B] Process pre-extracted TIFF stacks")
    while True:
        choice = input("Enter A or B: ").strip().lower()
        if choice in ("a", "lif"):
            return "lif"
        if choice in ("b", "tif", "tiff"):
            return "tiff"
        print("Please enter 'A' or 'B'.")


def _prompt_denoise_lines() -> bool:
    """Ask whether to apply scanner-line denoising as a preprocessing step."""
    print("\nApply scanner-line noise removal as preprocessing?")
    print("  Removes horizontal line artifacts from sparse fluorescence data.")
    print("  Safe to enable: only modifies pixels in detected line-corrupted")
    print("  rows that are isolated (not part of larger biological blobs).")
    while True:
        choice = input("Enable line denoising? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no", ""):
            return False
        print("Please enter 'y' or 'n'.")


def _gui_select_lif_files() -> list[Path]:
    """Open a file dialog to select one or more .lif files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Select .lif file(s)",
        filetypes=[("Leica Image Format", "*.lif"), ("All files", "*.*")],
    )
    root.destroy()
    if not paths:
        print("[ERROR] No .lif file selected.")
        sys.exit(1)
    selected = [Path(p) for p in paths]
    print(f"[INFO] Selected {len(selected)} .lif file(s):")
    for p in selected:
        print(f"  {p}")
    return selected


def _gui_select_tiff_files() -> list[Path]:
    """Open a file dialog to select one or more TIFF stack files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Select TIFF stack file(s)  [e.g. DAPI_stack.tif, Reference_stack.tif ...]",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    if not paths:
        print("[ERROR] No TIFF file selected.")
        sys.exit(1)
    selected = [Path(p) for p in paths]
    print(f"[INFO] Selected {len(selected)} TIFF file(s):")
    for p in selected:
        print(f"  {p}")
    return selected


def _derive_series_and_base_from_tiffs(
    tif_paths: list[Path],
) -> tuple[list[Path], Path]:
    """Group selected tif files by series dir and derive a common base_dir.

    Expected layout: <base>/<lif_name>/<series_name>/stacks/<channel>_stack.tif
    If a file is not inside a folder named 'stacks', its parent is treated as
    the series dir directly.
    Returns (series_dirs, base_dir).
    """
    series_dirs: dict[Path, None] = {}  # ordered set via insertion-order dict
    for p in tif_paths:
        if p.parent.name.lower() == "stacks":
            series_dirs[p.parent.parent] = None
        else:
            series_dirs[p.parent] = None

    series_list = list(series_dirs.keys())

    # base_dir is two levels above each series dir (series -> lif -> base).
    grandparents = {s.parent.parent for s in series_list}
    if len(grandparents) == 1:
        base_dir = grandparents.pop()
    else:
        all_parts = [list(s.parts) for s in series_list]
        common = all_parts[0]
        for parts in all_parts[1:]:
            common = [a for a, b in zip(common, parts) if a == b]
        base_dir = Path(*common) if common else series_list[0].parent.parent

    return series_list, base_dir


def main():
    print("=" * 70)
    print(" LIF / TIFF volumetric drift-correction pipeline")
    print("=" * 70)

    mode = _prompt_input_mode()
    denoise_lines = _prompt_denoise_lines()

    if mode == "lif":
        lif_files = _gui_select_lif_files()
        base_dir = lif_files[0].parent
        run_pipeline_lif(
            lif_files=lif_files,
            base_dir=base_dir,
            required_channels=3,
            required_z=20,
            channel_order=(0, 1, 2),
            channel_labels=("DAPI", "Reference", "Target"),
            jump_threshold_voxels=5.0,
            fatal_jump_threshold=10.0,
            smoothing_sigma=1.0,
            denoise_lines=denoise_lines,
            denoise_detection_neighborhood=31,
            denoise_detection_threshold=4.0,
            denoise_min_punctum_size=5,
            verbose=True,
        )
    else:
        import datetime
        from collections import defaultdict
        tif_files = _gui_select_tiff_files()
        series_dirs, base_dir = _derive_series_and_base_from_tiffs(tif_files)
        channel_labels = ("DAPI", "Reference", "Target")
        required_z = 20
        run_ts = datetime.datetime.now().isoformat(timespec="seconds")
        results_by_lif: dict = defaultdict(list)

        # Process exactly the series the user picked via the file dialog.
        # (We don't rediscover via run_pipeline_tiff because that would walk
        # the whole base_dir, which may include series the user didn't select.)
        for sdir in series_dirs:
            series = load_series_from_tiff_dir(sdir, channel_labels=channel_labels)
            if series is None:
                print(f"[SKIP] Could not load series from {sdir}")
                continue
            z = series.stacks[channel_labels[0]].shape[0]
            if z < required_z:
                print(f"[SKIP] {sdir.name} (z={z}, required={required_z})")
                continue
            res = process_series(
                series=series,
                base_dir=base_dir,
                channel_labels=channel_labels,
                jump_threshold_voxels=5.0,
                fatal_jump_threshold=10.0,
                smoothing_sigma=1.0,
                denoise_lines=denoise_lines,
                denoise_detection_neighborhood=31,
                denoise_detection_threshold=4.0,
                denoise_min_punctum_size=5,
                verbose=True,
            )
            results_by_lif[series.lif_name].append(res)

        if not results_by_lif:
            print("[INFO] No TIFF series matched.")
        else:
            for lif_name, series_results in results_by_lif.items():
                lif_dir = base_dir / sanitize_name(lif_name)
                print(f"\n[INFO] Generating report for '{lif_name}'...")
                generate_lif_report(
                    lif_name=lif_name,
                    lif_dir=lif_dir,
                    series_results=series_results,
                    run_timestamp=run_ts,
                    source_path=None,
                )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
