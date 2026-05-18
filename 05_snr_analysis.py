#!/usr/bin/env python
# coding: utf-8

# # SNR Analysis
# 
# 03 rigid registration으로 생성된 registered TIFF stack에서 Reference/Target channel의 전체 z-stack SNR을 계산합니다.
# 
# - DAPI channel은 제외합니다.
# - 각 z slice마다 MAD(mean absolute deviation) thresholding으로 signal/background binary mask를 만듭니다.
# - SNR은 `mean(signal_pixels) / std(background_pixels)`로 계산합니다.
# - 하나의 LIF 폴더에 포함된 series별 SNR figure를 모아 PDF로 저장합니다.
# 

# ## 0. Setup
# 

# In[ ]:


from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "analysis_results"
PDF_DIR = RESULTS_DIR / "snr_pdf_by_lif"
MASK_PNG_DIR = RESULTS_DIR / "snr_binary_mask_png"
RESULTS_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
MASK_PNG_DIR.mkdir(exist_ok=True)

CHANNELS = ("Reference", "Target")
MAD_MULTIPLIER = 3
MAD_MULTIPLIER_SWEEP = list(range(1, 11))
NOISE_METHOD = "std"
IMPLEMENTED = False

print({
    "project_dir": str(PROJECT_DIR),
    "data_dir": str(DATA_DIR),
    "results_dir": str(RESULTS_DIR),
    "pdf_dir": str(PDF_DIR),
    "mask_png_dir": str(MASK_PNG_DIR),
    "channels": CHANNELS,
    "mad_multiplier": MAD_MULTIPLIER,
    "mad_multiplier_sweep": MAD_MULTIPLIER_SWEEP,
})


# ## 1. Input / Output Contract
# 

# In[ ]:


INPUTS = {
    "registered_channel_stacks": "data/<lif_name>/<series_name>/registered_stacks/{Reference,Target}_stack_registered.tif",
    "registered_multichannel_stack": "data/<lif_name>/<series_name>/registered_stacks/<series_name>_stack_registered.tif",
}

OUTPUTS = {
    "all_z_snr_csv": "analysis_results/snr_all_z_registered_stacks_mad.csv",
    "lif_pdf_reports": "analysis_results/snr_pdf_by_lif/<lif_name>_snr_all_z.pdf",
    "middle_z_binary_mask_png": "analysis_results/snr_binary_mask_png/<lif>__<series>__<channel>__z<idx>__binary_mask.png",
    "middle_z_raw_mask_qc_png": "analysis_results/snr_binary_mask_png/<lif>__<series>__<channel>__z<idx>__raw_mask_qc.png",
}

print("Inputs:")
for key, value in INPUTS.items():
    print(f"  {key}: {value}")
print("Outputs:")
for key, value in OUTPUTS.items():
    print(f"  {key}: {value}")


# ## 2. Algorithm TODO
# 

# In[ ]:


ALGORITHM = [
    "registered_stacks 폴더가 있는 모든 series 탐색",
    "Reference/Target channel TIFF만 로드하고 DAPI는 제외",
    "각 z slice에서 non-zero pixel 기준 MAD threshold 계산",
    "binary mask: background=0, signal=1",
    "signal_pixels = z_slice[mask == 1], background_pixels = z_slice[mask == 0]",
    "SNR = mean(signal_pixels) / std(background_pixels)",
    "middle-z raw/binary mask QC PNG 저장",
    "전체 결과 CSV 저장 및 lif_name 단위 multi-page PDF 저장",
    "PDF에서 MAD multiplier 1~10 thresholding 결과 비교",
]

for i, item in enumerate(ALGORITHM, 1):
    print(f"{i}. {item}")


# ## 3. Discovery Helpers
# 

# In[ ]:


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unnamed"


def discover_registered_series(data_dir: Path = DATA_DIR, channels: tuple[str, ...] = CHANNELS) -> pd.DataFrame:
    rows = []
    if not data_dir.exists():
        return pd.DataFrame(rows)

    for registered_dir in sorted(data_dir.glob("*/*/registered_stacks")):
        series_dir = registered_dir.parent
        channel_paths = {channel: registered_dir / f"{channel}_stack_registered.tif" for channel in channels}
        available = [channel for channel, path in channel_paths.items() if path.exists()]
        if not available:
            continue

        rows.append({
            "lif_name": series_dir.parent.name,
            "series_name": series_dir.name,
            "series_dir": series_dir,
            "registered_stacks_dir": registered_dir,
            "available_channels": available,
            "n_available_channels": len(available),
        })

    return pd.DataFrame(rows)


series_df = discover_registered_series()
print(f"Discovered {len(series_df)} registered series")
display(series_df[["lif_name", "series_name", "registered_stacks_dir", "available_channels"]])


# ## 4. Implementation Area

# In[ ]:


def create_mad_signal_mask(image_2d: np.ndarray, mad_multiplier: float = MAD_MULTIPLIER):
    """
    MAD thresholding for a single 2D z slice.
    Threshold = mean(non-zero pixels) + mad_multiplier * mean(abs(non-zero pixels - mean)).
    Returns a uint8 binary mask where background=0 and signal=1.
    """
    non_zero = image_2d[image_2d > 0].astype(np.float32)
    if non_zero.size == 0:
        threshold = 0.0
    else:
        center = float(np.mean(non_zero))
        mean_absolute_deviation = float(np.mean(np.abs(non_zero - center)))
        threshold = center + mad_multiplier * mean_absolute_deviation

    mask = (image_2d > threshold).astype(np.uint8)
    return mask, threshold


def calculate_slice_snr(z_slice: np.ndarray, mask: np.ndarray) -> dict:
    signal_pixels = z_slice[mask == 1]
    background_pixels = z_slice[mask == 0]

    mean_signal = float(np.mean(signal_pixels)) if signal_pixels.size else np.nan
    std_signal = float(np.std(signal_pixels)) if signal_pixels.size else np.nan
    mean_background = float(np.mean(background_pixels)) if background_pixels.size else np.nan
    noise = float(np.std(background_pixels)) if background_pixels.size else np.nan

    if signal_pixels.size == 0 or not np.isfinite(noise):
        snr = np.nan
    elif noise == 0:
        snr = np.inf
    else:
        snr = mean_signal / noise

    return {
        "mean_signal": mean_signal,
        "std_signal": std_signal,
        "mean_background": mean_background,
        "noise": noise,
        "snr": snr,
        "n_signal_pixels": int(signal_pixels.size),
        "n_background_pixels": int(background_pixels.size),
    }


def calculate_stack_snr(stack_path: Path, lif_name: str, series_name: str, channel: str) -> list[dict]:
    stack = tifffile.imread(stack_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack for {stack_path}, got shape {stack.shape}")

    rows = []
    n_z, n_y, n_x = stack.shape
    for z_idx in range(n_z):
        z_slice = stack[z_idx]
        mask, threshold = create_mad_signal_mask(z_slice)
        metrics = calculate_slice_snr(z_slice, mask)
        rows.append({
            "lif_name": lif_name,
            "series_name": series_name,
            "channel": channel,
            "stack_file": stack_path.name,
            "z_slice": z_idx,
            "n_z": n_z,
            "height": n_y,
            "width": n_x,
            "threshold_method": "mad_mean_abs_dev",
            "mad_multiplier": MAD_MULTIPLIER,
            "threshold": float(threshold),
            **metrics,
        })
    return rows


def resolve_stack_and_middle_z(tiff_df: pd.DataFrame, lif_name: str, series_name: str):
    tiff_df = tiff_df.sort_values("z_slice")
    n_z = int(tiff_df["n_z"].iloc[0])
    z_idx = max(n_z // 2 - 1, 0)
    metric_row = tiff_df[tiff_df["z_slice"] == z_idx].iloc[0]

    stack_path = DATA_DIR / lif_name / series_name / "registered_stacks" / metric_row["stack_file"]
    stack = tifffile.imread(stack_path)
    z_slice = stack[z_idx]
    return z_slice, z_idx, n_z, metric_row


# In[ ]:


IMPLEMENTED = True

results = []
errors = []

for _, row in series_df.iterrows():
    lif_name = row["lif_name"]
    series_name = row["series_name"]
    registered_dir = Path(row["registered_stacks_dir"])
    print(f"Processing {lif_name}/{series_name}")

    for channel in CHANNELS:
        stack_path = registered_dir / f"{channel}_stack_registered.tif"
        if not stack_path.exists():
            print(f"  missing {channel}: {stack_path.name}")
            continue

        try:
            channel_rows = calculate_stack_snr(stack_path, lif_name, series_name, channel)
            results.extend(channel_rows)
            print(f"  {channel}: {len(channel_rows)} z slices")
        except Exception as exc:
            errors.append({
                "lif_name": lif_name,
                "series_name": series_name,
                "channel": channel,
                "stack_path": str(stack_path),
                "error": repr(exc),
            })
            print(f"  ERROR {channel}: {exc}")

results_df = pd.DataFrame(results)
errors_df = pd.DataFrame(errors)

print("")
print(f"Calculated {len(results_df)} SNR rows")
if not errors_df.empty:
    print(f"Encountered {len(errors_df)} errors")
    display(errors_df)

display(results_df.head())


# In[ ]:


def plot_lif_summary(lif_df: pd.DataFrame, lif_name: str):
    summary = (
        lif_df.groupby(["series_name", "channel"], as_index=False)
        .agg(
            mean_snr=("snr", "mean"),
            median_snr=("snr", "median"),
            min_snr=("snr", "min"),
            max_snr=("snr", "max"),
            n_z=("z_slice", "count"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle(f"SNR summary: {lif_name}", fontsize=14)

    series_names = sorted(summary["series_name"].unique())
    x = np.arange(len(series_names))
    width = 0.36
    offsets = np.linspace(-width / 2, width / 2, len(CHANNELS)) if len(CHANNELS) > 1 else [0]

    for offset, channel in zip(offsets, CHANNELS):
        ch_df = summary[summary["channel"] == channel].set_index("series_name").reindex(series_names)
        axes[0].bar(x + offset, ch_df["mean_snr"], width=width, label=channel)
    axes[0].set_ylabel("Mean SNR across z")
    axes[0].set_xlabel("Series")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(series_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    text_lines = [
        f"Series count: {lif_df['series_name'].nunique()}",
        f"Rows: {len(lif_df)}",
        f"Channels: {', '.join(sorted(lif_df['channel'].unique()))}",
        f"Threshold: mean(non-zero) + {MAD_MULTIPLIER} * mean absolute deviation",
        "SNR: mean(signal pixels) / std(background pixels)",
        "First page chart: mean SNR grouped by series and channel",
    ]
    axes[1].axis("off")
    axes[1].text(0.01, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=11)

    return fig


def plot_series_snr(series_df: pd.DataFrame, lif_name: str, series_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle(f"{lif_name} / {series_name}", fontsize=13)

    for channel in CHANNELS:
        ch_df = series_df[series_df["channel"] == channel].sort_values("z_slice")
        if ch_df.empty:
            continue
        axes[0, 0].plot(ch_df["z_slice"], ch_df["snr"], marker="o", markersize=2.5, linewidth=1, label=channel)
        axes[0, 1].plot(ch_df["z_slice"], ch_df["threshold"], marker="o", markersize=2.5, linewidth=1, label=channel)
        axes[1, 0].plot(ch_df["z_slice"], ch_df["n_signal_pixels"], marker="o", markersize=2.5, linewidth=1, label=channel)
        axes[1, 1].plot(ch_df["z_slice"], ch_df["noise"], marker="o", markersize=2.5, linewidth=1, label=channel)

    axes[0, 0].set_title("SNR by z slice")
    axes[0, 0].set_ylabel("SNR")
    axes[0, 1].set_title("MAD threshold by z slice")
    axes[0, 1].set_ylabel("Threshold")
    axes[1, 0].set_title("Signal pixels by z slice")
    axes[1, 0].set_ylabel("Pixel count")
    axes[1, 1].set_title("Background noise by z slice")
    axes[1, 1].set_ylabel("std(background)")

    for ax in axes.ravel():
        ax.set_xlabel("z slice")
        ax.grid(alpha=0.3)
        ax.legend()

    return fig


def get_display_limits(z_slice: np.ndarray):
    positive = z_slice[z_slice > 0]
    if positive.size:
        return np.percentile(positive, [1, 99.5])
    return None, None


def save_middle_z_mask_pngs(
    tiff_df: pd.DataFrame,
    lif_name: str,
    series_name: str,
    channel: str,
    png_dir: Path = MASK_PNG_DIR,
) -> dict[str, Path]:
    z_slice, z_idx, _, _ = resolve_stack_and_middle_z(tiff_df, lif_name, series_name)
    mask, threshold = create_mad_signal_mask(z_slice)
    vmin, vmax = get_display_limits(z_slice)

    stem = "__".join([
        safe_filename(lif_name),
        safe_filename(series_name),
        safe_filename(channel),
        f"z{z_idx:03d}",
    ])
    raw_path = png_dir / f"{stem}__raw.png"
    mask_path = png_dir / f"{stem}__binary_mask.png"
    qc_path = png_dir / f"{stem}__raw_mask_qc.png"

    plt.imsave(raw_path, z_slice, cmap="gray", vmin=vmin, vmax=vmax)
    plt.imsave(mask_path, mask * 255, cmap="gray", vmin=0, vmax=255)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle(
        f"{lif_name} / {series_name} / {channel} / z{z_idx} / MAD x{MAD_MULTIPLIER} / threshold {threshold:.4f}",
        fontsize=9,
    )
    axes[0].imshow(z_slice, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Raw intensity")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest", resample=False)
    axes[1].set_title("Binary mask: 0/1")
    axes[1].axis("off")
    fig.savefig(qc_path, dpi=300)
    plt.close(fig)

    unique_values = np.unique(mask)
    if not np.array_equal(unique_values, np.array([0], dtype=np.uint8)) and not np.array_equal(
        unique_values, np.array([1], dtype=np.uint8)
    ) and not np.array_equal(unique_values, np.array([0, 1], dtype=np.uint8)):
        raise ValueError(f"Mask is not binary for {lif_name}/{series_name}/{channel}/z{z_idx}: {unique_values}")

    return {
        "raw": raw_path,
        "binary_mask": mask_path,
        "raw_mask_qc": qc_path,
    }


def plot_middle_z_mask_page(tiff_df: pd.DataFrame, lif_name: str, series_name: str, channel: str):
    z_slice, z_idx, n_z, metric_row = resolve_stack_and_middle_z(tiff_df, lif_name, series_name)
    mask, _ = create_mad_signal_mask(z_slice)
    vmin, vmax = get_display_limits(z_slice)

    fig, axes = plt.subplots(1, 3, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle(f"Middle z mask QC: {lif_name} / {series_name} / {channel}", fontsize=12)

    axes[0].imshow(z_slice, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"Raw intensity\nZ{z_idx} of {n_z}")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest", resample=False)
    axes[1].set_title("Binary mask\nbackground=0, signal=1")
    axes[1].axis("off")

    text_lines = [
        f"TIFF: {metric_row['stack_file']}",
        f"Channel: {channel}",
        f"Full z stack number: {n_z}",
        f"Selected z index: {z_idx}",
        f"Threshold method: {metric_row['threshold_method']}",
        f"Threshold: {float(metric_row['threshold']):.4f}",
        f"Background count: {int(metric_row['n_background_pixels'])}",
        f"Signal count: {int(metric_row['n_signal_pixels'])}",
        f"Background noise std: {float(metric_row['noise']):.4f}",
        f"SNR: {float(metric_row['snr']):.4f}",
    ]
    axes[2].axis("off")
    axes[2].text(0.0, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=10)

    return fig


def plot_multiplier_sweep_page(
    tiff_df: pd.DataFrame,
    lif_name: str,
    series_name: str,
    channel: str,
    multipliers: list[int] = MAD_MULTIPLIER_SWEEP,
):
    z_slice, z_idx, n_z, _ = resolve_stack_and_middle_z(tiff_df, lif_name, series_name)

    fig, axes = plt.subplots(2, 5, figsize=(11, 8.5), constrained_layout=True)
    fig.suptitle(
        f"MAD multiplier threshold sweep: {lif_name} / {series_name} / {channel} / Z{z_idx} of {n_z}",
        fontsize=11,
    )

    for ax, multiplier in zip(axes.ravel(), multipliers):
        mask, threshold = create_mad_signal_mask(z_slice, mad_multiplier=multiplier)
        signal_count = int(mask.sum())
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest", resample=False)
        ax.set_title(f"x{multiplier}: th={threshold:.2f}\nsignal={signal_count}", fontsize=8)
        ax.axis("off")

    return fig


def save_lif_pdf_reports(results_df: pd.DataFrame, pdf_dir: Path = PDF_DIR) -> list[Path]:
    pdf_paths = []
    if results_df.empty:
        return pdf_paths

    for lif_name, lif_df in results_df.groupby("lif_name", sort=True):
        pdf_path = pdf_dir / f"{safe_filename(lif_name)}_snr_all_z.pdf"
        with PdfPages(pdf_path) as pdf:
            fig = plot_lif_summary(lif_df, lif_name)
            pdf.savefig(fig)
            plt.close(fig)

            for series_name, one_series_df in lif_df.groupby("series_name", sort=True):
                fig = plot_series_snr(one_series_df, lif_name, series_name)
                pdf.savefig(fig)
                plt.close(fig)

                for channel in CHANNELS:
                    one_tiff_df = one_series_df[one_series_df["channel"] == channel]
                    if one_tiff_df.empty:
                        continue
                    png_paths = save_middle_z_mask_pngs(one_tiff_df, lif_name, series_name, channel)
                    print(f"Saved mask PNG: {png_paths['binary_mask']}")
                    print(f"Saved raw/mask QC PNG: {png_paths['raw_mask_qc']}")

                    fig = plot_middle_z_mask_page(one_tiff_df, lif_name, series_name, channel)
                    pdf.savefig(fig)
                    plt.close(fig)

                    fig = plot_multiplier_sweep_page(one_tiff_df, lif_name, series_name, channel)
                    pdf.savefig(fig)
                    plt.close(fig)

        pdf_paths.append(pdf_path)
        print(f"Saved PDF: {pdf_path}")

    return pdf_paths


if not results_df.empty:
    csv_path = RESULTS_DIR / "snr_all_z_registered_stacks_mad.csv"
    results_df.to_csv(csv_path, index=False)
    pdf_paths = save_lif_pdf_reports(results_df)

    print("")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved {len(pdf_paths)} PDF report(s)")

    summary_df = (
        results_df.groupby(["lif_name", "series_name", "channel"], as_index=False)
        .agg(mean_snr=("snr", "mean"), median_snr=("snr", "median"), n_z=("z_slice", "count"))
    )
    display(summary_df)
else:
    csv_path = None
    pdf_paths = []
    print("No results to save.")


# ## 5. Outputs

# In[ ]:


EXPECTED_OUTPUTS = [
    RESULTS_DIR / "snr_all_z_registered_stacks_mad.csv",
    PDF_DIR / "<lif_name>_snr_all_z.pdf (summary + z plots + middle-z mask QC + MAD multiplier sweep pages)",
    MASK_PNG_DIR / "<lif>__<series>__<channel>__z<idx>__raw.png",
    MASK_PNG_DIR / "<lif>__<series>__<channel>__z<idx>__binary_mask.png",
    MASK_PNG_DIR / "<lif>__<series>__<channel>__z<idx>__raw_mask_qc.png",
]

for output in EXPECTED_OUTPUTS:
    print(output)
