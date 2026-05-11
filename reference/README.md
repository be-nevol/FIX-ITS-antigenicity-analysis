# FiX-ITS Reference Code

This folder keeps the reference implementation for the LIF/TIFF volumetric
drift-correction workflow used by the analysis notebooks.

## Files

- `lif_pipeline.py` - canonical reference implementation. Use this file for
  imports, command-line runs, and future edits.
- `lif_processing_notebook.ipynb` - notebook driver for interactive inspection,
  parameter tuning, single-series tests, and batch execution. It now imports
  from `lif_pipeline`.
- `lif_pipeline_final.py` - older full snapshot retained for comparison.
- `lif_pipeline_json.py` - older JSON/per-channel TIFF snapshot retained for
  comparison.

## Main Capabilities

- Reads Leica `.lif` files or pre-extracted TIFF stacks.
- Performs sequential slice-by-slice 2D translation drift correction with
  SimpleITK.
- Optionally removes horizontal scanner-line artifacts before registration.
- Writes Fiji/ImageJ-compatible multichannel hyperstack TIFFs with calibration,
  LUTs, and registration provenance.
- Generates side-by-side GIFs, registered MIP PNGs, `metadata.json`, per-LIF
  CSV summaries, denoising validation PNGs, and PDF QC reports.

## Output Layout

```text
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
  reports/
    <lif_name>_series_summary.csv
    <lif_name>_drift_correction_report.pdf
```

## Notes

The canonical file supersedes the two older snapshots. Keeping the snapshots in
place avoids breaking any external references while making the intended import
target explicit.
