# FiX-ITS
This repository contains the complete Python-based processing and analysis pipeline characterization of the sub-synaptic distribution of synaptic proteins and adhesion molecules at nanoscale resolution, including pre- and postsynaptic scaffolding proteins (e.g. PSD-95, Homer), active zone components, and trans-synaptic adhesion complexes. Contents include: a complete pre-processing pipeline for confocal Z-stack microscope images including drift correction and scanner noise subtraction; quantitative analysis of expansion ratio and anisotropix expansion error; quantitative analysis for immunostaining quality assessment. For pre-processing, QC outputs include: 
- Cover page: pipeline version, run timestamp, source file, series count, clipping events
- Summary table: one row per series with Z counts, pixel sizes, denoising status, circuit-breaker flags
- Per-series pages: inter-slice jump bar chart (colour-coded by registration outcome), 2D drift trajectory, registration parameter block, denoising diagnostics
- Worst-slice denoising validation panel (if denoising was run): three-panel figure matching the notebook block 7 preview — Original | Cleaned | Difference
