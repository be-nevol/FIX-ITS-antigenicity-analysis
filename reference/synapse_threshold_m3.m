function [mask, th] = synapse_threshold_m3(A)
% SYNAPSE_THRESHOLD_M3  Median-subtract + cross-kernel + spherical convolution
%                       thresholding for synaptic object segmentation.
%
% This function implements Method 3 from threshold_comparison_test.m as a
% standalone, parameter-free thresholding routine for use in the Synapse
% Analysis pipeline.  All kernel parameters are fixed internally so that
% the threshold is derived entirely from the data — no user-tunable
% multipliers or percentile cutoffs are required.
%
% Algorithm
% ---------
%   Step 1a  Subtract the volume median to remove the DC background offset.
%            Negative residuals are clipped to zero.
%
%   Step 1b  Apply a 2-D cross kernel [0,1,0; 1,1,1; 0,1,0]/5 slice-by-slice
%            in XY.  This mild lateral smoothing spreads genuine signal without
%            broadening background, and suppresses single-voxel noise spikes
%            before the gate step.  imfilter is used with 'replicate' boundary
%            padding to avoid edge artefacts.
%
%   Step 2   Coarse gate: retain only voxels above max/10.  The mean intensity
%            of surviving voxels becomes the threshold th.  This automatically
%            scales to object brightness without any user input.
%
%   Step 3   Subtract th from the cross-kernel smoothed volume; convolve with
%            a 5×5×5 normalised spherical kernel (radius ≤ 2.5 voxels).
%            Voxels where the convolution result exceeds zero form the
%            foreground mask.  The spherical convolution enforces 3-D spatial
%            continuity — isolated voxels that survive step 2 but lack
%            volumetric support are suppressed.
%
% Usage
% -----
%   [mask, th] = synapse_threshold_m3(A)
%
% Inputs
%   A    – 3-D double array  [H × W × Z]
%          Pass the voxel-expanded, pre-cast-to-double sub-volume
%          (i.e. the exim1 / exim2 arrays from Synapse_Analysis).
%          Gaussian pre-filtering is NOT applied here — it is the caller's
%          responsibility if desired (see note below).
%
% Outputs
%   mask – logical 3-D foreground mask  [H × W × Z]
%   th   – scalar threshold value derived from the data (useful for
%          logging / display in the segmentation figure title)
%
% Note on Gaussian pre-filtering
% --------------------------------
%   The original Synapse_Analysis pipeline applied imgaussfilt3 with sigma
%   0.5 (ch1) or 1.0 (ch2) before thresholding.  Method 3 is designed to
%   operate on the raw expanded volume — the cross-kernel in step 1b provides
%   its own mild smoothing.  However, if pre-filtering is desired for
%   consistency with other downstream steps, filter A before calling this
%   function.  The threshold and mask are both returned so the caller can
%   choose how to use them.
%
% Kernel pre-computation
% ----------------------
%   Kernels are computed once at function load time as persistent variables
%   so they are not rebuilt on every call when processing multiple objects
%   in a loop.

% ── Persistent kernels (built once per MATLAB session) ───────────────────
persistent cross_k2d k7

if isempty(cross_k2d)
    % 2-D cross kernel for XY slice-by-slice smoothing
    cross_k2d = [0 1 0; 1 1 1; 0 1 0] / 5;

    % 5×5×5 spherical kernel: include voxels within radius 2.5
    [kx, ky, kz] = ndgrid(-2:2, -2:2, -2:2);
    k7_raw = double(sqrt(kx.^2 + ky.^2 + kz.^2) <= 2.5);
    k7     = k7_raw / sum(k7_raw(:));   % normalise to unit sum
end

% ── Step 1a: median subtract, clip to zero ───────────────────────────────
A1 = A - median(A(:));
A1 = max(A1, 0);

% ── Step 1b: 2-D cross-kernel smooth, slice by slice ────────────────────
A1s = zeros(size(A1));
for z = 1:size(A1, 3)
    A1s(:,:,z) = imfilter(A1(:,:,z), cross_k2d, 'replicate', 'same');
end

% ── Step 2: coarse gate → mean-of-survivors threshold ───────────────────
coarse_gate   = max(A1s(:)) / 10 ;
bright_voxels = A1s(A1s > coarse_gate);

if isempty(bright_voxels)
    % No signal above the coarse gate — return empty mask
    mask = false(size(A));
    th   = coarse_gate;
    return;
end

th = mean(bright_voxels);

% ── Step 3: subtract th → spherical convolution → threshold at zero ─────
A2   = A1s - th;
A2   = convn(A2, k7, 'same');
mask = A2 > 0;

end
