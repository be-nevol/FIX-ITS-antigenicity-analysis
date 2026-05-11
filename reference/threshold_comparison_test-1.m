%% threshold_comparison_test.m
%
% Interactive test script to compare segmentation methods on manually
% selected synaptic objects.  Mirrors the object-selection workflow from
% Synapse_Analysis_v3_xcorr.m so results are directly comparable.
%
% Methods compared (displayed side-by-side per object):
%   1. Original  – multithresh (Otsu) x multiplier  (current pipeline)
%   2. Adaptive  – Gaussian-residual local threshold (proposed)
%   3. Percentile– foreground defined by top-N% of voxels
%
% Output: one figure per selected object, saved to <pathName>/thresh_test/
%
% Tunable parameters are grouped in the "PARAMETERS" section below.

%% ── FILE SELECTION ───────────────────────────────────────────────────────
[fileNames, pathName] = uigetfile( ...
    {'*.tif','TIFF Files';'*.*','All Files'}, ...
    'Select image(s) for threshold test', 'MultiSelect', 'on');
if isequal(fileNames, 0); disp('Cancelled.'); return; end
if ~iscell(fileNames); fileNames = {fileNames}; end

out_dir = fullfile(pathName, 'thresh_test');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% ── CHANNEL SETUP ────────────────────────────────────────────────────────
num_ch_str = inputdlg({'Number of channels in image:'}, ...
    'Channel count', [1 50], {'2'});
num_ch = str2double(num_ch_str{1});

channel_list   = arrayfun(@(x) sprintf('Channel %d', x), 1:num_ch, ...
    'UniformOutput', false);
[sel_idx, ok]  = listdlg('ListString', channel_list, ...
    'SelectionMode', 'multiple', ...
    'PromptString',  'Select 2 channels to test:', ...
    'ListSize',      [300 150], 'Name', 'Channel Selection');
if ~ok; disp('Cancelled.'); return; end

name_dlg   = inputdlg( ...
    {sprintf('Name for Channel %d:', sel_idx(1)), ...
     sprintf('Name for Channel %d:', sel_idx(2))}, ...
    'Channel Names', [1 50], ...
    {sprintf('Ch%d', sel_idx(1)), sprintf('Ch%d', sel_idx(2))});
ch_names = name_dlg;

%% ── PARAMETERS ───────────────────────────────────────────────────────────
EXPAND_Z        = 3;       % Z-expansion factor (match pipeline value)
MIN_VOLUME      = 1;       % minimum object size in voxels
CROP_HALF       = 50;      % half-width of crop around selected point (px)

% Original method
OTSU_MULT_CH1   = 4;       % multithresh multiplier for channel 1
OTSU_MULT_CH2   = 1;       % multithresh multiplier for channel 2

% Adaptive Gaussian-residual method
SIGMA_SIGNAL    = 0.5;     % signal smoothing (sigma, px)
SIGMA_BG_XY     = 7;       % background kernel XY (px)
% Z sigma is auto-scaled by EXPAND_Z below
SENSITIVITY     = 0.10;    % fraction above local background → foreground

% Percentile method
PCT_CH1         = 97;      % top N% foreground for channel 1  (lower = more)
PCT_CH2         = 95;      % top N% foreground for channel 2

%% ── LOAD FIRST FILE ──────────────────────────────────────────────────────
image_path = fullfile(pathName, fileNames{1});
info       = imfinfo(image_path);
num_slices = numel(info) / num_ch;

fprintf('Loading %s  (%d slices x %d channels)...\n', ...
    fileNames{1}, num_slices, num_ch);

image1 = zeros(info(1).Height, info(1).Width, num_slices, 'uint16');
image2 = zeros(info(1).Height, info(1).Width, num_slices, 'uint16');
for k = 1:num_slices
    image1(:,:,k) = imread(image_path, (k-1)*num_ch + sel_idx(1));
    image2(:,:,k) = imread(image_path, (k-1)*num_ch + sel_idx(2));
end
fprintf('Done loading.\n');

%% ── OBJECT SELECTION (same MIP display as pipeline) ─────────────────────
ch1g   = imgaussfilt3(image1, 0.5);
ch2g   = imgaussfilt3(image2, 0.5);
ch1f   = adapthisteq(max(ch1g, [], 3), 'ClipLimit', 0.0005) * 30;
ch2f   = adapthisteq(max(ch2g, [], 3), 'ClipLimit', 0.0005) * 20;
merged = cat(3, ch1f, ch2f, zeros(size(ch1f), 'like', ch1f));

fig_sel = figure('Name', 'Select objects – press Enter when done', ...
    'Units', 'normalized', 'Position', [0.05 0.1 0.6 0.8]);
imshow(merged);
title('Click on synaptic objects, then press Enter', 'FontSize', 12);
[x_pts, y_pts] = getpts(fig_sel);
close(fig_sel);

if isempty(x_pts)
    disp('No points selected – exiting.');
    return;
end
n_pts = length(x_pts);
fprintf('%d object(s) selected.\n', n_pts);

%% ── PER-OBJECT COMPARISON LOOP ───────────────────────────────────────────
for obj = 1:n_pts

    com     = round([x_pts(obj), y_pts(obj)]);
    x_start = max(1,            com(1) - CROP_HALF);
    x_end   = min(size(image1,2), com(1) + CROP_HALF);
    y_start = max(1,            com(2) - CROP_HALF);
    y_end   = min(size(image1,1), com(2) + CROP_HALF);

    crop1   = image1(y_start:y_end, x_start:x_end, :);
    crop2   = image2(y_start:y_end, x_start:x_end, :);

    % Z-expand (match pipeline)
    SIGMA_BG_Z = max(1, round(SIGMA_BG_XY / EXPAND_Z));
    exim1 = expand(crop1, [1, 1, EXPAND_Z]);
    exim2 = expand(crop2, [1, 1, EXPAND_Z]);

    fprintf('\nObject %d/%d  (crop [%d-%d, %d-%d])...\n', ...
        obj, n_pts, x_start, x_end, y_start, y_end);

    % ── METHOD 1: Original (Otsu x multiplier) ───────────────────────────
    gf1_orig  = imgaussfilt3(exim1, 0.5);
    gf2_orig  = imgaussfilt3(exim2, 1.0);
    thr1_orig = multithresh(gf1_orig) * OTSU_MULT_CH1;
    thr2_orig = multithresh(gf2_orig) * OTSU_MULT_CH2;
    bm1_orig  = gf1_orig >= thr1_orig;
    bm2_orig  = gf2_orig >= thr2_orig;
    bm1_orig  = bwareaopen(imfill(bm1_orig, 'holes'), MIN_VOLUME, 26);
    bm2_orig  = bwareaopen(imfill(bm2_orig, 'holes'), MIN_VOLUME, 26);

    % ── METHOD 2: Adaptive Gaussian-residual ─────────────────────────────
    gf1_sig   = imgaussfilt3(double(exim1), SIGMA_SIGNAL);
    gf2_sig   = imgaussfilt3(double(exim2), SIGMA_SIGNAL);
    bg1       = imgaussfilt3(double(exim1), [SIGMA_BG_XY SIGMA_BG_XY SIGMA_BG_Z]);
    bg2       = imgaussfilt3(double(exim2), [SIGMA_BG_XY SIGMA_BG_XY SIGMA_BG_Z]);
    resid1    = gf1_sig - bg1;
    resid2    = gf2_sig - bg2;
    thr1_adap = SENSITIVITY * max(bg1(:));
    thr2_adap = SENSITIVITY * max(bg2(:));
    bm1_adap  = resid1 > thr1_adap;
    bm2_adap  = resid2 > thr2_adap;
    bm1_adap  = bwareaopen(imfill(bm1_adap, 'holes'), MIN_VOLUME, 26);
    bm2_adap  = bwareaopen(imfill(bm2_adap, 'holes'), MIN_VOLUME, 26);

    % ── METHOD 3: Percentile ─────────────────────────────────────────────
    thr1_pct  = prctile(double(exim1(:)), PCT_CH1);
    thr2_pct  = prctile(double(exim2(:)), PCT_CH2);
    bm1_pct   = double(exim1) >= thr1_pct;
    bm2_pct   = double(exim2) >= thr2_pct;
    bm1_pct   = bwareaopen(imfill(bm1_pct, 'holes'), MIN_VOLUME, 26);
    bm2_pct   = bwareaopen(imfill(bm2_pct, 'holes'), MIN_VOLUME, 26);

    % ── FIGURE ───────────────────────────────────────────────────────────
    % Layout:  4 rows x 2 cols
    %   Row 1: Original image MIPs  (ch1 | ch2)
    %   Row 2: Method 1 overlay     (ch1 | ch2)
    %   Row 3: Method 2 overlay     (ch1 | ch2)
    %   Row 4: Method 3 overlay     (ch1 | ch2)
    %
    % Each overlay: green = foreground mask, grey = MIP background

    mip1 = mat2gray(max(double(exim1), [], 3));
    mip2 = mat2gray(max(double(exim2), [], 3));

    fig = figure('Name', sprintf('Threshold Comparison – Object %d', obj), ...
        'Units', 'normalized', 'Position', [0.05 0.05 0.85 0.90], ...
        'Color', 'w');

    tlo = tiledlayout(fig, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tlo, ...
        sprintf('Threshold Comparison  |  Object %d  |  %s (left)  vs  %s (right)', ...
            obj, ch_names{1}, ch_names{2}), ...
        'FontSize', 12, 'FontWeight', 'bold');

    % Row 1 – raw MIPs
    ax = nexttile(tlo); imshow(mip1, []); 
    title(ax, sprintf('%s – Raw MIP', ch_names{1}), 'FontSize', 9);
    ax = nexttile(tlo); imshow(mip2, []);
    title(ax, sprintf('%s – Raw MIP', ch_names{2}), 'FontSize', 9);

    % Helper: overlay mask on MIP
    overlay = @(mip, mask) cat(3, mip .* ~mask, ...
                                    min(1, mip + 0.55*mask), ...
                                    mip .* ~mask);

    % Row 2 – Method 1
    mip1_lbl = max(label2rgb(max(uint32(bm1_orig) .* ...
        uint32(bwlabeln(bm1_orig,26)), [], 3), 'jet', 'k', 'shuffle'), [], 3);
    ax = nexttile(tlo);
    imshow(overlay(mip1, max(bm1_orig,[],3)), []);
    title(ax, sprintf('Method 1: Otsu x%d  |  thr=%.1f  |  %d vox', ...
        OTSU_MULT_CH1, thr1_orig, nnz(bm1_orig)), 'FontSize', 8);

    ax = nexttile(tlo);
    imshow(overlay(mip2, max(bm2_orig,[],3)), []);
    title(ax, sprintf('Method 1: Otsu x%d  |  thr=%.1f  |  %d vox', ...
        OTSU_MULT_CH2, thr2_orig, nnz(bm2_orig)), 'FontSize', 8);

    % Row 3 – Method 2
    ax = nexttile(tlo);
    imshow(overlay(mip1, max(bm1_adap,[],3)), []);
    title(ax, sprintf('Method 2: Gauss-residual  |  sens=%.2f  |  %d vox', ...
        SENSITIVITY, nnz(bm1_adap)), 'FontSize', 8);

    ax = nexttile(tlo);
    imshow(overlay(mip2, max(bm2_adap,[],3)), []);
    title(ax, sprintf('Method 2: Gauss-residual  |  sens=%.2f  |  %d vox', ...
        SENSITIVITY, nnz(bm2_adap)), 'FontSize', 8);

    % Row 4 – Method 3
    ax = nexttile(tlo);
    imshow(overlay(mip1, max(bm1_pct,[],3)), []);
    title(ax, sprintf('Method 3: Top %.0f%%  |  thr=%.1f  |  %d vox', ...
        100-PCT_CH1, thr1_pct, nnz(bm1_pct)), 'FontSize', 8);

    ax = nexttile(tlo);
    imshow(overlay(mip2, max(bm2_pct,[],3)), []);
    title(ax, sprintf('Method 3: Top %.0f%%  |  thr=%.1f  |  %d vox', ...
        100-PCT_CH2, thr2_pct, nnz(bm2_pct)), 'FontSize', 8);

    % Save
    out_path = fullfile(out_dir, sprintf('thresh_comparison_obj%02d.png', obj));
    exportgraphics(fig, out_path, 'Resolution', 150);
    fprintf('  Saved: %s\n', out_path);

end  % object loop

fprintf('\nDone. Figures saved to:\n  %s\n', out_dir);
msgbox(sprintf('Threshold comparison complete.\n%d object(s) processed.\nFigures saved to thresh_test/', n_pts));
