%% ROI_selection_04.m
% Manual ROI/object selection for downstream signal-intensity analysis.
%
% Workflow follows the object-selection and segmentation logic in
% reference/Synapse_Analysis_v5_xcorr.m:
%   1. Discover registered Reference and Target 3-D TIFF stack pairs.
%   2. Display each image pair sequentially as a Reference/Target merged MIP.
%   3. Click object centers, then press Enter to save and advance.
%   4. Crop a fixed XY box around each point across the full Z-stack.
%   5. Segment Reference and Target crops with Method 3
%      (median-subtract + XY cross-kernel + 3-D spherical convolution).
%   6. Save crops, masks, QC panels, object metadata, and roi_manifest.csv.
%
% Selection rule for antigenicity analysis:
%   Click synaptic objects by Reference signal. Target is shown only for
%   context, to avoid target-brightness selection bias across conditions.

clear; clc;

script_dir = fileparts(mfilename('fullpath'));
reference_dir = fullfile(script_dir, 'reference');
if exist(reference_dir, 'dir')
    addpath(reference_dir);
end

%% Fixed parameters
params.crop_radius_xy = 50;      % matches Synapse_Analysis_v5_xcorr.m
params.expand_z = 3;            % matches EXPAND_Z in Synapse_Analysis_v5_xcorr.m
params.min_volume_voxels = 1;   % same default prompt value as reference
params.display_low_pct = 1;
params.display_high_pct = 99.8;
params.reference_smooth_sigma = 0.5;
params.target_smooth_sigma = 0.5;
params.target_dilate_radius = 3;  % reference pipeline dilates one channel
params.start_at_series_index = 1;
params.batch_timestamp_override = '';

%% Live log for VSCode terminal monitoring
results_dir = fullfile(script_dir, 'analysis_results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end
live_log_path = fullfile(results_dir, 'roi_selection_live.log');
if exist(live_log_path, 'file')
    delete(live_log_path);
end
diary(live_log_path);
diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>
fprintf('Live log: %s\n', live_log_path);

%% Discover and process all registered stack pairs
data_root = fullfile(script_dir, 'data');
if strlength(string(params.batch_timestamp_override)) > 0
    batch_timestamp = char(params.batch_timestamp_override);
else
    batch_timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
end
series_list = discover_registered_series(data_root);

if isempty(series_list)
    error('No registered Reference/Target stack pairs were found under %s.', data_root);
end

fprintf('\nDiscovered %d registered stack pair(s):\n', numel(series_list));
disp(struct2table(rmfield(series_list, {'reference_stack_path', 'target_stack_path', 'series_dir'})));
fprintf('Fixed parameters:\n');
disp(struct2table(params));

batch_manifest = table();
for series_idx = params.start_at_series_index:numel(series_list)
    manifest = process_registered_series(series_list(series_idx), params, ...
        batch_timestamp, series_idx, numel(series_list));
    if ~isempty(manifest)
        batch_manifest = [batch_manifest; manifest]; %#ok<AGROW>
    end
end
batch_manifest_path = fullfile(results_dir, sprintf('roi_selection_batch_%s.csv', batch_timestamp));
if ~isempty(batch_manifest)
    writetable(batch_manifest, batch_manifest_path);
end

fprintf('\nBatch ROI selection complete.\n');
fprintf('Batch manifest: %s\n', batch_manifest_path);

%% Local helper functions
function series_list = discover_registered_series(data_root)
    series_dirs = dir(fullfile(data_root, '*', '*'));
    series_dirs = series_dirs([series_dirs.isdir]);
    series_dirs = series_dirs(~ismember({series_dirs.name}, {'.', '..'}));
    series_list = struct( ...
        'sample_id', {}, 'condition', {}, 'image_id', {}, ...
        'series_dir', {}, 'reference_stack_path', {}, 'target_stack_path', {});

    for i = 1:numel(series_dirs)
        series_dir = fullfile(series_dirs(i).folder, series_dirs(i).name);
        registered_dir = fullfile(series_dir, 'registered_stacks');
        if ~exist(registered_dir, 'dir')
            continue;
        end

        reference_stack_path = fullfile(registered_dir, 'Reference_stack_registered.tif');
        target_stack_path = fullfile(registered_dir, 'Target_stack_registered.tif');
        if ~exist(reference_stack_path, 'file') || ~exist(target_stack_path, 'file')
            fprintf('[SKIP] Missing Reference/Target pair: %s\n', registered_dir);
            continue;
        end

        [~, image_id] = fileparts(series_dir);
        [~, sample_id] = fileparts(fileparts(series_dir));
        series_list(end + 1).sample_id = sample_id; %#ok<SAGROW>
        series_list(end).condition = char(infer_condition(image_id));
        series_list(end).image_id = image_id;
        series_list(end).series_dir = series_dir;
        series_list(end).reference_stack_path = reference_stack_path;
        series_list(end).target_stack_path = target_stack_path;
    end

    if ~isempty(series_list)
        sort_key = strcat(string({series_list.sample_id}), " | ", string({series_list.image_id}));
        [~, order] = sort(sort_key);
        series_list = series_list(order);
    end
end

function manifest = process_registered_series(series_info, params, batch_timestamp, series_idx, n_series)
    sample_id = series_info.sample_id;
    condition = series_info.condition;
    image_id = series_info.image_id;
    series_dir = series_info.series_dir;
    reference_stack_path = series_info.reference_stack_path;
    target_stack_path = series_info.target_stack_path;

    fprintf('\n[%d/%d] %s | %s | %s\n', series_idx, n_series, sample_id, condition, image_id);
    fprintf('Loading Reference: %s\n', reference_stack_path);
    reference_raw = read_tiff_volume(reference_stack_path);
    fprintf('Loading Target   : %s\n', target_stack_path);
    target_raw = read_tiff_volume(target_stack_path);

    if ~isequal(size(reference_raw), size(target_raw))
        error('Reference and Target stack sizes must match for %s.', image_id);
    end

    [height, width, z_slices] = size(reference_raw);

    reference_display = normalize_percentile_for_display( ...
        max(imgaussfilt3(reference_raw, params.reference_smooth_sigma), [], 3), ...
        params.display_low_pct, params.display_high_pct);
    target_display = normalize_percentile_for_display( ...
        max(imgaussfilt3(target_raw, params.target_smooth_sigma), [], 3), ...
        params.display_low_pct, params.display_high_pct);

    merged = cat(3, target_display, reference_display, zeros(size(reference_display)));

    fig = figure('Name', sprintf('ROI selection %d/%d', series_idx, n_series), ...
        'Color', 'w', 'Position', [100 100 1100 950]);
    imshow(merged);
    title({sprintf('[%d/%d] %s', series_idx, n_series, image_id), ...
           'Click Reference-positive object centers. Target is red, Reference is green.', ...
           'Press Enter to save this image and advance. Press Enter without points to skip.'}, ...
           'Interpreter', 'none');
    [x_pts, y_pts] = getpts(fig);
    if isvalid(fig)
        close(fig);
    end

    points = [x_pts, y_pts];
    in_bounds = points(:,1) >= 1 & points(:,1) <= width & ...
        points(:,2) >= 1 & points(:,2) <= height;
    if any(~in_bounds)
        fprintf('[WARN] Ignoring %d point(s) outside the image bounds for %s.\n', ...
            sum(~in_bounds), image_id);
        points = points(in_bounds, :);
    end

    n_objects = size(points, 1);
    if n_objects == 0
        fprintf('[SKIP] No ROI/object centers selected for %s.\n', image_id);
        manifest = table();
        return;
    end
    fprintf('%d object center(s) selected.\n', n_objects);

    roi_root = fullfile(series_dir, 'manual_rois', sprintf('roi_%s', batch_timestamp));
    if ~exist(roi_root, 'dir')
        mkdir(roi_root);
    end

    manifest = table();
    for object_id = 1:n_objects
        center_x = round(points(object_id, 1));
        center_y = round(points(object_id, 2));

        x_start = max(1, center_x - params.crop_radius_xy);
        x_end = min(width, center_x + params.crop_radius_xy);
        y_start = max(1, center_y - params.crop_radius_xy);
        y_end = min(height, center_y + params.crop_radius_xy);

        if x_start > x_end || y_start > y_end
            fprintf('[WARN] Skipping object %03d for %s because crop bounds are empty.\n', ...
                object_id, image_id);
            continue;
        end

        reference_crop = reference_raw(y_start:y_end, x_start:x_end, :);
        target_crop = target_raw(y_start:y_end, x_start:x_end, :);

        reference_expanded = expand_z_volume(reference_crop, params.expand_z);
        target_expanded = expand_z_volume(target_crop, params.expand_z);

        [reference_mask, reference_threshold] = synapse_threshold_m3_local(double(reference_expanded));
        [target_mask, target_threshold] = synapse_threshold_m3_local(double(target_expanded));

        reference_mask = bwareaopen(reference_mask, params.min_volume_voxels, 26);
        target_mask = bwareaopen(target_mask, params.min_volume_voxels, 26);
        if params.target_dilate_radius > 0
            target_mask = imdilate(target_mask, strel('disk', params.target_dilate_radius));
        end
        reference_mask = imfill(reference_mask, 'holes');
        target_mask = imfill(target_mask, 'holes');

        [reference_label, n_reference_objects] = bwlabeln(reference_mask, 26);
        [target_label, n_target_objects] = bwlabeln(target_mask, 26);

        object_folder = fullfile(roi_root, sprintf('object_%03d', object_id));
        if ~exist(object_folder, 'dir')
            mkdir(object_folder);
        end

        write_volume_tiff(reference_crop, fullfile(object_folder, 'reference_crop.tif'));
        write_volume_tiff(target_crop, fullfile(object_folder, 'target_crop.tif'));
        write_volume_tiff(uint8(reference_mask), fullfile(object_folder, 'reference_object_mask.tif'));
        write_volume_tiff(uint8(target_mask), fullfile(object_folder, 'target_object_mask.tif'));
        write_volume_tiff(uint32(reference_label), fullfile(object_folder, 'reference_object_labels.tif'));
        write_volume_tiff(uint32(target_label), fullfile(object_folder, 'target_object_labels.tif'));

        valid_roi_mask = true(size(target_expanded));
        write_volume_tiff(uint8(valid_roi_mask), fullfile(object_folder, 'valid_roi_mask.tif'));

        qc_path = fullfile(object_folder, 'roi_segmentation_qc.png');
        save_roi_qc(reference_expanded, target_expanded, reference_mask, target_mask, ...
            object_id, reference_threshold, target_threshold, qc_path);

        metadata = struct();
        metadata.sample_id = sample_id;
        metadata.condition = condition;
        metadata.image_id = image_id;
        metadata.object_id = object_id;
        metadata.center_x = center_x;
        metadata.center_y = center_y;
        metadata.crop_x_start = x_start;
        metadata.crop_x_end = x_end;
        metadata.crop_y_start = y_start;
        metadata.crop_y_end = y_end;
        metadata.z_start = 1;
        metadata.z_end = z_slices;
        metadata.expand_z = params.expand_z;
        metadata.min_volume_voxels = params.min_volume_voxels;
        metadata.reference_threshold_m3 = reference_threshold;
        metadata.target_threshold_m3 = target_threshold;
        metadata.n_reference_objects = n_reference_objects;
        metadata.n_target_objects = n_target_objects;
        metadata.reference_stack_path = reference_stack_path;
        metadata.target_stack_path = target_stack_path;
        metadata.object_folder = object_folder;
        metadata.qc_path = qc_path;
        write_json(metadata, fullfile(object_folder, 'object_metadata.json'));

        row = table( ...
            string(sample_id), string(condition), string(image_id), object_id, ...
            center_x, center_y, x_start, x_end, y_start, y_end, 1, z_slices, ...
            params.expand_z, params.min_volume_voxels, ...
            reference_threshold, target_threshold, ...
            n_reference_objects, n_target_objects, ...
            string(reference_stack_path), string(target_stack_path), ...
            string(fullfile(object_folder, 'reference_crop.tif')), ...
            string(fullfile(object_folder, 'target_crop.tif')), ...
            string(fullfile(object_folder, 'reference_object_mask.tif')), ...
            string(fullfile(object_folder, 'target_object_mask.tif')), ...
            string(fullfile(object_folder, 'valid_roi_mask.tif')), ...
            string(qc_path), ...
            'VariableNames', { ...
                'sample_id','condition','image_id','object_id', ...
                'center_x','center_y','crop_x_start','crop_x_end', ...
                'crop_y_start','crop_y_end','z_start','z_end', ...
                'expand_z','min_volume_voxels', ...
                'reference_threshold_m3','target_threshold_m3', ...
                'n_reference_objects','n_target_objects', ...
                'reference_stack_path','target_stack_path', ...
                'reference_crop_path','target_crop_path', ...
                'reference_object_mask_path','target_object_mask_path', ...
                'valid_roi_mask_path','qc_path'});
        manifest = [manifest; row]; %#ok<AGROW>
    end

    manifest_path = fullfile(roi_root, 'roi_manifest.csv');
    writetable(manifest, manifest_path);

    params_path = fullfile(roi_root, 'roi_selection_parameters.json');
    params_struct = params;
    params_struct.sample_id = sample_id;
    params_struct.condition = condition;
    params_struct.image_id = image_id;
    params_struct.reference_stack_path = reference_stack_path;
    params_struct.target_stack_path = target_stack_path;
    params_struct.roi_root = roi_root;
    params_struct.batch_timestamp = batch_timestamp;
    write_json(params_struct, params_path);

    fprintf('Saved: %s\n', roi_root);
end

function vol = read_tiff_volume(path)
    vol = tiffreadVolume(path);
    if ndims(vol) ~= 3
        error('Expected a 3-D TIFF stack: %s', path);
    end
end

function write_volume_tiff(vol, out_path)
    if exist(out_path, 'file')
        delete(out_path);
    end
    if isa(vol, 'uint32')
        max_value = max(vol(:));
        if max_value > double(intmax('uint16'))
            error('Cannot write %s as uint16 label TIFF; max label is %d.', out_path, max_value);
        end
        vol = uint16(vol);
    end
    for z = 1:size(vol, 3)
        mode = 'overwrite';
        if z > 1
            mode = 'append';
        end
        imwrite(vol(:,:,z), out_path, 'tif', 'WriteMode', mode);
    end
end

function out = normalize_percentile_for_display(img, low_pct, high_pct)
    img = double(img);
    lo = prctile(img(:), low_pct);
    hi = prctile(img(:), high_pct);
    if hi <= lo
        out = zeros(size(img));
    else
        out = min(max((img - lo) ./ (hi - lo), 0), 1);
    end
end

function expanded = expand_z_volume(vol, expand_z)
    if expand_z <= 1
        expanded = vol;
        return;
    end
    if exist('expand', 'file') == 2
        expanded = expand(vol, [1, 1, expand_z]);
    else
        expanded = repelem(vol, 1, 1, expand_z);
    end
end

function [mask, th] = synapse_threshold_m3_local(A)
    if isempty(A)
        mask = false(size(A));
        th = 0;
        return;
    end

    persistent cross_k2d k7
    if isempty(cross_k2d)
        cross_k2d = [0 1 0; 1 1 1; 0 1 0] / 5;
        [kx, ky, kz] = ndgrid(-2:2, -2:2, -2:2);
        k7_raw = double(sqrt(kx.^2 + ky.^2 + kz.^2) <= 2.5);
        k7 = k7_raw / sum(k7_raw(:));
    end

    A1 = A - median(A(:));
    A1 = max(A1, 0);
    A1s = zeros(size(A1));
    for z = 1:size(A1, 3)
        A1s(:,:,z) = imfilter(A1(:,:,z), cross_k2d, 'replicate', 'same');
    end

    coarse_gate = max(A1s(:)) / 10;
    if isempty(coarse_gate) || ~isfinite(coarse_gate)
        mask = false(size(A));
        th = 0;
        return;
    end

    bright_voxels = A1s(A1s > coarse_gate);
    if isempty(bright_voxels)
        mask = false(size(A));
        th = coarse_gate;
        return;
    end

    th = mean(bright_voxels);
    A2 = A1s - th;
    A2 = convn(A2, k7, 'same');
    mask = A2 > 0;
end

function save_roi_qc(reference_expanded, target_expanded, reference_mask, target_mask, ...
    object_id, reference_threshold, target_threshold, out_path)
    ref_mip = normalize_percentile_for_display(max(reference_expanded, [], 3), 1, 99.8);
    tgt_mip = normalize_percentile_for_display(max(target_expanded, [], 3), 1, 99.8);
    ref_mask_mip = max(reference_mask, [], 3);
    tgt_mask_mip = max(target_mask, [], 3);

    overlay_ref = cat(3, ref_mip .* ~ref_mask_mip, ...
                         min(1, ref_mip + 0.55 * ref_mask_mip), ...
                         ref_mip .* ~ref_mask_mip);
    overlay_tgt = cat(3, min(1, tgt_mip + 0.55 * tgt_mask_mip), ...
                         tgt_mip .* ~tgt_mask_mip, ...
                         tgt_mip .* ~tgt_mask_mip);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 700]);
    tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; imshow(ref_mip); title('Reference crop MIP');
    nexttile; imshow(tgt_mip); title('Target crop MIP');
    nexttile; imshow(cat(3, tgt_mip, ref_mip, zeros(size(ref_mip))));
    title('Target red / Reference green');
    nexttile; imshow(ref_mask_mip); title(sprintf('Reference M3 mask (th=%.2f)', reference_threshold));
    nexttile; imshow(tgt_mask_mip); title(sprintf('Target M3 mask (th=%.2f)', target_threshold));
    nexttile; imshow(max(overlay_ref, overlay_tgt)); title(sprintf('Object %03d masks overlay', object_id));
    exportgraphics(fig, out_path, 'Resolution', 200);
    close(fig);
end

function condition = infer_condition(series_name)
    s = lower(series_name);
    if contains(s, 'fix')
        condition = "FiX-ITS";
    elseif contains(s, 'multiexr') || contains(s, 'multi')
        condition = "multi-ExR";
    elseif contains(s, 'panexm') || contains(s, 'panex')
        condition = "pan-ExM-t";
    else
        condition = "unknown";
    end
end

function write_json(s, out_path)
    fid = fopen(out_path, 'w');
    if fid < 0
        error('Could not write JSON file: %s', out_path);
    end
    cleanup = onCleanup(@() fclose(fid));
    txt = jsonencode(s, 'PrettyPrint', true);
    fwrite(fid, txt, 'char');
end
