%%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%% ~~~~~~~~~ Analysis pipeline for Antigenicity assessment ~~~~~~~~~~~~~%%
%%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%% User prompt for image selection capable of accepting multiple input images
%%% Assumes TIFF and prompts user for specifying the number of color channels
%%% User is requested to select channels for analysis
%%% Data will be saved to .csv files based on User input of Channel name
%%% If multiple images are selected then these will be appended with ...
%%% column headers re-stated before the start of new data
%%%
%%% XCORR COMPATIBILITY EDITS  (all marked %%% XCORR EDIT):
%%%
%%%   EDIT 1  Hardcoded paths removed; all outputs use objects_folder / pathName.
%%%
%%%   EDIT 2  labeled_images_map and raw_images_map declared before the loop.
%%%           Volumes are allocated in EXPANDED Z space (Z x 3) so the arrays
%%%           handed to xcorr3D_analysis are consistent with the centroids that
%%%           dist_vectors_v3 sees (centroids are computed on exim1/2).
%%%
%%%   EDIT 3  Object ID written as integer (%d) in both CSVs.
%%%           Label value painted into the full-field volume equals j, matching
%%%           the Object column exactly.  No offset arithmetic — j is the
%%%           unique ID shared by CSV, label volume, and xcorr3D_analysis.
%%%
%%%   EDIT 4  Crop coordinates stored in crop_pts during the synapse-selection
%%%           loop.  The analysis loop (j) indexes crop_pts(j,:) directly,
%%%           removing the implicit assumption that pts rows and channel cell
%%%           rows are always aligned.
%%%
%%%   EDIT 5  Corrected Z scale (native_z_scale / EXPAND_Z) built once and
%%%           passed to BOTH dist_vectors_v3 AND xcorr3D_analysis.
%%%           Centroids are in expanded-Z pixel space; using the native scale
%%%           would over-estimate Z distances by EXPAND_Z.
%%%
%%%   EDIT 6  Full-field volumes written in expanded Z space (no collapse back
%%%           to native Z).  xcorr3D_analysis sub-volumes are therefore
%%%           isotropic-Z, consistent with the corrected scale passed in EDIT 5.

if exist('reference_image_path_override','var') && ...
   exist('target_image_path_override','var')
    reference_image_path = reference_image_path_override;
    target_image_path    = target_image_path_override;
    [reference_path, reference_file, ref_ext] = fileparts(reference_image_path);
    [target_path, target_file, tgt_ext]       = fileparts(target_image_path);
    reference_file = [reference_file ref_ext];
    target_file    = [target_file tgt_ext];
else
    [reference_file, reference_path] = uigetfile( ...
        {'*.tif','TIFF Files';'*.*','All Files'}, ...
        'Select Reference 3D stack TIFF');
    if isequal(reference_file, 0)
        error('Reference stack selection cancelled.');
    end

    [target_file, target_path] = uigetfile( ...
        {'*.tif','TIFF Files';'*.*','All Files'}, ...
        'Select Target 3D stack TIFF');
    if isequal(target_file, 0)
        error('Target stack selection cancelled.');
    end

    reference_image_path = fullfile(reference_path, reference_file);
    target_image_path    = fullfile(target_path, target_file);
end

reference_folder = fileparts(reference_image_path);
target_folder    = fileparts(target_image_path);
if ~strcmp(reference_folder, target_folder)
    error('Reference and Target stacks must be selected from the same folder.');
end

pathName = reference_folder;
[~, analysis_label] = fileparts(pathName);
fileNames = {analysis_label};

%%% XCORR EDIT 1a ────────────────────────────────────────────────────────
objects_folder = fullfile(pathName, ...
    sprintf('objects_%s', datestr(now, 'yyyymmdd_HHMMSS')));
if ~exist(objects_folder, 'dir')
    mkdir(objects_folder);
end
%%% ──────────────────────────────────────────────────────────────────────

%% Parameters
if exist('min_volume_override','var')
    min_volume = min_volume_override;
else
    t_inputs   = inputdlg({'Object Volume Threshold :'}, ...
        'Expected Synaptic Signal Volume (pixels)',[1 100],{'1'});
    min_volume = str2double(t_inputs{1});
end

num_ch = 2;

[~, ref_default_name, ~] = fileparts(reference_file);
[~, tgt_default_name, ~] = fileparts(target_file);
ref_default_name = regexprep(ref_default_name, '_stack$', '');
tgt_default_name = regexprep(tgt_default_name, '_stack$', '');

if exist('channel_names_override','var')
    channel_names = channel_names_override;
else
    channel_names = inputdlg( ...
        {'Reference channel name:','Target channel name:'}, ...
        'Channel Names',[1 100], {ref_default_name, tgt_default_name});
    if isempty(channel_names) || numel(channel_names) < 2
        error('Channel naming cancelled.');
    end
end

if exist('comp_vectors_override','var')
    comp_vectors = comp_vectors_override;
else
    comp_vectors = questdlg('Compute vectors between channel centroids?', ...
        'Vector Analysis','Yes','No','Yes');
end

if strcmp(comp_vectors,'Yes')
    if exist('scale_input_override','var')
        scale_input = scale_input_override;
    else
        scale_input = inputdlg( ...
            {'XY pixel size (um/pixel):','Z pixel size (um/pixel):'}, ...
            'Pixel Scaling Factors',[1 50],{'0.172','0.179'});
    end
end

%%% XCORR EDIT 2a ────────────────────────────────────────────────────────
%%% Voxel-expansion factor used by expand().
%%% If you change the expand() call below, update this constant too.
EXPAND_Z = 3;

%%% Maps keyed by channel_name, populated at end of the analysis loop.
labeled_images_map = containers.Map('KeyType','char','ValueType','any');
raw_images_map     = containers.Map('KeyType','char','ValueType','any');
%%% ──────────────────────────────────────────────────────────────────────

ch_csv_init = containers.Map('KeyType','char','ValueType','logical');

for file_idx = 1:length(fileNames)

    ref_info   = imfinfo(reference_image_path);
    tgt_info   = imfinfo(target_image_path);
    num_slices = numel(ref_info);

    if num_ch > 1

        if numel(ref_info) ~= numel(tgt_info)
            error('Reference and Target stacks must have the same number of Z slices.');
        end

        %% Load raw channel volumes
        fprintf('Loading %s - %s\n', fileNames{file_idx}, channel_names{1});
        ch1 = tiffreadVolume(reference_image_path);

        fprintf('Loading %s - %s\n', fileNames{file_idx}, channel_names{2});
        ch2 = tiffreadVolume(target_image_path);

        if ~isequal(size(ch1), size(ch2))
            error('Reference and Target stack sizes must match.');
        end

        ch1_raw = ch1;
        ch2_raw = ch2;

        if exist('apply_drift_correction_override','var')
            apply_drift_correction = apply_drift_correction_override;
        else
            apply_drift_correction = questdlg( ...
                'Apply sequential slice drift correction before ROI selection?', ...
                'Slice Drift Correction','Yes','No','Yes');
        end

        if strcmp(apply_drift_correction,'Yes')
            if exist('registration_channel_override','var')
                reg_choice = registration_channel_override;
            else
                reg_choice = questdlg( ...
                    'Which channel should drive translation registration?', ...
                    'Registration Channel','Reference','Target','Reference');
            end

            reg_channel_idx = 1;
            if strcmp(reg_choice,'Target')
                reg_channel_idx = 2;
            end

            fprintf('\nRunning slice drift correction using %s channel...\n', ...
                channel_names{reg_channel_idx});
            [ch1, ch2, drift_transforms] = register_slice_drift_translation( ...
                ch1, ch2, reg_channel_idx);

            write_volume_tiff(ch1, fullfile(objects_folder, ...
                sprintf('%s_registered_stack.tif', channel_names{1})));
            write_volume_tiff(ch2, fullfile(objects_folder, ...
                sprintf('%s_registered_stack.tif', channel_names{2})));

            drift_fig = figure('Name','Slice Drift Correction Preview', ...
                'Position',[100 100 1200 500]);
            tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
            nexttile;
            imshow(cat(3, ...
                adapthisteq(max(ch1,[],3),'ClipLimit',0.0005) * 30, ...
                adapthisteq(max(ch2,[],3),'ClipLimit',0.0005) * 20, ...
                zeros(size(ch1,1), size(ch1,2))));
            title('Registered MIP');
            nexttile;
            plot(drift_transforms(:,1), '-o', 'LineWidth', 1);
            hold on;
            plot(drift_transforms(:,2), '-o', 'LineWidth', 1);
            hold off;
            xlabel('Slice Pair');
            ylabel('Translation (pixels)');
            legend({'dX','dY'}, 'Location','best');
            title('Per-slice Translation');
            nexttile;
            plot(vecnorm(drift_transforms, 2, 2), '-o', 'LineWidth', 1);
            xlabel('Slice Pair');
            ylabel('Shift magnitude (pixels)');
            title('Drift Magnitude');
            exportgraphics(drift_fig, fullfile(objects_folder, ...
                'slice_drift_correction_preview.png'));
            close(drift_fig);

            create_registration_metrics_figure( ...
                ch1_raw, ch2_raw, ch1, ch2, drift_transforms, ...
                reg_channel_idx, channel_names, objects_folder);

            fprintf('Slice drift correction complete. Registered stacks saved in:\n  %s\n', ...
                objects_folder);

            if exist('registration_only_override','var')
                registration_only = registration_only_override;
            else
                registration_only = questdlg( ...
                    'Save registration outputs only and skip downstream analysis?', ...
                    'Registration Only','Yes','No','No');
            end

            if strcmp(registration_only,'Yes')
                fprintf('Registration-only mode selected. Skipping ROI analysis and CSV outputs.\n');
                continue;
            end
        end

        %% Merged MIP for manual synapse selection
        ch1g   = imgaussfilt3(ch1, 0.5);
        ch2g   = imgaussfilt3(ch2, 0.5);
        ch1f   = normalize_percentile_for_display(max(ch1g,[],3), 1, 99.8);
        ch2f   = normalize_percentile_for_display(max(ch2g,[],3), 1, 99.8);
        merged = cat(3, ch1f, ch2f, zeros(size(ch1f)));
        if exist('roi_points_override','var')
            pts = roi_points_override;
        else
            subplot(1,1,1);
            imshow(merged);
            title('Select objects --> Press Enter to Proceed');
            [x, y] = getpts;
            close;
            pts = [x, y];
        end
        n_pts  = size(pts, 1);
        ob2    = round(n_pts / 2);
        tiledlayout(ob2, ob2);

        %%% XCORR EDIT 4a ────────────────────────────────────────────────
        %%% Store crop bounds alongside saved TIFFs so the analysis loop
        %%% can recover them by index without relying on pts alignment.
        crop_pts = zeros(n_pts, 4);   % [x_start x_end y_start y_end]
        %%% ──────────────────────────────────────────────────────────────

        %% Crop and save each selected synapse
        for i = 1:n_pts
            com     = round(pts(i,:));
            x_start = max(1,          com(1) - 50);
            x_end   = min(size(ch1,2), com(1) + 50);
            y_start = max(1,          com(2) - 50);
            y_end   = min(size(ch1,1), com(2) + 50);
            z_max   = size(ch1, 3);

            %%% XCORR EDIT 4b: record crop bounds indexed by i
            crop_pts(i,:) = [x_start, x_end, y_start, y_end];

            crop_ch1 = ch1(y_start:y_end, x_start:x_end, 1:z_max);
            crop_ch2 = ch2(y_start:y_end, x_start:x_end, 1:z_max);

            object_folder_path = fullfile(objects_folder, sprintf('object_%d',i));
            if ~exist(object_folder_path,'dir'); mkdir(object_folder_path); end

            crop_ch1_mip = normalize_percentile_for_display(max(crop_ch1,[],3), 1, 99.8);
            crop_ch2_mip = normalize_percentile_for_display(max(crop_ch2,[],3), 1, 99.8);
            mx = cat(3, crop_ch1_mip, crop_ch2_mip, ...
                        zeros(size(crop_ch1,1), size(crop_ch1,2)));
            nexttile; imshow(mx);

            if max(mx(:)) > 0
                %%% XCORR EDIT 1b: portable paths
                imwrite(uint16(mx), fullfile(objects_folder, ...
                    sprintf('maxproj_%d.tif',i)), 'tif');
                f1 = fullfile(object_folder_path, sprintf('ch1_object_%d.tif',i));
                f2 = fullfile(object_folder_path, sprintf('ch2_object_%d.tif',i));
                for o = 1:z_max
                    mode = 'overwrite'; if o > 1; mode = 'append'; end
                    imwrite(crop_ch1(:,:,o), f1,'tif','WriteMode',mode);
                    imwrite(crop_ch2(:,:,o), f2,'tif','WriteMode',mode);
                end
            end
        end

        %% Read saved crops back from disk
        %%% XCORR EDIT 1c: portable parentDir
        allFolders    = regexp(genpath(objects_folder), ...
            ['[^' regexptranslate('escape', pathsep) ']+'], 'match');
        allFolders(1) = [];
        channel1 = {};
        channel2 = {};
        %%% XCORR EDIT 4c: also store per-object crop bounds in read-back order
        crop_pts_ordered = zeros(0, 4);

        for i = 1:length(allFolders)
            folder     = allFolders{i};
            imageFiles = dir(fullfile(folder,'*.tif'));
            if length(imageFiles) >= 2
                channel1{end+1} = tiffreadVolume(fullfile(folder, imageFiles(1).name)); 
                channel2{end+1} = tiffreadVolume(fullfile(folder, imageFiles(2).name)); 
                % Recover matching crop_pts row by parsing object index from folder name
                tok = regexp(folder, 'object_(\d+)$','tokens');
                if ~isempty(tok)
                    obj_i = str2double(tok{1}{1});
                    if obj_i <= size(crop_pts,1)
                        crop_pts_ordered(end+1,:) = crop_pts(obj_i,:); 
                    else
                        crop_pts_ordered(end+1,:) = [0 0 0 0]; 
                    end
                end
            end
        end

        %% Initialise CSVs
        %%% XCORR EDIT 1d: portable csv paths
        csv_path1 = fullfile(objects_folder, ...
            sprintf('%s_analysis_results.csv', channel_names{1}));
        csv_path2 = fullfile(objects_folder, ...
            sprintf('%s_analysis_results.csv', channel_names{2}));

        fid = fopen(csv_path1,'w');
        fprintf(fid,'Image,Object,Channel,Volume,MeanIntensity,X,Y,Z,EigVX,EigVY,EigVZ,MajorAxis\n');
        fclose(fid);
        fid = fopen(csv_path2,'w');
        fprintf(fid,'Image,Object,Channel,Volume,MeanIntensity,X,Y,Z,EigVX,EigVY,EigVZ,MajorAxis\n');
        fclose(fid);
        ch_csv_init(channel_names{1}) = true;
        ch_csv_init(channel_names{2}) = true;
        fprintf('CSV files initialised.\n');

        %%% XCORR EDIT 2b ────────────────────────────────────────────────
        %%% Pre-allocate full-field label and raw volumes in EXPANDED Z
        %%% space.  Centroids from regionprops3 are in exim1/2 coords, so
        %%% the label volume must also be in that same expanded space.
        full_H  = size(ch1,1);
        full_W  = size(ch1,2);
        full_Z  = size(ch1,3) * EXPAND_Z;   % expanded Z dimension

        lbl_vol_ch1 = zeros(full_H, full_W, full_Z, 'uint32');
        lbl_vol_ch2 = zeros(full_H, full_W, full_Z, 'uint32');
        raw_vol_ch1 = zeros(full_H, full_W, full_Z, 'double');
        raw_vol_ch2 = zeros(full_H, full_W, full_Z, 'double');
        %%% ──────────────────────────────────────────────────────────────

        %% Per-object analysis loop
        for j = 1:length(channel1)
            image1 = channel1{1,j};
            image2 = channel2{1,j};

            exim1 = expand(image1, [1, 1, EXPAND_Z]);
            exim2 = expand(image2, [1, 1, EXPAND_Z]);

            mid_slice = cat(2, max(exim1,[],3)*30, max(exim2,[],3)*15);

            % ── Thresholding: Method 3 (median-subtract + spherical convolution)
            % synapse_threshold_m3 operates on the raw expanded double volume.
            % It returns the binary foreground mask and the derived threshold
            % value th (used for display in the figure title).
            % bwareaopen and imfill are applied here after the call so the
            % pipeline post-processing steps are unchanged.
            [binmask1, th1] = synapse_threshold_m3(double(exim1));
            [binmask2, th2] = synapse_threshold_m3(double(exim2));
            mask_step1 = cat(2, max(binmask1,[],3), max(binmask2,[],3));
            se = strel('disk',3);
            binmask1   = bwareaopen(binmask1, min_volume, 26);
            binmask2a   = bwareaopen(binmask2, min_volume, 26);
            binmask2   = imdilate(binmask2a,se);
            mask_step2 = cat(2, max(binmask1,[],3), max(binmask2,[],3));

            binmask1   = imfill(binmask1,'holes');
            binmask2   = imfill(binmask2,'holes');
            mask_step3 = cat(2, max(binmask1,[],3), max(binmask2,[],3));

            [labeled1, ~] = bwlabeln(binmask1, 26);
            [labeled2, ~] = bwlabeln(binmask2, 26);

            props1 = regionprops3(labeled1, exim1, ...
                'Centroid','Volume','MeanIntensity','BoundingBox','EigenVectors','EigenValues');
            props2 = regionprops3(labeled2, exim2, ...
                'Centroid','Volume','MeanIntensity','BoundingBox','EigenVectors','EigenValues');

            % Valid object filter: volume threshold only.
            % Method 3 does not produce a fixed intensity threshold comparable
            % to the Otsu multiplier, so MeanIntensity filtering is removed.
            % Objects are accepted if they meet the minimum volume criterion.
            valid_obs1 = props1.Volume >= min_volume;
            valid_obs2 = props2.Volume >= min_volume;

            cent1    = props1.Centroid(valid_obs1,:);
            cent2    = props2.Centroid(valid_obs2,:);
            vol1     = props1.Volume(valid_obs1,:);
            vol2     = props2.Volume(valid_obs2,:);
            avg_int1 = props1.MeanIntensity(valid_obs1,:);
            avg_int2 = props2.MeanIntensity(valid_obs2,:);
            eigv1    = props1.EigenVectors(valid_obs1,:);
            eigv2    = props2.EigenVectors(valid_obs2,:);
            d1       = props1.EigenValues(valid_obs1,:);
            d2       = props2.EigenValues(valid_obs2,:);

            vob1 = size(cent1,1);  vob2 = size(cent2,1);
            paxis1 = zeros(vob1,3);  paxis2 = zeros(vob2,3);
            dim1   = zeros(vob1,3);  dim2   = zeros(vob2,3);
            for z  = 1:vob1; paxis1(z,:) = (eigv1{z}(:,1))'; dim1(z,:) = (d1{z}(1,:))'; end
            for zz = 1:vob2; paxis2(zz,:)= (eigv2{zz}(:,1))';dim2(zz,:)= (d2{zz}(1,:))';end

            %%% XCORR EDIT 3a ─────────────────────────────────────────────
            %%% Write Object as INTEGER (%d).
            %%% j is the unique object ID used by both the CSV and the label
            %%% volume — no offset arithmetic needed.
            fid = fopen(csv_path1,'a');
            for i = 1:size(cent1,1)
                fprintf(fid,'%s,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                    fileNames{file_idx}, j, channel_names{1}, ...
                    vol1(i), avg_int1(i), ...
                    cent1(i,1), cent1(i,2), cent1(i,3), ...
                    paxis1(i,2), paxis1(i,1), paxis1(i,3), dim1(i));
            end
            fclose(fid);

            fid = fopen(csv_path2,'a');
            for ii = 1:size(cent2,1)
                fprintf(fid,'%s,%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f\n', ...
                    fileNames{file_idx}, j, channel_names{2}, ...
                    vol2(ii), avg_int2(ii), ...
                    cent2(ii,1), cent2(ii,2), cent2(ii,3), ...
                    paxis2(ii,2), paxis2(ii,1), paxis2(ii,3), dim2(ii));
            end
            fclose(fid);
            %%% ──────────────────────────────────────────────────────────

            %%% XCORR EDIT 2c / 3b / 4d ───────────────────────────────────
            %%% Write this object's expanded labeled and raw arrays into the
            %%% full-field volume at the correct spatial position.
            %%%
            %%% - Crop bounds recovered from crop_pts_ordered(j,:) — safe
            %%%   even if some objects were skipped during the read-back.
            %%% - Label value = j (integer), matching the CSV Object column.
            %%% - NO Z collapse: arrays remain in expanded space (full_Z).
            %%% - Existing labels in overlapping regions are preserved
            %%%   (earlier objects are not overwritten).

            if j <= size(crop_pts_ordered,1) && any(crop_pts_ordered(j,:))
                xs_j = crop_pts_ordered(j,1);
                xe_j = crop_pts_ordered(j,2);
                ys_j = crop_pts_ordered(j,3);
                ye_j = crop_pts_ordered(j,4);

                h_crop = ye_j - ys_j + 1;
                w_crop = xe_j - xs_j + 1;
                z_crop = min(size(labeled1,3), full_Z);

                % Trim labeled / raw arrays to match actual crop footprint
                lbl1_w = labeled1(1:h_crop, 1:w_crop, 1:z_crop);
                lbl2_w = labeled2(1:h_crop, 1:w_crop, 1:z_crop);
                raw1_w = double(exim1(1:h_crop, 1:w_crop, 1:z_crop));
                raw2_w = double(exim2(1:h_crop, 1:w_crop, 1:z_crop));

                obj_mask1 = lbl1_w > 0;
                obj_mask2 = lbl2_w > 0;

                % Paint label = j; do not overwrite previously placed objects
                lbl_vol_ch1(ys_j:ye_j, xs_j:xe_j, 1:z_crop) = ...
                    lbl_vol_ch1(ys_j:ye_j, xs_j:xe_j, 1:z_crop) .* uint32(~obj_mask1) + ...
                    uint32(obj_mask1) * uint32(j);

                lbl_vol_ch2(ys_j:ye_j, xs_j:xe_j, 1:z_crop) = ...
                    lbl_vol_ch2(ys_j:ye_j, xs_j:xe_j, 1:z_crop) .* uint32(~obj_mask2) + ...
                    uint32(obj_mask2) * uint32(j);

                raw_vol_ch1(ys_j:ye_j, xs_j:xe_j, 1:z_crop) = ...
                    raw_vol_ch1(ys_j:ye_j, xs_j:xe_j, 1:z_crop) + raw1_w;

                raw_vol_ch2(ys_j:ye_j, xs_j:xe_j, 1:z_crop) = ...
                    raw_vol_ch2(ys_j:ye_j, xs_j:xe_j, 1:z_crop) + raw2_w;
            else
                warning('Object %d: crop bounds not found — skipped from volume maps.',j);
            end
            %%% ──────────────────────────────────────────────────────────

            %% Display processing steps
            figure('Name', sprintf('%s - Object %d', fileNames{file_idx}, j), ...
                'Position',[100 100 1200 800]);
            tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
            nexttile; imshow(mid_slice);  title('Original Image');
            nexttile; imshow(mask_step1);
            title(sprintf('M3 Threshold  (th1=%.1f | th2=%.1f)', th1, th2));
            nexttile; imshow(mask_step2); title('Small Objects Removed');
            nexttile; imshow(mask_step3); title('Holes Filled');
            nexttile;
            imshow(cat(2, label2rgb(max(labeled1,[],3),'jet','k','shuffle'), ...
                          label2rgb(max(labeled2,[],3),'jet','k','shuffle')));
            title('Labeled Objects');
            nexttile; imshow(mid_slice); hold on; axis off;
            % Draw principal axis for ALL reference (ch2) objects
            for k = 1:size(cent2,1)
                quiver([cent2(k,1), cent2(k,1)], [cent2(k,2), cent2(k,2)], ...
                    [-paxis2(k,2)*30,  paxis2(k,2)*30], ...
                    [-paxis2(k,1)*30,  paxis2(k,1)*30], ...
                    'Color','r','LineWidth',1,'AutoScale','off');
            end
            % Plot centroid marker for ALL target (ch1) objects
            for k = 1:size(cent1,1)
                plot(cent1(k,1), cent1(k,2), 'o', ...
                    'MarkerSize',5,'MarkerFaceColor','g','MarkerEdgeColor','g');
            end
            hold off; title('Detected Objects');
            
            sgtitle(sprintf('%s - Object %d Pipeline', fileNames{file_idx}, j), ...
                'FontSize',14,'FontWeight','bold');
            %%% XCORR EDIT 1e: portable export path
            exportgraphics(gcf, fullfile(objects_folder, ...
                sprintf('Object%0.0d_segmentation.jpg',j)));
            fprintf('Object %d complete.\n',j);

        end  % j loop

        %%% XCORR EDIT 2d: store completed full-field maps ────────────────
        labeled_images_map(channel_names{1}) = lbl_vol_ch1;
        labeled_images_map(channel_names{2}) = lbl_vol_ch2;
        raw_images_map(channel_names{1})     = raw_vol_ch1;
        raw_images_map(channel_names{2})     = raw_vol_ch2;
        fprintf('Volume maps populated (%d objects, expanded Z=%d slices).\n', ...
            length(channel1), full_Z);
        %%% ──────────────────────────────────────────────────────────────

        msgbox(sprintf('Analysis complete for %d channels', numel(channel_names)));

    end  % num_ch > 1
end  % file loop

%% Vector analysis and cross-correlation
registration_only_active = ...
    exist('registration_only','var') && strcmp(registration_only,'Yes');

if ~registration_only_active && exist('channel_names','var') && length(channel_names) >= 2
    if strcmp(comp_vectors,'Yes')

        threshold_input = inputdlg('Enter distance threshold (um):', ...
            'Distance Threshold',[1 50],{'2.7'});

        if ~isempty(threshold_input)
            distance_threshold = str2double(threshold_input{1});

            %%% XCORR EDIT 5 ──────────────────────────────────────────────
            %%% Build the corrected scale for expanded-Z centroid space.
            %%% Centroids from regionprops3 on exim1/2 have Z coordinates
            %%% that are EXPAND_Z x larger than native pixel indices.
            %%% Both dist_vectors_v3 AND xcorr3D_analysis must receive the
            %%% same corrected Z scale so distances are correct in um.
            xy_sc = str2double(scale_input{1});
            z_sc  = str2double(scale_input{2}) / EXPAND_Z;
            scale_input_expanded = {num2str(xy_sc,'%.6f'), num2str(z_sc,'%.6f')};

            fprintf('\nUsing corrected Z scale: %.6f um/px (native %.6f / %d)\n', ...
                z_sc, str2double(scale_input{2}), EXPAND_Z);

            [results_table, pair_info] = dist_vectors_v3( ...
                objects_folder, channel_names, distance_threshold, scale_input_expanded);
            %%% ──────────────────────────────────────────────────────────

            if ~isempty(results_table)
                fprintf('\nVector analysis complete. %d pairs found.\n', ...
                    height(results_table));
            end
        end

        run_xcorr = questdlg( ...
            'Run 3-D cross-correlation analysis on detected pairs?', ...
            '3-D Cross-Correlation','Yes','No','Yes');

        if strcmp(run_xcorr,'Yes') && ~isempty(results_table)
            if isKey(labeled_images_map, channel_names{1}) && ...
               isKey(labeled_images_map, channel_names{2})

                %%% XCORR EDIT 5 (cont.): pass corrected scale to xcorr3D_analysis
                xcorr3D_analysis(objects_folder, channel_names, scale_input_expanded, ...
                                 labeled_images_map, raw_images_map);

                fprintf('\n3-D cross-correlation outputs written to:\n');
                fprintf('  %s_to_%s_xcorr_full.csv\n',    channel_names{1}, channel_names{2});
                fprintf('  %s_to_%s_xcorr_summary.csv\n', channel_names{1}, channel_names{2});
                fprintf('  %s_to_%s_xcorr_rawdata.csv\n', channel_names{1}, channel_names{2});
                fprintf('  %s_to_%s_xcorr_figure.fig/.png\n', channel_names{1}, channel_names{2});
            else
                warndlg(['Cross-correlation skipped: volume maps missing. ' ...
                    'Ensure all objects were processed in the current session.'], ...
                    'XCorr Warning');
            end
        end

    end
end
%% Compile unified distance summary
% Reads: vectors CSV + both xcorr summary CSVs
% Writes: <Ch1>_to_<Ch2>_distance_summary.csv
%
% Columns in output:
%   Pair, C2C_um, PDist_um, Angle_Btw_deg,
%   Tangential_um,          <- sqrt(C2C^2 - PDist^2)
%   Sarkar_XCorr_um, Sarkar_GCorrPeak, Sarkar_HMS_offset_nm,
%   Hybrid_XCorr_um, Hybrid_GCorrPeak, Hybrid_HMS_offset_nm,
%   PDist_vs_C2C_ratio,     <- PDist/C2C — 1.0 = purely normal, 0 = purely tangential
%   Sarkar_vs_PDist_ratio,  <- Sarkar_XCorr / PDist
%   Hybrid_vs_PDist_ratio   <- Hybrid_XCorr / PDist

if ~registration_only_active
    vec_path   = fullfile(objects_folder, ...
        sprintf('%s_to_%s_vectors.csv',    channel_names{1}, channel_names{2}));
    sark_path  = fullfile(objects_folder, ...
        sprintf('%s_to_%s_xcorr_summary_sarkar.csv', channel_names{1}, channel_names{2}));
    hyb_path   = fullfile(objects_folder, ...
        sprintf('%s_to_%s_xcorr_summary_hybrid.csv', channel_names{1}, channel_names{2}));

    if isfile(vec_path) && isfile(sark_path) && isfile(hyb_path)
        vT  = readtable(vec_path);
        sT  = readtable(sark_path);
        hT  = readtable(hyb_path);

        % Build pair label to match xcorr summary format
        pair_lbl = arrayfun(@(i) sprintf('%s%d->%s%d', ...
            channel_names{1}, vT.Ch1_Index(i), ...
            channel_names{2}, vT.Ch2_Index(i)), ...
            (1:height(vT))', 'UniformOutput', false);

        n = height(vT);
        c2c_um      = vT.Distance_um;
        pdist_um_v  = vT.PDist_um;
        angle_deg   = vT.Angle_Btw;
        tang_um     = sqrt(max(0, c2c_um.^2 - pdist_um_v.^2));

        % Match xcorr rows to vector rows by pair label
        sark_xc  = nan(n,1);  sark_gr = nan(n,1);  sark_hms = nan(n,1);
        hyb_xc   = nan(n,1);  hyb_gr  = nan(n,1);  hyb_hms  = nan(n,1);

        for i = 1:n
            si = find(strcmp(sT.Pair, pair_lbl{i}), 1);
            hi = find(strcmp(hT.Pair, pair_lbl{i}), 1);
            if ~isempty(si)
                sark_xc(i)  = sT.XCorr_Offset_um(si);
                sark_gr(i)  = sT.GCorr_Peak(si);
                sark_hms(i) = sT.HMS_offset_nm(si);
            end
            if ~isempty(hi)
                hyb_xc(i)  = hT.XCorr_Offset_um(hi);
                hyb_gr(i)  = hT.GCorr_Peak(hi);
                hyb_hms(i) = hT.HMS_offset_nm(hi);
            end
        end

        pdist_c2c_ratio  = pdist_um_v  ./ max(c2c_um, eps);
        sark_pdist_ratio = sark_xc     ./ max(pdist_um_v, eps);
        hyb_pdist_ratio  = hyb_xc      ./ max(pdist_um_v, eps);

        out_T = table(pair_lbl, c2c_um, pdist_um_v, angle_deg, tang_um, ...
            sark_xc, sark_gr, sark_hms, ...
            hyb_xc,  hyb_gr,  hyb_hms, ...
            pdist_c2c_ratio, sark_pdist_ratio, hyb_pdist_ratio, ...
            'VariableNames', { ...
                'Pair', 'C2C_um', 'PDist_um', 'Angle_Btw_deg', 'Tangential_um', ...
                'Sarkar_XCorr_um', 'Sarkar_GCorrPeak', 'Sarkar_HMS_offset_nm', ...
                'Hybrid_XCorr_um', 'Hybrid_GCorrPeak', 'Hybrid_HMS_offset_nm', ...
                'PDist_C2C_ratio', 'Sarkar_PDist_ratio', 'Hybrid_PDist_ratio'});

        out_path = fullfile(objects_folder, ...
            sprintf('%s_to_%s_distance_summary.csv', channel_names{1}, channel_names{2}));
        writetable(out_T, out_path);
        fprintf('\nUnified distance summary: %s\n', out_path);
    else
        warning('Distance summary skipped — one or more input CSVs not found.');
    end
end

function [reg_ch1, reg_ch2, trans] = register_slice_drift_translation(ch1, ch2, reg_channel_idx)
% Sequential 2-D translation-only registration across Z slices.

[r, c, vol] = size(ch1);
reg_ch1 = zeros(r, c, vol, 'like', ch1);
reg_ch2 = zeros(r, c, vol, 'like', ch2);
trans   = zeros(max(vol - 1, 0), 2);

reg_ch1(:,:,1) = ch1(:,:,1);
reg_ch2(:,:,1) = ch2(:,:,1);

if vol <= 1
    return;
end

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 100;
optimizer.MaximumStepLength = 0.01;
optimizer.MinimumStepLength = 1e-5;
optimizer.RelaxationFactor  = 0.5;

fprintf('Registering slices sequentially...\n');
for k = 2:vol
    if reg_channel_idx == 1
        fixed  = reg_ch1(:,:,k-1);
        moving = ch1(:,:,k);
    else
        fixed  = reg_ch2(:,:,k-1);
        moving = ch2(:,:,k);
    end

    fprintf('  slice %d/%d\n', k, vol);
    tform = imregtform(moving, fixed, 'translation', optimizer, metric);
    trans(k-1,:) = tform.T(3,1:2);

    if k > 2
        prev_t = trans(k-2,:);
        curr_t = trans(k-1,:);
        if norm(curr_t - prev_t) > 5
            fprintf('    Large jump detected. Retrying with previous transform.\n');
            init = affine2d([1 0 0; 0 1 0; prev_t 1]);
            tform = imregtform(moving, fixed, 'translation', ...
                optimizer, metric, 'InitialTransformation', init);
            trans(k-1,:) = tform.T(3,1:2);
        end
    end

    output_ref = imref2d(size(fixed));
    reg_ch1(:,:,k) = imwarp(ch1(:,:,k), tform, 'OutputView', output_ref);
    reg_ch2(:,:,k) = imwarp(ch2(:,:,k), tform, 'OutputView', output_ref);
end
end

function write_volume_tiff(vol, out_path)
for z = 1:size(vol,3)
    mode = 'overwrite';
    if z > 1
        mode = 'append';
    end
    imwrite(vol(:,:,z), out_path, 'tif', 'WriteMode', mode);
end
end

function create_registration_qc_figure(ch1_before, ch2_before, ch1_after, ch2_after, channel_names, objects_folder)
slab_size = 15;
num_slices = size(ch1_before, 3);
slab_start = max(1, floor((num_slices - slab_size) / 2) + 1);
slab_end   = min(num_slices, slab_start + slab_size - 1);

fig = figure('Name','Registration QC', 'Position',[100 100 1600 900], ...
    'ToolBar', 'none', 'MenuBar', 'none');
tiledlayout(3,4,'TileSpacing','compact','Padding','compact');

nexttile;
imshow(build_rgb_mip(ch1_before, ch2_before));
title(sprintf('Before MIP (%s/%s)', channel_names{1}, channel_names{2}));

nexttile;
imshow(build_rgb_mip(ch1_after, ch2_after));
title(sprintf('After MIP (%s/%s)', channel_names{1}, channel_names{2}));

nexttile;
imshow(build_rgb_slab_mip(ch1_before, ch2_before, slab_start, slab_end));
title(sprintf('Before Slab MIP (Z %d-%d)', slab_start, slab_end));

nexttile;
imshow(build_rgb_slab_mip(ch1_after, ch2_after, slab_start, slab_end));
title(sprintf('After Slab MIP (Z %d-%d)', slab_start, slab_end));

nexttile;
imshow(build_ortho_rgb(ch1_before, ch2_before, 'xy'));
title('Before XY Mid-slice');

nexttile;
imshow(build_ortho_rgb(ch1_after, ch2_after, 'xy'));
title('After XY Mid-slice');

nexttile;
imshow(build_ortho_rgb(ch1_before, ch2_before, 'xz'));
title('Before XZ Mid-slice');

nexttile;
imshow(build_ortho_rgb(ch1_after, ch2_after, 'xz'));
title('After XZ Mid-slice');

nexttile;
imshow(build_ortho_rgb(ch1_before, ch2_before, 'yz'));
title('Before YZ Mid-slice');

nexttile;
imshow(build_ortho_rgb(ch1_after, ch2_after, 'yz'));
title('After YZ Mid-slice');

nexttile([1 2]);
show_volume_outline_overlay(ch1_after, ch2_after);
title('Registered Volume Overview');

exportgraphics(fig, fullfile(objects_folder, 'registration_qc_overview.png'));
close(fig);
end

function create_registration_metrics_figure(ch1_before, ch2_before, ch1_after, ch2_after, ...
    drift_transforms, reg_channel_idx, channel_names, objects_folder)
if reg_channel_idx == 1
    reg_before = ch1_before;
    reg_after  = ch1_after;
else
    reg_before = ch2_before;
    reg_after  = ch2_after;
end

num_slices = size(reg_before, 3);
pair_idx = (2:num_slices)';

if isempty(drift_transforms)
    cumulative_shift = zeros(1, 2);
    step_shift = zeros(1, 2);
else
    cumulative_shift = [0 0; cumsum(drift_transforms, 1)];
    step_shift = drift_transforms;
end

rmse_before = nan(max(num_slices - 1, 1), 1);
rmse_after  = nan(max(num_slices - 1, 1), 1);
for k = 2:num_slices
    rmse_before(k-1) = compute_slice_pair_rmse(reg_before(:,:,k), reg_before(:,:,k-1));
    rmse_after(k-1)  = compute_slice_pair_rmse(reg_after(:,:,k), reg_after(:,:,k-1));
end

fig = figure('Name','Registration Metrics', 'Position',[100 100 1500 850], ...
    'ToolBar', 'none', 'MenuBar', 'none');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
if size(cumulative_shift,1) > 1
    quiver(cumulative_shift(1:end-1,1), cumulative_shift(1:end-1,2), ...
        step_shift(:,1), step_shift(:,2), 0, 'LineWidth', 1.2, ...
        'Color', [0 0.4470 0.7410], 'MaxHeadSize', 0.6);
    hold on;
    plot(cumulative_shift(:,1), cumulative_shift(:,2), '-o', ...
        'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.1, 'MarkerSize', 4);
    text(cumulative_shift(1,1), cumulative_shift(1,2), '  Z1', 'FontSize', 9, ...
        'VerticalAlignment', 'bottom');
    text(cumulative_shift(end,1), cumulative_shift(end,2), ...
        sprintf('  Z%d', size(cumulative_shift,1)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    hold off;
else
    plot(0, 0, 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
end
axis image;
grid on;
xlabel('X shift (pixels)');
ylabel('Y shift (pixels)');
title(sprintf('Displacement Vector Map (%s-driven)', channel_names{reg_channel_idx}));

nexttile;
imagesc([zeros(1,2); drift_transforms]);
colormap(gca, parula);
cb = colorbar;
cb.Label.String = 'Translation (pixels)';
yticks(1:num_slices);
yticklabels(arrayfun(@(v) sprintf('Z%d', v), 1:num_slices, 'UniformOutput', false));
xticks([1 2]);
xticklabels({'dX','dY'});
xlabel('Component');
ylabel('Slice');
title('Per-slice Translation Components');

nexttile;
if num_slices > 1
    plot(pair_idx, rmse_before(1:num_slices-1), '-o', 'LineWidth', 1.2, ...
        'DisplayName', 'Before');
    hold on;
    plot(pair_idx, rmse_after(1:num_slices-1), '-o', 'LineWidth', 1.2, ...
        'DisplayName', 'After');
    hold off;
    legend('Location','best');
else
    plot(1, 0, 'o');
end
grid on;
xlabel('Slice');
ylabel('RMSE (intensity)');
title(sprintf('Slice-pair RMSE in %s channel', channel_names{reg_channel_idx}));

nexttile;
if num_slices > 1
    improvement = rmse_before(1:num_slices-1) - rmse_after(1:num_slices-1);
    bar(pair_idx, improvement, 'FaceColor', [0.4660 0.6740 0.1880]);
    yline(0, '--k');
    grid on;
    xlabel('Slice');
    ylabel('\DeltaRMSE (Before - After)');
    title(sprintf('Mean improvement = %.4g', mean(improvement, 'omitnan')));
else
    axis off;
    text(0.5, 0.5, 'Single-slice stack: no RMSE comparison available.', ...
        'HorizontalAlignment', 'center');
end

metrics_png = fullfile(objects_folder, 'registration_displacement_rmse.png');
metrics_fig = fullfile(objects_folder, 'registration_displacement_rmse.fig');
exportgraphics(fig, metrics_png);
savefig(fig, metrics_fig);
close(fig);
end

function val = compute_slice_pair_rmse(img_a, img_b)
diff_img = double(img_a) - double(img_b);
val = sqrt(mean(diff_img(:).^2, 'omitnan'));
end

function rgb = build_rgb_mip(ch1, ch2)
rgb = cat(3, ...
    normalize_for_display(max(ch1,[],3), 0.0005) * 0.95, ...
    normalize_for_display(max(ch2,[],3), 0.0005) * 0.85, ...
    zeros(size(ch1,1), size(ch1,2)));
end

function rgb = build_rgb_slab_mip(ch1, ch2, z_start, z_end)
rgb = cat(3, ...
    normalize_for_display(max(ch1(:,:,z_start:z_end),[],3), 0.0005) * 0.95, ...
    normalize_for_display(max(ch2(:,:,z_start:z_end),[],3), 0.0005) * 0.85, ...
    zeros(size(ch1,1), size(ch1,2)));
end

function rgb = build_ortho_rgb(ch1, ch2, plane_name)
mid_y = round(size(ch1,1) / 2);
mid_x = round(size(ch1,2) / 2);
mid_z = round(size(ch1,3) / 2);

switch lower(plane_name)
    case 'xy'
        a = ch1(:,:,mid_z);
        b = ch2(:,:,mid_z);
    case 'xz'
        a = format_ortho_plane_for_display(squeeze(ch1(mid_y,:,:)));
        b = format_ortho_plane_for_display(squeeze(ch2(mid_y,:,:)));
    case 'yz'
        a = format_ortho_plane_for_display(squeeze(ch1(:,mid_x,:)));
        b = format_ortho_plane_for_display(squeeze(ch2(:,mid_x,:)));
    otherwise
        error('Unknown orthogonal plane: %s', plane_name);
end

rgb = cat(3, ...
    normalize_for_display(a, 0.0005) * 0.95, ...
    normalize_for_display(b, 0.0005) * 0.85, ...
    zeros(size(a)));
end

function out = normalize_for_display(img, clip_limit)
img = double(img);
if max(img(:)) <= 0
    out = zeros(size(img));
    return;
end
out = adapthisteq(mat2gray(img), 'ClipLimit', clip_limit);
end

function show_volume_outline_overlay(ch1, ch2)
mid_y = round(size(ch1,1) / 2);
mid_x = round(size(ch1,2) / 2);
mid_z = round(size(ch1,3) / 2);

xy = build_rgb_mip(ch1, ch2);

imshow(xy);
hold on;
xline(mid_x, 'c-', 'LineWidth', 1);
yline(mid_y, 'c-', 'LineWidth', 1);
text(10, 20, sprintf('XY @ Z=%d', mid_z), 'Color', 'w', 'FontWeight', 'bold');
hold off;
end

function out = format_ortho_plane_for_display(img)
img = img';

if isempty(img)
    out = img;
    return;
end

target_height = max(size(img,1) * 12, 180);
out = imresize(img, [target_height, size(img,2)], 'nearest');
end

function out = normalize_percentile_for_display(img, low_pct, high_pct)
img = double(img);
if max(img(:)) <= 0
    out = zeros(size(img));
    return;
end

lo = prctile(img(:), low_pct);
hi = prctile(img(:), high_pct);
if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
    out = mat2gray(img);
    return;
end

img = min(max(img, lo), hi);
out = mat2gray(img, [lo hi]);
out = adapthisteq(out, 'ClipLimit', 0.0005);
end
