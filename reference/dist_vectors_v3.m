function [results_table, pair_info] = dist_vectors_v3(pathName, channel_names, distance_threshold, scale_input_expanded)
% DIST_VECTORS_V3 - Computes vectors between centroids from different channels
%
% Inputs:
%   pathName           - Path to directory containing channel CSV files
%   channel_names      - Cell array of channel names (e.g., {'DAPI', 'GFP'})
%   distance_threshold - Maximum distance in microns (default: 2.7)
%   scale_input        - Cell array: {xy_scale_string, z_scale_string} in µm/pixel
%
% Outputs:
%   results_table - Table containing pair information and vectors
%   pair_info     - Structure with detailed pair information

    % Set default threshold if not provided
    if nargin < 3 || isempty(distance_threshold)
        distance_threshold = 2.7;
    end
    
    % Validate inputs
    if length(channel_names) < 2
        error('At least 2 channel names are required for comparison.');
    end
    
    fprintf('\n=== Computing Centroid Vectors ===\n');
    fprintf('Channels: %s to %s\n', channel_names{1}, channel_names{2});
    
    if nargin < 4 || isempty(scale_input_expanded)
        fprintf('No scaling factors provided. Using pixel units.\n');
        xy_scale = 1;
        z_scale = 1;
    else
        xy_scale = str2double(scale_input_expanded{1}); % µm/pixel
        z_scale = str2double(scale_input_expanded{2});   % µm/pixel
    end
    
    fprintf('XY scale: %.4f µm/pixel\n', xy_scale);
    fprintf('Z scale: %.4f µm/slice\n', z_scale);
    fprintf('Distance threshold: %.1f µm\n', distance_threshold);

    %% Read CSV data for both channels
    ch1_filename = sprintf('%s_analysis_results.csv', channel_names{1}); % Target
    ch2_filename = sprintf('%s_analysis_results.csv', channel_names{2}); % Reference

    % Use pathName passed from caller (portable)
    ch1_path = fullfile(pathName, ch1_filename);
    ch2_path = fullfile(pathName, ch2_filename);

    % Check if files exist
    if ~isfile(ch1_path)
        error('Channel 1 CSV file not found: %s', ch1_path);
    end
    if ~isfile(ch2_path)
        error('Channel 2 CSV file not found: %s', ch2_path);
    end
    
    fprintf('Reading channel 1: %s\n', ch1_filename);
    % Detect header import options and preserve original header name
    opts1 = detectImportOptions(ch1_path,'filetype','text','readvariablenames',1,'Range',1);
    opts1.VariableNamingRule = 'preserve';
    ch1_data = readtable(ch1_path, opts1);
    

    fprintf('Reading channel 2: %s\n', ch2_filename);
    opts2 = detectImportOptions(ch2_path);
    opts2.VariableNamingRule = 'preserve';
    ch2_data = readtable(ch2_path, opts2);

    obj1_id = ch1_data.Object;
    obj2_id = ch2_data.Object;
    centroids_ch1_px = [ch1_data.X, ch1_data.Y, ch1_data.Z]; % Target   
    centroids_ch2_px = [ch2_data.X, ch2_data.Y, ch2_data.Z]; % Reference
    paxis = [ch2_data.EigVX, ch2_data.EigVY, ch2_data.EigVZ];
    major_axis_len = ch2_data.MajorAxis;  % Renamed from 'eval' to avoid shadowing built-in

    %% Build object correspondence mask (vectorized)
    % Each entry mask(i,j) is true if obj2_id(i) == obj1_id(j)
    mask = (obj2_id == obj1_id');
    fprintf('Object correspondence mask: %d x %d\n', size(mask,1), size(mask,2));

    %% Convert to microns (scaled coordinates)
    scale_vec = [xy_scale, xy_scale, z_scale];
    centroids_ch1_um = centroids_ch1_px .* scale_vec;
    centroids_ch2_um = centroids_ch2_px .* scale_vec;
    rcent1_px = round(centroids_ch1_px);
    rcent2_px = round(centroids_ch2_px);
    rcent1_um = rcent1_px .* scale_vec;
    rcent2_um = rcent2_px .* scale_vec;
    
    fprintf('Channel 1: %d centroids\n', size(centroids_ch1_px, 1));
    fprintf('Channel 2: %d centroids\n', size(centroids_ch2_px, 1));

    ref_num = size(centroids_ch2_um, 1);
    drop = 0;
    
    if ref_num == 0
        fprintf('No centroid pairs found within %.2f µm.\n', distance_threshold);
        results_table = [];
        pair_info = [];
        return;
    end

    %% Pre-allocate to upper bound (ref_num * max possible targets), trim later
    max_pairs = ref_num * size(centroids_ch1_px, 1);  % absolute upper bound
    alloc_size = min(max_pairs, 10000);  % reasonable initial allocation

    vectors_px = zeros(alloc_size, 3);
    vectors_um = zeros(alloc_size, 3);
    vector_magnitudes_px = zeros(alloc_size, 1);
    vector_magnitudes_um = zeros(alloc_size, 1);
    synsep = zeros(alloc_size, 1);
    angle_vals = zeros(alloc_size, 1);
    
    pair_info = struct();
    pair_info.ch1_index = zeros(alloc_size, 1);
    pair_info.ch2_index = zeros(alloc_size, 1);
    pair_info.ch1_centroid_px = zeros(alloc_size, 3);
    pair_info.ch2_centroid_px = zeros(alloc_size, 3);
    pair_info.ch1_centroid_um = zeros(alloc_size, 3);
    pair_info.ch2_centroid_um = zeros(alloc_size, 3);
    pair_info.ch_image = cell(alloc_size, 1);
    pair_info.distance_threshold = distance_threshold;
    pair_info.xy_scale = xy_scale;
    pair_info.z_scale = z_scale;
    
    j = 0;  % Valid pair counter — only incremented on successful storage

    %% Axis unit vectors for angle computation
    xvec = [1 0 0];
    yvec = [0 1 0];
    zvec = [0 0 1];

    %% Iterate through Reference Channel objects
    for i = 1:ref_num
        x = centroids_ch2_um(i,1);
        y = centroids_ch2_um(i,2);
        z = centroids_ch2_um(i,3);
        
        % Bounding box in microns for proximity search
        xT = x + distance_threshold;
        yT = y + distance_threshold;
        zT = z + distance_threshold;
        xt = x - distance_threshold;
        yt = y - distance_threshold;
        zt = z - distance_threshold;
        
        % Normalize principal axis eigenvector to guarantee unit length
        axis_raw = paxis(i,:);
        axis_norm = norm(axis_raw);
        if axis_norm < eps
            fprintf('\nSkipping ref%d: degenerate eigenvector\n', obj2_id(i));
            continue;
        end
        axis_vec = axis_raw / axis_norm;  % Unit eigenvector
        
        major_len_i = major_axis_len(i);
        
        fprintf('\nO ref%d', obj2_id(i));
        
        % Find Ch1 objects within bounding-box distance of current Ch2 reference
        ch1_ind = find(rcent1_um(:,1) < xT & rcent1_um(:,1) > xt & ...
                       rcent1_um(:,2) < yT & rcent1_um(:,2) > yt & ...
                       rcent1_um(:,3) < zT & rcent1_um(:,3) > zt);
        
        ch1_num = length(ch1_ind);
        ch2_ind = i;
        
        if ch1_num < 1
            continue;  % No nearby targets
        end
        
        % 5% margin of major axis length
        m = major_len_i * 0.05;
        L = (major_len_i / 2) - m;
        fprintf(' L:%.3f', L);
        
        % Process each candidate Ch1 object
        for a = 1:ch1_num
            target_idx = ch1_ind(a);
            
            % Separation vector in pixel space
            pos2_ck = centroids_ch2_px(ch2_ind, :);
            pos1_ck = centroids_ch1_px(target_idx, :);
            vec_ck = pos1_ck - pos2_ck;
            uvec = norm(vec_ck);
            
            if uvec < eps
                continue;  % Identical centroids, skip
            end
            
            % Projection of separation vector onto principal axis
            proj = dot(vec_ck, axis_vec);
            
            % Angle between centroid-centroid vector and principal axis
            % Clamp to [-1,1] to prevent complex results from floating-point error
            cos_angle = proj / uvec;  % axis_vec is unit, so no division by norm needed
            cos_angle = max(-1, min(1, cos_angle));
            th_bet = acosd(cos_angle);
            
            % Perpendicular distance with ANISOTROPIC scaling
            % Residual vector in pixels, then scale each component properly
            residual_px = vec_ck - (proj * axis_vec);
            residual_um = residual_px .* scale_vec;
            pdist_val = norm(residual_um);
            
            % Validity check: matching objects, projection within axis bounds,
            % and angle not near-parallel (2°–178° range)
            if mask(i, target_idx) == 1 && abs(proj) < L && th_bet > 2 && th_bet < 178
                j = j + 1;
                
                % Grow arrays if needed
                if j > size(vectors_px, 1)
                    grow = max(1000, j);
                    vectors_px(end+1:end+grow, :) = 0;
                    vectors_um(end+1:end+grow, :) = 0;
                    vector_magnitudes_px(end+1:end+grow) = 0;
                    vector_magnitudes_um(end+1:end+grow) = 0;
                    synsep(end+1:end+grow) = 0;
                    angle_vals(end+1:end+grow) = 0;
                    pair_info.ch1_index(end+1:end+grow) = 0;
                    pair_info.ch2_index(end+1:end+grow) = 0;
                    pair_info.ch1_centroid_px(end+1:end+grow, :) = 0;
                    pair_info.ch2_centroid_px(end+1:end+grow, :) = 0;
                    pair_info.ch1_centroid_um(end+1:end+grow, :) = 0;
                    pair_info.ch2_centroid_um(end+1:end+grow, :) = 0;
                    pair_info.ch_image{end+grow} = [];
                end
                
                pair_info.ch1_index(j) = obj1_id(target_idx);
                pair_info.ch2_index(j) = obj2_id(ch2_ind);
                angle_vals(j) = th_bet;
                
                % Positions in pixels
                pos1_px = centroids_ch1_px(target_idx, :);
                pos2_px = centroids_ch2_px(ch2_ind, :);
                
                % Positions in microns
                pos1_um = centroids_ch1_um(target_idx, :);
                pos2_um = centroids_ch2_um(ch2_ind, :);
                
                % Compute separation vectors
                vectors_px(j, :) = pos2_px - pos1_px;
                vectors_um(j, :) = pos2_um - pos1_um;
                vector_magnitudes_px(j) = norm(vectors_px(j,:));
                vector_magnitudes_um(j) = norm(vectors_um(j,:));
                synsep(j) = pdist_val;
                
                pair_info.ch1_centroid_px(j, :) = pos1_px;
                pair_info.ch2_centroid_px(j, :) = pos2_px;
                pair_info.ch1_centroid_um(j, :) = pos1_um;
                pair_info.ch2_centroid_um(j, :) = pos2_um;
                pair_info.ch_image{j} = ch2_data.Image{ch2_ind};
                
                fprintf(' :::%d-%d~o', obj1_id(target_idx), j);
            else
                drop = drop + 1;
            end
        end
    end

    %% Trim arrays to actual pair count
    if j == 0
        fprintf('\nNo valid centroid pairs found within %.2f µm.\n', distance_threshold);
        results_table = [];
        pair_info = [];
        return;
    end

    vectors_px = vectors_px(1:j, :);
    vectors_um = vectors_um(1:j, :);
    vector_magnitudes_px = vector_magnitudes_px(1:j);
    vector_magnitudes_um = vector_magnitudes_um(1:j);
    synsep = synsep(1:j);
    angle_vals = angle_vals(1:j);
    pair_info.ch1_index = pair_info.ch1_index(1:j);
    pair_info.ch2_index = pair_info.ch2_index(1:j);
    pair_info.ch1_centroid_px = pair_info.ch1_centroid_px(1:j, :);
    pair_info.ch2_centroid_px = pair_info.ch2_centroid_px(1:j, :);
    pair_info.ch1_centroid_um = pair_info.ch1_centroid_um(1:j, :);
    pair_info.ch2_centroid_um = pair_info.ch2_centroid_um(1:j, :);
    pair_info.ch_image = pair_info.ch_image(1:j);

    fprintf('\n\nDropped pairs (failed validity): %d\n', drop);
    fprintf('Valid pairs retained: %d\n', j);

    %% Build results table
    results_table = table(pair_info.ch1_index, pair_info.ch2_index, ...
                          pair_info.ch_image, ...
                          pair_info.ch1_centroid_px(:,1), pair_info.ch1_centroid_px(:,2), pair_info.ch1_centroid_px(:,3), ...
                          pair_info.ch2_centroid_px(:,1), pair_info.ch2_centroid_px(:,2), pair_info.ch2_centroid_px(:,3), ...
                          pair_info.ch1_centroid_um(:,1), pair_info.ch1_centroid_um(:,2), pair_info.ch1_centroid_um(:,3), ...
                          pair_info.ch2_centroid_um(:,1), pair_info.ch2_centroid_um(:,2), pair_info.ch2_centroid_um(:,3), ...
                          vectors_px(:,1), vectors_px(:,2), vectors_px(:,3), ...
                          vectors_um(:,1), vectors_um(:,2), vectors_um(:,3), ...
                          angle_vals, vector_magnitudes_px, vector_magnitudes_um, synsep, ...
                          'VariableNames', {'Ch1_Index', 'Ch2_Index', ...
                                           'Image', ...
                                           'Ch1_X_px', 'Ch1_Y_px', 'Ch1_Z_px', ...
                                           'Ch2_X_px', 'Ch2_Y_px', 'Ch2_Z_px', ...
                                           'Ch1_X_um', 'Ch1_Y_um', 'Ch1_Z_um', ...
                                           'Ch2_X_um', 'Ch2_Y_um', 'Ch2_Z_um', ...
                                           'Vector_dX_px', 'Vector_dY_px', 'Vector_dZ_px', ...
                                           'Vector_dX_um', 'Vector_dY_um', 'Vector_dZ_um', ...
                                           'Angle_Btw', 'Distance_px', 'Distance_um', 'PDist_um'});
    
    %% Save results (portable path)
    output_filename = sprintf('%s_to_%s_vectors.csv', channel_names{1}, channel_names{2});
    output_path = fullfile(pathName, output_filename);
    writetable(results_table, output_path);
    fprintf('Results saved to: %s\n', output_filename);
    
    %% Display summary statistics (computed on trimmed data only)
    fprintf('\n=== Vector Analysis Summary ===\n');
    fprintf('Total valid pairs: %d\n', j);
    fprintf('Mean distance: %.2f µm (%.2f px)\n', mean(vector_magnitudes_um), mean(vector_magnitudes_px));
    fprintf('Std distance: %.2f µm (%.2f px)\n', std(vector_magnitudes_um), std(vector_magnitudes_px));
    fprintf('Min distance: %.2f µm (%.2f px)\n', min(vector_magnitudes_um), min(vector_magnitudes_px));
    fprintf('Max distance: %.2f µm (%.2f px)\n', max(vector_magnitudes_um), max(vector_magnitudes_px));
    fprintf('Mean vector (µm): [%.2f, %.2f, %.2f]\n', mean(vectors_um(:,1)), mean(vectors_um(:,2)), mean(vectors_um(:,3)));
    fprintf('Mean vector (px): [%.2f, %.2f, %.2f]\n', mean(vectors_px(:,1)), mean(vectors_px(:,2)), mean(vectors_px(:,3)));
    fprintf('Mean perpendicular distance: %.2f µm\n', mean(synsep));
    fprintf('Mean angle of separation: %.2f°\n', mean(angle_vals));

end
