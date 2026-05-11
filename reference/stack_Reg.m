[fileNames, pathName] = uigetfile({'*.tif', 'TIFF Files'; '*.*', 'All Files'},'Select images for analysis', 'MultiSelect', 'on');

if ~iscell(fileNames)
    fileNames = {fileNames};
end

num_ch = 3;

for file_idx = 1:length(fileNames)
    % Load Volumetric stack image
    image_path = fullfile(pathName, fileNames{file_idx});
    info = imfinfo(image_path);
    num_slices = numel(info)/num_ch;
    
    if num_ch > 1  % Multichannel image%
        if file_idx == 1
        channel_list = arrayfun(@(x) sprintf('Channel %d', x), 1:num_ch, 'UniformOutput', false);
        [selected_indices, tf] = listdlg('ListString', channel_list, 'SelectionMode', 'multiple', 'PromptString', 'Select channels to analyze:', 'ListSize', [300 150], 'Name', 'Channel Selection');
               
          % Get channel names from user
          num_selected = length(selected_indices);
          
          for ch_idx = 1:num_selected
              channel_num = selected_indices(ch_idx);
              for k = 1:num_slices
                % Frame index: (slice-1) * num_ch + channel 
                frame_idx = (k - 1) * num_ch + channel_num;
                image(:,:,k) = imread(image_path, frame_idx);
              end
            
            if ch_idx == 1
                ch1 = image;
            elseif ch_idx == 2
                ch2 = image;
            elseif ch_idx == 3
                ch3 = image;
            else
            end    
          end   
        end
    end
end    

[r,c,vol] = size(ch2);

%% Start drift correction
regvol1 = zeros(r, c, vol, 'like', ch1);
regvol2 = zeros(r, c, vol, 'like', ch2);
regvol3 = zeros(r, c, vol, 'like', ch3);
trans = zeros(vol-1,2);

regvol1(:,:,1) = ch1(:,:,1);
regvol2(:,:,1) = ch2(:,:,1);
regvol3(:,:,1) = ch3(:,:,1);

[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 100;
optimizer.MaximumStepLength = 0.01;
optimizer.MinimumStepLength = 1e-5;
optimizer.RelaxationFactor = 0.5;

% Register each slice to the previous slice
fprintf('Registering slices sequentially...\n');
fprintf('Registering--')
for k = 2:vol
    % Fixed image: previous registered slice
    fixed = regvol2(:,:,k-1);
    fprintf('%0.0d--',k) % Display progress
    % Moving image: current slice
    moving2 = ch2(:,:,k); % Reference Channel
    moving1 = ch1(:,:,k);
    moving3 = ch3(:,:,k);
    
    % Compute translation-only transformation
    tform = imregtform(moving2, fixed, 'translation', optimizer, metric);
    trans(k-1,:) = tform.T(3,1:2);
    if k>2
       pret = trans(k-2,:);
       curt = trans(k-1,:);
       jump = norm (curt-pret);
       if jump > 5
           fprintf('\nLarge jump detected\n')
           % Repeat registration guided by previous transform
           init = affine2d([1 0 0; 0 1 0; pret 1]);
           tform = imregtform(moving2,fixed,'translation',optimizer,metric,'InitialTransformation',init);
           trans(k-1,:) = tform.T(3,1:2);
       end
    end   
    % Apply transformation from Reference channel registration to all
    % channels
    regvol2(:,:,k) = imwarp(moving2, tform, 'OutputView', imref2d(size(fixed)));
    regvol1(:,:,k) = imwarp(moving1, tform, 'OutputView', imref2d(size(fixed)));
    regvol3(:,:,k) = imwarp(moving3, tform, 'OutputView', imref2d(size(fixed)));

    
end
%%
fprintf('Registration complete!\n'); 
        % Create maximum projection for displaying result
        ch2f = adapthisteq(max(regvol2,[],3),'ClipLimit',0.0005)*25;
        ch3f = adapthisteq(max(regvol3,[],3),'ClipLimit',0.0001)*50;
        ch1f = max(regvol1,[],3)/10;
        ch2r = adapthisteq(max(ch2,[],3),'ClipLimit',0.0005)*25;
        ch3r = adapthisteq(max(ch3,[],3),'ClipLimit',0.0001)*50;
        ch1r = max(ch1,[],3)/10;
        merged = cat(3,ch3f,ch2f,ch1f);  
        orig = cat(3,ch3r,ch2r,ch1r);

for o = 1:vol
    if o == 1
      % Write the first frame and create the file
      imwrite(regvol1(:, :, o), 'reg_ch1.tif', 'tif', 'WriteMode', 'overwrite');
      imwrite(regvol2(:, :, o), 'reg_ch2.tif', 'tif', 'WriteMode', 'overwrite');        
      imwrite(regvol3(:, :, o), 'reg_ch3.tif', 'tif', 'WriteMode', 'overwrite'); 
    else 
      imwrite(regvol1(:, :, o), 'reg_ch1.tif', 'tif', 'WriteMode', 'append');
      imwrite(regvol2(:, :, o), 'reg_ch2.tif', 'tif', 'WriteMode', 'append');        
      imwrite(regvol3(:, :, o), 'reg_ch3.tif', 'tif', 'WriteMode', 'append'); 
    end
end


figure;
subplot(1,2,1); imshow(orig); title('Original');
subplot(1,2,2); imshow(merged); title('Registered');