function xcorr3D_analysis(pathName, channel_names, scale_input, labeled_images_map, raw_images_map, distance_nm, cap_factor)
% XCORR3D_ANALYSIS  3-D cross-correlation between every reference-target
%                   object pair found in dist_vectors_v3 results.
%
% ── Compatibility ────────────────────────────────────────────────────────
%   Drop-in replacement.  Called from Synapse_Analysis_v3_xcorr.m with
%   identical 5 arguments — no changes required in the calling script.
%
%   Optional arguments:
%     distance_nm  – [min_nm, max_nm] XC peak search annulus.
%                    Default: auto-estimated from vectors CSV PDist_um.
%     cap_factor   – intensity-cap divisor in sarkar_preprocess.
%                    Sarkar original = 6.  Default = 6.
%
% ── Dual-pipeline mode ───────────────────────────────────────────────────
%   Two cross-correlation pipelines are run per pair and saved separately:
%
%   SARKAR pipeline  (outputs suffixed _sarkar)
%     Mask     : sarkar_preprocess  (full-volume convn cross-kernel,
%                adaptive threshold, spherical convolution)
%     Intensity: Sarkar A3/B3 — background-subtracted, raw-restored,
%                intensity-capped at max/cap_factor
%
%   HYBRID pipeline  (outputs suffixed _hybrid)
%     Mask     : synapse_threshold_m3  (slice-by-slice imfilter cross-kernel,
%                same adaptive threshold, spherical convolution)
%                → tighter boundary, sub-cluster-preserving
%     Intensity: Sarkar A3/B3 recomputed inside the M3 mask boundary
%                → Sarkar background subtraction and cap applied, but
%                  spatial support limited to M3-segmented voxels
%
% ── Outputs (written to pathName) ────────────────────────────────────────
%   Sarkar : *_xcorr_full_sarkar.csv  | *_xcorr_summary_sarkar.csv
%            *_xcorr_rawdata_sarkar.csv
%            *_xcorr_figure_sarkar.fig/.png
%            *_xcorr_segcheck_sarkar.png
%
%   Hybrid : *_xcorr_full_hybrid.csv  | *_xcorr_summary_hybrid.csv
%            *_xcorr_rawdata_hybrid.csv
%            *_xcorr_figure_hybrid.fig/.png
%            *_xcorr_segcheck_hybrid.png

% ── 0.  Arguments ────────────────────────────────────────────────────────
    if nargin < 3 || isempty(scale_input)
        xy_scale = 1;   z_scale = 1;
    else
        xy_scale = str2double(scale_input{1});
        z_scale  = str2double(scale_input{2});
    end
    if nargin < 6 || isempty(distance_nm);  distance_nm = [];  end
    if nargin < 7 || isempty(cap_factor);   cap_factor  = 6;   end

    pixel_nm = xy_scale * 1000;
    ch1_name = channel_names{1};
    ch2_name = channel_names{2};

    fprintf('\n=== 3-D XCorr Analysis — Dual Pipeline (Sarkar + Hybrid) ===\n');
    fprintf('Reference : %s   Target : %s\n', ch2_name, ch1_name);
    fprintf('XY scale  : %.4f um/px  (%.2f nm/px)   Z scale : %.4f um/px\n', ...
        xy_scale, pixel_nm, z_scale);
    fprintf('Cap factor: %g\n', cap_factor);

% ── 1.  Load vector-analysis results ─────────────────────────────────────
    vectors_csv = fullfile(pathName, ...
        sprintf('%s_to_%s_vectors.csv', ch1_name, ch2_name));
    if ~isfile(vectors_csv)
        error('xcorr3D_analysis: vectors file not found:\n  %s', vectors_csv);
    end
    vT      = readtable(vectors_csv);
    n_pairs = height(vT);
    fprintf('Loaded %d valid pairs.\n', n_pairs);
    if n_pairs == 0; warning('xcorr3D_analysis: no pairs to process.'); return; end

    if isempty(distance_nm)
        med_nm      = median(vT.PDist_um) * 1000;
        distance_nm = [max(10, med_nm * 0.3), med_nm * 2.5 + 50];
        fprintf('Auto distance constraint: [%.0f, %.0f] nm\n', ...
            distance_nm(1), distance_nm(2));
    end

% ── 2.  Pull labeled & raw volumes ───────────────────────────────────────
    for cn = {ch1_name, ch2_name}
        if ~isKey(labeled_images_map, cn{1})
            error('xcorr3D_analysis: labeled_images_map missing key "%s".', cn{1}); end
        if ~isKey(raw_images_map, cn{1})
            error('xcorr3D_analysis: raw_images_map missing key "%s".', cn{1}); end
    end
    lbl_ch1 = labeled_images_map(ch1_name);
    lbl_ch2 = labeled_images_map(ch2_name);
    raw_ch1 = double(raw_images_map(ch1_name));
    raw_ch2 = double(raw_images_map(ch2_name));

% ── 3.  Pre-allocate result arrays — one struct per pipeline ──────────────
    pair_label  = cell(n_pairs, 1);
    c2c_dist_um = nan(n_pairs, 1);
    pdist_um    = nan(n_pairs, 1);

    S = alloc_results(n_pairs);   % Sarkar
    H = alloc_results(n_pairs);   % Hybrid

% ── 4.  rmax ──────────────────────────────────────────────────────────────
    rmax = max(8, round(120 / pixel_nm));
    fprintf('rmax = %d px  (%.0f nm)\n', rmax, rmax * pixel_nm);

% ── 5.  Figures ───────────────────────────────────────────────────────────
    fig_h_px  = min(0.95, 0.20 * n_pairs + 0.06);
    qc_h_px   = min(0.95, 0.18 * n_pairs + 0.06);
    N_CORR    = 4;   % XY | XZ | YZ | g(r)
    N_QC      = 6;   % raw | mask | capped  ×2 channels

    [fig_s,  tlo_s]   = make_figure(sprintf('XCorr Sarkar: %s→%s',ch1_name,ch2_name), ...
        fig_h_px, n_pairs, N_CORR, ...
        sprintf('Sarkar G(r)  |  %s (target) vs %s (reference)',ch1_name,ch2_name));

    [fig_hf, tlo_hf]  = make_figure(sprintf('XCorr Hybrid: %s→%s',ch1_name,ch2_name), ...
        fig_h_px, n_pairs, N_CORR, ...
        sprintf('Hybrid G(r) [M3 mask + Sarkar intensity]  |  %s vs %s',ch1_name,ch2_name));

    [fig_qcs, tlo_qcs] = make_figure(sprintf('QC Sarkar: %s/%s',ch1_name,ch2_name), ...
        qc_h_px, n_pairs, N_QC, ...
        sprintf('Sarkar preprocessing QC  |  cap=%g  |  %s / %s',cap_factor,ch1_name,ch2_name));

    [fig_qch, tlo_qch] = make_figure(sprintf('QC Hybrid: %s/%s',ch1_name,ch2_name), ...
        qc_h_px, n_pairs, N_QC, ...
        sprintf('Hybrid (M3 mask) QC  |  cap=%g  |  %s / %s',cap_factor,ch1_name,ch2_name));

% ── 6.  Per-pair loop ─────────────────────────────────────────────────────
    for p = 1:n_pairs

        ch1_id    = vT.Ch1_Index(p);
        ch2_id    = vT.Ch2_Index(p);
        label_str = sprintf('%s%d->%s%d', ch1_name, ch1_id, ch2_name, ch2_id);
        pair_label{p}  = label_str;
        c2c_dist_um(p) = vT.Distance_um(p);
        pdist_um(p)    = vT.PDist_um(p);

        fprintf('\n[%d/%d]  %s\n', p, n_pairs, label_str);

        % Label masks
        mask1_lbl = (lbl_ch1 == ch1_id);
        mask2_lbl = (lbl_ch2 == ch2_id);

        if ~any(mask1_lbl(:)) || ~any(mask2_lbl(:))
            warning('  Object label not found — skipping pair.');
            nan_blk  = build_nan_block(label_str);
            nan_raw  = build_nan_raw_block(label_str, rmax);
            S.full_blocks{p} = nan_blk;  S.raw_blocks{p} = nan_raw;
            H.full_blocks{p} = nan_blk;  H.raw_blocks{p} = nan_raw;
            fill_skip_tiles(tlo_s,   N_CORR, label_str);
            fill_skip_tiles(tlo_hf,  N_CORR, label_str);
            fill_skip_tiles(tlo_qcs, N_QC,   label_str);
            fill_skip_tiles(tlo_qch, N_QC,   label_str);
            continue;
        end

        % Sub-volume extraction
        bb1 = get_bbox(mask1_lbl);  bb2 = get_bbox(mask2_lbl);
        PAD = 5;
        rlo = max(1,               min(bb1(1),bb2(1))-PAD);
        rhi = min(size(lbl_ch1,1), max(bb1(2),bb2(2))+PAD);
        clo = max(1,               min(bb1(3),bb2(3))-PAD);
        chi = min(size(lbl_ch1,2), max(bb1(4),bb2(4))+PAD);
        zlo = max(1,               min(bb1(5),bb2(5))-PAD);
        zhi = min(size(lbl_ch1,3), max(bb1(6),bb2(6))+PAD);

        sub1_raw = raw_ch1(rlo:rhi, clo:chi, zlo:zhi);
        sub2_raw = raw_ch2(rlo:rhi, clo:chi, zlo:zhi);

        % ── Sarkar preprocessing ──────────────────────────────────────────
        [sA2, sA3, s_maskA, sNa, sVa] = sarkar_preprocess(sub1_raw, cap_factor);
        [sB2, sB3, s_maskB, sNb, sVb] = sarkar_preprocess(sub2_raw, cap_factor);

        % ── Hybrid: M3 masks + Sarkar intensity ───────────────────────────
        % Pre-filter with sigma=0.5 before M3 (noise robustness for PMT data)
        m3_mA = double(imfill(bwareaopen( ...
            synapse_threshold_m3(imgaussfilt3(double(sub1_raw),0.5)),1,26),'holes'));
        m3_mB = double(imfill(bwareaopen( ...
            synapse_threshold_m3(imgaussfilt3(double(sub2_raw),0.5)),1,26),'holes'));

        % Apply M3 mask to Sarkar background-subtracted intensities
        hA2 = sA2 .* m3_mA;
        hB2 = sB2 .* m3_mB;
        % Intensity cap on hybrid masked volumes
        cap_a = max(hA2(:))/cap_factor; if cap_a<eps; cap_a=1; end
        cap_b = max(hB2(:))/cap_factor; if cap_b<eps; cap_b=1; end
        hA3 = min(hA2, cap_a);
        hB3 = min(hB2, cap_b);
        % Veatch normalisation scalars
        hNa = sum(hA2(:).*m3_mA(:)); if hNa<eps; hNa=eps; end
        hVa = sum(hA3(:));           if hVa<eps; hVa=eps; end
        hNb = sum(hB2(:).*m3_mB(:)); if hNb<eps; hNb=eps; end
        hVb = sum(hB3(:));           if hVb<eps; hVb=eps; end

        % ── QC figures ────────────────────────────────────────────────────
        raw1_mip = max(sub1_raw,[],3);
        raw2_mip = max(sub2_raw,[],3);

        plot_qc_row(tlo_qcs, raw1_mip, max(s_maskA,[],3), max(sA3,[],3), ...
            ch1_name, label_str, cap_factor);
        plot_qc_row(tlo_qcs, raw2_mip, max(s_maskB,[],3), max(sB3,[],3), ...
            ch2_name, '', cap_factor);

        plot_qc_row(tlo_qch, raw1_mip, max(m3_mA,[],3), max(hA3,[],3), ...
            ch1_name, label_str, cap_factor);
        plot_qc_row(tlo_qch, raw2_mip, max(m3_mB,[],3), max(hB3,[],3), ...
            ch2_name, '', cap_factor);

        drawnow;

        % ── Run Sarkar pipeline ───────────────────────────────────────────
        [s_res, s_full, s_raw] = run_xcorr_pipeline( ...
            sA2,sA3,s_maskA,sNa,sVa, sB2,sB3,s_maskB,sNb,sVb, ...
            rmax, pixel_nm, xy_scale, z_scale, distance_nm, ...
            label_str, vT, p, cap_factor, 'Sarkar');

        S = store_results(S, p, s_res, s_full, s_raw);
        plot_corr_row(tlo_s, s_res, xy_scale, z_scale, label_str, ...
            ch1_name, ch2_name, rmax);

        % ── Run Hybrid pipeline ───────────────────────────────────────────
        [h_res, h_full, h_raw] = run_xcorr_pipeline( ...
            hA2,hA3,m3_mA,hNa,hVa, hB2,hB3,m3_mB,hNb,hVb, ...
            rmax, pixel_nm, xy_scale, z_scale, distance_nm, ...
            label_str, vT, p, cap_factor, 'Hybrid');

        H = store_results(H, p, h_res, h_full, h_raw);
        plot_corr_row(tlo_hf, h_res, xy_scale, z_scale, label_str, ...
            ch1_name, ch2_name, rmax);

        fprintf('  [Sarkar] XCorr=%.3f um  G(r0)=%.3f  HMS_offset=%.1f nm\n', ...
            S.xcorr_offset_um(p), S.gcorr_peak(p), S.hms_offset_nm(p));
        fprintf('  [Hybrid] XCorr=%.3f um  G(r0)=%.3f  HMS_offset=%.1f nm\n', ...
            H.xcorr_offset_um(p), H.gcorr_peak(p), H.hms_offset_nm(p));

        drawnow;
    end  % pair loop

% ── 7.  Save figures ──────────────────────────────────────────────────────
    base = @(tag) fullfile(pathName, ...
        sprintf('%s_to_%s_xcorr_%s', ch1_name, ch2_name, tag));

    save_fig(fig_s,   base('figure_sarkar'));
    save_fig(fig_hf,  base('figure_hybrid'));
    exportgraphics(set_visible(fig_qcs), [base('segcheck_sarkar') '.png'], 'Resolution',200);
    exportgraphics(set_visible(fig_qch), [base('segcheck_hybrid') '.png'], 'Resolution',200);
    fprintf('\nAll figures saved.\n');

% ── 8.  Write CSVs ────────────────────────────────────────────────────────
    write_full_csv( [base('full_sarkar')    '.csv'], S.full_blocks);
    write_raw_csv(  [base('rawdata_sarkar') '.csv'], S.raw_blocks);
    write_summary(  [base('summary_sarkar') '.csv'], pair_label, c2c_dist_um, pdist_um, S, 'Sarkar');

    write_full_csv( [base('full_hybrid')    '.csv'], H.full_blocks);
    write_raw_csv(  [base('rawdata_hybrid') '.csv'], H.raw_blocks);
    write_summary(  [base('summary_hybrid') '.csv'], pair_label, c2c_dist_um, pdist_um, H, 'Hybrid');

    fprintf('\n=== XCorr Analysis Complete ===\n');
    fprintf('Sarkar outputs: *_xcorr_*_sarkar.*\n');
    fprintf('Hybrid outputs: *_xcorr_*_hybrid.*\n');
end


% ═══════════════════════════════════════════════════════════════════════════
%  PIPELINE RUNNER
% ═══════════════════════════════════════════════════════════════════════════

function [res, full_block, raw_block] = run_xcorr_pipeline( ...
        A2, A3, maskA, Na, Va, B2, B3, maskB, Nb, Vb, ...
        rmax, pixel_nm, xy_scale, z_scale, distance_nm, ...
        label_str, vT, p, cap_factor, tag)

    pad_sz  = size(A2) + rmax;

    NP      = real(fftshift(ifftn( fftn(A3,pad_sz).*conj(fftn(B3,pad_sz)) )));
    G1_full = (Va*Vb/Na/Nb) .* real(fftshift(ifftn( ...
                  fftn(A2.*maskA,pad_sz).*conj(fftn(B2.*maskB,pad_sz)) )));
    NPmask  = max(0, max(NP(:))/4 - NP);
    G1_full = G1_full ./ max(NP, eps);
    G1_full(NPmask > 0) = 0;

    L0 = size(G1_full);
    L  = floor(L0/2 + 1);

    % Distance-constrained peak search
    G0 = real(fftshift(ifftn( ...
             fftn(A3.*maskA,size(A3)+rmax).*conj(fftn(B3.*maskB,size(B3)+rmax)) )));
    G0(NPmask > 0) = 0;

    [gx,gy,gz] = ndgrid(1:size(G0,1), 1:size(G0,2), 1:size(G0,3));
    Lv = floor(size(G0)/2+1);
    d_ctr = sqrt(((gx-Lv(1))*pixel_nm).^2 + ...
                 ((gy-Lv(2))*pixel_nm).^2 + ...
                 ((gz-Lv(3))*pixel_nm).^2);
    G_search = G0 .* (d_ctr >= distance_nm(1) & d_ctr <= distance_nm(2));
    Gm = max(G_search(:));

    if Gm <= 0
        warning('  [%s] No XC peak in annulus [%.0f %.0f] nm — global max.', ...
            tag, distance_nm(1), distance_nm(2));
        [~,pk] = max(G1_full(:));
        [mx,my,mz] = ind2sub(L0, pk);
    else
        [mx,my,mz] = ind2sub(size(G_search), find(G_search==Gm,1));
    end

    xyzshift = [mx-L(1), my-L(2), mz-L(3)];
    mxyz     = floor(xyzshift + L0/2 + 1);

    lag_r_px = xyzshift(1);  lag_r_um = lag_r_px * xy_scale;
    lag_c_px = xyzshift(2);  lag_c_um = lag_c_px * xy_scale;
    lag_z_px = xyzshift(3);  lag_z_um = lag_z_px * z_scale;
    xcorr_offset_px = sqrt(lag_r_px^2+lag_c_px^2+lag_z_px^2);
    xcorr_offset_um = sqrt(lag_r_um^2+lag_c_um^2+lag_z_um^2);

    % Radial g(r) profiles
    g_cross = radial_profile_gr(G1_full, mxyz, rmax);

    NP_aa = real(fftshift(ifftn(fftn(A3,pad_sz).*conj(fftn(A3,pad_sz)))));
    G_aa  = (Va^2/Na^2) .* real(fftshift(ifftn( ...
                fftn(A2.*maskA,pad_sz).*conj(fftn(A2.*maskA,pad_sz)))));
    NPm_aa = max(0, max(NP_aa(:))/4 - NP_aa);
    G_aa  = G_aa ./ max(NP_aa,eps);  G_aa(NPm_aa>0) = 0;
    g_auto1 = radial_profile_gr(G_aa, floor(size(G_aa)/2+1), rmax);

    NP_bb = real(fftshift(ifftn(fftn(B3,pad_sz).*conj(fftn(B3,pad_sz)))));
    G_bb  = (Vb^2/Nb^2) .* real(fftshift(ifftn( ...
                fftn(B2.*maskB,pad_sz).*conj(fftn(B2.*maskB,pad_sz)))));
    NPm_bb = max(0, max(NP_bb(:))/4 - NP_bb);
    G_bb  = G_bb ./ max(NP_bb,eps);  G_bb(NPm_bb>0) = 0;
    g_auto2 = radial_profile_gr(G_bb, floor(size(G_bb)/2+1), rmax);

    r_nm   = (0:rmax) * pixel_nm;
    hms_c  = poly3_halfmax(r_nm, g_cross);
    hms_a1 = poly3_halfmax(r_nm, g_auto1);
    hms_a2 = poly3_halfmax(r_nm, g_auto2);
    hms_d  = hms_c - hms_a1;

    peak_val = G1_full(mxyz(1), mxyz(2), mxyz(3));

    % Result struct
    res.xcorr_offset_um = xcorr_offset_um;
    res.gcorr_peak      = g_cross(1);
    res.hms_corr_nm     = hms_c;
    res.hms_auto1_nm    = hms_a1;
    res.hms_auto2_nm    = hms_a2;
    res.hms_offset_nm   = hms_d;
    res.G1_full         = G1_full;
    res.L               = L;
    res.L0              = L0;
    res.mxyz            = mxyz;
    res.lag_r_um        = lag_r_um;
    res.lag_c_um        = lag_c_um;
    res.lag_z_um        = lag_z_um;
    res.r_nm            = r_nm;
    res.g_cross         = g_cross;
    res.g_auto1         = g_auto1;
    res.g_auto2         = g_auto2;
    res.peak_val        = peak_val;

    full_block = { ...
        sprintf('[%s] %s',tag,label_str), '', '',               '';
        'Metric',                 'Value (px)',    'Value (um/nm)',     'Notes';
        'Center-to-Center Dist',  vT.Distance_px(p), vT.Distance_um(p),'Euclidean centroid separation';
        'PDist (perp. dist)',      '',             vT.PDist_um(p),     'Perp dist to principal axis';
        'XCorr Lag dR',           lag_r_px,        lag_r_um,           'Row lag at G(r) peak';
        'XCorr Lag dC',           lag_c_px,        lag_c_um,           'Col lag at G(r) peak';
        'XCorr Lag dZ',           lag_z_px,        lag_z_um,           'Z lag at G(r) peak';
        'XCorr Peak Offset Dist', xcorr_offset_px, xcorr_offset_um,    'Euclidean lag offset at peak';
        'G(r) Peak Value',        '',              peak_val,           'G(r~0) Veatch-normalised';
        'HMS Cross-corr (nm)',    '',              hms_c,              'Half-max shift of cross-corr g(r)';
        'HMS Auto ch1 (nm)',      '',              hms_a1,             'Half-max of ch1 autocorr g(r)';
        'HMS Auto ch2 (nm)',      '',              hms_a2,             'Half-max of ch2 autocorr g(r)';
        'HMS Offset (nm)',        '',              hms_d,              'HMS_corr - HMS_auto1  [Sarkar metric]';
        'Cap Factor',             '',              cap_factor,         'Intensity cap divisor';
        'Mask Source',            '',              tag,                'Sarkar or Hybrid (M3 mask)';
        'Angle Between Objects',  '',              vT.Angle_Btw(p),   'Degrees';
    };

    raw_block = build_raw_block_gr( ...
        sprintf('[%s] %s',tag,label_str), r_nm, g_cross, g_auto1, g_auto2);
end


% ═══════════════════════════════════════════════════════════════════════════
%  UTILITY HELPERS
% ═══════════════════════════════════════════════════════════════════════════

function R = alloc_results(n)
    R.xcorr_offset_um = nan(n,1);
    R.gcorr_peak      = nan(n,1);
    R.hms_corr_nm     = nan(n,1);
    R.hms_auto1_nm    = nan(n,1);
    R.hms_auto2_nm    = nan(n,1);
    R.hms_offset_nm   = nan(n,1);
    R.full_blocks     = cell(n,1);
    R.raw_blocks      = cell(n,1);
end

function R = store_results(R, p, res, full_block, raw_block)
    R.xcorr_offset_um(p) = res.xcorr_offset_um;
    R.gcorr_peak(p)      = res.gcorr_peak;
    R.hms_corr_nm(p)     = res.hms_corr_nm;
    R.hms_auto1_nm(p)    = res.hms_auto1_nm;
    R.hms_auto2_nm(p)    = res.hms_auto2_nm;
    R.hms_offset_nm(p)   = res.hms_offset_nm;
    R.full_blocks{p}     = full_block;
    R.raw_blocks{p}      = raw_block;
end

function [fig, tlo] = make_figure(name, height, n_rows, n_cols, ttl)
    fig = figure('Name',name,'Units','normalized', ...
        'Position',[0.02 0.02 0.96 height],'Color','w','Visible','off');
    tlo = tiledlayout(fig, n_rows, n_cols, 'TileSpacing','compact','Padding','compact');
    title(tlo, ttl, 'FontSize',10,'FontWeight','bold');
end

function save_fig(fig, base_path)
    fig.Visible = 'on';
    savefig(fig, [base_path '.fig']);
    exportgraphics(fig, [base_path '.png'], 'Resolution',200);
    fprintf('Figure saved: %s.fig/.png\n', base_path);
end

function fig = set_visible(fig)
    fig.Visible = 'on';
end

function plot_corr_row(tlo, res, xy_scale, z_scale, label_str, ch1_name, ch2_name, rmax)
    G = res.G1_full;
    L = res.L;
    mc = min(max(res.mxyz, rmax+1), size(G)-rmax);

    lr = ((1:size(G,1))-L(1))*xy_scale;
    lc = ((1:size(G,2))-L(2))*xy_scale;
    lz = ((1:size(G,3))-L(3))*z_scale;

    % XY
    ax = nexttile(tlo);
    imagesc(ax,lc,lr,squeeze(G(:,:,mc(3))));
    axis(ax,'image'); colormap(ax,'parula'); colorbar(ax,'FontSize',6);
    hold(ax,'on'); plot(ax,res.lag_c_um,res.lag_r_um,'r+','MarkerSize',10,'LineWidth',1.5); hold(ax,'off');
    xlabel(ax,'Lag C (um)','FontSize',7); ylabel(ax,'Lag R (um)','FontSize',7);
    title(ax,sprintf('%s XY G(r0)=%.2f',label_str,res.peak_val),'FontSize',8,'Interpreter','none');
    set(ax,'FontSize',7);

    % XZ
    ax = nexttile(tlo);
    imagesc(ax,lz,lr,squeeze(G(:,mc(2),:)));
    axis(ax,'image'); colormap(ax,'parula'); colorbar(ax,'FontSize',6);
    hold(ax,'on'); plot(ax,res.lag_z_um,res.lag_r_um,'r+','MarkerSize',10,'LineWidth',1.5); hold(ax,'off');
    xlabel(ax,'Lag Z (um)','FontSize',7); ylabel(ax,'Lag R (um)','FontSize',7);
    title(ax,sprintf('%s XZ',label_str),'FontSize',8,'Interpreter','none');
    set(ax,'FontSize',7);

    % YZ
    ax = nexttile(tlo);
    imagesc(ax,lz,lc,squeeze(G(mc(1),:,:)));
    axis(ax,'image'); colormap(ax,'parula'); colorbar(ax,'FontSize',6);
    hold(ax,'on'); plot(ax,res.lag_z_um,res.lag_c_um,'r+','MarkerSize',10,'LineWidth',1.5); hold(ax,'off');
    xlabel(ax,'Lag Z (um)','FontSize',7); ylabel(ax,'Lag C (um)','FontSize',7);
    title(ax,sprintf('%s YZ',label_str),'FontSize',8,'Interpreter','none');
    set(ax,'FontSize',7);

    % g(r)
    yellow = [0.929,0.694,0.125];
    g_a1 = res.g_auto1; g_a1(isnan(g_a1)) = 0;
    g_a2 = res.g_auto2; g_a2(isnan(g_a2)) = 0;
    g_c  = res.g_cross; g_c(isnan(g_c))   = 0;
    ax = nexttile(tlo);
    plot(ax,res.r_nm,g_a1,'Color',yellow,'LineWidth',2.5); hold(ax,'on');
    plot(ax,res.r_nm,g_a2,'m','LineWidth',2.5);
    plot(ax,res.r_nm,g_c, 'k','LineWidth',2.5);
    if ~isnan(res.hms_corr_nm);  xline(ax,res.hms_corr_nm, 'k--','LineWidth',1.2); end
    if ~isnan(res.hms_auto1_nm); xline(ax,res.hms_auto1_nm,'--','Color',yellow,'LineWidth',1.2); end
    hold(ax,'off');
    xlabel(ax,'Radial lag (nm)','FontSize',7); ylabel(ax,'G(r)','FontSize',7);
    legend(ax,sprintf('%s auto',ch1_name),sprintf('%s auto',ch2_name),'Cross-corr', ...
        'FontSize',6,'Location','northeast');
    title(ax,sprintf('HMS_{corr}=%.0f nm  HMS_{offset}=%.0f nm', ...
        res.hms_corr_nm,res.hms_offset_nm),'FontSize',8,'Interpreter','tex');
    set(ax,'FontSize',7); grid(ax,'on');
end

function plot_qc_row(tlo_qc, raw_mip, mask_mip, cap_mip, ch_name, label_str, cap_factor)
    ax = nexttile(tlo_qc);
    imagesc(ax, raw_mip); axis(ax,'image','off'); colormap(ax,'gray');
    if ~isempty(label_str)
        title(ax,sprintf('%s raw\n%s',ch_name,label_str),'FontSize',7,'Interpreter','none');
    else
        title(ax,sprintf('%s raw',ch_name),'FontSize',7,'Interpreter','none');
    end

    ax = nexttile(tlo_qc);
    imagesc(ax, mask_mip); axis(ax,'image','off'); colormap(ax,'gray');
    title(ax,sprintf('%s mask',ch_name),'FontSize',7,'Interpreter','none');

    ax = nexttile(tlo_qc);
    imagesc(ax, cap_mip); axis(ax,'image','off'); colormap(ax,'hot');
    cb = colorbar(ax,'Location','eastoutside'); cb.FontSize = 6;
    title(ax,sprintf('%s capped (÷%g)',ch_name,cap_factor),'FontSize',7,'Interpreter','none');
end


% ═══════════════════════════════════════════════════════════════════════════
%  ALGORITHM HELPERS  (sarkar_preprocess, synapse_threshold_m3, g(r), HMS)
% ═══════════════════════════════════════════════════════════════════════════

function [A2, A3, maskA, Na, Va] = sarkar_preprocess(A1, cap_factor)
    if nargin < 2; cap_factor = 6; end
    A1 = double(A1);
    n = 5; k7 = zeros(n,n,n);
    for ii=1:n; for jj=1:n; for kk=1:n
        if sqrt((ii-(n+1)/2)^2+(jj-(n+1)/2)^2+(kk-(n+1)/2)^2)<=n/2
            k7(ii,jj,kk)=1; end
    end; end; end
    k7 = k7/sum(k7(:));
    k2 = [0,1,0;1,1,1;0,1,0]/5;
    bk = median(A1(:));
    
    A1 = max(0, convn(A1-bk, k2, 'same'));
    A_coarse = max(0, A1 - max(A1(:))/3);
    pv = A_coarse(A_coarse>0);
    if isempty(pv); th=0; else; th=mean(pv); end
    A2    = max(0, A1-th);
    A2    = convn(A2, k7, 'same');
    maskA = double(A2>0);
    A2    = A1.*maskA;
    cap   = max(A2(:))/cap_factor; if cap<eps; cap=1; end
    A3 = min(A2, cap);
    Na = sum(A2(:).*maskA(:)); if Na<eps; Na=eps; end
    Va = sum(A3(:));            if Va<eps; Va=eps; end
end


function [mask, th] = synapse_threshold_m3(A)
    persistent cross_k2d k7_m3
    if isempty(cross_k2d)
        % 2-D cross kernel for XY slice-by-slice smoothing
        cross_k2d = [0 1 0; 1 1 1; 0 1 0]/5;
        % 5×5×5  kernel: include voxels within radius 2.5
        [kx,ky,kz] = ndgrid(-2:2,-2:2,-2:2);
        k7r = double(sqrt(kx.^2+ky.^2+kz.^2)<=2.5);
        k7_m3 = k7r/sum(k7r(:)); % normalise to unit sum
    end
 % ── Step 1a: median subtract, clip to zero ───────────────────────────────
    A1 = max(0, A - median(A(:)));
    A1s = zeros(size(A1));
 % ── Step 1b: 2-D cross-kernel smooth, slice by slice ────────────────────
    for z = 1:size(A1,3)
        A1s(:,:,z) = imfilter(A1(:,:,z), cross_k2d, 'replicate', 'same');
    end
 % ── Step 2: coarse gate → mean-of-survivors threshold ───────────────────   
    coarse_gate = max(A1s(:))/10;
    bv = A1s(A1s>coarse_gate);
    
    if isempty(bv); mask=false(size(A)); th=coarse_gate; return; end
    % No signal above the coarse gate — return empty mas
    th = mean(bv);
    mask = convn(A1s-th, k7_m3, 'same') > 0;
end


function g = radial_profile_gr(G1, mxyz, rmax)
    g  = NaN(1,rmax+1);
    sz = size(G1);
    if any(mxyz-rmax<1) || any(mxyz+rmax>sz); return; end
    G = G1(mxyz(1)-rmax:mxyz(1)+rmax, ...
           mxyz(2)-rmax:mxyz(2)+rmax, ...
           mxyz(3)-rmax:mxyz(3)+rmax);
    [xv,yv,zv] = ndgrid(-rmax:rmax,-rmax:rmax,-rmax:rmax);
    r_grid = sqrt(xv.^2+yv.^2+zv.^2);
    Ar = reshape(r_grid,1,[]); Avals = reshape(G,1,[]);
    nz = Avals~=0; Ar=Ar(nz); Avals=Avals(nz);
    if isempty(Ar); return; end
    [rr,idx] = sort(Ar); vv = Avals(idx);
    r_axis = 0:floor(max(rr));
    [~,bin] = histc(rr, r_axis-0.5); %#ok<HISTC>
    for j = 1:rmax+1
        m = (bin==j); n2=sum(m);
        if n2>0; g(j)=sum(m.*vv)/n2; end
    end
end


function hms = poly3_halfmax(r_nm, g)
    hms = NaN;
    if isempty(g) || all(isnan(g)); return; end
    g_fit = g; g_fit(isnan(g_fit)) = 0;
    target = max(g_fit)*0.5;
    if target<=0; return; end
    rv = 0:0.1:1000;
    try
        fo = fit(r_nm(:),g_fit(:),'poly3');
        f_r = fo(rv);
        [~,ind] = min(abs(f_r-target));
        hms = rv(ind);
    catch; hms = NaN; end
end


function bb = get_bbox(mask3d)
    idx = find(mask3d);
    [r,c,z] = ind2sub(size(mask3d),idx);
    bb = [min(r) max(r) min(c) max(c) min(z) max(z)];
end


function fill_skip_tiles(tlo, ncols, label_str)
    for k = 1:ncols
        ax = nexttile(tlo); axis(ax,'off');
        text(ax,0.5,0.5,sprintf('%s\n(skipped)',label_str), ...
            'HorizontalAlignment','center','Units','normalized', ...
            'FontSize',8,'Color',[0.5 0.5 0.5]);
    end
end


function block = build_nan_block(label_str)
    block = {label_str,'','','';
             'Metric','Value (px)','Value (um/nm)','Notes';
             'Center-to-Center Dist',NaN,NaN,'Object not found';
             'PDist (perp. dist)','',NaN,'';
             'XCorr Lag dR',NaN,NaN,'';
             'XCorr Lag dC',NaN,NaN,'';
             'XCorr Lag dZ',NaN,NaN,'';
             'XCorr Peak Offset Dist',NaN,NaN,'';
             'G(r) Peak Value','',NaN,'';
             'HMS Cross-corr (nm)','',NaN,'';
             'HMS Auto ch1 (nm)','',NaN,'';
             'HMS Auto ch2 (nm)','',NaN,'';
             'HMS Offset (nm)','',NaN,'';
             'Cap Factor','',NaN,'';
             'Mask Source','','','';
             'Angle Between Objects','',NaN,''};
end


function block = build_nan_raw_block(label_str, rmax)
    n = rmax+1;
    block = [{label_str,'','','','','',''};
             {'r_nm','G_cross','G_auto1','G_auto2','','',''};
             num2cell(NaN(n,7))];
end


function block = build_raw_block_gr(label_str, r_nm, g_cross, g_auto1, g_auto2)
    block = [{label_str,'','','','','',''};
             {'r_nm','G_cross','G_auto1','G_auto2','','',''}];
    for k = 1:length(r_nm)
        block = [block; {r_nm(k),g_cross(k),g_auto1(k),g_auto2(k),'','',''}]; %#ok<AGROW>
    end
end


function write_summary(filepath, pair_label, c2c_dist_um, pdist_um, R, tag)
    try
        T = table(pair_label, c2c_dist_um, pdist_um, ...
            R.xcorr_offset_um, R.gcorr_peak, ...
            R.hms_corr_nm, R.hms_auto1_nm, R.hms_auto2_nm, R.hms_offset_nm, ...
            'VariableNames',{'Pair','CenterToCenter_um','PDist_um', ...
                'XCorr_Offset_um','GCorr_Peak', ...
                'HMS_corr_nm','HMS_auto1_nm','HMS_auto2_nm','HMS_offset_nm'});
        writetable(T, filepath);
        fprintf('Summary %-8s: %s\n', tag, filepath);
    catch ME
        warning('write_summary failed (%s): %s', tag, ME.message);
        fid = fopen(filepath,'w');
        if fid<0; return; end
        fprintf(fid,'Pair,CenterToCenter_um,PDist_um,XCorr_Offset_um,GCorr_Peak,HMS_corr_nm,HMS_auto1_nm,HMS_auto2_nm,HMS_offset_nm\n');
        for i = 1:length(pair_label)
            fprintf(fid,'%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n', ...
                pair_label{i},c2c_dist_um(i),pdist_um(i), ...
                R.xcorr_offset_um(i),R.gcorr_peak(i), ...
                R.hms_corr_nm(i),R.hms_auto1_nm(i),R.hms_auto2_nm(i),R.hms_offset_nm(i));
        end
        fclose(fid);
    end
end


function write_full_csv(filepath, blocks)
    valid = ~cellfun(@isempty, blocks);
    if ~any(valid); return; end
    bk = blocks(valid); nb = length(bk);
    nr = size(bk{1},1);  nc = nb*5-1;
    C  = repmat({''},nr,nc);
    for b = 1:nb
        cs = (b-1)*5+1; blk=bk{b};
        for r = 1:min(nr,size(blk,1))
            for c = 1:min(4,size(blk,2))
                v = blk{r,c};
                if isnumeric(v); C{r,cs+c-1}=val2str(v,'%.6g');
                else;            C{r,cs+c-1}=v; end
            end
        end
    end
    write_cell_csv(filepath, C);
    fprintf('Full CSV: %s\n', filepath);
end


function write_raw_csv(filepath, raw_blocks)
    valid = ~cellfun(@isempty, raw_blocks);
    if ~any(valid); return; end
    bk = raw_blocks(valid); nb = length(bk);
    mr = max(cellfun(@(b)size(b,1), bk));
    C  = repmat({''},mr,nb*7-1);
    for b = 1:nb
        blk=bk{b}; cs=(b-1)*7+1;
        for r = 1:size(blk,1)
            for c = 1:min(6,size(blk,2))
                v = blk{r,c};
                if isnumeric(v); C{r,cs+c-1}=val2str(v,'%.8g');
                else;            C{r,cs+c-1}=v; end
            end
        end
    end
    write_cell_csv(filepath, C);
    fprintf('Raw CSV : %s\n', filepath);
end


function write_cell_csv(filepath, C)
    fid = fopen(filepath,'w');
    if fid<0; error('xcorr3D_analysis: cannot open: %s',filepath); end
    [nr,nc] = size(C);
    for r = 1:nr
        parts = cell(1,nc);
        for c = 1:nc
            v = C{r,c};
            if isnumeric(v); v=val2str(v,'%.8g'); end
            if any(v==','); v=['"' v '"']; end
            parts{c} = v;
        end
        fprintf(fid,'%s\n', strjoin(parts,','));
    end
    fclose(fid);
end


function s = val2str(v, fmt)
    if nargin<2; fmt='%.6g'; end
    if isnan(v); s='NaN'; elseif isinf(v); s=num2str(v); else; s=num2str(v,fmt); end
end
