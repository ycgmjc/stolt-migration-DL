% =========================================================================
% ptTargets_fk_multi_acq.m
%
% This script is a modified version of the F-k migration benchmark script
% from BentoBox (https://gitlab.com/mj66/bentobox).
%
% Copyright 2022, Marko Jakovljevic & Louise Zhuang
% Licensed under the Apache License, Version 2.0
%
% Modifications:
% - Integrated `fk_multi_benchmark` to intercept intermediate frequency loops.
% - Added 4D reshaping and cache packaging to export training matrices 
%   for PyTorch deep learning pipelines.
% =========================================================================


% set paths
clear all
addpath(genpath('../'))

%%
%%%%%%%%%%%%%%%%%%%%%
% GENERATE THE DATA %
%%%%%%%%%%%%%%%%%%%%%

run genFieldIIData

%%
% DAS

ax_span = time*c/2;
foc_spacing = ax_span(2) - ax_span(1);
range_start = ax_span(1);
range_end = ax_span(end);
range_axes = range_start:foc_spacing:range_end;
num_foci = length(range_axes);

x_size = 20e-3;
nlines_x = 256;
rx_x = (linspace(0,1,nlines_x)-0.5)*x_size;
dc_rx = 0;
dc_tx = 0;

beams_mono = zeros(num_foci,nlines_x);
beams_multi = zeros(num_foci,nlines_x);

beam_file = fullfile('cached_data/beams_cached.mat');
if isfile(beam_file)
    % ============================================================
    % LOAD EVERYTHING FROM CACHE
    % ============================================================
    S = load(beam_file);

    beams_multi   = S.beams_multi;
    beams_mono    = S.beams_mono;

    % raw multistatic data + parameters
    full_sch_data = S.full_sch_data;   % T x N x M
    sch_data_mono = S.sch_data_mono;
    start_times   = S.start_times;
    tshift        = S.tshift;
    fs            = S.fs;
    f0            = S.f0;
    rxAptPos      = S.rxAptPos;
    txAptPos      = S.txAptPos;
    time          = S.time;
    c             = S.c;
    N_elements    = S.N_elements;

else

    % tic
    for ii=1:nlines_x
    
        beam_origin = [rx_x(ii) 0 0];
        beam_direction = [0 0 1];
        foc_pts = make_beam(range_start,range_end,beam_direction,beam_origin,foc_spacing);
    
        % multistatic
        foc_data = focus_fs(time', full_sch_data, foc_pts, rxAptPos, txAptPos, dc_rx, dc_tx, c);
        beams_multi(:,ii) = double(squeeze(sum(sum(foc_data,3),2)));
    
        % monostatic
        foc_tmp = zeros(num_foci,N_elements);
        for idnum = 1:N_elements        
            foc_tmp(:,idnum) = focus_data(time',sch_data_mono(:,idnum),foc_pts,rxAptPos(idnum,:),txAptPos(idnum,:),dc_rx, c);
        end
        beams_mono(:,ii) = squeeze(sum(foc_tmp,2));
        
        display(['Focused line # ' num2str(ii)])
    end

    % Save DAS beams + RAW RF + parameters
    save(beam_file, ...
         'beams_multi','beams_mono', ...
         'full_sch_data','sch_data_mono', ...
         'start_times','tshift', ...
         'fs','f0','rxAptPos','txAptPos', ...
         'time','c','N_elements', ...
         '-v7');
end
    % toc

%%
% F-k migration

start_time_stolt = start_times - tshift;
f_params.fs = fs;
f_params.f0 = f0;


%   foc_data_fk : final F-k image   (Nz x Nx)
%   t, x_data     : axes
%   in_b        : per-ku inputs     (H x W x U)
%   out_b       : per-ku outputs    (H x W x U)
[foc_data_fk, t, x_data, in_b, out_b, raw_b] = fk_multi_benchmark(full_sch_data, f_params, rxAptPos, start_time_stolt, c);
% [foc_data_fk, t, x_data]= fk_multi(full_sch_data, f_params, rxAptPos, start_time_stolt, c);
% [foc_data_fk, t, x_data]= fk_multi_benchmark(full_sch_data, f_params, rxAptPos, start_time_stolt, c);

raw_file = fullfile('cached_data/raw_input_fk.mat');
raw_b = single(raw_b);
[H, W, U] = size(raw_b);
raw_data  = reshape(raw_b,  [1, H, W, U]);  % B=1
fprintf('raw_b size:  H=%d, W=%d, U=%d\n', H, W, U);
save(raw_file, 'raw_data', '-v7');

ax_data = t*c/2;
train_file = fullfile('cached_data/training_cache_fk.mat');
if ~isfile(train_file)
    scale = max(abs(out_b(:)));
    if scale == 0
        scale = 1;  
    end
    
    %in_b  = in_b  / scale;
    %out_b = out_b / scale;
    
    fprintf('After scaling, in_b abs min/max:  %g  %g\n', ...
            min(abs(in_b(:))), max(abs(in_b(:))));
    fprintf('After scaling, out_b abs min/max: %g  %g\n', ...
            min(abs(out_b(:))), max(abs(out_b(:))));

    in_b = single(in_b);
    out_b = single(out_b);
    [H, W, U] = size(in_b);
    inputs  = reshape(in_b,  [1, H, W, U]);  % B=1
    outputs = reshape(out_b, [1, H, W, U]);  % B=1

    [H, W, U] = size(in_b);
    fprintf('in_b size:  H=%d, W=%d, U=%d\n', H, W, U);

    S = whos('inputs');
    disp(S.size);  % should print: 1   H   W   U
    fprintf('in_b abs min/max:  %g  %g\n', min(abs(in_b(:))), max(abs(in_b(:))));
    fprintf('out_b abs min/max: %g  %g\n', min(abs(out_b(:))), max(abs(out_b(:))));

    

    save(train_file, 'inputs', 'outputs', 'scale', '-v7');
    disp('[TRAIN CACHE] Saved');
    
else
    %{
    S = load(train_file);

    inputs  = S.inputs;   % B x H x W x U
    outputs = S.outputs;

    % Convert new tensors to single
    in_b  = single(in_b);
    out_b = single(out_b);

    % Check size consistency
    [H_old, W_old, U_old] = size(inputs, 2:4);
    [H, W, U] = size(in_b);

    if H ~= H_old || W ~= W_old || U ~= U_old
        error('New sample size does not match existing cache size.');
    end

    % Append along B
    inputs  = cat(1, inputs,  reshape(in_b,  [1, H, W, U]));
    outputs = cat(1, outputs, reshape(out_b, [1, H, W, U]));

    save(train_file, 'inputs', 'outputs', '-v7.3');
    disp('[TRAIN CACHE] Appended new sample.');
    %}
end

%%
%%%%%%%%%%%%%%%%%%
% B-mode images %%
%%%%%%%%%%%%%%%%%%

env_fk = abs(foc_data_fk);
env_fk = env_fk/max(env_fk(:));

% upsample the final images
% F-k
ncols = size(env_fk,2);
original_spacing = linspace(0,1,ncols);
upsampled_spacing = linspace(0,1,ncols*4);
env_fk_interp = (interp1(original_spacing,env_fk',upsampled_spacing))';
x_us_fk = (interp1(original_spacing,x_data',upsampled_spacing))';

%%
%%%%%%%%%%%%%%%%%%%%
%   VISUALIZATION %%
%%%%%%%%%%%%%%%%%%%%

clim = [-80 0];
figure('PaperPositionMode', 'auto');
s1 = subplot(1,1,1);
imagesc(x_us_fk, ax_data, db(env_fk));
axis image; axis off;
colormap(gray); caxis(clim)
title('F-k multi')
set(gca,'FontSize',10)

%save('cached_data/fk_vis_vars.mat', 'x_data', 'ax_data');

