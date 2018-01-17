function [state, rect, values] = tracker_dsst_initialize(im, region, varargin)

target_sz = [region(4), region(3)];
pos = [region(2), region(1)] + floor(target_sz/2);
params.init_pos = floor(pos) + floor(target_sz/2);

params.wsize = floor(target_sz);

%parameters according to the paper
params.padding = 1.0;         			% extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
params.learning_rate = 0.025;			% tracking model learning rate (denoted "eta" in the paper)
params.number_of_scales = 33;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.02;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples

params.visualization = 1;

% parameters
padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
lambda = params.lambda;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_max_area = params.scale_model_max_area;


init_target_sz = target_sz;
% target size att scale = 1
base_target_sz = target_sz;

% window size, taking padding into account
sz = floor(base_target_sz * (1 + padding));

% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
output_sigma = sqrt(prod(base_target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));



% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));

% store pre-computed translation filter cosine window
cos_window = single(hann(sz(1)) * hann(sz(2))');
% store pre-computed scale filter cosine window
if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));
end;

% scale factors
ss = 1:nScales;
scaleFactors = scale_step.^(ceil(nScales/2) - ss);

% compute the resize dimensions used for feature extraction in the scale
% estimation
scale_model_factor = 1;
if prod(init_target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
end
scale_model_sz = floor(init_target_sz * scale_model_factor);

currentScaleFactor = 1;

min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));


xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
% calculate the translation filter update
xlf = fft2(xl);
new_hf_num = bsxfun(@times, yf, conj(xlf));
new_hf_den = sum(xlf .* conj(xlf), 3);

% extract the training sample feature map for the scale filter
xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

% calculate the scale filter update
xsf = fft(xs,[],2);
new_sf_num = bsxfun(@times, ysf, conj(xsf));
new_sf_den = sum(xsf .* conj(xsf), 1);

% first frame, train with a single image
state.hf_den = new_hf_den;
state.hf_num = new_hf_num;

state.sf_den = new_sf_den;
state.sf_num = new_sf_num;

% calculate the new target size
state.target_sz = floor(base_target_sz * currentScaleFactor);
state.base_target_sz = base_target_sz;
state.currentScaleFactor = currentScaleFactor;
state.lambda = lambda;
state.pos = pos;
state.cos_window = cos_window;
state.sz = sz;
rect = floor(region);
state.scaleFactors = scaleFactors;
state.scale_window = scale_window;
state.scale_model_sz = scale_model_sz;
state.min_scale_factor = min_scale_factor;
state.max_scale_factor = max_scale_factor;
state.yf = yf;
state.ysf = ysf;
state.learning_rate = learning_rate;
end
