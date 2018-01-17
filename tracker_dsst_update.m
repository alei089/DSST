function [state, rect, values] = tracker_dsst_update(state, im, varargin)

target_sz = state.target_sz;
pos = state.pos;
sz = state.sz;
currentScaleFactor = state.currentScaleFactor;
cos_window = state.cos_window;
hf_num = state.hf_num;
hf_den = state.hf_den;
currentScaleFactor = state.currentScaleFactor;
base_target_sz = state.base_target_sz;
scaleFactors = state.scaleFactors;
scale_window = state.scale_window;
scale_model_sz = state.scale_model_sz;
sf_num = state.sf_num;
sf_den = state.sf_den;
lambda = state.lambda;
yf = state.yf;
ysf = state.ysf;
min_scale_factor = state.min_scale_factor;
max_scale_factor = state.max_scale_factor;
learning_rate = state.learning_rate;

xt = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
% calculate the correlation response of the translation filter
xtf = fft2(xt);
response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));

% find the maximum translation response
[row, col] = find(response == max(response(:)), 1);

% update the position
pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);

% extract the test sample feature map for the scale filter
xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

% calculate the correlation response of the scale filter
xsf = fft(xs,[],2);
scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));

% find the maximum scale response
recovered_scale = find(scale_response == max(scale_response(:)), 1);

xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
% update the scale
currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
if currentScaleFactor < min_scale_factor
    currentScaleFactor = min_scale_factor;
elseif currentScaleFactor > max_scale_factor
    currentScaleFactor = max_scale_factor;
end
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

state.hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
state.hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
state.sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
state.sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;

% calculate the new target size
state.target_sz = floor(base_target_sz * currentScaleFactor);
state.base_target_sz = base_target_sz;
state.currentScaleFactor = currentScaleFactor;
state.lambda = lambda;
state.pos = pos;
state.cos_window = cos_window;
state.sz = sz;
rect= [pos([2,1]) - state.target_sz([2,1])/2, state.target_sz([2,1])];
state.scaleFactors = scaleFactors;

state.scale_window = scale_window;
state.scale_model_sz = scale_model_sz;
end
