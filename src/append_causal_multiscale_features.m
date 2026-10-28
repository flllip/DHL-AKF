function xMs = append_causal_multiscale_features(xBase, windows)
% Append causal moving-average views for multi-scale temporal encoding.
% xBase: [C x T] or [C x T x B]
% xMs  : [C*numel(scales) x T] / [C*numel(scales) x T x B]

if nargin < 2 || isempty(windows)
    windows = [1, 3, 7, 15];
end

if all(windows ~= 1)
    windows = [1, windows(:)'];
else
    windows = windows(:)';
end

if ndims(xBase) == 2
    xBase = reshape(xBase, size(xBase, 1), size(xBase, 2), 1);
    squeezeBack = true;
else
    squeezeBack = false;
end

[C, T, B] = size(xBase);
xMs = zeros(C * numel(windows), T, B, 'like', xBase);

for wi = 1:numel(windows)
    w = windows(wi);
    xAvg = zeros(C, T, B, 'like', xBase);
    for b = 1:B
        for t = 1:T
            t0 = max(1, t - w + 1);
            xAvg(:, t, b) = mean(xBase(:, t0:t, b), 2);
        end
    end
    rowIdx = (wi - 1) * C + (1:C);
    xMs(rowIdx, :, :) = xAvg;
end

if squeezeBack
    xMs = xMs(:, :, 1);
end
end
