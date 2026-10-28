function [yHat, deltaPredNorm, yMeas, logVarNorm] = decode_lstm_residual_output(netOut, measInput, normStats, domain)
% Decode residual-style LSTM output.
% domain='normalized'  : measInput is normalized measurement channels XNorm(3:8,:,:)
% domain='real'        : measInput is real measurement in physical units

if nargin < 4 || isempty(domain)
    domain = 'real';
end

% After extractdata(forward(...)), MATLAB may materialize sequence outputs as
% [C x B x T] while measurement tensors are still [C x T x B].
if ~isa(netOut, 'dlarray') && ~isa(measInput, 'dlarray') && ndims(netOut) == 3 && ndims(measInput) == 3
    if size(netOut, 2) == size(measInput, 3) && size(netOut, 3) == size(measInput, 2)
        netOut = permute(netOut, [1, 3, 2]);
    end
end

deltaPredNorm = netOut(1:6, :, :);
logVarNorm = max(min(netOut(7:12, :, :), 5), -8);

shape = ones(1, max(ndims(deltaPredNorm), 2));
shape(1) = 6;
muDelta = reshape(getResidualMean(normStats), shape);
sigDelta = reshape(getResidualScale(normStats), shape);

switch lower(domain)
    case {'normalized', 'norm'}
        if ~isfield(normStats, 'measMap')
            error('normStats.measMap is required for normalized residual decoding.');
        end
        scale = reshape(normStats.measMap.scale, shape);
        bias = reshape(normStats.measMap.bias, shape);
        yMeas = measInput .* scale + bias;
        if isfield(normStats, 'deltaMap')
            deltaScale = reshape(normStats.deltaMap.scale, shape);
            deltaBias = reshape(normStats.deltaMap.bias, shape);
            yHat = yMeas + deltaBias + deltaScale .* deltaPredNorm;
        else
            yHat = yMeas + deltaPredNorm;
        end

    case {'real', 'measurement'}
        yMeas = measInput;
        yHat = yMeas + muDelta + sigDelta .* deltaPredNorm;

    otherwise
        error('Unsupported decode domain: %s', domain);
end
end

function muDelta = getResidualMean(normStats)
if isfield(normStats, 'muDelta')
    muDelta = normStats.muDelta;
else
    muDelta = zeros(6, 1, 'like', normStats.sigY);
end
end

function sigDelta = getResidualScale(normStats)
if isfield(normStats, 'sigDelta')
    sigDelta = normStats.sigDelta;
else
    sigDelta = normStats.sigY;
end
sigDelta = max(sigDelta, 1e-6);
end
