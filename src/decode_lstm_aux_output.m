function [muDenoise, muSubst, varAux, pAbn, pSubst, logVarNorm, airAux] = decode_lstm_aux_output(netOut, normStats, domain, measInput)
% Decode auxiliary-measurement network output.
% Output layout:
%   1:6   mu_denoise / delta_denoise
%   7:12  mu_subst   / delta_subst
%   13:18 logvar_aux
%   19    p_abn logit
%   20    p_subst logit
%   21:23 air-data auxiliary head [VT, alpha, beta] if available
% A 17-channel single-head model omits 7:12; later fields shift by six.

if nargin < 3 || isempty(domain)
    domain = 'real';
end
if nargin < 4
    measInput = [];
end

muDNorm = netOut(1:6, :, :);
if size(netOut, 1) == 17
    muSNorm = muDNorm;
    offset = -6;
else
    muSNorm = netOut(7:12, :, :);
    offset = 0;
end
logVarNorm = max(min(netOut(13+offset:18+offset, :, :), 5), -6);
pAbn = sigmoid(netOut(19+offset, :, :));
pSubst = sigmoid(netOut(20+offset, :, :));

if size(netOut, 1) >= 17
    airNorm = netOut(21+offset:23+offset, :, :);
else
    airNorm = zeros(3, size(netOut, 2), size(netOut, 3), 'like', netOut);
end

shape6 = ones(1, max(ndims(muDNorm), 2));
shape6(1) = 6;
muY = reshape(normStats.muY, shape6);
sigY = reshape(normStats.sigY, shape6);
varLower = reshape(normStats.varLower, shape6);
varUpper = reshape(normStats.varUpper, shape6);

shape3 = ones(1, max(ndims(airNorm), 2));
shape3(1) = 3;
hasAirStats = isfield(normStats, 'muAir') && isfield(normStats, 'sigAir');
if hasAirStats
    muAir = reshape(normStats.muAir, shape3);
    sigAir = reshape(normStats.sigAir, shape3);
end

useResidualMode = isfield(normStats, 'muDelta') && isfield(normStats, 'sigDelta') && ~isempty(measInput);
if useResidualMode
    muDelta = reshape(normStats.muDelta, shape6);
    sigDelta = reshape(normStats.sigDelta, shape6);
    measInput = cast(measInput, 'like', muDNorm);
    deltaDReal = muDelta + sigDelta .* muDNorm;
    deltaSReal = muDelta + sigDelta .* muSNorm;
    muDReal = measInput + deltaDReal;
    muSReal = measInput + deltaSReal;
end

switch lower(domain)
    case {'normalized', 'norm'}
        if useResidualMode
            muDenoise = (muDReal - muY) ./ sigY;
            muSubst = (muSReal - muY) ./ sigY;
        else
            muDenoise = muDNorm;
            muSubst = muSNorm;
        end
        varAux = max(min(exp(logVarNorm), varUpper ./ (sigY .^ 2)), varLower ./ (sigY .^ 2));
        if hasAirStats
            airAux = airNorm;
        else
            airAux = zeros(size(airNorm), 'like', airNorm);
        end

    case {'real', 'measurement'}
        if useResidualMode
            muDenoise = muDReal;
            muSubst = muSReal;
        else
            muDenoise = muY + sigY .* muDNorm;
            muSubst = muY + sigY .* muSNorm;
        end
        varAux = max(min(exp(logVarNorm) .* (sigY .^ 2), varUpper), varLower);
        if hasAirStats
            airAux = muAir + sigAir .* airNorm;
        else
            airAux = zeros(size(airNorm), 'like', airNorm);
        end

    otherwise
        error('Unsupported decode domain: %s', domain);
end
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
