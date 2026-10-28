function feat = compose_stage2_meta_features(baseFeat, lambda1, pAbn, pSubst, logVarNorm, muDenoise, muSubst, measurements, featureCfg, metaState)
% Compose the stage-2 meta-feature used by SVR and gate models.

baseFeat = reshape(double(baseFeat), 34, []);
lambda1 = reshape(double(lambda1), 1, []);
pAbn = reshape(double(pAbn), 1, []);
pSubst = reshape(double(pSubst), 1, []);
logVarNorm = reshape(double(logVarNorm), 6, []);
muDenoise = reshape(double(muDenoise), 6, []);
muSubst = reshape(double(muSubst), 6, []);
measurements = reshape(double(measurements), 6, []);

if nargin < 9 || isempty(featureCfg)
    featureCfg = stage2_meta_feature_spec("legacy");
else
    featureCfg = stage2_meta_feature_spec(featureCfg);
end
if nargin < 10 || isempty(metaState)
    metaState = struct();
end

deltaD = muDenoise - measurements;
deltaS = muSubst - measurements;
feat = [baseFeat; lambda1; pAbn; pSubst; logVarNorm; deltaD; deltaS];

if featureCfg.extraDim <= 0
    return;
end

innovation = baseFeat(29:34, :);
ctxInnovation = localResolveMetaValue(metaState, 'ctxInnovation', sqrt(mean(innovation .^ 2, 1)), size(feat, 2));
prevMode = localResolveMetaValue(metaState, 'prevMode', zeros(1, size(feat, 2)), size(feat, 2));
prevLambda1 = localResolveMetaValue(metaState, 'prevLambda1', zeros(1, size(feat, 2)), size(feat, 2));
fallbackStreak = localResolveMetaValue(metaState, 'fallbackStreak', zeros(1, size(feat, 2)), size(feat, 2));
timeSinceLastAnomalyEnd = localResolveMetaValue(metaState, 'timeSinceLastAnomalyEnd', zeros(1, size(feat, 2)), size(feat, 2));
timeSinceLastAnomalyEnd = min(max(timeSinceLastAnomalyEnd, 0), featureCfg.timeSinceCapSec);

feat = [feat; prevMode; prevLambda1; fallbackStreak; timeSinceLastAnomalyEnd; ctxInnovation];
end

function value = localResolveMetaValue(metaState, fieldName, defaultValue, count)
value = defaultValue;
if isstruct(metaState) && isfield(metaState, fieldName)
    value = metaState.(fieldName);
end
value = double(value);
if isscalar(value)
    value = repmat(value, 1, count);
else
    value = reshape(value, 1, []);
    if numel(value) ~= count
        error('Meta feature field %s size mismatch.', fieldName);
    end
end
end
