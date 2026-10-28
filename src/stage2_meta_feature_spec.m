function cfg = stage2_meta_feature_spec(inputArg)
% Resolve stage-2 meta-feature configuration.

base = struct( ...
    'featureSet', "legacy", ...
    'baseFeatDim', 34, ...
    'extraNames', strings(0, 1), ...
    'extraDim', 0, ...
    'featureDim', 55, ...
    'timeSinceCapSec', 5.0);

if nargin < 1 || isempty(inputArg)
    cfg = base;
    return;
end

if isstruct(inputArg)
    cfg = base;
    names = fieldnames(inputArg);
    for i = 1:numel(names)
        cfg.(names{i}) = inputArg.(names{i});
    end
    if ~isfield(cfg, 'featureSet') || strlength(string(cfg.featureSet)) == 0
        cfg.featureSet = base.featureSet;
    end
    cfg = localApplyFeatureSet(cfg);
    return;
end

cfg = base;
cfg.featureSet = string(inputArg);
cfg = localApplyFeatureSet(cfg);
end

function cfg = localApplyFeatureSet(cfg)
switch lower(char(cfg.featureSet))
    case {'legacy', 'base'}
        cfg.featureSet = "legacy";
        cfg.extraNames = strings(0, 1);
    case {'recovery_augmented', 'r3'}
        cfg.featureSet = "recovery_augmented";
        cfg.extraNames = [ ...
            "prevMode"; ...
            "prevLambda1"; ...
            "fallbackStreak"; ...
            "timeSinceLastAnomalyEnd"; ...
            "ctxInnovation"];
    otherwise
        error('Unsupported stage2 meta feature set: %s', string(cfg.featureSet));
end

cfg.extraDim = numel(cfg.extraNames);
cfg.featureDim = cfg.baseFeatDim + 1 + 1 + 1 + 6 + 6 + 6 + cfg.extraDim;
cfg.timeSinceCapSec = max(double(cfg.timeSinceCapSec), 0);
end
