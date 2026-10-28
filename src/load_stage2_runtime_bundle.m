function bundle = load_stage2_runtime_bundle(modelFile)
% Load shared runtime assets for stage-2 online/replay execution.

defaultRcDiag = [0.44, 0.44, 0.44, 0.2, 0.2, 0.2] .^ 2;

bundle = struct();
bundle.modelFile = string(modelFile);
bundle.net = [];
bundle.normStats = struct();
bundle.svrBundle = struct('available', false);
bundle.gateBundle = struct('available', false);
bundle.trainSummary = struct();
bundle.runtime = localDefaultRuntime(defaultRcDiag);

S = load(modelFile, 'net', 'normStats', 'svrBundle', 'gateBundle', 'trainSummary');
bundle.net = S.net;
bundle.normStats = S.normStats;
bundle.normStats.sigIn(bundle.normStats.sigIn < 1e-6) = 1;
bundle.normStats.sigY(bundle.normStats.sigY < 1e-6) = 1;
if isfield(S, 'svrBundle')
    bundle.svrBundle = S.svrBundle;
end
if isfield(S, 'gateBundle')
    bundle.gateBundle = S.gateBundle;
end

if isfield(S, 'trainSummary')
    bundle.trainSummary = S.trainSummary;
    bundle.runtime = localApplyTrainSummary(bundle.runtime, S.trainSummary);
end

bundle.runtime.svrAvailable = isstruct(bundle.svrBundle) && isfield(bundle.svrBundle, 'available') && logical(bundle.svrBundle.available);
bundle.runtime.gateAvailable = isstruct(bundle.gateBundle) && isfield(bundle.gateBundle, 'available') && logical(bundle.gateBundle.available);

bundle.runtime = localApplyWorkspaceOverrides(bundle.runtime);
bundle.runtime = localResolveGateCalibration(bundle.runtime);
bundle.runtime = localApplyRcScale(bundle.runtime);
bundle.runtime.Rc = diag(bundle.runtime.RcDiag);
end

function runtime = localDefaultRuntime(defaultRcDiag)
runtime = struct();
runtime.stageMode = "stage1_denoise_only";
runtime.enableSubst = false;
runtime.usePredictedVariance = false;
runtime.gammaD = ones(6, 1);
runtime.gammaS = ones(6, 1);
runtime.substEnterProb = 0.60;
runtime.substExitProb = 0.40;
runtime.substEnterCount = 2;
runtime.substExitCount = 3;
runtime.qBoostHorizon = 5;
runtime.baseRcDiag = double(defaultRcDiag(:));
runtime.RcDiag = double(defaultRcDiag(:));
runtime.rcScale = 1.0;
runtime.runtimeModelMode = "denoise_only";
runtime.gateCalibrationEnabled = false;
runtime.gateCalibrationFile = "";
runtime.gateCalibrationUsed = false;
runtime.gateCalibrationSource = "base_rc";
runtime.gateCalibrationStats = struct();
runtime.cleanCalibrationStats = struct('available', false, 'usedOnlineRc', false);
runtime.ctxLen = 120;
runtime.Rc = diag(runtime.RcDiag);
runtime.svrAvailable = false;
runtime.gateAvailable = false;
runtime.gateThresholds = struct('T1', 16.812, 'T2', 14.449, 'T1ch', 6.635, 'T2ch', 5.024);
runtime.metaFeatureConfig = stage2_meta_feature_spec("legacy");
runtime.recoveryEnabled = true;
runtime.recoveryFallbackEnterCount = 12;
runtime.recoveryLambdaEnter = 2 * runtime.gateThresholds.T1;
runtime.recoveryCtxRawOkMinChannels = 4;
runtime.recoveryHoldOffAuxCount = 50;
runtime.recoveryQBoostHorizon = 100;
runtime.recoveryExitCleanCount = 5;
end

function runtime = localApplyTrainSummary(runtime, ts)
if isfield(ts, 'stageMode')
    runtime.stageMode = string(ts.stageMode);
end
if isfield(ts, 'enableSubst')
    runtime.enableSubst = logical(ts.enableSubst);
end
if isfield(ts, 'usePredictedVariance')
    runtime.usePredictedVariance = logical(ts.usePredictedVariance);
end
if isfield(ts, 'gammaD') && numel(ts.gammaD) == 6
    runtime.gammaD = max(double(ts.gammaD(:)), 0.05);
end
if isfield(ts, 'gammaS') && numel(ts.gammaS) == 6
    runtime.gammaS = max(double(ts.gammaS(:)), 0.05);
end
if isfield(ts, 'substEnterProb')
    runtime.substEnterProb = double(ts.substEnterProb);
end
if isfield(ts, 'substExitProb')
    runtime.substExitProb = double(ts.substExitProb);
end
if isfield(ts, 'substEnterCount')
    runtime.substEnterCount = max(1, round(double(ts.substEnterCount)));
end
if isfield(ts, 'substExitCount')
    runtime.substExitCount = max(1, round(double(ts.substExitCount)));
end
if isfield(ts, 'qBoostHorizon')
    runtime.qBoostHorizon = max(1, round(double(ts.qBoostHorizon)));
end
if isfield(ts, 'recoveryEnabled')
    runtime.recoveryEnabled = logical(ts.recoveryEnabled);
end
if isfield(ts, 'recoveryFallbackEnterCount')
    runtime.recoveryFallbackEnterCount = max(1, round(double(ts.recoveryFallbackEnterCount)));
end
if isfield(ts, 'recoveryLambdaEnter')
    runtime.recoveryLambdaEnter = max(double(ts.recoveryLambdaEnter), runtime.gateThresholds.T1);
end
if isfield(ts, 'recoveryCtxRawOkMinChannels')
    runtime.recoveryCtxRawOkMinChannels = max(1, min(6, round(double(ts.recoveryCtxRawOkMinChannels))));
end
if isfield(ts, 'recoveryHoldOffAuxCount')
    runtime.recoveryHoldOffAuxCount = max(1, round(double(ts.recoveryHoldOffAuxCount)));
end
if isfield(ts, 'recoveryQBoostHorizon')
    runtime.recoveryQBoostHorizon = max(1, round(double(ts.recoveryQBoostHorizon)));
end
if isfield(ts, 'recoveryExitCleanCount')
    runtime.recoveryExitCleanCount = max(1, round(double(ts.recoveryExitCleanCount)));
end
if isfield(ts, 'baseRcDiag') && numel(ts.baseRcDiag) == 6
    runtime.baseRcDiag = max(double(ts.baseRcDiag(:)), 1e-9);
    runtime.RcDiag = runtime.baseRcDiag;
end
if isfield(ts, 'sequenceLength')
    runtime.ctxLen = max(20, min(240, round(double(ts.sequenceLength))));
end
if isfield(ts, 'gateThresholds')
    runtime.gateThresholds = localMergeGateThresholds(runtime.gateThresholds, ts.gateThresholds);
end
runtime.recoveryLambdaEnter = max(runtime.recoveryLambdaEnter, runtime.gateThresholds.T1);
if isfield(ts, 'metaFeatureConfig')
    runtime.metaFeatureConfig = stage2_meta_feature_spec(ts.metaFeatureConfig);
end

if isfield(ts, 'runtimeModelMode')
    runtime.runtimeModelMode = string(ts.runtimeModelMode);
elseif runtime.enableSubst && ~strcmpi(runtime.stageMode, "stage1_denoise_only")
    runtime.runtimeModelMode = "dual_head";
else
    runtime.runtimeModelMode = "denoise_only";
end

if isfield(ts, 'runtimeGateCalib')
    calibCfg = ts.runtimeGateCalib;
    if isstruct(calibCfg)
        if isfield(calibCfg, 'enabled')
            runtime.gateCalibrationEnabled = logical(calibCfg.enabled);
        end
        if isfield(calibCfg, 'useOnlineRc')
            runtime.gateCalibrationEnabled = logical(calibCfg.useOnlineRc);
        end
        if isfield(calibCfg, 'file')
            runtime.gateCalibrationFile = string(calibCfg.file);
        elseif isfield(calibCfg, 'defaultFile')
            runtime.gateCalibrationFile = string(calibCfg.defaultFile);
        end
    elseif ischar(calibCfg) || isstring(calibCfg)
        runtime.gateCalibrationEnabled = true;
        runtime.gateCalibrationFile = string(calibCfg);
    end
end
end

function runtime = localApplyWorkspaceOverrides(runtime)
try
    if evalin('base', "exist('stage2RuntimeModelModeOverride','var')")
        modeValue = evalin('base', 'stage2RuntimeModelModeOverride');
        if ischar(modeValue) || isstring(modeValue)
            runtime.runtimeModelMode = string(modeValue);
        end
    end
catch
end

try
    if evalin('base', "exist('stage2GateCalibrationOverride','var')")
        calibValue = evalin('base', 'stage2GateCalibrationOverride');
        if ischar(calibValue) || isstring(calibValue)
            runtime.gateCalibrationFile = string(calibValue);
            runtime.gateCalibrationEnabled = true;
        elseif isstruct(calibValue) && isfield(calibValue, 'file')
            runtime.gateCalibrationFile = string(calibValue.file);
            if isfield(calibValue, 'enabled')
                runtime.gateCalibrationEnabled = logical(calibValue.enabled);
            else
                runtime.gateCalibrationEnabled = true;
            end
        end
    end
catch
end

try
    if evalin('base', "exist('stage2UseGateCalibrationOverride','var')")
        runtime.gateCalibrationEnabled = logical(evalin('base', 'stage2UseGateCalibrationOverride'));
    end
catch
end

try
    if evalin('base', "exist('stage2GateThresholdOverride','var')")
        override = evalin('base', 'stage2GateThresholdOverride');
        if isstruct(override)
            runtime.gateThresholds = localMergeGateThresholds(runtime.gateThresholds, override);
        end
    end
catch
end

try
    if evalin('base', "exist('stage2RuntimeOverride','var')")
        override = evalin('base', 'stage2RuntimeOverride');
        if isstruct(override)
            runtime = localMergeRuntimeOverrides(runtime, override);
        end
    end
catch
end
end

function runtime = localResolveGateCalibration(runtime)
runtime.RcDiag = runtime.baseRcDiag;
runtime.gateCalibrationUsed = false;
runtime.gateCalibrationSource = "base_rc";
runtime.cleanCalibrationStats = struct('available', false, 'usedOnlineRc', false);

if ~runtime.gateCalibrationEnabled
    return;
end

file = char(runtime.gateCalibrationFile);
if isempty(strtrim(file)) || ~isfile(file)
    return;
end

C = load(file);
if isfield(C, 'gateCalibration')
    gateCalibration = C.gateCalibration;
elseif isfield(C, 'cleanCalibration')
    gateCalibration = C.cleanCalibration;
else
    return;
end

if ~isfield(gateCalibration, 'Rc_online_diag') || numel(gateCalibration.Rc_online_diag) ~= 6
    return;
end

runtime.RcDiag = max(double(gateCalibration.Rc_online_diag(:)), 1e-9);
runtime.gateCalibrationUsed = true;
runtime.gateCalibrationSource = string(file);
runtime.gateCalibrationStats = gateCalibration;

runtime.cleanCalibrationStats = struct( ...
    'available', true, ...
    'usedOnlineRc', true, ...
    'file', string(file), ...
    'Rc_online_diag', runtime.RcDiag, ...
    'scaleFactor', localGetField(gateCalibration, 'scaleFactor', NaN), ...
    'lambda1_clean_p95', localGetField(gateCalibration, 'lambda1_clean_p95', NaN), ...
    'cleanModeShare', localGetField(gateCalibration, 'cleanModeShare', NaN), ...
    'cleanFallbackShare', localGetField(gateCalibration, 'cleanFallbackShare', NaN), ...
    'warmupIgnoreSec', localGetField(gateCalibration, 'warmupIgnoreSec', NaN), ...
    'fullClean', localGetField(gateCalibration, 'fullClean', struct()), ...
    'postWarmupClean', localGetField(gateCalibration, 'postWarmupClean', struct()));
end

function runtime = localApplyRcScale(runtime)
if ~isfield(runtime, 'rcScale')
    runtime.rcScale = 1.0;
end

rcScale = double(runtime.rcScale);
if ~(isscalar(rcScale) && isfinite(rcScale) && rcScale > 0)
    rcScale = 1.0;
end
runtime.rcScale = rcScale;

runtime.RcDiag = max(double(runtime.RcDiag(:)) * rcScale, 1e-9);
runtime.baseRcDiag = max(double(runtime.baseRcDiag(:)) * rcScale, 1e-9);
if isfield(runtime, 'cleanCalibrationStats') && isstruct(runtime.cleanCalibrationStats)
    runtime.cleanCalibrationStats.rcScale = rcScale;
    if isfield(runtime.cleanCalibrationStats, 'available') && runtime.cleanCalibrationStats.available
        runtime.cleanCalibrationStats.Rc_online_diag = runtime.RcDiag;
    end
end
end

function value = localGetField(S, field, defaultValue)
if isstruct(S) && isfield(S, field)
    value = S.(field);
else
    value = defaultValue;
end
end

function out = localMergeGateThresholds(base, override)
out = base;
if ~isstruct(override)
    return;
end

names = fieldnames(base);
for i = 1:numel(names)
    key = names{i};
    if isfield(override, key)
        out.(key) = double(override.(key));
    end
end
end

function runtime = localMergeRuntimeOverrides(runtime, override)
names = fieldnames(override);
for i = 1:numel(names)
    key = names{i};
    if strcmp(key, 'gateThresholds') && isstruct(override.(key))
        runtime.gateThresholds = localMergeGateThresholds(runtime.gateThresholds, override.(key));
    else
        runtime.(key) = override.(key);
    end
end

if isfield(runtime, 'gammaD') && numel(runtime.gammaD) == 6
    runtime.gammaD = max(double(runtime.gammaD(:)), 0.05);
end
if isfield(runtime, 'gammaS') && numel(runtime.gammaS) == 6
    runtime.gammaS = max(double(runtime.gammaS(:)), 0.05);
end
if isfield(runtime, 'baseRcDiag') && numel(runtime.baseRcDiag) == 6
    runtime.baseRcDiag = max(double(runtime.baseRcDiag(:)), 1e-9);
end
if isfield(runtime, 'RcDiag') && numel(runtime.RcDiag) == 6
    runtime.RcDiag = max(double(runtime.RcDiag(:)), 1e-9);
end
if isfield(runtime, 'rcScale')
    rcScale = double(runtime.rcScale);
    if ~(isscalar(rcScale) && isfinite(rcScale) && rcScale > 0)
        rcScale = 1.0;
    end
    runtime.rcScale = rcScale;
end
if isfield(runtime, 'recoveryFallbackEnterCount')
    runtime.recoveryFallbackEnterCount = max(1, round(double(runtime.recoveryFallbackEnterCount)));
end
if isfield(runtime, 'recoveryCtxRawOkMinChannels')
    runtime.recoveryCtxRawOkMinChannels = max(1, min(6, round(double(runtime.recoveryCtxRawOkMinChannels))));
end
if isfield(runtime, 'recoveryHoldOffAuxCount')
    runtime.recoveryHoldOffAuxCount = max(1, round(double(runtime.recoveryHoldOffAuxCount)));
end
if isfield(runtime, 'recoveryQBoostHorizon')
    runtime.recoveryQBoostHorizon = max(1, round(double(runtime.recoveryQBoostHorizon)));
end
if isfield(runtime, 'recoveryExitCleanCount')
    runtime.recoveryExitCleanCount = max(1, round(double(runtime.recoveryExitCleanCount)));
end
if isfield(runtime, 'ctxLen')
    runtime.ctxLen = max(20, min(240, round(double(runtime.ctxLen))));
end
runtime.recoveryLambdaEnter = max(double(runtime.recoveryLambdaEnter), double(runtime.gateThresholds.T1));
end
