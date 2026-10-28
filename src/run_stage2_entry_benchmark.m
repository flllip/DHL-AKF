function result = run_stage2_entry_benchmark(entryOrLogs, runtimeCfg, options)
% Run one log entry through the shared Stage-2 EKF/Fusion replay path.

if nargin < 2 || isempty(runtimeCfg)
    runtimeCfg = struct();
end
if nargin < 3 || isempty(options)
    options = struct();
end

[logs, entryName] = localUnpackEntry(entryOrLogs);

dt = double(localField(logs, 'sampleTime', localField(options, 'sampleTime', 0.05)));
stateMat = double(logs.state);
controlMat = double(logs.controls);

if isfield(logs, 'measurements_eval')
    measMat = double(logs.measurements_eval);
    measurementSource = "measurements_eval";
else
    measMat = double(logs.measurements);
    measurementSource = "measurements";
end

if isfield(logs, 'measurements')
    measurementNominal = double(logs.measurements);
else
    measurementNominal = measMat;
end
measurementTrue = localOptionalMatrix(logs, {'Yclean', 'Ytrue'}, 6);

windTrue = double(logs.vw_true);
airTrue = double(logs.air_true);
N = min([size(stateMat, 2), size(controlMat, 2), size(measMat, 2), ...
    size(windTrue, 2), size(airTrue, 2)]);
if ~isempty(measurementNominal)
    N = min(N, size(measurementNominal, 2));
end
if ~isempty(measurementTrue)
    N = min(N, size(measurementTrue, 2));
end

stateMat = stateMat(:, 1:N);
controlMat = controlMat(:, 1:N);
measMat = measMat(:, 1:N);
measurementNominal = measurementNominal(:, 1:N);
windTrue = windTrue(:, 1:N);
airTrue = airTrue(:, 1:N);
if ~isempty(measurementTrue)
    measurementTrue = measurementTrue(:, 1:N);
end

replayFault = localBuildReplayFault(logs, measMat, dt, N);

overrideState = localPushRuntimeOverrides(runtimeCfg);
cleanupObj = onCleanup(@() localRestoreRuntimeOverrides(overrideState)); %#ok<NASGU>

collectDebug = logical(localField(options, 'collectDebug', true));
replayOptions = localField(options, 'replayOptions', struct());
optionFields = {'collectLstmAux', 'collectAirDirectStandalone', 'airDirectModelFile', ...
    'collectSadsMlp', 'collectSadsLstm', 'sadsMlpModelFile', 'sadsLstmModelFile'};
for i = 1:numel(optionFields)
    key = optionFields{i};
    if isfield(options, key)
        replayOptions.(key) = options.(key);
    end
end
clear ekf_baseline_step lstm_ekf_fusion_step lstm_always_step
replay = run_stage2_replay(stateMat, controlMat, measMat, dt, collectDebug, replayOptions);
replay = localRelabelAeroBiasChannelFallbackMode(replay, logs);

online = struct();
online.N = replay.N;
online.time = replay.time;
online.state = stateMat(:, 1:replay.N);
online.controls = controlMat(:, 1:replay.N);
online.measurementNominal = measurementNominal(:, 1:replay.N);
online.measurementFault = measMat(:, 1:replay.N);
online.measurementTrue = localTrimOptional(measurementTrue, replay.N);
online.windTrue = windTrue(:, 1:replay.N);
online.airTrue = airTrue(:, 1:replay.N);
online.windEkf = replay.windEkf;
online.windFuse = replay.windFuse;
online.airEkf = replay.airEkf;
online.airFuse = replay.airFuse;
online.windLstmAux = replay.windLstmAux;
online.airLstmAux = replay.airLstmAux;
online.airLstmDirectAux = replay.airLstmDirectAux;
online.airLstmDirectStandalone = replay.airLstmDirectStandalone;
online.airSadsMlp = replay.airSadsMlp;
online.airSadsLstm = replay.airSadsLstm;
online.mode = replay.mode;
online.lambda1 = replay.lambda1;
online.lambda2 = replay.lambda2;
online.anomalyMask = replayFault.anomaly_mask;
online.anomalyType = replayFault.anomaly_type;
online.anomalySeverity = replayFault.anomaly_severity;
online.substLabel = replayFault.subst_label;
online.aeroBiasMask = localLogMatrix(logs, {'aero_bias_mask'}, 1, replay.N, zeros(1, replay.N));
online.aeroBiasWeight = localLogMatrix(logs, {'aero_bias_weight'}, 1, replay.N, zeros(1, replay.N));
online.aeroBiasMeta = localField(logs, 'aero_bias_meta', struct());

modeStats = localBuildModeStats(online.mode, replayFault);
lambdaStats = localBuildLambdaStats(online.lambda1, online.lambda2, replayFault);
scheduleStats = localBuildScheduleStats(replayFault);
candidateStats = localBuildCandidateStats(replay.debug, replay.mode, replayFault);
gateStats = localBuildGateStats(modeStats, lambdaStats, candidateStats, replay.runtime);
validation = localDefaultValidation();

result = struct();
result.generatedAt = string(datetime('now'));
result.entryName = entryName;
result.measurementSource = measurementSource;
result.sampleTime = dt;
result.runtimeConfig = runtimeCfg;
result.online = online;
result.replay = replay;
result.replayFault = replayFault;
result.modeStats = modeStats;
result.lambdaStats = lambdaStats;
result.scheduleStats = scheduleStats;
result.candidateStats = candidateStats;
result.gateStats = gateStats;
result.metrics = localBuildMetrics(online, replay, replayFault, struct(), modeStats, validation);
result.airMeasurementProxy = localBuildAirMeasurementProxy(measMat);
end

function [logs, entryName] = localUnpackEntry(entryOrLogs)
if isstruct(entryOrLogs) && isfield(entryOrLogs, 'logs')
    logs = entryOrLogs.logs;
    entryName = string(localField(entryOrLogs, 'name', "entry"));
else
    logs = entryOrLogs;
    entryName = "entry";
end
end

function replayFault = localBuildReplayFault(logs, measurements, dt, N)
replayFault = struct();
replayFault.measurements = measurements(:, 1:N);
replayFault.anomaly_mask = localLogMatrix(logs, {'anomaly_mask'}, 6, N, zeros(6, N));
replayFault.anomaly_type = localLogMatrix(logs, {'anomaly_type'}, 1, N, zeros(1, N));
replayFault.anomaly_severity = localLogMatrix(logs, {'anomaly_severity'}, 1, N, zeros(1, N));
replayFault.subst_label = localLogMatrix(logs, {'eval_subst_label', 'subst_label'}, 1, N, zeros(1, N));
replayFault.aero_bias_mask = localLogMatrix(logs, {'aero_bias_mask'}, 1, N, zeros(1, N));
replayFault.aero_bias_weight = localLogMatrix(logs, {'aero_bias_weight'}, 1, N, zeros(1, N));
replayFault.sampleTime = dt;
replayFault.N = N;
replayFault.schedule = struct('source', 'logs', 'sampleTime', dt, 'N', N);
end

function value = localLogMatrix(logs, names, expectedRows, N, defaultValue)
value = defaultValue;
for i = 1:numel(names)
    name = names{i};
    if isfield(logs, name)
        value = localToRows(double(logs.(name)), expectedRows, name);
        break;
    end
end

if size(value, 2) < N
    value(:, end + 1:N) = 0;
end
value = value(:, 1:N);
end

function value = localOptionalMatrix(logs, names, expectedRows)
value = [];
for i = 1:numel(names)
    name = names{i};
    if isfield(logs, name)
        value = localToRows(double(logs.(name)), expectedRows, name);
        return;
    end
end
end

function value = localTrimOptional(value, N)
if isempty(value)
    return;
end
value = value(:, 1:N);
end

function M = localToRows(value, expectedRows, name)
value = squeeze(value);
if isempty(value)
    M = zeros(expectedRows, 0);
    return;
end
if isvector(value)
    if expectedRows == 1
        value = value(:)';
    else
        value = value(:);
    end
end
if size(value, 1) == expectedRows
    M = value;
elseif size(value, 2) == expectedRows
    M = value';
else
    error('%s has incompatible shape: %dx%d.', name, size(value, 1), size(value, 2));
end
end

function state = localPushRuntimeOverrides(runtimeCfg)
state = struct();
state.hadModelFile = evalin('base', "exist('stage2ModelFileOverride','var')");
state.hadModelMode = evalin('base', "exist('stage2RuntimeModelModeOverride','var')");
state.hadUseGate = evalin('base', "exist('stage2UseGateCalibrationOverride','var')");
state.hadGateFile = evalin('base', "exist('stage2GateCalibrationOverride','var')");
state.hadGateThresholds = evalin('base', "exist('stage2GateThresholdOverride','var')");
state.hadRuntimeOverride = evalin('base', "exist('stage2RuntimeOverride','var')");

if state.hadModelFile, state.modelFile = evalin('base', 'stage2ModelFileOverride'); end
if state.hadModelMode, state.modelMode = evalin('base', 'stage2RuntimeModelModeOverride'); end
if state.hadUseGate, state.useGate = evalin('base', 'stage2UseGateCalibrationOverride'); end
if state.hadGateFile, state.gateFile = evalin('base', 'stage2GateCalibrationOverride'); end
if state.hadGateThresholds, state.gateThresholds = evalin('base', 'stage2GateThresholdOverride'); end
if state.hadRuntimeOverride, state.runtimeOverride = evalin('base', 'stage2RuntimeOverride'); end

if isfield(runtimeCfg, 'modelFile')
    assignin('base', 'stage2ModelFileOverride', char(runtimeCfg.modelFile));
end
if isfield(runtimeCfg, 'modelMode')
    assignin('base', 'stage2RuntimeModelModeOverride', char(runtimeCfg.modelMode));
end
if isfield(runtimeCfg, 'useGateCalibration')
    assignin('base', 'stage2UseGateCalibrationOverride', logical(runtimeCfg.useGateCalibration));
end
if isfield(runtimeCfg, 'gateCalibrationFile')
    assignin('base', 'stage2GateCalibrationOverride', char(runtimeCfg.gateCalibrationFile));
end
if isfield(runtimeCfg, 'gateThresholds')
    assignin('base', 'stage2GateThresholdOverride', runtimeCfg.gateThresholds);
end

runtimeOverride = localBuildRuntimeOverride(runtimeCfg);
if ~isempty(fieldnames(runtimeOverride))
    assignin('base', 'stage2RuntimeOverride', runtimeOverride);
elseif isfield(runtimeCfg, 'runtimeOverride')
    evalin('base', 'clear stage2RuntimeOverride');
end
end

function localRestoreRuntimeOverrides(state)
if state.hadModelFile
    assignin('base', 'stage2ModelFileOverride', state.modelFile);
else
    evalin('base', 'clear stage2ModelFileOverride');
end
if state.hadModelMode
    assignin('base', 'stage2RuntimeModelModeOverride', state.modelMode);
else
    evalin('base', 'clear stage2RuntimeModelModeOverride');
end
if state.hadUseGate
    assignin('base', 'stage2UseGateCalibrationOverride', state.useGate);
else
    evalin('base', 'clear stage2UseGateCalibrationOverride');
end
if state.hadGateFile
    assignin('base', 'stage2GateCalibrationOverride', state.gateFile);
else
    evalin('base', 'clear stage2GateCalibrationOverride');
end
if state.hadGateThresholds
    assignin('base', 'stage2GateThresholdOverride', state.gateThresholds);
else
    evalin('base', 'clear stage2GateThresholdOverride');
end
if state.hadRuntimeOverride
    assignin('base', 'stage2RuntimeOverride', state.runtimeOverride);
else
    evalin('base', 'clear stage2RuntimeOverride');
end
end

function runtimeOverride = localBuildRuntimeOverride(runtimeCfg)
runtimeOverride = struct();
if isfield(runtimeCfg, 'runtimeOverride') && isstruct(runtimeCfg.runtimeOverride)
    runtimeOverride = runtimeCfg.runtimeOverride;
end

fields = {'gammaD', 'gammaS', 'rcScale', 'usePredictedVariance', ...
    'recoveryEnabled', 'recoveryFallbackEnterCount', 'recoveryLambdaEnter', ...
    'recoveryCtxRawOkMinChannels', 'recoveryHoldOffAuxCount', ...
    'recoveryQBoostHorizon', 'recoveryExitCleanCount', 'ctxLen'};
for i = 1:numel(fields)
    key = fields{i};
    if isfield(runtimeCfg, key)
        runtimeOverride.(key) = runtimeCfg.(key);
    end
end
end

function replay = localRelabelAeroBiasChannelFallbackMode(replay, logs)
if ~isstruct(logs) || ~isfield(logs, 'aero_bias_mask') || ...
        ~isstruct(replay) || ~isfield(replay, 'mode') || ...
        ~isfield(replay, 'debug') || ~isfield(replay.debug, 'selectedSource')
    return;
end

N = numel(replay.mode);
aeroMask = localLogMatrix(logs, {'aero_bias_mask'}, 1, N, zeros(1, N)) ~= 0;
selectedSource = double(replay.debug.selectedSource);
if size(selectedSource, 1) ~= 6 && size(selectedSource, 2) == 6
    selectedSource = selectedSource';
end
if size(selectedSource, 1) ~= 6
    return;
end
selectedSource = selectedSource(:, 1:min(N, size(selectedSource, 2)));
if size(selectedSource, 2) < N
    selectedSource(:, end + 1:N) = 0;
end

modeVec = round(double(replay.mode(:)'));
usedChannelFallback = any(selectedSource ~= 0, 1) & any(selectedSource ~= 1, 1);
relabelMask = aeroMask & modeVec == 2 & usedChannelFallback;
if any(relabelMask)
    replay.mode(relabelMask) = 3;
end
replay.debug.aeroBiasModeRelabel = relabelMask;
end

function proxy = localBuildAirMeasurementProxy(measMat)
uvw = double(measMat(1:3, :));
vt = sqrt(sum(uvw .^ 2, 1));
alpha = atan2(uvw(3, :), uvw(1, :));
beta = asin(max(min(uvw(2, :) ./ max(vt, 1e-9), 1), -1));
proxy = [vt; alpha; beta];
end

function metrics = localBuildMetrics(online, replay, replayFault, referenceBenchmark, modeStats, validation)
metrics = struct();
metrics.measurementReplayRmse = sqrt(mean((online.measurementFault - replayFault.measurements) .^ 2, 2));
metrics.measurementReplayMaxAbs = max(abs(online.measurementFault - replayFault.measurements), [], 2);

maskAbn = any(replayFault.anomaly_mask ~= 0, 1);
metrics.online = localMetricBundle(online.windEkf, online.windFuse, online.airEkf, online.airFuse, ...
    online.windTrue, online.airTrue, maskAbn);
metrics.replay = localMetricBundle(replay.windEkf, replay.windFuse, replay.airEkf, replay.airFuse, ...
    online.windTrue, online.airTrue, maskAbn);

metrics.mode.cleanShare = modeStats.clean.mode01Share;
metrics.mode.severeShare = modeStats.severe.mode23Share;
metrics.mode.substHitShare = modeStats.subst.mode23Share;
metrics.mode.rawShare = modeStats.overall.share(1);
metrics.mode.denoiseShare = modeStats.overall.share(2);
metrics.mode.substShare = modeStats.overall.share(3);
metrics.mode.fallbackShare = modeStats.overall.share(4);
metrics.mode.maneuverAeroDenoiseShare = modeStats.maneuverAero.share(2);
metrics.mode.maneuverAeroSubstShare = modeStats.maneuverAero.share(3);
metrics.mode.maneuverAeroFallbackShare = modeStats.maneuverAero.share(4);

deltaWind = abs(metrics.online.rmseWindFuse - metrics.replay.rmseWindFuse);
deltaAir = abs(metrics.online.rmseAirFuse - metrics.replay.rmseAirFuse);
relWind = deltaWind ./ max(abs(metrics.replay.rmseWindFuse), 1e-9);
relAir = deltaAir ./ max(abs(metrics.replay.rmseAirFuse), 1e-9);

metrics.consistency.relWind = relWind;
metrics.consistency.relAir = relAir;
metrics.consistency.absWind = deltaWind;
metrics.consistency.absAir = deltaAir;
metrics.consistency.passWind = all((relWind <= validation.relRmseTol) | (deltaWind <= validation.absWindTol));
metrics.consistency.passVt = (relAir(1) <= validation.relRmseTol) || (deltaAir(1) <= validation.absVtTol);
metrics.consistency.passAlpha = (relAir(2) <= validation.relRmseTol) || (deltaAir(2) <= validation.absAngleTolRad);
metrics.consistency.passBeta = (relAir(3) <= validation.relRmseTol) || (deltaAir(3) <= validation.absAngleTolRad);
metrics.consistency.pass = metrics.consistency.passWind && metrics.consistency.passVt && ...
    metrics.consistency.passAlpha && metrics.consistency.passBeta;

metrics.pass.fusionBeatsEkfOnline = all(metrics.online.rmseWindFuse < metrics.online.rmseWindEkf) && ...
    all(metrics.online.rmseAirFuse < metrics.online.rmseAirEkf) && ...
    all(metrics.online.p95AirImprovePct > 0);
metrics.pass.cleanMode = metrics.mode.cleanShare >= validation.cleanModeShareMin;
metrics.pass.severeMode = isnan(metrics.mode.severeShare) || metrics.mode.severeShare >= validation.severeModeShareMin;
metrics.pass.referenceAvailable = ~isempty(fieldnames(referenceBenchmark));
metrics.pass.all = metrics.pass.fusionBeatsEkfOnline && metrics.consistency.pass && ...
    metrics.pass.cleanMode && metrics.pass.severeMode;

if metrics.pass.referenceAvailable
    metrics.referenceDelta.rmseWindFuse = metrics.online.rmseWindFuse - referenceBenchmark.rmseWindFuse;
    metrics.referenceDelta.rmseAirFuse = metrics.online.rmseAirFuse - referenceBenchmark.rmseAirFuse;
else
    metrics.referenceDelta = struct();
end
end

function bundle = localMetricBundle(windEkf, windFuse, airEkf, airFuse, windTrue, airTrue, maskAbn)
windErrEkf = windEkf - windTrue;
windErrFuse = windFuse - windTrue;
airErrEkf = airEkf - airTrue;
airErrFuse = airFuse - airTrue;

bundle.rmseWindEkf = sqrt(mean(windErrEkf .^ 2, 2));
bundle.rmseWindFuse = sqrt(mean(windErrFuse .^ 2, 2));
bundle.rmseAirEkf = sqrt(mean(airErrEkf .^ 2, 2));
bundle.rmseAirFuse = sqrt(mean(airErrFuse .^ 2, 2));
bundle.windImprovePct = 100 * (1 - bundle.rmseWindFuse ./ max(bundle.rmseWindEkf, 1e-9));
bundle.airImprovePct = 100 * (1 - bundle.rmseAirFuse ./ max(bundle.rmseAirEkf, 1e-9));
bundle.p95AirEkf = localP95(airErrEkf(:, maskAbn));
bundle.p95AirFuse = localP95(airErrFuse(:, maskAbn));
bundle.p95AirImprovePct = 100 * (1 - bundle.p95AirFuse ./ max(bundle.p95AirEkf, 1e-9));
end

function modeStats = localBuildModeStats(modeVec, replayFault)
modeVec = round(modeVec(:)');
maskClean = replayFault.anomaly_severity == 0;
maskSevere = replayFault.subst_label ~= 0 | replayFault.anomaly_severity >= 3;
maskSubst = replayFault.subst_label ~= 0;
maskManeuverAero = false(size(maskClean));
if isfield(replayFault, 'aero_bias_mask')
    maskManeuverAero = replayFault.aero_bias_mask ~= 0;
end

modeStats = struct();
modeStats.overall = localModeSlice(modeVec, true(size(modeVec)));
modeStats.clean = localModeSlice(modeVec, maskClean);
modeStats.severe = localModeSlice(modeVec, maskSevere);
modeStats.subst = localModeSlice(modeVec, maskSubst);
modeStats.maneuverAero = localModeSlice(modeVec, maskManeuverAero);
end

function stats = localModeSlice(modeVec, mask)
mask = logical(mask(:)');
stats = struct();
stats.count = sum(mask);
if stats.count <= 0
    stats.hist = zeros(1, 4);
    stats.share = nan(1, 4);
    stats.mode01Share = NaN;
    stats.mode23Share = NaN;
    stats.fallbackShare = NaN;
    return;
end
histVal = histcounts(modeVec(mask), -0.5:1:3.5);
stats.hist = histVal;
stats.share = histVal / max(sum(histVal), 1);
stats.mode01Share = mean(ismember(modeVec(mask), [0 1]));
stats.mode23Share = mean(ismember(modeVec(mask), [2 3]));
stats.fallbackShare = mean(modeVec(mask) == 3);
end

function lambdaStats = localBuildLambdaStats(lambda1, lambda2, replayFault)
maskClean = replayFault.anomaly_severity == 0;
maskSevere = replayFault.subst_label ~= 0 | replayFault.anomaly_severity >= 3;
maskManeuverAero = false(size(maskClean));
if isfield(replayFault, 'aero_bias_mask')
    maskManeuverAero = replayFault.aero_bias_mask ~= 0;
end
lambdaStats = struct();
lambdaStats.lambda1 = struct();
lambdaStats.lambda1.clean = localQuantileStats(lambda1(maskClean));
lambdaStats.lambda1.severe = localQuantileStats(lambda1(maskSevere));
lambdaStats.lambda1.maneuverAero = localQuantileStats(lambda1(maskManeuverAero));
lambdaStats.lambda2 = struct();
lambdaStats.lambda2.clean = localQuantileStats(lambda2(maskClean));
lambdaStats.lambda2.severe = localQuantileStats(lambda2(maskSevere));
lambdaStats.lambda2.maneuverAero = localQuantileStats(lambda2(maskManeuverAero));
lambdaStats.lambda1_clean_p95 = lambdaStats.lambda1.clean.p95;
lambdaStats.lambda2_severe_p50 = lambdaStats.lambda2.severe.p50;
lambdaStats.lambda2_maneuver_aero_p50 = lambdaStats.lambda2.maneuverAero.p50;
end

function stats = localQuantileStats(x)
x = double(x(:));
stats = struct('count', numel(x), 'p05', NaN, 'p50', NaN, 'p95', NaN, 'p99', NaN);
if isempty(x)
    return;
end
q = prctile(x, [5 50 95 99]);
stats.p05 = q(1);
stats.p50 = q(2);
stats.p95 = q(3);
stats.p99 = q(4);
end

function scheduleStats = localBuildScheduleStats(replayFault)
scheduleStats = struct();
scheduleStats.cleanCount = sum(replayFault.anomaly_severity == 0);
scheduleStats.abnormalCount = sum(any(replayFault.anomaly_mask ~= 0, 1));
scheduleStats.severeCount = sum(replayFault.subst_label ~= 0 | replayFault.anomaly_severity >= 3);
scheduleStats.substLabelCount = sum(replayFault.subst_label ~= 0);
typeCounts = histcounts(replayFault.anomaly_type(replayFault.anomaly_type > 0), 0.5:1:6.5);
scheduleStats.typeCounts = typeCounts;
if sum(typeCounts) > 0
    scheduleStats.typeShare = typeCounts / sum(typeCounts);
else
    scheduleStats.typeShare = zeros(1, 6);
end
scheduleStats.severityCounts = histcounts(replayFault.anomaly_severity(replayFault.anomaly_severity > 0), 0.5:1:3.5);
end

function candidateStats = localBuildCandidateStats(debug, modeVec, replayFault)
candidateStats = struct();
candidateStats.available = isstruct(debug) && isfield(debug, 'preferredCandidate');
candidateStats.rawCount = sum(round(modeVec(:)') == 0);
candidateStats.denoiseCount = sum(round(modeVec(:)') == 1);
candidateStats.substitutionCount = sum(round(modeVec(:)') == 2);
candidateStats.fallbackCount = sum(round(modeVec(:)') == 3);

if ~candidateStats.available
    candidateStats.preferredCounts = zeros(1, 2);
    candidateStats.acceptedCounts = zeros(1, 2);
    candidateStats.bestCounts = zeros(1, 2);
    candidateStats.skipUpdateCount = 0;
    candidateStats.cleanFallbackCount = sum(round(modeVec(replayFault.anomaly_severity == 0)) == 3);
    return;
end

candidateStats.preferredCounts = histcounts(debug.preferredCandidate(debug.preferredCandidate > 0), 0.5:1:2.5);
candidateStats.acceptedCounts = histcounts(debug.acceptedCandidate(debug.acceptedCandidate > 0), 0.5:1:2.5);
candidateStats.bestCounts = histcounts(debug.bestCandidate(debug.bestCandidate > 0), 0.5:1:2.5);
candidateStats.skipUpdateCount = sum(debug.skipUpdate);
candidateStats.cleanFallbackCount = sum(round(modeVec(replayFault.anomaly_severity == 0)) == 3);
candidateStats.usedGateCalibrationCount = sum(debug.usedGateCalibration);
if isfield(replayFault, 'aero_bias_mask') && isfield(debug, 'pSubst')
    aeroMask = logical(replayFault.aero_bias_mask(:)');
    pSubst = double(debug.pSubst(:)');
    pSubst = pSubst(1:min(numel(pSubst), numel(aeroMask)));
    aeroMask = aeroMask(1:numel(pSubst));
    if any(aeroMask)
        candidateStats.maneuverAeroPSubstMean = mean(pSubst(aeroMask));
        candidateStats.maneuverAeroPSubstMax = max(pSubst(aeroMask));
    else
        candidateStats.maneuverAeroPSubstMean = NaN;
        candidateStats.maneuverAeroPSubstMax = NaN;
    end
end
if isfield(debug, 'gateWeights')
    candidateStats.gateWeightMean = mean(debug.gateWeights, 3);
else
    candidateStats.gateWeightMean = zeros(3, 6);
end
end

function gateStats = localBuildGateStats(modeStats, lambdaStats, candidateStats, runtimeInfo)
gateStats = struct();
gateStats.thresholds = localField(runtimeInfo, 'gateThresholds', struct('T1', 16.812, 'T2', 14.449, 'T1ch', 6.635, 'T2ch', 5.024));
gateStats.runtimeModelMode = localField(runtimeInfo, 'runtimeModelMode', "unknown");
gateStats.useGateCalibration = logical(localField(runtimeInfo, 'gateCalibrationUsed', false));
gateStats.gateCalibrationSource = localField(runtimeInfo, 'gateCalibrationSource', "base_rc");
gateStats.cleanMode01Share = modeStats.clean.mode01Share;
gateStats.cleanFallbackShare = modeStats.clean.fallbackShare;
gateStats.severeMode23Share = modeStats.severe.mode23Share;
gateStats.maneuverAeroMode1Share = modeStats.maneuverAero.share(2);
gateStats.maneuverAeroMode2Share = modeStats.maneuverAero.share(3);
gateStats.maneuverAeroFallbackShare = modeStats.maneuverAero.fallbackShare;
gateStats.lambda1_clean_p95 = lambdaStats.lambda1.clean.p95;
gateStats.lambda2_severe_p50 = lambdaStats.lambda2.severe.p50;
gateStats.lambda2_maneuver_aero_p50 = lambdaStats.lambda2.maneuverAero.p50;
gateStats.preferredCandidateCounts = candidateStats.preferredCounts;
gateStats.acceptedCandidateCounts = candidateStats.acceptedCounts;
gateStats.maneuverAeroPSubstMean = localField(candidateStats, 'maneuverAeroPSubstMean', NaN);
gateStats.maneuverAeroPSubstMax = localField(candidateStats, 'maneuverAeroPSubstMax', NaN);
end

function validation = localDefaultValidation()
validation = struct();
validation.relRmseTol = 0.15;
validation.absWindTol = 0.8;
validation.absVtTol = 1.5;
validation.absAngleTolRad = deg2rad(0.08);
validation.cleanModeShareMin = 0.85;
validation.severeModeShareMin = 0.60;
end

function p95 = localP95(errMat)
if isempty(errMat)
    p95 = [NaN; NaN; NaN];
else
    p95 = prctile(abs(errMat').', 95, 2);
end
end

function value = localField(S, field, defaultValue)
if isstruct(S) && isfield(S, field)
    value = S.(field);
else
    value = defaultValue;
end
end
