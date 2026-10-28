defaults = stage2_10ms_defaults();

replayCfg = struct();
replayCfg.modelFile = 'lstm_akf_net_10ms_windenv_maneuver_aerobias.mat';
replayCfg.resultFile = 'stage2_synthetic_online_replay_gamma5_20_windenv_sads_results.mat';
replayCfg.summaryCsvFile = 'stage2_synthetic_online_replay_gamma5_20_windenv_sads_summary.csv';
replayCfg.figurePrefix = 'stage2_synthetic_online_replay_gamma5_20_windenv_sads';
replayCfg.enablePlots = true;
replayCfg.airDirectModelFile = 'lstm_air_direct_net_10ms_windenv.mat';
replayCfg.collectAirDirectStandalone = [];
replayCfg.sadsMlpModelFile = 'sads_cascade_mlp_10ms_windenv.mat';
replayCfg.sadsLstmModelFile = 'sads_seq_lstm_10ms_windenv.mat';
replayCfg.collectSadsMlp = [];
replayCfg.collectSadsLstm = [];
replayCfg.aeroBias = struct('enabled', false);
replayCfg.runtimeOverride = struct();

if exist('stage2SyntheticReplayOverride', 'var') && isstruct(stage2SyntheticReplayOverride)
    replayCfg = localMergeStruct(replayCfg, stage2SyntheticReplayOverride);
end

modelFile = char(replayCfg.modelFile);
resultFile = char(replayCfg.resultFile);
summaryCsvFile = char(replayCfg.summaryCsvFile);
figurePrefix = char(replayCfg.figurePrefix);
enablePlots = logical(replayCfg.enablePlots);
airDirectModelFile = char(replayCfg.airDirectModelFile);
sadsMlpModelFile = char(replayCfg.sadsMlpModelFile);
sadsLstmModelFile = char(replayCfg.sadsLstmModelFile);
if isempty(replayCfg.collectAirDirectStandalone)
    collectAirDirectStandalone = exist(airDirectModelFile, 'file') == 2;
else
    collectAirDirectStandalone = logical(replayCfg.collectAirDirectStandalone);
end
if isempty(replayCfg.collectSadsMlp)
    collectSadsMlp = exist(sadsMlpModelFile, 'file') == 2;
else
    collectSadsMlp = logical(replayCfg.collectSadsMlp);
end
if isempty(replayCfg.collectSadsLstm)
    collectSadsLstm = exist(sadsLstmModelFile, 'file') == 2;
else
    collectSadsLstm = logical(replayCfg.collectSadsLstm);
end

S = load(modelFile, 'trainSummary');
if isfield(S, 'trainSummary')
    trainSummary = S.trainSummary;
else
    trainSummary = struct();
end

syntheticCfg = localResolveSyntheticCfg(defaults.synthetic, trainSummary);
dataset = generate_full_envelope_synthetic_logs(syntheticCfg);
entries = localSelectBenchmarkSyntheticEntries(dataset.test, trainSummary);

runtimeCfg = defaults.runtime;
runtimeCfg.modelFile = modelFile;
runtimeCfg.modelMode = 'dual_head';
runtimeCfg.useGateCalibration = true;
runtimeCfg.gammaD = 5 * ones(6, 1);
runtimeCfg.gammaS = 20 * ones(6, 1);
if isstruct(replayCfg.runtimeOverride) && ~isempty(fieldnames(replayCfg.runtimeOverride))
    runtimeCfg.runtimeOverride = localMergeStruct(localField(runtimeCfg, 'runtimeOverride', struct()), ...
        replayCfg.runtimeOverride);
end
if isstruct(replayCfg.runtimeOverride) && ~isempty(fieldnames(replayCfg.runtimeOverride))
    runtimeCfg.runtimeOverride = replayCfg.runtimeOverride;
end

entryResults = struct([]);
%for i = 1:numel(entries)
for i = 1:1
    fprintf('Synthetic online replay %d/%d | %s\n', i, numel(entries), entries(i).name);
    entryForReplay = localApplyManeuverAeroBias(entries(i), replayCfg.aeroBias, i);
    replayOptions = struct('collectLstmAux', true);
    replayOptions.collectAirDirectStandalone = collectAirDirectStandalone;
    replayOptions.airDirectModelFile = airDirectModelFile;
    replayOptions.collectSadsMlp = collectSadsMlp;
    replayOptions.collectSadsLstm = collectSadsLstm;
    replayOptions.sadsMlpModelFile = sadsMlpModelFile;
    replayOptions.sadsLstmModelFile = sadsLstmModelFile;
    entryResult = run_stage2_entry_benchmark(entryForReplay, runtimeCfg, replayOptions);
    if i == 1
        entryResults = repmat(entryResult, numel(entries), 1);
    else
        entryResults(i) = entryResult;
    end
end

aggregate = localAggregateEntryResults(entryResults);
referenceBenchmark = localField(trainSummary, 'benchmark', struct());

results = struct();
results.generatedAt = string(datetime('now'));
results.modelFile = string(modelFile);
results.config = struct( ...
    'modelFile', string(modelFile), ...
    'modelMode', string(runtimeCfg.modelMode), ...
    'useGateCalibration', logical(runtimeCfg.useGateCalibration), ...
    'gammaD', runtimeCfg.gammaD, ...
    'gammaS', runtimeCfg.gammaS, ...
    'resultFile', string(resultFile), ...
    'summaryCsvFile', string(summaryCsvFile), ...
    'figurePrefix', string(figurePrefix), ...
    'airDirectModelFile', string(airDirectModelFile), ...
    'collectAirDirectStandalone', logical(collectAirDirectStandalone), ...
    'sadsMlpModelFile', string(sadsMlpModelFile), ...
    'sadsLstmModelFile', string(sadsLstmModelFile), ...
    'collectSadsMlp', logical(collectSadsMlp), ...
    'collectSadsLstm', logical(collectSadsLstm), ...
    'aeroBias', replayCfg.aeroBias, ...
    'enablePlots', logical(enablePlots));
results.syntheticConfig = syntheticCfg;
results.referenceBenchmark = referenceBenchmark;
results.referenceBenchmarkSourceFiles = localField(trainSummary, 'benchmarkSourceFiles', strings(0, 1));
results.provenance = localBuildSyntheticTeachingProvenance(syntheticCfg);
results.entries = entryResults;
results.benchmark = aggregate.benchmark;
results.metrics = aggregate.metrics;
results.modeStats = aggregate.modeStats;
results.lambdaStats = aggregate.lambdaStats;
results.scheduleStats = aggregate.scheduleStats;
results.syntheticVsReference = localBuildReferenceDelta(aggregate.benchmark, referenceBenchmark);
results.representativeEntry = localSelectRepresentativeEntry(entryResults);
if enablePlots
    results.figureFiles = plot_stage2_synthetic_replay_overview(results, struct('prefix', figurePrefix));
else
    results.figureFiles = strings(0, 1);
end

save(resultFile, 'results');
localWriteSummaryCsv(results, summaryCsvFile);

fprintf('saved results: %s\n', resultFile);
fprintf('saved summary: %s\n', summaryCsvFile);
localPrintReplaySummary(results);

function cfg = localResolveSyntheticCfg(defaultCfg, trainSummary)
if ~isstruct(trainSummary)
    cfg = localFallbackSyntheticCfg(defaultCfg);
    return;
end

if isfield(trainSummary, 'sourceInfo') && isstruct(trainSummary.sourceInfo)
    sourceInfo = trainSummary.sourceInfo;
    if isfield(sourceInfo, 'syntheticConfig')
        cfg = sourceInfo.syntheticConfig;
        return;
    elseif isfield(sourceInfo, 'config')
        cfg = sourceInfo.config;
        return;
    end
end

if isfield(trainSummary, 'synthetic') && isstruct(trainSummary.synthetic)
    cfg = trainSummary.synthetic;
    return;
end

cfg = localFallbackSyntheticCfg(defaultCfg);
end

function cfg = localFallbackSyntheticCfg(defaultCfg)
baseDataset = generate_full_envelope_synthetic_logs();
cfg = localMergeStruct(baseDataset.config, defaultCfg);
end

function out = localMergeStruct(base, override)
out = base;
if ~isstruct(override)
    return;
end
names = fieldnames(override);
for i = 1:numel(names)
    key = names{i};
    if isfield(out, key) && isstruct(out.(key)) && isstruct(override.(key))
        out.(key) = localMergeStruct(out.(key), override.(key));
    else
        out.(key) = override.(key);
    end
end
end

function entryOut = localApplyManeuverAeroBias(entryIn, inputCfg, entryIndex)
cfg = localResolveManeuverAeroBiasCfg(inputCfg);
entryOut = entryIn;
if ~cfg.enabled || ~isstruct(entryIn) || ~isfield(entryIn, 'logs')
    return;
end

logs = entryIn.logs;
need = {'state', 'controls', 'measurements', 'Yclean', 'vw_true'};
for i = 1:numel(need)
    if ~isfield(logs, need{i})
        return;
    end
end

stateMat = double(logs.state);
controlMat = double(logs.controls);
measNominal = double(logs.measurements);
yClean = double(logs.Yclean);
windTrue = double(logs.vw_true);
N = min([size(stateMat, 2), size(controlMat, 2), size(measNominal, 2), ...
    size(yClean, 2), size(windTrue, 2)]);
if N <= 1
    return;
end

[mask, weight, indicators] = localManeuverAeroBiasWeight(logs, yClean, cfg, N);
coeffScale = localManeuverAeroCoeffScale(cfg, entryIndex);
delta = zeros(6, N);
for k = 1:N
    scaleK = 1 + weight(k) * (coeffScale - 1);
    yBias = wind_estimation_model_numeric_aerobias( ...
        stateMat(:, k), controlMat(:, k), windTrue(:, k), scaleK);
    delta(:, k) = yBias - yClean(:, k);
end

logs.measurements(:, 1:N) = measNominal(:, 1:N) + delta;
if isfield(logs, 'measurements_eval')
    measEval = double(logs.measurements_eval);
    nEval = min(N, size(measEval, 2));
    faultResidual = measEval(:, 1:nEval) - measNominal(:, 1:nEval);
    measEvalOut = logs.measurements(:, 1:nEval) + faultResidual;
    if isfield(logs, 'anomaly_type')
        type4 = localRow(logs.anomaly_type, nEval) == 4;
        measEvalOut(:, type4) = measEval(:, type4);
    end
    logs.measurements_eval(:, 1:nEval) = measEvalOut;
end

logs.aero_bias_mask = double(mask);
logs.aero_bias_weight = double(weight);
logs.aero_bias_meta = struct( ...
    'enabled', true, ...
    'mode', "maneuver_window_coeff_scale", ...
    'coeffOrder', ["Cx"; "Cy"; "Cz"; "Cl"; "Cm"; "Cn"], ...
    'coeffScaleFull', coeffScale(:), ...
    'activeFrameCount', sum(mask), ...
    'activeDurationSec', sum(mask) * localField(logs, 'sampleTime', 0.01), ...
    'indicators', indicators, ...
    'config', cfg);
entryOut.logs = logs;
end

function cfg = localResolveManeuverAeroBiasCfg(inputCfg)
cfg = struct( ...
    'enabled', false, ...
    'minPct', 0.10, ...
    'maxPct', 0.20, ...
    'mode', "signed", ...
    'seed', 20260514, ...
    'quantilePct', 90, ...
    'alphaRateThresholdDegPerSec', NaN, ...
    'rampSec', 0.35, ...
    'mergeGapSec', 0.15, ...
    'minDurationSec', 0.08);
if islogical(inputCfg) || isnumeric(inputCfg)
    cfg.enabled = logical(inputCfg);
    return;
end
if isstruct(inputCfg)
    cfg = localMergeStruct(cfg, inputCfg);
end
cfg.enabled = logical(cfg.enabled);
cfg.minPct = max(0, double(cfg.minPct));
cfg.maxPct = max(cfg.minPct, double(cfg.maxPct));
cfg.quantilePct = min(max(double(cfg.quantilePct), 50), 99.9);
cfg.seed = double(cfg.seed);
cfg.rampSec = max(0, double(cfg.rampSec));
cfg.mergeGapSec = max(0, double(cfg.mergeGapSec));
cfg.minDurationSec = max(0, double(cfg.minDurationSec));
cfg.mode = string(cfg.mode);
end

function coeffScale = localManeuverAeroCoeffScale(cfg, entryIndex)
rngState = rng;
cleanupObj = onCleanup(@() rng(rngState)); %#ok<NASGU>
rng(double(cfg.seed) + double(entryIndex), 'twister');
mag = cfg.minPct + (cfg.maxPct - cfg.minPct) * rand(6, 1);
switch lower(string(cfg.mode))
    case {"all_positive", "positive", "plus"}
        sgn = ones(6, 1);
    case {"all_negative", "negative", "minus"}
        sgn = -ones(6, 1);
    case {"longitudinal"}
        sgn = [1; 0; 1; 0; 1; 0];
    otherwise
        sgn = sign(rand(6, 1) - 0.5);
        sgn(sgn == 0) = 1;
end
coeffScale = 1 + sgn .* mag;
end

function [mask, weight, indicators] = localManeuverAeroBiasWeight(logs, yClean, cfg, N)
dt = double(localField(logs, 'sampleTime', 0.01));
if ~(isfinite(dt) && dt > 0)
    dt = 0.01;
end
stateMat = double(logs.state(:, 1:N));
controlMat = double(logs.controls(:, 1:N));
qAbs = abs(stateMat(11, :));
rAbs = abs(stateMat(12, :));
deAbs = abs(controlMat(2, :));
if isfield(logs, 'air_true') && size(logs.air_true, 2) >= N
    alpha = double(logs.air_true(2, 1:N));
else
    alpha = atan2(stateMat(9, :), max(abs(stateMat(7, :)), 1e-9));
end
alphaRateDeg = abs([0, diff(rad2deg(alpha))] / dt);
nzProxy = abs(double(yClean(3, 1:N))) / 32.17;

qThr = localPercentileThreshold(qAbs, cfg.quantilePct);
rThr = localPercentileThreshold(rAbs, cfg.quantilePct);
deThr = localPercentileThreshold(deAbs, cfg.quantilePct);
nzThr = localPercentileThreshold(nzProxy, cfg.quantilePct);
if isfinite(cfg.alphaRateThresholdDegPerSec)
    alphaRateThr = double(cfg.alphaRateThresholdDegPerSec);
else
    alphaRateThr = localPercentileThreshold(alphaRateDeg, cfg.quantilePct);
end

rawMask = (qAbs > qThr) | (rAbs > rThr) | (deAbs > deThr) | ...
    (alphaRateDeg > alphaRateThr) | (nzProxy > nzThr);
rawMask = localMergeShortGaps(rawMask, round(cfg.mergeGapSec / dt));
mask = localMinimumDurationMask(rawMask, round(cfg.minDurationSec / dt));
weight = localRaisedCosineWeight(mask, round(cfg.rampSec / dt));

indicators = struct( ...
    'qAbsThresholdRadPerSec', qThr, ...
    'rAbsThresholdRadPerSec', rThr, ...
    'deltaEAbsThresholdDeg', deThr, ...
    'alphaRateThresholdDegPerSec', alphaRateThr, ...
    'nzProxyThresholdG', nzThr);
end

function thr = localPercentileThreshold(x, pct)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    thr = inf;
else
    thr = prctile(x, pct);
end
end

function maskOut = localMergeShortGaps(maskIn, maxGap)
maskOut = logical(maskIn(:)');
if maxGap <= 0 || isempty(maskOut)
    return;
end
edges = diff([true, maskOut, true]);
starts = find(edges == -1);
stops = find(edges == 1) - 1;
for i = 1:numel(starts)
    isInteriorGap = starts(i) > 1 && stops(i) < numel(maskOut);
    if isInteriorGap && stops(i) >= starts(i) && (stops(i) - starts(i) + 1) <= maxGap
        maskOut(starts(i):stops(i)) = true;
    end
end
end

function maskOut = localMinimumDurationMask(maskIn, minSamples)
maskIn = logical(maskIn(:)');
maskOut = false(size(maskIn));
if minSamples <= 1
    maskOut = maskIn;
    return;
end
edges = diff([false, maskIn, false]);
starts = find(edges == 1);
stops = find(edges == -1) - 1;
for i = 1:numel(starts)
    if stops(i) - starts(i) + 1 >= minSamples
        maskOut(starts(i):stops(i)) = true;
    end
end
end

function weight = localRaisedCosineWeight(mask, rampSamples)
mask = logical(mask(:)');
weight = double(mask);
if rampSamples <= 0 || isempty(mask)
    return;
end
edges = diff([false, mask, false]);
starts = find(edges == 1);
stops = find(edges == -1) - 1;
for i = 1:numel(starts)
    s = starts(i);
    e = stops(i);
    left = max(1, s - rampSamples):max(1, s - 1);
    if ~isempty(left)
        x = (numel(left):-1:1) / (numel(left) + 1);
        weight(left) = max(weight(left), 0.5 * (1 + cos(pi * x)));
    end
    right = min(numel(mask), e + 1):min(numel(mask), e + rampSamples);
    if ~isempty(right)
        x = (1:numel(right)) / (numel(right) + 1);
        weight(right) = max(weight(right), 0.5 * (1 + cos(pi * x)));
    end
end
weight = min(max(weight, 0), 1);
end

function row = localRow(x, N)
row = double(x);
if size(row, 1) ~= 1 && size(row, 2) == 1
    row = row';
end
row = row(1, 1:min(N, numel(row)));
if numel(row) < N
    row(end + 1:N) = 0;
end
end

function entries = localSelectBenchmarkSyntheticEntries(entries, trainSummary)
if ~isstruct(trainSummary) || ~isfield(trainSummary, 'benchmarkSourceFiles')
    return;
end

sourceFiles = string(trainSummary.benchmarkSourceFiles(:));
entryNames = string({entries.name});
keep = ismember(entryNames, sourceFiles);
if any(keep)
    entries = entries(keep);
end
end

function aggregate = localAggregateEntryResults(entryResults)
allWindErrBase = [];
allWindErrFuse = [];
allAirErrBase = [];
allAirErrFuse = [];
allAirErrSadsMlp = [];
allAirErrSadsLstm = [];
allAirErrBaseAbn = [];
allAirErrFuseAbn = [];
modeHist = [];
lambda1 = [];
lambda2 = [];
anomalySeverity = [];
substLabel = [];
anomalyMask = [];

for i = 1:numel(entryResults)
    r = entryResults(i);
    vwTrue = r.online.windTrue;
    airTrue = r.online.airTrue;
    abnMask = any(r.replayFault.anomaly_mask ~= 0, 1);

    allWindErrBase = [allWindErrBase, r.replay.windEkf - vwTrue]; %#ok<AGROW>
    allWindErrFuse = [allWindErrFuse, r.replay.windFuse - vwTrue]; %#ok<AGROW>
    allAirErrBase = [allAirErrBase, r.replay.airEkf - airTrue]; %#ok<AGROW>
    allAirErrFuse = [allAirErrFuse, r.replay.airFuse - airTrue]; %#ok<AGROW>
    airSadsMlp = localReplayAir(r.replay, 'airSadsMlp', airTrue);
    airSadsLstm = localReplayAir(r.replay, 'airSadsLstm', airTrue);
    if ~isempty(airSadsMlp)
        allAirErrSadsMlp = [allAirErrSadsMlp, airSadsMlp - airTrue]; %#ok<AGROW>
    end
    if ~isempty(airSadsLstm)
        allAirErrSadsLstm = [allAirErrSadsLstm, airSadsLstm - airTrue]; %#ok<AGROW>
    end
    allAirErrBaseAbn = [allAirErrBaseAbn, r.replay.airEkf(:, abnMask) - airTrue(:, abnMask)]; %#ok<AGROW>
    allAirErrFuseAbn = [allAirErrFuseAbn, r.replay.airFuse(:, abnMask) - airTrue(:, abnMask)]; %#ok<AGROW>
    modeHist = [modeHist, r.replay.mode]; %#ok<AGROW>
    lambda1 = [lambda1, r.replay.lambda1]; %#ok<AGROW>
    lambda2 = [lambda2, r.replay.lambda2]; %#ok<AGROW>
    anomalySeverity = [anomalySeverity, r.replayFault.anomaly_severity]; %#ok<AGROW>
    substLabel = [substLabel, r.replayFault.subst_label]; %#ok<AGROW>
    anomalyMask = [anomalyMask, r.replayFault.anomaly_mask]; %#ok<AGROW>
end

benchmark = struct();
benchmark.rmseWindBase = sqrt(mean(allWindErrBase .^ 2, 2));
benchmark.rmseWindFuse = sqrt(mean(allWindErrFuse .^ 2, 2));
benchmark.rmseAirBase = sqrt(mean(allAirErrBase .^ 2, 2));
benchmark.rmseAirFuse = sqrt(mean(allAirErrFuse .^ 2, 2));
benchmark.rmseAirSadsMlp = localRmseOrNan(allAirErrSadsMlp, 3);
benchmark.rmseAirSadsLstm = localRmseOrNan(allAirErrSadsLstm, 3);
benchmark.maeAirSadsMlp = localMaeOrNan(allAirErrSadsMlp, 3);
benchmark.maeAirSadsLstm = localMaeOrNan(allAirErrSadsLstm, 3);
benchmark.airImprovePct = 100 * (1 - benchmark.rmseAirFuse ./ max(benchmark.rmseAirBase, 1e-9));
benchmark.p95AirBaseAbn = localP95(allAirErrBaseAbn);
benchmark.p95AirFuseAbn = localP95(allAirErrFuseAbn);
benchmark.airP95ImprovePct = 100 * (1 - benchmark.p95AirFuseAbn ./ max(benchmark.p95AirBaseAbn, 1e-9));
benchmark.modeShare = histcounts(modeHist, -0.5:1:3.5) / max(numel(modeHist), 1);
benchmark.passAverage = mean(benchmark.airImprovePct) > 0 && mean(benchmark.airP95ImprovePct) > 0;

metrics = struct();
metrics.online.rmseWindEkf = benchmark.rmseWindBase;
metrics.online.rmseWindFuse = benchmark.rmseWindFuse;
metrics.online.rmseAirEkf = benchmark.rmseAirBase;
metrics.online.rmseAirFuse = benchmark.rmseAirFuse;
metrics.online.rmseAirSadsMlp = benchmark.rmseAirSadsMlp;
metrics.online.rmseAirSadsLstm = benchmark.rmseAirSadsLstm;
metrics.online.maeAirSadsMlp = benchmark.maeAirSadsMlp;
metrics.online.maeAirSadsLstm = benchmark.maeAirSadsLstm;
metrics.online.airImprovePct = benchmark.airImprovePct;
metrics.online.p95AirEkf = benchmark.p95AirBaseAbn;
metrics.online.p95AirFuse = benchmark.p95AirFuseAbn;
metrics.online.p95AirImprovePct = benchmark.airP95ImprovePct;
metrics.mode.rawShare = benchmark.modeShare(1);
metrics.mode.denoiseShare = benchmark.modeShare(2);
metrics.mode.substShare = benchmark.modeShare(3);
metrics.mode.fallbackShare = benchmark.modeShare(4);

modeStats = localModeStats(modeHist, anomalySeverity, substLabel);
lambdaStats = localLambdaStats(lambda1, lambda2, anomalySeverity, substLabel);
scheduleStats = struct( ...
    'cleanCount', sum(anomalySeverity == 0), ...
    'abnormalCount', sum(any(anomalyMask ~= 0, 1)), ...
    'severeCount', sum(substLabel ~= 0 | anomalySeverity >= 3), ...
    'substLabelCount', sum(substLabel ~= 0));

aggregate = struct('benchmark', benchmark, 'metrics', metrics, ...
    'modeStats', modeStats, 'lambdaStats', lambdaStats, 'scheduleStats', scheduleStats);
end

function air = localReplayAir(replay, fieldName, airTrue)
air = [];
if ~isstruct(replay) || ~isfield(replay, fieldName) || isempty(replay.(fieldName))
    return;
end
air = double(replay.(fieldName));
if size(air, 1) ~= 3 && size(air, 2) == 3
    air = air';
end
if size(air, 1) ~= 3
    air = [];
    return;
end
N = size(airTrue, 2);
air = air(:, 1:min(N, size(air, 2)));
if size(air, 2) < N
    air(:, end + 1:N) = NaN;
end
end

function r = localRmseOrNan(errMat, rows)
if isempty(errMat)
    r = nan(rows, 1);
else
    r = sqrt(mean(errMat .^ 2, 2, 'omitnan'));
end
end

function m = localMaeOrNan(errMat, rows)
if isempty(errMat)
    m = nan(rows, 1);
else
    m = mean(abs(errMat), 2, 'omitnan');
end
end

function delta = localBuildReferenceDelta(benchmark, referenceBenchmark)
delta = struct('available', false);
if ~isstruct(referenceBenchmark) || ~isfield(referenceBenchmark, 'rmseAirFuse')
    return;
end

delta.available = true;
delta.rmseWindBase = benchmark.rmseWindBase - referenceBenchmark.rmseWindBase;
delta.rmseWindFuse = benchmark.rmseWindFuse - referenceBenchmark.rmseWindFuse;
delta.rmseAirBase = benchmark.rmseAirBase - referenceBenchmark.rmseAirBase;
delta.rmseAirFuse = benchmark.rmseAirFuse - referenceBenchmark.rmseAirFuse;
end

function provenance = localBuildSyntheticTeachingProvenance(syntheticCfg)
provenance = struct();
provenance.dataType = "synthetic_replay";
provenance.sourceType = "generate_full_envelope_synthetic_logs";
provenance.intendedUse = "teaching_demo";
provenance.isMeasuredFlightData = false;
provenance.isPhysicalFlightRecord = false;
provenance.selectionPolicy = "representative_case_closest_to_median_metrics";
provenance.explanation = "Results come from simulator-generated full-envelope replay and are intended for reproducible teaching/demo displays, not real-flight reporting.";
provenance.sampleTime = localField(syntheticCfg, 'sampleTime', NaN);
provenance.sequenceLength = localField(syntheticCfg, 'sequenceLength', NaN);
end

function representative = localSelectRepresentativeEntry(entryResults)
featureNames = ["rmse_wind_fuse_n", "rmse_wind_fuse_e", ...
    "rmse_vt_fuse", "rmse_alpha_fuse_deg", "rmse_beta_fuse_deg", ...
    "severe_share_pct", "fallback_share_pct"];
representative = struct( ...
    'entryIndex', 1, ...
    'entryName', "", ...
    'selectionMode', "closest_to_median_metrics", ...
    'distanceScore', NaN, ...
    'featureNames', featureNames, ...
    'selectedFeatureVector', nan(1, numel(featureNames)), ...
    'medianFeatureVector', nan(1, numel(featureNames)));
if isempty(entryResults)
    return;
end

F = nan(numel(entryResults), numel(featureNames));
entryNames = strings(numel(entryResults), 1);
for i = 1:numel(entryResults)
    r = entryResults(i);
    entryNames(i) = string(localField(r, 'entryName', sprintf('entry_%02d', i)));
    F(i, :) = [ ...
        double(r.metrics.online.rmseWindFuse(1)), ...
        double(r.metrics.online.rmseWindFuse(2)), ...
        double(r.metrics.online.rmseAirFuse(1)), ...
        double(rad2deg(r.metrics.online.rmseAirFuse(2))), ...
        double(rad2deg(r.metrics.online.rmseAirFuse(3))), ...
        100 * mean(r.replayFault.subst_label ~= 0 | r.replayFault.anomaly_severity >= 3), ...
        100 * double(localField(r.metrics.mode, 'fallbackShare', NaN))];
end

med = median(F, 1, 'omitnan');
q = prctile(F, [25 75], 1);
scale = max(q(2, :) - q(1, :), 1e-6);
dist = mean(abs((F - med) ./ scale), 2, 'omitnan');
dist(~isfinite(dist)) = inf;
[bestDist, idx] = min(dist);
if ~isfinite(bestDist)
    idx = 1;
    bestDist = NaN;
end

representative.entryIndex = idx;
representative.entryName = entryNames(idx);
representative.distanceScore = bestDist;
representative.selectedFeatureVector = F(idx, :);
representative.medianFeatureVector = med;
end

function localWriteSummaryCsv(results, csvFile)
rows = strings(numel(results.entries) + 1, 1);
rmseWindEkfN = zeros(numel(rows), 1);
rmseWindEkfE = zeros(numel(rows), 1);
rmseWindFuseN = zeros(numel(rows), 1);
rmseWindFuseE = zeros(numel(rows), 1);
rmseVtEkf = zeros(numel(rows), 1);
rmseAlphaEkf = zeros(numel(rows), 1);
rmseBetaEkf = zeros(numel(rows), 1);
rmseVtFuse = zeros(numel(rows), 1);
rmseAlphaFuse = zeros(numel(rows), 1);
rmseBetaFuse = zeros(numel(rows), 1);
rmseVtSadsMlp = nan(numel(rows), 1);
rmseAlphaSadsMlp = nan(numel(rows), 1);
rmseBetaSadsMlp = nan(numel(rows), 1);
rmseVtSadsLstm = nan(numel(rows), 1);
rmseAlphaSadsLstm = nan(numel(rows), 1);
rmseBetaSadsLstm = nan(numel(rows), 1);
lambda1CleanP95 = zeros(numel(rows), 1);
lambda2SevereP50 = zeros(numel(rows), 1);
fallbackShare = zeros(numel(rows), 1);

for i = 1:numel(results.entries)
    r = results.entries(i);
    rows(i) = r.entryName;
    [rmseWindEkfN(i), rmseWindEkfE(i), rmseWindFuseN(i), rmseWindFuseE(i), ...
        rmseVtEkf(i), rmseAlphaEkf(i), rmseBetaEkf(i), ...
        rmseVtFuse(i), rmseAlphaFuse(i), rmseBetaFuse(i)] = localMetricRow(r.metrics.online);
    [rmseVtSadsMlp(i), rmseAlphaSadsMlp(i), rmseBetaSadsMlp(i)] = ...
        localAirRmseRow(localReplayAir(r.replay, 'airSadsMlp', r.online.airTrue), r.online.airTrue);
    [rmseVtSadsLstm(i), rmseAlphaSadsLstm(i), rmseBetaSadsLstm(i)] = ...
        localAirRmseRow(localReplayAir(r.replay, 'airSadsLstm', r.online.airTrue), r.online.airTrue);
    lambda1CleanP95(i) = r.lambdaStats.lambda1_clean_p95;
    lambda2SevereP50(i) = r.lambdaStats.lambda2_severe_p50;
    fallbackShare(i) = r.metrics.mode.fallbackShare;
end

idx = numel(rows);
rows(idx) = "ALL";
[rmseWindEkfN(idx), rmseWindEkfE(idx), rmseWindFuseN(idx), rmseWindFuseE(idx), ...
    rmseVtEkf(idx), rmseAlphaEkf(idx), rmseBetaEkf(idx), ...
    rmseVtFuse(idx), rmseAlphaFuse(idx), rmseBetaFuse(idx)] = localMetricRow(results.metrics.online);
[rmseVtSadsMlp(idx), rmseAlphaSadsMlp(idx), rmseBetaSadsMlp(idx)] = ...
    localAirMetricVectorRow(results.metrics.online.rmseAirSadsMlp);
[rmseVtSadsLstm(idx), rmseAlphaSadsLstm(idx), rmseBetaSadsLstm(idx)] = ...
    localAirMetricVectorRow(results.metrics.online.rmseAirSadsLstm);
lambda1CleanP95(idx) = results.lambdaStats.lambda1_clean_p95;
lambda2SevereP50(idx) = results.lambdaStats.lambda2_severe_p50;
fallbackShare(idx) = results.metrics.mode.fallbackShare;

summaryTable = table(rows, rmseWindEkfN, rmseWindEkfE, rmseWindFuseN, rmseWindFuseE, ...
    rmseVtEkf, rmseAlphaEkf, rmseBetaEkf, rmseVtFuse, rmseAlphaFuse, rmseBetaFuse, ...
    rmseVtSadsMlp, rmseAlphaSadsMlp, rmseBetaSadsMlp, ...
    rmseVtSadsLstm, rmseAlphaSadsLstm, rmseBetaSadsLstm, ...
    lambda1CleanP95, lambda2SevereP50, 100 * fallbackShare, ...
    'VariableNames', {'entry', 'rmse_wind_ekf_n', 'rmse_wind_ekf_e', ...
    'rmse_wind_fuse_n', 'rmse_wind_fuse_e', 'rmse_vt_ekf', ...
    'rmse_alpha_ekf_deg', 'rmse_beta_ekf_deg', 'rmse_vt_fuse', ...
    'rmse_alpha_fuse_deg', 'rmse_beta_fuse_deg', ...
    'rmse_vt_sads_mlp', 'rmse_alpha_sads_mlp_deg', 'rmse_beta_sads_mlp_deg', ...
    'rmse_vt_sads_lstm', 'rmse_alpha_sads_lstm_deg', 'rmse_beta_sads_lstm_deg', ...
    'lambda1_clean_p95', ...
    'lambda2_severe_p50', 'fallback_share_pct'});
writetable(summaryTable, csvFile);
end

function [windEkfN, windEkfE, windFuseN, windFuseE, vtEkf, alphaEkf, betaEkf, vtFuse, alphaFuse, betaFuse] = localMetricRow(m)
windEkfN = double(m.rmseWindEkf(1));
windEkfE = double(m.rmseWindEkf(2));
windFuseN = double(m.rmseWindFuse(1));
windFuseE = double(m.rmseWindFuse(2));
vtEkf = double(m.rmseAirEkf(1));
alphaEkf = double(rad2deg(m.rmseAirEkf(2)));
betaEkf = double(rad2deg(m.rmseAirEkf(3)));
vtFuse = double(m.rmseAirFuse(1));
alphaFuse = double(rad2deg(m.rmseAirFuse(2)));
betaFuse = double(rad2deg(m.rmseAirFuse(3)));
end

function [vt, alphaDeg, betaDeg] = localAirRmseRow(airEst, airTrue)
if isempty(airEst)
    vt = NaN;
    alphaDeg = NaN;
    betaDeg = NaN;
    return;
end
rmseAir = sqrt(mean((airEst - airTrue) .^ 2, 2));
[vt, alphaDeg, betaDeg] = localAirMetricVectorRow(rmseAir);
end

function [vt, alphaDeg, betaDeg] = localAirMetricVectorRow(v)
if isempty(v) || numel(v) < 3
    vt = NaN;
    alphaDeg = NaN;
    betaDeg = NaN;
else
    vt = double(v(1));
    alphaDeg = double(rad2deg(v(2)));
    betaDeg = double(rad2deg(v(3)));
end
end

function modeStats = localModeStats(modeVec, anomalySeverity, substLabel)
maskClean = anomalySeverity == 0;
maskSevere = substLabel ~= 0 | anomalySeverity >= 3;
maskSubst = substLabel ~= 0;
modeStats = struct();
modeStats.overall = localModeSlice(modeVec, true(size(modeVec)));
modeStats.clean = localModeSlice(modeVec, maskClean);
modeStats.severe = localModeSlice(modeVec, maskSevere);
modeStats.subst = localModeSlice(modeVec, maskSubst);
end

function stats = localModeSlice(modeVec, mask)
mask = logical(mask(:)');
stats = struct('count', sum(mask), 'hist', zeros(1, 4), 'share', nan(1, 4), ...
    'mode01Share', NaN, 'mode23Share', NaN, 'fallbackShare', NaN);
if stats.count <= 0
    return;
end
stats.hist = histcounts(round(modeVec(mask)), -0.5:1:3.5);
stats.share = stats.hist / max(sum(stats.hist), 1);
stats.mode01Share = mean(ismember(round(modeVec(mask)), [0 1]));
stats.mode23Share = mean(ismember(round(modeVec(mask)), [2 3]));
stats.fallbackShare = mean(round(modeVec(mask)) == 3);
end

function lambdaStats = localLambdaStats(lambda1, lambda2, anomalySeverity, substLabel)
maskClean = anomalySeverity == 0;
maskSevere = substLabel ~= 0 | anomalySeverity >= 3;
lambdaStats = struct();
lambdaStats.lambda1.clean = localQuantileStats(lambda1(maskClean));
lambdaStats.lambda1.severe = localQuantileStats(lambda1(maskSevere));
lambdaStats.lambda2.clean = localQuantileStats(lambda2(maskClean));
lambdaStats.lambda2.severe = localQuantileStats(lambda2(maskSevere));
lambdaStats.lambda1_clean_p95 = lambdaStats.lambda1.clean.p95;
lambdaStats.lambda2_severe_p50 = lambdaStats.lambda2.severe.p50;
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

function localPrintReplaySummary(results)
fprintf('\n==== Synthetic Online Replay gammaD=5 gammaS=20 ====\n');
if isfield(results, 'provenance')
    fprintf('Data source      : %s | intended use: %s | real flight: %d\n', ...
        char(string(localField(results.provenance, 'dataType', ""))), ...
        char(string(localField(results.provenance, 'intendedUse', ""))), ...
        logical(localField(results.provenance, 'isPhysicalFlightRecord', false)));
end
fprintf('Wind RMSE EKF    : [%.4f %.4f] ft/s\n', results.benchmark.rmseWindBase(1), results.benchmark.rmseWindBase(2));
fprintf('Wind RMSE Fusion : [%.4f %.4f] ft/s\n', results.benchmark.rmseWindFuse(1), results.benchmark.rmseWindFuse(2));
fprintf('Air RMSE EKF     : VT %.4f | alpha %.4f deg | beta %.4f deg\n', ...
    results.benchmark.rmseAirBase(1), rad2deg(results.benchmark.rmseAirBase(2)), rad2deg(results.benchmark.rmseAirBase(3)));
fprintf('Air RMSE Fusion  : VT %.4f | alpha %.4f deg | beta %.4f deg\n', ...
    results.benchmark.rmseAirFuse(1), rad2deg(results.benchmark.rmseAirFuse(2)), rad2deg(results.benchmark.rmseAirFuse(3)));
if isfield(results.benchmark, 'rmseAirSadsMlp') && any(isfinite(results.benchmark.rmseAirSadsMlp))
    fprintf('Air RMSE SADS-MLP : VT %.4f | alpha %.4f deg | beta %.4f deg\n', ...
        results.benchmark.rmseAirSadsMlp(1), rad2deg(results.benchmark.rmseAirSadsMlp(2)), ...
        rad2deg(results.benchmark.rmseAirSadsMlp(3)));
end
if isfield(results.benchmark, 'rmseAirSadsLstm') && any(isfinite(results.benchmark.rmseAirSadsLstm))
    fprintf('Air RMSE SADS-LSTM: VT %.4f | alpha %.4f deg | beta %.4f deg\n', ...
        results.benchmark.rmseAirSadsLstm(1), rad2deg(results.benchmark.rmseAirSadsLstm(2)), ...
        rad2deg(results.benchmark.rmseAirSadsLstm(3)));
end
fprintf('lambda1 clean p95 = %.4f | lambda2 severe p50 = %.4f | fallback share = %.2f%%\n', ...
    results.lambdaStats.lambda1_clean_p95, results.lambdaStats.lambda2_severe_p50, ...
    100 * results.metrics.mode.fallbackShare);
if isfield(results, 'entries') && ~isempty(results.entries) && ...
        isfield(results.entries(1), 'gateStats') && ...
        isfield(results.entries(1).gateStats, 'maneuverAeroMode1Share')
    gs = results.entries(1).gateStats;
    fprintf('Maneuver aero: mode1 %.2f%% | mode2 %.2f%% | fallback %.2f%% | lambda2 p50 %.4f | pSubst mean/max %.4f / %.4f\n', ...
        100 * gs.maneuverAeroMode1Share, 100 * gs.maneuverAeroMode2Share, ...
        100 * gs.maneuverAeroFallbackShare, gs.lambda2_maneuver_aero_p50, ...
        gs.maneuverAeroPSubstMean, gs.maneuverAeroPSubstMax);
end
if results.syntheticVsReference.available
    fprintf('Delta vs trainSummary benchmark air fuse: VT %.6f | alpha %.6f deg | beta %.6f deg\n', ...
        results.syntheticVsReference.rmseAirFuse(1), ...
        rad2deg(results.syntheticVsReference.rmseAirFuse(2)), ...
        rad2deg(results.syntheticVsReference.rmseAirFuse(3)));
end
if isfield(results, 'representativeEntry') && isstruct(results.representativeEntry)
    fprintf('Representative teaching case: %s (entry %d, distance %.4f)\n', ...
        char(string(localField(results.representativeEntry, 'entryName', ""))), ...
        double(localField(results.representativeEntry, 'entryIndex', 1)), ...
        double(localField(results.representativeEntry, 'distanceScore', NaN)));
end
end
