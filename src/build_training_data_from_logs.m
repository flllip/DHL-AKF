function [XBaseAll, YAll, meta, frameInfo] = build_training_data_from_logs(logs, cfg)
% Build supervised tensors for the stage-2 auxiliary-measurement network.
%
% Outputs:
%   XBaseAll [34 x T x S]
%   YAll     [23 x T x S]
%     1:6   nominal pseudo-measurement target
%     7:12  physical-reference target for the substitution head
%     13:15 air_true = [VT; alpha; beta]
%     16:18 reserved
%     19    frame-level abnormal label
%     20    frame-level substitution label
%     21    frame-level scenario code: 0 clean, 1 mild, 2 severe
%     22    model-mismatch weight
%     23    sensor abnormal label

if nargin < 2 || isempty(cfg)
    cfg = localDefaultCfg();
end

if ~isfield(logs, 'Yclean')
    error('Missing logs.Yclean.');
end

stateMat = double(logs.state);
controlMat = double(logs.controls);
measMat = double(logs.measurements);
yCleanMat = double(logs.Yclean);

if isfield(logs, 'air_true')
    airTrueMat = double(logs.air_true);
else
    airTrueMat = zeros(3, size(stateMat, 2));
    for k = 1:size(stateMat, 2)
        if isfield(logs, 'vw_true')
            vwRef = logs.vw_true(:, k);
        elseif isfield(logs, 'vw_prior')
            vwRef = logs.vw_prior(:, k);
        else
            vwRef = zeros(2, 1);
        end
        [vt, alpha, beta] = reconstruct_airdata_from_state_wind(stateMat(:, k), vwRef(1:2));
        airTrueMat(:, k) = [vt; alpha; beta];
    end
end

if isfield(logs, 'sampleTime')
    sampleTime = double(logs.sampleTime);
else
    sampleTime = cfg.sampleTime;
end

noiseStd = std(measMat - yCleanMat, 0, 2);
noiseStd = max(noiseStd, 1e-6);
variants = localBuildTrainingVariants(logs, cfg, stateMat, controlMat, measMat, yCleanMat, sampleTime);
numVariants = numel(variants);

if isfield(logs, 'sequenceLength')
    T = double(logs.sequenceLength);
else
    T = cfg.sequenceLength;
end
if isfield(logs, 'stride')
    stride = double(logs.stride);
else
    stride = cfg.stride;
end

N = size(stateMat, 2);
if N < T
    error('Not enough samples: N=%d < sequenceLength=%d.', N, T);
end

startIdx = 1:stride:(N - T + 1);
numBaseWindows = numel(startIdx);
useLoggedFaultLabels = isfield(logs, 'loggedFaultLabels') && logical(logs.loggedFaultLabels);
entrySource = "synthetic";
if isfield(logs, 'label_alignment_ok') || isfield(logs, 'exportSource')
    entrySource = "logged";
end

if useLoggedFaultLabels
    numSeqTotal = numBaseWindows * numVariants;
else
    scenarioCodes = [0, 1, 2];
    numSeqTotal = numBaseWindows * numel(scenarioCodes) * numVariants;
end

XBaseAll = zeros(34, T, numSeqTotal, 'single');
YAll = zeros(23, T, numSeqTotal, 'single');
needFrameInfo = nargout >= 4;
if needFrameInfo
    frameInfo = struct();
    frameInfo.anomalyMask = zeros(6, T, numSeqTotal, 'single');
    frameInfo.abnormalLabel = zeros(1, T, numSeqTotal, 'single');
    frameInfo.substLabel = zeros(1, T, numSeqTotal, 'single');
    frameInfo.scenarioCode = zeros(1, T, numSeqTotal, 'single');
else
    frameInfo = struct();
end

outIdx = 0;
meta.scenarioCounts = zeros(1, 3);
meta.abnormalFrames = 0;
meta.substitutionFrames = 0;
meta.sequenceTags = repmat(struct( ...
    'source', "", ...
    'hasSubst', false, ...
    'hasSevere', false, ...
    'hasFailureWindow', false, ...
    'dominantScenario', 0), numSeqTotal, 1);

for v = 1:numVariants
    measMatV = variants(v).measurements;
    teacherXPred = variants(v).teacherXPred;
    teacherZPred = variants(v).teacherZPred;
    teacherLambda1 = variants(v).teacherLambda1;
    teacherAnchorMat = variants(v).teacherAnchor;
    dropoutRefMat = variants(v).dropoutReference;
    modelMismatchWeightMat = variants(v).modelMismatchWeight;

for s = 1:numBaseWindows
    idx = startIdx(s):(startIdx(s) + T - 1);

    stateSeq = stateMat(:, idx);
    controlSeq = controlMat(:, idx);
    measSeq = measMatV(:, idx);
    yCleanSeq = yCleanMat(:, idx);
    airTrueSeq = airTrueMat(:, idx);
    teacherXSeq = teacherXPred(:, idx);
    teacherZSeq = teacherZPred(:, idx);
    teacherLambdaSeq = teacherLambda1(:, idx);
    teacherAnchorSeq = teacherAnchorMat(:, idx);
    dropoutRefSeq = dropoutRefMat(:, idx);
    modelMismatchWeightSeq = reshape(modelMismatchWeightMat(1, idx), [1, T]);

    if useLoggedFaultLabels
        outIdx = outIdx + 1;
        measScenario = measSeq;
        abnMask = logs.anomaly_mask(:, idx);
        sensorAbnLabel = double(any(abnMask ~= 0, 1));
        substLabel = localGetLabelSeq(logs, 'subst_label', idx, T);
        severitySeq = localGetLabelSeq(logs, 'anomaly_severity', idx, T);
        scenarioCodeSeq = zeros(1, T);
        scenarioCodeSeq(sensorAbnLabel ~= 0) = 1;
        scenarioCodeSeq((severitySeq >= 3) | (substLabel ~= 0)) = 2;
        scenarioCodeSeq = max(scenarioCodeSeq, double(modelMismatchWeightSeq > 0));
        lambdaSeq = teacherLambdaSeq;
        xPredSeq = teacherXSeq;
        zPredSeq = teacherZSeq;

        [XBaseAll(:, :, outIdx), YAll(:, :, outIdx)] = localBuildScenario( ...
            stateSeq, controlSeq, measScenario, yCleanSeq, airTrueSeq, xPredSeq, zPredSeq, lambdaSeq, ...
            abnMask, sensorAbnLabel, substLabel, scenarioCodeSeq, modelMismatchWeightSeq);
        if needFrameInfo
            frameInfo.anomalyMask(:, :, outIdx) = single(abnMask ~= 0);
            frameInfo.abnormalLabel(1, :, outIdx) = single(max(sensorAbnLabel, double(modelMismatchWeightSeq > 0)));
            frameInfo.substLabel(1, :, outIdx) = single(substLabel);
            frameInfo.scenarioCode(1, :, outIdx) = single(scenarioCodeSeq);
        end

        dominantScenario = localDominantScenarioCode(scenarioCodeSeq);
        meta.scenarioCounts(dominantScenario + 1) = meta.scenarioCounts(dominantScenario + 1) + 1;
        meta.abnormalFrames = meta.abnormalFrames + sum(max(sensorAbnLabel, double(modelMismatchWeightSeq > 0)));
        meta.substitutionFrames = meta.substitutionFrames + sum(substLabel);
        meta.sequenceTags(outIdx).source = "logged";
        meta.sequenceTags(outIdx).hasSubst = any(substLabel ~= 0);
        meta.sequenceTags(outIdx).hasSevere = any(scenarioCodeSeq == 2);
        meta.sequenceTags(outIdx).hasFailureWindow = localHasFailureWindow(logs, idx, scenarioCodeSeq);
        meta.sequenceTags(outIdx).dominantScenario = dominantScenario;
    else
        for sc = scenarioCodes
            outIdx = outIdx + 1;
            switch sc
                case 0
                    measScenario = measSeq;
                    abnMask = zeros(6, T);
                    sensorAbnLabel = zeros(1, T);
                    substLabel = zeros(1, T);
                    lambdaSeq = teacherLambdaSeq;
                    xPredSeq = teacherXSeq;
                    zPredSeq = teacherZSeq;
                    scenarioCodeSeq = zeros(1, T);

                case 1
                    [measScenario, abnMask, ~, ~, substLabel] = ...
                        injectSequenceAnomalies(measSeq, dropoutRefSeq, noiseStd, cfg.anomaly.mild);
                    sensorAbnLabel = double(any(abnMask ~= 0, 1));
                    teacherScenario = build_teacher_anchor_signals(stateSeq, controlSeq, teacherAnchorSeq, measScenario, sampleTime);
                    lambdaSeq = teacherScenario.lambda1;
                    xPredSeq = teacherScenario.xPred;
                    zPredSeq = teacherScenario.zPred;
                    scenarioCodeSeq = ones(1, T);

                otherwise
                    [measScenario, abnMask, ~, ~, substLabel] = ...
                        injectSequenceAnomalies(measSeq, dropoutRefSeq, noiseStd, cfg.anomaly.severe);
                    sensorAbnLabel = double(any(abnMask ~= 0, 1));
                    teacherScenario = build_teacher_anchor_signals(stateSeq, controlSeq, teacherAnchorSeq, measScenario, sampleTime);
                    lambdaSeq = teacherScenario.lambda1;
                    xPredSeq = teacherScenario.xPred;
                    zPredSeq = teacherScenario.zPred;
                    scenarioCodeSeq = 2 * ones(1, T);
            end
            scenarioCodeSeq = max(scenarioCodeSeq, double(modelMismatchWeightSeq > 0));

            [XBaseAll(:, :, outIdx), YAll(:, :, outIdx)] = localBuildScenario( ...
                stateSeq, controlSeq, measScenario, yCleanSeq, airTrueSeq, xPredSeq, zPredSeq, lambdaSeq, ...
                abnMask, sensorAbnLabel, substLabel, scenarioCodeSeq, modelMismatchWeightSeq);
            if needFrameInfo
                frameInfo.anomalyMask(:, :, outIdx) = single(abnMask ~= 0);
                frameInfo.abnormalLabel(1, :, outIdx) = single(max(sensorAbnLabel, double(modelMismatchWeightSeq > 0)));
                frameInfo.substLabel(1, :, outIdx) = single(substLabel);
                frameInfo.scenarioCode(1, :, outIdx) = single(scenarioCodeSeq);
            end

            meta.scenarioCounts(sc + 1) = meta.scenarioCounts(sc + 1) + 1;
            meta.abnormalFrames = meta.abnormalFrames + sum(max(sensorAbnLabel, double(modelMismatchWeightSeq > 0)));
            meta.substitutionFrames = meta.substitutionFrames + sum(substLabel);
            meta.sequenceTags(outIdx).source = entrySource;
            meta.sequenceTags(outIdx).hasSubst = any(substLabel ~= 0);
            meta.sequenceTags(outIdx).hasSevere = sc == 2;
            meta.sequenceTags(outIdx).hasFailureWindow = sc == 2;
            meta.sequenceTags(outIdx).dominantScenario = sc;
        end
    end
end
end

meta.numSequences = numSeqTotal;
meta.numBaseWindows = numBaseWindows;
meta.sequenceLength = T;
meta.numSamples = N;
meta.sampleTime = sampleTime;
end

function variants = localBuildTrainingVariants(logs, cfg, stateMat, controlMat, measMat, yCleanMat, sampleTime)
[teacherXPred, teacherZPred, teacherLambda1] = localBuildTeacherFromLogs( ...
    logs, stateMat, controlMat, measMat, sampleTime);

template = struct( ...
    'measurements', measMat, ...
    'teacherXPred', teacherXPred, ...
    'teacherZPred', teacherZPred, ...
    'teacherLambda1', teacherLambda1, ...
    'teacherAnchor', measMat, ...
    'dropoutReference', teacherZPred, ...
    'modelMismatchWeight', zeros(1, size(measMat, 2)), ...
    'aeroCoeffScale', ones(6, 1), ...
    'isAeroBiased', false, ...
    'isManeuverAeroBiased', false);
variants = template;

aeroCfg = localResolveAeroBiasCfg(cfg);
if aeroCfg.enabled && aeroCfg.augmentCopies >= 1 && isfield(logs, 'vw_true')
    rngState = rng;
    cleanupObj = onCleanup(@() rng(rngState)); %#ok<NASGU>
    rng(double(aeroCfg.seed) + localSeedOffset(logs, stateMat), 'twister');

    for i = 1:aeroCfg.augmentCopies
        coeffScale = localRandomAeroCoeffScale(aeroCfg);
        measBiased = localApplyAeroBiasToMeasurements( ...
            stateMat, controlMat, measMat, yCleanMat, double(logs.vw_true), coeffScale);
        teacher = build_teacher_anchor_signals(stateMat, controlMat, yCleanMat, measBiased, sampleTime);

        v = template;
        v.measurements = measBiased;
        v.teacherXPred = double(teacher.xPred);
        v.teacherZPred = double(teacher.zPred);
        v.teacherLambda1 = double(teacher.lambda1);
        v.teacherAnchor = yCleanMat;
        v.dropoutReference = double(teacher.zPred) + (measBiased - measMat);
        v.aeroCoeffScale = coeffScale(:);
        v.isAeroBiased = true;
        variants(end + 1) = v; %#ok<AGROW>
    end
end

maneuverCfg = localResolveManeuverAeroBiasCfg(cfg);
if maneuverCfg.enabled && maneuverCfg.augmentCopies >= 1 && isfield(logs, 'vw_true')
    rngState = rng;
    cleanupObj = onCleanup(@() rng(rngState)); %#ok<NASGU>
    rng(double(maneuverCfg.seed) + localSeedOffset(logs, stateMat), 'twister');
    [~, weight] = localManeuverAeroBiasWeight(logs, yCleanMat, maneuverCfg);

    for i = 1:maneuverCfg.augmentCopies
        coeffScale = localRandomAeroCoeffScale(maneuverCfg);
        [measBiased, modelMismatchWeight] = localApplyManeuverAeroBiasToMeasurements( ...
            stateMat, controlMat, measMat, yCleanMat, double(logs.vw_true), coeffScale, weight);
        teacher = build_teacher_anchor_signals(stateMat, controlMat, yCleanMat, measBiased, sampleTime);

        v = template;
        v.measurements = measBiased;
        v.teacherXPred = double(teacher.xPred);
        v.teacherZPred = double(teacher.zPred);
        v.teacherLambda1 = double(teacher.lambda1);
        v.teacherAnchor = yCleanMat;
        v.dropoutReference = double(teacher.zPred) + (measBiased - measMat);
        v.modelMismatchWeight = modelMismatchWeight;
        v.aeroCoeffScale = coeffScale(:);
        v.isManeuverAeroBiased = true;
        variants(end + 1) = v; %#ok<AGROW>
    end
end
end

function [teacherXPred, teacherZPred, teacherLambda1] = localBuildTeacherFromLogs(logs, stateMat, controlMat, measMat, sampleTime)
if isfield(logs, 'teacher_xPred') && isfield(logs, 'teacher_zPred') && isfield(logs, 'teacher_lambda1')
    teacherXPred = double(logs.teacher_xPred);
    teacherZPred = double(logs.teacher_zPred);
    teacherLambda1 = double(logs.teacher_lambda1);
elseif isfield(logs, 'measurements_nominal')
    teacher = build_teacher_anchor_signals( ...
        stateMat, controlMat, double(logs.measurements_nominal), measMat, sampleTime);
    teacherXPred = double(teacher.xPred);
    teacherZPred = double(teacher.zPred);
    teacherLambda1 = double(teacher.lambda1);
else
    teacher = build_teacher_signals(stateMat, controlMat, measMat, sampleTime);
    teacherXPred = double(teacher.xPred);
    teacherZPred = double(teacher.zPred);
    teacherLambda1 = double(teacher.lambda1);
end
end

function aeroCfg = localResolveAeroBiasCfg(cfg)
aeroCfg = struct('enabled', false, 'minPct', 0.05, 'maxPct', 0.10, ...
    'augmentCopies', 0, 'seed', 20260507, 'mode', "per_entry_signed");
if ~isstruct(cfg) || ~isfield(cfg, 'aeroBias') || ~isstruct(cfg.aeroBias)
    return;
end
names = fieldnames(cfg.aeroBias);
for i = 1:numel(names)
    aeroCfg.(names{i}) = cfg.aeroBias.(names{i});
end
aeroCfg.enabled = logical(aeroCfg.enabled);
aeroCfg.minPct = max(0, double(aeroCfg.minPct));
aeroCfg.maxPct = max(aeroCfg.minPct, double(aeroCfg.maxPct));
aeroCfg.augmentCopies = max(0, round(double(aeroCfg.augmentCopies)));
aeroCfg.seed = double(aeroCfg.seed);
aeroCfg.mode = string(aeroCfg.mode);
end

function offset = localSeedOffset(logs, stateMat)
offset = 0;
if isfield(logs, 'flightIdx')
    offset = offset + 101 * round(double(logs.flightIdx));
end
if isfield(logs, 'datasetRole')
    roleText = char(string(logs.datasetRole));
    offset = offset + 17 * sum(double(roleText));
end
cols = unique(round(linspace(1, size(stateMat, 2), min(20, size(stateMat, 2)))));
sample = stateMat(:, cols);
offset = offset + mod(round(sum(abs(sample(:)))), 1000000);
end

function coeffScale = localRandomAeroCoeffScale(aeroCfg)
mag = aeroCfg.minPct + (aeroCfg.maxPct - aeroCfg.minPct) * rand(6, 1);
switch lower(string(aeroCfg.mode))
    case {"all_positive", "positive", "plus"}
        signVec = ones(6, 1);
    case {"all_negative", "negative", "minus"}
        signVec = -ones(6, 1);
    case {"longitudinal"}
        signVec = [1; 0; 1; 0; 1; 0];
    otherwise
        signVec = sign(rand(6, 1) - 0.5);
        signVec(signVec == 0) = 1;
end
coeffScale = 1 + signVec .* mag;
end

function measOut = localApplyAeroBiasToMeasurements(stateMat, controlMat, measMat, yCleanMat, windTrue, coeffScale)
coeffScale = double(coeffScale(:));
N = min([size(stateMat, 2), size(controlMat, 2), size(measMat, 2), ...
    size(yCleanMat, 2), size(windTrue, 2)]);
yBias = zeros(6, N);
for k = 1:N
    yBias(:, k) = wind_estimation_model_numeric_aerobias( ...
        stateMat(:, k), controlMat(:, k), windTrue(:, k), coeffScale);
end
measOut = measMat(:, 1:N) + (yBias - yCleanMat(:, 1:N));
end

function maneuverCfg = localResolveManeuverAeroBiasCfg(cfg)
maneuverCfg = struct( ...
    'enabled', false, ...
    'augmentCopies', 0, ...
    'minPct', 0.10, ...
    'maxPct', 0.20, ...
    'mode', "signed", ...
    'seed', 20260515, ...
    'quantilePct', 90, ...
    'alphaRateThresholdDegPerSec', NaN, ...
    'rampSec', 0.35, ...
    'mergeGapSec', 0.15, ...
    'minDurationSec', 0.08);
if ~isstruct(cfg) || ~isfield(cfg, 'maneuverAeroBias') || ~isstruct(cfg.maneuverAeroBias)
    return;
end
names = fieldnames(cfg.maneuverAeroBias);
for i = 1:numel(names)
    maneuverCfg.(names{i}) = cfg.maneuverAeroBias.(names{i});
end
maneuverCfg.enabled = logical(maneuverCfg.enabled);
maneuverCfg.augmentCopies = max(0, round(double(maneuverCfg.augmentCopies)));
maneuverCfg.minPct = max(0, double(maneuverCfg.minPct));
maneuverCfg.maxPct = max(maneuverCfg.minPct, double(maneuverCfg.maxPct));
maneuverCfg.mode = string(maneuverCfg.mode);
maneuverCfg.seed = double(maneuverCfg.seed);
maneuverCfg.quantilePct = min(max(double(maneuverCfg.quantilePct), 50), 99.9);
maneuverCfg.rampSec = max(0, double(maneuverCfg.rampSec));
maneuverCfg.mergeGapSec = max(0, double(maneuverCfg.mergeGapSec));
maneuverCfg.minDurationSec = max(0, double(maneuverCfg.minDurationSec));
end

function [measOut, modelMismatchWeight] = localApplyManeuverAeroBiasToMeasurements( ...
    stateMat, controlMat, measMat, yCleanMat, windTrue, coeffScale, weight)
coeffScale = double(coeffScale(:));
N = min([size(stateMat, 2), size(controlMat, 2), size(measMat, 2), ...
    size(yCleanMat, 2), size(windTrue, 2), numel(weight)]);
measOut = measMat;
modelMismatchWeight = zeros(1, size(measMat, 2));
if N < 1
    return;
end

yBias = zeros(6, N);
for k = 1:N
    yBias(:, k) = wind_estimation_model_numeric_aerobias( ...
        stateMat(:, k), controlMat(:, k), windTrue(:, k), coeffScale);
end
weightRow = reshape(double(weight(1:N)), [1, N]);
modelMismatchWeight(1, 1:N) = weightRow;
measOut(:, 1:N) = measMat(:, 1:N) + (yBias - yCleanMat(:, 1:N)) .* repmat(weightRow, 6, 1);
end

function [mask, weight] = localManeuverAeroBiasWeight(logs, yClean, cfg)
N = min([size(logs.state, 2), size(logs.controls, 2), size(yClean, 2)]);
dt = double(localField(logs, 'sampleTime', localField(cfg, 'sampleTime', 0.01)));
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
alphaRateDeg = abs([0, diff(alpha * 180 / pi)] / dt);
nzProxy = abs(double(yClean(3, 1:N))) / 32.17;

qThr = localPercentileThreshold(qAbs, cfg.quantilePct);
rThr = localPercentileThreshold(rAbs, cfg.quantilePct);
deThr = localPercentileThreshold(deAbs, cfg.quantilePct);
nzThr = localPercentileThreshold(nzProxy, cfg.quantilePct);
if isfield(cfg, 'alphaRateThresholdDegPerSec') && isfinite(cfg.alphaRateThresholdDegPerSec)
    alphaRateThr = double(cfg.alphaRateThresholdDegPerSec);
else
    alphaRateThr = localPercentileThreshold(alphaRateDeg, cfg.quantilePct);
end

rawMask = (qAbs > qThr) | (rAbs > rThr) | (deAbs > deThr) | ...
    (alphaRateDeg > alphaRateThr) | (nzProxy > nzThr);
rawMask = localMergeShortGaps(rawMask, round(cfg.mergeGapSec / dt));
mask = localMinimumDurationMask(rawMask, round(cfg.minDurationSec / dt));
weight = localRaisedCosineWeight(mask, round(cfg.rampSec / dt));
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

function value = localField(s, name, defaultValue)
value = defaultValue;
if isstruct(s) && isfield(s, name)
    value = s.(name);
end
end

function [XSeq, YSeq] = localBuildScenario(stateSeq, controlSeq, measSeq, yCleanSeq, airTrueSeq, xPredSeq, zPredSeq, lambdaSeq, abnMask, sensorAbnLabel, substLabel, scenarioCodeSeq, modelMismatchWeightSeq)
T = size(measSeq, 2);
XSeq = zeros(34, T, 'single');
YSeq = zeros(23, T, 'single');
ySubTargetSeq = yCleanSeq;
modelMismatchWeightSeq = reshape(double(modelMismatchWeightSeq), [1, T]);
sensorAbnLabel = reshape(double(sensorAbnLabel), [1, T]);
abnLabel = max(sensorAbnLabel, double(modelMismatchWeightSeq > 0));

for kk = 1:T
    xBase = compose_lstm_fusion_input(xPredSeq(:, kk), measSeq(:, kk), stateSeq(:, kk), controlSeq(:, kk), zPredSeq(:, kk));
    XSeq(:, kk) = single(xBase);
    YSeq(1:6, kk) = single(yCleanSeq(:, kk));
    YSeq(7:12, kk) = single(ySubTargetSeq(:, kk));
    YSeq(13:15, kk) = single(airTrueSeq(:, kk));
    YSeq(16, kk) = single(lambdaSeq(kk));
    YSeq(17:18, kk) = single(0);
    YSeq(19, kk) = single(abnLabel(kk));
    YSeq(20, kk) = single(substLabel(kk));
    YSeq(21, kk) = single(scenarioCodeSeq(kk));
    YSeq(22, kk) = single(modelMismatchWeightSeq(kk));
    YSeq(23, kk) = single(sensorAbnLabel(kk));
end
end

function seq = localGetLabelSeq(logs, fieldName, idx, T)
if isfield(logs, fieldName)
    values = double(logs.(fieldName));
    if isvector(values)
        seq = reshape(values(idx), [1, T]);
    else
        seq = reshape(values(:, idx), [1, T]);
    end
else
    seq = zeros(1, T);
end
end

function code = localDominantScenarioCode(scenarioCodeSeq)
counts = [sum(scenarioCodeSeq == 0), sum(scenarioCodeSeq == 1), sum(scenarioCodeSeq == 2)];
[~, idx] = max(counts);
code = idx - 1;
end

function tf = localHasFailureWindow(logs, idx, scenarioCodeSeq)
tf = any(scenarioCodeSeq == 2);
if isfield(logs, 'mode')
    modeSeq = double(logs.mode(:, idx));
    tf = tf || any(modeSeq(:) == 3);
end
end

function cfg = localDefaultCfg()
cfg.sequenceLength = 120;
cfg.stride = 12;
cfg.sampleTime = 0.05;
cfg.anomaly.mild = localAnomalyCfg([1, 1], [0.9, 1.4, 1.8], [6, 28], [1, 2, 5, 6], false);
cfg.anomaly.severe = localAnomalyCfg([1, 2], [1.2, 2.0, 3.0], [10, 60], [1, 2, 3, 4, 5, 6], false);
cfg.aeroBias = struct('enabled', false, 'minPct', 0.05, 'maxPct', 0.10, ...
    'augmentCopies', 0, 'seed', 20260507, 'mode', "per_entry_signed");
cfg.maneuverAeroBias = struct('enabled', false, 'augmentCopies', 0, ...
    'minPct', 0.10, 'maxPct', 0.20, 'mode', "signed", 'seed', 20260515, ...
    'quantilePct', 90, 'rampSec', 0.35, 'mergeGapSec', 0.15, 'minDurationSec', 0.08);
end

function cfg = localAnomalyCfg(numEventsRange, severityAmp, segmentRange, typePool, forceSubst)
cfg.numEventsRange = numEventsRange;
cfg.severityAmp = severityAmp;
cfg.segmentRange = segmentRange;
cfg.typePool = typePool;
cfg.substTypes = [3, 4, 6];
cfg.forceSubst = forceSubst;
end

function [measOut, anomalyMask, anomalyType, anomalySeverity, substLabel] = injectSequenceAnomalies(measIn, zPred, noiseStd, cfgAnomaly)
[C, T] = size(measIn);
measOut = measIn;
anomalyMask = zeros(C, T);
anomalyType = zeros(1, T);
anomalySeverity = zeros(1, T);
substLabel = zeros(1, T);

numEvents = randi(cfgAnomaly.numEventsRange);

for ev = 1:numEvents
    typeId = cfgAnomaly.typePool(randi(numel(cfgAnomaly.typePool)));
    severityId = randi([1, 3]);
    amp = cfgAnomaly.severityAmp(severityId);
    segLen = randi(cfgAnomaly.segmentRange);
    segLen = min(segLen, T);
    t0 = randi([1, T - segLen + 1]);
    idx = t0:(t0 + segLen - 1);
    chCount = randi([1, min(3, C)]);
    channels = randperm(C, chCount);

    switch typeId
        case 1  % impulsive spike
            spikePts = idx(rand(1, numel(idx)) < 0.35);
            if isempty(spikePts)
                spikePts = idx(randi(numel(idx)));
            end
            for ch = channels
                measOut(ch, spikePts) = measOut(ch, spikePts) + ...
                    amp * noiseStd(ch) * sign(randn(1, numel(spikePts))) .* (3 + rand(1, numel(spikePts)));
            end

        case 2  % bias drift
            drift = linspace(0, amp, numel(idx));
            for ch = channels
                measOut(ch, idx) = measOut(ch, idx) + noiseStd(ch) * drift;
            end

        case 3  % stuck/hold
            for ch = channels
                holdVal = measOut(ch, max(1, t0 - 1));
                measOut(ch, idx) = holdVal;
            end

        case 4  % dropout
            for ch = channels
                measOut(ch, idx) = zPred(ch, idx);
            end

        case 5  % scale factor
            scale = 1 + 0.08 * amp * sign(randn);
            for ch = channels
                measOut(ch, idx) = scale * measOut(ch, idx);
            end

        otherwise % short delay/misalignment
            lag = min(randi([1, 4]), t0 - 1);
            for ch = channels
                if lag > 0
                    measOut(ch, idx) = measOut(ch, idx - lag);
                end
            end
    end

    anomalyMask(channels, idx) = 1;
    anomalyType(idx) = typeId;
    anomalySeverity(idx) = max(anomalySeverity(idx), severityId);
    if cfgAnomaly.forceSubst
        substLabel(idx) = 1;
    elseif severityId >= 3 || any(cfgAnomaly.substTypes == typeId)
        substLabel(idx) = 1;
    end
end
end
