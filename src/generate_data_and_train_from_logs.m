% Train the final stage-2 EKF-LSTM fusion network.
%
% Default workflow:
%   1) generate full-flight-envelope synthetic logs
%   2) train one- or two-regression-head auxiliary-measurement network
%   3) calibrate fixed second-test covariance
%   4) evaluate hold-out full-envelope benchmark against EKF
%
rng(42);

cfg.data.source = "synthetic_full_envelope";
cfg.synthetic = localDefaultSyntheticCfg();
cfg.model.profile = "auto";
cfg.model.regressionHeads = 2;
cfg.model.hidden = [96, 64];
cfg.model.fc = 64;
cfg.model.dropout = 0.15;
cfg.model.multiscaleWindows = [1, 3, 7, 15];
cfg.train.numEpochs = 35;
cfg.train.miniBatchSize = 16;
cfg.train.learnRate = 1e-3;
cfg.train.outputModelFile = "lstm_akf_net.mat";
cfg.train.initModelFile = "";
cfg.train.seed = [];
cfg.train.skipBenchmark = false;
cfg.loss.mode = "stage2_dual_head_multitask";
cfg.loss.huberDelta = 1.0;
cfg.loss.anchorWeight = 0.0;
cfg.loss.substWeight = 0.8;
cfg.loss.airWeight = 0.05;
cfg.loss.clsWeight = 0.15;
cfg.loss.nllWeight = 0.0;
cfg.loss.cleanResidualWeight = 0.25;
cfg.loss.obsWeightPower = 0.5;
cfg.loss.obsWeightClip = [0.25, 4.0];
cfg.loss.channelWeights = ones(6, 1, 'single');
cfg.loss.maneuverAeroDenoiseBoost = 2.0;
cfg.loss.maneuverAeroAirBoost = 0.25;
cfg.eval.substThreshold = 0.60;
cfg.eval.batchSize = [];
cfg.filter.baseRcDiag = single([0.44, 0.44, 0.44, 0.2, 0.2, 0.2] .^ 2);
cfg.filter.substEnterProb = 0.70;
cfg.filter.substExitProb = 0.35;
cfg.filter.substEnterCount = 10;
cfg.filter.substExitCount = 15;
cfg.filter.qBoostHorizon = 25;
cfg.filter.gateThresholds = struct('T1', 16.812, 'T2', 14.449, 'T1ch', 6.635, 'T2ch', 5.024);
cfg.filter.recoveryEnabled = true;
cfg.filter.recoveryFallbackEnterCount = 12;
cfg.filter.recoveryLambdaEnter = 2 * cfg.filter.gateThresholds.T1;
cfg.filter.recoveryCtxRawOkMinChannels = 4;
cfg.filter.recoveryHoldOffAuxCount = 50;
cfg.filter.recoveryQBoostHorizon = 100;
cfg.filter.recoveryExitCleanCount = 5;
cfg.data.loggedPattern = "real_training_logs_stage2_*.mat";
cfg.data.mix = struct('syntheticFraction', 0.50, 'loggedFraction', 0.50, 'loggedOversampleFactor', 2);
cfg.runtime.modelMode = "dual_head";
cfg.runtime.usePredictedVariance = false;
cfg.runtime.useGateCalibration = true;
cfg.runtime.gateCalibrationFile = "stage2_clean_gate_calibration.mat";
cfg.runtime.gateThresholds = cfg.filter.gateThresholds;
cfg.train.oofEpochs = 4;
cfg.meta = struct();
cfg.meta.enabled = false;
cfg.meta.devUseValEntries = true;
cfg.meta.maxRejectSamplesSvr = 7000;
cfg.meta.maxRejectSamplesGate = 18000;
cfg.meta.svrBoxConstraint = 5;
cfg.meta.gateReg = 0.01;
cfg.meta.gateLearnRate = 0.15;
cfg.meta.gateNumIters = 220;
cfg.meta.rejectThresholdSource = "T1";
cfg.meta.oofRmseImproveMinPct = 5.0;
cfg.meta.oofBetterShareMin = 0.55;
cfg.meta.holdoutRmseRatioMax = 1.02;
cfg.meta.holdoutBetterShareMin = 0.50;
cfg.meta.recoveryCleanWindowSec = 2.0;
cfg.meta.channelRuleMode = "oof_holdout_dual_screen";
cfg.meta.featureSet = "legacy";
cfg.meta.featureConfig = stage2_meta_feature_spec(cfg.meta.featureSet);
cfg.meta.includeRecoveryCleanReject = false;
cfg.meta.recoveryCleanRejectWeightBoost = 1.0;
cfg.loss.rejectBoostFactor = 1.0;
cfg.loss.cleanZeroResidualWeight = 0.05;
cfg.loss.substRejectBoost = 1.0;

if evalin('base', "exist('trainOverride','var')")
    override = evalin('base', 'trainOverride');
    if isstruct(override)
        cfg = mergeStruct(cfg, override);
    else
        warning('trainOverride exists but is not struct, ignored.');
    end
end

cfg.loss.gateThresholds = cfg.filter.gateThresholds;

[trainEntries, valEntries, testEntries, duplicateInfo, sourceInfo] = buildDataSource(cfg);

[XTrainBase, YTrainRaw, trainMeta, trainSeqInfo] = collectDataset(trainEntries, cfg.synthetic);
[XValBase, YValRaw, valMeta, valSeqInfo] = collectDataset(valEntries, cfg.synthetic);
[XTestBase, YTestRaw, testMeta, testSeqInfo] = collectDataset(testEntries, cfg.synthetic);

if string(cfg.data.source) == "mixed_finetune"
    [XTrainBase, YTrainRaw, trainSeqInfo] = rebalanceMixedDataset(XTrainBase, YTrainRaw, trainSeqInfo, cfg.data.mix);
end

[obsInfoLike, obsRowNorm] = estimateObservationChannelInfo(trainEntries, cfg.filter.baseRcDiag);
cfg.loss.channelWeights = computeObservationChannelWeights(obsInfoLike, cfg.loss);
fprintf('Observation info-like [ud vd wd pd qd rd]: ');
disp(obsInfoLike(:)');
fprintf('Loss channel weights [ud vd wd pd qd rd]: ');
disp(cfg.loss.channelWeights(:)');

XTrainRaw = append_causal_multiscale_features(XTrainBase, cfg.model.multiscaleWindows);
XValRaw = append_causal_multiscale_features(XValBase, cfg.model.multiscaleWindows);
XTestRaw = append_causal_multiscale_features(XTestBase, cfg.model.multiscaleWindows);

numTrainSeq = size(XTrainRaw, 3);
numValSeq = size(XValRaw, 3);
numTestSeq = size(XTestRaw, 3);
if numTrainSeq < 10 || numValSeq < 1 || numTestSeq < 1
    error('Dataset split too small: train=%d val=%d test=%d.', numTrainSeq, numValSeq, numTestSeq);
end

if cfg.model.profile == "auto"
    if numTrainSeq >= 8000
        activeProfile = "large";
    elseif numTrainSeq >= 2500
        activeProfile = "medium";
    else
        activeProfile = "small";
    end
else
    activeProfile = cfg.model.profile;
end

switch activeProfile
    case "large"
        cfg.model.hidden = [128, 96];
        cfg.model.fc = 96;
        cfg.model.dropout = 0.20;
        cfg.train.miniBatchSize = 32;
    case "medium"
        cfg.model.hidden = [96, 64];
        cfg.model.fc = 64;
        cfg.model.dropout = 0.15;
        cfg.train.miniBatchSize = 16;
    otherwise
        cfg.model.hidden = [64, 48];
        cfg.model.fc = 48;
        cfg.model.dropout = 0.10;
        cfg.train.miniBatchSize = 8;
end

if isempty(cfg.eval.batchSize) || cfg.eval.batchSize <= 0
    cfg.eval.batchSize = cfg.train.miniBatchSize;
end

if ~ismember(double(cfg.model.regressionHeads), [1, 2])
    error('regressionHeads must be 1 or 2.');
end
if ~isempty(cfg.train.seed)
    rng(double(cfg.train.seed));
end

fprintf('Active training profile: %s | train=%d | val=%d | test=%d\n', ...
    activeProfile, numTrainSeq, numValSeq, numTestSeq);

useInitModel = strlength(string(cfg.train.initModelFile)) > 0 && isfile(char(cfg.train.initModelFile));
if useInitModel
    initModelFile = char(cfg.train.initModelFile);
    SInit = load(initModelFile, 'net', 'normStats', 'trainSummary');
    net = SInit.net;
    normStats = SInit.normStats;
    fprintf('Fine-tune init model: %s\n', initModelFile);
else
    muIn = mean(XTrainRaw, [2, 3]);
    sigIn = std(XTrainRaw, 0, [2, 3]);
    sigIn(sigIn < 1e-6) = 1;

    muY = mean(YTrainRaw(1:6, :, :), [2, 3]);
    sigY = std(YTrainRaw(1:6, :, :), 0, [2, 3]);
    sigY(sigY < 1e-6) = 1;

    deltaTrainRaw = YTrainRaw(1:6, :, :) - XTrainBase(5:10, :, :);
    muDelta = mean(deltaTrainRaw, [2, 3]);
    sigDelta = std(deltaTrainRaw, 0, [2, 3]);
    sigDelta(sigDelta < 1e-6) = 1;

    muAir = mean(YTrainRaw(13:15, :, :), [2, 3]);
    sigAir = std(YTrainRaw(13:15, :, :), 0, [2, 3]);
    sigAir(sigAir < 1e-6) = 1;

    normStats.muIn = single(muIn);
    normStats.sigIn = single(sigIn);
    normStats.muY = single(muY);
    normStats.sigY = single(sigY);
    normStats.muDelta = single(muDelta);
    normStats.sigDelta = single(sigDelta);
    normStats.muAir = single(muAir);
    normStats.sigAir = single(sigAir);

    layers = [
        sequenceInputLayer(size(XTrainRaw, 1), "Name", "input")
        lstmLayer(cfg.model.hidden(1), "OutputMode", "sequence", "Name", "lstm1")
        dropoutLayer(cfg.model.dropout, "Name", "drop1")
        lstmLayer(cfg.model.hidden(2), "OutputMode", "sequence", "Name", "lstm2")
        fullyConnectedLayer(cfg.model.fc, "Name", "fc1")
        reluLayer("Name", "relu1")
        fullyConnectedLayer(23, "Name", "fc_out")
    ];
    fullNet = dlnetwork(layerGraph(layers));
    if cfg.model.regressionHeads == 1
        % Select the denoising rows plus all shared auxiliary outputs.
        % This also pairs their initialization with the two-head model.
        keepRows = [1:6, 13:23];
        initializedLayers = fullNet.Layers;
        fullOutput = initializedLayers(end);
        initializedLayers(end) = fullyConnectedLayer(17, "Name", "fc_out", ...
            "Weights", fullOutput.Weights(keepRows, :), ...
            "Bias", fullOutput.Bias(keepRows));
        net = dlnetwork(layerGraph(initializedLayers));
    else
        net = fullNet;
    end
end

normStats.varLower = single(0.05 * cfg.filter.baseRcDiag(:));
normStats.varUpper = single(10.0 * cfg.filter.baseRcDiag(:));
normStats.multiscaleWindows = single(cfg.model.multiscaleWindows(:));
normStats.baseInputDim = int32(size(XTrainBase, 1));
normStats.netInputDim = int32(size(XTrainRaw, 1));
normStats.sigIn(normStats.sigIn < 1e-6) = 1;
normStats.sigY(normStats.sigY < 1e-6) = 1;

clear XValBase XTestBase XTrainBase

trailingAvg = [];
trailingAvgSq = [];
iteration = 0;
numItersPerEpoch = ceil(numTrainSeq / cfg.train.miniBatchSize);

fprintf('Training started...\n');
fprintf('Loss mode: %s\n', cfg.loss.mode);
if ~isempty(cfg.train.seed)
    rng(double(cfg.train.seed) + 1000);
end

for epoch = 1:cfg.train.numEpochs
    perm = randperm(numTrainSeq);
    epochLoss = 0;

    for ii = 1:numItersPerEpoch
        iteration = iteration + 1;
        batchStart = (ii - 1) * cfg.train.miniBatchSize + 1;
        batchEnd = min(ii * cfg.train.miniBatchSize, numTrainSeq);
        batchIdx = perm(batchStart:batchEnd);

        XBatch = dlarray(normalizeInputBatch(XTrainRaw(:, :, batchIdx), normStats), "CTB");
        YBatch = dlarray(single(YTrainRaw(:, :, batchIdx)), "CTB");

        [loss, gradients] = dlfeval(@modelGradientsStage2, net, XBatch, YBatch, normStats, cfg.loss);
        [net, trailingAvg, trailingAvgSq] = adamupdate( ...
            net, gradients, trailingAvg, trailingAvgSq, iteration, cfg.train.learnRate);

        epochLoss = epochLoss + double(gather(extractdata(loss)));
    end

    valLoss = computeDatasetLossBatched(net, XValRaw, YValRaw, normStats, cfg.loss, cfg.eval.batchSize);
    fprintf('Epoch %3d/%3d | TrainLoss %.6f | ValLoss %.6f\n', ...
        epoch, cfg.train.numEpochs, epochLoss / numItersPerEpoch, ...
        valLoss);
end

trainMetrics = evaluateDatasetMetrics(net, XTrainRaw, YTrainRaw, normStats, cfg.eval);
valMetrics = evaluateDatasetMetrics(net, XValRaw, YValRaw, normStats, cfg.eval);
testMetrics = evaluateDatasetMetrics(net, XTestRaw, YTestRaw, normStats, cfg.eval);

gammaD = deriveCalibratedRScale(valMetrics.mild.rmseDenoise, valMetrics.mild.rmseMeas, 0.08, 1.0);
gammaS = deriveCalibratedRScale(valMetrics.severe.rmseSubst, valMetrics.severe.rmseMeas, 0.05, 1.0);

printMetrics('Train', trainMetrics);
printMetrics('Val', valMetrics);
printMetrics('Test', testMetrics);
fprintf('Calibrated gammaD [ud vd wd pd qd rd]: ');
disp(gammaD(:)');
fprintf('Calibrated gammaS [ud vd wd pd qd rd]: ');
disp(gammaS(:)');

svrBundle = struct('available', false);
gateBundle = struct('available', false);
metaSummary = struct('available', false);
holdoutResidualSummary = struct('available', false);
metaTrainingEnabled = localMetaTrainingEnabled(cfg);
if metaTrainingEnabled
    fprintf('Training OOF meta models (SVR + gate)...\n');
    metaEntries = trainEntries;
    if isfield(cfg.meta, 'devUseValEntries') && cfg.meta.devUseValEntries
        metaEntries = [metaEntries(:); valEntries(:)];
    end
    [svrBundle, gateBundle, metaSummary] = trainStage2MetaModelsOOF(metaEntries, cfg);
    if isstruct(svrBundle) && isfield(svrBundle, 'available') && logical(svrBundle.available)
        ruleMode = localResolveChannelRuleMode(cfg.meta);
        if strcmpi(ruleMode, "oof_holdout_dual_screen")
            holdoutResidualSummary = localEvaluateResidualHoldout(testEntries, net, normStats, cfg, svrBundle);
            svrBundle = localApplyResidualHoldoutRule(svrBundle, holdoutResidualSummary, cfg.meta);
        end
        metaSummary.holdoutResidualSummary = holdoutResidualSummary;
        metaSummary.channelRule = svrBundle.channelRule;
        metaSummary.enabledResidualChannels = logical(svrBundle.enabledMask(:));
        metaSummary.enabledResidualChannelCount = sum(metaSummary.enabledResidualChannels);
    end
end

trainSummary = cfg;
trainSummary.parameterCount = sum(cellfun(@numel, net.Learnables.Value));
trainSummary.activeProfile = activeProfile;
trainSummary.lossMode = char(cfg.loss.mode);
if metaTrainingEnabled
    trainSummary.stageMode = 'stage2_two_stage_meta';
else
    trainSummary.stageMode = 'stage2_dual_head';
end
trainSummary.enableSubst = true;
runtimeUsePredictedVariance = isfield(cfg, 'runtime') && isfield(cfg.runtime, 'usePredictedVariance') && ...
    logical(cfg.runtime.usePredictedVariance);
lossUsesNll = isfield(cfg.loss, 'nllWeight') && cfg.loss.nllWeight > 0;
trainSummary.usePredictedVariance = runtimeUsePredictedVariance || lossUsesNll;
if trainSummary.usePredictedVariance
    trainSummary.varianceMode = 'nll_predicted_aux_cov';
else
    trainSummary.varianceMode = 'fixed_calibrated_test_cov';
end
trainSummary.gammaD = single(gammaD(:));
trainSummary.gammaS = single(gammaS(:));
trainSummary.sourceInfo = sourceInfo;
trainSummary.duplicateInfo = duplicateInfo;
trainSummary.trainMeta = trainMeta;
trainSummary.valMeta = valMeta;
trainSummary.testMeta = testMeta;
trainSummary.trainMetrics = trainMetrics;
trainSummary.valMetrics = valMetrics;
trainSummary.testMetrics = testMetrics;
trainSummary.observationInfoLike = single(obsInfoLike(:));
trainSummary.observationRowNorm = single(obsRowNorm(:));
trainSummary.lossChannelWeights = single(cfg.loss.channelWeights(:));
trainSummary.generatedAt = string(datetime('now'));
trainSummary.sampleTime = cfg.synthetic.sampleTime;
trainSummary.baseRcDiag = single(cfg.filter.baseRcDiag(:));
trainSummary.substEnterProb = cfg.filter.substEnterProb;
trainSummary.substExitProb = cfg.filter.substExitProb;
trainSummary.substEnterCount = cfg.filter.substEnterCount;
trainSummary.substExitCount = cfg.filter.substExitCount;
trainSummary.qBoostHorizon = cfg.filter.qBoostHorizon;
trainSummary.recoveryEnabled = logical(cfg.filter.recoveryEnabled);
trainSummary.recoveryFallbackEnterCount = cfg.filter.recoveryFallbackEnterCount;
trainSummary.recoveryLambdaEnter = cfg.filter.recoveryLambdaEnter;
trainSummary.recoveryCtxRawOkMinChannels = cfg.filter.recoveryCtxRawOkMinChannels;
trainSummary.recoveryHoldOffAuxCount = cfg.filter.recoveryHoldOffAuxCount;
trainSummary.recoveryQBoostHorizon = cfg.filter.recoveryQBoostHorizon;
trainSummary.recoveryExitCleanCount = cfg.filter.recoveryExitCleanCount;
trainSummary.sequenceLength = cfg.synthetic.sequenceLength;
trainSummary.runtimeModelMode = char(cfg.runtime.modelMode);
trainSummary.gateThresholds = cfg.filter.gateThresholds;
trainSummary.metaFeatureConfig = localResolveMetaFeatureConfig(cfg.meta);
trainSummary.residualChannelRuleMode = localResolveChannelRuleMode(cfg.meta);
trainSummary.runtimeGateCalib = struct( ...
    'enabled', logical(cfg.runtime.useGateCalibration), ...
    'useOnlineRc', logical(cfg.runtime.useGateCalibration), ...
    'file', char(cfg.runtime.gateCalibrationFile));
trainSummary.initModelFile = string(cfg.train.initModelFile);
trainSummary.onlineInputLayout = "xPred4 + meas6 + zAux12 + zPred6 + innovation6";
trainSummary.onlineDenormSnippet = "decode_lstm_aux_output handles the saved 17/23-channel layout";
trainSummary.metaSummary = metaSummary;

save(char(cfg.train.outputModelFile), 'net', 'normStats', 'svrBundle', 'gateBundle', 'trainSummary');

clear XTrainRaw YTrainRaw XValRaw YValRaw XTestRaw YTestRaw
if isfield(cfg.train, 'skipBenchmark') && cfg.train.skipBenchmark
    fprintf('Skipping training-script benchmark; paired ablation evaluator runs separately.\n');
else
runtimeOverrideState = localPushRuntimeOverrides(char(cfg.train.outputModelFile), cfg.runtime);
try
    benchmark = evaluateFilterBenchmark(testEntries);
catch ME
    localRestoreRuntimeOverrides(runtimeOverrideState);
    rethrow(ME);
end
localRestoreRuntimeOverrides(runtimeOverrideState);
trainSummary.benchmark = benchmark;
trainSummary.benchmarkRuntimeMode = char(cfg.runtime.modelMode);
trainSummary.benchmarkUseGateCalibration = logical(cfg.runtime.useGateCalibration);
trainSummary.benchmarkSourceFiles = string({testEntries.name});
save(char(cfg.train.outputModelFile), 'net', 'normStats', 'svrBundle', 'gateBundle', 'trainSummary');

fprintf('\n===== Hold-out Flight Benchmark =====\n');
fprintf('Wind RMSE EKF      : [%.4f %.4f] ft/s\n', benchmark.rmseWindBase(1), benchmark.rmseWindBase(2));
fprintf('Wind RMSE Fusion   : [%.4f %.4f] ft/s\n', benchmark.rmseWindFuse(1), benchmark.rmseWindFuse(2));
fprintf('VT RMSE            : EKF %.4f | Fusion %.4f ft/s\n', benchmark.rmseAirBase(1), benchmark.rmseAirFuse(1));
fprintf('alpha RMSE         : EKF %.4f | Fusion %.4f deg\n', rad2deg(benchmark.rmseAirBase(2)), rad2deg(benchmark.rmseAirFuse(2)));
fprintf('beta RMSE          : EKF %.4f | Fusion %.4f deg\n', rad2deg(benchmark.rmseAirBase(3)), rad2deg(benchmark.rmseAirFuse(3)));
fprintf('Air RMSE improve%%  : [%.2f %.2f %.2f]\n', benchmark.airImprovePct(1), benchmark.airImprovePct(2), benchmark.airImprovePct(3));
fprintf('Abnormal p95 improve%% : [%.2f %.2f %.2f]\n', benchmark.airP95ImprovePct(1), benchmark.airP95ImprovePct(2), benchmark.airP95ImprovePct(3));
end
fprintf('Saved trained model to %s\n', char(cfg.train.outputModelFile));

function [loss, gradients] = modelGradientsStage2(net, X, Y, normStats, cfgLoss)
YPred = forward(net, X);
loss = computeStage2Loss(YPred, X, Y, normStats, cfgLoss);
gradients = dlgradient(loss, net.Learnables);
end

function loss = modelLossStage2(net, X, Y, normStats, cfgLoss)
YPred = forward(net, X);
loss = computeStage2Loss(YPred, X, Y, normStats, cfgLoss);
end

function lossMean = computeDatasetLossBatched(net, XRaw, YRaw, normStats, cfgLoss, batchSize)
numSeq = size(XRaw, 3);
batchSize = max(1, min(batchSize, numSeq));
lossAccum = 0;
weightAccum = 0;

for startIdx = 1:batchSize:numSeq
    batchIdx = startIdx:min(startIdx + batchSize - 1, numSeq);
    XBatch = dlarray(normalizeInputBatch(XRaw(:, :, batchIdx), normStats), 'CTB');
    YBatch = dlarray(single(YRaw(:, :, batchIdx)), 'CTB');
    batchLoss = dlfeval(@modelLossStage2, net, XBatch, YBatch, normStats, cfgLoss);
    batchWeight = numel(batchIdx);
    lossAccum = lossAccum + double(gather(extractdata(batchLoss))) * batchWeight;
    weightAccum = weightAccum + batchWeight;
end

lossMean = lossAccum / max(weightAccum, 1);
end

function loss = computeStage2Loss(YPred, XNorm, Y, normStats, cfgLoss)
if size(YPred, 1) == 17
    deltaDNorm = YPred(1:6, :, :);
    deltaSNorm = deltaDNorm;
    offset = -6;
else
    deltaDNorm = YPred(1:6, :, :);
    deltaSNorm = YPred(7:12, :, :);
    offset = 0;
end
logVarNorm = max(min(YPred(13+offset:18+offset, :, :), 5), -6);
pAbn = sigmoid(YPred(19+offset, :, :));
pSubst = sigmoid(YPred(20+offset, :, :));
airNorm = YPred(21+offset:23+offset, :, :);

measRaw = recoverMeasurementRaw(XNorm, normStats);
yCleanRaw = Y(1:6, :, :);
ySubTargetRaw = Y(7:12, :, :);
deltaTrueDRaw = yCleanRaw - measRaw;
deltaTrueSRaw = ySubTargetRaw - measRaw;
deltaTrueDNorm = normalizeDeltaTarget(deltaTrueDRaw, normStats);
deltaTrueSNorm = normalizeDeltaTarget(deltaTrueSRaw, normStats);
airTrueNorm = normalizeAirTarget(Y(13:15, :, :), normStats);

abn = Y(19, :, :);
subst = Y(20, :, :);
scenarioCode = Y(21, :, :);
modelMismatchWeight = zeros(size(subst), 'like', subst);
if size(Y, 1) >= 22
    modelMismatchWeight = Y(22, :, :);
end
lambdaSeq = Y(16, :, :);
gateThresholds = localGetGateThresholds(cfgLoss);
rejectMask = lambdaSeq > gateThresholds.T1;

maskD = ones(size(subst), 'like', subst);
if isfield(cfgLoss, 'maneuverAeroDenoiseBoost') && cfgLoss.maneuverAeroDenoiseBoost > 1
    maskD = maskD .* (1 + (cfgLoss.maneuverAeroDenoiseBoost - 1) * modelMismatchWeight);
end
maskS = single(subst > 0);
maskAir = 1 + 0.25 * (scenarioCode == 2);
if isfield(cfgLoss, 'maneuverAeroAirBoost') && cfgLoss.maneuverAeroAirBoost > 0
    maskAir = maskAir .* (1 + cfgLoss.maneuverAeroAirBoost * modelMismatchWeight);
end
if isfield(cfgLoss, 'rejectBoostFactor') && cfgLoss.rejectBoostFactor > 1
    maskD = maskD .* (1 + (cfgLoss.rejectBoostFactor - 1) * rejectMask);
end
if isfield(cfgLoss, 'substRejectBoost') && cfgLoss.substRejectBoost > 1
    maskS = maskS .* (1 + (cfgLoss.substRejectBoost - 1) * rejectMask .* subst);
end

lossD = weightedHuberLossMasked(deltaDNorm - deltaTrueDNorm, cfgLoss.huberDelta, cfgLoss.channelWeights, maskD);
lossS = weightedHuberLossMasked(deltaSNorm - deltaTrueSNorm, cfgLoss.huberDelta, cfgLoss.channelWeights, maskS);
lossAir = maskedHuberLoss3(airNorm - airTrueNorm, cfgLoss.huberDelta, maskAir);

lossNLL = 0;
if isfield(cfgLoss, 'nllWeight') && cfgLoss.nllWeight > 0
    chosenDelta = deltaDNorm .* (1 - subst) + deltaSNorm .* subst;
    deltaTrueChosenNorm = deltaTrueDNorm .* (1 - subst) + deltaTrueSNorm .* subst;
    lossNLL = cfgLoss.nllWeight * weightedGaussianNLL( ...
        chosenDelta - deltaTrueChosenNorm, logVarNorm, cfgLoss.channelWeights, 0.5 + 0.5 * abn);
end
lossCls = binaryCrossEntropyProb(pSubst, subst) + 0.25 * binaryCrossEntropyProb(pAbn, abn);
cleanMask = 1 - abn;
zeroDeltaNorm = normalizeDeltaTarget(0 * measRaw, normStats);
lossCleanResidual = weightedHuberLossMasked(deltaDNorm - zeroDeltaNorm, cfgLoss.huberDelta, cfgLoss.channelWeights, cleanMask);
lossCleanZero = 0;
if isfield(cfgLoss, 'cleanZeroResidualWeight') && cfgLoss.cleanZeroResidualWeight > 0
    lossCleanZero = cfgLoss.cleanZeroResidualWeight * ...
        weightedHuberLossMasked(deltaSNorm - zeroDeltaNorm, cfgLoss.huberDelta, cfgLoss.channelWeights, cleanMask);
end

loss = lossD + cfgLoss.substWeight * lossS + ...
    cfgLoss.airWeight * lossAir + cfgLoss.clsWeight * lossCls + ...
    lossNLL + cfgLoss.cleanResidualWeight * lossCleanResidual + lossCleanZero;
end

function measRaw = recoverMeasurementRaw(XNorm, normStats)
muMeas = reshape(normStats.muIn(5:10), [6, 1, 1]);
sigMeas = reshape(normStats.sigIn(5:10), [6, 1, 1]);
measRaw = XNorm(5:10, :, :) .* sigMeas + muMeas;
end

function deltaNorm = normalizeDeltaTarget(deltaRaw, normStats)
muDelta = reshape(normStats.muDelta, [6, 1, 1]);
sigDelta = reshape(normStats.sigDelta, [6, 1, 1]);
deltaNorm = (deltaRaw - muDelta) ./ sigDelta;
end

function airNorm = normalizeAirTarget(airRaw, normStats)
muAir = reshape(normStats.muAir, [3, 1, 1]);
sigAir = reshape(normStats.sigAir, [3, 1, 1]);
airNorm = (airRaw - muAir) ./ sigAir;
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

function loss = weightedHuberLossMasked(residual, delta, channelWeights, sampleMask)
weights = reshape(single(channelWeights), [numel(channelWeights), 1, 1]);
mask = ones(6, 1, 1, 'single') .* sampleMask;
lossMap = huberElementwise(residual, delta) .* weights .* mask;
denom = max(sum(weights .* mask, 'all'), 1e-9);
loss = sum(lossMap, 'all') / denom;
end

function loss = maskedHuberLoss3(residual, delta, mask)
lossMap = huberElementwise(residual, delta) .* mask;
denom = max(sum(mask, 'all'), 1e-9);
loss = sum(lossMap, 'all') / denom;
end

function loss = weightedGaussianNLL(residual, logVarNorm, channelWeights, sampleMask)
weights = reshape(single(channelWeights), [numel(channelWeights), 1, 1]);
mask = ones(6, 1, 1, 'single') .* sampleMask;
lossMap = 0.5 * (logVarNorm + (residual .^ 2) ./ exp(logVarNorm));
lossMap = lossMap .* weights .* mask;
denom = max(sum(weights .* mask, 'all'), 1e-9);
loss = sum(lossMap, 'all') / denom;
end

function out = huberElementwise(residual, delta)
absRes = abs(residual);
quadMask = absRes <= delta;
out = 0.5 * quadMask .* (residual .^ 2) + ...
    (~quadMask) .* (delta * (absRes - 0.5 * delta));
end

function loss = binaryCrossEntropyProb(prob, label)
prob = min(max(prob, 1e-6), 1 - 1e-6);
loss = mean(-(label .* log(prob) + (1 - label) .* log(1 - prob)), 'all');
end

function metrics = evaluateDatasetMetrics(net, XRaw, YRaw, normStats, evalCfg)
YPred = predictDatasetBatched(net, XRaw, normStats, evalCfg.batchSize);

yClean = YRaw(1:6, :, :);
yMeas = XRaw(5:10, :, :);
[muD, muS, ~, pAbn, pSubst, ~, airAux] = decode_lstm_aux_output(YPred, normStats, 'real', yMeas);
useSubst = single(pSubst >= evalCfg.substThreshold);
yChosen = muD .* (1 - useSubst) + muS .* useSubst;

airTrue = YRaw(13:15, :, :);
scenarioCode = round(YRaw(21, :, :));
maskClean = scenarioCode == 0;
maskMild = scenarioCode == 1;
maskSevere = scenarioCode == 2;
maskAbn = scenarioCode >= 1;
maskManeuverAero = false(size(maskClean));
if size(YRaw, 1) >= 22
    maskManeuverAero = YRaw(22, :, :) > 0;
end
maskAll = true(size(maskClean));

metrics.all = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskAll);
metrics.clean = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskClean);
metrics.mild = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskMild);
metrics.severe = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskSevere);
metrics.abnormal = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskAbn);
metrics.maneuverAero = computeMetricBundle(yChosen, muD, muS, yMeas, yClean, airAux, airTrue, maskManeuverAero);
metrics.maneuverAero.pAbnMean = maskedScalarMean(pAbn, maskManeuverAero);
metrics.maneuverAero.pAbnMax = maskedScalarMax(pAbn, maskManeuverAero);
metrics.maneuverAero.pSubstMean = maskedScalarMean(pSubst, maskManeuverAero);
metrics.maneuverAero.pSubstMax = maskedScalarMax(pSubst, maskManeuverAero);
metrics.abnF1 = computeF1(pAbn >= 0.5, YRaw(19, :, :) >= 0.5);
metrics.substF1 = computeF1(pSubst >= evalCfg.substThreshold, YRaw(20, :, :) >= 0.5);
metrics.gammaCandidateD = deriveCalibratedRScale(metrics.mild.rmseDenoise, metrics.mild.rmseMeas, 0.08, 1.0);
metrics.gammaCandidateS = deriveCalibratedRScale(metrics.severe.rmseSubst, metrics.severe.rmseMeas, 0.05, 1.0);
metrics.passClean = all(metrics.clean.rmseChosen <= 1.05 * metrics.clean.rmseMeas);
metrics.passMild = sum(metrics.mild.rmseChosen < metrics.mild.rmseMeas) >= 4;
metrics.passSevere = sum(metrics.severe.rmseSubst < metrics.severe.rmseMeas) >= 4;
metrics.passLag = all(abs(metrics.all.bestLagChosen) <= 1);
metrics.passStage2 = metrics.passClean && metrics.passMild && metrics.passSevere && metrics.passLag;
end

function YPredAll = predictDatasetBatched(net, XRaw, normStats, batchSize)
numSeq = size(XRaw, 3);
batchSize = max(1, min(batchSize, numSeq));
YPredAll = zeros(size(net.Layers(end).Weights, 1), size(XRaw, 2), numSeq, 'single');

for startIdx = 1:batchSize:numSeq
    batchIdx = startIdx:min(startIdx + batchSize - 1, numSeq);
    YPredBatch = predictNormalizedBatch(net, XRaw(:, :, batchIdx), normStats);
    YPredAll(:, :, batchIdx) = YPredBatch;
end
end

function YPred = predictNormalizedBatch(net, XRawBatch, normStats)
XBatch = dlarray(normalizeInputBatch(XRawBatch, normStats), 'CTB');
YPred = gather(extractdata(forward(net, XBatch)));
if ndims(YPred) == 3 && size(YPred, 2) == size(XRawBatch, 3) && size(YPred, 3) == size(XRawBatch, 2)
    YPred = permute(YPred, [1, 3, 2]);
end
YPred = single(YPred);
end

function XNorm = normalizeInputBatch(XRaw, normStats)
muIn = reshape(normStats.muIn, [], 1, 1);
sigIn = reshape(normStats.sigIn, [], 1, 1);
XNorm = (single(XRaw) - muIn) ./ sigIn;
end

function bundle = computeMetricBundle(yChosen, muD, muS, yMeas, yTrue, airAux, airTrue, mask)
bundle.count = sum(mask(:));
bundle.rmseChosen = maskedChannelError(yChosen, yTrue, mask, 'rmse');
bundle.rmseDenoise = maskedChannelError(muD, yTrue, mask, 'rmse');
bundle.rmseSubst = maskedChannelError(muS, yTrue, mask, 'rmse');
bundle.rmseMeas = maskedChannelError(yMeas, yTrue, mask, 'rmse');
bundle.maeChosen = maskedChannelError(yChosen, yTrue, mask, 'mae');
bundle.bestLagChosen = estimateBestLagByChannelMasked(yChosen, yTrue, mask, 12);
bundle.improvementPct = 1 - bundle.rmseChosen ./ max(bundle.rmseMeas, 1e-9);
bundle.airRmse = maskedChannelError(airAux, airTrue, mask, 'rmse');
end

function val = maskedScalarMean(x, mask)
x = double(x);
mask = logical(mask);
if ~any(mask(:))
    val = NaN;
    return;
end
val = mean(x(mask), 'all');
end

function val = maskedScalarMax(x, mask)
x = double(x);
mask = logical(mask);
if ~any(mask(:))
    val = NaN;
    return;
end
val = max(x(mask), [], 'all');
end

function val = maskedChannelError(a3, b3, mask3, mode)
a = reshape(a3, size(a3, 1), []);
b = reshape(b3, size(b3, 1), []);
mask = logical(mask3(:)');
val = nan(size(a, 1), 1);

for ch = 1:size(a, 1)
    aa = a(ch, mask);
    bb = b(ch, mask);
    if isempty(aa)
        continue;
    end
    switch lower(mode)
        case 'rmse'
            val(ch) = sqrt(mean((aa - bb) .^ 2));
        case 'mae'
            val(ch) = mean(abs(aa - bb));
        otherwise
            error('Unsupported mode: %s', mode);
    end
end
end

function f1 = computeF1(predMask, trueMask)
pred = logical(predMask(:));
truth = logical(trueMask(:));
tp = sum(pred & truth);
fp = sum(pred & ~truth);
fn = sum(~pred & truth);
precision = tp / max(tp + fp, 1);
recall = tp / max(tp + fn, 1);
f1 = 2 * precision * recall / max(precision + recall, 1e-9);
end

function gamma = deriveCalibratedRScale(rmseAux, rmseMeas, lo, hi)
ratio2 = (rmseAux(:) ./ max(rmseMeas(:), 1e-6)) .^ 2;
ratio2(~isfinite(ratio2)) = 1;
gamma = min(max(ratio2, lo), hi);
end

function [entriesTrain, entriesVal, entriesTest, duplicateInfo, sourceInfo] = buildDataSource(cfg)
duplicateInfo = struct([]);

switch cfg.data.source
    case "synthetic_full_envelope"
        dataset = generate_full_envelope_synthetic_logs(cfg.synthetic);
        entriesTrain = dataset.train;
        entriesVal = dataset.val;
        entriesTest = dataset.test;
        sourceInfo = struct('type', "synthetic_full_envelope", 'config', cfg.synthetic);

    case "mixed_finetune"
        dataset = generate_full_envelope_synthetic_logs(cfg.synthetic);
        [loggedTrain, loggedVal, loggedTest, duplicateInfo, loggedInfo] = localLoadLoggedEntries(cfg);
        entriesTrain = [dataset.train; loggedTrain];
        entriesVal = [dataset.val; loggedVal];
        entriesTest = [dataset.test; loggedTest];
        sourceInfo = struct('type', "mixed_finetune", 'syntheticConfig', cfg.synthetic, 'loggedInfo', loggedInfo);

    otherwise
        [entriesTrain, entriesVal, entriesTest, duplicateInfo, loggedInfo] = localLoadLoggedEntries(cfg);
        sourceInfo = struct('type', "logged", 'files', loggedInfo.files);
end
end

function [entriesTrain, entriesVal, entriesTest, duplicateInfo, info] = localLoadLoggedEntries(cfg)
pattern = char(cfg.data.loggedPattern);
candidateFiles = dir(pattern);
if isempty(candidateFiles)
    if isfile('real_training_logs.mat')
        candidateFiles = dir('real_training_logs.mat');
    elseif string(cfg.data.source) == "mixed_finetune"
        fprintf('No logged flights found for %s | mixed_finetune will run synthetic-only.\n', pattern);
        entriesTrain = repmat(struct('name', "", 'folder', "", 'datenum', 0, 'logs', struct()), 0, 1);
        entriesVal = entriesTrain;
        entriesTest = entriesTrain;
        duplicateInfo = struct([]);
        info = struct('files', strings(0, 1), 'pattern', string(pattern), ...
            'rejectedFiles', strings(0, 1), 'rejectedReasons', strings(0, 1));
        return;
    else
        error('Missing training logs for pattern: %s', pattern);
    end
end

if numel(candidateFiles) < 3
    if string(cfg.data.source) == "mixed_finetune"
        fprintf('Logged flights found: %d | mixed_finetune will use available aligned logs as train-only augmentation.\n', numel(candidateFiles));
    else
        error('Need at least 3 logged flights, got %d.', numel(candidateFiles));
    end
end

[entries, duplicateInfo] = collectUniqueLogFiles(candidateFiles);
[entries, rejectedInfo] = localFilterAlignedLoggedEntries(entries, 1);
if ~isempty(rejectedInfo)
    fprintf('Filtered %d logged flights due to label alignment.\n', numel(rejectedInfo));
end

if numel(entries) >= 3
    entriesTrain = entries(1:end-2);
    entriesVal = entries(end-1);
    entriesTest = entries(end);
elseif string(cfg.data.source) == "mixed_finetune"
    fprintf('Aligned logged flights available for augmentation: %d\n', numel(entries));
    entriesTrain = entries;
    entriesVal = entries([]);
    entriesTest = entries([]);
else
    error('Need at least 3 aligned logged flights for train/val/test split.');    
end
info = struct( ...
    'files', string({entries.name}), ...
    'pattern', string(pattern), ...
    'rejectedFiles', string({rejectedInfo.file}), ...
    'rejectedReasons', string({rejectedInfo.reason}));
end

function [infoLike, rowNorm] = estimateObservationChannelInfo(entries, rcDiag)
accum = zeros(6, 1);
sampleCount = 0;

for i = 1:numel(entries)
    logs = entries(i).logs;
    xPredMat = logs.teacher_xPred;
    zPredMat = logs.teacher_zPred;
    N = size(logs.state, 2);
    sampleIdx = unique(round(linspace(1, N, min(N, 240))));

    for k = sampleIdx
        H2 = compute_jacobian_numerical(xPredMat(1:2, k), logs.state(:, k), logs.controls(:, k), zPredMat(:, k));
        accum = accum + sum(H2 .^ 2, 2);
        sampleCount = sampleCount + 1;
    end
end

rowNorm = accum / max(sampleCount, 1);
infoLike = rowNorm ./ max(double(rcDiag(:)), 1e-9);
end

function channelWeights = computeObservationChannelWeights(infoLike, cfgLoss)
infoSafe = max(double(infoLike(:)), 1e-9);
raw = (infoSafe / mean(infoSafe)) .^ cfgLoss.obsWeightPower;
raw = min(max(raw, cfgLoss.obsWeightClip(1)), cfgLoss.obsWeightClip(2));
channelWeights = single(raw / mean(raw));
end

function lags = estimateBestLagByChannelMasked(a3, b3, mask3, maxLag)
a = reshape(a3, size(a3, 1), []);
b = reshape(b3, size(b3, 1), []);
mask = logical(mask3(:)');
lags = zeros(size(a, 1), 1);

for ch = 1:size(a, 1)
    aa = a(ch, mask);
    bb = b(ch, mask);
    if numel(aa) < 8
        continue;
    end
    bestRmse = inf;
    for lag = -maxLag:maxLag
        [a1, b1] = alignByLag1D(aa, bb, lag);
        if isempty(a1)
            continue;
        end
        rmse = sqrt(mean((a1 - b1) .^ 2));
        if rmse < bestRmse
            bestRmse = rmse;
            lags(ch) = lag;
        end
    end
end
end

function [a1, b1] = alignByLag1D(a, b, lag)
N = min(numel(a), numel(b));
a = a(1:N);
b = b(1:N);

if lag > 0
    a1 = a(1:(N - lag));
    b1 = b((1 + lag):N);
elseif lag < 0
    L = -lag;
    a1 = a((1 + L):N);
    b1 = b(1:(N - L));
else
    a1 = a;
    b1 = b;
end
end

function printMetrics(tag, metrics)
printBundle([tag ' All'], metrics.all);
printBundle([tag ' Clean'], metrics.clean);
printBundle([tag ' Mild'], metrics.mild);
printBundle([tag ' Severe'], metrics.severe);
if isfield(metrics, 'maneuverAero') && metrics.maneuverAero.count > 0
    printBundle([tag ' ManeuverAero'], metrics.maneuverAero);
    fprintf('%s ManeuverAero pAbn mean/max: %.4f / %.4f | pSubst mean/max: %.4f / %.4f\n', ...
        tag, metrics.maneuverAero.pAbnMean, metrics.maneuverAero.pAbnMax, ...
        metrics.maneuverAero.pSubstMean, metrics.maneuverAero.pSubstMax);
end
fprintf('%s abnormal F1: %.4f | substitution F1: %.4f\n', tag, metrics.abnF1, metrics.substF1);
fprintf('%s stage2 pass [clean mild severe lag overall]: %d %d %d %d %d\n', ...
    tag, metrics.passClean, metrics.passMild, metrics.passSevere, metrics.passLag, metrics.passStage2);
end

function printBundle(tag, bundle)
fprintf('%s RMSE (chosen aux vs Yclean): ', tag);
disp(bundle.rmseChosen(:)');
fprintf('%s RMSE (mu_denoise vs Yclean): ', tag);
disp(bundle.rmseDenoise(:)');
fprintf('%s RMSE (meas vs Yclean): ', tag);
disp(bundle.rmseMeas(:)');
fprintf('%s RMSE (mu_subst vs Yclean): ', tag);
disp(bundle.rmseSubst(:)');
fprintf('%s Air RMSE [VT alpha beta]: ', tag);
disp(bundle.airRmse(:)');
fprintf('%s RMSE improvement pct: ', tag);
disp((100 * bundle.improvementPct(:))');
end

function benchmark = evaluateFilterBenchmark(entries)
allWindErrBase = [];
allWindErrFuse = [];
allAirErrBase = [];
allAirErrFuse = [];
allAirErrBaseAbn = [];
allAirErrFuseAbn = [];
modeHist = [];

for i = 1:numel(entries)
    entryResult = run_stage2_entry_benchmark(entries(i));
    vwTrue = entryResult.online.windTrue;
    airTrue = entryResult.online.airTrue;
    abnMask = any(entryResult.replayFault.anomaly_mask ~= 0, 1);

    allWindErrBase = [allWindErrBase, entryResult.replay.windEkf - vwTrue]; %#ok<AGROW>
    allWindErrFuse = [allWindErrFuse, entryResult.replay.windFuse - vwTrue]; %#ok<AGROW>
    allAirErrBase = [allAirErrBase, entryResult.replay.airEkf - airTrue]; %#ok<AGROW>
    allAirErrFuse = [allAirErrFuse, entryResult.replay.airFuse - airTrue]; %#ok<AGROW>
    allAirErrBaseAbn = [allAirErrBaseAbn, entryResult.replay.airEkf(:, abnMask) - airTrue(:, abnMask)]; %#ok<AGROW>
    allAirErrFuseAbn = [allAirErrFuseAbn, entryResult.replay.airFuse(:, abnMask) - airTrue(:, abnMask)]; %#ok<AGROW>
    modeHist = [modeHist, entryResult.replay.mode]; %#ok<AGROW>
end

benchmark.rmseWindBase = sqrt(mean(allWindErrBase .^ 2, 2));
benchmark.rmseWindFuse = sqrt(mean(allWindErrFuse .^ 2, 2));
benchmark.rmseAirBase = sqrt(mean(allAirErrBase .^ 2, 2));
benchmark.rmseAirFuse = sqrt(mean(allAirErrFuse .^ 2, 2));
benchmark.airImprovePct = 100 * (1 - benchmark.rmseAirFuse ./ max(benchmark.rmseAirBase, 1e-9));

benchmark.p95AirBaseAbn = localP95(allAirErrBaseAbn);
benchmark.p95AirFuseAbn = localP95(allAirErrFuseAbn);
benchmark.airP95ImprovePct = 100 * (1 - benchmark.p95AirFuseAbn ./ max(benchmark.p95AirBaseAbn, 1e-9));
benchmark.modeShare = histcounts(modeHist, -0.5:1:3.5) / max(numel(modeHist), 1);
benchmark.passAverage = mean(benchmark.airImprovePct) > 0 && mean(benchmark.airP95ImprovePct) > 0;
end

function p95 = localP95(errMat)
if isempty(errMat)
    p95 = [NaN; NaN; NaN];
else
    p95 = prctile(abs(errMat').', 95, 2);
end
end

function [entries, duplicateInfo] = collectUniqueLogFiles(logFiles)
[~, order] = sort([logFiles.datenum], 'ascend');
logFiles = logFiles(order);

entries = repmat(struct('name', "", 'folder', "", 'datenum', 0, 'logs', struct()), numel(logFiles), 1);
duplicateInfo = repmat(struct('file', "", 'duplicateOf', ""), numel(logFiles), 1);
entryCount = 0;
dupCount = 0;

for i = 1:numel(logFiles)
    file = logFiles(i);
    S = load(fullfile(file.folder, file.name));
    if ~isfield(S, 'logs')
        error('File %s does not contain variable: logs', file.name);
    end

    dupIdx = 0;
    for j = 1:entryCount
        if isDuplicateLogs(S.logs, entries(j).logs)
            dupIdx = j;
            break;
        end
    end

    if dupIdx > 0
        dupCount = dupCount + 1;
        duplicateInfo(dupCount).file = string(file.name);
        duplicateInfo(dupCount).duplicateOf = string(entries(dupIdx).name);
        continue;
    end

    entryCount = entryCount + 1;
    entries(entryCount).name = string(file.name);
    entries(entryCount).folder = string(file.folder);
    entries(entryCount).datenum = file.datenum;
    entries(entryCount).logs = S.logs;
end

entries = entries(1:entryCount);
duplicateInfo = duplicateInfo(1:dupCount);
end

function [entriesOut, rejectedInfo] = localFilterAlignedLoggedEntries(entriesIn, residualTol)
if nargin < 2 || isempty(residualTol)
    residualTol = 1;
end

keepMask = false(numel(entriesIn), 1);
rejectedInfo = repmat(struct('file', "", 'reason', "", 'maxResidualLag', NaN), numel(entriesIn), 1);
rejectCount = 0;

for i = 1:numel(entriesIn)
    [keepMask(i), reason, maxResidualLag] = localCheckLabelAlignment(entriesIn(i).logs, residualTol);
    if ~keepMask(i)
        rejectCount = rejectCount + 1;
        rejectedInfo(rejectCount).file = string(entriesIn(i).name);
        rejectedInfo(rejectCount).reason = string(reason);
        rejectedInfo(rejectCount).maxResidualLag = maxResidualLag;
        fprintf('Skipping %s | %s\n', entriesIn(i).name, reason);
    end
end

entriesOut = entriesIn(keepMask);
rejectedInfo = rejectedInfo(1:rejectCount);
end

function [tf, reason, maxResidualLag] = localCheckLabelAlignment(logs, residualTol)
tf = false;
reason = "missing label_alignment_ok";
maxResidualLag = inf;

hasResidualPerChannel = isfield(logs, 'label_lag_residual_per_channel') && ...
    numel(logs.label_lag_residual_per_channel) >= 1;
if hasResidualPerChannel
    residualLag = double(logs.label_lag_residual_per_channel(:));
    maxResidualLag = max(abs(residualLag));
elseif isfield(logs, 'label_lag_samples_residual')
    maxResidualLag = abs(double(logs.label_lag_samples_residual));
end

if ~isfield(logs, 'label_alignment_ok')
    if isfinite(maxResidualLag)
        reason = sprintf('missing label_alignment_ok (max residual lag %.0f)', maxResidualLag);
    end
    return;
end

if ~logical(logs.label_alignment_ok)
    if isfinite(maxResidualLag)
        reason = sprintf('label_alignment_ok=false (max residual lag %.0f)', maxResidualLag);
    else
        reason = 'label_alignment_ok=false';
    end
    return;
end

if ~hasResidualPerChannel
    if isfinite(maxResidualLag)
        reason = sprintf('missing residual-per-channel lag (scalar residual %.0f)', maxResidualLag);
    else
        reason = 'missing residual-per-channel lag';
    end
    return;
end

if ~(isfinite(maxResidualLag) && maxResidualLag <= residualTol)
    reason = sprintf('max residual lag %.0f > %d', maxResidualLag, residualTol);
    return;
end

tf = true;
reason = "";
end

function tf = isDuplicateLogs(a, b)
required = {'state', 'controls', 'measurements', 'Yclean'};
tf = true;
for i = 1:numel(required)
    field = required{i};
    if ~isfield(a, field) || ~isfield(b, field) || ~isequal(a.(field), b.(field))
        tf = false;
        return;
    end
end
end

function [XAll, YAll, metaList, seqInfo, frameInfo] = collectDataset(entries, dataCfg)
XAll = [];
YAll = [];
metaList = repmat(struct('file', "", 'numSequences', 0, 'numSamples', 0, ...
    'scenarioCounts', zeros(1, 3), 'abnormalFrames', 0, 'substitutionFrames', 0), numel(entries), 1);
seqInfo = repmat(struct('source', "", 'hasSubst', false, 'hasSevere', false, 'hasFailureWindow', false, 'dominantScenario', 0), 0, 1);
frameInfo = struct('anomalyMask', [], 'abnormalLabel', [], 'substLabel', [], 'scenarioCode', []);

for i = 1:numel(entries)
    [Xi, Yi, meta, frameInfoI] = build_training_data_from_logs(entries(i).logs, dataCfg);
    XAll = cat(3, XAll, Xi);
    YAll = cat(3, YAll, Yi);
    seqInfo = [seqInfo; meta.sequenceTags(:)]; %#ok<AGROW>
    if isempty(frameInfo.anomalyMask)
        frameInfo = frameInfoI;
    else
        frameInfo.anomalyMask = cat(3, frameInfo.anomalyMask, frameInfoI.anomalyMask);
        frameInfo.abnormalLabel = cat(3, frameInfo.abnormalLabel, frameInfoI.abnormalLabel);
        frameInfo.substLabel = cat(3, frameInfo.substLabel, frameInfoI.substLabel);
        frameInfo.scenarioCode = cat(3, frameInfo.scenarioCode, frameInfoI.scenarioCode);
    end

    metaList(i).file = string(entries(i).name);
    metaList(i).numSequences = meta.numSequences;
    metaList(i).numSamples = meta.numSamples;
    metaList(i).scenarioCounts = meta.scenarioCounts;
    metaList(i).abnormalFrames = meta.abnormalFrames;
    metaList(i).substitutionFrames = meta.substitutionFrames;

    fprintf('Loaded %s | sequences=%d | clean=%d mild=%d severe=%d\n', ...
        entries(i).name, meta.numSequences, meta.scenarioCounts(1), meta.scenarioCounts(2), meta.scenarioCounts(3));
end
end

function [XOut, YOut, seqInfoOut] = rebalanceMixedDataset(XIn, YIn, seqInfoIn, mixCfg)
if isempty(seqInfoIn)
    XOut = XIn;
    YOut = YIn;
    seqInfoOut = seqInfoIn;
    return;
end

source = string({seqInfoIn.source});
idxSynthetic = find(source == "synthetic");
idxLogged = find(source == "logged");

if isempty(idxSynthetic) || isempty(idxLogged)
    XOut = XIn;
    YOut = YIn;
    seqInfoOut = seqInfoIn;
    return;
end

failureIdx = idxLogged([seqInfoIn(idxLogged).hasFailureWindow]);
loggedPool = idxLogged;
oversampleFactor = max(1, round(mixCfg.loggedOversampleFactor));
for i = 2:oversampleFactor
    loggedPool = [loggedPool, failureIdx]; %#ok<AGROW>
end

nEach = min(numel(idxSynthetic), numel(loggedPool));
idxSynthetic = idxSynthetic(randperm(numel(idxSynthetic), nEach));
loggedPool = loggedPool(randperm(numel(loggedPool), nEach));
selected = [idxSynthetic, loggedPool];
selected = selected(randperm(numel(selected)));

XOut = XIn(:, :, selected);
YOut = YIn(:, :, selected);
seqInfoOut = seqInfoIn(selected);
end

function cfg = localDefaultSyntheticCfg()
cfg.seed = 11;
cfg.saveToDisk = false;
cfg.sampleTime = 0.05;
cfg.flightDuration = 180;
cfg.numTrain = 10;
cfg.numVal = 3;
cfg.numTest = 3;
cfg.sequenceLength = 120;
cfg.stride = 12;
cfg.segmentDurationRange = [8, 22];
cfg.altitudeRange = [5000, 40000];
cfg.vtRange = [320, 850];
cfg.alphaRange = [-5, 24];
cfg.betaRange = [-10, 10];
cfg.phiRange = [-75, 75];
cfg.thetaRange = [-25, 20];
cfg.controlRanges.thrust = [1500, 19000];
cfg.controlRanges.de = [-25, 25];
cfg.controlRanges.da = [-21.5, 21.5];
cfg.controlRanges.dr = [-30, 30];
cfg.windMean = [0; 0];
cfg.windSigma = [1.8; 1.8];
cfg.windTimeConstant = 6.0;
cfg.windShear = localDefaultWindShearCfg();
cfg.windTurbulence = localDefaultWindTurbulenceCfg();
cfg.windGust = localDefaultWindGustCfg();
cfg.gustEventRange = [4, 7];
cfg.gustDurationRange = [20, 80];
cfg.gustAmplitude = [20, 20];
cfg.sensorNoiseStd = [0.22; 0.22; 0.22; 0.028; 0.028; 0.028];
cfg.sensorBiasStd = [0.004; 0.004; 0.004; 0.0012; 0.0012; 0.0012];
cfg.sensorBiasRho = 0.995;
cfg.windPriorStd = [3.0; 3.0];
cfg.anomaly.mild = localStage2AnomalyCfg([1, 1], [0.9, 1.4, 1.8], [6, 28], [1, 2, 5, 6], false);
cfg.anomaly.severe = localStage2AnomalyCfg([1, 2], [1.2, 2.0, 3.0], [10, 60], [1, 2, 3, 4, 5, 6], false);
cfg.aeroBias = struct('enabled', false, 'minPct', 0.05, 'maxPct', 0.10, ...
    'augmentCopies', 0, 'seed', 20260507, 'mode', "per_entry_signed");
cfg.maneuverAeroBias = struct('enabled', false, 'augmentCopies', 0, ...
    'minPct', 0.10, 'maxPct', 0.20, 'mode', "signed", 'seed', 20260515, ...
    'quantilePct', 90, 'rampSec', 0.35, 'mergeGapSec', 0.15, 'minDurationSec', 0.08);
cfg.evalAnomaly.numEventsRange = [5, 9];
cfg.evalAnomaly.severityAmp = [1.0, 1.8, 2.8];
cfg.evalAnomaly.segmentRange = [12, 90];
cfg.evalAnomaly.substTypes = [3, 4, 6];
cfg.evalAnomaly.channelScale = [2.5, 5.0, 5.0, 0.45, 0.25, 0.12];
end

function cfg = localDefaultWindShearCfg()
cfg.enabled = true;
cfg.referenceAltitude = 2708.1006;
cfg.lowerAltitude = 15;
cfg.saturatedWindMps = [-30 * 0.5144; 20 * 0.5144];
cfg.mpsToFtps = 3.28084;
end

function cfg = localDefaultWindTurbulenceCfg()
sigu = 0.15 * sqrt(900 * 0.5144 ^ 2 + 400 * 0.5144 ^ 2);
cfg.enabled = true;
cfg.randomSampleTime = 0.05;
cfg.randomVariance = [0.2; 0.2];
cfg.tau = [2.5; 2.5];
cfg.numeratorGain = [10; 10];
cfg.sigmaMps = [sigu; sigu];
cfg.scale = 1.0;
cfg.mpsToFtps = 3.28084;
end

function cfg = localDefaultWindGustCfg()
cfg.enabled = true;
cfg.startTimeRangeSec = [0, 60];
cfg.startTimeSec = nan(4, 1);
cfg.axis = [1; 2; 1; 2];
cfg.lengthM = [120; 150; 150; 150];
cfg.amplitudeMps = [10; -12; -8; 8];
cfg.mpsToFtps = 3.28084;
end

function cfg = localStage2AnomalyCfg(numEventsRange, severityAmp, segmentRange, typePool, forceSubst)
cfg.numEventsRange = numEventsRange;
cfg.severityAmp = severityAmp;
cfg.segmentRange = segmentRange;
cfg.typePool = typePool;
cfg.substTypes = [3, 4, 6];
cfg.forceSubst = forceSubst;
end

function [svrBundle, gateBundle, metaSummary] = trainStage2MetaModelsOOF(entries, cfg)
svrBundle = struct('available', false);
gateBundle = struct('available', false);
metaSummary = struct('available', false, 'numFolds', 0, 'numRejectSamples', 0);

if numel(entries) < 2
    warning('Meta-model OOF training skipped: need at least two entries.');
    return;
end

thresholds = localGetGateThresholds(cfg.filter);
featureCfg = localResolveMetaFeatureConfig(cfg.meta);
metaData = struct('X', zeros(featureCfg.featureDim, 0), 'yRaw', zeros(6, 0), 'yTrue', zeros(6, 0), ...
    'muD', zeros(6, 0), 'muS', zeros(6, 0), 'sampleWeight', zeros(1, 0), ...
    'scenarioCode', zeros(1, 0), 'lambda1', zeros(1, 0), ...
    'anomalyMask', zeros(6, 0), 'abnormalFrame', zeros(1, 0), ...
    'severeFrame', zeros(1, 0), 'substLabel', zeros(1, 0), ...
    'recoveryCleanRejectFrame', zeros(1, 0));

for fold = 1:numel(entries)
    fprintf('OOF fold %d/%d | holdout=%s\n', fold, numel(entries), entries(fold).name);
    foldTrain = entries;
    foldTrain(fold) = [];
    [netFold, normStatsFold] = localTrainFoldStage2Model(foldTrain, entries(fold), cfg);
    foldData = localCollectRejectMetaSamples(entries(fold), netFold, normStatsFold, cfg, thresholds);
    metaData.X = [metaData.X, foldData.X]; %#ok<AGROW>
    metaData.yRaw = [metaData.yRaw, foldData.yRaw]; %#ok<AGROW>
    metaData.yTrue = [metaData.yTrue, foldData.yTrue]; %#ok<AGROW>
    metaData.muD = [metaData.muD, foldData.muD]; %#ok<AGROW>
    metaData.muS = [metaData.muS, foldData.muS]; %#ok<AGROW>
    metaData.sampleWeight = [metaData.sampleWeight, foldData.sampleWeight]; %#ok<AGROW>
    metaData.scenarioCode = [metaData.scenarioCode, foldData.scenarioCode]; %#ok<AGROW>
    metaData.lambda1 = [metaData.lambda1, foldData.lambda1]; %#ok<AGROW>
    metaData.anomalyMask = [metaData.anomalyMask, foldData.anomalyMask]; %#ok<AGROW>
    metaData.abnormalFrame = [metaData.abnormalFrame, foldData.abnormalFrame]; %#ok<AGROW>
    metaData.severeFrame = [metaData.severeFrame, foldData.severeFrame]; %#ok<AGROW>
    metaData.substLabel = [metaData.substLabel, foldData.substLabel]; %#ok<AGROW>
    metaData.recoveryCleanRejectFrame = [metaData.recoveryCleanRejectFrame, foldData.recoveryCleanRejectFrame]; %#ok<AGROW>
end

if isempty(metaData.sampleWeight)
    warning('Meta-model OOF training skipped: reject region is empty.');
    return;
end

svrBundle = localTrainSvrResidualBundle(metaData, cfg.meta, thresholds);
rHatAll = localPredictSvrResidualBundle(svrBundle, metaData.X, false);
svrBundle = localApplyResidualChannelRule(svrBundle, metaData, rHatAll, cfg.meta);
rHat = localPredictSvrResidualBundle(svrBundle, metaData.X);
muR = metaData.muD + rHat;
gateBundle = localTrainGateBundle(metaData, muR, cfg.meta, thresholds);

metaSummary.available = true;
metaSummary.numFolds = numel(entries);
metaSummary.numRejectSamples = size(metaData.X, 2);
metaSummary.rmseRaw = sqrt(mean((metaData.yRaw - metaData.yTrue) .^ 2, 2));
metaSummary.rmseMuD = sqrt(mean((metaData.muD - metaData.yTrue) .^ 2, 2));
metaSummary.rmseMuR = sqrt(mean((muR - metaData.yTrue) .^ 2, 2));
metaSummary.improvedChannelsBySVR = sum(metaSummary.rmseMuR < metaSummary.rmseMuD);
metaSummary.rmseMuS = sqrt(mean((metaData.muS - metaData.yTrue) .^ 2, 2));
metaSummary.featureDim = size(metaData.X, 1);
metaSummary.metaFeatureConfig = featureCfg;
metaSummary.rejectThresholdT1 = thresholds.T1;
metaSummary.channelNames = ["u", "v", "w", "p", "q", "r"];
metaSummary.expertQuality = localBuildOOFExpertQualitySummary(metaData, muR);
metaSummary.expertQuality.recoveryCleanReject = localBuildExpertSliceSummary(metaData, muR, metaData.recoveryCleanRejectFrame ~= 0);
metaSummary.channelRule = localBuildResidualChannelRuleSummary(metaData, muR, svrBundle.enabledMask);
metaSummary.enabledResidualChannels = logical(svrBundle.enabledMask(:));
metaSummary.enabledResidualChannelCount = sum(metaSummary.enabledResidualChannels);
end

function [net, normStats] = localTrainFoldStage2Model(trainEntries, valEntries, cfg)
[XTrainBase, YTrainRaw] = collectDataset(trainEntries, cfg.synthetic);
[XValBase, YValRaw] = collectDataset(valEntries, cfg.synthetic);
[obsInfoLike, ~] = estimateObservationChannelInfo(trainEntries, cfg.filter.baseRcDiag);

lossCfg = cfg.loss;
lossCfg.channelWeights = computeObservationChannelWeights(obsInfoLike, lossCfg);

XTrainRaw = append_causal_multiscale_features(XTrainBase, cfg.model.multiscaleWindows);
XValRaw = append_causal_multiscale_features(XValBase, cfg.model.multiscaleWindows);
numTrainSeq = size(XTrainRaw, 3);

[hidden, fcDim, dropout, miniBatchSize] = localResolveProfileSettings(cfg, numTrainSeq);
normStats = localBuildStage2NormStats(XTrainBase, XTrainRaw, YTrainRaw, cfg.filter.baseRcDiag);
layers = [
    sequenceInputLayer(size(XTrainRaw, 1), "Name", "input")
    lstmLayer(hidden(1), "OutputMode", "sequence", "Name", "lstm1")
    dropoutLayer(dropout, "Name", "drop1")
    lstmLayer(hidden(2), "OutputMode", "sequence", "Name", "lstm2")
    fullyConnectedLayer(fcDim, "Name", "fc1")
    reluLayer("Name", "relu1")
    fullyConnectedLayer(23, "Name", "fc_out")
];
net = dlnetwork(layerGraph(layers));

trailingAvg = [];
trailingAvgSq = [];
iteration = 0;
numEpochs = max(1, min(cfg.train.numEpochs, cfg.train.oofEpochs));
numItersPerEpoch = ceil(numTrainSeq / miniBatchSize);

for epoch = 1:numEpochs
    perm = randperm(numTrainSeq);
    for ii = 1:numItersPerEpoch
        iteration = iteration + 1;
        batchStart = (ii - 1) * miniBatchSize + 1;
        batchEnd = min(ii * miniBatchSize, numTrainSeq);
        batchIdx = perm(batchStart:batchEnd);
        XBatch = dlarray(normalizeInputBatch(XTrainRaw(:, :, batchIdx), normStats), "CTB");
        YBatch = dlarray(single(YTrainRaw(:, :, batchIdx)), "CTB");
        [loss, gradients] = dlfeval(@modelGradientsStage2, net, XBatch, YBatch, normStats, lossCfg);
        [net, trailingAvg, trailingAvgSq] = adamupdate( ...
            net, gradients, trailingAvg, trailingAvgSq, iteration, cfg.train.learnRate);
        if ~isfinite(double(gather(extractdata(loss))))
            error('OOF fold training diverged.');
        end
    end
    if ~isempty(XValBase)
        %#ok<NASGU>
        valLoss = computeDatasetLossBatched(net, XValRaw, YValRaw, normStats, lossCfg, miniBatchSize);
    end
end
end

function [hidden, fcDim, dropout, miniBatchSize] = localResolveProfileSettings(cfg, numTrainSeq)
if cfg.model.profile == "auto"
    if numTrainSeq >= 8000
        profile = "large";
    elseif numTrainSeq >= 2500
        profile = "medium";
    else
        profile = "small";
    end
else
    profile = cfg.model.profile;
end

switch profile
    case "large"
        hidden = [128, 96];
        fcDim = 96;
        dropout = 0.20;
        miniBatchSize = 32;
    case "medium"
        hidden = [96, 64];
        fcDim = 64;
        dropout = 0.15;
        miniBatchSize = 16;
    otherwise
        hidden = [64, 48];
        fcDim = 48;
        dropout = 0.10;
        miniBatchSize = 8;
end
end

function normStats = localBuildStage2NormStats(XBase, XRaw, YRaw, baseRcDiag)
muIn = mean(XRaw, [2, 3]);
sigIn = std(XRaw, 0, [2, 3]);
sigIn(sigIn < 1e-6) = 1;

muY = mean(YRaw(1:6, :, :), [2, 3]);
sigY = std(YRaw(1:6, :, :), 0, [2, 3]);
sigY(sigY < 1e-6) = 1;

deltaTrainRaw = YRaw(1:6, :, :) - XBase(5:10, :, :);
muDelta = mean(deltaTrainRaw, [2, 3]);
sigDelta = std(deltaTrainRaw, 0, [2, 3]);
sigDelta(sigDelta < 1e-6) = 1;

muAir = mean(YRaw(13:15, :, :), [2, 3]);
sigAir = std(YRaw(13:15, :, :), 0, [2, 3]);
sigAir(sigAir < 1e-6) = 1;

normStats = struct();
normStats.muIn = single(muIn);
normStats.sigIn = single(sigIn);
normStats.muY = single(muY);
normStats.sigY = single(sigY);
normStats.muDelta = single(muDelta);
normStats.sigDelta = single(sigDelta);
normStats.muAir = single(muAir);
normStats.sigAir = single(sigAir);
normStats.varLower = single(0.05 * baseRcDiag(:));
normStats.varUpper = single(10.0 * baseRcDiag(:));
normStats.baseInputDim = int32(size(XBase, 1));
normStats.netInputDim = int32(size(XRaw, 1));
end

function foldData = localCollectRejectMetaSamples(entries, net, normStats, cfg, thresholds)
featureCfg = localResolveMetaFeatureConfig(cfg.meta);
foldData = struct('X', zeros(featureCfg.featureDim, 0), 'yRaw', zeros(6, 0), 'yTrue', zeros(6, 0), ...
    'muD', zeros(6, 0), 'muS', zeros(6, 0), 'sampleWeight', zeros(1, 0), ...
    'scenarioCode', zeros(1, 0), 'lambda1', zeros(1, 0), ...
    'anomalyMask', zeros(6, 0), 'abnormalFrame', zeros(1, 0), ...
    'severeFrame', zeros(1, 0), 'substLabel', zeros(1, 0), ...
    'recoveryCleanRejectFrame', zeros(1, 0));

for i = 1:numel(entries)
    if isfield(entries(i), 'logs') && isstruct(entries(i).logs) && ...
            isfield(entries(i).logs, 'loggedFaultLabels') && ~logical(entries(i).logs.loggedFaultLabels)
        continue;
    end
    [XBase, YRaw, ~, ~, frameInfo] = collectDataset(entries(i), cfg.synthetic);
    XNet = append_causal_multiscale_features(XBase, cfg.model.multiscaleWindows);
    batchSize = max(1, min(cfg.train.miniBatchSize, size(XNet, 3)));
    YPred = predictDatasetBatched(net, XNet, normStats, batchSize);

    yMeas = XBase(5:10, :, :);
    [muD, muS, ~, pAbn, pSubst, logVarNorm] = decode_lstm_aux_output(YPred, normStats, 'real', yMeas);
    muD = localClipAuxTensor(muD, yMeas, normStats);
    muS = localClipAuxTensor(muS, yMeas, normStats);
    lambda1 = reshape(YRaw(16, :, :), 1, []);
    scenarioCode = reshape(YRaw(21, :, :), 1, []);
    sampleTime = localResolveEntrySampleTime(entries(i), cfg.synthetic.sampleTime);
    metaState = localBuildMetaFeatureState(XBase, lambda1, sampleTime, thresholds, featureCfg);
    recoveryMask = localBuildRecoveryCleanFrameMaskTensor(frameInfo.anomalyMask, sampleTime, cfg.meta.recoveryCleanWindowSec);
    recoveryCleanReject = reshape(double(recoveryMask), 1, []) ~= 0 & ...
        all(reshape(double(frameInfo.anomalyMask), 6, []) == 0, 1) & ...
        (lambda1 > thresholds.T1);
    rejectMask = lambda1 > thresholds.T1;
    if ~any(rejectMask)
        continue;
    end

    feat = compose_stage2_meta_features(XBase, YRaw(16, :, :), pAbn, pSubst, logVarNorm, muD, muS, yMeas, featureCfg, metaState);
    yRawFlat = reshape(yMeas, 6, []);
    yTrueFlat = reshape(YRaw(1:6, :, :), 6, []);
    muDFlat = reshape(muD, 6, []);
    muSFlat = reshape(muS, 6, []);
    anomalyMaskFlat = reshape(double(frameInfo.anomalyMask), 6, []);
    abnormalFrame = reshape(double(frameInfo.abnormalLabel), 1, []);
    substLabel = reshape(double(frameInfo.substLabel), 1, []);
    severeFrame = double((scenarioCode == 2) | (substLabel ~= 0));
    sampleWeight = 1 + 0.5 * (scenarioCode == 2) + min(max((lambda1 - thresholds.T1) / max(thresholds.T1, 1e-6), 0), 2);
    if isfield(cfg.meta, 'includeRecoveryCleanReject') && cfg.meta.includeRecoveryCleanReject
        sampleWeight = sampleWeight + double(cfg.meta.recoveryCleanRejectWeightBoost) * double(recoveryCleanReject);
    end

    foldData.X = [foldData.X, feat(:, rejectMask)]; %#ok<AGROW>
    foldData.yRaw = [foldData.yRaw, yRawFlat(:, rejectMask)]; %#ok<AGROW>
    foldData.yTrue = [foldData.yTrue, yTrueFlat(:, rejectMask)]; %#ok<AGROW>
    foldData.muD = [foldData.muD, muDFlat(:, rejectMask)]; %#ok<AGROW>
    foldData.muS = [foldData.muS, muSFlat(:, rejectMask)]; %#ok<AGROW>
    foldData.sampleWeight = [foldData.sampleWeight, sampleWeight(rejectMask)]; %#ok<AGROW>
    foldData.scenarioCode = [foldData.scenarioCode, scenarioCode(rejectMask)]; %#ok<AGROW>
    foldData.lambda1 = [foldData.lambda1, lambda1(rejectMask)]; %#ok<AGROW>
    foldData.anomalyMask = [foldData.anomalyMask, anomalyMaskFlat(:, rejectMask)]; %#ok<AGROW>
    foldData.abnormalFrame = [foldData.abnormalFrame, abnormalFrame(rejectMask)]; %#ok<AGROW>
    foldData.severeFrame = [foldData.severeFrame, severeFrame(rejectMask)]; %#ok<AGROW>
    foldData.substLabel = [foldData.substLabel, substLabel(rejectMask)]; %#ok<AGROW>
    foldData.recoveryCleanRejectFrame = [foldData.recoveryCleanRejectFrame, double(recoveryCleanReject(rejectMask))]; %#ok<AGROW>
end
end

function yClip = localClipAuxTensor(yAux, measurements, normStats)
if ~(isstruct(normStats) && isfield(normStats, 'sigDelta'))
    yClip = yAux;
    return;
end

deltaCap = reshape(3 * double(normStats.sigDelta(:)), [6, 1, 1]);
deltaAux = double(yAux) - double(measurements);
deltaAux = min(max(deltaAux, -deltaCap), deltaCap);
yClip = double(measurements) + deltaAux;
end

function svrBundle = localTrainSvrResidualBundle(metaData, metaCfg, thresholds)
featureCfg = localResolveMetaFeatureConfig(metaCfg);
svrBundle = struct();
svrBundle.available = true;
svrBundle.featureDim = size(metaData.X, 1);
svrBundle.featureConfig = featureCfg;
svrBundle.thresholds = thresholds;
svrBundle.channels = cell(6, 1);
svrBundle.enabledMask = true(6, 1);
svrBundle.channelRule = struct();

for ch = 1:6
    idx = localSelectSampleIndices(metaData.sampleWeight, metaCfg.maxRejectSamplesSvr);
    X = metaData.X(:, idx)';
    y = (metaData.yTrue(ch, idx) - metaData.muD(ch, idx))';
    w = metaData.sampleWeight(idx)';
    epsilon = max(1e-4, 0.1 * std(y));
    model = fitrsvm(X, y, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 'auto', ...
        'Standardize', true, ...
        'BoxConstraint', metaCfg.svrBoxConstraint, ...
        'Epsilon', epsilon, ...
        'Weights', w);
    svrBundle.channels{ch} = compact(model);
end
end

function rHat = localPredictSvrResidualBundle(svrBundle, X, applyEnableMask)
if nargin < 3
    applyEnableMask = true;
end

X = double(X);
rHat = zeros(6, size(X, 2));
enabledMask = true(6, 1);
if applyEnableMask && isstruct(svrBundle) && isfield(svrBundle, 'enabledMask') && numel(svrBundle.enabledMask) == 6
    enabledMask = logical(svrBundle.enabledMask(:));
end

for ch = 1:6
    if applyEnableMask && ~enabledMask(ch)
        continue;
    end
    if isempty(svrBundle.channels{ch})
        continue;
    end
    rHat(ch, :) = predict(svrBundle.channels{ch}, X')';
end
end

function svrBundle = localApplyResidualChannelRule(svrBundle, metaData, rHatAll, metaCfg)
muRAll = metaData.muD + rHatAll;
channelSummary = localBuildExpertSliceSummary(metaData, muRAll, metaData.anomalyMask ~= 0);
rmseImproveMinPct = 5.0;
betterShareMin = 0.55;
if nargin >= 4 && isstruct(metaCfg)
    if isfield(metaCfg, 'oofRmseImproveMinPct')
        rmseImproveMinPct = double(metaCfg.oofRmseImproveMinPct);
    end
    if isfield(metaCfg, 'oofBetterShareMin')
        betterShareMin = double(metaCfg.oofBetterShareMin);
    end
end
improvePct = 100 * (1 - channelSummary.rmseMuR ./ max(channelSummary.rmseRaw, 1e-9));
betterShare = channelSummary.betterShareMuR;
enabledMask = channelSummary.countByChannel > 0 & isfinite(improvePct) & ...
    (improvePct > rmseImproveMinPct) & (betterShare > betterShareMin);

svrBundle.enabledMask = logical(enabledMask(:));
svrBundle.channelRule = struct( ...
    'source', "reject_anomalous_channel_oof", ...
    'rmseImproveMinPct', rmseImproveMinPct, ...
    'betterShareMin', betterShareMin, ...
    'countByChannel', channelSummary.countByChannel, ...
    'rmseRaw', channelSummary.rmseRaw, ...
    'rmseMuR', channelSummary.rmseMuR, ...
    'rmseImprovePct', improvePct, ...
    'betterShareMuR', betterShare, ...
    'enabledMaskOOF', logical(enabledMask(:)), ...
    'enabledMask', logical(enabledMask(:)));

fprintf('Residual expert channel rule (reject anomalous channel OOF):\n');
for ch = 1:6
    fprintf('  ch%d | count=%d | raw=%.4f | muR=%.4f | improve=%.2f%% | betterShare=%.2f%% | enable=%d\n', ...
        ch, channelSummary.countByChannel(ch), channelSummary.rmseRaw(ch), ...
        channelSummary.rmseMuR(ch), improvePct(ch), 100 * betterShare(ch), enabledMask(ch));
end
end

function summary = localBuildOOFExpertQualitySummary(metaData, muR)
summary = struct();
summary.reject = localBuildExpertSliceSummary(metaData, muR, true(1, size(metaData.yTrue, 2)));
summary.rejectAbnormal = localBuildExpertSliceSummary(metaData, muR, metaData.abnormalFrame ~= 0);
summary.rejectSevere = localBuildExpertSliceSummary(metaData, muR, metaData.severeFrame ~= 0);
summary.rejectAnomalousChannel = localBuildExpertSliceSummary(metaData, muR, metaData.anomalyMask ~= 0);
summary.recoveryCleanReject = localBuildExpertSliceSummary(metaData, muR, metaData.recoveryCleanRejectFrame ~= 0);
end

function summary = localBuildResidualChannelRuleSummary(metaData, muR, enabledMask)
summary = localBuildExpertSliceSummary(metaData, muR, metaData.anomalyMask ~= 0);
summary.rule = "OOF reject_anomalous_channel: rmseImprovePct>5 && betterShareMuR>0.55";
summary.rmseImproveMuRVsRawPct = 100 * (1 - summary.rmseMuR ./ max(summary.rmseRaw, 1e-9));
summary.enabledMask = logical(enabledMask(:));
end

function holdoutSummary = localEvaluateResidualHoldout(entries, net, normStats, cfg, svrBundle)
holdoutSummary = struct('available', false);
if isempty(entries)
    return;
end

featureCfg = localResolveMetaFeatureConfig(cfg.meta);
windowSec = 2.0;
if isfield(cfg, 'meta') && isfield(cfg.meta, 'recoveryCleanWindowSec')
    windowSec = double(cfg.meta.recoveryCleanWindowSec);
end

holdData = struct( ...
    'yRaw', zeros(6, 0), ...
    'yTrue', zeros(6, 0), ...
    'muD', zeros(6, 0), ...
    'muR', zeros(6, 0), ...
    'muS', zeros(6, 0), ...
    'abnormalChannelMask', zeros(6, 0), ...
    'recoveryCleanFrame', zeros(1, 0));

for i = 1:numel(entries)
    [XBase, YRaw, ~, ~, frameInfo] = collectDataset(entries(i), cfg.synthetic);
    XNet = append_causal_multiscale_features(XBase, cfg.model.multiscaleWindows);
    batchSize = max(1, min(cfg.train.miniBatchSize, size(XNet, 3)));
    YPred = predictDatasetBatched(net, XNet, normStats, batchSize);

    yMeas = XBase(5:10, :, :);
    [muD, muS, ~, pAbn, pSubst, logVarNorm] = decode_lstm_aux_output(YPred, normStats, 'real', yMeas);
    muD = localClipAuxTensor(muD, yMeas, normStats);
    muS = localClipAuxTensor(muS, yMeas, normStats);
    sampleTime = localResolveEntrySampleTime(entries(i), cfg.synthetic.sampleTime);
    metaState = localBuildMetaFeatureState(XBase, reshape(YRaw(16, :, :), 1, []), sampleTime, localGetGateThresholds(cfg.filter), featureCfg);
    metaFeat = compose_stage2_meta_features(XBase, YRaw(16, :, :), pAbn, pSubst, logVarNorm, muD, muS, yMeas, featureCfg, metaState);
    rHat = localPredictSvrResidualBundle(svrBundle, reshape(metaFeat, size(metaFeat, 1), []));
    muR = localClipAuxTensor(reshape(muD, 6, []) + rHat, reshape(yMeas, 6, []), normStats);

    recoveryMask = localBuildRecoveryCleanFrameMaskTensor(frameInfo.anomalyMask, sampleTime, windowSec);

    holdData.yRaw = [holdData.yRaw, reshape(yMeas, 6, [])]; %#ok<AGROW>
    holdData.yTrue = [holdData.yTrue, reshape(YRaw(1:6, :, :), 6, [])]; %#ok<AGROW>
    holdData.muD = [holdData.muD, reshape(muD, 6, [])]; %#ok<AGROW>
    holdData.muR = [holdData.muR, muR]; %#ok<AGROW>
    holdData.muS = [holdData.muS, reshape(muS, 6, [])]; %#ok<AGROW>
    holdData.abnormalChannelMask = [holdData.abnormalChannelMask, reshape(double(frameInfo.anomalyMask ~= 0), 6, [])]; %#ok<AGROW>
    holdData.recoveryCleanFrame = [holdData.recoveryCleanFrame, reshape(double(recoveryMask), 1, [])]; %#ok<AGROW>
end

if isempty(holdData.yTrue)
    return;
end

holdoutSummary.available = true;
holdoutSummary.channelNames = ["u", "v", "w", "p", "q", "r"];
holdoutSummary.recoveryCleanWindowSec = windowSec;
holdoutSummary.abnormalChannel = localBuildHoldoutSliceSummary(holdData, holdData.abnormalChannelMask ~= 0);
holdoutSummary.recoveryClean = localBuildHoldoutSliceSummary(holdData, holdData.recoveryCleanFrame ~= 0);
end

function summary = localBuildHoldoutSliceSummary(holdData, maskInput)
mask = localNormalizeExpertMask(maskInput, size(holdData.yTrue));
summary = localComputeExpertMetrics(holdData.yTrue, holdData.yRaw, holdData.muD, holdData.muR, holdData.muS, mask);
summary.frameCount = sum(any(mask, 1));
summary.countByChannel = reshape(summary.countByChannel, [6, 1]);
end

function sampleTime = localResolveEntrySampleTime(entry, defaultSampleTime)
sampleTime = double(defaultSampleTime);
if isstruct(entry) && isfield(entry, 'logs') && isstruct(entry.logs) && isfield(entry.logs, 'sampleTime')
    sampleTime = double(entry.logs.sampleTime);
end
if ~(isfinite(sampleTime) && sampleTime > 0)
    sampleTime = double(defaultSampleTime);
end
end

function recoveryMask = localBuildRecoveryCleanFrameMaskTensor(anomalyMaskTensor, sampleTime, windowSec)
[~, T, S] = size(anomalyMaskTensor);
recoveryMask = false(1, T, S);
timeVec = (0:T-1) * sampleTime;
for s = 1:S
    anomalyFrame = squeeze(any(anomalyMaskTensor(:, :, s) ~= 0, 1));
    recoveryMask(1, :, s) = localBuildRecoveryCleanFrameMaskSeq(anomalyFrame, timeVec, windowSec);
end
end

function mask = localBuildRecoveryCleanFrameMaskSeq(anomalyFrame, timeVec, windowSec)
anomalyFrame = logical(anomalyFrame(:)');
timeVec = double(timeVec(:)');
mask = false(size(anomalyFrame));
lastEndTime = -inf;
for i = 1:numel(anomalyFrame)
    if anomalyFrame(i)
        lastEndTime = timeVec(i);
        continue;
    end
    if isfinite(lastEndTime) && (timeVec(i) - lastEndTime) <= windowSec
        mask(i) = true;
    end
end
end

function svrBundle = localApplyResidualHoldoutRule(svrBundle, holdoutSummary, metaCfg)
ratioMax = 1.02;
betterShareMin = 0.50;
if nargin >= 3 && isstruct(metaCfg)
    if isfield(metaCfg, 'holdoutRmseRatioMax')
        ratioMax = double(metaCfg.holdoutRmseRatioMax);
    end
    if isfield(metaCfg, 'holdoutBetterShareMin')
        betterShareMin = double(metaCfg.holdoutBetterShareMin);
    end
end

oofEnabledMask = logical(svrBundle.enabledMask(:));
abnormalPass = false(6, 1);
recoveryPass = false(6, 1);
if isstruct(holdoutSummary) && isfield(holdoutSummary, 'available') && logical(holdoutSummary.available)
    abnormalPass = localBuildHoldoutChannelPass(holdoutSummary.abnormalChannel, ratioMax, betterShareMin);
    recoveryPass = localBuildHoldoutChannelPass(holdoutSummary.recoveryClean, ratioMax, betterShareMin);
end

enabledMask = oofEnabledMask & abnormalPass & recoveryPass;
svrBundle.enabledMask = logical(enabledMask(:));
svrBundle.channelRule.source = "oof_holdout_dual_screen";
svrBundle.channelRule.holdoutRmseRatioMax = ratioMax;
svrBundle.channelRule.holdoutBetterShareMin = betterShareMin;
svrBundle.channelRule.enabledMaskOOF = oofEnabledMask;
svrBundle.channelRule.holdoutAbnormalPass = abnormalPass;
svrBundle.channelRule.holdoutRecoveryPass = recoveryPass;
svrBundle.channelRule.enabledMask = logical(enabledMask(:));

fprintf('Residual expert hold-out screen (abnormal + recovery):\n');
for ch = 1:6
    fprintf('  ch%d | oof=%d | abnormal=%d | recovery=%d | final=%d\n', ...
        ch, oofEnabledMask(ch), abnormalPass(ch), recoveryPass(ch), enabledMask(ch));
end
end

function passMask = localBuildHoldoutChannelPass(summary, ratioMax, betterShareMin)
passMask = summary.countByChannel > 0 & ...
    isfinite(summary.rmseMuR) & isfinite(summary.rmseRaw) & ...
    (summary.rmseMuR <= ratioMax .* max(summary.rmseRaw, 1e-9)) & ...
    (summary.betterShareMuR >= betterShareMin);
passMask = logical(passMask(:));
end

function ruleMode = localResolveChannelRuleMode(metaCfg)
ruleMode = "oof_holdout_dual_screen";
if nargin >= 1 && isstruct(metaCfg) && isfield(metaCfg, 'channelRuleMode')
    ruleMode = string(metaCfg.channelRuleMode);
end
end

function tf = localMetaTrainingEnabled(cfg)
tf = false;
if ~(isstruct(cfg) && isfield(cfg, 'meta') && isfield(cfg.meta, 'enabled') && cfg.meta.enabled)
    return;
end

mode = "dual_head";
if isfield(cfg, 'runtime') && isfield(cfg.runtime, 'modelMode')
    mode = string(cfg.runtime.modelMode);
end
tf = any(strcmpi(mode, ["two_stage_svr_residual", "two_stage_moe_svr"]));
end

function featureCfg = localResolveMetaFeatureConfig(metaCfg)
featureCfg = stage2_meta_feature_spec("legacy");
if nargin < 1 || ~isstruct(metaCfg)
    return;
end
if isfield(metaCfg, 'featureSet') && strlength(string(metaCfg.featureSet)) > 0
    featureCfg = stage2_meta_feature_spec(metaCfg.featureSet);
elseif isfield(metaCfg, 'featureConfig') && ~isempty(metaCfg.featureConfig)
    featureCfg = stage2_meta_feature_spec(metaCfg.featureConfig);
end
end

function metaState = localBuildMetaFeatureState(XBase, lambda1, sampleTime, thresholds, featureCfg)
metaState = struct();
featureCfg = stage2_meta_feature_spec(featureCfg);
if featureCfg.extraDim <= 0
    return;
end

[~, T, S] = size(XBase);
lambda1 = reshape(double(lambda1), T, S)';
ctxInnovation = squeeze(sqrt(mean(double(XBase(29:34, :, :)) .^ 2, 1)));
if isvector(ctxInnovation)
    ctxInnovation = reshape(ctxInnovation, [T, S])';
else
    ctxInnovation = double(ctxInnovation)';
end

prevMode = zeros(S, T);
prevLambda1 = zeros(S, T);
fallbackStreak = zeros(S, T);
timeSinceLastAnomalyEnd = zeros(S, T);

for s = 1:S
    prevModeVal = 0;
    prevLambdaVal = 0;
    fallbackVal = 0;
    timeSinceVal = 0;
    anomalyProxyActive = false;
    for t = 1:T
        prevMode(s, t) = prevModeVal;
        prevLambda1(s, t) = prevLambdaVal;
        fallbackStreak(s, t) = fallbackVal;
        timeSinceLastAnomalyEnd(s, t) = min(timeSinceVal, featureCfg.timeSinceCapSec);

        rejectNow = lambda1(s, t) > thresholds.T1;
        modeProxy = 3 * double(rejectNow);
        if modeProxy == 3
            fallbackVal = fallbackVal + 1;
        else
            fallbackVal = 0;
        end

        if rejectNow
            anomalyProxyActive = true;
            timeSinceVal = 0;
        else
            if anomalyProxyActive
                anomalyProxyActive = false;
                timeSinceVal = 0;
            else
                timeSinceVal = timeSinceVal + sampleTime;
            end
        end

        prevModeVal = modeProxy;
        prevLambdaVal = lambda1(s, t);
    end
end

metaState.prevMode = reshape(prevMode', 1, []);
metaState.prevLambda1 = reshape(prevLambda1', 1, []);
metaState.fallbackStreak = reshape(fallbackStreak', 1, []);
metaState.timeSinceLastAnomalyEnd = reshape(timeSinceLastAnomalyEnd', 1, []);
metaState.ctxInnovation = reshape(double(ctxInnovation'), 1, []);
end

function summary = localBuildExpertSliceSummary(metaData, muR, maskInput)
mask = localNormalizeExpertMask(maskInput, size(metaData.yTrue));
summary = localComputeExpertMetrics(metaData.yTrue, metaData.yRaw, metaData.muD, muR, metaData.muS, mask);
summary.frameCount = sum(any(mask, 1));
summary.countByChannel = reshape(summary.countByChannel, [6, 1]);
end

function mask = localNormalizeExpertMask(maskInput, targetSize)
if isempty(maskInput)
    mask = false(targetSize);
elseif isvector(maskInput)
    mask = repmat(logical(maskInput(:)'), targetSize(1), 1);
else
    mask = logical(maskInput);
end

if ~isequal(size(mask), targetSize)
    error('Expert slice mask size mismatch.');
end
end

function summary = localComputeExpertMetrics(yTrue, yRaw, muD, muR, muS, mask)
summary = struct();
summary.countByChannel = sum(mask, 2);
summary.totalCount = sum(mask(:));

[summary.rmseRaw, summary.maeRaw] = localMaskedExpertError(yRaw, yTrue, mask);
[summary.rmseMuD, summary.maeMuD] = localMaskedExpertError(muD, yTrue, mask);
[summary.rmseMuR, summary.maeMuR] = localMaskedExpertError(muR, yTrue, mask);
[summary.rmseMuS, summary.maeMuS] = localMaskedExpertError(muS, yTrue, mask);

summary.betterShareRaw = nan(6, 1);
summary.betterShareMuD = localMaskedBetterShare(muD, yRaw, yTrue, mask);
summary.betterShareMuR = localMaskedBetterShare(muR, yRaw, yTrue, mask);
summary.betterShareMuS = localMaskedBetterShare(muS, yRaw, yTrue, mask);
end

function [rmseVal, maeVal] = localMaskedExpertError(pred, target, mask)
rmseVal = nan(6, 1);
maeVal = nan(6, 1);
for ch = 1:6
    idx = find(mask(ch, :));
    if isempty(idx)
        continue;
    end
    err = double(pred(ch, idx) - target(ch, idx));
    rmseVal(ch) = sqrt(mean(err .^ 2));
    maeVal(ch) = mean(abs(err));
end
end

function share = localMaskedBetterShare(pred, raw, target, mask)
share = nan(6, 1);
for ch = 1:6
    idx = find(mask(ch, :));
    if isempty(idx)
        continue;
    end
    errPred = abs(double(pred(ch, idx) - target(ch, idx)));
    errRaw = abs(double(raw(ch, idx) - target(ch, idx)));
    share(ch) = mean(errPred < errRaw);
end
end

function gateBundle = localTrainGateBundle(metaData, muR, metaCfg, thresholds)
featureCfg = localResolveMetaFeatureConfig(metaCfg);
idx = localSelectSampleIndices(metaData.sampleWeight, metaCfg.maxRejectSamplesGate);
X = metaData.X(:, idx);
w = metaData.sampleWeight(idx);
muX = mean(X, 2);
sigX = std(X, 0, 2);
sigX(sigX < 1e-6) = 1;
Xn = (X - muX) ./ sigX;

W = zeros(3, size(Xn, 1), 6, 'single');
b = zeros(3, 6, 'single');
for ch = 1:6
    experts = [metaData.yRaw(ch, idx); muR(ch, idx); metaData.muS(ch, idx)];
    yTrue = metaData.yTrue(ch, idx);
    softLabel = localBuildExpertSoftLabels(experts, yTrue);
    [W(:, :, ch), b(:, ch)] = localTrainSoftmaxGateModel(Xn, softLabel, w, metaCfg);
end

gateBundle = struct();
gateBundle.available = true;
gateBundle.featureDim = size(X, 1);
gateBundle.featureConfig = featureCfg;
gateBundle.muX = single(muX);
gateBundle.sigX = single(sigX);
gateBundle.W = W;
gateBundle.b = b;
gateBundle.thresholds = thresholds;
end

function softLabel = localBuildExpertSoftLabels(experts, yTrue)
err = (experts - reshape(double(yTrue), 1, [])) .^ 2;
scale = median(err(:));
scale = max(scale, 1e-6);
score = exp(-err / scale);
softLabel = score ./ max(sum(score, 1), 1e-9);
end

function [W, b] = localTrainSoftmaxGateModel(Xn, softLabel, sampleWeight, metaCfg)
numFeat = size(Xn, 1);
W = zeros(3, numFeat);
b = zeros(3, 1);
lr = metaCfg.gateLearnRate;
sampleWeight = reshape(double(sampleWeight), 1, []);
sampleWeight = sampleWeight / max(sum(sampleWeight), 1e-9);

for iter = 1:metaCfg.gateNumIters
    score = W * Xn + b;
    score = score - max(score, [], 1);
    prob = exp(score);
    prob = prob ./ max(sum(prob, 1), 1e-9);
    diff = (prob - softLabel) .* sampleWeight;
    gradW = diff * Xn' + metaCfg.gateReg * W;
    gradB = sum(diff, 2);
    W = W - lr * gradW;
    b = b - lr * gradB;
    lr = max(lr * 0.995, 0.02 * metaCfg.gateLearnRate);
end
end

function idx = localSelectSampleIndices(sampleWeight, maxCount)
N = numel(sampleWeight);
if isempty(maxCount) || maxCount <= 0 || N <= maxCount
    idx = 1:N;
    return;
end
perm = randperm(N, maxCount);
idx = sort(perm);
end

function thresholds = localGetGateThresholds(S)
thresholds = struct('T1', 16.812, 'T2', 14.449, 'T1ch', 6.635, 'T2ch', 5.024);
if ~isstruct(S)
    return;
end

if isfield(S, 'gateThresholds') && isstruct(S.gateThresholds)
    src = S.gateThresholds;
else
    src = S;
end

fields = fieldnames(thresholds);
for i = 1:numel(fields)
    key = fields{i};
    if isfield(src, key)
        thresholds.(key) = double(src.(key));
    end
end
end

function state = localPushRuntimeOverrides(modelFile, runtimeCfg)
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

assignin('base', 'stage2ModelFileOverride', modelFile);
assignin('base', 'stage2RuntimeModelModeOverride', char(runtimeCfg.modelMode));
assignin('base', 'stage2UseGateCalibrationOverride', logical(runtimeCfg.useGateCalibration));
runtimeOverride = localBuildRuntimeOverride(runtimeCfg);
if isempty(fieldnames(runtimeOverride))
    evalin('base', 'clear stage2RuntimeOverride');
else
    assignin('base', 'stage2RuntimeOverride', runtimeOverride);
end
if isfield(runtimeCfg, 'gateCalibrationFile')
    assignin('base', 'stage2GateCalibrationOverride', char(runtimeCfg.gateCalibrationFile));
else
    assignin('base', 'stage2GateCalibrationOverride', '');
end
if isfield(runtimeCfg, 'gateThresholds')
    assignin('base', 'stage2GateThresholdOverride', runtimeCfg.gateThresholds);
else
    evalin('base', 'clear stage2GateThresholdOverride');
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
fields = {'gammaD', 'gammaS', 'usePredictedVariance', ...
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

function out = mergeStruct(base, override)
out = base;
f = fieldnames(override);
for i = 1:numel(f)
    k = f{i};
    if isfield(out, k) && isstruct(out.(k)) && isstruct(override.(k))
        out.(k) = mergeStruct(out.(k), override.(k));
    else
        out.(k) = override.(k);
    end
end
end

