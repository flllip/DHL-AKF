function [Vw_est, air_est, mode_flag, lambda_1, lambda_2, debugInfo] = lstm_ekf_fusion_step(state, controls, measurements, dt, reset)
% Stage-2 EKF-LSTM fusion estimator with two-stage testing.
%
% mode_flag:
%   0 -> raw measurement accepted
%   1 -> denoise auxiliary measurement accepted
%   2 -> substitution auxiliary measurement accepted
%   3 -> channel-wise fallback / conservative update

persistent xHat P RcGate RcUpdateBase qAccelBase qBoostCounter qBoostHorizon initialized
persistent xCtx PCtx
persistent T1 T2 T1ch T2ch
persistent net normStats ctxLen baseBuf
persistent svrBundle gateBundle
persistent metaFeatureConfig metaPrevMode metaPrevLambda1 metaFallbackStreak metaTimeSinceAnomalyEnd metaAnomalyProxyActive
persistent stageMode runtimeModelMode enableSubst usePredictedVariance gammaD gammaS
persistent substEnterProb substExitProb substEnterCount substExitCount
persistent substActive substHighCounter substLowCounter
persistent recoveryEnabled recoveryFallbackEnterCount recoveryLambdaEnter recoveryCtxRawOkMinChannels
persistent recoveryHoldOffAuxCount recoveryQBoostHorizon recoveryExitCleanCount
persistent recoveryActive recoveryHoldOffCounter recoveryFallbackStreak recoveryHighLambdaStreak
persistent recoveryCleanCounter recoveryPending reanchorCount
persistent runtimeInfo

if isempty(initialized)
    initialized = false;
end

debugInfo = struct( ...
    'preferredCandidate', 0, ...
    'acceptedCandidate', 0, ...
    'bestCandidate', 0, ...
    'lambdaCandMuR', 0, ...
    'lambdaCandMuS', 0, ...
    'useSubstPrimary', false, ...
    'substActive', false, ...
    'pSubst', 0, ...
    'skipUpdate', false, ...
    'fallbackValidCount', 0, ...
    'rawScore', zeros(6, 1), ...
    'auxScore', zeros(6, 1), ...
    'selectedSource', zeros(6, 1), ...
    'dominantFallbackChannel', 0, ...
    'recoveryActive', false, ...
    'recoveryTriggered', false, ...
    'ctxRawScore', zeros(6, 1), ...
    'ctxRawOkCount', 0, ...
    'reanchorCount', 0, ...
    'usedGateCalibration', false, ...
    'pAbn', 0, ...
    'muD', zeros(6, 1), ...
    'muR', zeros(6, 1), ...
    'muS', zeros(6, 1), ...
    'gateWeights', zeros(3, 6), ...
    'lambda2Mix', 0, ...
    'runtime', struct());

if ~localUseLegacy4StateFusion()
    [Vw_est, air_est, mode_flag, lambda_1, lambda_2, debugInfo] = ...
        localTwoStateFusionStep(state, controls, measurements, dt, reset, debugInfo);
    return;
end

if ~initialized || reset ~= 0
    xHat = zeros(4, 1);
    P = diag([10, 10, 2, 2]);
    xCtx = zeros(4, 1);
    PCtx = diag([10, 10, 2, 2]);
    qAccelBase = [0.20, 0.20] .^ 2;
    qBoostCounter = 0;

    T1 = 16.812;
    T2 = 14.449;
    T1ch = 6.635;
    T2ch = 5.024;

    modelFile = resolve_stage2_model_file('lstm_akf_net.mat');
    bundle = load_stage2_runtime_bundle(modelFile);
    net = bundle.net;
    normStats = bundle.normStats;
    svrBundle = bundle.svrBundle;
    gateBundle = bundle.gateBundle;
    runtimeInfo = bundle.runtime;
    metaFeatureConfig = stage2_meta_feature_spec(runtimeInfo.metaFeatureConfig);
    RcGate = runtimeInfo.Rc;
    RcUpdateBase = diag(runtimeInfo.baseRcDiag);
    stageMode = char(runtimeInfo.stageMode);
    runtimeModelMode = char(runtimeInfo.runtimeModelMode);
    enableSubst = logical(runtimeInfo.enableSubst);
    usePredictedVariance = logical(runtimeInfo.usePredictedVariance);
    gammaD = runtimeInfo.gammaD;
    gammaS = runtimeInfo.gammaS;
    substEnterProb = runtimeInfo.substEnterProb;
    substExitProb = runtimeInfo.substExitProb;
    substEnterCount = runtimeInfo.substEnterCount;
    substExitCount = runtimeInfo.substExitCount;
    qBoostHorizon = runtimeInfo.qBoostHorizon;
    recoveryEnabled = logical(runtimeInfo.recoveryEnabled);
    recoveryFallbackEnterCount = runtimeInfo.recoveryFallbackEnterCount;
    recoveryLambdaEnter = runtimeInfo.recoveryLambdaEnter;
    recoveryCtxRawOkMinChannels = runtimeInfo.recoveryCtxRawOkMinChannels;
    recoveryHoldOffAuxCount = runtimeInfo.recoveryHoldOffAuxCount;
    recoveryQBoostHorizon = runtimeInfo.recoveryQBoostHorizon;
    recoveryExitCleanCount = runtimeInfo.recoveryExitCleanCount;
    ctxLen = runtimeInfo.ctxLen;

    substActive = false;
    substHighCounter = 0;
    substLowCounter = 0;
    recoveryActive = false;
    recoveryHoldOffCounter = 0;
    recoveryFallbackStreak = 0;
    recoveryHighLambdaStreak = 0;
    recoveryCleanCounter = 0;
    recoveryPending = false;
    reanchorCount = 0;
    metaPrevMode = 0;
    metaPrevLambda1 = 0;
    metaFallbackStreak = 0;
    metaTimeSinceAnomalyEnd = 0;
    metaAnomalyProxyActive = false;
    baseBuf = zeros(double(normStats.baseInputDim), ctxLen, 'single');
    thresholds = runtimeInfo.gateThresholds;
    T1 = thresholds.T1;
    T2 = thresholds.T2;
    T1ch = thresholds.T1ch;
    T2ch = thresholds.T2ch;
    initialized = true;
end

dt = localSanitizeDt(dt);
[F, Qk] = localCVModel(dt, qAccelBase, qBoostCounter > 0);
xPred = F * xHat;
PPred = F * P * F' + Qk;

zPred = localModelObs(state, controls, xPred(1:2));
H2 = compute_jacobian_numerical(xPred(1:2), state, controls, zPred);
H = [H2, zeros(6, 2)];

[Fctx, Qctx] = localCVModel(dt, qAccelBase, false);
xCtxPred = Fctx * xCtx;
PCtxPred = Fctx * PCtx * Fctx' + Qctx;
zCtxPred = localModelObs(state, controls, xCtxPred(1:2));
H2Ctx = compute_jacobian_numerical(xCtxPred(1:2), state, controls, zCtxPred);
HCtx = [H2Ctx, zeros(6, 2)];
[yCtxAnchor, RCtxAnchor] = localBuildRecoveryRawUpdate(measurements(:), zCtxPred, HCtx * PCtxPred * HCtx' + RcUpdateBase, RcUpdateBase, T1ch);
SCtxGate = HCtx * PCtxPred * HCtx' + RcUpdateBase;
SCtx = HCtx * PCtxPred * HCtx' + RCtxAnchor;
KCtx = (PCtxPred * HCtx') / SCtx;
xCtx = xCtxPred + KCtx * (yCtxAnchor - zCtxPred);
Ictx = eye(4);
PCtx = (Ictx - KCtx * HCtx) * PCtxPred * (Ictx - KCtx * HCtx)' + KCtx * RCtxAnchor * KCtx';

baseFeat = compose_lstm_fusion_input(xCtxPred, measurements(:), state, controls, zCtxPred);
baseBuf = localPushBuffer(baseBuf, single(baseFeat));
netSeq = append_causal_multiscale_features(baseBuf(:, any(baseBuf, 1)), double(normStats.multiscaleWindows(:)'));
netSeqN = (netSeq - normStats.muIn) ./ normStats.sigIn;

e1 = measurements(:) - zPred;
S1 = H * PPred * H' + RcGate;
lambda_1 = localMahalanobis(e1, S1);
lambda_2 = 0;
skipUpdate = false;
ctxRawScore = ((measurements(:) - zCtxPred) .^ 2) ./ max(diag(SCtxGate), 1e-9);
ctxRawOkCount = sum(ctxRawScore <= T1ch);
metaRuntimeState = struct( ...
    'prevMode', metaPrevMode, ...
    'prevLambda1', metaPrevLambda1, ...
    'fallbackStreak', metaFallbackStreak, ...
    'timeSinceLastAnomalyEnd', metaTimeSinceAnomalyEnd, ...
    'ctxInnovation', sqrt(mean((measurements(:) - zCtxPred(:)) .^ 2)));
debugInfo.ctxRawScore = ctxRawScore;
debugInfo.ctxRawOkCount = ctxRawOkCount;
debugInfo.recoveryActive = logical(recoveryActive);
debugInfo.reanchorCount = double(reanchorCount);
triggerRecovery = false;

if recoveryEnabled && ctxRawOkCount >= recoveryCtxRawOkMinChannels
    highLambdaReady = logical(lambda_1 > recoveryLambdaEnter) && ...
        (recoveryHighLambdaStreak + 1 >= recoveryFallbackEnterCount);
    fallbackReady = recoveryFallbackStreak >= recoveryFallbackEnterCount;
    triggerRecovery = (~recoveryActive && recoveryHoldOffCounter <= 0) && (recoveryPending || highLambdaReady || fallbackReady);
end

if triggerRecovery
    recoveryActive = true;
    recoveryHoldOffCounter = recoveryHoldOffAuxCount;
    recoveryCleanCounter = 0;
    recoveryPending = false;
    reanchorCount = reanchorCount + 1;
    xHat = xCtx;
    P = PCtx;
    qBoostCounter = max(qBoostCounter, recoveryQBoostHorizon);
    debugInfo.recoveryTriggered = true;
    debugInfo.recoveryActive = true;
    debugInfo.reanchorCount = double(reanchorCount);
end

useRecoveryRawOnly = recoveryEnabled && recoveryActive && recoveryHoldOffCounter > 0;

if useRecoveryRawOnly
    xPred = xCtxPred;
    PPred = PCtxPred;
    zPred = zCtxPred;
    H = HCtx;
    S1 = H * PPred * H' + RcGate;
    lambda_1 = localMahalanobis(measurements(:) - zPred, S1);
    [yUpdate, RUpdate, selectedSource, dominantFallbackChannel] = ...
        localBuildRecoveryRawUpdate(measurements(:), zCtxPred, SCtx, RcUpdateBase, T1ch);
    lambda_2 = localMahalanobis(yUpdate - zPred, H * PPred * H' + RUpdate);
    debugInfo.rawScore = ctxRawScore;
    debugInfo.selectedSource = selectedSource;
    debugInfo.dominantFallbackChannel = dominantFallbackChannel;
    debugInfo.fallbackValidCount = sum(selectedSource ~= 2);
    if all(selectedSource == 0)
        mode_flag = 0;
    else
        mode_flag = 3;
    end
elseif strcmpi(runtimeModelMode, 'denoise_always')
    [muDenoise, muSubst, varAux, pAbn, pSubst] = localPredictAux(net, netSeqN, normStats, measurements(:));
    [yCand, Rtest, RupdateCand] = localBuildCandidate(1, muDenoise, muSubst, varAux, RcGate, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements(:));

    yUpdate = yCand;
    RUpdate = RupdateCand;
    lambda_2 = localMahalanobis(yCand - zPred, H * PPred * H' + Rtest);
    mode_flag = 1;
    debugInfo.preferredCandidate = 1;
    debugInfo.acceptedCandidate = 1;
    debugInfo.bestCandidate = 1;
    debugInfo.pAbn = double(pAbn);
    debugInfo.pSubst = double(pSubst);
elseif strcmpi(runtimeModelMode, 'channel_blend_always')
    [muDenoise, muSubst, varAux, pAbn, pSubst] = localPredictAux(net, netSeqN, normStats, measurements(:));
    [yCand, Rtest, RupdateCand] = localBuildCandidate(1, muDenoise, muSubst, varAux, RcGate, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements(:));
    lambda_2 = localMahalanobis(yCand - zPred, H * PPred * H' + Rtest);
    [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
        localChannelFallback(measurements(:), yCand, zPred, S1, H * PPred * H' + Rtest, RcUpdateBase, RupdateCand, T1ch, T2ch);
    if validCount < 3
        yUpdate = measurements(:);
        RUpdate = RcUpdateBase;
        mode_flag = 0;
    else
        mode_flag = localInferBlendMode(yUpdate, measurements(:), yCand);
    end
    debugInfo.preferredCandidate = 1;
    debugInfo.acceptedCandidate = 1;
    debugInfo.bestCandidate = 1;
    debugInfo.fallbackValidCount = validCount;
    debugInfo.rawScore = rawScore;
    debugInfo.auxScore = auxScore;
    debugInfo.selectedSource = selectedSource;
    debugInfo.dominantFallbackChannel = dominantFallbackChannel;
    debugInfo.pAbn = double(pAbn);
    debugInfo.pSubst = double(pSubst);
elseif lambda_1 <= T1
    yUpdate = measurements(:);
    RUpdate = RcUpdateBase;
    mode_flag = 0;
else
    [muDenoise, muSubst, varAux, pAbn, pSubst, logVarNorm] = localPredictAux(net, netSeqN, normStats, measurements(:));
    [useSubstPrimary, substActive, substHighCounter, substLowCounter] = ...
        localUpdateSubstState(pSubst, substActive, substHighCounter, substLowCounter, ...
        substEnterProb, substExitProb, substEnterCount, substExitCount, ...
        (strcmpi(runtimeModelMode, 'dual_head') || strcmpi(runtimeModelMode, 'two_stage_svr_residual')) && enableSubst);

    debugInfo.useSubstPrimary = logical(useSubstPrimary);
    debugInfo.substActive = logical(substActive);
    debugInfo.pAbn = double(pAbn);
    debugInfo.pSubst = double(pSubst);
    debugInfo.muD = muDenoise;
    debugInfo.muS = muSubst;

    if strcmpi(runtimeModelMode, 'two_stage_svr_residual') && isstruct(svrBundle) && isfield(svrBundle, 'available') && logical(svrBundle.available)
        muResidual = localBuildResidualExpertMeasurement(baseFeat, lambda_1, muDenoise, muSubst, pAbn, pSubst, logVarNorm, measurements(:), svrBundle, normStats, metaFeatureConfig, metaRuntimeState);
        debugInfo.muR = muResidual;
        if enableSubst && useSubstPrimary
            candidateOrder = [2, 1];
        elseif enableSubst
            candidateOrder = [1, 2];
        else
            candidateOrder = 1;
        end

        accepted = false;
        bestLambda = inf;
        bestY = measurements(:);
        bestRtest = RcGate;
        bestRupdate = RcUpdateBase;
        bestMode = 0;
        debugInfo.preferredCandidate = candidateOrder(1);

        for candidate = candidateOrder
            if candidate == 2
                [yCand, Rtest, RupdateCand, modeCandidate] = ...
                    localBuildCandidate(2, muDenoise, muSubst, varAux, RcGate, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements(:));
            else
                yCand = muResidual;
                Rtest = diag(gammaD .* diag(RcGate));
                RupdateCand = diag(gammaD .* diag(RcUpdateBase));
                modeCandidate = 1;
            end

            lambdaCand = localMahalanobis(yCand - zPred, H * PPred * H' + Rtest);
            if candidate == 2
                debugInfo.lambdaCandMuS = lambdaCand;
            else
                debugInfo.lambdaCandMuR = lambdaCand;
            end
            if lambdaCand < bestLambda
                bestLambda = lambdaCand;
                bestY = yCand;
                bestRtest = Rtest;
                bestRupdate = RupdateCand;
                bestMode = modeCandidate;
            end
            if lambdaCand <= T2
                accepted = true;
                lambda_2 = lambdaCand;
                yUpdate = yCand;
                RUpdate = RupdateCand;
                mode_flag = modeCandidate;
                debugInfo.acceptedCandidate = candidate;
                break;
            end
        end

        debugInfo.bestCandidate = bestMode;
        if ~accepted
            [ctxAccepted, yCtx, Rctx, lambdaCtx, selectedCtx] = localBuildContextAnchorCandidate( ...
                measurements(:), zCtxPred, SCtxGate, RcGate, RcUpdateBase, HCtx, PCtxPred, ...
                T1ch, T2, recoveryCtxRawOkMinChannels);
            if ctxAccepted
                recoveryActive = true;
                recoveryHoldOffCounter = recoveryHoldOffAuxCount;
                recoveryCleanCounter = 0;
                recoveryPending = false;
                reanchorCount = reanchorCount + 1;
                qBoostCounter = max(qBoostCounter, recoveryQBoostHorizon);
                xPred = xCtxPred;
                PPred = PCtxPred;
                zPred = zCtxPred;
                H = HCtx;
                lambda_2 = lambdaCtx;
                yUpdate = yCtx;
                RUpdate = Rctx;
                mode_flag = 1;
                debugInfo.acceptedCandidate = 3;
                debugInfo.bestCandidate = 3;
                debugInfo.recoveryTriggered = true;
                debugInfo.recoveryActive = true;
                debugInfo.reanchorCount = double(reanchorCount);
                debugInfo.selectedSource = selectedCtx;
                debugInfo.fallbackValidCount = sum(selectedCtx == 0);
                debugInfo.rawScore = ctxRawScore;
            else
                [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
                    localChannelFallback(measurements(:), bestY, zPred, S1, H * PPred * H' + bestRtest, RcUpdateBase, bestRupdate, T1ch, T2ch);
                lambdaHybrid = localMahalanobis(yUpdate - zPred, H * PPred * H' + RUpdate);
                lambda_2 = lambdaHybrid;
                debugInfo.fallbackValidCount = validCount;
                debugInfo.rawScore = rawScore;
                debugInfo.auxScore = auxScore;
                debugInfo.selectedSource = selectedSource;
                debugInfo.dominantFallbackChannel = dominantFallbackChannel;
                debugInfo.lambdaHybrid = lambdaHybrid;
                if validCount >= 3 && lambdaHybrid <= T2
                    mode_flag = bestMode;
                    debugInfo.acceptedCandidate = max(bestMode, 1);
                else
                    mode_flag = 3;
                end
                if validCount < 3
                    skipUpdate = true;
                    qBoostCounter = qBoostHorizon;
                    yUpdate = measurements(:);
                    RUpdate = RcUpdateBase;
                    lambda_2 = bestLambda;
                end
            end
        end
    elseif strcmpi(runtimeModelMode, 'two_stage_moe_svr') && localMetaModelsAvailable(svrBundle, gateBundle)
        [yMix, RtestMix, RupdateMix, lambdaMix, modeCandidate, mixDebug] = localBuildMoECandidate( ...
            baseFeat, lambda_1, muDenoise, muSubst, pAbn, pSubst, logVarNorm, ...
            measurements(:), RcGate, RcUpdateBase, gammaD, gammaS, zPred, H, PPred, ...
            svrBundle, gateBundle, normStats, T1, metaFeatureConfig, metaRuntimeState);
        lambda_2 = lambdaMix;
        debugInfo.preferredCandidate = 1;
        debugInfo.bestCandidate = modeCandidate;
        debugInfo.muR = mixDebug.muR;
        debugInfo.gateWeights = mixDebug.gateWeights;
        debugInfo.lambda2Mix = lambdaMix;
        if lambdaMix <= T2
            yUpdate = yMix;
            RUpdate = RupdateMix;
            mode_flag = modeCandidate;
            debugInfo.acceptedCandidate = modeCandidate;
        else
            [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
                localChannelFallback(measurements(:), yMix, zPred, S1, H * PPred * H' + RtestMix, RcUpdateBase, RupdateMix, T1ch, T2ch);
            mode_flag = 3;
            debugInfo.fallbackValidCount = validCount;
            debugInfo.rawScore = rawScore;
            debugInfo.auxScore = auxScore;
            debugInfo.selectedSource = selectedSource;
            debugInfo.dominantFallbackChannel = dominantFallbackChannel;
            if validCount < 3
                skipUpdate = true;
                qBoostCounter = qBoostHorizon;
                yUpdate = measurements(:);
                RUpdate = RcUpdateBase;
            end
        end
    elseif strcmpi(runtimeModelMode, 'denoise_only_no_second_test')
        [yCand, Rtest, RupdateCand] = localBuildCandidate(1, muDenoise, muSubst, varAux, RcGate, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements(:));
        lambda_2 = localMahalanobis(yCand - zPred, H * PPred * H' + Rtest);
        yUpdate = yCand;
        RUpdate = RupdateCand;
        mode_flag = 1;
        debugInfo.preferredCandidate = 1;
        debugInfo.acceptedCandidate = 1;
        debugInfo.bestCandidate = 1;
    else

        if (strcmpi(runtimeModelMode, 'dual_head') || strcmpi(runtimeModelMode, 'two_stage_moe_svr')) && enableSubst
            if useSubstPrimary
                candidateOrder = [2, 1];
            else
                candidateOrder = [1, 2];
            end
        else
            candidateOrder = 1;
        end

        accepted = false;
        bestLambda = inf;
        bestY = measurements(:);
        bestRtest = RcGate;
        bestRupdate = RcUpdateBase;
        bestMode = 0;
        debugInfo.preferredCandidate = candidateOrder(1);

        for candidate = candidateOrder
            [yCand, Rtest, RupdateCand, modeCandidate] = ...
                localBuildCandidate(candidate, muDenoise, muSubst, varAux, RcGate, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements(:));

            e2 = yCand - zPred;
            S2 = H * PPred * H' + Rtest;
            lambdaCand = localMahalanobis(e2, S2);

            if lambdaCand < bestLambda
                bestLambda = lambdaCand;
                bestY = yCand;
                bestRtest = Rtest;
                bestRupdate = RupdateCand;
                bestMode = modeCandidate;
            end
            if candidate == 2
                debugInfo.lambdaCandMuS = lambdaCand;
            elseif candidate == 1
                debugInfo.lambdaCandMuR = lambdaCand;
            end

            if lambdaCand <= T2
                accepted = true;
                lambda_2 = lambdaCand;
                yUpdate = yCand;
                RUpdate = RupdateCand;
                mode_flag = modeCandidate;
                debugInfo.acceptedCandidate = candidate;
                break;
            end
        end

        debugInfo.bestCandidate = bestMode;

        if ~accepted
            [ctxAccepted, yCtx, Rctx, lambdaCtx, selectedCtx] = localBuildContextAnchorCandidate( ...
                measurements(:), zCtxPred, SCtxGate, RcGate, RcUpdateBase, HCtx, PCtxPred, ...
                T1ch, T2, recoveryCtxRawOkMinChannels);
            if ctxAccepted
                recoveryActive = true;
                recoveryHoldOffCounter = recoveryHoldOffAuxCount;
                recoveryCleanCounter = 0;
                recoveryPending = false;
                reanchorCount = reanchorCount + 1;
                qBoostCounter = max(qBoostCounter, recoveryQBoostHorizon);
                xPred = xCtxPred;
                PPred = PCtxPred;
                zPred = zCtxPred;
                H = HCtx;
                lambda_2 = lambdaCtx;
                yUpdate = yCtx;
                RUpdate = Rctx;
                mode_flag = 1;
                debugInfo.acceptedCandidate = 3;
                debugInfo.bestCandidate = 3;
                debugInfo.recoveryTriggered = true;
                debugInfo.recoveryActive = true;
                debugInfo.reanchorCount = double(reanchorCount);
                debugInfo.selectedSource = selectedCtx;
                debugInfo.fallbackValidCount = sum(selectedCtx == 0);
                debugInfo.rawScore = ctxRawScore;
            else
                [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
                    localChannelFallback(measurements(:), bestY, zPred, S1, H * PPred * H' + bestRtest, RcUpdateBase, bestRupdate, T1ch, T2ch);
                lambdaHybrid = localMahalanobis(yUpdate - zPred, H * PPred * H' + RUpdate);
                lambda_2 = lambdaHybrid;
                debugInfo.fallbackValidCount = validCount;
                debugInfo.rawScore = rawScore;
                debugInfo.auxScore = auxScore;
                debugInfo.selectedSource = selectedSource;
                debugInfo.dominantFallbackChannel = dominantFallbackChannel;
                debugInfo.lambdaHybrid = lambdaHybrid;
                if validCount >= 3 && lambdaHybrid <= T2
                    mode_flag = bestMode;
                    debugInfo.acceptedCandidate = max(bestMode, 1);
                else
                    mode_flag = 3;
                end
                if validCount < 3
                    skipUpdate = true;
                    qBoostCounter = qBoostHorizon;
                    yUpdate = measurements(:);
                    RUpdate = RcUpdateBase;
                    lambda_2 = bestLambda;
                end
            end
        end
    end
end

if skipUpdate
    xHat = xPred;
    P = PPred;
    debugInfo.skipUpdate = true;
else
    SUpdate = H * PPred * H' + RUpdate;
    K = (PPred * H') / SUpdate;
    xHat = xPred + K * (yUpdate - zPred);
    I = eye(4);
    P = (I - K * H) * PPred * (I - K * H)' + K * RUpdate * K';
end

if qBoostCounter > 0
    qBoostCounter = qBoostCounter - 1;
end

if mode_flag == 3
    recoveryFallbackStreak = recoveryFallbackStreak + 1;
else
    recoveryFallbackStreak = 0;
end

if lambda_1 > recoveryLambdaEnter
    recoveryHighLambdaStreak = recoveryHighLambdaStreak + 1;
else
    recoveryHighLambdaStreak = 0;
end

if recoveryEnabled && ~recoveryActive && recoveryFallbackStreak >= recoveryFallbackEnterCount
    recoveryPending = true;
elseif mode_flag ~= 3
    recoveryPending = false;
end

if recoveryActive
    if recoveryHoldOffCounter > 0
        recoveryHoldOffCounter = recoveryHoldOffCounter - 1;
    end
    if lambda_1 <= T1 && mode_flag ~= 3
        recoveryCleanCounter = recoveryCleanCounter + 1;
    else
        recoveryCleanCounter = 0;
    end
    if recoveryHoldOffCounter <= 0 && recoveryCleanCounter >= recoveryExitCleanCount
        recoveryActive = false;
        recoveryCleanCounter = 0;
        recoveryPending = false;
    end
end

[metaPrevMode, metaPrevLambda1, metaFallbackStreak, metaTimeSinceAnomalyEnd, metaAnomalyProxyActive] = ...
    localUpdateMetaRuntimeState(metaPrevMode, metaPrevLambda1, metaFallbackStreak, ...
    metaTimeSinceAnomalyEnd, metaAnomalyProxyActive, mode_flag, lambda_1, T1, dt);

[VT, alpha, beta] = reconstruct_airdata_from_state_wind(state, xHat(1:2));
Vw_est = xHat(1:2);
air_est = [VT; alpha; beta];
debugInfo.substActive = logical(substActive);
debugInfo.recoveryActive = logical(recoveryActive);
debugInfo.reanchorCount = double(reanchorCount);
debugInfo.usedGateCalibration = runtimeInfo.gateCalibrationUsed;
debugInfo.runtime = runtimeInfo;
end

function tf = localUseLegacy4StateFusion()
tf = false;
try
    if evalin('base', "exist('stage2UseLegacy4StateFusionOverride','var')")
        tf = logical(evalin('base', 'stage2UseLegacy4StateFusionOverride'));
    end
catch
    tf = false;
end
end

function [Vw_est, air_est, mode_flag, lambda_1, lambda_2, debugInfo] = ...
    localTwoStateFusionStep(state, controls, measurements, dt, reset, debugInfo)
persistent xHat2 P2 xGate2 PGate2 initialized2
persistent xCtx4 PCtx4 qAccelCtx4
persistent RcGate2 RcUpdate2 T1_2 T2_2 T1ch_2 T2ch_2
persistent net2 normStats2 ctxLen2 baseBuf2
persistent runtimeInfo2 runtimeModelMode2 enableSubst2 usePredictedVariance2 gammaD2 gammaS2
persistent substEnterProb2 substExitProb2 substEnterCount2 substExitCount2
persistent substActive2 substHighCounter2 substLowCounter2
persistent oracleAux2 oracleEnabled2 oracleIdx2

if isempty(initialized2)
    initialized2 = false;
end

if ~initialized2 || reset ~= 0
    xHat2 = [0; 0];
    P2 = diag([10, 10]);
    xGate2 = [0; 0];
    PGate2 = diag([10, 10]);
    xCtx4 = zeros(4, 1);
    PCtx4 = diag([10, 10, 2, 2]);
    qAccelCtx4 = [0.20, 0.20] .^ 2;

    modelFile = resolve_stage2_model_file('lstm_akf_net.mat');
    bundle = load_stage2_runtime_bundle(modelFile);
    net2 = bundle.net;
    normStats2 = bundle.normStats;
    runtimeInfo2 = bundle.runtime;
    RcGate2 = runtimeInfo2.Rc;
    RcUpdate2 = diag(runtimeInfo2.baseRcDiag);
    thresholds = runtimeInfo2.gateThresholds;
    runtimeInfo2.gateThresholds = thresholds;
    T1_2 = thresholds.T1;
    T2_2 = thresholds.T2;
    T1ch_2 = thresholds.T1ch;
    T2ch_2 = thresholds.T2ch;
    runtimeModelMode2 = char(runtimeInfo2.runtimeModelMode);
    enableSubst2 = logical(runtimeInfo2.enableSubst);
    usePredictedVariance2 = logical(runtimeInfo2.usePredictedVariance);
    gammaD2 = runtimeInfo2.gammaD;
    gammaS2 = runtimeInfo2.gammaS;
    substEnterProb2 = runtimeInfo2.substEnterProb;
    substExitProb2 = runtimeInfo2.substExitProb;
    substEnterCount2 = runtimeInfo2.substEnterCount;
    substExitCount2 = runtimeInfo2.substExitCount;
    substActive2 = false;
    substHighCounter2 = 0;
    substLowCounter2 = 0;
    ctxLen2 = runtimeInfo2.ctxLen;
    baseBuf2 = zeros(double(normStats2.baseInputDim), ctxLen2, 'single');
    [oracleEnabled2, oracleAux2] = localLoadOracleAuxMeasurements();
    oracleIdx2 = 1;
    initialized2 = true;
end

dt = localSanitizeDt(dt);
state = reshape(state, [12, 1]);
controls = reshape(controls, [4, 1]);
measurements = reshape(measurements, [6, 1]);

Q2 = max(dt / 0.05, 1e-3) * diag([0.15, 0.15] .^ 2);
xPred2 = xHat2;
PPred2 = P2 + Q2;
zPred = localModelObs(state, controls, xPred2);
H = compute_jacobian_numerical(xPred2, state, controls, zPred);
SFinalGate = H * PPred2 * H' + RcGate2;

xGatePred = xGate2;
PGatePred = PGate2 + Q2;
zGatePred = localModelObs(state, controls, xGatePred);
HGate = compute_jacobian_numerical(xGatePred, state, controls, zGatePred);
e1 = measurements - zGatePred;
S1 = HGate * PGatePred * HGate' + RcGate2;
lambda_1 = localMahalanobis(e1, S1);
lambda_2 = 0;

KGate = (PGatePred * HGate') / (HGate * PGatePred * HGate' + RcUpdate2);
xGate2 = xGatePred + KGate * (measurements - zGatePred);
Igate = eye(2);
PGate2 = (Igate - KGate * HGate) * PGatePred * (Igate - KGate * HGate)' + KGate * RcUpdate2 * KGate';

rawScore = (e1 .^ 2) ./ max(diag(S1), 1e-9);
debugInfo.rawScore = rawScore;
debugInfo.ctxRawScore = rawScore;
debugInfo.ctxRawOkCount = sum(rawScore <= T1ch_2);
debugInfo.fallbackValidCount = sum(rawScore <= T1ch_2);

[Fctx, Qctx] = localCVModel(dt, qAccelCtx4, false);
xCtxPred = Fctx * xCtx4;
PCtxPred = Fctx * PCtx4 * Fctx' + Qctx;
zCtxPred = localModelObs(state, controls, xCtxPred(1:2));
H2Ctx = compute_jacobian_numerical(xCtxPred(1:2), state, controls, zCtxPred);
HCtx = [H2Ctx, zeros(6, 2)];
SCtx = HCtx * PCtxPred * HCtx' + RcUpdate2;
KCtx = (PCtxPred * HCtx') / SCtx;
xCtx4 = xCtxPred + KCtx * (measurements - zCtxPred);
Ictx = eye(4);
PCtx4 = (Ictx - KCtx * HCtx) * PCtxPred * (Ictx - KCtx * HCtx)' + KCtx * RcUpdate2 * KCtx';

baseFeat = compose_lstm_fusion_input(xCtxPred, measurements, state, controls, zCtxPred);
baseBuf2 = localPushBuffer(baseBuf2, single(baseFeat));
netSeq = append_causal_multiscale_features(baseBuf2(:, any(baseBuf2, 1)), double(normStats2.multiscaleWindows(:)'));
netSeqN = (netSeq - normStats2.muIn) ./ normStats2.sigIn;

yUpdate = measurements;
RUpdate = RcUpdate2;
mode_flag = 0;

needAux = lambda_1 > T1_2 || strcmpi(runtimeModelMode2, 'denoise_always') || ...
    strcmpi(runtimeModelMode2, 'channel_blend_always');

if needAux
    [muDenoise, muSubst, varAux, pAbn, pSubst] = localPredictAux(net2, netSeqN, normStats2, measurements);
    if oracleEnabled2 && ~isempty(oracleAux2)
        oracleCol = min(oracleIdx2, size(oracleAux2, 2));
        muDenoise = oracleAux2(:, oracleCol);
        muSubst = oracleAux2(:, oracleCol);
        varAux = zeros(6, 1);
        pAbn = 1;
        pSubst = 1;
    end
    debugInfo.muD = muDenoise;
    debugInfo.muS = muSubst;
    debugInfo.pAbn = double(pAbn);
    debugInfo.pSubst = double(pSubst);

    [useSubstPrimary, substActive2, substHighCounter2, substLowCounter2] = ...
        localUpdateSubstState(pSubst, substActive2, substHighCounter2, substLowCounter2, ...
        substEnterProb2, substExitProb2, substEnterCount2, substExitCount2, ...
        strcmpi(runtimeModelMode2, 'dual_head') && enableSubst2);
    debugInfo.useSubstPrimary = logical(useSubstPrimary);
    debugInfo.substActive = logical(substActive2);

    if strcmpi(runtimeModelMode2, 'denoise_always')
        [yCand, Rtest, RupdateCand] = localTwoStateCandidate( ...
            1, muDenoise, muSubst, varAux, RcGate2, RcUpdate2, ...
            gammaD2, gammaS2, usePredictedVariance2, normStats2, measurements, oracleEnabled2);
        yUpdate = yCand;
        RUpdate = RupdateCand;
        lambda_2 = localMahalanobis(yCand - zPred, H * PPred2 * H' + Rtest);
        mode_flag = 1;
        debugInfo.preferredCandidate = 1;
        debugInfo.acceptedCandidate = 1;
        debugInfo.bestCandidate = 1;
    elseif strcmpi(runtimeModelMode2, 'channel_blend_always')
        [yCand, Rtest, RupdateCand] = localTwoStateCandidate( ...
            1, muDenoise, muSubst, varAux, RcGate2, RcUpdate2, ...
            gammaD2, gammaS2, usePredictedVariance2, normStats2, measurements, oracleEnabled2);
        lambda_2 = localMahalanobis(yCand - zPred, H * PPred2 * H' + Rtest);
        [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
            localChannelFallback(measurements, yCand, zPred, SFinalGate, H * PPred2 * H' + Rtest, ...
            RcUpdate2, RupdateCand, T1ch_2, T2ch_2);
        mode_flag = localInferBlendMode(yUpdate, measurements, yCand);
        if validCount < 3
            mode_flag = 3;
        end
        debugInfo.preferredCandidate = 1;
        debugInfo.acceptedCandidate = 1;
        debugInfo.bestCandidate = 1;
        debugInfo.fallbackValidCount = validCount;
        debugInfo.rawScore = rawScore;
        debugInfo.auxScore = auxScore;
        debugInfo.selectedSource = selectedSource;
        debugInfo.dominantFallbackChannel = dominantFallbackChannel;
    elseif lambda_1 > T1_2
        if enableSubst2 && useSubstPrimary
            candidateOrder = [2, 1];
        elseif enableSubst2
            candidateOrder = [1, 2];
        else
            candidateOrder = 1;
        end

        accepted = false;
        bestLambda = inf;
        bestY = measurements;
        bestRtest = RcGate2;
        bestRupdate = RcUpdate2;
        bestMode = 0;
        debugInfo.preferredCandidate = candidateOrder(1);

        for candidate = candidateOrder
            [yCand, Rtest, RupdateCand, modeCandidate] = localTwoStateCandidate( ...
                candidate, muDenoise, muSubst, varAux, RcGate2, RcUpdate2, ...
                gammaD2, gammaS2, usePredictedVariance2, normStats2, measurements, oracleEnabled2);
            lambdaCand = localMahalanobis(yCand - zPred, H * PPred2 * H' + Rtest);
            if candidate == 2
                debugInfo.lambdaCandMuS = lambdaCand;
            else
                debugInfo.lambdaCandMuR = lambdaCand;
            end
            if lambdaCand < bestLambda
                bestLambda = lambdaCand;
                bestY = yCand;
                bestRtest = Rtest;
                bestRupdate = RupdateCand;
                bestMode = modeCandidate;
            end
            if lambdaCand <= T2_2
                accepted = true;
                [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
                    localChannelFallback(measurements, yCand, zPred, SFinalGate, H * PPred2 * H' + Rtest, ...
                    RcUpdate2, RupdateCand, T1ch_2, T2ch_2);
                lambda_2 = localMahalanobis(yUpdate - zPred, H * PPred2 * H' + RUpdate);
                if validCount >= 3
                    mode_flag = modeCandidate;
                else
                    mode_flag = 3;
                end
                debugInfo.acceptedCandidate = candidate;
                debugInfo.fallbackValidCount = validCount;
                debugInfo.rawScore = rawScore;
                debugInfo.auxScore = auxScore;
                debugInfo.selectedSource = selectedSource;
                debugInfo.dominantFallbackChannel = dominantFallbackChannel;
                break;
            end
        end

        debugInfo.bestCandidate = bestMode;
        if ~accepted
            [yUpdate, RUpdate, validCount, rawScore, auxScore, selectedSource, dominantFallbackChannel] = ...
                localChannelFallback(measurements, bestY, zPred, SFinalGate, H * PPred2 * H' + bestRtest, ...
                RcUpdate2, bestRupdate, T1ch_2, T2ch_2);
            lambda_2 = localMahalanobis(yUpdate - zPred, H * PPred2 * H' + RUpdate);
            mode_flag = 3;
            debugInfo.fallbackValidCount = validCount;
            debugInfo.rawScore = rawScore;
            debugInfo.auxScore = auxScore;
            debugInfo.selectedSource = selectedSource;
            debugInfo.dominantFallbackChannel = dominantFallbackChannel;
        end
    end
end

SUpdate = H * PPred2 * H' + RUpdate;
K = (PPred2 * H') / SUpdate;
xHat2 = xPred2 + K * (yUpdate - zPred);
I = eye(2);
P2 = (I - K * H) * PPred2 * (I - K * H)' + K * RUpdate * K';

if oracleEnabled2
    oracleIdx2 = oracleIdx2 + 1;
end

[VT, alpha, beta] = reconstruct_airdata_from_state_wind(state, xHat2);
Vw_est = xHat2;
air_est = [VT; alpha; beta];
debugInfo.usedGateCalibration = runtimeInfo2.gateCalibrationUsed;
debugInfo.runtime = runtimeInfo2;
end

function [yCand, Rtest, Rupdate, modeCandidate] = localTwoStateCandidate(candidate, muDenoise, muSubst, varAux, RcTestBase, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements, bypassClip)
if bypassClip
    switch candidate
        case 2
            yCand = muSubst(:);
            Rtest = diag(gammaS .* diag(RcTestBase));
            RupdateBase = diag(gammaS .* diag(RcUpdateBase));
            modeCandidate = 2;
        otherwise
            yCand = muDenoise(:);
            Rtest = diag(gammaD .* diag(RcTestBase));
            RupdateBase = diag(gammaD .* diag(RcUpdateBase));
            modeCandidate = 1;
    end
    if usePredictedVariance
        Rupdate = diag(max(diag(RupdateBase), varAux(:)));
    else
        Rupdate = RupdateBase;
    end
    return;
end

[yCand, Rtest, Rupdate, modeCandidate] = localBuildCandidate( ...
    candidate, muDenoise, muSubst, varAux, RcTestBase, RcUpdateBase, ...
    gammaD, gammaS, usePredictedVariance, normStats, measurements);
end

function [enabled, oracleAux] = localLoadOracleAuxMeasurements()
enabled = false;
oracleAux = [];
try
    if evalin('base', "exist('stage2OracleAuxMeasurementsOverride','var')")
        oracleAux = double(evalin('base', 'stage2OracleAuxMeasurementsOverride'));
        if size(oracleAux, 1) == 6 && size(oracleAux, 2) >= 1
            enabled = true;
        else
            oracleAux = [];
        end
    end
catch
    enabled = false;
    oracleAux = [];
end
end

function [muDenoise, muSubst, varAux, pAbn, pSubst, logVarNorm] = localPredictAux(net, netSeqN, normStats, measurements)
dlIn = dlarray(reshape(single(netSeqN), [size(netSeqN, 1), size(netSeqN, 2), 1]), 'CTB');
dlOut = predict(net, dlIn);
outData = double(gather(extractdata(dlOut)));
if ndims(outData) >= 3 && size(outData, 3) > 1 && size(outData, 2) == 1
    out = outData(:, 1, end);
elseif ndims(outData) >= 3
    out = outData(:, end, 1);
else
    out = outData(:, end);
end
[muDenoise, muSubst, varAux, pAbn, pSubst, logVarNorm] = decode_lstm_aux_output(out, normStats, 'real', measurements);
end

function tf = localMetaModelsAvailable(svrBundle, gateBundle)
tf = isstruct(svrBundle) && isfield(svrBundle, 'available') && logical(svrBundle.available) && ...
    isstruct(gateBundle) && isfield(gateBundle, 'available') && logical(gateBundle.available);
end

function muResidual = localBuildResidualExpertMeasurement(baseFeat, lambda1, muDenoise, muSubst, pAbn, pSubst, logVarNorm, measurements, svrBundle, normStats, featureCfg, metaState)
metaFeat = compose_stage2_meta_features(baseFeat, lambda1, pAbn, pSubst, logVarNorm, muDenoise, muSubst, measurements, featureCfg, metaState);
rHat = localPredictSvrRuntime(svrBundle, metaFeat);
muResidual = localClipAuxMeasurement(muDenoise + rHat, measurements, localResolveDeltaCap(normStats));
end

function [yMix, RtestMix, RupdateMix, lambdaMix, modeCandidate, debugMix] = localBuildMoECandidate(baseFeat, lambda1, muDenoise, muSubst, pAbn, pSubst, logVarNorm, measurements, RcGate, RcUpdateBase, gammaD, gammaS, zPred, H, PPred, svrBundle, gateBundle, normStats, T1, featureCfg, metaState)
metaFeat = compose_stage2_meta_features(baseFeat, lambda1, pAbn, pSubst, logVarNorm, muDenoise, muSubst, measurements, featureCfg, metaState);
rHat = localPredictSvrRuntime(svrBundle, metaFeat);
muR = muDenoise + rHat;
gateWeights = localPredictGateRuntime(gateBundle, metaFeat);

deltaCap = localResolveDeltaCap(normStats);
muR = localClipAuxMeasurement(muR, measurements, deltaCap);
muSubst = localClipAuxMeasurement(muSubst, measurements, deltaCap);
muDenoise = localClipAuxMeasurement(muDenoise, measurements, deltaCap);

severity = min(max((double(lambda1) - T1) / max(0.5 * T1, 1e-6), 0), 1);
gateWeights(2:3, :) = severity * gateWeights(2:3, :);
gateWeights(1, :) = 1 - sum(gateWeights(2:3, :), 1);
gateWeights = max(gateWeights, 0);
gateWeights = gateWeights ./ max(sum(gateWeights, 1), 1e-9);

yMix = gateWeights(1, :)' .* measurements(:) + ...
    gateWeights(2, :)' .* muR(:) + ...
    gateWeights(3, :)' .* muSubst(:);

RDTest = diag(gammaD .* diag(RcGate));
RSTest = diag(gammaS .* diag(RcGate));
RDUpdate = diag(gammaD .* diag(RcUpdateBase));
RSUpdate = diag(gammaS .* diag(RcUpdateBase));
RtestDiag = zeros(6, 1);
RupdateDiag = zeros(6, 1);
for ii = 1:6
    wRaw = gateWeights(1, ii);
    wR = gateWeights(2, ii);
    wS = gateWeights(3, ii);
    RtestDiag(ii) = wRaw^2 * RcGate(ii, ii) + ...
        wR^2 * RDTest(ii, ii) + ...
        wS^2 * RSTest(ii, ii) + ...
        0.05 * RcGate(ii, ii);
    RupdateDiag(ii) = wRaw^2 * RcUpdateBase(ii, ii) + ...
        wR^2 * RDUpdate(ii, ii) + ...
        wS^2 * RSUpdate(ii, ii) + ...
        0.05 * RcUpdateBase(ii, ii);
end
RtestMix = diag(max(RtestDiag, 1e-9));
RupdateMix = diag(max(RupdateDiag, 1e-9));
lambdaMix = localMahalanobis(yMix - zPred, H * PPred * H' + RtestMix);

if mean(gateWeights(3, :)) > mean(gateWeights(2, :))
    modeCandidate = 2;
else
    modeCandidate = 1;
end

debugMix = struct('muR', muR, 'gateWeights', gateWeights);
end

function rHat = localPredictSvrRuntime(svrBundle, metaFeat)
rHat = zeros(6, 1);
for ch = 1:6
    if isfield(svrBundle, 'enabledMask') && numel(svrBundle.enabledMask) == 6 && ~logical(svrBundle.enabledMask(ch))
        continue;
    end
    if isempty(svrBundle.channels{ch})
        continue;
    end
    rHat(ch) = predict(svrBundle.channels{ch}, metaFeat(:)');
end
end

function gateWeights = localPredictGateRuntime(gateBundle, metaFeat)
Xn = (double(metaFeat(:)) - double(gateBundle.muX(:))) ./ double(gateBundle.sigX(:));
gateWeights = zeros(3, 6);
for ch = 1:6
    score = double(gateBundle.W(:, :, ch)) * Xn + double(gateBundle.b(:, ch));
    gateWeights(:, ch) = localSoftmax(score);
end
end

function prob = localSoftmax(score)
score = score - max(score);
prob = exp(score);
prob = prob / max(sum(prob), 1e-9);
end

function deltaCap = localResolveDeltaCap(normStats)
if isstruct(normStats) && isfield(normStats, 'sigDelta')
    deltaCap = 3 * double(normStats.sigDelta(:));
else
    deltaCap = inf(6, 1);
end
end

function yClip = localClipAuxMeasurement(yAux, measurements, deltaCap)
deltaAux = yAux(:) - measurements(:);
deltaAux = min(max(deltaAux, -deltaCap(:)), deltaCap(:));
yClip = measurements(:) + deltaAux;
end

function [yUpdate, RUpdate, selectedSource, dominantFallbackChannel] = localBuildRecoveryRawUpdate(measurements, zCtxPred, SCtx, RcUpdateBase, T1ch)
yUpdate = measurements(:);
RUpdate = RcUpdateBase;
selectedSource = zeros(6, 1);
ctxRawScore = ((measurements(:) - zCtxPred(:)) .^ 2) ./ max(diag(SCtx), 1e-9);
for i = 1:6
    if ctxRawScore(i) <= T1ch
        RUpdate(i, i) = RcUpdateBase(i, i);
        selectedSource(i) = 0;
    else
        RUpdate(i, i) = 10 * RcUpdateBase(i, i);
        selectedSource(i) = 2;
    end
end

[~, dominantFallbackChannel] = max(ctxRawScore ./ max(T1ch, 1e-9));
if ~any(selectedSource == 2)
    dominantFallbackChannel = 0;
end
end

function [accepted, yCtx, Rctx, lambdaCtx, selectedSource] = localBuildContextAnchorCandidate(measurements, zCtxPred, SCtx, RcGate, RcUpdateBase, HCtx, PCtxPred, T1ch, T2, minOkChannels)
ctxRawScore = ((measurements(:) - zCtxPred(:)) .^ 2) ./ max(diag(SCtx), 1e-9);
rawKeepMask = true(6, 1);
rawKeepMask(ctxRawScore > 0.75 * T1ch) = false;
[yCtx, Rctx, RtestCtx, selectedSource, lambdaCtx] = localAssembleContextAnchorCandidate( ...
    measurements, zCtxPred, rawKeepMask, RcGate, RcUpdateBase, HCtx, PCtxPred);

while lambdaCtx > T2 && sum(rawKeepMask) > minOkChannels
    scoreMasked = ctxRawScore;
    scoreMasked(~rawKeepMask) = -inf;
    [~, worstIdx] = max(scoreMasked);
    rawKeepMask(worstIdx) = false;
    [yCtx, Rctx, RtestCtx, selectedSource, lambdaCtx] = localAssembleContextAnchorCandidate( ...
        measurements, zCtxPred, rawKeepMask, RcGate, RcUpdateBase, HCtx, PCtxPred);
end

accepted = (sum(rawKeepMask) >= minOkChannels) && any(selectedSource == 3) && (lambdaCtx <= T2);
end

function [yCtx, Rctx, RtestCtx, selectedSource, lambdaCtx] = localAssembleContextAnchorCandidate(measurements, zCtxPred, rawKeepMask, RcGate, RcUpdateBase, HCtx, PCtxPred)
yCtx = measurements(:);
RtestCtx = RcGate;
Rctx = RcUpdateBase;
selectedSource = zeros(6, 1);
for i = 1:6
    if ~rawKeepMask(i)
        yCtx(i) = zCtxPred(i);
        RtestCtx(i, i) = 12 * RcGate(i, i);
        Rctx(i, i) = 4 * RcUpdateBase(i, i);
        selectedSource(i) = 3;
    end
end
lambdaCtx = localMahalanobis(yCtx - zCtxPred, HCtx * PCtxPred * HCtx' + RtestCtx);
end

function [prevMode, prevLambda1, fallbackStreak, timeSinceLastAnomalyEnd, anomalyProxyActive] = ...
    localUpdateMetaRuntimeState(prevMode, prevLambda1, fallbackStreak, timeSinceLastAnomalyEnd, anomalyProxyActive, modeFlag, lambda1, T1, dt)
if modeFlag == 3
    fallbackStreak = fallbackStreak + 1;
else
    fallbackStreak = 0;
end

anomalyProxyNow = (lambda1 > T1) || (modeFlag == 3);
if anomalyProxyNow
    anomalyProxyActive = true;
    timeSinceLastAnomalyEnd = 0;
else
    if anomalyProxyActive
        anomalyProxyActive = false;
        timeSinceLastAnomalyEnd = 0;
    else
        timeSinceLastAnomalyEnd = timeSinceLastAnomalyEnd + dt;
    end
end

prevMode = modeFlag;
prevLambda1 = lambda1;
end

function [yCand, Rtest, Rupdate, modeCandidate] = localBuildCandidate(candidate, muDenoise, muSubst, varAux, RcTestBase, RcUpdateBase, gammaD, gammaS, usePredictedVariance, normStats, measurements)
switch candidate
    case 2
        yCand = muSubst;
        Rtest = diag(gammaS .* diag(RcTestBase));
        RupdateBase = diag(gammaS .* diag(RcUpdateBase));
        modeCandidate = 2;
    otherwise
        yCand = muDenoise;
        Rtest = diag(gammaD .* diag(RcTestBase));
        RupdateBase = diag(gammaD .* diag(RcUpdateBase));
        modeCandidate = 1;
end

deltaAux = yCand - measurements(:);
if isfield(normStats, 'sigDelta')
    deltaCap = 3 * double(normStats.sigDelta(:));
    deltaAux = min(max(deltaAux, -deltaCap), deltaCap);
    yCand = measurements(:) + deltaAux;
end

if usePredictedVariance
    Rupdate = diag(max(diag(RupdateBase), varAux(:)));
else
    Rupdate = RupdateBase;
end
end

function [useSubstNow, substActive, highCounter, lowCounter] = localUpdateSubstState(pSubst, substActive, highCounter, lowCounter, enterProb, exitProb, enterCount, exitCount, enableSubst)
if ~enableSubst
    useSubstNow = false;
    substActive = false;
    highCounter = 0;
    lowCounter = 0;
    return;
end

if pSubst >= enterProb
    highCounter = highCounter + 1;
    lowCounter = 0;
elseif pSubst < exitProb
    lowCounter = lowCounter + 1;
    highCounter = 0;
else
    highCounter = 0;
    lowCounter = 0;
end

if ~substActive && highCounter >= enterCount
    substActive = true;
    highCounter = 0;
end
if substActive && lowCounter >= exitCount
    substActive = false;
    lowCounter = 0;
end

useSubstNow = substActive || (pSubst >= enterProb && enterCount <= 1);
end

function [F, Q] = localCVModel(dt, qAccelBase, boosted)
boost = 1.0;
if boosted
    boost = 3.0;
end

qN = boost * qAccelBase(1);
qE = boost * qAccelBase(2);

F = [1, 0, dt, 0; ...
     0, 1, 0, dt; ...
     0, 0, 1,  0; ...
     0, 0, 0,  1];

Qn = qN * [dt^4 / 4, dt^3 / 2; dt^3 / 2, dt^2];
Qe = qE * [dt^4 / 4, dt^3 / 2; dt^3 / 2, dt^2];

Q = [Qn(1,1), 0,       Qn(1,2), 0; ...
     0,       Qe(1,1), 0,       Qe(1,2); ...
     Qn(2,1), 0,       Qn(2,2), 0; ...
     0,       Qe(2,1), 0,       Qe(2,2)];
end

function dt = localSanitizeDt(dt)
dt = double(dt);
if ~(isfinite(dt) && dt > 0)
    dt = 0.02;
end
end

function z = localModelObs(state, controls, vw)
[ud, vd, wd, pd, qd, rd] = wind_estimation_model_numeric(state, controls, vw);
z = [ud; vd; wd; pd; qd; rd];
end

function lambda = localMahalanobis(e, S)
S = 0.5 * (S + S');
S = S + 1e-9 * eye(size(S));
lambda = real(e' * (S \ e));
end

function buf = localPushBuffer(buf, feat)
used = find(any(buf, 1), 1, 'last');
if isempty(used)
    used = 0;
end

if used < size(buf, 2)
    buf(:, used + 1) = feat;
else
    buf(:, 1:end-1) = buf(:, 2:end);
    buf(:, end) = feat;
end
end

function [ySel, RSel, validCount, rawScoreVec, auxScoreVec, selectedSource, dominantFallbackChannel] = localChannelFallback(yRaw, yAux, zPred, S1, S2, Rc, Raux, T1ch, T2ch)
ySel = yRaw;
RSel = 10 * Rc;
validCount = 0;
rawScoreVec = zeros(6, 1);
auxScoreVec = zeros(6, 1);
selectedSource = zeros(6, 1);

for i = 1:6
    rawScore = (yRaw(i) - zPred(i))^2 / max(S1(i, i), 1e-9);
    auxScore = (yAux(i) - zPred(i))^2 / max(S2(i, i), 1e-9);
    rawScoreVec(i) = rawScore;
    auxScoreVec(i) = auxScore;
    rawOk = rawScore <= T1ch;
    auxOk = auxScore <= T2ch;

    if rawOk && auxOk
        if auxScore <= rawScore
            ySel(i) = yAux(i);
            RSel(i, i) = Raux(i, i);
            selectedSource(i) = 1;
        else
            ySel(i) = yRaw(i);
            RSel(i, i) = Rc(i, i);
            selectedSource(i) = 0;
        end
        validCount = validCount + 1;
    elseif auxOk
        ySel(i) = yAux(i);
        RSel(i, i) = Raux(i, i);
        selectedSource(i) = 1;
        validCount = validCount + 1;
    elseif rawOk
        ySel(i) = yRaw(i);
        RSel(i, i) = Rc(i, i);
        selectedSource(i) = 0;
        validCount = validCount + 1;
    else
        ySel(i) = yRaw(i);
        RSel(i, i) = 10 * Rc(i, i);
        selectedSource(i) = 2;
    end
end
dominantFallbackChannel = localResolveDominantFallbackChannel(rawScoreVec, auxScoreVec, selectedSource, T1ch, T2ch);
end

function idx = localResolveDominantFallbackChannel(rawScoreVec, auxScoreVec, selectedSource, T1ch, T2ch)
idx = 0;
severity = -inf(size(rawScoreVec));
bothReject = selectedSource == 2;
if any(bothReject)
    severity(bothReject) = max(rawScoreVec(bothReject) / max(T1ch, 1e-9), ...
        auxScoreVec(bothReject) / max(T2ch, 1e-9));
else
    rawChosen = selectedSource == 0;
    if any(rawChosen)
        severity(rawChosen) = auxScoreVec(rawChosen) / max(T2ch, 1e-9);
    else
        severity = auxScoreVec / max(T2ch, 1e-9);
    end
end

[bestScore, idxBest] = max(severity);
if isfinite(bestScore) && bestScore > 0
    idx = idxBest;
end
end

function modeFlag = localInferBlendMode(ySel, yRaw, yAux)
tol = 1e-9;
rawCount = sum(abs(ySel - yRaw) <= tol);
auxCount = sum(abs(ySel - yAux) <= tol);
if rawCount == numel(ySel)
    modeFlag = 0;
elseif auxCount == numel(ySel)
    modeFlag = 1;
else
    modeFlag = 3;
end
end
