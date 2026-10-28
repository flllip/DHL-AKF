function replay = run_stage2_replay(stateMat, controlMat, measMat, dt, collectDebug, options)
% Run EKF and fusion replay on sampled signals using the shared runtime path.

if nargin < 5
    collectDebug = true;
end
if nargin < 6
    options = struct();
end

clear ekf_baseline_step lstm_ekf_fusion_step lstm_always_step air_direct_lstm_step predict_sads_air_direct_sequence
oracleState = localPushOracleAuxOverride(options);
cleanupObj = onCleanup(@() localRestoreOracleAuxOverride(oracleState));
airDirectState = localPushAirDirectOverride(options);
cleanupAirObj = onCleanup(@() localRestoreAirDirectOverride(airDirectState));
collectLstmAux = isstruct(options) && isfield(options, 'collectLstmAux') && logical(options.collectLstmAux);
collectAirDirectStandalone = isstruct(options) && isfield(options, 'collectAirDirectStandalone') && logical(options.collectAirDirectStandalone);
sadsMlpModelFile = localOptionChar(options, 'sadsMlpModelFile', 'sads_cascade_mlp_10ms_windenv.mat');
sadsLstmModelFile = localOptionChar(options, 'sadsLstmModelFile', 'sads_seq_lstm_10ms_windenv.mat');
collectSadsMlp = isstruct(options) && isfield(options, 'collectSadsMlp') && logical(options.collectSadsMlp);
collectSadsLstm = isstruct(options) && isfield(options, 'collectSadsLstm') && logical(options.collectSadsLstm);

N = min([size(stateMat, 2), size(controlMat, 2), size(measMat, 2)]);
windEkf = zeros(2, N);
airEkf = zeros(3, N);
lambdaEkf = zeros(1, N);
windFuse = zeros(2, N);
airFuse = zeros(3, N);
mode = zeros(1, N);
lambda1 = zeros(1, N);
lambda2 = zeros(1, N);
windLstmAux = [];
airLstmAux = [];
airLstmDirectAux = [];
airLstmDirectStandalone = [];
airSadsMlp = [];
airSadsLstm = [];
lambdaLstmAux = [];
yHatLstmAux = [];
if collectLstmAux
    windLstmAux = zeros(2, N);
    airLstmAux = zeros(3, N);
    airLstmDirectAux = zeros(3, N);
    lambdaLstmAux = zeros(1, N);
    yHatLstmAux = zeros(6, N);
end
if collectAirDirectStandalone
    airLstmDirectStandalone = zeros(3, N);
end

debug = struct();
runtimeInfo = struct();
if collectDebug
    debug.preferredCandidate = zeros(1, N);
    debug.acceptedCandidate = zeros(1, N);
    debug.bestCandidate = zeros(1, N);
    debug.lambdaCandMuR = zeros(1, N);
    debug.lambdaCandMuS = zeros(1, N);
    debug.useSubstPrimary = false(1, N);
    debug.substActive = false(1, N);
    debug.pSubst = zeros(1, N);
    debug.skipUpdate = false(1, N);
    debug.fallbackValidCount = zeros(1, N);
    debug.rawScore = zeros(6, N);
    debug.auxScore = zeros(6, N);
    debug.selectedSource = zeros(6, N);
    debug.dominantFallbackChannel = zeros(1, N);
    debug.recoveryActive = false(1, N);
    debug.recoveryTriggered = false(1, N);
    debug.ctxRawScore = zeros(6, N);
    debug.ctxRawOkCount = zeros(1, N);
    debug.reanchorCount = zeros(1, N);
    debug.usedGateCalibration = false(1, N);
    debug.pAbn = zeros(1, N);
    debug.muD = zeros(6, N);
    debug.muR = zeros(6, N);
    debug.muS = zeros(6, N);
    debug.gateWeights = zeros(3, 6, N);
    debug.lambda2Mix = zeros(1, N);
end

for k = 1:N
    reset = double(k == 1);
    [windEkf(:, k), airEkf(:, k), lambdaEkf(k)] = ekf_baseline_step( ...
        stateMat(:, k), controlMat(:, k), measMat(:, k), reset);

    if collectLstmAux
        [windLstmAux(:, k), airLstmAux(:, k), lambdaLstmAux(k), yHatLstmAux(:, k), airLstmDirectAux(:, k)] = ...
            lstm_always_step(stateMat(:, k), controlMat(:, k), measMat(:, k), reset);
    end
    if collectAirDirectStandalone
        airLstmDirectStandalone(:, k) = air_direct_lstm_step( ...
            stateMat(:, k), controlMat(:, k), measMat(:, k), reset);
    end

    if collectDebug
        [windFuse(:, k), airFuse(:, k), mode(k), lambda1(k), lambda2(k), debugStep] = ...
            lstm_ekf_fusion_step(stateMat(:, k), controlMat(:, k), measMat(:, k), dt, reset);
        debug.preferredCandidate(k) = localGetDebugField(debugStep, 'preferredCandidate', 0);
        debug.acceptedCandidate(k) = localGetDebugField(debugStep, 'acceptedCandidate', 0);
        debug.bestCandidate(k) = localGetDebugField(debugStep, 'bestCandidate', 0);
        debug.lambdaCandMuR(k) = localGetDebugField(debugStep, 'lambdaCandMuR', 0);
        debug.lambdaCandMuS(k) = localGetDebugField(debugStep, 'lambdaCandMuS', 0);
        debug.useSubstPrimary(k) = logical(localGetDebugField(debugStep, 'useSubstPrimary', false));
        debug.substActive(k) = logical(localGetDebugField(debugStep, 'substActive', false));
        debug.pSubst(k) = localGetDebugField(debugStep, 'pSubst', 0);
        debug.pAbn(k) = localGetDebugField(debugStep, 'pAbn', 0);
        debug.skipUpdate(k) = logical(localGetDebugField(debugStep, 'skipUpdate', false));
        debug.fallbackValidCount(k) = localGetDebugField(debugStep, 'fallbackValidCount', 0);
        debug.rawScore(:, k) = localGetDebugField(debugStep, 'rawScore', zeros(6, 1));
        debug.auxScore(:, k) = localGetDebugField(debugStep, 'auxScore', zeros(6, 1));
        debug.selectedSource(:, k) = localGetDebugField(debugStep, 'selectedSource', zeros(6, 1));
        debug.dominantFallbackChannel(k) = localGetDebugField(debugStep, 'dominantFallbackChannel', 0);
        debug.recoveryActive(k) = logical(localGetDebugField(debugStep, 'recoveryActive', false));
        debug.recoveryTriggered(k) = logical(localGetDebugField(debugStep, 'recoveryTriggered', false));
        debug.ctxRawScore(:, k) = localGetDebugField(debugStep, 'ctxRawScore', zeros(6, 1));
        debug.ctxRawOkCount(k) = localGetDebugField(debugStep, 'ctxRawOkCount', 0);
        debug.reanchorCount(k) = localGetDebugField(debugStep, 'reanchorCount', 0);
        debug.usedGateCalibration(k) = logical(localGetDebugField(debugStep, 'usedGateCalibration', false));
        debug.muD(:, k) = localGetDebugField(debugStep, 'muD', zeros(6, 1));
        debug.muR(:, k) = localGetDebugField(debugStep, 'muR', zeros(6, 1));
        debug.muS(:, k) = localGetDebugField(debugStep, 'muS', zeros(6, 1));
        debug.gateWeights(:, :, k) = localGetDebugField(debugStep, 'gateWeights', zeros(3, 6));
        debug.lambda2Mix(k) = localGetDebugField(debugStep, 'lambda2Mix', 0);
        if k == 1 && isfield(debugStep, 'runtime')
            runtimeInfo = debugStep.runtime;
        end
    else
        [windFuse(:, k), airFuse(:, k), mode(k), lambda1(k), lambda2(k)] = ...
            lstm_ekf_fusion_step(stateMat(:, k), controlMat(:, k), measMat(:, k), dt, reset);
    end
end

if collectSadsMlp
    if exist(sadsMlpModelFile, 'file') == 2
        airSadsMlp = predict_sads_air_direct_sequence( ...
            sadsMlpModelFile, stateMat(:, 1:N), controlMat(:, 1:N), measMat(:, 1:N));
    else
        warning('SADS MLP model file not found: %s', sadsMlpModelFile);
    end
end

if collectSadsLstm
    if exist(sadsLstmModelFile, 'file') == 2
        airSadsLstm = predict_sads_air_direct_sequence( ...
            sadsLstmModelFile, stateMat(:, 1:N), controlMat(:, 1:N), measMat(:, 1:N));
    else
        warning('SADS LSTM model file not found: %s', sadsLstmModelFile);
    end
end

replay = struct();
replay.N = N;
replay.time = (0:N-1) * dt;
replay.windEkf = windEkf;
replay.airEkf = airEkf;
replay.lambdaEkf = lambdaEkf;
replay.windFuse = windFuse;
replay.airFuse = airFuse;
replay.windLstmAux = windLstmAux;
replay.airLstmAux = airLstmAux;
replay.airLstmDirectAux = airLstmDirectAux;
replay.airLstmDirectStandalone = airLstmDirectStandalone;
replay.airSadsMlp = airSadsMlp;
replay.airSadsLstm = airSadsLstm;
replay.lambdaLstmAux = lambdaLstmAux;
replay.yHatLstmAux = yHatLstmAux;
replay.mode = mode;
replay.lambda1 = lambda1;
replay.lambda2 = lambda2;
replay.debug = debug;
replay.runtime = runtimeInfo;
end

function state = localPushOracleAuxOverride(options)
state = struct();
state.hadOracle = evalin('base', "exist('stage2OracleAuxMeasurementsOverride','var')");
if state.hadOracle
    state.oracle = evalin('base', 'stage2OracleAuxMeasurementsOverride');
end

oracleAux = [];
if isnumeric(options)
    oracleAux = options;
elseif isstruct(options)
    if isfield(options, 'oracleAuxMeasurements')
        oracleAux = options.oracleAuxMeasurements;
    elseif isfield(options, 'oracleCleanAux')
        oracleAux = options.oracleCleanAux;
    end
end

if ~isempty(oracleAux)
    assignin('base', 'stage2OracleAuxMeasurementsOverride', double(oracleAux));
elseif state.hadOracle
    evalin('base', 'clear stage2OracleAuxMeasurementsOverride');
end
end

function localRestoreOracleAuxOverride(state)
if state.hadOracle
    assignin('base', 'stage2OracleAuxMeasurementsOverride', state.oracle);
else
    evalin('base', 'clear stage2OracleAuxMeasurementsOverride');
end
end

function state = localPushAirDirectOverride(options)
state = struct();
state.hadModel = evalin('base', "exist('stage2AirDirectModelFileOverride','var')");
if state.hadModel
    state.modelFile = evalin('base', 'stage2AirDirectModelFileOverride');
end

if isstruct(options) && isfield(options, 'airDirectModelFile') && ~isempty(options.airDirectModelFile)
    assignin('base', 'stage2AirDirectModelFileOverride', char(options.airDirectModelFile));
elseif state.hadModel
    evalin('base', 'clear stage2AirDirectModelFileOverride');
end
end

function localRestoreAirDirectOverride(state)
if isfield(state, 'hadModel') && state.hadModel
    assignin('base', 'stage2AirDirectModelFileOverride', state.modelFile);
else
    evalin('base', 'clear stage2AirDirectModelFileOverride');
end
end

function value = localGetDebugField(S, field, defaultValue)
if isstruct(S) && isfield(S, field)
    value = S.(field);
else
    value = defaultValue;
end
end

function value = localOptionChar(options, field, defaultValue)
if isstruct(options) && isfield(options, field) && ~isempty(options.(field))
    value = char(options.(field));
else
    value = char(defaultValue);
end
end
