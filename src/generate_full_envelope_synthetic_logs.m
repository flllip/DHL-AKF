function dataset = generate_full_envelope_synthetic_logs(cfg)
% Generate synthetic full-flight-envelope logs for stage-2 EKF-LSTM training.

if nargin < 1 || isempty(cfg)
    cfg = localDefaultCfg();
else
    cfg = localMergeStruct(localDefaultCfg(), cfg);
end

rng(cfg.seed);

dataset.train = localBuildEntries("train", cfg.numTrain, cfg);
dataset.val = localBuildEntries("val", cfg.numVal, cfg);
dataset.test = localBuildEntries("test", cfg.numTest, cfg);
dataset.config = cfg;
end

function entries = localBuildEntries(role, count, cfg)
entries = repmat(struct('name', "", 'folder', "", 'datenum', 0, 'logs', struct()), count, 1);

for i = 1:count
    logs = localGenerateSingleFlight(role, i, cfg);
    entries(i).name = sprintf('synthetic_%s_flight_%02d', char(role), i);
    entries(i).folder = pwd;
    entries(i).datenum = now + i / 1e5;
    entries(i).logs = logs;

    if cfg.saveToDisk
        fileName = sprintf('full_envelope_%s_%02d.mat', char(role), i);
        save(fileName, 'logs');
    end
end
end

function logs = localGenerateSingleFlight(role, flightIdx, cfg)
dt = cfg.sampleTime;
N = round(cfg.flightDuration / dt);
t = (0:N - 1) * dt;

[modeId, targets] = localGenerateFlightTargets(cfg, N, dt, flightIdx);

windNav = localGenerateWind(cfg, N, dt, flightIdx, targets);
state = zeros(12, N);
controls = zeros(4, N);
airTrue = zeros(3, N);
pos = zeros(3, 1);
psi = targets.psi0;

for k = 1:N
    phi = targets.phi(k);
    theta = targets.theta(k);
    psi = localWrapToPi(psi + targets.r(k) * dt);

    vt = targets.vt(k);
    alpha = targets.alpha(k);
    beta = targets.beta(k);

    airBody = [vt * cos(alpha) * cos(beta); ...
               vt * sin(beta); ...
               vt * sin(alpha) * cos(beta)];

    [~, ~, ~, ~, vwBody] = reconstruct_airdata_from_state_wind( ...
        [0; 0; targets.alt(k); phi; theta; psi; 0; 0; 0; 0; 0; 0], windNav(:, k));
    bodyVel = airBody + vwBody;

    Cnb = localCnb(phi, theta, psi);
    navVel = Cnb' * bodyVel;
    pos = pos + navVel * dt;
    pos(3) = targets.alt(k);

    state(:, k) = [pos(1); pos(2); pos(3); phi; theta; psi; ...
        bodyVel(1); bodyVel(2); bodyVel(3); targets.p(k); targets.q(k); targets.r(k)];
    controls(:, k) = [targets.thrust(k); targets.de(k); targets.da(k); targets.dr(k)];
    airTrue(:, k) = [vt; alpha; beta];
end

yClean = zeros(6, N);
for k = 1:N
    yClean(:, k) = localModelObs(state(:, k), controls(:, k), windNav(:, k));
end

[measNominal, vwPrior] = localNominalSensorModel(yClean, windNav, cfg, N);
teacherNominal = build_teacher_signals(state, controls, measNominal, dt);
[measEval, anomalyMask, anomalyType, anomalySeverity, substLabel] = ...
    localInjectFlightAnomalies(measNominal, teacherNominal.zPred, cfg.evalAnomaly);

logs = struct();
logs.state = state;
logs.controls = controls;
logs.measurements = measNominal;
logs.measurements_eval = measEval;
logs.Yclean = yClean;
logs.Ytrue = yClean;
logs.air_true = airTrue;
logs.vw_true = windNav;
logs.vw_prior = vwPrior;
logs.teacher_xPred = teacherNominal.xPred;
logs.teacher_zPred = teacherNominal.zPred;
logs.teacher_lambda1 = teacherNominal.lambda1;
logs.anomaly_mask = anomalyMask;
logs.anomaly_type = anomalyType;
logs.anomaly_severity = anomalySeverity;
logs.eval_subst_label = substLabel;
logs.flight_mode = modeId;
logs.datasetRole = string(role);
logs.sampleTime = dt;
logs.sequenceLength = cfg.sequenceLength;
logs.stride = cfg.stride;
logs.flightIdx = flightIdx;
logs.flightTime = t;
end

function [modeId, targets] = localGenerateFlightTargets(cfg, N, dt, flightIdx)
t = (0:N - 1) * dt;
segMin = round(cfg.segmentDurationRange(1) / dt);
segMax = round(cfg.segmentDurationRange(2) / dt);

alt0 = cfg.altitudeRange(1) + rand * diff(cfg.altitudeRange);
vt0 = cfg.vtRange(1) + rand * diff(cfg.vtRange);
alpha0 = deg2rad(-2 + 8 * rand);
beta0 = deg2rad(-2 + 4 * rand);
phi0 = deg2rad(-10 + 20 * rand);
theta0 = deg2rad(-2 + 4 * rand);
psi0 = deg2rad(-20 + 40 * rand);

knotIdx = 1;
knotAlt = alt0;
knotVt = vt0;
knotAlpha = alpha0;
knotBeta = beta0;
knotPhi = phi0;
knotTheta = theta0;
knotP = 0;
knotQ = 0;
knotR = 0;
knotThrust = 7000;
knotDe = -1.0;
knotDa = 0;
knotDr = 0;
knotModes = 1;

curAlt = alt0;
while knotIdx(end) < N
    remain = N - knotIdx(end);
    segLen = min(remain, randi([segMin, segMax]));
    nextIdx = knotIdx(end) + segLen;
    mode = localSelectMode(curAlt, cfg);
    target = localSampleModeTarget(mode, curAlt, cfg, flightIdx);

    knotIdx(end + 1) = nextIdx; %#ok<AGROW>
    knotAlt(end + 1) = target.alt; %#ok<AGROW>
    knotVt(end + 1) = target.vt; %#ok<AGROW>
    knotAlpha(end + 1) = target.alpha; %#ok<AGROW>
    knotBeta(end + 1) = target.beta; %#ok<AGROW>
    knotPhi(end + 1) = target.phi; %#ok<AGROW>
    knotTheta(end + 1) = target.theta; %#ok<AGROW>
    knotP(end + 1) = target.p; %#ok<AGROW>
    knotQ(end + 1) = target.q; %#ok<AGROW>
    knotR(end + 1) = target.r; %#ok<AGROW>
    knotThrust(end + 1) = target.thrust; %#ok<AGROW>
    knotDe(end + 1) = target.de; %#ok<AGROW>
    knotDa(end + 1) = target.da; %#ok<AGROW>
    knotDr(end + 1) = target.dr; %#ok<AGROW>
    knotModes(end + 1) = target.modeId; %#ok<AGROW>
    curAlt = target.alt;
end

if knotIdx(end) ~= N
    knotIdx(end) = N;
end

targets.alt = localSmoothInterp(knotIdx, knotAlt, N);
targets.vt = localSmoothInterp(knotIdx, knotVt, N);
targets.alpha = localSmoothInterp(knotIdx, knotAlpha, N);
targets.beta = localSmoothInterp(knotIdx, knotBeta, N);
targets.phi = localSmoothInterp(knotIdx, knotPhi, N);
targets.theta = localSmoothInterp(knotIdx, knotTheta, N);
targets.p = localSmoothInterp(knotIdx, knotP, N);
targets.q = localSmoothInterp(knotIdx, knotQ, N);
targets.r = localSmoothInterp(knotIdx, knotR, N);
targets.thrust = localSmoothInterp(knotIdx, knotThrust, N);
targets.de = localSmoothInterp(knotIdx, knotDe, N);
targets.da = localSmoothInterp(knotIdx, knotDa, N);
targets.dr = localSmoothInterp(knotIdx, knotDr, N);
targets.psi0 = psi0;

modeId = zeros(1, N);
for i = 1:(numel(knotIdx) - 1)
    idx = knotIdx(i):min(knotIdx(i + 1), N);
    modeId(idx) = knotModes(i + 1);
end

targets.alt = min(max(targets.alt, cfg.altitudeRange(1)), cfg.altitudeRange(2));
targets.vt = min(max(targets.vt, cfg.vtRange(1)), cfg.vtRange(2));
targets.alpha = min(max(targets.alpha, deg2rad(cfg.alphaRange(1))), deg2rad(cfg.alphaRange(2)));
targets.beta = min(max(targets.beta, deg2rad(cfg.betaRange(1))), deg2rad(cfg.betaRange(2)));
targets.phi = min(max(targets.phi, deg2rad(cfg.phiRange(1))), deg2rad(cfg.phiRange(2)));
targets.theta = min(max(targets.theta, deg2rad(cfg.thetaRange(1))), deg2rad(cfg.thetaRange(2)));
targets.de = min(max(targets.de, cfg.controlRanges.de(1)), cfg.controlRanges.de(2));
targets.da = min(max(targets.da, cfg.controlRanges.da(1)), cfg.controlRanges.da(2));
targets.dr = min(max(targets.dr, cfg.controlRanges.dr(1)), cfg.controlRanges.dr(2));
targets.thrust = min(max(targets.thrust, cfg.controlRanges.thrust(1)), cfg.controlRanges.thrust(2));
end

function mode = localSelectMode(curAlt, cfg)
if curAlt < cfg.altitudeRange(1) + 3000
    candidates = ["climb", "recovery", "turn"];
elseif curAlt > cfg.altitudeRange(2) - 3000
    candidates = ["descent", "recovery", "turn"];
else
    candidates = ["cruise", "climb", "descent", "turn", "aggressive", "highalpha", "recovery"];
end
mode = candidates(randi(numel(candidates)));
end

function target = localSampleModeTarget(mode, curAlt, cfg, flightIdx)
target = struct('modeId', 0, 'alt', curAlt, 'vt', 500, 'alpha', 0, 'beta', 0, ...
    'phi', 0, 'theta', 0, 'p', 0, 'q', 0, 'r', 0, 'thrust', 7000, 'de', 0, 'da', 0, 'dr', 0);

switch char(mode)
    case 'cruise'
        target.modeId = 1;
        target.alt = curAlt + (-1200 + 2400 * rand);
        target.vt = 420 + 240 * rand;
        target.alpha = deg2rad(-2 + 9 * rand);
        target.beta = deg2rad(-4 + 8 * rand);
        target.phi = deg2rad(-20 + 40 * rand);
        target.theta = deg2rad(-4 + 9 * rand);
        target.p = deg2rad(-12 + 24 * rand);
        target.q = deg2rad(-8 + 16 * rand);
        target.r = deg2rad(-10 + 20 * rand);
        target.thrust = 5000 + 5000 * rand;
        target.de = -4 + 6 * rand;
        target.da = -5 + 10 * rand;
        target.dr = -4 + 8 * rand;

    case 'climb'
        target.modeId = 2;
        target.alt = curAlt + (2500 + 5000 * rand);
        target.vt = 360 + 180 * rand;
        target.alpha = deg2rad(4 + 10 * rand);
        target.beta = deg2rad(-4 + 8 * rand);
        target.phi = deg2rad(-25 + 50 * rand);
        target.theta = deg2rad(4 + 14 * rand);
        target.p = deg2rad(-18 + 36 * rand);
        target.q = deg2rad(4 + 18 * rand);
        target.r = deg2rad(-12 + 24 * rand);
        target.thrust = 10000 + 7000 * rand;
        target.de = -10 + 6 * rand;
        target.da = -7 + 14 * rand;
        target.dr = -5 + 10 * rand;

    case 'descent'
        target.modeId = 3;
        target.alt = curAlt - (2500 + 5000 * rand);
        target.vt = 480 + 260 * rand;
        target.alpha = deg2rad(-4 + 8 * rand);
        target.beta = deg2rad(-5 + 10 * rand);
        target.phi = deg2rad(-30 + 60 * rand);
        target.theta = deg2rad(-18 + 10 * rand);
        target.p = deg2rad(-20 + 40 * rand);
        target.q = deg2rad(-16 + 10 * rand);
        target.r = deg2rad(-14 + 28 * rand);
        target.thrust = 2500 + 5500 * rand;
        target.de = -1 + 8 * rand;
        target.da = -8 + 16 * rand;
        target.dr = -6 + 12 * rand;

    case 'turn'
        target.modeId = 4;
        turnSign = sign(rand - 0.5);
        if turnSign == 0
            turnSign = 1;
        end
        target.alt = curAlt + (-1500 + 3000 * rand);
        target.vt = 430 + 220 * rand;
        target.alpha = deg2rad(0 + 10 * rand);
        target.beta = deg2rad(turnSign * (1 + 5 * rand));
        target.phi = deg2rad(turnSign * (25 + 35 * rand));
        target.theta = deg2rad(-3 + 10 * rand);
        target.p = deg2rad(turnSign * (10 + 25 * rand));
        target.q = deg2rad(-8 + 12 * rand);
        target.r = deg2rad(turnSign * (6 + 20 * rand));
        target.thrust = 6000 + 6500 * rand;
        target.de = -6 + 8 * rand;
        target.da = turnSign * (6 + 10 * rand);
        target.dr = turnSign * (2 + 6 * rand);

    case 'aggressive'
        target.modeId = 5;
        target.alt = curAlt + (-3500 + 7000 * rand);
        target.vt = 420 + 360 * rand;
        target.alpha = deg2rad(0 + 18 * rand);
        target.beta = deg2rad(-8 + 16 * rand);
        target.phi = deg2rad(-70 + 140 * rand);
        target.theta = deg2rad(-20 + 35 * rand);
        target.p = deg2rad(-45 + 90 * rand);
        target.q = deg2rad(-30 + 60 * rand);
        target.r = deg2rad(-28 + 56 * rand);
        target.thrust = 7000 + 10000 * rand;
        target.de = -16 + 22 * rand;
        target.da = -18 + 36 * rand;
        target.dr = -12 + 24 * rand;

    case 'highalpha'
        target.modeId = 6;
        target.alt = curAlt + (-1800 + 3600 * rand);
        target.vt = 320 + 180 * rand;
        target.alpha = deg2rad(12 + 12 * rand);
        target.beta = deg2rad(-6 + 12 * rand);
        target.phi = deg2rad(-35 + 70 * rand);
        target.theta = deg2rad(4 + 16 * rand);
        target.p = deg2rad(-22 + 44 * rand);
        target.q = deg2rad(4 + 24 * rand);
        target.r = deg2rad(-18 + 36 * rand);
        target.thrust = 9000 + 8500 * rand;
        target.de = -18 + 12 * rand;
        target.da = -10 + 20 * rand;
        target.dr = -8 + 16 * rand;

    otherwise % recovery
        target.modeId = 7;
        target.alt = curAlt + (-800 + 1600 * rand);
        target.vt = 480 + 120 * rand;
        target.alpha = deg2rad(-1 + 7 * rand);
        target.beta = deg2rad(-3 + 6 * rand);
        target.phi = deg2rad(-10 + 20 * rand);
        target.theta = deg2rad(-2 + 6 * rand);
        target.p = deg2rad(-8 + 16 * rand);
        target.q = deg2rad(-6 + 12 * rand);
        target.r = deg2rad(-8 + 16 * rand);
        target.thrust = 5500 + 4500 * rand;
        target.de = -4 + 5 * rand;
        target.da = -4 + 8 * rand;
        target.dr = -3 + 6 * rand;
end

target.alt = min(max(target.alt, cfg.altitudeRange(1)), cfg.altitudeRange(2));
if mod(flightIdx, 4) == 0 && target.modeId == 1
    target.beta = target.beta + deg2rad(2.5 * sign(randn));
end
end

function seq = localSmoothInterp(knotIdx, knotVal, N)
xq = 1:N;
seq = interp1(knotIdx, knotVal, xq, 'pchip');
seq = smoothdata(seq, 'gaussian', 7);
end

function windNav = localGenerateWind(cfg, N, dt, flightIdx, targets)
windNav = localGenerateMeanWind(cfg, N, dt);
windNav = windNav + localGenerateWindShear(cfg, targets, N);
windNav = windNav + localGenerateTurbulence(cfg, N, dt);
windNav = windNav + localGenerateDiscreteGusts(cfg, targets, N, dt, flightIdx);
end

function windNav = localGenerateMeanWind(cfg, N, dt)
windNav = zeros(2, N);
rho = exp(-dt / cfg.windTimeConstant);
sigma = cfg.windSigma(:) * sqrt(1 - rho^2);
windNav(:, 1) = cfg.windMean(:) + 3 * randn(2, 1);

for k = 2:N
    windNav(:, k) = cfg.windMean(:) + rho * (windNav(:, k - 1) - cfg.windMean(:)) + sigma .* randn(2, 1);
end
end

function shearNav = localGenerateWindShear(cfg, targets, N)
shearNav = zeros(2, N);
if ~isfield(cfg, 'windShear') || ~isfield(cfg.windShear, 'enabled') || ~cfg.windShear.enabled
    return;
end

altFt = localTargetRow(targets, 'alt', N, 0);
refAlt = double(cfg.windShear.referenceAltitude);
lowerAlt = double(cfg.windShear.lowerAltitude);
denom = max((refAlt - lowerAlt) ^ 2, 1e-9);
profile = ((refAlt - altFt) .^ 2) / denom;
profile = min(max(profile, 0), 1);

shearMps = cfg.windShear.saturatedWindMps(:) .* profile;
shearNav = double(cfg.windShear.mpsToFtps) * shearMps;
end

function turbNav = localGenerateTurbulence(cfg, N, dt)
turbNav = zeros(2, N);
if ~isfield(cfg, 'windTurbulence') || ~isfield(cfg.windTurbulence, 'enabled') || ~cfg.windTurbulence.enabled
    return;
end

tcfg = cfg.windTurbulence;
sampleStride = max(1, round(double(tcfg.randomSampleTime) / dt));
tau = max(double(tcfg.tau(:)), 1e-6);
gain = double(tcfg.numeratorGain(:));
sigmaMps = double(tcfg.sigmaMps(:));
noiseStd = sqrt(double(tcfg.randomVariance(:)));
scale = double(tcfg.scale);
mpsToFtps = double(tcfg.mpsToFtps);

state = zeros(2, 1);
u = noiseStd .* randn(2, 1);
for k = 1:N
    if k == 1 || mod(k - 1, sampleStride) == 0
        u = noiseStd .* randn(2, 1);
    end
    a = exp(-dt ./ tau);
    state = a .* state + (1 - a) .* gain .* u;
    turbNav(:, k) = mpsToFtps * scale * sigmaMps .* state;
end
end

function gustNav = localGenerateDiscreteGusts(cfg, targets, N, dt, flightIdx)
gustNav = zeros(2, N);
if isfield(cfg, 'windGust') && isfield(cfg.windGust, 'enabled') && cfg.windGust.enabled
    gustNav = localGenerateSimulinkLikeGusts(cfg.windGust, targets, N, dt, flightIdx);
    return;
end

numGusts = cfg.gustEventRange(1) + mod(flightIdx, diff(cfg.gustEventRange) + 1);
for i = 1:numGusts
    len = randi(cfg.gustDurationRange);
    t0 = randi([1, max(1, N - len + 1)]);
    axisIdx = randi(2);
    amp = cfg.gustAmplitude(axisIdx) * (0.5 + rand);
    tau = linspace(0, pi, len);
    gust = 0.5 * amp * (1 - cos(tau));
    if rand < 0.5
        gust = -gust;
    end
    idx = t0:(t0 + len - 1);
    gustNav(axisIdx, idx) = gustNav(axisIdx, idx) + gust;
end
end

function gustNav = localGenerateSimulinkLikeGusts(gcfg, targets, N, dt, flightIdx) %#ok<INUSD>
gustNav = zeros(2, N);
t = (0:N - 1) * dt;
vtMps = 0.3048 * localTargetRow(targets, 'vt', N, 500);
axes = round(double(gcfg.axis(:)));
lengthM = double(gcfg.lengthM(:));
ampMps = double(gcfg.amplitudeMps(:));
startTimes = localResolveGustStartTimes(gcfg, numel(axes));

for i = 1:numel(axes)
    axisIdx = min(max(axes(i), 1), 2);
    d = max(lengthM(i), 1e-6);
    dist = 0;
    for k = 1:N
        if t(k) >= startTimes(i)
            x = min(dist, d);
            gustNav(axisIdx, k) = gustNav(axisIdx, k) + ...
                double(gcfg.mpsToFtps) * 0.5 * ampMps(i) * (1 - cos(pi * x / d));
            dist = min(dist + vtMps(k) * dt, d);
        end
    end
end
end

function startTimes = localResolveGustStartTimes(gcfg, count)
startTimes = nan(count, 1);
if isfield(gcfg, 'startTimeSec') && numel(gcfg.startTimeSec) >= count
    startTimes = double(gcfg.startTimeSec(:));
    startTimes = startTimes(1:count);
end

rangeSec = double(gcfg.startTimeRangeSec(:)');
for i = 1:count
    if ~isfinite(startTimes(i))
        startTimes(i) = rangeSec(1) + rand * diff(rangeSec);
    end
end
end

function row = localTargetRow(targets, fieldName, N, defaultValue)
if isstruct(targets) && isfield(targets, fieldName)
    row = double(targets.(fieldName));
    row = row(:)';
else
    row = defaultValue * ones(1, N);
end

if numel(row) < N
    row(end + 1:N) = row(end);
end
row = row(1:N);
end

function [measNominal, vwPrior] = localNominalSensorModel(yClean, windNav, cfg, N)
measNominal = yClean;
bias = zeros(6, 1);
rho = cfg.sensorBiasRho;

for k = 1:N
    bias = rho * bias + cfg.sensorBiasStd(:) .* randn(6, 1);
    white = cfg.sensorNoiseStd(:) .* randn(6, 1);
    measNominal(:, k) = measNominal(:, k) + bias + white;
end

vwPrior = windNav + cfg.windPriorStd(:) .* randn(2, N);
vwPrior = smoothdata(vwPrior, 2, 'gaussian', 11);
end

function [measOut, anomalyMask, anomalyType, anomalySeverity, substLabel] = localInjectFlightAnomalies(measIn, zPred, cfg)
[C, N] = size(measIn);
measOut = measIn;
anomalyMask = zeros(C, N);
anomalyType = zeros(1, N);
anomalySeverity = zeros(1, N);
substLabel = zeros(1, N);

numEvents = randi(cfg.numEventsRange);
for ev = 1:numEvents
    typeId = randi(6);
    severityId = randi([2, 3]);
    len = randi(cfg.segmentRange);
    len = min(len, N);
    t0 = randi([1, N - len + 1]);
    idx = t0:(t0 + len - 1);
    chCount = randi([1, min(3, C)]);
    channels = randperm(C, chCount);
    amp = cfg.severityAmp(severityId);

    switch typeId
        case 1  % burst spike
            for ch = channels
                spikePts = idx(rand(1, numel(idx)) < 0.45);
                if isempty(spikePts)
                    spikePts = idx(randi(numel(idx)));
                end
                measOut(ch, spikePts) = measOut(ch, spikePts) + amp * cfg.channelScale(ch) * sign(randn(1, numel(spikePts)));
            end

        case 2  % bias drift
            drift = linspace(0, amp, numel(idx));
            for ch = channels
                measOut(ch, idx) = measOut(ch, idx) + cfg.channelScale(ch) * drift;
            end

        case 3  % stuck
            for ch = channels
                holdVal = measOut(ch, max(1, t0 - 1));
                measOut(ch, idx) = holdVal;
            end

        case 4  % dropout to model prediction
            for ch = channels
                measOut(ch, idx) = zPred(ch, idx);
            end

        case 5  % scale factor
            scale = 1 + 0.12 * amp * sign(randn);
            for ch = channels
                measOut(ch, idx) = scale * measOut(ch, idx);
            end

        otherwise % short delay
            lag = min(randi([1, 5]), t0 - 1);
            for ch = channels
                if lag > 0
                    measOut(ch, idx) = measOut(ch, idx - lag);
                end
            end
    end

    anomalyMask(channels, idx) = 1;
    anomalyType(idx) = typeId;
    anomalySeverity(idx) = max(anomalySeverity(idx), severityId);
    substLabel(idx) = max(substLabel(idx), double(severityId >= 2 || any(cfg.substTypes == typeId)));
end
end

function y = localModelObs(state, controls, vw)
[ud, vd, wd, pd, qd, rd] = wind_estimation_model_numeric(state, controls, vw);
y = [ud; vd; wd; pd; qd; rd];
end

function Cnb = localCnb(phi, theta, psi)
Cnb = [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta); ...
       sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi), ...
       sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), ...
       sin(phi) * cos(theta); ...
       cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi), ...
       cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), ...
       cos(phi) * cos(theta)];
end

function a = localWrapToPi(a)
a = atan2(sin(a), cos(a));
end

function cfg = localDefaultCfg()
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
