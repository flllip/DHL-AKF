function [Vw_est, air_est, lambda_k,xPred] = ekf_baseline_step(state, controls, measurements, reset)
% Baseline EKF step without LSTM branch.
% Inputs/outputs use the same conventions as lstm_ekf_fusion_step.

persistent xHat P Q Rc T_alpha stepTime initialized

if isempty(initialized)
    initialized = false;
end

if ~initialized || isempty(Q) || isempty(Rc)
    modelFile = resolve_stage2_model_file('lstm_akf_net.mat');
    stepTime = resolve_stage2_step_time(0.05, modelFile);
    Q = max(stepTime / 0.05, 1e-3) * diag([0.15, 0.15] .^ 2);
    Rc = diag([0.44, 0.44, 0.44, 0.2, 0.2, 0.2] .^ 2);
    T_alpha = 16.812;
end

if ~initialized || reset ~= 0 || isempty(xHat) || isempty(P)
    xHat = [0; 0];
    P = diag([10, 10]);
    initialized = true;
end

xPred = xHat;
PPred = P + Q;
zPred = modelObs(state, controls, xPred);
H = compute_jacobian_numerical(xPred, state, controls, zPred);

e = measurements(:) - zPred;
S = H * PPred * H' + Rc;
lambda_k = mahalanobisValue(e, S); %#ok<NASGU>

K = (PPred * H') / S;
xHat = xPred + K * e;

I = eye(2);
P = (I - K * H) * PPred * (I - K * H)' + K * Rc * K';

[VT, alpha, beta] = reconstructAtmosphere(state, xHat);

Vw_est = xHat;
air_est = [VT; alpha; beta];
end

function y = modelObs(state, controls, vw)
[ud, vd, wd, pd, qd, rd] = wind_estimation_model_numeric(state, controls, vw);
y = [ud; vd; wd; pd; qd; rd];
end

function lambda = mahalanobisValue(e, S)
S = 0.5 * (S + S');
S = S + 1e-9 * eye(size(S));
lambda = real(e' * (S \ e));
end

function [vt, alpha, beta] = reconstructAtmosphere(state, vw)
phi = state(4); theta = state(5); psi = state(6);
u = state(7); v = state(8); w = state(9);

Cnb = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta); ...
       sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), ...
       sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), ...
       sin(phi)*cos(theta); ...
       cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), ...
       cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), ...
       cos(phi)*cos(theta)];

vwBody = Cnb * [vw(1); vw(2); 0];
vtBody = [u; v; w] - vwBody;

vt = max(norm(vtBody), 1e-6);
alpha = atan2(vtBody(3), vtBody(1));
beta = asin(max(min(vtBody(2) / vt, 1), -1));
end

