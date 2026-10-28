function teacher = build_teacher_signals(stateMat, controlMat, measMat, sampleTime, varargin)
% Build teacher prediction signals using a 4D constant-velocity wind model.
%
% teacher.xPred    [4 x N]
% teacher.zPred    [6 x N]
% teacher.lambda1  [1 x N]

cfg.rcDiag = [0.44, 0.44, 0.44, 0.2, 0.2, 0.2] .^ 2;
cfg.qAccel = [0.20, 0.20] .^ 2;
cfg.x0 = zeros(4, 1);
cfg.P0 = diag([10, 10, 2, 2]);

if ~isempty(varargin)
    override = varargin{1};
    if isstruct(override)
        f = fieldnames(override);
        for i = 1:numel(f)
            cfg.(f{i}) = override.(f{i});
        end
    end
end

dt = double(sampleTime);
if ~(isfinite(dt) && dt > 0)
    dt = 0.02;
end

N = size(stateMat, 2);
xHat = cfg.x0;
P = cfg.P0;
Rc = diag(cfg.rcDiag);

teacher.xPred = zeros(4, N);
teacher.zPred = zeros(6, N);
teacher.lambda1 = zeros(1, N);

for k = 1:N
    [F, Q] = localCVModel(dt, cfg.qAccel);
    xPred = F * xHat;
    PPred = F * P * F' + Q;

    zPred = localModelObs(stateMat(:, k), controlMat(:, k), xPred(1:2));
    H2 = compute_jacobian_numerical(xPred(1:2), stateMat(:, k), controlMat(:, k), zPred);
    H = [H2, zeros(6, 2)];

    innov = measMat(:, k) - zPred;
    S = H * PPred * H' + Rc;
    lambda1 = real(innov' * (S \ innov));

    K = (PPred * H') / S;
    xHat = xPred + K * innov;
    I = eye(4);
    P = (I - K * H) * PPred * (I - K * H)' + K * Rc * K';

    teacher.xPred(:, k) = xPred;
    teacher.zPred(:, k) = zPred;
    teacher.lambda1(k) = lambda1;
end
end

function [F, Q] = localCVModel(dt, qAccel)
qN = qAccel(1);
qE = qAccel(2);

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

function z = localModelObs(state, controls, vw)
[ud, vd, wd, pd, qd, rd] = wind_estimation_model_numeric(state, controls, vw);
z = [ud; vd; wd; pd; qd; rd];
end
