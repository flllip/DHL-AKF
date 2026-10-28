function [measurementFault, info, ctx] = stage2_fault_runtime_step(ctx, measurementNominal, state, controls)
% Apply one scheduled stage-2 online fault step.

ctx.k = ctx.k + 1;
k = ctx.k;
measurementNominal = reshape(double(measurementNominal), [6, 1]);
state = reshape(double(state), [12, 1]);
controls = reshape(double(controls), [4, 1]);

if k > ctx.schedule.N
    measurementFault = measurementNominal;
    info = localEmptyInfo(ctx.channelCount);
    return;
end

[F, Q] = localCVModel(ctx.sampleTime, ctx.teacherQAccel);
xPred = F * ctx.teacherXHat;
PPred = F * ctx.teacherP * F' + Q;
zPred = localModelObs(state, controls, xPred(1:2));
H2 = compute_jacobian_numerical(xPred(1:2), state, controls, zPred);
H = [H2, zeros(6, 2)];
innov = measurementNominal - zPred;
S = H * PPred * H' + ctx.teacherRc;
K = (PPred * H') / S;
ctx.teacherXHat = xPred + K * innov;
I = eye(4);
ctx.teacherP = (I - K * H) * PPred * (I - K * H)' + K * ctx.teacherRc * K';

measurementFault = measurementNominal;
activeIds = ctx.schedule.activeEventIds{k};
for ii = 1:numel(activeIds)
    event = ctx.schedule.events(activeIds(ii));
    relIdx = k - event.startIdx + 1;
    ch = event.channels;
    switch event.typeId
        case 1
            if relIdx <= numel(event.spikeMask) && event.spikeMask(relIdx)
                delta = event.amp .* event.channelScale(:) .* event.spikeSign(:, relIdx);
                measurementFault(ch) = measurementFault(ch) + delta;
            end
        case 2
            measurementFault(ch) = measurementFault(ch) + event.channelScale(:) .* event.drift(relIdx);
        case 3
            if isempty(ctx.holdValues{activeIds(ii)})
                if k > 1 && all(isfinite(ctx.outputHist(ch, k - 1)))
                    ctx.holdValues{activeIds(ii)} = ctx.outputHist(ch, k - 1);
                else
                    ctx.holdValues{activeIds(ii)} = measurementFault(ch);
                end
            end
            measurementFault(ch) = ctx.holdValues{activeIds(ii)};
        case 4
            measurementFault(ch) = zPred(ch);
        case 5
            measurementFault(ch) = event.scale * measurementFault(ch);
        otherwise
            lag = event.lag;
            if lag > 0 && (k - lag) >= 1 && all(isfinite(ctx.outputHist(ch, k - lag)))
                measurementFault(ch) = ctx.outputHist(ch, k - lag);
            end
    end
end

ctx.outputHist(:, k) = measurementFault;
info = localInfo(ctx.schedule, k);
end

function info = localEmptyInfo(channelCount)
info = struct();
info.anomalyMask = zeros(channelCount, 1);
info.anomalyType = 0;
info.anomalySeverity = 0;
info.substLabel = 0;
end

function info = localInfo(schedule, k)
info = struct();
info.anomalyMask = schedule.anomalyMask(:, k);
info.anomalyType = schedule.anomalyType(k);
info.anomalySeverity = schedule.anomalySeverity(k);
info.substLabel = schedule.substLabel(k);
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
