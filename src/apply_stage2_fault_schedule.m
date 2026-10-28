function replay = apply_stage2_fault_schedule(measurementNominal, stateMat, controlMat, schedule)
% Replay a shared stage-2 fault schedule on sampled online signals.

measurementNominal = double(measurementNominal);
stateMat = double(stateMat);
controlMat = double(controlMat);

if size(measurementNominal, 1) ~= 6
    error('measurementNominal must be 6xN.');
end
if size(stateMat, 1) ~= 12
    error('stateMat must be 12xN.');
end
if size(controlMat, 1) ~= 4
    error('controlMat must be 4xN.');
end

N = min([size(measurementNominal, 2), size(stateMat, 2), size(controlMat, 2), double(schedule.N)]);
ctx = stage2_fault_runtime_init(schedule);

measurementFault = zeros(6, N);
anomalyMask = zeros(6, N);
anomalyType = zeros(1, N);
anomalySeverity = zeros(1, N);
substLabel = zeros(1, N);

for k = 1:N
    [measurementFault(:, k), info, ctx] = stage2_fault_runtime_step( ...
        ctx, measurementNominal(:, k), stateMat(:, k), controlMat(:, k));
    anomalyMask(:, k) = info.anomalyMask(:);
    anomalyType(k) = info.anomalyType;
    anomalySeverity(k) = info.anomalySeverity;
    substLabel(k) = info.substLabel;
end

replay = struct();
replay.measurements = measurementFault;
replay.anomaly_mask = anomalyMask;
replay.anomaly_type = anomalyType;
replay.anomaly_severity = anomalySeverity;
replay.subst_label = substLabel;
replay.sampleTime = double(schedule.sampleTime);
replay.N = N;
replay.schedule = schedule;
end
