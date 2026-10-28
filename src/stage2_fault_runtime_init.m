function ctx = stage2_fault_runtime_init(schedule)
% Initialize shared online fault runtime state.

if nargin < 1 || isempty(schedule)
    error('stage2_fault_runtime_init requires a non-empty schedule.');
end

ctx = struct();
ctx.schedule = schedule;
ctx.k = 0;
ctx.sampleTime = double(schedule.sampleTime);
ctx.teacherXHat = zeros(4, 1);
ctx.teacherP = diag([10, 10, 2, 2]);
ctx.teacherQAccel = [0.20, 0.20] .^ 2;
ctx.teacherRc = diag([0.44, 0.44, 0.44, 0.2, 0.2, 0.2] .^ 2);
ctx.channelCount = double(schedule.channelCount);
ctx.outputHist = nan(ctx.channelCount, schedule.N);
ctx.holdValues = cell(numel(schedule.events), 1);
end
