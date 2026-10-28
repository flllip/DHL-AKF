function cfg = stage2_fault_default_cfg(sampleTime)
% Default stage-2 online fault schedule config.

if nargin < 1 || isempty(sampleTime)
    sampleTime = 0.01;
end

scale = max(round(0.05 / sampleTime), 1);

cfg = struct();
cfg.sampleTime = sampleTime;
cfg.numEventsRange = [5, 9];
cfg.severityAmp = [1.0, 1.8, 2.8];
cfg.segmentRange = scale * [12, 90];
cfg.typePool = [1, 2, 3, 4, 5, 6];
cfg.substTypes = [3, 4, 6];
cfg.channelScale = [2.5; 5.0; 5.0; 0.45; 0.25; 0.12];
cfg.maxChannelsPerEvent = 3;
end
