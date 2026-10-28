function schedule = build_stage2_fault_schedule(numSamples, cfg, seed)
% Build a deterministic stage-2 fault event table.

if nargin < 2 || isempty(cfg)
    cfg = stage2_fault_default_cfg();
end
if nargin < 3 || isempty(seed)
    seed = 42;
end

rngState = rng;
cleanupObj = onCleanup(@() rng(rngState)); %#ok<NASGU>
rng(seed);
N = max(1, round(double(numSamples)));
channelCount = numel(cfg.channelScale);
numEvents = localResolveNumEvents(cfg.numEventsRange);

baseEvent = struct( ...
    'typeId', 0, ...
    'severityId', 0, ...
    'amp', 0, ...
    'channels', [], ...
    'channelScale', [], ...
    'startIdx', 1, ...
    'endIdx', 1, ...
    'length', 1, ...
    'scale', 1, ...
    'lag', 0, ...
    'subst', false, ...
    'drift', [], ...
    'spikeMask', [], ...
    'spikeSign', []);

events = repmat(baseEvent, numEvents, 1);
activeEventIds = cell(1, N);
anomalyMask = zeros(channelCount, N);
anomalyType = zeros(1, N);
anomalySeverity = zeros(1, N);
substLabel = zeros(1, N);

for ev = 1:numEvents
    typeId = cfg.typePool(randi(numel(cfg.typePool)));
    severityId = randi([1, 3]);
    amp = cfg.severityAmp(severityId);
    segLen = min(randi(cfg.segmentRange), N);
    startIdx = randi([1, N - segLen + 1]);
    endIdx = startIdx + segLen - 1;
    chCount = randi([1, min(cfg.maxChannelsPerEvent, channelCount)]);
    channels = sort(randperm(channelCount, chCount));

    event = baseEvent;
    event.typeId = typeId;
    event.severityId = severityId;
    event.amp = amp;
    event.channels = channels(:)';
    event.channelScale = double(cfg.channelScale(channels));
    event.startIdx = startIdx;
    event.endIdx = endIdx;
    event.length = segLen;
    event.subst = severityId >= 3 || any(cfg.substTypes == typeId);

    switch typeId
        case 1
            spikeMask = rand(1, segLen) < 0.45;
            if ~any(spikeMask)
                spikeMask(randi(segLen)) = true;
            end
            spikeSign = sign(randn(chCount, segLen));
            spikeSign(spikeSign == 0) = 1;
            event.spikeMask = spikeMask;
            event.spikeSign = spikeSign;

        case 2
            event.drift = linspace(0, amp, segLen);

        case 5
            event.scale = 1 + 0.08 * amp * sign(randn);

        case 6
            event.lag = min(randi([1, 4]), startIdx - 1);
    end

    events(ev) = event;
    idx = startIdx:endIdx;
    anomalyMask(channels, idx) = 1;
    anomalyType(idx) = typeId;
    anomalySeverity(idx) = max(anomalySeverity(idx), severityId);
    if event.subst
        substLabel(idx) = 1;
    end
    for k = idx
        activeEventIds{k}(end + 1) = ev; %#ok<AGROW>
    end
end

schedule = struct();
schedule.seed = seed;
schedule.sampleTime = cfg.sampleTime;
schedule.N = N;
schedule.config = cfg;
schedule.events = events;
schedule.activeEventIds = activeEventIds;
schedule.anomalyMask = anomalyMask;
schedule.anomalyType = anomalyType;
schedule.anomalySeverity = anomalySeverity;
schedule.substLabel = substLabel;
schedule.channelCount = channelCount;
schedule.generatedAt = string(datetime('now'));
end

function numEvents = localResolveNumEvents(rangeValue)
rangeValue = double(rangeValue(:)');
if isempty(rangeValue)
    numEvents = 0;
    return;
end

if numel(rangeValue) == 1
    low = max(0, round(rangeValue(1)));
    high = low;
else
    low = max(0, round(min(rangeValue)));
    high = max(0, round(max(rangeValue)));
end

if high <= 0
    numEvents = 0;
elseif low == high
    numEvents = low;
else
    numEvents = randi([low, high]);
end
end
