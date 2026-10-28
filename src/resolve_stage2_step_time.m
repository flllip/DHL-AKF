function dt = resolve_stage2_step_time(defaultDt, modelFile)
% Resolve stage-2 runtime sample time from override or model summary.

if nargin < 1 || isempty(defaultDt)
    defaultDt = 0.05;
end
if nargin < 2
    modelFile = '';
end

dt = double(defaultDt);

try
    if evalin('base', "exist('stage2StepTimeOverride','var')")
        dtCand = double(evalin('base', 'stage2StepTimeOverride'));
        if isfinite(dtCand) && dtCand > 0
            dt = dtCand;
            return;
        end
    end
catch
end

if isempty(modelFile) || ~isfile(modelFile)
    return;
end

try
    S = load(modelFile, 'trainSummary');
    if isfield(S, 'trainSummary') && isfield(S.trainSummary, 'sampleTime')
        dtCand = double(S.trainSummary.sampleTime);
        if isfinite(dtCand) && dtCand > 0
            dt = dtCand;
        end
    end
catch
end
end
