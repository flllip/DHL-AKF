function modelFile = resolve_stage2_model_file(defaultModelFile)
% Resolve stage-2 runtime model file from base workspace override.

if nargin < 1 || isempty(defaultModelFile)
    defaultModelFile = 'lstm_akf_net.mat';
end

modelFile = char(defaultModelFile);

try
    if evalin('base', "exist('stage2ModelFileOverride','var')")
        candidate = evalin('base', 'stage2ModelFileOverride');
        if isstring(candidate) || ischar(candidate)
            candidate = char(candidate);
            if ~isempty(strtrim(candidate))
                modelFile = candidate;
            end
        end
    end
catch
    modelFile = char(defaultModelFile);
end
end
