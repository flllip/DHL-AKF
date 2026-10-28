ov = stage2_10ms_defaults();

trainOverride = struct();
trainOverride.data = ov.data;
trainOverride.synthetic = ov.synthetic;
trainOverride.model = struct();
trainOverride.model.multiscaleWindows = ov.model.multiscaleWindows;
trainOverride.filter = ov.filter;
trainOverride.train = struct();
trainOverride.train.outputModelFile = ov.train.outputModelFile;
trainOverride.train.initModelFile = ov.train.initModelFile;
trainOverride.train.numEpochs = ov.train.numEpochs;
trainOverride.train.oofEpochs = ov.train.oofEpochs;
trainOverride.loss = ov.loss;
trainOverride.meta = ov.meta;
trainOverride.runtime = ov.runtime;

if exist('stage2TrainOverride', 'var') && isstruct(stage2TrainOverride)
    trainOverride = localMergeStruct(trainOverride, stage2TrainOverride);
end

generate_data_and_train_from_logs

function out = localMergeStruct(base, override)
out = base;
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
