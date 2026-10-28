ov = stage2_10ms_defaults();

stage2TrainOverride = struct();
stage2TrainOverride.train = struct();
stage2TrainOverride.train.outputModelFile = "lstm_akf_net_10ms_windenv_aerobias5_10.mat";
stage2TrainOverride.train.initModelFile = "";
stage2TrainOverride.train.numEpochs = ov.train.numEpochs;
stage2TrainOverride.train.oofEpochs = ov.train.oofEpochs;

stage2TrainOverride.synthetic = struct();
stage2TrainOverride.synthetic.aeroBias = struct( ...
    'enabled', true, ...
    'minPct', 0.05, ...
    'maxPct', 0.10, ...
    'augmentCopies', 1, ...
    'seed', 20260507);

stage2TrainOverride.runtime = struct();
stage2TrainOverride.runtime.modelFile = stage2TrainOverride.train.outputModelFile;

run_stage2_10ms_training;

clear stage2TrainOverride
