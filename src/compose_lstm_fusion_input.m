function x = compose_lstm_fusion_input(xPredAug, measurements, state, controls, zPred)
% Compose the 34D fusion-network input used by training and online inference.
% Layout:
%   xPred_aug[4]
%   measurement[6]
%   zAux[12]
%   zPred[6]
%   innovation_raw[6]

xPredAug = reshape(xPredAug, [4, 1]);
measurements = reshape(measurements, [6, 1]);
zPred = reshape(zPred, [6, 1]);

zAux = compose_lstm_aux_features(state, controls);
innovation = measurements - zPred;

x = [xPredAug; measurements; zAux; zPred; innovation];
end
