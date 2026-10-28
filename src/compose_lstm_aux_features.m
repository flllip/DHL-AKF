function z = compose_lstm_aux_features(state, controls, varargin)
% Shared 12D auxiliary features for LSTM training and inference.
% varargin is accepted for backward-compatible call sites that pass xPred.

phi = state(4);
theta = state(5);
thrust = controls(1);
de = controls(2);
da = controls(3);
dr = controls(4);
u = state(7);
v = state(8);
w = state(9);
p = state(10);
q = state(11);
r = state(12);

z = [u; v; w; p; q; r; phi; theta; thrust; de; da; dr];
end
