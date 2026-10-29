# DHL-AKF

Core MATLAB code for **A physics-constrained dual-head LSTM-assisted adaptive Kalman filtering framework for aircraft air data estimation**.

## Core code

| Component | Main files |
| --- | --- |
| Online DHL-AKF estimator | `lstm_ekf_fusion_step.m` |
| Network training and multitask losses | `generate_data_and_train_from_logs.m` |
| Main configuration | `stage2_10ms_defaults.m` |
| Causal multiscale features | `append_causal_multiscale_features.m` |
| Input feature construction | `compose_lstm_aux_features.m`, `compose_lstm_fusion_input.m` |
| Network output decoding | `decode_lstm_aux_output.m` |

**The remaining supporting code and implementation details are being organized and documented. Coming soon.**

This repository currently contains selected core MATLAB source files. Supporting modules, datasets, trained weights, and reproduction instructions will be added in a later release. The current release is not a standalone reproduction package.
