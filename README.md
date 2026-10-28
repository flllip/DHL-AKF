# DHL-AKF

Core MATLAB code for **A physics-constrained dual-head LSTM-assisted adaptive Kalman filtering framework for aircraft air data estimation**.

## Release status

This initial release provides the core implementation of the dual-head LSTM-assisted adaptive Kalman filter, including observation recovery, two-stage innovation consistency tests, and channel-wise conservative fallback.

**The remaining code and implementation details are being organized and documented. Coming soon.**

This is a partial source release. Pretrained network weights, generated flight datasets, Simulink models, external aircraft-model dependencies, the complete baseline implementations, and processor-in-the-loop deployment code are not included in this initial release. The provided sources alone are not a complete end-to-end reproduction package.

## Included code

All MATLAB files are in `src/`, retaining their original filenames and contents.

| Component | Main files |
| --- | --- |
| Online DHL-AKF estimator | `lstm_ekf_fusion_step.m` |
| Network training and multitask losses | `generate_data_and_train_from_logs.m` |
| Training entry points and configuration | `run_stage2_10ms_training*.m`, `stage2_10ms_defaults.m` |
| Synthetic flight logs and training sequences | `generate_full_envelope_synthetic_logs.m`, `build_training_data_from_logs.m` |
| Causal multiscale features | `append_causal_multiscale_features.m`, `compose_lstm_aux_features.m` |
| Candidate decoding and runtime loading | `decode_lstm_aux_output.m`, `decode_lstm_residual_output.m`, `load_stage2_runtime_bundle.m` |
| Observation models and air-data reconstruction | `wind_estimation_model_numeric*.m`, `reconstruct_airdata_from_state_wind.m` |
| Sensor degradation and replay evaluation | `build_stage2_fault_schedule.m`, `apply_stage2_fault_schedule.m`, `run_stage2_replay.m`, `run_stage2_entry_benchmark.m` |

## Environment

The implementation uses MATLAB. Network training and inference require Deep Learning Toolbox. Complete dependency setup, runnable examples, and reproduction instructions are being prepared and will be added with the remaining code.
