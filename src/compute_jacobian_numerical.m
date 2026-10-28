function H = compute_jacobian_numerical(Vw_nominal, state, controls, z_nominal)
    % 使用有限差分法计算雅可比矩阵 H (6x2)
    % H(:,1) = d(Measurements)/d(Vw_N)
    % H(:,2) = d(Measurements)/d(Vw_E)
    
    delta = 0.05; % 扰动步长，单位 m/s 或 ft/s (取决于你的模型单位，这里是ft/s)
    H = zeros(6, 2);
    
    % --- 对 Vw_N (状态1) 求偏导 ---
%     Vw_pert_N = Vw_nominal;
%     Vw_pert_N(1) = Vw_pert_N(1) + delta;
% 
%     [ud_N, vd_N, wd_N, pd_N, qd_N, rd_N] = wind_estimation_model_numeric(state, controls, Vw_pert_N);
%     z_pert_N = [ud_N; vd_N; wd_N; pd_N; qd_N; rd_N];
% 
%     H(:, 1) = (z_pert_N - z_nominal) / delta;
% 
%     % --- 对 Vw_E (状态2) 求偏导 ---
%     Vw_pert_E = Vw_nominal;
%     Vw_pert_E(2) = Vw_pert_E(2) + delta;
% 
%     [ud_E, vd_E, wd_E, pd_E, qd_E, rd_E] = wind_estimation_model_numeric(state, controls, Vw_pert_E);
%     z_pert_E = [ud_E; vd_E; wd_E; pd_E; qd_E; rd_E];
% 
%     H(:, 2) = (z_pert_E - z_nominal) / delta;
% end
% --- 对 Vw_N (状态1) 求偏导 ---
    Vw_plus_N = Vw_nominal;  
    Vw_plus_N(1) = Vw_plus_N(1) + delta;
    Vw_minus_N = Vw_nominal; Vw_minus_N(1) = Vw_minus_N(1) -delta;
    
    [u_pN, v_pN, w_pN, p_pN, q_pN, r_pN] =wind_estimation_model_numeric(state, controls, Vw_plus_N);
    [u_mN, v_mN, w_mN, p_mN, q_mN, r_mN] =wind_estimation_model_numeric(state, controls, Vw_minus_N);
    
    H(:, 1) = ([u_pN; v_pN; w_pN; p_pN; q_pN; r_pN] - [u_mN; v_mN; w_mN; p_mN; q_mN; r_mN]) / (2 * delta);
    
    % --- 对 Vw_E (状态2) 求偏导 ---
    Vw_plus_E = Vw_nominal;  
    Vw_plus_E(2) = Vw_plus_E(2) +delta;
    Vw_minus_E = Vw_nominal; 
    Vw_minus_E(2) = Vw_minus_E(2) -delta;
    
    [u_pE, v_pE, w_pE, p_pE, q_pE, r_pE] =wind_estimation_model_numeric(state, controls, Vw_plus_E);
    [u_mE, v_mE, w_mE, p_mE, q_mE, r_mE] =wind_estimation_model_numeric(state, controls, Vw_minus_E);
    
    H(:, 2) = ([u_pE; v_pE; w_pE; p_pE; q_pE; r_pE] - [u_mE; v_mE; w_mE; p_mE; q_mE; r_mE]) / (2 *delta);
end