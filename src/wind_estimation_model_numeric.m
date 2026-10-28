function [Udot, Vdot, Wdot, Pdot, Qdot, Rdot] = wind_estimation_model_numeric( state, controls,Vw)
% �������ٹ���ģ�� - ��Ӧ�����й�ʽ(8)
% ����:
%   Vw_N, Vw_E: ���ٵı���Ͷ������ (���ű���)
%   state: ������״̬���� [npos, epos, alt, phi, theta, psi, Ug, Vg, Wg, P, Q, R]
%   controls: ������������ [T, el, ail, rud, lef]
% ���:
%   Udot, Vdot, Wdot, Pdot, Qdot, Rdot: ������������ֵ

    % ��״̬��������ȡ����
    phi = state(4);     % ��ת�� (rad)
    theta = state(5);   % ������ (rad)
    psi = state(6);     % ƫ���� (rad)
     %U = state(7)*cos(state(8))*cos(state(9));  % ��������ϵx���ٶ� (ft/s)
    % V = state(7)*sin(state(9));       % ��������ϵy���ٶ� (ft/s)
   %  W = state(7)*sin(state(8))*cos(state(9));       % ��������ϵz���ٶ� (ft/s)
   U = state(7);
   V=state(8);
   W=state(9);
P = state(10);      % ��ת���ٶ� (rad/s)
    Q = state(11);      % �������ٶ� (rad/s)
    R = state(12);      % ƫ�����ٶ� (rad/s)m
    alt = state(3);     % �߶� (ft)
    
    % �ӿ�����������ȡ����
    T = controls(1);    % ���� (lb)
    el = controls(2);   % ������ (deg)
    ail = controls(3);  % ���� (deg)
    rud = controls(4);  % ����� (deg)
    
    % ��������
    g = 32.17;          % �������ٶ�, ft/s^2
    m = 636.94;         % ����, slugs
    B = 30.0;           % ��չ, ft
    S = 300.0;          % �������, ft^2
    cbar = 11.32;       % ƽ�������ҳ�, ft
%     xcgr = 0.35;        % �ο�����λ��
%     xcg = 0.30;         % ʵ������λ��
    
    % ���Ծ�
    Jx = 9496.0;
    Jy = 55814.0;
    Jz = 63100.0;
    Jxz = 982.0;
    
    % ����1: �����ٴӵ�������ϵת������������ϵ
    % ��������ϵ����������ϵ��ת������
    C_N_B = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta);
            sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), ...
            sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), ...
            sin(phi)*cos(theta);
            cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), ...
            cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), ...
            cos(phi)*cos(theta)];
    
    % �����ڵ�������ϵ�� (���费ֱ����Ϊ0)
    Vw_N=Vw(1);
    Vw_E=Vw(2);
    Vw_NED = [Vw_N; Vw_E; 0];
    
    % �����ڻ�������ϵ��
    Vw_B = C_N_B * Vw_NED;
    
    % ����2: ��������� (��ʽ3)
    % �����ڻ�������ϵ��
    Vg_B = [U; V; W];
    
    % ������ڻ�������ϵ��
    Vt_B = Vg_B - Vw_B;
    
    % ����3: ������ټ���������� (��ʽ5)
    Vt_x = Vt_B(1);
    Vt_y = Vt_B(2);
    Vt_z = Vt_B(3);
    
    V_T = sqrt(Vt_x^2 + Vt_y^2 + Vt_z^2);  % �����ģֵ
    alpha = atan2(Vt_z, Vt_x);             % ���� (rad)
    beta = asin(Vt_y / V_T);               % �໬�� (rad)
    
    % ���Ƕ�ת��Ϊ�� (F-16ģ����Ҫ)
    alpha_deg = alpha * 180/pi;
    beta_deg = beta * 180/pi;
    
    % ����4: ����F-16�ͱ�������ģ�ͼ�����������ϵ��
    [Cx_tot, Cy_tot, Cz_tot, Cl_tot, Cm_tot, Cn_tot] = ...
        f16_lofi_aerodynamic_coeffs(alpha_deg, beta_deg, el, ail, rud, P, Q, R, V_T);
    
    % ����5: ���㶯ѹ
    [~, qbar, ~] = atmos_model(alt, V_T);  % �߶Ⱥ�����ټ��㶯ѹ
    
    % ����6: ��������������
    F_x = qbar * S * Cx_tot + T;
    F_y = qbar * S * Cy_tot;
    F_z = qbar * S * Cz_tot;
    
    M_x = Cl_tot * qbar * S * B;
    M_y = Cm_tot * qbar * S * cbar;
    M_z = Cn_tot * qbar * S * B;
    
    % ����7: �����߼��ٶȺͽǼ��ٶ�
    % �߼��ٶ�
    Udot = R*V - Q*W - g*sin(theta) + F_x/m;
    Vdot = P*W - R*U + g*cos(theta)*sin(phi) + F_y/m;
    Wdot = Q*U - P*V + g*cos(theta)*cos(phi) + F_z/m;
    
    % �Ǽ��ٶ�
    denom = Jx*Jz - Jxz*Jxz;
    Pdot = (Jz*M_x + Jxz*M_z - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + Jxz*(Jx-Jy+Jz)*P*Q)/denom;
    Qdot = (M_y + (Jz-Jx)*P*R - Jxz*(P*P-R*R))/Jy;
    Rdot = (Jx*M_z + Jxz*M_x + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - Jxz*(Jx-Jy+Jz)*Q*R)/denom;
end