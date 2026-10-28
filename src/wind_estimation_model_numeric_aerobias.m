function y = wind_estimation_model_numeric_aerobias(state, controls, Vw, coeffScale)
% Nominal F-16 observation model with multiplicative aerodynamic coefficient bias.
% coeffScale order: [Cx, Cy, Cz, Cl, Cm, Cn].

if nargin < 4 || isempty(coeffScale)
    coeffScale = ones(6, 1);
end
coeffScale = double(coeffScale(:));
if isscalar(coeffScale)
    coeffScale = coeffScale * ones(6, 1);
end
if numel(coeffScale) ~= 6
    error('coeffScale must be scalar or a 6-element vector.');
end

phi = state(4);
theta = state(5);
psi = state(6);
U = state(7);
V = state(8);
W = state(9);
P = state(10);
Q = state(11);
R = state(12);
alt = state(3);

T = controls(1);
el = controls(2);
ail = controls(3);
rud = controls(4);

g = 32.17;
m = 636.94;
B = 30.0;
S = 300.0;
cbar = 11.32;

Jx = 9496.0;
Jy = 55814.0;
Jz = 63100.0;
Jxz = 982.0;

C_N_B = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta); ...
         sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), ...
         sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), ...
         sin(phi)*cos(theta); ...
         cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi), ...
         cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi), ...
         cos(phi)*cos(theta)];

Vw_B = C_N_B * [Vw(1); Vw(2); 0];
Vt_B = [U; V; W] - Vw_B;

V_T = max(norm(Vt_B), 1e-6);
alpha = atan2(Vt_B(3), Vt_B(1));
beta = asin(max(min(Vt_B(2) / V_T, 1), -1));
alpha_deg = alpha * 180 / pi;
beta_deg = beta * 180 / pi;

[Cx_tot, Cy_tot, Cz_tot, Cl_tot, Cm_tot, Cn_tot] = ...
    f16_lofi_aerodynamic_coeffs(alpha_deg, beta_deg, el, ail, rud, P, Q, R, V_T);

coeff = [Cx_tot; Cy_tot; Cz_tot; Cl_tot; Cm_tot; Cn_tot] .* coeffScale;
Cx_tot = coeff(1);
Cy_tot = coeff(2);
Cz_tot = coeff(3);
Cl_tot = coeff(4);
Cm_tot = coeff(5);
Cn_tot = coeff(6);

[~, qbar, ~] = atmos_model(alt, V_T);

F_x = qbar * S * Cx_tot + T;
F_y = qbar * S * Cy_tot;
F_z = qbar * S * Cz_tot;

M_x = Cl_tot * qbar * S * B;
M_y = Cm_tot * qbar * S * cbar;
M_z = Cn_tot * qbar * S * B;

Udot = R * V - Q * W - g * sin(theta) + F_x / m;
Vdot = P * W - R * U + g * cos(theta) * sin(phi) + F_y / m;
Wdot = Q * U - P * V + g * cos(theta) * cos(phi) + F_z / m;

denom = Jx * Jz - Jxz * Jxz;
Pdot = (Jz * M_x + Jxz * M_z - ...
    (Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + ...
    Jxz * (Jx - Jy + Jz) * P * Q) / denom;
Qdot = (M_y + (Jz - Jx) * P * R - Jxz * (P * P - R * R)) / Jy;
Rdot = (Jx * M_z + Jxz * M_x + ...
    (Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - ...
    Jxz * (Jx - Jy + Jz) * Q * R) / denom;

y = [Udot; Vdot; Wdot; Pdot; Qdot; Rdot];
end
