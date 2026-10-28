function [vt, alpha, beta, vtBody, vwBody] = reconstruct_airdata_from_state_wind(state, vwNav)
% Reconstruct air-relative parameters from body state and horizontal wind.

state = reshape(state, [12, 1]);
vwNav = reshape(vwNav, [2, 1]);

phi = state(4);
theta = state(5);
psi = state(6);
u = state(7);
v = state(8);
w = state(9);

Cnb = [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta); ...
       sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi), ...
       sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi), ...
       sin(phi) * cos(theta); ...
       cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi), ...
       cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi), ...
       cos(phi) * cos(theta)];

vwBody = Cnb * [vwNav(1); vwNav(2); 0];
vtBody = [u; v; w] - vwBody;

vt = max(norm(vtBody), 1e-6);
alpha = atan2(vtBody(3), vtBody(1));
beta = asin(max(min(vtBody(2) / vt, 1), -1));
end
