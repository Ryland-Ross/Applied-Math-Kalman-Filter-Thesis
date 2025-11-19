% Ryland Ross
% Colgate University
% 11/19/25
% 2-D KF, EKF, UKF, and KF+Drag Comparison
% Projectile motion with quadratic air resistance

clc; clear; close all;

%% --- Simulation Parameters ---
Delta_t = 0.01;        % time step (s)
T = 7;                 % total time (s)
t = 0:Delta_t:T;
N = length(t);

%% --- Physical Parameters ---
g = 9.81;              % gravity (m/s^2)
c = 0.1;               % drag coefficient (1/m)
Q = 0.01 * eye(4);     % process noise covariance

%% --- Measurement Model ---
H = [1 0 0 0; 0 0 1 0];   % measure positions only
sigma_x = 0.5; sigma_y = 0.5; % measurement uncertainty
R = diag([sigma_x^2, sigma_y^2]);  % measurement noise

%% --- Initialization ---
x_hat_0_0 = [0; 30; 0; 25];   % [p_x0; v_x0; p_y0; v_y0]
P_0_0 = eye(4);

x_true = zeros(4,N);
x_true(:,1) = x_hat_0_0;
z = zeros(2,N);

%% --- True Nonlinear Dynamics ---
for k = 2:N
    p_x = x_true(1,k-1); v_x = x_true(2,k-1);
    p_y = x_true(3,k-1); v_y = x_true(4,k-1);
    v = sqrt(v_x^2 + v_y^2);

    x_true(1,k) = p_x + v_x*Delta_t;
    x_true(2,k) = v_x - c*v_x*v*Delta_t;
    x_true(3,k) = p_y + v_y*Delta_t;
    x_true(4,k) = v_y - (g + c*v_y*v)*Delta_t;
end

%% --- Monte Carlo Runs for Averaged RMSE ---
numRuns = 100;
RMSE_pos_all = zeros(2,4,numRuns); 
RMSE_vel_all = zeros(2,4,numRuns);

for runIdx = 1:numRuns
    % regenerate noise for every run so there is a difference
    z(1,:) = x_true(1,:) + sigma_x*randn(1,N);
    z(2,:) = x_true(3,:) + sigma_y*randn(1,N);

%% --- Linear KF with Gravity (ignoring drag) using control input ---
x_hat_KF = zeros(4,N);
x_hat_KF(:,1) = x_hat_0_0;
P_k_k = P_0_0;
u = [0; -g];
B = [0 0; Delta_t 0; 0 0; 0 Delta_t];
A = [1 Delta_t 0 0; 0 1 0 0; 0 0 1 Delta_t; 0 0 0 1];

for k = 2:N
    % --- Prediction Step ---
    x_prev = x_hat_KF(:,k-1);
    x_pred = A*x_prev + B*u;
    P_pred = A*P_k_k*A' + Q;

    % --- Update Step ---
    K_k = P_pred*H' / (H*P_pred*H' + R);
    x_hat_KF(:,k) = x_pred + K_k*(z(:,k) - H*x_pred);
    P_k_k = (eye(4) - K_k*H)*P_pred;
end

%% --- Extended Kalman Filter (Nonlinear with Drag) ---
x_hat_EKF = zeros(4,N); x_hat_EKF(:,1) = x_hat_0_0;
P_k_k = P_0_0;

for k = 2:N
    x_prev = x_hat_EKF(:,k-1);
    v_x = x_prev(2); v_y = x_prev(4);
    v = sqrt(v_x^2 + v_y^2);

    f = [x_prev(1) + v_x*Delta_t;
         v_x - c*v_x*v*Delta_t;
         x_prev(3) + v_y*Delta_t;
         v_y - (g + c*v_y*v)*Delta_t];

    F_k = [1, Delta_t, 0, 0;
           0, 1 - c*Delta_t*(v + (v_x^2)/v), 0, -c*Delta_t*v_x*v_y/v;
           0, 0, 1, Delta_t;
           0, -c*Delta_t*v_x*v_y/v, 0, 1 - c*Delta_t*(v + (v_y^2)/v)];

    P_pred = F_k*P_k_k*F_k' + Q;
    K_k = P_pred*H'/(H*P_pred*H' + R);
    x_hat_EKF(:,k) = f + K_k*(z(:,k) - H*f);
    P_k_k = (eye(4)-K_k*H)*P_pred;
end

%% --- Linear KF with Drag added manually ---
x_hat_KF_drag = zeros(4,N);
x_hat_KF_drag(:,1) = x_hat_0_0;
P_k_k = P_0_0;

for k = 2:N
    x_prev = x_hat_KF_drag(:,k-1);
    v_x = x_prev(2); v_y = x_prev(4);
    v = sqrt(v_x^2 + v_y^2);

    u_drag = [-c*v_x*v*Delta_t;
              -(g + c*v_y*v)*Delta_t];

    x_pred = [x_prev(1) + v_x*Delta_t;
              v_x + u_drag(1);
              x_prev(3) + v_y*Delta_t;
              v_y + u_drag(2)];

    P_pred = A*P_k_k*A' + Q;

    K_k = P_pred*H' / (H*P_pred*H' + R);
    x_hat_KF_drag(:,k) = x_pred + K_k*(z(:,k) - H*x_pred);
    P_k_k = (eye(4)-K_k*H)*P_pred;
end

%% --- Unscented Kalman Filter (UKF) ---
x_hat_UKF = zeros(4,N);
x_hat_UKF(:,1) = x_hat_0_0;
P_k_k = P_0_0;

alpha = 1e-3;    % small positive constant
beta = 2;        % optimal for Gaussian distributions
kappa = 0;       % secondary scaling parameter
n = 4;           % state dimension
lambda = alpha^2 * (n + kappa) - n;
Wm = [lambda / (n + lambda) repmat(1 / (2 * (n + lambda)), 1, 2*n)];
Wc = Wm; Wc(1) = Wc(1) + (1 - alpha^2 + beta);

for k = 2:N
    % --- Generate Sigma Points ---
    S = chol((n + lambda)*P_k_k, 'lower');
    X_sigma = [x_hat_UKF(:,k-1), x_hat_UKF(:,k-1) + S, x_hat_UKF(:,k-1) - S];

    % --- Propagate Through Nonlinear Dynamics ---
    X_pred = zeros(n, 2*n+1);
    for i = 1:(2*n+1)
        x_i = X_sigma(:,i);
        v_x = x_i(2); v_y = x_i(4);
        v = sqrt(v_x^2 + v_y^2);
        X_pred(:,i) = [x_i(1) + v_x*Delta_t;
                       v_x - c*v_x*v*Delta_t;
                       x_i(3) + v_y*Delta_t;
                       v_y - (g + c*v_y*v)*Delta_t];
    end

    % --- Predicted Mean and Covariance ---
    x_pred = X_pred * Wm';
    P_pred = Q;
    for i = 1:(2*n+1)
        diff = X_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i)*(diff*diff');
    end

    % --- Predict Measurements ---
    Z_sigma = H * X_pred;
    z_pred = Z_sigma * Wm';
    P_zz = R;
    P_xz = zeros(n, size(H,1));
    for i = 1:(2*n+1)
        P_zz = P_zz + Wc(i)*(Z_sigma(:,i) - z_pred)*(Z_sigma(:,i) - z_pred)';
        P_xz = P_xz + Wc(i)*(X_pred(:,i) - x_pred)*(Z_sigma(:,i) - z_pred)';
    end

    % --- Kalman Gain and Update ---
    K_k = P_xz / P_zz;
    x_hat_UKF(:,k) = x_pred + K_k*(z(:,k) - z_pred);
    P_k_k = P_pred - K_k*P_zz*K_k';
end

%% --- RMSE Calculations ---
RMSE_pos_all(:,1,runIdx) = sqrt(mean((x_hat_KF([1,3],:) - x_true([1,3],:)).^2,2));
RMSE_pos_all(:,2,runIdx) = sqrt(mean((x_hat_EKF([1,3],:) - x_true([1,3],:)).^2,2));
RMSE_pos_all(:,3,runIdx) = sqrt(mean((x_hat_KF_drag([1,3],:) - x_true([1,3],:)).^2,2));
RMSE_pos_all(:,4,runIdx) = sqrt(mean((x_hat_UKF([1,3],:) - x_true([1,3],:)).^2,2));

RMSE_vel_all(:,1,runIdx) = sqrt(mean((x_hat_KF([2,4],:) - x_true([2,4],:)).^2,2));
RMSE_vel_all(:,2,runIdx) = sqrt(mean((x_hat_EKF([2,4],:) - x_true([2,4],:)).^2,2));
RMSE_vel_all(:,3,runIdx) = sqrt(mean((x_hat_KF_drag([2,4],:) - x_true([2,4],:)).^2,2));
RMSE_vel_all(:,4,runIdx) = sqrt(mean((x_hat_UKF([2,4],:) - x_true([2,4],:)).^2,2));
end

RMSE_pos_avg = mean(RMSE_pos_all,3);
RMSE_vel_avg = mean(RMSE_vel_all,3);

fprintf('--- Average RMSE over %d runs ---\n', numRuns);
fprintf('Filter       | X Pos | Y Pos | X Vel | Y Vel\n');
fprintf('--------------------------------------------\n');
fprintf('KF           | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,1), RMSE_pos_avg(2,1), RMSE_vel_avg(1,1), RMSE_vel_avg(2,1));
fprintf('KF+Drag      | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,3), RMSE_pos_avg(2,3), RMSE_vel_avg(1,3), RMSE_vel_avg(2,3));
fprintf('EKF          | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,2), RMSE_pos_avg(2,2), RMSE_vel_avg(1,2), RMSE_vel_avg(2,2));
fprintf('UKF          | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,4), RMSE_pos_avg(2,4), RMSE_vel_avg(1,4), RMSE_vel_avg(2,4));

%% --- Plots: Position vs Time ---
figure;
subplot(2,1,1);
plot(t, x_true(1,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(1,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(1,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(1,:), 'r', 'LineWidth', 1.5);
plot(t, x_hat_UKF(1,:), 'm', 'LineWidth', 1.5);
scatter(t, z(1,:), 10, 'c', 'filled', 'MarkerFaceAlpha', 0.2);
xlabel('Time [s]'); ylabel('X Position [m]');
legend('True','KF','EKF','KF+Drag','UKF','Measurements'); grid on; title('X Position');

subplot(2,1,2);
plot(t, x_true(3,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(3,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(3,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(3,:), 'r', 'LineWidth', 1.5);
plot(t, x_hat_UKF(3,:), 'm', 'LineWidth', 1.5);
scatter(t, z(2,:), 10, 'c', 'filled', 'MarkerFaceAlpha', 0.2);
xlabel('Time [s]'); ylabel('Y Position [m]');
legend('True','KF','EKF','KF+Drag','UKF','Measurements'); grid on; title('Y Position');

%% --- Plots: Velocity vs Time ---
figure;
subplot(2,1,1);
plot(t, x_true(2,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(2,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(2,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(2,:), 'r', 'LineWidth', 1.5);
plot(t, x_hat_UKF(2,:), 'm', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('X Velocity [m/s]');
legend('True','KF','EKF','KF+Drag','UKF'); grid on; title('X Velocity');

subplot(2,1,2);
plot(t, x_true(4,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(4,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(4,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(4,:), 'r', 'LineWidth', 1.5);
plot(t, x_hat_UKF(4,:), 'm', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y Velocity [m/s]');
legend('True','KF','EKF','KF+Drag','UKF'); grid on; title('Y Velocity');

%% --- Trajectory Plot ---
figure;
plot(x_true(1,:), x_true(3,:), 'k', 'LineWidth', 2); hold on;
plot(x_hat_KF(1,:), x_hat_KF(3,:), 'b--', 'LineWidth', 1.5);
plot(x_hat_EKF(1,:), x_hat_EKF(3,:), 'g--', 'LineWidth', 1.5);
plot(x_hat_KF_drag(1,:), x_hat_KF_drag(3,:), 'r--', 'LineWidth', 1.5);
plot(x_hat_UKF(1,:), x_hat_UKF(3,:), 'm--', 'LineWidth', 1.5);
scatter(z(1,:), z(2,:), 15, 'c', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('X [m]'); ylabel('Y [m]');
legend('True','KF','EKF','KF+Drag','UKF','Measurements'); grid on;
title('Projectile Trajectory Comparison');

%% --- RMSE Comparison Bar Plot ---
figure;
filter_names = {'KF','EKF','KF+Drag','UKF'};
bar_data_pos = RMSE_pos_avg';
bar_data_vel = RMSE_vel_avg';
subplot(2,1,1);
bar(bar_data_pos);
set(gca,'XTickLabel',filter_names);
ylabel('RMSE [m]'); title('Position RMSE Comparison'); grid on;
legend('X','Y');

subplot(2,1,2);
bar(bar_data_vel);
set(gca,'XTickLabel',filter_names);
ylabel('RMSE [m/s]'); title('Velocity RMSE Comparison'); grid on;
legend('X','Y');