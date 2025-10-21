% Ryland Ross
% Colgate University
% 10/21/25
% 2-D EKF, KF, and KF+Drag Comparison
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
sigma_x = 0.5; sigma_y = 0.5; %measurement uncertainty
R = diag([sigma_x^2, sigma_y^2]);  % measurement noise

%% --- Initialization ---
x_hat_0_0 = [0; 30; 0; 25];   % [p_x0; v_x0; p_y0; v_y0] (m, m/s)
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

% perform 10 runs so RMSE can be averaged 
numRuns = 10;
RMSE_pos_all = zeros(2,3,numRuns); 
RMSE_vel_all = zeros(2,3,numRuns);

for runIdx = 1:numRuns
    % regenerate noise for every run so there is a difference
    z(1,:) = x_true(1,:) + sigma_x*randn(1,N);
    z(2,:) = x_true(3,:) + sigma_y*randn(1,N);

%% --- Linear KF with Gravity (ignoring drag) using control input ---
x_hat_KF = zeros(4,N);
x_hat_KF(:,1) = x_hat_0_0;
P_k_k = P_0_0;

% Control input: only gravity in y
u = [0; -g];

% maps acceleration to position & velocity
B = [0 0;
     Delta_t 0;
     0 0;
     0 Delta_t];

A = [1 Delta_t 0 0;
     0 1      0 0;
     0 0      1 Delta_t;
     0 0      0 1];

for k = 2:N
    % --- Prediction Step ---
    x_prev = x_hat_KF(:,k-1);
    x_pred = A*x_prev + B*u;

    % Covariance prediction
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

%% --- Linear KF with Drag added manually --

x_hat_KF_drag = zeros(4,N);    % preallocate state estimates
x_hat_KF_drag(:,1) = x_hat_0_0; % initialize with initial state
P_k_k = P_0_0;                 % initial covariance

for k = 2:N
    % Get previous state 
    x_prev = x_hat_KF_drag(:,k-1);
    v_x = x_prev(2); 
    v_y = x_prev(4);
    v = sqrt(v_x^2 + v_y^2);  % total speed magnitude
    
    % Compute drag input
    u_drag = [-c*v_x*v*Delta_t;   % change in x-velocity due to drag
              -(g + c*v_y*v)*Delta_t]; % change in y-velocity due to drag + gravity

    % Linear KF prediction (manual inclusion of drag)
    % x_pred = previous position + velocity*dt + drag contribution
    x_pred = [x_prev(1) + v_x*Delta_t;  % x-position update
              v_x + u_drag(1);          % x-velocity update
              x_prev(3) + v_y*Delta_t;  % y-position update
              v_y + u_drag(2)];         % y-velocity update

    % Covariance prediction 
    P_pred = A*P_k_k*A' + Q;

    % Kalman gain and update step
    K_k = P_pred*H' / (H*P_pred*H' + R);
    x_hat_KF_drag(:,k) = x_pred + K_k*(z(:,k) - H*x_pred);
    P_k_k = (eye(4)-K_k*H)*P_pred;
end


    RMSE_pos_all(:,1,runIdx) = sqrt(mean((x_hat_KF([1,3],:) - x_true([1,3],:)).^2,2));
    RMSE_pos_all(:,2,runIdx) = sqrt(mean((x_hat_EKF([1,3],:) - x_true([1,3],:)).^2,2));
    RMSE_pos_all(:,3,runIdx) = sqrt(mean((x_hat_KF_drag([1,3],:) - x_true([1,3],:)).^2,2));

    RMSE_vel_all(:,1,runIdx) = sqrt(mean((x_hat_KF([2,4],:) - x_true([2,4],:)).^2,2));
    RMSE_vel_all(:,2,runIdx) = sqrt(mean((x_hat_EKF([2,4],:) - x_true([2,4],:)).^2,2));
    RMSE_vel_all(:,3,runIdx) = sqrt(mean((x_hat_KF_drag([2,4],:) - x_true([2,4],:)).^2,2));
end

RMSE_pos_avg = mean(RMSE_pos_all,3); % average across runs
RMSE_vel_avg = mean(RMSE_vel_all,3);

fprintf('--- Average RMSE over %d runs ---\n', numRuns);
fprintf('Filter       | X Pos | Y Pos | X Vel | Y Vel\n');
fprintf('--------------------------------------------\n');
fprintf('KF           | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,1), RMSE_pos_avg(2,1), RMSE_vel_avg(1,1), RMSE_vel_avg(2,1));
fprintf('KF+Drag      | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,3), RMSE_pos_avg(2,3), RMSE_vel_avg(1,3), RMSE_vel_avg(2,3));
fprintf('EKF          | %.3f | %.3f | %.3f | %.3f\n', RMSE_pos_avg(1,2), RMSE_pos_avg(2,2), RMSE_vel_avg(1,2), RMSE_vel_avg(2,2));


%% --- Plots ---

% Position vs Time
figure;
subplot(2,1,1);
plot(t, x_true(1,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(1,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(1,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(1,:), 'r', 'LineWidth', 1.5);
scatter(t, z(1,:), 10, 'c', 'filled', 'MarkerFaceAlpha', 0.2);
xlabel('Time [s]'); ylabel('X Position [m]');
legend('True','KF','EKF','KF+Drag','Measurements'); grid on; title('X Position');

subplot(2,1,2);
plot(t, x_true(3,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(3,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(3,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(3,:), 'r', 'LineWidth', 1.5);
scatter(t, z(2,:), 10, 'm', 'filled', 'MarkerFaceAlpha', 0.2);
xlabel('Time [s]'); ylabel('Y Position [m]');
legend('True','KF','EKF','KF+Drag','Measurements'); grid on; title('Y Position');

% Velocity vs Time
figure;
subplot(2,1,1);
plot(t, x_true(2,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(2,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(2,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(2,:), 'r', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('X Velocity [m/s]');
legend('True','KF','EKF','KF+Drag'); grid on; title('X Velocity');

subplot(2,1,2);
plot(t, x_true(4,:), 'k', 'LineWidth', 2); hold on;
plot(t, x_hat_KF(4,:), 'b', 'LineWidth', 1.5);
plot(t, x_hat_EKF(4,:), 'g', 'LineWidth', 1.5);
plot(t, x_hat_KF_drag(4,:), 'r', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y Velocity [m/s]');
legend('True','KF','EKF','KF+Drag'); grid on; title('Y Velocity');

% Trajectory Plot
figure;
plot(x_true(1,:), x_true(3,:), 'k', 'LineWidth', 2); hold on;
plot(x_hat_KF(1,:), x_hat_KF(3,:), 'b--', 'LineWidth', 1.5);
plot(x_hat_EKF(1,:), x_hat_EKF(3,:), 'g--', 'LineWidth', 1.5);
plot(x_hat_KF_drag(1,:), x_hat_KF_drag(3,:), 'r--', 'LineWidth', 1.5);
scatter(z(1,:), z(2,:), 15, 'c', 'filled', 'MarkerFaceAlpha', 0.1);
xlabel('X [m]'); ylabel('Y [m]');
legend('True','KF','EKF','KF+Drag','Measurements'); grid on;
title('Projectile Trajectory Comparison');
