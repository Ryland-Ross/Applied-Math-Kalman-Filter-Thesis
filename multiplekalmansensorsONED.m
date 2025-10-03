% Ryland Ross
% 9/29/25
% 1-D Kalman Filter with Multiple Sensors. Uses similar framework to my
% other 1D ones, just adjusts z, H, and R basically
clc; clear; close all;

%% Simulation parameters
dt = 0.01;      % time step (s)
T = 5;         % total simulation time (s)
t = 0:dt:T;     
N = length(t);

%% True motion
a_true = 0.2;   % constant acceleration (m/s^2)
p0 = 0; v0 = 1; % initial position and velocity

p_true = p0 + v0*t + 0.5*a_true*t.^2;
v_true = v0 + a_true*t;

%% Sensor noise parameters
sigma_p1 = 0.5;    % std dev of first position sensor
sigma_p2 = 0.7;    % std dev of second position sensor
sigma_v  = 0.3;    % std dev of velocity sensor

%% Generate noisy measurements
z_p1 = p_true + sigma_p1*randn(1,N);
z_p2 = p_true + sigma_p2*randn(1,N);
z_v  = v_true + sigma_v*randn(1,N);

%% Choose measurement configuration
% Options:
% 1 = two position sensors only
% 2 = two positions + velocity sensor
config = 2;  %PUT CHOICE HERE

switch config %sets z, H, R depending on how many sensors I'm using
    case 1
        z = [z_p1; z_p2];
        H = [1 0; 1 0];
        R = diag([sigma_p1^2, sigma_p2^2]);
    case 2
        z = [z_p1; z_p2; z_v];
        H = [1 0; 1 0; 0 1];
        R = diag([sigma_p1^2, sigma_p2^2, sigma_v^2]);
    otherwise
        error('Invalid configuration selected');
end

%% Kalman Filter parameters
A = [1 dt; 0 1];          % state transition
B = [0.5*dt^2; dt];       % control input (acceleration)
Q = [0.01 0; 0 0.01];     % process noise covariance
x_est = zeros(2,N);       % state estimates
P = eye(2);               

%% Kalman Filter loop
for k = 2:N
    % Prediction
    x_pred = A*x_est(:,k-1) + B*a_true;
    P_pred = A*P*A' + Q;

    % Update
    K = P_pred*H'/(H*P_pred*H' + R);
    x_est(:,k) = x_pred + K*(z(:,k) - H*x_pred);
    P = (eye(2) - K*H)*P_pred;
end

%% Plot results
figure;

% Position plot
subplot(2,1,1)
plot(t, p_true,'k','LineWidth',2); hold on
if config == 1 || config == 2
    plot(t, z_p1,'r.'); 
    plot(t, z_p2,'g.');
end
plot(t, x_est(1,:),'b','LineWidth',1.5);
legend_entries = {'True Position'};
if config == 1 || config == 2
    legend_entries = [legend_entries, 'Pos Sensor 1','Pos Sensor 2'];
end
legend_entries = [legend_entries, 'KF Estimate'];
legend(legend_entries)
xlabel('Time [s]'); ylabel('Position [m]'); grid on

% Velocity plot
subplot(2,1,2)
plot(t, v_true,'k','LineWidth',2); hold on
if config == 2
   plot(t, z_v,'m.');
end
plot(t, x_est(2,:),'b','LineWidth',1.5);
legend_entries = {'True Velocity'};
if config == 2
    legend_entries = [legend_entries, 'Velocity Measurement'];
end
legend_entries = [legend_entries, 'KF Estimate'];
legend(legend_entries)
xlabel('Time [s]'); ylabel('Velocity [m/s]'); grid on

%% Compute estimation errors
pos_error = x_est(1,:) - p_true; % position error over time
vel_error = x_est(2,:) - v_true; % velocity error over time

% Root Mean Squared Error (RMSE)
rmse_pos = sqrt(mean(pos_error.^2));
rmse_vel = sqrt(mean(vel_error.^2));

fprintf('Position RMSE: %.4f m\n', rmse_pos);
fprintf('Velocity RMSE: %.4f m/s\n', rmse_vel);

