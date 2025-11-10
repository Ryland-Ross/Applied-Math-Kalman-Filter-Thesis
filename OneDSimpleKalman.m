% Ryland Ross
% 9/22/25
% 1-D Kalman Filter Implementation (with RMSE averaged over 10 runs)

% Imagine you have a car moving along the x-axis. A sensor measures its
% position every 0.1 seconds, and the sensor has an uncertainty (Gaussian
% noise) of sigma = 0.5 meters. The vehicle is constantly accelerating at
% 0.2 m/s^2.

% Goal: Estimate the position and velocity of the car at every time step,
% and compute the average RMSE of both position and velocity across 100 runs.

%% Initialize parameters
clc; clear; close all; 

dt = 0.01; % sensor time step (s)
T = 30;    % total simulation time (s)
t = 0:dt:T; % discrete time vector from 0 to T with step size dt
N = length(t); % total number of time steps

%% True motion (ground truth car trajectory)
a_true = 0.2; % constant, known acceleration (m/s^2)
p0 = 0;       % initial position (m)
v0 = 1;       % initial velocity (m)

% Ground truth using kinematic equations
p_true = p0 + v0*t + 0.5*a_true*t.^2; % position over time
v_true = v0 + a_true*t;               % velocity over time

%% Measurement model
sigma_meas = 0.5; % measurement noise standard deviation (m)

%% Monte Carlo: Run filter multiple times to average RMSE
numRuns = 100;                % number of independent runs
rmse_pos_all = zeros(1,numRuns); % store RMSE for position
rmse_vel_all = zeros(1,numRuns); % store RMSE for velocity

for run = 1:numRuns
    
    % Generate noisy position measurements for this run
    % (true position + Gaussian noise)
    z = p_true + sigma_meas*randn(size(t));
    
    % Kalman Filter matrices
    A = [1 dt; 0 1];          % state transition
    B = [0.5*dt^2; dt];       % control (acceleration input)
    H = [1 0];                % measurement: position only
    Q = [0.01 0; 0 0.01];     % process noise covariance
    R = sigma_meas^2;         % measurement noise covariance

    % Initialize estimates
    x_est = zeros(2,N);       % [position; velocity] estimates
    P = eye(2);               % initial covariance

    % Main Kalman filter loop through all time steps
    for k = 2:N
        % --- Prediction step ---
        x_pred = A*x_est(:,k-1) + B*a_true; % predict state
        P_pred = A*P*A' + Q;                % predict covariance

        % --- Update step ---
        K = P_pred*H'/(H*P_pred*H' + R);                 % Kalman gain
        x_est(:,k) = x_pred + K*(z(k) - H*x_pred);       % update estimate
        P = (eye(2) - K*H)*P_pred;                       % update covariance
    end
    
    % Compute RMSE for position and velocity estimates in this run
    rmse_pos_all(run) = sqrt(mean((x_est(1,:) - p_true).^2));
    rmse_vel_all(run) = sqrt(mean((x_est(2,:) - v_true).^2));
    
    % Store last run’s results for plotting
    if run == numRuns
        z_last = z;
        x_est_last = x_est;
    end
end

%% Average RMSE results
rmse_pos_avg = mean(rmse_pos_all);
rmse_vel_avg = mean(rmse_vel_all);

fprintf('Average Position RMSE over %d runs = %.4f meters\n', numRuns, rmse_pos_avg);
fprintf('Average Velocity RMSE over %d runs = %.4f m/s\n', numRuns, rmse_vel_avg);

%% Plots (only from the last run to visualize one trajectory)
% --- Position subplot ---
subplot(2,1,1)
plot(t, p_true,'k','LineWidth',2); hold on;   % true trajectory
plot(t, z_last,'r.');                         % noisy measurements
plot(t, x_est_last(1,:),'b','LineWidth',1.5); % KF estimated position
legend('True Position','Measurements','KF Estimate')
xlabel('Time [s]'); ylabel('Position [m]'); grid on;

% --- Velocity subplot ---
subplot(2,1,2)
plot(t, v_true,'k','LineWidth',2); hold on;   % true velocity
plot(t, x_est_last(2,:),'b','LineWidth',1.5); % KF estimated velocity
legend('True Velocity','KF Estimate')
xlabel('Time [s]'); ylabel('Velocity [m/s]'); grid on;
