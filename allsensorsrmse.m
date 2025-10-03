% Ryland Ross
% 9/29/25
% Finds average RMSE over 100 runs for multiple 1D sensor configurations
clc; clear;

%% Simulation parameters
dt = 0.01;      % time step (s)
T = 5;          % total simulation time (s)
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
sigma_v  = 0.1;    % std dev of velocity sensor

%% Kalman Filter parameters
A = [1 dt; 0 1];          % state transition
B = [0.5*dt^2; dt];       % control input (acceleration)
Q = [0.01 0; 0 0.01];     % process noise covariance

%% Number of Monte Carlo runs
numRuns = 100;

%% Configurations
configs = 1:3;
config_names = {'Single Position', 'Two Positions', 'Two Positions + Velocity'};

%% Preallocate results
avg_rmse_pos = zeros(1,length(configs));
avg_rmse_vel = zeros(1,length(configs));
std_rmse_pos = zeros(1,length(configs));
std_rmse_vel = zeros(1,length(configs));

for idx = 1:length(configs)
    config = configs(idx);
    rmse_pos_all = zeros(1,numRuns);
    rmse_vel_all = zeros(1,numRuns);

    for run = 1:numRuns
        %% Generate noisy measurements for this run
        z_p1 = p_true + sigma_p1*randn(1,N);
        z_p2 = p_true + sigma_p2*randn(1,N);
        z_v  = v_true + sigma_v*randn(1,N);

        %% Setup measurement matrices based on configuration
        switch config
            case 1
                z = z_p1;
                H = [1 0];
                R = sigma_p1^2;
            case 2
                z = [z_p1; z_p2];
                H = [1 0; 1 0];
                R = diag([sigma_p1^2, sigma_p2^2]);
            case 3
                z = [z_p1; z_p2; z_v];
                H = [1 0; 1 0; 0 1];
                R = diag([sigma_p1^2, sigma_p2^2, sigma_v^2]);
        end

        %% Initialize filter
        x_est = zeros(2,N);
        P = eye(2);

        %% Kalman Filter loop
        for k = 2:N
            x_pred = A*x_est(:,k-1) + B*a_true;
            P_pred = A*P*A' + Q;

            K = P_pred*H'/(H*P_pred*H' + R);
            x_est(:,k) = x_pred + K*(z(:,k) - H*x_pred);
            P = (eye(2) - K*H)*P_pred;
        end

        %% Compute RMSE for this run
        rmse_pos_all(run) = sqrt(mean((x_est(1,:) - p_true).^2));
        rmse_vel_all(run) = sqrt(mean((x_est(2,:) - v_true).^2));
    end

    %% Store average and std RMSEs
    avg_rmse_pos(idx) = mean(rmse_pos_all);
    avg_rmse_vel(idx) = mean(rmse_vel_all);
    std_rmse_pos(idx) = std(rmse_pos_all);
    std_rmse_vel(idx) = std(rmse_vel_all);
end

%% Display results in a table
T = table(config_names', avg_rmse_pos', std_rmse_pos', avg_rmse_vel', std_rmse_vel', ...
    'VariableNames', {'Configuration','Avg_RMSE_Position','Std_RMSE_Position','Avg_RMSE_Velocity','Std_RMSE_Velocity'});
disp(T);
