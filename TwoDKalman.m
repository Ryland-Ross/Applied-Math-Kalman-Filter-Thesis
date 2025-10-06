% Ryland Ross
% 9/30/25
% 2-D Kalman Filter with Multiple Sensors (x+y positions, x+y positions+velocities, double x+y positions+velocities)

clc; clear all; close all;

%% Simulation parameters
dt = 0.01;       % time step (s)
T = 8.156;       % total simulation time (s)
t = 0:dt:T;      
N = length(t);

%% True initial conditions
p0 = [0; 0];     % initial positions [px0; py0]
v0 = [10; 40];   % initial velocities [vx0; vy0] 

%% Accelerations to test
accels = [0 2];  % horizontal acceleration: 0 (free-fall) and 2 m/s^2 (thrust)
ay = -9.81;      % vertical acceleration (gravity)

%% Sensor noise parameters
sigma_px1 = 0.5; sigma_px2 = 0.3;  % horizontal position sensors
sigma_py1 = 0.5; sigma_py2 = 0.3;  % vertical position sensors
sigma_vx  = 0.1; sigma_vy  = 0.1;  % velocity sensors

%% Kalman Filter parameters
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1];   % state transition
B = [0.5*dt^2 0; dt 0; 0 0.5*dt^2; 0 dt];    % control input
Q = 0.01*eye(4);                              % process noise

%% Number of Monte Carlo runs
numRuns = 10;

%% Configurations
configs = 1:3;
config_names = {'Position only','Pos + Vel','Double Pos + Vel'};

%% Preallocate RMSE results
avg_rmse_px = zeros(length(accels), length(configs));
avg_rmse_py = zeros(length(accels), length(configs));
avg_rmse_vx = zeros(length(accels), length(configs));
avg_rmse_vy = zeros(length(accels), length(configs));

%% Loop over accelerations and measurement configurations
for a_idx = 1:length(accels)
    ax = accels(a_idx);
   
    % True trajectories
    px_true = p0(1) + v0(1)*t + 0.5*ax*t.^2;
    vx_true = v0(1) + ax*t;
    py_true = p0(2) + v0(2)*t + 0.5*ay*t.^2;
    vy_true = v0(2) + ay*t;
   
    for cfg_idx = 1:length(configs)
        cfg = configs(cfg_idx);
        rmse_px_all = zeros(1,numRuns);
        rmse_py_all = zeros(1,numRuns);
        rmse_vx_all = zeros(1,numRuns);
        rmse_vy_all = zeros(1,numRuns);
       
        for run = 1:numRuns
            %% Generate noisy measurements
            z_px1 = px_true + sigma_px1*randn(1,N);
            z_px2 = px_true + sigma_px2*randn(1,N);
            z_py1 = py_true + sigma_py1*randn(1,N);
            z_py2 = py_true + sigma_py2*randn(1,N);
            z_vx  = vx_true + sigma_vx*randn(1,N);
            z_vy  = vy_true + sigma_vy*randn(1,N);
           
            %% Measurement matrices
            switch cfg
                case 1 % Position only
                    z = [z_px1; z_py1];
                    H = [1 0 0 0; 0 0 1 0];
                    R = diag([sigma_px1^2, sigma_py1^2]);
                case 2 % Position + velocity
                    z = [z_px1; z_vx; z_py1; z_vy];
                    H = eye(4);
                    R = diag([sigma_px1^2, sigma_vx^2, sigma_py1^2, sigma_vy^2]);
                case 3 % Double position + velocity
                    z = [z_px1; z_vx; z_px2; z_vx; z_py1; z_vy; z_py2; z_vy];
                    H = [1 0 0 0; 0 1 0 0; 1 0 0 0; 0 1 0 0;
                         0 0 1 0; 0 0 0 1; 0 0 1 0; 0 0 0 1];
                    R = diag([sigma_px1^2, sigma_vx^2, sigma_px2^2, sigma_vx^2, ...
                              sigma_py1^2, sigma_vy^2, sigma_py2^2, sigma_vy^2]);
            end
           
            %% Initialize Kalman filter
            x_est = zeros(4,N);
            x_est(:,1) = [p0(1); v0(1); p0(2); v0(2)];
            P = eye(4);
           
            %% Kalman filter loop
            for k = 2:N
                % Prediction
                u = [ax; ay];
                x_pred = A*x_est(:,k-1) + B*u;
                P_pred = A*P*A' + Q;
               
                % Update
                K = P_pred*H'/(H*P_pred*H' + R);
                x_est(:,k) = x_pred + K*(z(:,k) - H*x_pred);
                P = (eye(size(K,1)) - K*H)*P_pred;
            end
           
            %% Compute RMSE
            rmse_px_all(run) = sqrt(mean((x_est(1,:) - px_true).^2));
            rmse_py_all(run) = sqrt(mean((x_est(3,:) - py_true).^2));
            rmse_vx_all(run) = sqrt(mean((x_est(2,:) - vx_true).^2));
            rmse_vy_all(run) = sqrt(mean((x_est(4,:) - vy_true).^2));
        end
       
        %% Store average RMSE
        avg_rmse_px(a_idx,cfg_idx) = mean(rmse_px_all);
        avg_rmse_py(a_idx,cfg_idx) = mean(rmse_py_all);
        avg_rmse_vx(a_idx,cfg_idx) = mean(rmse_vx_all);
        avg_rmse_vy(a_idx,cfg_idx) = mean(rmse_vy_all);

        %% ---- Visualization for this configuration ----
        figure('Name',sprintf('Kalman Filter: %s (ax=%.1f)', config_names{cfg_idx}, ax),...
               'NumberTitle','off');

        %% 3D Trajectory
        subplot(3,1,1);
        plot3(px_true, py_true, t, 'k', 'LineWidth', 3); hold on;
        plot3(x_est(1,:), x_est(3,:), t, 'r', 'LineWidth', 2.2);
        plot3(z_px1, z_py1, t, '.', 'Color', [0 0.6 1], 'MarkerSize', 5);
        xlabel('X Position [m]'); ylabel('Y Position [m]'); zlabel('Time [s]');
        title(sprintf('3D Trajectory - %s', config_names{cfg_idx}), 'FontWeight','bold');
        legend('True trajectory','Kalman estimate','Measurements','Location','best');
        grid on; view(35,25);

        %% Position vs Time
        subplot(3,1,2);
        hold on; grid on;
        plot(t, px_true, 'k', 'LineWidth', 3);
        plot(t, py_true, 'k--', 'LineWidth', 3);
        plot(t, x_est(1,:), 'r', 'LineWidth', 2.2);
        plot(t, x_est(3,:), 'b', 'LineWidth', 2.2);
        plot(t, z_px1, '.', 'Color', [0 0.6 1], 'MarkerSize', 5);
        plot(t, z_py1, '.', 'Color', [0.3 0.8 0.3], 'MarkerSize', 5);
        xlabel('Time [s]'); ylabel('Position [m]');
        legend('True x','True y','Estimated x','Estimated y','Measured x','Measured y','Location','best');
        title(sprintf('Position vs Time - %s', config_names{cfg_idx}), 'FontWeight','bold');

        %% Velocity vs Time
        subplot(3,1,3);
        hold on; grid on;
        plot(t, vx_true, 'k', 'LineWidth', 3);
        plot(t, vy_true, 'k--', 'LineWidth', 3);
        plot(t, x_est(2,:), 'r', 'LineWidth', 2.2);
        plot(t, x_est(4,:), 'b', 'LineWidth', 2.2);
        plot(t, z_vx, '.', 'Color', [0 0.6 1], 'MarkerSize', 5);
        plot(t, z_vy, '.', 'Color', [0.3 0.8 0.3], 'MarkerSize', 5);
        xlabel('Time [s]'); ylabel('Velocity [m/s]');
        legend('True vx','True vy','Estimated vx','Estimated vy','Measured vx','Measured vy','Location','best');
        title(sprintf('Velocity vs Time - %s', config_names{cfg_idx}), 'FontWeight','bold');
    end
end

%% Display results table
results = table;
for a_idx = 1:length(accels)
    for cfg_idx = 1:length(configs)
        results = [results; {accels(a_idx), config_names{cfg_idx}, ...
            avg_rmse_px(a_idx,cfg_idx), avg_rmse_py(a_idx,cfg_idx), ...
            avg_rmse_vx(a_idx,cfg_idx), avg_rmse_vy(a_idx,cfg_idx)}];
    end
end
results.Properties.VariableNames = {'Horizontal Acceleration (m/s^2)','Configuration','Avg_RMSE_px','Avg_RMSE_py','Avg_RMSE_vx','Avg_RMSE_vy'};
disp(results);
