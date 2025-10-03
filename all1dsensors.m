% Ryland Ross
% 9/29/25
% 1-D Kalman Filter with Multiple Sensors (Single, Two, and Velocity Sensors)
% Prompts user to select sensor configuration at runtime
clc; clear; close all;

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

%% Generate noisy measurements
z_p1 = p_true + sigma_p1*randn(1,N);
z_p2 = p_true + sigma_p2*randn(1,N);
z_v  = v_true + sigma_v*randn(1,N);

%% Prompt user to choose measurement configuration
fprintf('Select sensor configuration:\n');
fprintf('1 = Single position sensor\n');
fprintf('2 = Two position sensors\n');
fprintf('3 = Two positions + velocity sensor\n');
config = input('Enter configuration number (1, 2, or 3): ');

switch config
    case 1  % single position sensor
        z = z_p1;
        H = [1 0];
        R = sigma_p1^2;
    case 2  % two position sensors
        z = [z_p1; z_p2];
        H = [1 0; 1 0];
        R = diag([sigma_p1^2, sigma_p2^2]);
    case 3  % two position + velocity sensor
        z = [z_p1; z_p2; z_v];
        H = [1 0; 1 0; 0 1];
        R = diag([sigma_p1^2, sigma_p2^2, sigma_v^2]);
    otherwise
        error('Invalid configuration selected. Please enter 1, 2, or 3.');
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

%% Compute RMSE
rmse_pos = sqrt(mean((x_est(1,:) - p_true).^2));
rmse_vel = sqrt(mean((x_est(2,:) - v_true).^2));

fprintf('Configuration %d RMSE Position: %.4f m\n', config, rmse_pos);
fprintf('Configuration %d RMSE Velocity: %.4f m/s\n', config, rmse_vel);

%% Plot results
figure;

% Position plot
subplot(2,1,1)
plot(t, p_true,'k','LineWidth',2); hold on
if config == 1
    plot(t, z_p1,'r.');
elseif config == 2
    plot(t, z_p1,'r.'); 
    plot(t, z_p2,'g.');
elseif config == 3
    plot(t, z_p1,'r.'); 
    plot(t, z_p2,'g.');
end
plot(t, x_est(1,:),'b','LineWidth',1.5);
legend_entries = {'True Position'};
if config == 1
    legend_entries = [legend_entries, 'Position Measurement'];
elseif config == 2
    legend_entries = [legend_entries, 'Pos Sensor 1','Pos Sensor 2'];
elseif config == 3
    legend_entries = [legend_entries, 'Pos Sensor 1','Pos Sensor 2'];
end
legend_entries = [legend_entries, 'KF Estimate'];
legend(legend_entries)
xlabel('Time [s]'); ylabel('Position [m]'); grid on

% Velocity plot
subplot(2,1,2)
plot(t, v_true,'k','LineWidth',2); hold on
if config == 3
    plot(t, z_v,'m.');
end
plot(t, x_est(2,:),'b','LineWidth',1.5);
legend_entries = {'True Velocity'};
if config == 3
    legend_entries = [legend_entries, 'Velocity Measurement'];
end
legend_entries = [legend_entries, 'KF Estimate'];
legend(legend_entries)
xlabel('Time [s]'); ylabel('Velocity [m/s]'); grid on
