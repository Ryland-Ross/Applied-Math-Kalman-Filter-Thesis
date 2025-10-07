%% 3D Kalman Filter with Multiple Sensors
% Ryland Ross
% 10/6/25

clc; clear all; close all;

%% Simulation parameters
dt = 0.01;       % time step (s)
T = 5;       % total simulation time (s)
t = 0:dt:T;
N = length(t);

%% True initial conditions
p0 = [0; 0; 100];    % initial positions [px0; py0; pz0]
v0 = [20; 10; 0];     % initial velocities [vx0; vy0; vz0]

%% Accelerations
ax = 10;
ay = 6;
az = -9.81;

%% Sensor noise parameters
sigma_px1 = 0.5; sigma_px2 = 0.3;
sigma_py1 = 0.5; sigma_py2 = 0.3;
sigma_pz1 = 0.5; sigma_pz2 = 0.3;

sigma_vx = 0.1; sigma_vy = 0.1; sigma_vz = 0.1;

%% Kalman Filter parameters
A = [1 dt 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 dt 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 dt;
     0 0 0 0 0 1];

B = [0.5*dt^2 0 0;
     dt 0 0;
     0 0.5*dt^2 0;
     0 dt 0;
     0 0 0.5*dt^2;
     0 0 dt];

Q = 0.01*eye(6); % process noise

%% Number of Monte Carlo runs
numRuns = 40;

%% Measurement configurations
configs = 1:3;
config_names = {'Position only','Pos + Vel','Double Pos + Vel'};

%% Preallocate RMSE results
avg_rmse_px = zeros(length(configs),1);
avg_rmse_py = zeros(length(configs),1);
avg_rmse_pz = zeros(length(configs),1);
avg_rmse_vx = zeros(length(configs),1);
avg_rmse_vy = zeros(length(configs),1);
avg_rmse_vz = zeros(length(configs),1);

%% True trajectories
px_true = p0(1) + v0(1)*t + 0.5*ax*t.^2;
vx_true = v0(1) + ax*t;

py_true = p0(2) + v0(2)*t + 0.5*ay*t.^2;
vy_true = v0(2) + ay*t;

pz_true = p0(3) + v0(3)*t + 0.5*az*t.^2;
vz_true = v0(3) + az*t;

%% Loop over configurations
for cfg_idx = 1:length(configs)
    cfg = configs(cfg_idx);
    
    rmse_px_all = zeros(1,numRuns);
    rmse_py_all = zeros(1,numRuns);
    rmse_pz_all = zeros(1,numRuns);
    rmse_vx_all = zeros(1,numRuns);
    rmse_vy_all = zeros(1,numRuns);
    rmse_vz_all = zeros(1,numRuns);
    
    for run = 1:numRuns
        %% Generate noisy measurements (same as before)
        z_px1 = px_true + sigma_px1*randn(1,N);
        z_px2 = px_true + sigma_px2*randn(1,N);
        z_py1 = py_true + sigma_py1*randn(1,N);
        z_py2 = py_true + sigma_py2*randn(1,N);
        z_pz1 = pz_true + sigma_pz1*randn(1,N);
        z_pz2 = pz_true + sigma_pz2*randn(1,N);
        
        z_vx1 = vx_true + sigma_vx*randn(1,N);
        z_vx2 = vx_true + sigma_vx*randn(1,N);
        z_vy1 = vy_true + sigma_vy*randn(1,N);
        z_vy2 = vy_true + sigma_vy*randn(1,N);
        z_vz1 = vz_true + sigma_vz*randn(1,N);
        z_vz2 = vz_true + sigma_vz*randn(1,N);
        
        %% Measurement matrices
        switch cfg
            case 1
                z = [z_px1; z_py1; z_pz1];
                H = [1 0 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 0 1 0];
                R = diag([sigma_px1^2, sigma_py1^2, sigma_pz1^2]);
            case 2
                z = [z_px1; z_vx1; z_py1; z_vy1; z_pz1; z_vz1];
                H = eye(6);
                R = diag([sigma_px1^2, sigma_vx^2, sigma_py1^2, sigma_vy^2, sigma_pz1^2, sigma_vz^2]);
            case 3
                z = [z_px1; z_vx1; z_px2; z_vx2; z_py1; z_vy1; z_py2; z_vy2; z_pz1; z_vz1; z_pz2; z_vz2];
                H = [1 0 0 0 0 0;  % px1
                     0 1 0 0 0 0;  % vx1
                     1 0 0 0 0 0;  % px2
                     0 1 0 0 0 0;  % vx2
                     0 0 1 0 0 0;  % py1
                     0 0 0 1 0 0;  % vy1
                     0 0 1 0 0 0;  % py2
                     0 0 0 1 0 0;  % vy2
                     0 0 0 0 1 0;  % pz1
                     0 0 0 0 0 1;  % vz1
                     0 0 0 0 1 0;  % pz2
                     0 0 0 0 0 1]; % vz2
                R = diag([sigma_px1^2,sigma_vx^2,sigma_px2^2,sigma_vx^2,...
                          sigma_py1^2,sigma_vy^2,sigma_py2^2,sigma_vy^2,...
                          sigma_pz1^2,sigma_vz^2,sigma_pz2^2,sigma_vz^2]);
        end
        
        %% Initialize Kalman filter
        x_est = zeros(6,N);
        x_est(:,1) = [p0(1); v0(1); p0(2); v0(2); p0(3); v0(3)];
        P = eye(6);
        
        %% Kalman filter loop
        for k = 2:N
            u = [ax; ay; az];
            x_pred = A*x_est(:,k-1) + B*u;
            P_pred = A*P*A' + Q;
            
            K = P_pred*H'/(H*P_pred*H' + R);
            x_est(:,k) = x_pred + K*(z(:,k) - H*x_pred);
            P = (eye(size(K,1)) - K*H)*P_pred;
        end
        
        %% Compute RMSE
        rmse_px_all(run) = sqrt(mean((x_est(1,:) - px_true).^2));
        rmse_vx_all(run) = sqrt(mean((x_est(2,:) - vx_true).^2));
        rmse_py_all(run) = sqrt(mean((x_est(3,:) - py_true).^2));
        rmse_vy_all(run) = sqrt(mean((x_est(4,:) - vy_true).^2));
        rmse_pz_all(run) = sqrt(mean((x_est(5,:) - pz_true).^2));
        rmse_vz_all(run) = sqrt(mean((x_est(6,:) - vz_true).^2));
        
    end
    
    %% Store average RMSE
    avg_rmse_px(cfg_idx) = mean(rmse_px_all);
    avg_rmse_py(cfg_idx) = mean(rmse_py_all);
    avg_rmse_pz(cfg_idx) = mean(rmse_pz_all);
    avg_rmse_vx(cfg_idx) = mean(rmse_vx_all);
    avg_rmse_vy(cfg_idx) = mean(rmse_vy_all);
    avg_rmse_vz(cfg_idx) = mean(rmse_vz_all);
    
   %% --- Plotting: Subplots for Position and Velocity ---
figure('Name',sprintf('Kalman Filter - %s', config_names{cfg_idx}),'NumberTitle','off');

% Position vs Time
subplot(2,1,1); hold on; grid on;
plot(t, px_true,'r','LineWidth',2);        % True x
plot(t, py_true,'b','LineWidth',2);        % True y
plot(t, pz_true,'g','LineWidth',2);        % True z
plot(t, x_est(1,:),'r--','LineWidth',1.5); % Estimated x
plot(t, x_est(3,:),'b--','LineWidth',1.5); % Estimated y
plot(t, x_est(5,:),'g--','LineWidth',1.5); % Estimated z

% Measurements
if cfg == 3
    plot(t, z_px1,'c.','MarkerSize',5); plot(t, z_px2,'m.','MarkerSize',5);
    plot(t, z_py1,'y.','MarkerSize',5); plot(t, z_py2,'k.','MarkerSize',5);
    plot(t, z_pz1,'b.','MarkerSize',5); plot(t, z_pz2,'r.','MarkerSize',5);
else
    plot(t, z_px1,'c.','MarkerSize',5);
    plot(t, z_py1,'m.','MarkerSize',5);
    plot(t, z_pz1,'k.','MarkerSize',5);
end

xlabel('Time [s]'); ylabel('Position [m]');
title('Position vs Time'); 
if cfg == 3
    legend('True x','True y','True z','Estimated x','Estimated y','Estimated z',...
           'Measured x1','Measured x2','Measured y1','Measured y2','Measured z1','Measured z2');
else
    legend('True x','True y','True z','Estimated x','Estimated y','Estimated z',...
           'Measured x','Measured y','Measured z');
end

% Velocity vs Time
subplot(2,1,2); hold on; grid on;
plot(t, vx_true,'r','LineWidth',2);        % True vx
plot(t, vy_true,'b','LineWidth',2);        % True vy
plot(t, vz_true,'g','LineWidth',2);        % True vz
plot(t, x_est(2,:),'r--','LineWidth',1.5); % Estimated vx
plot(t, x_est(4,:),'b--','LineWidth',1.5); % Estimated vy
plot(t, x_est(6,:),'g--','LineWidth',1.5); % Estimated vz

% Velocity measurements
if cfg == 3
    plot(t, z_vx1,'c.','MarkerSize',5); plot(t, z_vx2,'m.','MarkerSize',5);
    plot(t, z_vy1,'y.','MarkerSize',5); plot(t, z_vy2,'k.','MarkerSize',5);
    plot(t, z_vz1,'b.','MarkerSize',5); plot(t, z_vz2,'r.','MarkerSize',5);
else
    plot(t, z_vx1,'c.','MarkerSize',5);
    plot(t, z_vy1,'m.','MarkerSize',5);
    plot(t, z_vz1,'k.','MarkerSize',5);
end

xlabel('Time [s]'); ylabel('Velocity [m/s]');
title('Velocity vs Time'); 
if cfg == 3
    legend('True vx','True vy','True vz','Estimated vx','Estimated vy','Estimated vz',...
           'Measured vx1','Measured vx2','Measured vy1','Measured vy2','Measured vz1','Measured vz2');
else
    legend('True vx','True vy','True vz','Estimated vx','Estimated vy','Estimated vz',...
           'Measured vx','Measured vy','Measured vz');
end
end

%% Display results table
results = table(config_names', avg_rmse_px, avg_rmse_py, avg_rmse_pz, ...
                avg_rmse_vx, avg_rmse_vy, avg_rmse_vz, ...
                'VariableNames',{'Configuration','Avg_RMSE_px','Avg_RMSE_py','Avg_RMSE_pz','Avg_RMSE_vx','Avg_RMSE_vy','Avg_RMSE_vz'});
disp(results);
