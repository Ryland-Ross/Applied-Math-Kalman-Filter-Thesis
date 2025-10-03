%% Ryland Ross
% 9/23/25
% 1-D Kalman Filter: Sweep of Q and R (Position + Velocity)
% Provides visualization of how different R and Q valeus affect the KF

clc; clear; close all;

%% Simulation setup
dt = 0.1;         % time step
T = 5;            % total simulation time
t = 0:dt:T;       
N = length(t);

a_true = 0.2;     % true acceleration
p0 = 0; v0 = 1;   % initial conditions

% True motion
p_true = p0 + v0*t + 0.5*a_true*t.^2;
v_true = v0 + a_true*t;

%% KF Matrices
A = [1 dt; 0 1];
B = [0.5*dt^2; dt];
H = [1 0];

% Define low/med/high Q and R
Q_vals = {0.001*eye(2), 0.1*eye(2), 1.0*eye(2)};
R_vals = [0.1^2, 0.5^2, 2.0^2];
Q_labels = {'Low Q','Med Q','High Q'};
R_labels = {'Low R','Med R','High R'};

%% Store estimates and measurements
X_all = cell(length(Q_vals), length(R_vals));
Z_all = cell(length(Q_vals), length(R_vals)); % store corresponding measurements

for i = 1:length(Q_vals) % Loop through all Q
    for j = 1:length(R_vals) % Loop through all R
        Q = Q_vals{i};
        R = R_vals(j);

        sigma_meas = sqrt(R); % ensures sigma^2 = R for each R
        z = p_true + sigma_meas*randn(size(t));
        Z_all{i,j} = z; % store measurements

        % --- Run KF ---
        x_est = zeros(2,N);
        P = eye(2);

        for k = 2:N
            % Prediction
            x_pred = A*x_est(:,k-1) + B*a_true;
            P_pred = A*P*A' + Q;

            % Update
            K = P_pred*H'/(H*P_pred*H' + R);
            x_est(:,k) = x_pred + K*(z(k) - H*x_pred);
            P = (eye(2)-K*H)*P_pred;
        end

        X_all{i,j} = x_est;
    end
end

%% --- Position plots ---
figure('Name','Position');
for i = 1:length(Q_vals)
    for j = 1:length(R_vals)
        x_est = X_all{i,j};
        z = Z_all{i,j}; % get correct measurements
        subplot(length(Q_vals), length(R_vals), (i-1)*length(R_vals)+j);
        hold on; grid on;
        plot(t, p_true,'k','LineWidth',1.5);
        plot(t, z,'r.'); % plot measurements for this Q,R
        plot(t, x_est(1,:),'b','LineWidth',1.2);
        title(sprintf('%s / %s\nQ=%.3f, R=%.2f',Q_labels{i},R_labels{j},Q_vals{i}(1,1),R_vals(j)));
        if i==length(Q_vals), xlabel('Time [s]'); end
        if j==1, ylabel('Pos [m]'); end
        legend('True','Meas','KF','Location','best');
    end
end

%% --- Velocity plots ---
figure('Name','Velocity');
for i = 1:length(Q_vals)
    for j = 1:length(R_vals)
        x_est = X_all{i,j};
        subplot(length(Q_vals), length(R_vals), (i-1)*length(R_vals)+j);
        hold on; grid on;
        plot(t, v_true,'k','LineWidth',1.5);
        plot(t, x_est(2,:),'b','LineWidth',1.2);
        title(sprintf('%s / %s\nQ=%.3f, R=%.2f',Q_labels{i},R_labels{j},Q_vals{i}(1,1),R_vals(j)));
        if i==length(Q_vals), xlabel('Time [s]'); end
        if j==1, ylabel('Vel [m/s]'); end
        legend('True','KF','Location','best');
    end
end
