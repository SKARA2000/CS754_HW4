%% Clearing console and variables
clc; clear all;
addpath("../l1_ls_matlab");
%% Constructing the actual vectors(x) and the measurement matrix(phi)
n = 500; m = 200;
S = 18;
p = 0.5;
rng(7);
phi = randn(m, n);
phi = (phi < p); phi = (phi*2 - 1)/sqrt(m);
x = unifrnd(0, 1000, n, 1);
% Constructing a Sparse Vector
x = sparse_vec_generator(x, S, n);
y = phi*x;
std_dev = (norm(y, 1)*0.05)/m;
y = y + rand(m, 1)*std_dev;
Lambda = [5e-4, 1e-4, 5e-3, 1e-3, 0.01, 0.05, 0.1, 0.5, 1, 2, 5];
cross_val = 0.9*m;
V = m - cross_val;
rel_tol = 1e-6;

%% Generating the Validation and the Reconstruction Sets
y_v = []; phi_v = []; indices = [];
temp = y;
while nnz(temp) ~= cross_val
    index = randi(m);
    if temp(index) ~= 0
        temp(index) = 0;
        y_v = [y_v; y(index)];
        phi_v = [phi_v; phi(index, :)];
        indices = [indices index];
    end
end
phi_r = phi(setdiff(1:end, indices), :);
y_r = y(setdiff(1:end, indices));

%% Solving the l1-regularized least squares problem using different lambda values
X = []; val_err = []; x_axis = []; RMSE = []; x_axis1 = [];
for i = 1:size(Lambda, 2)
    [temp1, status] = l1_ls(phi_r, y_r, Lambda(i), rel_tol, 'true');
    X = [X temp1];
    disp(status);
    if strcmpi(status, 'solved')
        error = 0;
        x_axis = [x_axis log10(Lambda(i))];
        for j = 1:V
            error = error + (y_v(j) - phi_v(j, :)*temp1)^2;
        end
        val_err = [val_err (error/(m - cross_val))];
        RMSE = [RMSE (norm(temp1 - x, 2)/norm(x, 2))];
    end
%     x_axis1 = [x_axis1 log10(Lambda(i))];
%     RMSE = [RMSE (norm(temp1 - x, 2)/norm(x, 2))];
end
figure();
plot(x_axis, val_err);
grid on;
hold on;
plot(x_axis, RMSE*1000);
title("Errors versus {log_{10}\lambda}}: Mutually exclusive sets");
xlabel("{log_{10}\lambda}");
ylabel("Validation Error and RMSE");
legend('Validation Error', 'RMSE*1000')
saveas(gcf(), '../images/plot1.png');
disp("Part 1 Done");

%% Using non-disjoint but coincident Validation and Reconstruction sets
y_v = []; phi_v = []; indices = [];
temp = y;
while nnz(temp) ~= cross_val
    index = randi(m);
    if temp(index) ~= 0
        temp(index) = 0;
        y_v = [y_v; y(index)];
        phi_v = [phi_v; phi(index, :)];
    end
end
temp = y;
while nnz(temp) ~= V
    index = randi(m);
    if temp(index) ~= 0
        temp(index) = 0;
        y_r = [y_r; y(index)];
        phi_r = [phi_r; phi(index, :)];
    end
end
X = []; val_err = []; x_axis = []; RMSE = []; x_axis1 = [];
for i = 1:size(Lambda, 2)
    [temp1, status] = l1_ls(phi_r, y_r, Lambda(i), rel_tol, 'true');
    X = [X temp1];
    disp(status);
    if strcmpi(status, 'solved')
        error = 0;
        x_axis = [x_axis log10(Lambda(i))];
        for j = 1:V
            error = error + (y_v(j) - phi_v(j, :)*temp1)^2;
        end
        val_err = [val_err (error/(m - cross_val))];
        RMSE = [RMSE (norm(temp1 - x, 2)/norm(x, 2))];
    end
end
%%
figure();
plot(x_axis, val_err);
grid on;
hold on;
plot(x_axis, RMSE*50);
title("Errors versus {log_{10}\lambda}: Coincident sets");
xlabel("{log_{10}\lambda}");
ylabel("Validation Error and RMSE");
legend('Validation Error', 'RMSE*50')
saveas(gcf(), '../images/plot2.png');
disp("Part 2 Done");

%% Sparse vector generator
function [x, indices] = sparse_vec_generator(x, S, n)
    indices = [];
    while nnz(x) ~= S
        temp = randi(n);
        if x(temp) ~= 0
            x(temp) = 0;
            indices = [indices temp];
        end
    end
end