clear;
close all;
clc;
fileName = input('Please input fileName: ', 's');
n = input('Please input the number of polynomial bases n: ');
lambda = input('Please input lambda value: ');

% read data from file
[x, y] = textread(fileName, '%f%f', 'delimiter', ',');
input = [x y];
input = sortElements(input, 1, size(input, 1));
x = input(:, 1);
y = input(:, 2);

% data matrix
dataMat = zeros(length(x), n);
for i = 1:n
    dataMat(:, i) = pow(x, i - 1);
end

% identity matrix
dim = size(dataMat, 2);
ident = eye(dim);
fprintf('\n');

%% LSE method
w = getInverse(dataMat' * dataMat + lambda * ident) * dataMat' * y;
str = 'Fitting line:';
for i = n:-1:1
    str = [str ' ' getStr(w, i)];
end
disp('LSE');
disp(str);
error = calError(dataMat, w, y);
disp(['Total error: ' num2str(error)]);

%% Newton's Method in optimization
prev = zeros(n, 1);
flag = 0; % check whether the calculation is converged or not
hessian = 2 * (dataMat' * dataMat);
while flag == 0
    gradient = hessian * prev - 2 * dataMat' * y;
    curr = prev - getInverse(hessian) * gradient;
    flag = checkConvergence(curr, prev);
    prev = curr;
end
str = 'Fitting line:';
for i = n:-1:1
    str = [str ' ' getStr(curr, i)];
end
fprintf('\n');
disp('Newton''s Method:');
disp(str);
error = calError(dataMat, curr, y);
disp(['Total error: ' num2str(error)]);

%% visualization
figure;
% LSE method
subplot(211)
plot(x, y, 'ro', 'MarkerSize', 6);
hold on;
plot(x, calEst(x, w), 'b-', 'LineWidth', 2);
grid minor;
hold off;
title('LSE Method');
% Newton's Method in optimization
subplot(212)
plot(x, y, 'ro', 'MarkerSize', 6);
hold on;
plot(x, calEst(x, curr), 'b-', 'LineWidth', 2);
grid minor;
hold off;
title('Newton''s Method');