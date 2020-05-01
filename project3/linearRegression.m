clc;
clear;
close all;

%% Part 3: Bayesian Linear Regression
THRESHOLD = 5e-3;
precision = input('Please input the precision b for the initial prior w: ');
n = input('Please input the number of basis functions n: ');
errorVar = input('Please input the variance of error a: ');
weights = zeros(n, 1);
currMean = zeros(n, 1);
currVar = (1 / precision) * eye(n, n);
prevMean = currMean;
prevVar = currVar;

for i = 1:n
    str = ['Please input the weight of the basis function x^' num2str(i - 1) ': '];
    weights(i) = input(str);
end

flag = 1;
count = 0;
prevPredMean = 0;
prevPredVar = 0;
currPredMean = 0;
currPredVar = 0;
data = [];
while flag || count < 1000
    if count > 0
        fprintf(1, '-------------------------------------------------------------------------\n');
    end
    [x, y] = genDataPoint(errorVar, weights);
    data = [data; x y];
    count = count + 1;
    fprintf(1, 'Add data point (%f, %f)\n', x, y);
    phi = zeros(1, n);
    for i = 1:n
        phi(i) = x ^ (i - 1);
    end
    prevMean = currMean;
    prevVar = currVar;
    currVar = inv(inv(prevVar) + errorVar * (phi' * phi));
    currMean = currVar * (inv(prevVar) * prevMean + errorVar * y * phi');
    if count == 10
        meanTen = currMean;
        varTen = currVar;
    elseif count == 50
        meanFifty = currMean;
        varFifty = currVar;
    end
    prevPredMean = currPredMean;
    prevPredVar = currPredVar;
    currPredMean = phi * currMean;
    currPredVar = errorVar + (phi * currVar * phi');
    fprintf(1, '\nPosterior mean:\n');
    for i = 1:n
        fprintf(1, '%f\n', currMean(i));
    end
    fprintf(1, '\nPosterior variance:\n');
    for i = 1:n
        for j = 1:n
            if j > 1
                fprintf(1, ', ');
            end
            fprintf(1, '%f', currVar(i, j));
        end
        fprintf(1, '\n');
    end
    fprintf(1, '\nPredictive distribution ~ N(%f, %f)\n', currPredMean, currPredVar);
    if abs(currPredMean - prevPredMean) < THRESHOLD && abs(currPredVar - prevPredVar) < THRESHOLD
        flag = 0;
    end
end

figure;
subplot(221);
x = -2:0.01:2;
y = zeros(1, length(x));
errorV = errorVar * ones(1, length(x));
for i = 1:length(x)
    for j = 1:n
        y(i) = y(i) + weights(j) * x(i) ^ (j - 1);
    end
end
plot(x, y, 'k-');
hold on;
grid minor;
plot(x, y + errorV, 'r-');
plot(x, y - errorV, 'r-');
title('Ground Truth');

subplot(222);
phi = zeros(n, length(x));
for i = 1:length(x)
    for j = 1:n
        phi(j, i) = x(i) ^ (j - 1);
    end
    y(i) = currMean' * phi(:, i);
    errorV(i) = errorVar + phi(:, i)' * currVar * phi(:, i);
end
plot(x, y, 'k-');
hold on;
grid minor;
plot(x, y + errorV, 'r-');
plot(x, y - errorV, 'r-');
plot(data(:, 1), data(:, 2), 'bo', 'MarkerSize', 2);
title('Predictive Result');

subplot(223);
for i = 1:length(x)
    y(i) = meanTen' * phi(:, i);
    errorV(i) = errorVar + phi(:, i)' * varTen * phi(:, i);
end
plot(x, y, 'k-');
hold on;
grid minor;
plot(x, y + errorV, 'r-');
plot(x, y - errorV, 'r-');
plot(data(1:10, 1), data(1:10, 2), 'bo', 'MarkerSize', 2);
title('After 10 Samples');

subplot(224);
for i = 1:length(x)
    y(i) = meanFifty' * phi(:, i);
    errorV(i) = errorVar + phi(:, i)' * varFifty * phi(:, i);
end
plot(x, y, 'k-');
hold on;
grid minor;
plot(x, y + errorV, 'r-');
plot(x, y - errorV, 'r-');
plot(data(1:50, 1), data(1:50, 2), 'bo', 'MarkerSize', 2);
title('After 50 Samples');