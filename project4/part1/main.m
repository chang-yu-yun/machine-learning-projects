close all;
clc;
clear;

numOfData = input('Please input the number of data N: ');
% x1
mx1 = input('Please input the mean mx1: ');
vx1 = input('Please input the variance vx1: ');
% y1
my1 = input('Please input the mean my1: ');
vy1 = input('Please input the variance vy1: ');

% x2
mx2 = input('Please input the mean mx2: ');
vx2 = input('Please input the variance vx2: ');
% y2
my2 = input('Please input the mean my2: ');
vy2 = input('Please input the variance vy2: ');

c0 = zeros(numOfData, 2);
c1 = zeros(numOfData, 2);

for i = 1:numOfData
    c0(i, 1) = genGaussian(mx1, vx1);
    c0(i, 2) = genGaussian(my1, vy1);
    c1(i, 1) = genGaussian(mx2, vx2);
    c1(i, 2) = genGaussian(my2, vy2);
end

[w, g0, g1, correct, error] = gradientDescent(c0, c1);
fprintf(1, '\nGradient descent: \nw: \n');
for i = 1:length(w)
    fprintf(1, '%f\n', w(i));
end
fprintf(1, '\nConfusion matrix: \n');
fprintf(1, '               Predict cluster 1   Predict cluster 2\n');
fprintf(1, 'Is cluster 1%20d%20d\n', correct(1), error(1));
fprintf(1, 'Is cluster 2%20d%20d\n', error(2), correct(2));
fprintf(1, 'Sensitivity (Successfully predict cluster 1): %f\n', correct(1) / numOfData);
fprintf(1, 'Specificity (Successfully predict cluster 2): %f\n', correct(2) / numOfData);
fprintf(1, '\n------------------------------------------------------------\n');

[w, n0, n1, correct, error] = newtonMethod(c0, c1);
fprintf(1, '\nNewton''s method: \nw: \n');
for i = 1:length(w)
    fprintf(1, '%f\n', w(i));
end
fprintf(1, '\nConfusion matrix: \n');
fprintf(1, '               Predict cluster 1   Predict cluster 2\n');
fprintf(1, 'Is cluster 1%20d%20d\n', correct(1), error(1));
fprintf(1, 'Is cluster 2%20d%20d\n', error(2), correct(2));
fprintf(1, 'Sensitivity (Successfully predict cluster 1): %f\n', correct(1) / numOfData);
fprintf(1, 'Specificity (Successfully predict cluster 2): %f\n', correct(2) / numOfData);

figure;
subplot(131);
plot(c0(:,1), c0(:,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
hold on;
grid minor;
plot(c1(:,1), c1(:,2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
title('Ground Truth');

subplot(132);
plot(g0(:,1), g0(:,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
hold on;
grid minor;
plot(g1(:,1), g1(:,2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
title('Gradient Descent');

subplot(133);
plot(n0(:,1), n0(:,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
hold on;
grid minor;
plot(n1(:,1), n1(:,2), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
title('Newton''s Method');
