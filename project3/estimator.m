clc;
clear;

%% Part 2: Sequential Estimator
mean = input('Please input the mean of Gaussian distribution: ');
var = input('Please input the variance of Gaussian distribution: ');
fprintf(1, 'Data point source function: N(%f, %f)\n\n', mean, var);
THRESHOLD = 5e-3;

prevMean = 0;
prevM2 = 0;
prevVar = 0;

currMean = 0;
currM2 = 0;
currVar = 0;

count = 0;
flag = 1;

while flag
    prevMean = currMean;
    prevM2 = currM2;
    prevVar = currVar;
    x = genGaussian(mean, var);
    count = count + 1;
    currMean = prevMean + (x - prevMean) / count;
    currM2 = prevM2 + (x - prevMean) * (x - currMean);
    if count > 1
        currVar = currM2 / (count - 1);
    else
        currVar = 0;
    end
    fprintf(1, 'Add data point: %f\n', x);
    fprintf(1, 'Mean = %f\tVariance = %f\n', currMean, currVar);
    if abs(currMean - prevMean) <= THRESHOLD && abs(currVar - prevVar) <= THRESHOLD
        break;
    end
end