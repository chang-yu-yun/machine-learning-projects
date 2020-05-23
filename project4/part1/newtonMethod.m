function [weights, r0, r1, correct, error] = newtonMethod(c0, c1)
numOfData = size(c0, 1);
dim = size(c0, 2) + 1;
weights = zeros(dim, 1);
prevWeights = weights;
step = 0.05;
r0 = [];
r1 = [];
d0 = [ones(numOfData, 1) c0];
d1 = [ones(numOfData, 1) c1];
correct = zeros(1, 2);
error = zeros(1, 2);

% training
while 1
    for i = 1:numOfData
        % use c0 to train the weights
        t = 0;
        d = sigmoid(d0(i, :) * weights);
        gradient = (d - t) * d0(i, :)';
        hessian = d * (1 - d) * d0(i, :)' * d0(i, :);
        if rank(hessian) < size(hessian, 1)
            % hessian is not invertible
            weights = weights - step * gradient;
        else
            weights = weights - hessian \ gradient;
        end
        % use c1 to train the weights
        t = 1;
        d = sigmoid(d1(i, :) * weights);
        gradient = (d - t) * d1(i, :)';
        hessian = d * (1 - d) * d1(i, :)' * d1(i, :);
        if rank(hessian) < size(hessian, 1)
            % hessian is not invertible
            weights = weights - step * gradient;
        else
            weights = weights - hessian \ gradient;
        end
    end
    if checkConvergence(weights, prevWeights)
        break;
    else
        prevWeights = weights;
    end
end

% classification
for i = 1:numOfData
    d = sigmoid(d0(i, :) * weights);
    if d <= 0.5
        r0 = [r0; c0(i, :)];
        correct(1) = correct(1) + 1;
    else
        r1 = [r1; c0(i, :)];
        error(1) = error(1) + 1;
    end
    
    d = sigmoid(d1(i, :) * weights);
    if d <= 0.5
        r0 = [r0; c1(i, :)];
        error(2) = error(2) + 1;
    else
        r1 = [r1; c1(i, :)];
        correct(2) = correct(2) + 1;
    end
end
end

