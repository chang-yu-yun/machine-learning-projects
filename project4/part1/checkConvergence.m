function [flag] = checkConvergence(w1, w2)
flag = 1; % convergence indicator
threshold = 5e-4;

for i = 1:length(w1)
    if abs(w1(i) - w2(i)) > threshold
       flag = 0;
       return;
    end
end

end

