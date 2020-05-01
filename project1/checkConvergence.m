function [result] = checkConvergence(curr, prev)
result = 1;
dim = length(curr);
for i = 1:dim
    if abs(curr(i) - prev(i)) > 0.01
        result = 0;
        break;
    end
end
end

