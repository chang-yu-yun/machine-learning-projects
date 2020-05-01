function [error] = calError(dataMat, b, y)
error = 0;
n = length(y);
for i = 1:n
    error = error + (dataMat(i,:) * b - y(i))^2 ;
end
end

