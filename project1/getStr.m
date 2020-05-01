function [str] = getStr(b, i)
n = length(b);
if i > 1
    suffix = ['X^' num2str(i - 1)];
else
    suffix = [];
end

if i == n
    coeff = num2str(b(i));
else
    if b(i) >= 0
        coeff = ['+ ' num2str(b(i))];
    else
        coeff = ['- ' num2str(abs(b(i)))];
    end
end

str = [coeff suffix];
end

