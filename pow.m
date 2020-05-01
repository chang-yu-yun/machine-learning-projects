function [out] = pow(in, n)
out = ones(length(in), 1);
for i = 0:(n - 1)
    out = out .* in;
end
end

