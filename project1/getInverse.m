function [out] = getInverse(in)
n = size(in, 1);

L = zeros(n, n);
U = zeros(n, n);
B = zeros(n, n);
out = zeros(n, n);

% initialization
for i = 1:n
    L(i, i) = 1;
end

% LU factorization
for i = 1:n
    for j = i:n
        sum = 0;
        for k = 1:(i - 1)
            sum = sum + L(i, k) * U(k, j);
        end
        U(i, j) = in(i, j) - sum;
        sum = 0;
        for k = 1:(i - 1)
            sum = sum + L(j, k) * U(k, i);
        end
        L(j, i) = (in(j, i) - sum) / U(i, i);
    end
end

% forward elimination, solve LB = I
b = eye(n);
for i = 1:n
    B(1, i) = b(1, i) / L(1, 1);
    for k = 2:n
        sum = 0;
        for j = (k - 1):-1:1
            sum = sum + L(k, j) * B(j, i);
        end
        B(k, i) = (b(k, i) - sum) / L(k, k);
    end
end
% backward substitution, solve U/in = B
for i = 1:n
    out(n, i) = B(n, i) / U(n, n);
    for k = (n - 1):-1:1
        sum = 0;
        for j = (k + 1):n
            sum = sum + U(k, j)* out(j, i);
        end
        out(k, i) = (B(k, i)- sum) / U(k, k);
    end
end
end

