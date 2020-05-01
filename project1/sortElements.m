function [mat] = sortElements(mat, first, last)
%% Quicksort to sort input matrix
if first >= last
    return;
end

pivot = first;
left = first;
for right = (left + 1):last
    if mat(right, 1) < mat(pivot, 1)
        left = left + 1;
        temp = mat(left, :);
        mat(left, :) = mat(right, :);
        mat(right, :) = temp;
    end
end

temp = mat(left, :);
mat(left, :) = mat(pivot, :);
mat(pivot, :) = temp;

t1 = sortElements(mat, first, left - 1);
t2 = sortElements(mat, left + 1, last);
mat(first:(left - 1), :) = t1(first:(left - 1), :);
mat((left + 1):right, :) = t2((left + 1):right, :);
end

