function [X, Y, y] = LoadBatch(fname) 
% Updated!
    fprintf('Loading LoadBatch... ');
    K = 10;
    A = load(fname);
    X = transpose(A.data);
    X = cast(X,'double');
    n = size(X, 2);
    y = A.labels + 1;
    y = cast(y,'double');
    Y = zeros(K, n);
    for i = 1:n
        onehot = y(i);
        Y(onehot, i) = 1;
    end
    disp('Done!');
end
