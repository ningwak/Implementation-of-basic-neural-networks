function [gradW, gradb] = ComputeGradients(X, Y, P, S1, H, W, lambda)

gradW1 = zeros(size(W{1}));
gradb1 = zeros(size(W{1}, 1), 1);
gradW2 = zeros(size(W{2}));
gradb2 = zeros(size(W{2}, 1), 1);
for i = 1:size(X, 2)
    x = X(:, i);
    y = Y(:, i);
    p = P(:, i);
    s1 = S1(:, i);
    h = H(:, i);
    g = -(y - p)';
    gradb2 = gradb2 + g';
    gradW2 = gradW2 + g' * h';
    g = g * W{2};
    sind = zeros(size(s1, 1), size(s1, 2));
    sind(find(s1 > 0)) = 1;
    g = g * diag(sind);
    gradb1 = gradb1 + g';
    gradW1 = gradW1 + g' * x'; 
end
gradb1 = gradb1 / size(X, 2);
gradW1 = gradW1 / size(X, 2) + 2 * lambda * W{1};
gradb2 = gradb2 / size(X, 2);
gradW2 = gradW2 / size(X, 2) + 2 * lambda * W{2};

gradW = {gradW1, gradW2};
gradb = {gradb1, gradb2};
end