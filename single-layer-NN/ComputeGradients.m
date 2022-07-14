function [gradW, gradb] = ComputeGradients(X, Y, P, W, lambda)

gradW = zeros(size(Y, 1), size(X, 1));
gradb = zeros(size(Y, 1), 1);
for i = 1:size(X, 2)
    x = X(:, i);
    y = Y(:, i);
    p = P(:, i);
    g = -(y - p)';
    gradb = gradb + g';
    gradW = gradW + g' * x';
end
gradb = gradb / size(X, 2);
gradW = gradW / size(X, 2) + 2 * lambda * W;
end