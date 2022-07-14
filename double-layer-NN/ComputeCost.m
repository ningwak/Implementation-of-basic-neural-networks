function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    loss = -sum(log(sum(Y .* P, 1))) / size(X, 2);
    cost = loss + lambda * sumsqr(W{1}) + lambda * sumsqr(W{2});
end