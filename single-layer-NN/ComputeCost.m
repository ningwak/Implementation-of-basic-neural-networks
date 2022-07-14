function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    J = -sum(log(sum(Y .* P, 1))) / size(X, 2) + lambda * sumsqr(W);
end