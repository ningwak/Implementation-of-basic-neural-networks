function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~, index] = max(P);
acc = sum((index - 1)' == y) / size(X, 2);
end