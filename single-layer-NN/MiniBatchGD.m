function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda)

P = EvaluateClassifier(X, W, b);
[gradW, gradb] = ComputeGradients(X, Y, P, W, lambda);
Wstar = W - eta * gradW;
bstar = b - eta * gradb;
end

