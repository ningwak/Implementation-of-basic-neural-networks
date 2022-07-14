function [Wstar, bstar] = MiniBatchGD(X, Y, eta, W, b, lambda)

[P, s1, h] = EvaluateClassifier(X, W, b);
[gradW, gradb] = ComputeGradients(X, Y, P, s1, h, W, lambda);
for i = 1:size(W, 2)
    Wstar{i} = W{i} - eta * gradW{i};
    bstar{i} = b{i} - eta * gradb{i};
end
end


