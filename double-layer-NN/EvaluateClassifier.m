function [P, s1, h] = EvaluateClassifier(X, W, b)
    b1 = repmat(b{1}, 1, size(X, 2));
    s1 = W{1} * X + b1;
    h = max(0, s1);
    b2 = repmat(b{2}, 1, size(h, 2));
    s = W{2} * h + b2;
    P = softmax(s);
end