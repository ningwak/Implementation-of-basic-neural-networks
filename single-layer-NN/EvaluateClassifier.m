function P = EvaluateClassifier(X, W, b)
    s = W * X + b;
    P = softmax(s);
end