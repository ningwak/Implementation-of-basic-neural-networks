function [A, H, O, P, loss] = ForwardPass(RNN, X, Y, h0)
[~, n] = size(X);

H(:, 1) = h0;
loss = 0;
for t = 1:n
    A(:, t) = RNN.W * H(:, t) + RNN.U * X(:, t) + RNN.b;
    H(:, t + 1) = tanh(A(:, t));
    O(:, t) = RNN.V * H(:, t + 1) + RNN.c;
    P(:, t) = softmax(O(:, t));
    loss = loss - log(Y(:, t)' * P(:, t));
end
end