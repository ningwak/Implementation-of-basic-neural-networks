function loss = ComputeLoss(X, Y, RNN, h0)

[~, ~, ~, ~, loss] = ForwardPass(RNN, X, Y, h0);
end