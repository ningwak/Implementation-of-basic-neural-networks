function Grads = ComputeGradients(RNN, X, Y, P, H, A)
[d, n] = size(X);
m = size(H, 1);
grads_V = zeros(size(RNN.V));
grads_W = zeros(size(RNN.W));
grads_U = zeros(size(RNN.U));
grads_c = zeros(size(RNN.c));
grads_b = zeros(size(RNN.b));
grads_at = zeros(1, m);
for t = n:-1:1
    grads_pt = -Y(:, t)'/(Y(:, t)' * P(:, t));
    grads_ot = -(Y(:, t) -  P(:, t))';
    grads_V = grads_V + grads_ot' * H(:, t + 1)';
    % grads_ht = grads_ot * RNN.V;
    grads_c = grads_c + grads_ot';
    
    grads_ht = grads_ot * RNN.V + grads_at * RNN.W;
    grads_at = grads_ht * diag(1 - tanh(A(:, t)).^2);
    grads_b = grads_b + grads_at';
    grads_W = grads_W + grads_at' * H(:, t)';
    grads_U = grads_U + grads_at' * X(:, t)';
end
Grads.b = grads_b;
Grads.W = grads_W;
Grads.U = grads_U;
Grads.V = grads_V;
Grads.c = grads_c;
end