function Y = SynthesizeText(RNN, h0, x0, n, K)
x = x0;
h = h0;
Y = zeros(K, n);
for t = 1:n
    a = RNN.W * h + RNN.U * x + RNN.b;
    h = tanh(a);
    o = RNN.V * h + RNN.c;
    p = softmax(o);
    
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a >0);
    ii = ixs(1);
    Y(ii, t) = 1;
    x = Y(:, t);
end
    
end
