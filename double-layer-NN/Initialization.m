function [W, b] = Initialization(m, d, K)
deviation1 = 1/sqrt(d);
W1 = deviation1 .* randn(m, d);
b1 = deviation1 .* randn(m, 1);
deviation2 = 1/sqrt(m);
W2 = deviation2 .* randn(K, m);
b2 = deviation2 .* randn(K, 1);
W = {W1, W2};
b = {b1, b2};
end
