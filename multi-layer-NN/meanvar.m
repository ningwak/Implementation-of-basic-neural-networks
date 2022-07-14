function [mu, v] = meanvar(s)
n = size(s, 2);
mu = mean(s, 2);
v = var(s, 0, 2);
v = v * (n - 1) / n;
end