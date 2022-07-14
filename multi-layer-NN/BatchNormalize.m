function s_hat = BatchNormalize(s, mu, v)
s_hat = (diag((v + eps).^-0.5)) * (s - mu);
end