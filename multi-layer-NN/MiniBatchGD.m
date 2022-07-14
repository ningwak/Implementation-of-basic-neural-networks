function NetParams = MiniBatchGD(X, Y, eta, NetParams, lambda, alpha)

grads = ComputeGradients(X, Y, NetParams, lambda);
W = NetParams.W;
b = NetParams.b;
for i = 1:size(NetParams.W, 2)
    Wstar{i} = W{i} - eta * grads.W{i};
    bstar{i} = b{i} - eta * grads.b{i};
end
if NetParams.use_bn
    if numel(fieldnames(NetParams)) == 7
        mu_av = NetParams.mu_av;
        v_av = NetParams.v_av;
        mu = grads.mu;
        v = grads.v;
        for i = 1:(size(NetParams.W, 2) - 1)
            mu_av{i} = alpha * mu_av{i} + (1 - alpha) * mu{i};
            v_av{i} = alpha * v_av{i} + (1 - alpha) * v{i};
        end
    else
        mu_av = grads.mu;
        v_av = grads.v;
    end

    NetParams.mu_av = mu_av;
    NetParams.v_av = v_av;
    for i = 1:(size(NetParams.W, 2) - 1)
        gammastar{i} = NetParams.gammas{i} - eta * grads.gammas{i};
        betastar{i} = NetParams.betas{i} - eta * grads.betas{i};
    end
    NetParams.mu_av = mu_av;
    NetParams.v_av = v_av;
    NetParams.gammas = gammastar;
    NetParams.betas = betastar;
end
NetParams.W = Wstar;
NetParams.b = bstar;
end
