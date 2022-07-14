function Grads = ComputeGradients(X, Y, NetParams, lambda, varargin)

if nargin == 6
    mu = varargin{1};
    v = varargin{2};
end
W = NetParams.W;
nb = size(X, 2);
H{1} = X;
S_hat{1} = 0;
for i = 1 : size(NetParams.W, 2)
    b = repmat(NetParams.b{i}, 1, size(X, 2));
    S{i} = NetParams.W{i} * H{i} + b;
    if NetParams.use_bn
        if i == size(NetParams.W, 2)
            P = softmax(S{i});
        else
            if nargin ~= 6
                [mu{i}, v{i}] = meanvar(S{i});
            end
            S_hat{i} = BatchNormalize(S{i}, mu{i}, v{i});
            S_tilde = NetParams.gammas{i} .* S_hat{i} + NetParams.betas{i};
            H{i + 1} = max(0, S_tilde);
        end        
    else
        if i == size(NetParams.W, 2)
            P = softmax(S{i});
        else
            H{i + 1} = max(0, S{i});
        end
    end
end
G = -(Y - P);
for i = 1:size(W, 2)
    ii = size(W, 2) - i + 1;
    if (NetParams.use_bn) && (i ~= 1)
        gradGamma{ii} = (1/nb) * (G .* S_hat{ii}) * ones(nb, 1);
        gradBeta{ii} = (1/nb) * (G * ones(nb, 1));
        G = G .* (NetParams.gammas{ii} * ones(1, nb));
        G = BatchNormBackPass(G, S{ii}, mu{ii}, v{ii}); 
    end
    gradW{ii} = (1/nb) * G * H{ii}' + 2 * lambda * NetParams.W{ii};
    gradb{ii} = G * ones(nb, 1)/nb;
    G = W{ii}' * G;
    G = G .* sign(H{ii});
end
Grads.W = gradW;
Grads.b = gradb;
if NetParams.use_bn  
    Grads.mu = mu;
    Grads.v = v;
    Grads.gammas = gradGamma;
    Grads.betas = gradBeta;
end

end