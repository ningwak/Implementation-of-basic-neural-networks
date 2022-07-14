function P = EvaluateClassifier(X, NetParams, varargin)
H{1} = X;
S_hat{1} = 0;
for i = 1 : size(NetParams.W, 2) - 1
    b = repmat(NetParams.b{i}, 1, size(X, 2));
    S{i} = NetParams.W{i} * H{i} + b;
    if NetParams.use_bn
        if nargin == 4
            mu{i} = varargin{1}(i);
            v{i} = varargin{2}(i);
        else
            [mu{i}, v{i}] = meanvar(S{i});
        end
        S_hat{i} = BatchNormalize(S{i}, mu{i}, v{i});
        S_tilde = NetParams.gammas{i} .* S_hat{i} + NetParams.betas{i};
        H{i + 1} = max(0, S_tilde);       
    else
        H{i + 1} = max(0, S{i});
    end
end 
i = size(NetParams.W, 2);
b = repmat(NetParams.b{i}, 1, size(X, 2));
Sout = NetParams.W{i} * H{i} + b;
P = softmax(Sout);
end

