function NetParams = InitializationSig(X, Y, m, use_bn, sig)
    K = size(Y,1); 
    d = size(X,1);
    k = size(m,2)+1; 
    m = [K m d];
    NetParams.use_bn = use_bn; % 0 = no BN, 1 = BN
    for i = 1:k
        count = k - i + 1;
        NetParams.W{count} = sig.*randn(m(i), m(i + 1));
        NetParams.b{count} = zeros(m(i),1);
        if use_bn
            NetParams.gammas{count} = ones(m(i),1);
            NetParams.betas{count} = zeros(m(i),1);
        end
    end
end
