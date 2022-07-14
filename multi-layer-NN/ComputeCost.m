function [cost, loss] = ComputeCost(X, Y, NetParams, lambda, varargin)
    if nargin == 6
        mu = varargin{1};
        v = varargin{2};
        P = EvaluateClassifier(X, NetParams, 1);
    else
        P = EvaluateClassifier(X, NetParams);
    end
    % loss = trace(-log(Y'*P))/size(X, 2);
    loss = -sum(log(sum(Y .* P, 1))) / size(X, 2);
    cost = loss;
    for i = 1:size(NetParams.W, 2)
        temp = NetParams.W{i}.^2;
        cost = cost + lambda * sum(temp(:));
    end
end