function [Wstar, bstar, results] = MiniBatch(data, testData, W, b, parameters)
iterations = 0;
n = size(data.X, 2);
for i = 1:parameters.n_epochs
    for j = 1:n/parameters.n_batch
        if floor(iterations/(2 * parameters.n_s)) == round(iterations/(2 * parameters.n_s))
            parameters.eta = parameters.eta_min + (iterations - floor(iterations/(2 * parameters.n_s)) * 2 * parameters.n_s)/parameters.n_s * (parameters.eta_max - parameters.eta_min);
        else
            parameters.eta = parameters.eta_max - (iterations - (2 * floor(iterations/(2 * parameters.n_s)) + 1) * parameters.n_s)/parameters.n_s * (parameters.eta_max - parameters.eta_min);
        end
        j_start = (j-1)*parameters.n_batch + 1;
        j_end = j*parameters.n_batch;
        Xbatch = data.X(:, j_start:j_end);
        Ybatch = data.Y(:, j_start:j_end);
        
        [W, b] = MiniBatchGD(Xbatch, Ybatch, parameters.eta, W, b, parameters.lambda);
        iterations = iterations + 1;
        if mod(iterations, 10) == 1
            k = floor(iterations/10) + 1;
            [results.traincost(k), results.trainloss(k)] = ComputeCost(data.X, data.Y, W, b, parameters.lambda);
            [results.testcost(k), results.testloss(k)] = ComputeCost(testData.X, testData.Y, W, b, parameters.lambda);
            results.trainacc(k) = ComputeAccuracy(data.X, data.y, W, b);
            results.testacc(k) = ComputeAccuracy(testData.X, testData.y, W, b);            
        end
    end
    
%     [results.traincost(i), results.trainloss(i)] = ComputeCost(data.X, data.Y, W, b, parameters.lambda);
%     [results.testcost(i), results.testloss(i)] = ComputeCost(testData.X, testData.Y, W, b, parameters.lambda);
%     results.trainacc(i) = ComputeAccuracy(data.X, data.y, W, b);
%     results.testacc(i) = ComputeAccuracy(testData.X, testData.y, W, b);
end
% results.testacc = ComputeAccuracy(testData.X, testData.y, W, b);
Wstar = W;
bstar = b;
end