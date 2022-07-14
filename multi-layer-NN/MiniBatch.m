function [NetParams, results] = MiniBatch(data, testData, NetParams, parameters)
iterations = 0;
n = size(data.X, 2);
layers = size(NetParams.W, 2);
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
        
        NetParams = MiniBatchGD(Xbatch, Ybatch, parameters.eta, NetParams, parameters.lambda, 0.9);
        iterations = iterations + 1;
        if NetParams.use_bn
            mu_av = NetParams.mu_av{layers - 1};
            v_av = NetParams.v_av{layers - 1};
            if mod(iterations, 10) == 1
                k = floor(iterations/10) + 1;
                [results.traincost(k), results.trainloss(k)] = ComputeCost(data.X, data.Y, NetParams, parameters.lambda, mu_av, v_av);
                [results.testcost(k), results.testloss(k)] = ComputeCost(testData.X, testData.Y, NetParams, parameters.lambda, mu_av, v_av);
                P = EvaluateClassifier(data.X, NetParams);
                results.trainacc(k) = ComputeAccuracy(P, data.y);
                P = EvaluateClassifier(testData.X, NetParams);
                results.testacc(k) = ComputeAccuracy(P, testData.y);   
                results.testacc(k)
            end
        else
            if mod(iterations, 10) == 1
                k = floor(iterations/10) + 1;
                [results.traincost(k), results.trainloss(k)] = ComputeCost(data.X, data.Y, NetParams, parameters.lambda);
                [results.testcost(k), results.testloss(k)] = ComputeCost(testData.X, testData.Y, NetParams, parameters.lambda);
                P = EvaluateClassifier(data.X, NetParams);
                results.trainacc(k) = ComputeAccuracy(P, data.y);
                P = EvaluateClassifier(testData.X, NetParams);
                results.testacc(k) = ComputeAccuracy(P, testData.y); 
                results.testacc(k)
            end
        end
            
    end
    
%     [results.traincost(i), results.trainloss(i)] = ComputeCost(data.X, data.Y, W, b, parameters.lambda);
%     [results.testcost(i), results.testloss(i)] = ComputeCost(testData.X, testData.Y, W, b, parameters.lambda);
%     results.trainacc(i) = ComputeAccuracy(data.X, data.y, W, b);
%     results.testacc(i) = ComputeAccuracy(testData.X, testData.y, W, b);
end
P = EvaluateClassifier(testData.X, NetParams);
results.validationacc = ComputeAccuracy(P, testData.y); 
end