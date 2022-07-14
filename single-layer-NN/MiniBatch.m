function [Wstar, bstar, trainingLoss, validationLoss, testAccuracy] = MiniBatch(X, Y, testX, testY, testLabels, n_batch, eta, n_epoch, W, b, lambda)
    
    for i=1:n_epoch
        trainingLoss(i) = ComputeCost(X, Y, W, b, lambda);
        validationLoss(i) = ComputeCost(testX, testY, W, b, lambda);
        testAccuracy(i) = ComputeAccuracy(testX, testLabels, W, b) * 100;
        for j=1:size(X, 2)/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            W = W - eta*grad_W;
            b = b - eta*grad_b;
        end
    end
    
    Wstar = W;
    bstar = b;
    
end