clear;
lambda = 0;
eta = 0.1;
n_epochs = 40;
n_batch = 100;


[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validationX, validationY, validationy] = LoadBatch('data_batch_2.mat');

meanX = mean(trainX, 2);
stdX = std(trainX, 0, 2);
trainX = trainX - repmat(meanX, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(stdX, [1, size(trainX, 2)]);

meanX = mean(validationX, 2);
stdX = std(validationX, 0, 2);
validationX = validationX - repmat(meanX, [1, size(validationX, 2)]);
validationX = validationX ./ repmat(stdX, [1, size(validationX, 2)]);

d = size(trainX, 1);
n = size(trainX, 2);
K = size(trainY, 1);
deviation = 0.01;
W = deviation .* randn(K, d);
b = deviation .* randn(K, 1);

% P = EvaluateClassifier(trainX(:, 1:20), W, b);
% [agradW, agradb] = ComputeGradients(trainX(1:20, 1), trainY(:, 1),P, W(:, 1:20), lambda);
% [ngradb, ngradW] = ComputeGradsNum(trainX(1:20, 1), trainY(:, 1), W(:, 1:20), b, lambda, 1e-6);
% 
% eps = 0.1;
% eb = abs(ngradb - agradb) ./ max(eps, abs(ngradb) + abs(agradb));
% eW = abs(ngradW - agradW) ./ max(eps, abs(ngradW) + abs(agradW));

for i = 1:n_epochs
    for j = 1:n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = trainX(:, j_start:j_end);
        Ybatch = trainY(:, j_start:j_end);
        
        [W, b] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, lambda);
    end
    
    trainloss(i) = ComputeCost(trainX, trainY, W, b, lambda);
    validationloss(i) = ComputeCost(validationX, validationY, W, b, lambda);
    acc(i) = ComputeAccuracy(validationX, validationy, W, b);
end

mt = [];
for i=1:K
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
    mt = [mt s_im{i}];
end
figure();
montage(s_im, 'size', [1, K]);
inds = 1:n_epochs;
figure();
plot(inds, trainloss, inds, validationloss);
legend('train loss', 'validation loss');