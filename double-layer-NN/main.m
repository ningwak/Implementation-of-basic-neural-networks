%% 
clear all;
close all;
lambda = 1;
eta = 0.001;
n_epochs = 40;
n_batch = 100;

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validationX, validationY, validationy] = LoadBatch('data_batch_2.mat');

[train3X, train3Y, train3y] = LoadBatch('data_batch_3.mat');
[train4X, train4Y, train4y] = LoadBatch('data_batch_4.mat');
[train5X, train5Y, train5y] = LoadBatch('data_batch_5.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

trainX = trainX./255;
meanX = mean(trainX, 2);
stdX = std(trainX, 0, 2);
trainX = trainX - repmat(meanX, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(stdX, [1, size(trainX, 2)]);

validationX = validationX./255;
meanX = mean(validationX, 2);
stdX = std(validationX, 0, 2);
validationX = validationX - repmat(meanX, [1, size(validationX, 2)]);
validationX = validationX ./ repmat(stdX, [1, size(validationX, 2)]);

train2X = validationX;
train2Y = validationY;
train2y = validationy;

train3X = train3X./255;
meanX = mean(train3X, 2);
stdX = std(train3X, 0, 2);
train3X = train3X - repmat(meanX, [1, size(train3X, 2)]);
train3X = train3X ./ repmat(stdX, [1, size(train3X, 2)]);

train4X = train4X./255;
meanX = mean(train4X, 2);
stdX = std(train4X, 0, 2);
train4X = train4X - repmat(meanX, [1, size(train4X, 2)]);
train4X = train4X ./ repmat(stdX, [1, size(train4X, 2)]);

train5X = train5X./255;
meanX = mean(train5X, 2);
stdX = std(train5X, 0, 2);
train5X = train5X - repmat(meanX, [1, size(train5X, 2)]);
train5X = train5X ./ repmat(stdX, [1, size(train5X, 2)]);

testX = testX./255;
meanX = mean(testX, 2);
stdX = std(testX, 0, 2);
testX = testX - repmat(meanX, [1, size(testX, 2)]);
testX = testX ./ repmat(stdX, [1, size(testX, 2)]);

d = size(trainX, 1);
n = size(trainX, 2);
K = size(trainY, 1);
m = 50;

[W, b] = Initialization(m, d, K);
%% 

% check gradient computation 

% [P, s1, h] = EvaluateClassifier(trainX(1:20, 1:2), {W{1}(:, 1:20), W{2}}, b);
% [agradWt, agradbt] = ComputeGradients(trainX(1:20, 1:2), trainY(:, 1:2), P, s1, h, {W{1}(:, 1:20), W{2}}, lambda);
% [ngradbt, ngradWt] = ComputeGradsNum(trainX(1:20, 1:2), trainY(:, 1:2), {W{1}(:, 1:20), W{2}}, b, lambda, 1e-5);
% 
% eps = 0.1;
% ngradb = ngradbt{1};
% agradb = agradbt{1};
% ngradW = ngradWt{1};
% agradW = agradWt{1};
% eb1 = abs(ngradb - agradb) ./ max(eps, abs(ngradb) + abs(agradb));
% eW1 = abs(ngradW - agradW) ./ max(eps, abs(ngradW) + abs(agradW));
% 
% ngradb = ngradbt{2};
% agradb = agradbt{2};
% ngradW = ngradWt{2};
% agradW = agradWt{2};
% eb2 = abs(ngradb - agradb) ./ max(eps, abs(ngradb) + abs(agradb));
% eW2 = abs(ngradW - agradW) ./ max(eps, abs(ngradW) + abs(agradW));
%% 

% sanity check

% lambda = 0;
% eta = 0.001;
% n_epochs = 200;
% n_batch = 100;
% 
% for i = 1:n_epochs
%     for j = 1:n/n_batch
%         j_start = (j-1)*n_batch + 1;
%         j_end = j*n_batch;
%         Xbatch = trainX(:, j_start:j_end);
%         Ybatch = trainY(:, j_start:j_end);
%         
%         [W, b] = MiniBatchGD(Xbatch, Ybatch, eta, W, b, lambda);
%     end
%     
%     [trainloss(i), ~] = ComputeCost(trainX, trainY, W, b, lambda);
% end
% 
% inds = 1:n_epochs;
% figure();
% plot(inds, trainloss);
% legend('train loss');

%% 

data.X = trainX;
data.Y = trainY;
data.y = trainy;
testData.X = validationX;
testData.Y = validationY;
testData.y = validationy;

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.01;
parameters.n_batch = 100;
n_cycles = 1;

n = size(data.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;

[Wstar, bstar, results] = MiniBatch(data, testData, W, b, parameters);

inds = 1:size(results.trainacc, 2);
figure();
plot(inds, results.trainloss, inds, results.testloss);
legend('training', 'validation');
figure();
plot(inds, results.traincost, inds, results.testcost);
legend('training', 'validation');
figure();
plot(inds, results.trainacc, inds, results.testacc);
legend('training', 'validation');

testaccuracy = ComputeAccuracy(testX, testy, Wstar, bstar)

%% 
data.X = trainX;
data.Y = trainY;
data.y = trainy;
testData.X = validationX;
testData.Y = validationY;
testData.y = validationy;

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.01;
parameters.n_batch = 100;
n_cycles = 3;

n = size(data.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;

[Wstar, bstar, results] = MiniBatch(data, testData, W, b, parameters);

inds = 1:size(results.trainacc, 2);
figure();
plot(inds, results.trainloss, inds, results.testloss);
legend('training', 'validation');
figure();
plot(inds, results.traincost, inds, results.testcost);
legend('training', 'validation');
figure();
plot(inds, results.trainacc, inds, results.testacc);
legend('training', 'validation');

testaccuracy = ComputeAccuracy(testX, testy, Wstar, bstar)
%% 
l_min = -5;
l_max = -1;

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.01;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

n = size(completeData.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

for tries = 1:10
    l = l_min + (l_max - l_min)*rand(1, 1);
    lt(tries) = 10^l;
    parameters.lambda = lt(tries);
    [Wstar, bstar, results] = MiniBatch(completeData, validationData, W, b, parameters);
    Wstars{tries} = Wstar;
    bstars{tries} = bstar;
    resultss{tries} = results;
    resultss{tries}.testacc
end

% inds = 1:parameters.n_epochs;
% figure();
% for tries = 1:8
%     plot(inds, results.testloss);
%     legend(tries);
%     hold on;
% end
% 
% figure();
% for tries = 1:8
%     plot(inds, results.testcost);
%     legend(tries);
%     hold on;
% end
% 
% figure();
% for tries = 1:8
%     plot(inds, results.testacc);
%     legend(tries);
%     hold on;
% end

%% 
l_min = -4;
l_max = -3;

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.01;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

n = size(completeData.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

for tries = 1:10
    l = l_min + (l_max - l_min)*rand(1, 1);
    lt(tries) = 10^l;
    parameters.lambda = lt(tries);
    [Wstar, bstar, results] = MiniBatch(completeData, validationData, W, b, parameters);
    Wstars{tries} = Wstar;
    bstars{tries} = bstar;
    resultss{tries} = results;
    resultss{tries}.testacc
end

    
%% 

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.n_s = 900;
parameters.lambda = 0.0042;
parameters.n_batch = 100;

n_cycles = 4;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:9000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:9000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:9000)];

meanX = mean(completeData.X, 2);
stdX = std(completeData.X, 0, 2);
completeData.X = completeData.X - repmat(meanX, [1, size(completeData.X, 2)]);
completeData.X = completeData.X ./ repmat(stdX, [1, size(completeData.X, 2)]);

validationData.X = train5X(:, 9001:n);
validationData.Y = train5Y(:, 9001:n);
validationData.y = train5y(9001:n);

meanX = mean(validationData.X, 2);
stdX = std(validationData.X, 0, 2);
validationData.X = validationData.X - repmat(meanX, [1, size(validationData.X, 2)]);
validationData.X = validationData.X ./ repmat(stdX, [1, size(validationData.X, 2)]);

n = size(completeData.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;

[Wstar, bstar, results] = MiniBatch(completeData, validationData, W, b, parameters);
inds = 1:size(results.trainacc, 2);
figure();
plot(inds, results.trainloss, inds, results.testloss);
legend('training', 'validation');
figure();
plot(inds, results.traincost, inds, results.testcost);
legend('training', 'validation');
figure();
plot(inds, results.trainacc, inds, results.testacc);
legend('training', 'validation');

testaccuracy = ComputeAccuracy(testX, testy, Wstar, bstar);



