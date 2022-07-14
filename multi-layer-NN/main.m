%% 
clear all;
close all;

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[train2X, train2Y, train2y] = LoadBatch('data_batch_2.mat');
[train3X, train3Y, train3y] = LoadBatch('data_batch_3.mat');
[train4X, train4Y, train4y] = LoadBatch('data_batch_4.mat');
[train5X, train5Y, train5y] = LoadBatch('data_batch_5.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% trainX = trainX./255;
% meanX = mean(trainX, 2);
% stdX = std(trainX, 0, 2);
% trainX = trainX - repmat(meanX, [1, size(trainX, 2)]);
% trainX = trainX ./ repmat(stdX, [1, size(trainX, 2)]);

testX = testX./255;
meanX = mean(testX, 2);
stdX = std(testX, 0, 2);
testX = testX - repmat(meanX, [1, size(testX, 2)]);
testX = testX ./ repmat(stdX, [1, size(testX, 2)]);

d = size(trainX, 1);
K = size(trainY, 1);

%% 

% check gradient computation without bn
lambda = 0;
m = [50, 50]; % hidden layers
use_bn = 0;
NetParams = Initialization(trainX(:, 1:20), trainY(:, 1:20), m, use_bn);
% [P, S, S_hat, H] = EvaluateClassifier(trainX(:, 1:20), NetParams);
Grads = ComputeGradients(trainX(:, 1:20), trainY(:, 1:20), NetParams, lambda);
GradsNum = ComputeGradsNumSlow(trainX(:, 1:20), trainY(:, 1:20), NetParams, lambda, 1e-5);
eps = 0.1;
for i = 1:size(NetParams.W, 2)
    error_W{i} = abs(norm(Grads.W{i})-norm(GradsNum.W{i})) ./ max(eps, norm(abs(Grads.W{i}))+norm(abs(GradsNum.W{i})));
    error_b{i} = abs(norm(Grads.b{i})-norm(GradsNum.b{i})) ./ max(eps, norm(abs(Grads.b{i}))+norm(abs(GradsNum.b{i})));
end
%% 

% check gradient computation with bn
lambda = 0;
m = [50, 50]; % hidden layers
use_bn = 1;
NetParams = Initialization(trainX(:, 1:20), trainY(:, 1:20), m, use_bn);
% [P, S, S_hat, H] = EvaluateClassifier(trainX(:, 1:20), NetParams);
Grads = ComputeGradients(trainX(:, 1:20), trainY(:, 1:20), NetParams, lambda);
GradsNum = ComputeGradsNumSlow(trainX(:, 1:20), trainY(:, 1:20), NetParams, lambda, 1e-5);
eps = 0.1;
%relative error
k = size(NetParams.W, 2);
for i = 1:k
    error_W{i} = abs(norm(Grads.W{i})-norm(GradsNum.W{i})) ./ max(eps, norm(abs(Grads.W{i}))+norm(abs(GradsNum.W{i})));
    error_b{i} = abs(norm(Grads.b{i})-norm(GradsNum.b{i})) ./ max(eps, norm(abs(Grads.b{i}))+norm(abs(GradsNum.b{i})));
    if (NetParams.use_bn) && (i<k)
        error_ga{i} = abs(norm(Grads.gammas{i})-norm(GradsNum.gammas{i})) ./ max(eps, norm(abs(Grads.gammas{i}))+norm(abs(GradsNum.gammas{i})));
        error_be{i} = abs(norm(Grads.betas{i})-norm(GradsNum.betas{i})) ./ max(eps, norm(abs(Grads.betas{i}))+norm(abs(GradsNum.betas{i})));
    end
end

%% 
% without bn

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.005;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

% completeData.X = [trainX, train2X, train3X];
% completeData.Y = [trainY, train2Y, train3Y];
% completeData.y = [trainy; train2y; train3y];

completeData.X = completeData.X./255;
meanX = mean(completeData.X, 2);
stdX = std(completeData.X, 0, 2);
completeData.X = completeData.X - repmat(meanX, [1, size(completeData.X, 2)]);
completeData.X = completeData.X ./ repmat(stdX, [1, size(completeData.X, 2)]);

n = size(completeData.X, 2);
parameters.n_s = 5*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

validationData.X = validationData.X./255;
meanX = mean(validationData.X, 2);
stdX = std(validationData.X, 0, 2);
validationData.X = validationData.X - repmat(meanX, [1, size(validationData.X, 2)]);
validationData.X = validationData.X ./ repmat(stdX, [1, size(validationData.X, 2)]);

% m = [50, 50]; % hidden layers
m = [10, 10, 10, 10, 20, 20, 30, 50];
use_bn = 0;
NetParams = Initialization(completeData.X, completeData.Y, m, use_bn);

[NetParams, results] = MiniBatch(completeData, validationData, NetParams, parameters);

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

P = EvaluateClassifier(testX, NetParams);
testaccuracy = ComputeAccuracy(P, testy);  

%% 
% with bn

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.005;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

% completeData.X = [trainX, train2X, train3X];
% completeData.Y = [trainY, train2Y, train3Y];
% completeData.y = [trainy; train2y; train3y];

completeData.X = completeData.X./255;
meanX = mean(completeData.X, 2);
stdX = std(completeData.X, 0, 2);
completeData.X = completeData.X - repmat(meanX, [1, size(completeData.X, 2)]);
completeData.X = completeData.X ./ repmat(stdX, [1, size(completeData.X, 2)]);

n = size(completeData.X, 2);
parameters.n_s = 5*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

validationData.X = validationData.X./255;
meanX = mean(validationData.X, 2);
stdX = std(validationData.X, 0, 2);
validationData.X = validationData.X - repmat(meanX, [1, size(validationData.X, 2)]);
validationData.X = validationData.X ./ repmat(stdX, [1, size(validationData.X, 2)]);

% m = [50, 50]; % hidden layers
m = [10, 10, 10, 10, 20, 20, 30, 50];
use_bn = 1;
NetParams = Initialization(completeData.X, completeData.Y, m, use_bn);

[NetParams, results] = MiniBatch(completeData, validationData, NetParams, parameters);

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

P = EvaluateClassifier(testX, NetParams);
testaccuracy = ComputeAccuracy(P, testy);  

%% 
% lambda search
% l_min = -5;
% l_max = -1;
l_min = -4;
l_max = -2;

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.005;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

completeData.X = completeData.X./255;
meanX = mean(completeData.X, 2);
stdX = std(completeData.X, 0, 2);
completeData.X = completeData.X - repmat(meanX, [1, size(completeData.X, 2)]);
completeData.X = completeData.X ./ repmat(stdX, [1, size(completeData.X, 2)]);

n = size(completeData.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

validationData.X = validationData.X./255;
meanX = mean(validationData.X, 2);
stdX = std(validationData.X, 0, 2);
validationData.X = validationData.X - repmat(meanX, [1, size(validationData.X, 2)]);
validationData.X = validationData.X ./ repmat(stdX, [1, size(validationData.X, 2)]);

m = [50, 50]; % hidden layers
use_bn = 1;
NetParams = Initialization(completeData.X, completeData.Y, m, use_bn);

for tries = 1:5
    l = l_min + (l_max - l_min)*rand(1, 1);
    lt(tries) = 10^l;
    parameters.lambda = lt(tries);
    [NetParams, results] = MiniBatch(completeData, validationData, NetParams, parameters);
    Wstars{tries} = NetParams.W;
    bstars{tries} = NetParams.b;
    gammastars{tries} = NetParams.gammas;
    betastars{tries} = NetParams.betas;
    resultss{tries} = results;
    resultss{tries}.validationacc
end
%% 
% sensitivity

parameters.eta_min = 1e-5;
parameters.eta_max = 1e-1;
parameters.lambda = 0.0009;
parameters.n_batch = 100;

n_cycles = 2;

completeData.X = [trainX, train2X, train3X, train4X, train5X(:, 1:5000)];
completeData.Y = [trainY, train2Y, train3Y, train4Y, train5Y(:, 1:5000)];
completeData.y = [trainy; train2y; train3y; train4y; train5y(1:5000)];

completeData.X = completeData.X./255;
meanX = mean(completeData.X, 2);
stdX = std(completeData.X, 0, 2);
completeData.X = completeData.X - repmat(meanX, [1, size(completeData.X, 2)]);
completeData.X = completeData.X ./ repmat(stdX, [1, size(completeData.X, 2)]);

n = size(completeData.X, 2);
parameters.n_s = 2*floor(n/parameters.n_batch);
parameters.n_epochs = (n_cycles*2*parameters.n_s*parameters.n_batch)/n;
n1 = size(trainX, 2);
validationData.X = train5X(:, 5001:n1);
validationData.Y = train5Y(:, 5001:n1);
validationData.y = train5y(5001:n1);

validationData.X = validationData.X./255;
meanX = mean(validationData.X, 2);
stdX = std(validationData.X, 0, 2);
validationData.X = validationData.X - repmat(meanX, [1, size(validationData.X, 2)]);
validationData.X = validationData.X ./ repmat(stdX, [1, size(validationData.X, 2)]);

m = [50, 50]; % hidden layers
use_bn = 1;
NetParams = InitializationSig(completeData.X, completeData.Y, m, use_bn, 1e-3);

[NetParams, results] = MiniBatch(completeData, validationData, NetParams, parameters);

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

P = EvaluateClassifier(testX, NetParams);
testaccuracy = ComputeAccuracy(P, testy);  
