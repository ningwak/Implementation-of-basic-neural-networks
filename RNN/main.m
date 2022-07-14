clear all;
close all;

book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);
K = size(book_chars, 2);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i = 1: K
    char_to_ind(book_chars(i)) = i;
    ind_to_char(int32(i)) = book_chars(i);
end

m = 100;
GDParams.eta = .1;
seq_length = 25;

sig = .01;

RNN.U = randn(m, K) * sig;
RNN.W = randn(m, m) * sig;
RNN.V = randn(K, m) * sig;

RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);

% X_chars = book_data(1:seq_length);
% Y_chars = book_data(2:seq_length + 1);
% 
% X = zeros(K, seq_length);
% Y = zeros(K, seq_length);
% 
% % convert to one-hot matrices
% for i = 1:seq_length
%     X(char_to_ind(X_chars(i)), i) = 1;
%     Y(char_to_ind(Y_chars(i)), i) = 1;
% end
% 
% h0 = zeros(m, 1);

%% 
% gradient check
[A, H, O, P, loss] = ForwardPass(RNN, X, Y, h0);

h = 1e-4;
AGrads = ComputeGradients(RNN, X, Y, P, H, A);
NGrads = ComputeGradsNum(X, Y, RNN, h);

eps = 1e-5;

for f = fieldnames(RNN)'
    Ag = AGrads.(f{1});
    Ng = NGrads.(f{1});
    error = abs(norm(Ag)-norm(Ng)) ./ max(eps, norm(abs(Ag))+norm(abs(Ng)));
    errors.(f{1}) = error;
end
errors
%% 
% train

GDParams.n_epochs = 10;

e = 1;
iter = 1;

for f = fieldnames(RNN)'
    GDParams.M.(f{1}) = zeros(size(RNN.(f{1})));
end
for e_poch = 1:GDParams.n_epochs
    hprev = zeros(m, 1);
    while e <= length(book_data) - seq_length - 1
        X_chars = book_data(e:e + seq_length - 1);
        Y_chars = book_data(e + 1:e + seq_length);

        X = zeros(K, seq_length);
        Y = zeros(K, seq_length);

        % convert to one-hot matrices
        for i = 1:seq_length
            X(char_to_ind(X_chars(i)), i) = 1;
            Y(char_to_ind(Y_chars(i)), i) = 1;
        end

        [A, H, ~, P, loss] = ForwardPass(RNN, X, Y, hprev);
        [RNN, GDParams] = MinibatchGD(RNN, X, Y, GDParams, A, H, P);
        [~, n] = size(X);
        hprev = H(:, n + 1);
        if iter == 1 && e == 1
            smooth_loss = loss;
            SY = SynthesizeText(RNN, hprev, X, 200, K);
            for j = 1: size(SY, 2)
                text(j) = ind_to_char(find(SY(:, j) == 1));
            end
            disp(['iter = 0, smooth_loss = ' num2str(smooth_loss) ', Text: ' text]);
        end
        smooth_loss = .999 * smooth_loss + .001 * loss;
        smooth_loss_seq(iter) = smooth_loss;
        
%         if mod(iter, 100) == 0
%             disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
%         end
        
        if mod(iter, 10000) == 0
            SY = SynthesizeText(RNN, hprev, X, 200, K);
            for j = 1: size(SY, 2)
                text(j) = ind_to_char(find(SY(:, j) == 1));
            end
            disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss) ', Text: ' text]);
        end
    
        e = e + seq_length;
        iter = iter + 1;
    end
    e = 1;
end

inds = 1:iter - 1;
figure();
plot(inds, smooth_loss_seq);

%% 
SY = SynthesizeText(RNN, hprev, X, 1000, K);
for j = 1: size(SY, 2)
    text(j) = ind_to_char(find(SY(:, j) == 1));
end
disp(['Text: ' text]);
