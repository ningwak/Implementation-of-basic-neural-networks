function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = im2double(A.data');
    y = A.labels;
    Y = y == 0:max(y);
    Y = Y.';
end