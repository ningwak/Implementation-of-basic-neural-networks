function acc = ComputeAccuracy(P, y)
[~, index] = max(P);
acc = sum(index' == y) / size(y, 1);
end