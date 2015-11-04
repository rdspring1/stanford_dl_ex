function [ Y ] = sumLSH( poolDim, X)

iterations = floor(numel(X) / poolDim);

Y = zeros(iterations, 1);
for i = 1:iterations
    start = (i-1) * poolDim+1;
    Y(i) = mean(X(1, start:start+poolDim-1));
end
end