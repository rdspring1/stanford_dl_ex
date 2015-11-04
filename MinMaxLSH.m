function [ Y ] = MinMaxLSH( poolDim, X, permutation )

iterations = floor(numel(X) / poolDim);

Y = zeros(iterations * 2, 1);
for i = 1:iterations
    idx = 2*(i-1)+1;
    start = (i-1) * poolDim+1;
    Y(idx) = max(X(1, start:start+poolDim-1));
    Y(idx+1) = min(X(1, start:start+poolDim-1));
end
%Y = shuffle(Y, permutation);

end