function [ Y ] = interpolateLSH( scale, originalDim, X)
    X1 = reshape(X, originalDim, originalDim);
    Y1 = imresize(X1, scale);
    Y = reshape(Y1, numel(Y1), 1);
end