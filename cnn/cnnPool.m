function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        poolImage = zeros(convolvedDim / poolDim, convolvedDim / poolDim);
        
        % Obtain the feature (filterDim x filterDim) needed during the convolution
        filter = ones(poolDim, poolDim);

        % Obtain the image
        im = squeeze(convolvedFeatures(:, :, filterNum, imageNum));

        % Convolve "filter" with "im", adding the result to convolvedImage
        % be sure to do a 'valid' convolution
        convolvedImage = convn(im, filter, 'valid');
        
        % subsampling and average
        for row = 1:poolDim:size(convolvedImage,1)
            for col = 1:poolDim:size(convolvedImage,2)
                poolImage((row-1)/poolDim + 1, (col-1)/poolDim + 1) = convolvedImage(row, col) / (poolDim * poolDim);
            end
        end
        
        pooledFeatures(:, :, filterNum, imageNum) = poolImage;
    end
end

end

