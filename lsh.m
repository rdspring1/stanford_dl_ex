addpath ./common
addpath ./multilayer_supervised

[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1e-2;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
num_nodes = size(stack{1}.W, 1);
num_points = 16;
%example = data_train(:,1);
example = rand(28);
example = reshape(example, 784, 1);
    
poolDims = [3];

X = zeros(num_points, 1);
for idx = 1:num_points
   X(idx) = idx; 
end

for idx = 1:numel(poolDims)
    poolDim = poolDims(idx);
    %cDim = ei.input_dim / poolDim;
    cDim = 36;
    permutation = randperm(ei.input_dim);
    poolExample = interpolateLSH(0.2, 28, example');
    %poolExample = sumLSH(poolDim, example');

    dist = zeros(num_nodes, 1);
    for i = 1:num_nodes
        dist(i) = cosine_distance(stack{1}.W(i,:), example);
    end

    [sorted_dist, sorted_idx] = sort(dist);

    poolNodes = zeros(num_nodes, cDim);
    for i = 1:num_nodes
        poolNodes(i,:) = interpolateLSH(0.2, 28, stack{1}.W(sorted_idx(i), :));
        %poolNodes(i,:) = sumLSH(poolDim, stack{1}.W(sorted_idx(i), :));
    end

    poolDist = zeros(num_nodes, 1);
    for i = 1:num_nodes
       poolDist(i) = cosine_distance(poolNodes(i,:), poolExample);
    end

    y = zeros(num_points, 1);
    poolDim = floor(numel(poolDist) / num_points);
    for i = 1:num_points
        start = (i-1) * poolDim+1;
        y(i) = mean(poolDist(start:start+poolDim-1, 1));
    end
    
    plot(poolDist)
    range(y)
    range(poolDist)
    range(dist)
    figure;
    plot(y)
    
    p = polyfit(X, y, 1)
    yfit =  p(1) * X + p(2);
    yresid = y - yfit;
    SSresid = sum(yresid.^2);
    SStotal = (length(y)-1) * var(y);
    rsq = 1 - SSresid/SStotal
end
