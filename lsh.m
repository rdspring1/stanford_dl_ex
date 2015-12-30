addpath ./common
addpath ./multilayer_supervised

[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 2;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [1e3, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1e-2;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
num_nodes = size(stack{1}.W, 1);
num_points = 16;

example = data_train(:,1);
%example = normrnd(0, 5, ei.input_dim, 1);
sigma = example * example';
[U,S,V] = svd(sigma);

cDim = 25;
poolExample = U(:,1:cDim)' * example;

X = normrnd(0,1,ei.input_dim, 1);
sigma1 = X * X';
[U1,S1,V1] = svd(sigma);

dist = zeros(num_nodes, 1);
for i = 1:num_nodes
    dist(i) = cosine_distance(stack{1}.W(i,:), example);
end

[sorted_dist, sorted_idx] = sort(dist);
poolNodes = zeros(num_nodes, cDim);
for i = 1:num_nodes
    poolNodes(i,:) = U1(:,1:cDim)' *  stack{1}.W(sorted_idx(i), :)';
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
