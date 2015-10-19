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

example = data_train(:,1);
s_example = sign(example);

max_dist = 0;
min_dist = 2;
max_node = -1;
min_node = -1;
for i = 1:size(stack{1}.W, 1)
    dist = cosine_distance(stack{1}.W(i,:), example);
    
    if dist > max_dist
        max_dist = dist;
        max_node = i;
    end
    
    if dist < min_dist
        min_dist = dist;
        min_node = i;
    end
end

max_node = stack{1}.W(max_node, :);
min_node = stack{1}.W(min_node, :);
smin = sign(min_node);
smax = sign(max_node);