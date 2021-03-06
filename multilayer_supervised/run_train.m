% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce 100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [500, 500, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1e-2;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'relu';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options.epochs = 1;
options.minibatch = 1;
options.alpha = 1e-3;
options.momentum = .95;

%% run training
opt_params = minFuncSGD(@(x,y,z) supervised_dnn_cost(x,y,z,ei,false),params,data_train,labels_train,options);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, data_test, [], ei, true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test) * 100;
fprintf('test accuracy: %2.1f%%\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, data_train, [], ei, true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train) * 100;
fprintf('train accuracy: %2.1f%%\n', acc_train);

%% check gradient
%average = grad_check(@supervised_dnn_cost, params, 10, ei, data_train, labels_train, false);
