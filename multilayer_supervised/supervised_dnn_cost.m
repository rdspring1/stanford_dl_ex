function [ cost, grad, pred] = supervised_dnn_cost(theta, data, labels, ei, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)
 
%% default values
po = false;
pred = [];
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);
sm = cell(numHidden, 1);

%% forward propagation
%%% YOUR CODE HERE %%%
hAct{1} = data;
for i = 1:numHidden
    Z = bsxfun(@plus, stack{i}.W * hAct{i}, stack{i}.b);
    mask = Z > 0;
    hAct{i+1} = Z .* mask;
    
    sparsity = ei.layer_sizes(i) - 100;
    [sorted_activations, ~] = sort(hAct{i+1});
    sm{i} = sparsityMask(hAct{i+1}, size(Z), sparsity, sorted_activations);
    %sm{i} = rand(size(Z,1), size(Z,2)) <= 0.5; % dropouts
    hAct{i+1} = hAct{i+1} .* sm{i};
end

weighted_sum = bsxfun(@plus, stack{numHidden+1}.W * hAct{numHidden+1}, stack{numHidden+1}.b);
hypothesis = exp(weighted_sum);
hAct{numHidden+2} = bsxfun(@rdivide, hypothesis, sum(hypothesis));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  pred = hAct{numHidden+2};
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
cost = 0;
m=size(data,2);
neg_norm_y_hat = hAct{numHidden+2};
assert(sum(neg_norm_y_hat > 0) == numel(neg_norm_y_hat));
for i = 1:m
       cost = cost + log(hAct{numHidden+2}(labels(i), i));
       neg_norm_y_hat(labels(i), i) = neg_norm_y_hat(labels(i), i) - 1;
end
cost = -cost;

%% compute gradients using back propagation
%%% YOUR CODE HERE %%%
gradStack{numHidden+1}.b = sum(neg_norm_y_hat, 2) ./ m;
gradStack{numHidden+1}.W = neg_norm_y_hat * hAct{numHidden+1}';

prev_layer_delta = neg_norm_y_hat;
for i = numHidden:-1:1
    dZ2 = hAct{i+1} > 0;
    d2 = (stack{numHidden+1}.W' * prev_layer_delta) .* dZ2;
    gradStack{i}.b = sum(d2, 2) ./ m;
    gradStack{i}.W = d2 * hAct{i}';
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end