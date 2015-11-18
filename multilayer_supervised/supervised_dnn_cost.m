function [ cost, grad, pred] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
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
    
    sparsity = ei.layer_sizes(i) - 65;
    [sorted_activations, ~] = sort(hAct{i+1});
    sm{i} = sparsityMask(hAct{i+1}, size(Z), sparsity, sorted_activations);
    hAct{i+1} = hAct{i+1} .* sm{i};
end
hAct(1,:) = [];

weighted_sum = bsxfun(@plus, stack{numHidden+1}.W * hAct{numHidden}, stack{numHidden+1}.b);
hypothesis = exp(weighted_sum);
hAct{numHidden+1} = bsxfun(@rdivide, hypothesis, sum(hypothesis));

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  pred = hAct{numHidden+1};
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
cost = 0;
m=size(data,2);
neg_norm_y_hat = hAct{numHidden+1};
for i = 1:m
       cost = cost + log(hAct{numHidden+1}(labels(i), i));
       neg_norm_y_hat(labels(i), i) = neg_norm_y_hat(labels(i), i) - 1;
end
cost = -cost;

%% compute gradients using back propagation
%%% YOUR CODE HERE %%%
gradStack{2}.b = sum(neg_norm_y_hat, 2) ./ m;
gradStack{2}.W = neg_norm_y_hat * hAct{1}';

dZ2 = hAct{1} > 0;
d2 = (stack{2}.W' * neg_norm_y_hat) .* dZ2;
gradStack{1}.b = sum(d2, 2) ./ m;
gradStack{1}.W = d2 * data';
gradStack{1}.W = gradStack{1}.W;

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end