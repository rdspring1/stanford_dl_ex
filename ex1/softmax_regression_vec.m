function [f,g] = softmax_regression_vec(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %

y_hat = theta' * X;
hypothesis = exp(y_hat);
norm_h = bsxfun(@rdivide, hypothesis, sum(hypothesis));
neg_norm_h = -norm_h;

for i = 1:m
    if y(i) ~= num_classes
       f = f + log(norm_h(y(i), i));
       neg_norm_h(y(i), i) = neg_norm_h(y(i), i) + 1;
    end
end
f = f * -1.0;

%groundTruth = full(sparse(y, 1:m, 1));
%groundTruth = groundTruth(1:num_classes-1,:);
g = neg_norm_h * X';
g = -1.0 * g';

g=g(:); % make gradient a vector for minFunc
