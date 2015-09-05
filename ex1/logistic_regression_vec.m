function [f,g] = logistic_regression_vec(theta, X, y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
y_hat = theta' * X;
sigmoid = @(x) 1/(1+exp(-x));
h = arrayfun(sigmoid, y_hat);

for i = 1:m
  cost = y(i) * log(h(i)) + (1-y(i)) * log(1-h(i));
  f = f + cost;
end
f = f * -1.0;

g = X * (h - y)';