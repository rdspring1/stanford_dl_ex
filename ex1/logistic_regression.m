function [f,g] = logistic_regression(theta, X, y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  n=size(X,1);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  weighted_sum = transpose(theta) * X;
  hypothesis = 1 ./ (1 + exp(-weighted_sum));
  for i = 1:m
      cost = y(i) * log(hypothesis(i)) + (1-y(i)) * log(1-hypothesis(i));
      f = f + cost;
  end
  f = f * -1.0;
  
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  
  for j = 1:n
     for i = 1:m
        g(j) = g(j) + X(j,i) * (hypothesis(i) - y(i));
     end
  end
