function [ Y ] = interpolateLSH( poolDim, originalDim, X)

X1 = reshape(X, originalDim, originalDim);
iterations = floor(numel(X) / poolDim);

Y = zeros(iterations, 1);
idx = 1;

for row = 1:originalDim
   for col = 1:poolDim:originalDim
       for i = 0:3
          for j = 0:3
              if row+i < originalDim && col+j < originalDim
                  Y(idx) = Y(idx) + X1(row+i, col+j); 
              end
          end
       end
       idx = idx + 1;
   end
end
end