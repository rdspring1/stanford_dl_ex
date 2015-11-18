function [ mask ] = sparsityMask( data, size, sparsity, sorted_activations )
%sparsityMask create a mask based on the desired level of sparsity

threshold = ones(size(1), size(2));

for j = 1:size(2)
    threshold(:,j) = threshold(:,j) .* sorted_activations(sparsity, j);
end

mask = data >= threshold;
end

