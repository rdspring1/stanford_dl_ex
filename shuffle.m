function [ Y ] = shuffle( Y, order )

for idx = 1:numel(order)
  order_idx = order(idx);
  temp = Y(order_idx);
  Y(order_idx) = Y(idx);
  Y(idx) = temp;
end
end