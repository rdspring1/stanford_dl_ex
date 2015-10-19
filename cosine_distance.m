function [ dist ] = cosine_distance( x, y )
%cosine_distance - reture cosine distance between two vectors
    dist = 1 - dot(x, y) / (norm(x) * norm(y));
end

