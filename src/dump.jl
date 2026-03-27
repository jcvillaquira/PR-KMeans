function regularized_k_means(X, lambda, iter_max, tol)
  n_data = size(data, 1)
  n_features = size(data, 2)
  output = Vector{Cluster}

  g1 = Cluster(Set{Int}(1:n_data), n_data,)
end

regularized_k_means(data, 0.1, 10, 0.01)


changeᵢ = length(Gᵢ) > 1 ? 1.0 / (length(Gᵢ) * (length(Gᵢ) - 1)) : -1.0
changeⱼ = length(Gⱼ) > 0 ? -1.0 / (length(Gⱼ) * (length(Gⱼ) + 1)) : 1.0
change_factor = changeᵢ + changeⱼ
distanceᵢ = length(Gᵢ) > 1 ? -(length(Gᵢ) / (length(Gᵢ) - 1)) * norm(x - Gᵢ.centroid)^2 : 0.0
distanceⱼ = length(Gⱼ) > 0 ? length(Gⱼ) / (length(Gⱼ) + 1) * norm(x - Gⱼ.centroid)^2 : 0.0
distance_factor = distanceᵢ + distanceⱼ
distance_factor = (length(Gⱼ) / (length(Gⱼ) + 1)) * norm(x - Gⱼ.centroid)^2 - (length(Gᵢ) / (length(Gᵢ) - 1)) * norm(x - Gᵢ.centroid)^2
