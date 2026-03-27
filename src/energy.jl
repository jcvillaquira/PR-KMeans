using LinearAlgebra

function compute_ΔE(x, Gᵢ, Gⱼ, λ)
  reg_factor, dist_factor = 0.0, 0.0
  if length(Gᵢ) > 1
    reg_factor += 1.0 / (length(Gᵢ) * (length(Gᵢ) - 1))
    dist_factor -= (length(Gᵢ) / (length(Gᵢ) - 1)) * norm(x - Gᵢ.centroid)^2
  else
    reg_factor -= 1.0
  end
  if length(Gⱼ) > 0
    reg_factor -= 1.0 / (length(Gⱼ) * (length(Gⱼ) + 1))
    dist_factor += length(Gⱼ) / (length(Gⱼ) + 1) * norm(x - Gⱼ.centroid)^2
  else
    reg_factor += 1
  end
  return λ * reg_factor + dist_factor
end

function compute_ΔE_merge(G1, G2, λ)
  if length(G1) * length(G2) == 0
    return 0.0
  end
  centroid_factor = (length(G1) * length(G1) / (length(G1) + length(G1))) * norm(G1.centroid - G2.centroid)^2
  regularization_factor = (1.0 / (length(G1) + length(G1))) - 1.0 / length(G1) - 1.0 / length(G2)
  return centroid_factor + λ * regularization_factor
end

function update_energy!(model)
  energy = 0.0
  for G in model.clusters.clusters
    if length(G) == 0
      continue
    end
    energy += model.λ / length(G)
    for nx in G.indices
      energy += norm(model.data[nx, :] .- G.centroid)^2
    end
  end
  model.energy = energy
end


