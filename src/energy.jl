include("utils.jl")

function compute_ΔE(x, Gᵢ, Gⱼ, λ, c)
  reg_factor, dist_factor = 0.0, 0.0
  if length(Gᵢ) > 1
    reg_factor += c / (length(Gᵢ) * (length(Gᵢ) - c))
    dist_factor -= (length(Gᵢ) * c / (length(Gᵢ) - c)) * distance2(x, Gᵢ.centroid)
  else
    reg_factor -= 1 / c
  end
  if length(Gⱼ) > 0
    reg_factor -= c / (length(Gⱼ) * (length(Gⱼ) + c))
    dist_factor += c * length(Gⱼ) / (length(Gⱼ) + c) * distance2(x, Gⱼ.centroid)
  else
    reg_factor += 1 / c
  end
  return λ * reg_factor + dist_factor
end

function compute_ΔE_merge(G1, G2, λ)
  if length(G1) * length(G2) == 0
    return 0.0
  end
  centroid_factor = (length(G1) * length(G1) / (length(G1) + length(G1))) * distance2(G1.centroid, G2.centroid)
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
      energy += distance2(model.data[nx, :], G.centroid)
    end
  end
  model.energy = energy
end

