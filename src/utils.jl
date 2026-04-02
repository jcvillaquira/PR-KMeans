using LinearAlgebra

function compute_centroid(data)
  return reshape(sum(data; dims=1) ./ size(data, 1), :)
end

function add_point!(x, nx, G, w)
  push!(G.indices, nx)
  G.centroid = (length(G) .* G.centroid + w .* x) ./ (length(G) + w)
  G.n_points += w
end

function remove_point!(x, nx, G, w)
  delete!(G.indices, nx)
  if length(G) == w
    G.centroid .*= 0.0
  else
    G.centroid = (length(G) .* G.centroid - w .* x) ./ (length(G) - w)
  end
  G.n_points -= w
end

function reassign_point!(x, nx, GG, dir, w)
  i, j = dir
  remove_point!(x, nx, GG[i], w)
  if length(GG[i]) == 0
    push!(GG.free, i)
  end
  add_point!(x, nx, GG[j], w)
  delete!(GG.free, j)
end

function compute_γ(G1, G2, λ_r)
  a = (length(G1) + length(G2)) / ((length(G1) - 1) * (length(G2) + 1))
  b = 2 * length(G2) * norm(G2.centroid - G1.centroid)
  c = -λ_r / (length(G1)^2 - length(G1)) + λ_r / (length(G2)^2 + length(G2)) - length(G2) * norm(G2.centroid - G1.centroid)^2 / (length(G2) + 1)
  return (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
end

