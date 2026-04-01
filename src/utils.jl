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
  if isempty(G.indices)
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

