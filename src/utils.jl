using LinearAlgebra

function compute_centroid(data)
  return reshape(sum(data; dims=1) ./ size(data, 1), :)
end

function add_point!(x, nx, G)
  push!(G.indices, nx)
  G.centroid = (G.n_points .* G.centroid + x) ./ (G.n_points + 1)
  G.n_points += 1
end

function remove_point!(x, nx, G)
  delete!(G.indices, nx)
  if isempty(G.indices)
    return nothing
  end
  G.centroid = (G.n_points .* G.centroid - x) ./ (G.n_points - 1)
  G.n_points -= 1
end

function reassign_point!(x, nx, GG, dir)
  i, j = dir
  remove_point!(x, nx, GG[i])
  if length(GG[i]) == 0
    push!(GG.free, i)
  end
  add_point!(x, nx, GG[j])
  delete!(GG.free, j)
end

