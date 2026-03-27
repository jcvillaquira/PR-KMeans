module RegularizedClustering

include("utils.jl")
include("energy.jl")
using Plots
using ProgressBars

export Model, update_energy!, run_model!, perform_iteration!, visualize

## CLUSTER struct

mutable struct Cluster
  indices::Set{Int}
  n_points::Int
  centroid::Vector{Float64}
end

Base.length(G::Cluster) = G.n_points

function Cluster(data::Matrix{Float64})
  n_points = size(data, 1)
  indices = Set{Int}(1:n_points)
  centroid = compute_centroid(data)
  Cluster(indices, n_points, centroid)
end

function Cluster(n_features::Int)
  Cluster(Set(), 0, zeros(n_features))
end

function Cluster()
  Cluster(2)
end

## CLUSTERS struct

mutable struct Clusters
  clusters::Vector{Cluster}
  n_features::Int
  free::Set{Int}
end

Base.length(GG::Clusters) = length(GG.clusters)
Base.getindex(GG::Clusters, k::Int) = GG.clusters[k]
Base.enumerate(GG::Clusters) = enumerate(GG.clusters)

function add_empty!(GG::Clusters)
  new_cluster = Cluster(GG.n_features)
  push!(GG.clusters, new_cluster)
  push!(GG.free, length(GG))
end

function Clusters(data::Matrix{Float64})
  data_cluster = [Cluster(data)]
  clusters = Clusters(data_cluster, size(data, 2), Set{Int}())
  add_empty!(clusters)
  return clusters
end

## REGULARIZED K-MEANS struct

mutable struct Model
  data::Matrix{Float64}
  clusters::Clusters
  colors::Vector{Int64}
  energy::Float64
  λ::Float64
  iter_max::Int
  tol::Float64
  iterations_done::Int64
end

Model(data::Matrix{Float64}, λ::Float64, iter_max::Int, tol::Float64) = begin
  clusters = Clusters(data)
  colors = ones(Int, size(data, 1))
  Model(data, clusters, colors, Inf, λ, iter_max, tol, 0)
end

function run_model!(model::Model)
  prev_energy = update_energy!(model)
  for _ in ProgressBar(1:model.iter_max)
    perform_iteration!(model)
    energy = update_energy!(model)
    if abs(energy - prev_energy) < model.tol
      break
    end
    prev_energy = energy
  end
end

function merge_step!(model::Model)
  merged = false
  for i in 1:length(model.clusters)
    G1 = model.clusters[i]
    for j in i+1:length(model.clusters)
      G2 = model.clusters[j]
      if compute_ΔE_merge(G1, G2, model.λ) < 0
        merge_clusters!(model.clusters, i, j, model.colors)
        merged = true
      end
    end
  end
  return merged
end

function perform_iteration!(model::Model)
  for (nx, x) in enumerate(eachrow(model.data))
    ΔE = 0.0
    i = model.colors[nx]
    j = i
    for (J, G_J) in enumerate(model.clusters)
      if J == i
        continue
      end
      ΔE_J = compute_ΔE(x, model.clusters[i], G_J, model.λ)
      j, ΔE = ΔE_J < ΔE ? (J, ΔE_J) : (j, ΔE)
    end
    if j != i
      reassign_point!(x, nx, model.clusters, i => j)
      model.colors[nx] = j
    end
    if j == length(model.clusters)
      add_empty!(model.clusters)
    end
    while merge_step!(model)
      continue
    end
  end
  model.iterations_done += 1
  nothing
end

function merge_clusters!(GG, i, j, colors)
  G1, G2 = GG.clusters[i], GG.clusters[j]
  G1.indices = union(G1.indices, G2.indices)
  G2.indices = Set{Int}()
  G1.centroid = (length(G1) .* G1.centroid .+ length(G2) .* G2.centroid) ./ (length(G1) + length(G2))
  G2.centroid .*= 0.0
  G1.n_points += G2.n_points
  G2.n_points = 0
  colors[colors.==j] .= i
  if G1.n_points > 0
    delete!(GG.free, i)
  end
  push!(GG.free, j)
end

function visualize(model::Model)
  n_features = size(model.data, 2)
  if n_features != 2
    throw("Visualization only available for 2d data.")
  end
  pl = scatter()
  colors = Set(model.colors)
  for c in colors
    mask = (model.colors .=== c)
    scatter!(pl, model.data[mask, 1], model.data[mask, 2])
  end
  pl
end

end

