module RegularizedClustering

include("utils.jl")
include("energy.jl")
using Plots
using ProgressBars
using DataStructures

export Model, ParallelModel, run_model!, refine!, visualize

## Auxiliary Types
const MatrixView{T} = SubArray{T,2,Matrix{T},<:Tuple,false}
const InputType{T} = Union{Matrix{T},MatrixView{T}}

## CLUSTER struct

mutable struct Cluster
  indices::Set{Int}
  n_points::Int
  centroid::Vector{Float64}
end

Base.length(G::Cluster) = G.n_points

function Cluster(data::InputType{Float64}; init::Bool=true)
  n_points = size(data, 1)
  indices = Set{Int}(1:n_points)
  if init
    centroid = compute_centroid(data)
  else
    n_points = 0
    centroid = zeros(Float64, size(data, 2))
  end
  Cluster(indices, n_points, centroid)
end

function Cluster(n_features::Int)
  Cluster(Set(), 0, zeros(Float64, n_features))
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

function Clusters(data::InputType{Float64}, weight::DefaultDict{Int,Int,Int})
  weighted = length(weight) > 0
  data_cluster = Cluster(data; init=!weighted)
  if weighted
    for (idx, row) in enumerate(eachrow(data))
      data_cluster.n_points += weight[idx]
      data_cluster.centroid .+= weight[idx] * row
    end
    data_cluster.centroid ./= data_cluster.n_points
  end
  clusters = Clusters([data_cluster], size(data, 2), Set{Int}())
  add_empty!(clusters)
  return clusters
end

## REGULARIZED K-MEANS struct

mutable struct Model
  data::InputType{Float64}
  clusters::Clusters
  colors::Vector{Int64}
  weight::DefaultDict{Int,Int,Int}
  energy::Float64
  name::String
  λ::Float64
  iter_max::Int
  tol::Float64
  iterations_done::Int64
end

Model(data::InputType{Float64}, λ::Float64, iter_max::Int, tol::Float64; weight=DefaultDict{Int,Int,Int}(1), name="") = begin
  clusters = Clusters(data, weight)
  colors = ones(Int, size(data, 1))
  Model(data, clusters, colors, weight, Inf, name, λ, iter_max, tol, 0)
end

function run_model!(model::Model)
  @info "Running model $(model.name)"
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
      ΔE_J = compute_ΔE(x, model.clusters[i], G_J, model.λ, model.weight[nx])
      j, ΔE = ΔE_J < ΔE ? (J, ΔE_J) : (j, ΔE)
    end
    if j != i
      reassign_point!(x, nx, model.clusters, i => j, model.weight[nx])
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


## PARALLEL MODEL STRUCT

mutable struct ParallelModel
  n_threads::Int
  data::InputType{Float64}
  sub_models::Vector{Model}
  cluster_model::Union{Model,Nothing}
  pointers::Vector{Tuple{Int,Int}}
  colors::Vector{Int}
  name::String
  parallel::Bool
  λ_c::Float64
  λ_g::Float64
  iter_max::Int
  tol::Float64
end

function ParallelModel(n_threads::Int, data::InputType{Float64}, λ_c::Float64, λ_g::Float64, iter_max::Int, tol::Float64; name="Parallel", parallel=true)
  size_batch = Int(ceil(size(data, 1) / n_threads))
  partition_index = Iterators.partition(1:size(data, 1), size_batch)
  sub_models = Vector{Model}(undef, n_threads)
  for (p, idx) in enumerate(partition_index)
    sub_data = data[idx, :]
    sub_models[p] = Model(sub_data, λ_c, iter_max, tol; name="$(name)_$(p)")
  end
  colors = ones(Int, size(data, 1))
  ParallelModel(n_threads, data, sub_models, nothing, Vector{Tuple{Int,Int}}(), colors, name, parallel, λ_c, λ_g, iter_max, tol)
end

function create_clusters_model!(p_model::ParallelModel)
  centroids = Vector{Vector{Float64}}()
  pointers = Vector{Tuple{Int,Int}}()
  weight = DefaultDict{Int,Int,Int}(1)
  for (nsm, sub_model) in enumerate(p_model.sub_models)
    for (ng, g) in enumerate(sub_model.clusters)
      if length(g) == 0
        continue
      end
      push!(pointers, (nsm, ng))
      push!(centroids, g.centroid)
      weight[length(centroids)] = length(g)
    end
  end
  p_model.pointers = pointers
  return collect(transpose(hcat(centroids...))), weight
end


function concat_cluster_groups!(p_model::ParallelModel)
  for (color, centroid_cluster) in enumerate(p_model.cluster_model.clusters)
    if length(centroid_cluster) == 0
      continue
    end
    for id_cluster in centroid_cluster.indices
      n_sm, n_g = p_model.pointers[id_cluster]
      sub_model = p_model.sub_models[n_sm]
      relative_index = collect(sub_model.clusters[n_g].indices)
      sub_model.colors[relative_index] .= color
    end
  end
  p_model.colors = vcat((sub_model.colors for sub_model in p_model.sub_models)...)
end


function run_model!(p_model::ParallelModel)
  @info "Running model $(p_model.name)"
  if p_model.parallel
    Threads.@threads for sub_model in p_model.sub_models
      run_model!(sub_model)
    end
  else
    for sub_model in p_model.sub_models
      run_model!(sub_model)
    end
  end
  centroid_data, weight = create_clusters_model!(p_model)
  p_model.cluster_model = Model(centroid_data, p_model.λ_g, p_model.iter_max, p_model.tol; weight=weight, name="Centroids")
  run_model!(p_model.cluster_model)
  concat_cluster_groups!(p_model)
  return nothing
end


function refine!(p_model::ParallelModel, λ_r::Float64)
  @info "Refining model $(p_model.name)"
  best_candidate = Vector{Int}(undef, size(p_model.data, 1))
  for _ in 1:(p_model.iter_max)
    done = refinement_step!(p_model, λ_r, best_candidate)
    if done # TODO: Add energy stopping condition
      break
    end
  end
end

function refinement_step!(p_model::ParallelModel, λ_r::Float64, best_candidate::Vector{Int})
  done = true
  minimum_ΔE = fill(Inf, size(p_model.data, 1))
  for (i, Gᵢ) in enumerate(p_model.cluster_model.clusters)
    if length(Gᵢ) == 0
      continue
    end
    for (j, Gⱼ) in enumerate(p_model.cluster_model.clusters)
      if (j == i) || (length(Gⱼ) == 0)
        continue
      end
      γ = compute_γ(Gᵢ, Gⱼ, λ_r)
      for c_id in Gᵢ.indices
        n_submodel, n_cluster = p_model.pointers[c_id]
        C = p_model.sub_models[n_submodel].clusters[n_cluster]
        ρ = maximum(norm(p_model.data[x_id, :] - Gᵢ.centroid) for x_id in C.indices)
        if ρ + norm(Gᵢ.centroid - C.centroid) > γ
          for x_id in C.indices
            x = p_model.data[x_id, :]
            if norm(x - Gᵢ.centroid) > γ
              ΔE = compute_ΔE(x, Gᵢ, Gⱼ, λ_r, 1)
              if ΔE < minimum_ΔE[x_id]
                minimum_ΔE[x_id] = ΔE
                best_candidate[x_id] = j
              end
            end
          end
        end
      end
    end
  end
  for (x_id, ΔE) in enumerate(minimum_ΔE)
    if isfinite(ΔE)
      i, j = p_model.colors[x_id], best_candidate[x_id]
      remove_point!(p_model.data[x_id, :], -1, p_model.cluster_model.clusters[i], 1)
      add_point!(p_model.data[x_id, :], -1, p_model.cluster_model.clusters[j], 1)
      delete!(p_model.cluster_model.clusters[j].indices, -1)
      p_model.colors[x_id] = j
      done = false
    end
  end
  return done
end


function visualize(model::Union{Model,ParallelModel}; path::Union{String,Nothing}=nothing, colors::Union{Vector,Nothing}=nothing)
  n_features = size(model.data, 2)
  if n_features != 2
    throw("Visualization only available for 2d data.")
  end
  pl = scatter()
  colors = isnothing(colors) ? model.colors : colors
  color_set = sort(collect(Set(colors)))
  for c in color_set
    mask = [isequal(c, cc) for cc in colors]
    scatter!(pl, model.data[mask, 1], model.data[mask, 2], label=c)
  end
  if !isnothing(path)
    savefig(pl, path)
  end
  return pl
end

end

