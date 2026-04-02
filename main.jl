using CSV

include("src/cluster.jl")
using .RegularizedClustering

## Parameters & Data
file = CSV.File(open("data/data_2d.csv"))
data = hcat(file.x, file.y)
iter_max = 10
tol = 0.1

## Sequential Version
λ = 10_000.0
model = Model(data, λ, iter_max, tol)
run_model!(model)
visualize(model)

## Parallel Version
n_threads = 8
λ_c = 10_000.0
λ_g = 1_000_000.0
λ_r = 10.0
p_model = ParallelModel(n_threads, data, λ_c, λ_g, iter_max, tol; parallel=true)
run_model!(p_model)
refine!(p_model, λ_r)
visualize(p_model)
