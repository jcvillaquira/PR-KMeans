# Parallel Regularized K-Means Algorithm
This is a Julia implementation of a parallel regularized k-means algorithm described in [[1]](#1).

![example](assets/example.png)

## Usage
A case of use is shown in the file `main.jl` by running `julia --project=. main.jl`.
```julia
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
n_threads = 14
λ_c = 10_000.0
λ_g = 0.1

p_model = ParallelModel(n_threads, data, λ_c, λ_g, iter_max, tol)
run_model!(p_model; parallel=true)
visualize(p_model)
```

## References
<a id="1">[1]</a> 
Benjamin McLaughlin and Sung Ha Kang (2023). 
A new parallel adaptive clustering and its application to streaming data.
