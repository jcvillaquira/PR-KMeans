# Parallel Regularized K-Means Algorithm
This is (eventually going to be) a Julia implementation of a parallel regularized k-means algorithm described in [[1]](#1).

## Usage
A case of use is shown in the file `main.jl` by running `julia --project=. main.jl`.
```julia
using CSV

include("src/cluster.jl")
using .RegularizedClustering

path = "data/data_2d.csv"
file = CSV.File(open(path))

data = hcat(file.x, file.y)

λ = 1_0000.0
iter_max = 10
tol = 0.1
model = Model(data, λ, iter_max, tol)
run_model!(model)
visualize(model)
```

## References
<a id="1">[1]</a> 
Benjamin McLaughlin and Sung Ha Kang (2023). 
A new parallel adaptive clustering and its application to streaming data.
