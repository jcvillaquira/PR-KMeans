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
