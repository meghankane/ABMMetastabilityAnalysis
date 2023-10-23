using ABMMetastabilityAnalysis
using JLD2

precomputed_dt = load("rate_matrices/ratemat_seed1.jld2")
rate_tensor = precomputed_dt["rates_tensor"]

rate_list = [copy(r) for r = eachslice(rate_tensor; dims=3)]
time_steps = axes(rate_tensor, 3) |> collect

augmented_rate_matrix(rate_list, time_steps)