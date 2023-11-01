using ABMMetastabilityAnalysis
using JLD2

precomputed_data = load("rate_matrices/ratemat_seed1.jld2")
rate_tensor = precomputed_data["rates_tensor"]

t = fill(0.01, 200)

augmented_rate_matrix(rate_tensor, t)