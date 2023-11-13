using ABMMetastabilityAnalysis
using JLD2


precomputed_data = load("rate_matrices/ratemat_seed1.jld2")
rate_tensor = precomputed_data["rates_tensor"]

Pt = mapslices(A -> transpose(exp(A)), rate_tensor[:, :, 1:end-1]; dims=(1, 2))
t = fill(0.01, 199)

augmented_rate_matrix(Pt, t)