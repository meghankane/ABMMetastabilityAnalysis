using ABMMetastabilityAnalysis
using Test, SparseArrays
using JLD2

@testset "Examples from the paper" begin
   # Section 5.1 - A simple two-state model
   Q_first_half = [-1.0 1.0; 1e-6 1e-6]
   Q_second_half = [1e-6 1e-6; 1.0 -1.0]
   list_q = repeat([Q_first_half, Q_second_half]; inner = 4)
   Q = reshape(collect(Iterators.flatten(list_q)), (2, 2, 8))
   t = fill(1.0, 8)

   # Loading ground truth
   J_true = load("J_sikorski_true.jld2")["J"]

   @test augmentedRateMatrix(Q, t) ≈ J_true
end