using ABMMetastabilityAnalysis

using Test, SparseArrays
using JLD2


# AugmentedMatrix.jl
@testset "Examples from the Augmented Jump Chain paper" begin
   # https://onlinelibrary.wiley.com/doi/full/10.1002/adts.202000274
   # Section 5.1 - A simple two-state model
   Q_first_half = [-1.0 1.0; 1e-6 1e-6]
   Q_second_half = [1e-6 1e-6; 1.0 -1.0]
   list_q = repeat([Q_first_half, Q_second_half]; inner = 4)
   Q = reshape(collect(Iterators.flatten(list_q)), (2, 2, 8))
   t = fill(1.0, 8)

   # Loading ground truth
   J_true = load("J_sikorski_true.jld2")["J"]

   @test augmented_rate_matrix(Q, t) ≈ J_true
end

# Cluster.jl
@testset "Optimal Number of Clusters Tests" begin
   eigenvalues = [7.0, 3.0, 1.0, 0.5]
   expected_clusters = 2
   @test optimal_num_clusters(eigenvalues) == expected_clusters

   # mix of positive and negative eigenvalues
   eigenvalues = [-3.5, -3.0, -0.5, 1.0, 2.0]
   expected_clusters = 3
   @test optimal_num_clusters(eigenvalues) == expected_clusters
end