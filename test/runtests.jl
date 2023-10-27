using ABMMetastabilityAnalysis
using SparseArrays
using Test

# AugmentedMatrix.jl
@testset "Examples from the Augmented Jump Chain paper" begin
   # https://onlinelibrary.wiley.com/doi/full/10.1002/adts.202000274
   # Section 5.1 - A simple two-state model
   i = [1, 1]
   j = [1, 2]
   v = [-1.0, 1.0]
   Q_first_half = sparse(i, j, v, 2, 2)
   ii = [2, 2]
   vv = [1.0, -1.0]
   Q_second_half = sparse(ii, j, vv, 2, 2)
   list_q = repeat([Q_first_half, Q_second_half]; inner = 4)
   Q = reshape(collect(Iterators.flatten(list_q)), (2, 2, 8))
   t = fill(1.0, 8)
end