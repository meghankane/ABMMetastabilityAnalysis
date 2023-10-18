using ABMMetastabilityAnalysis
using Test

@testset "Examples from the paper" begin
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