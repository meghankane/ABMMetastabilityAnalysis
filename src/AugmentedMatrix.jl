using LinearAlgebra, SparseArrays

#Used in one line in function below
@inline function invert_or_zero(x::T) where {T <: Real}
    if iszero(x)
        return zero(T)
    else
        return one(x) / x
    end
end

"""
    augmentedRateMatrix(rates_tensor::Array{T,3}, time_steps::Vector{T})

Computes the augmented rate matrix from several rate matrices and a succession
of time steps. The slices of `rates_tensor` are rate matrices that describe the
rates of transition between the states of a Discretized Markov process at times
for `t` in `time_steps`.

# Notation:
- `M`: is the number of disjoint intervals of time.
- `N`: is the number of states of the Discretized Markov process.
"""
function augmented_rate_matrix(rates_tensor::Array{T,3}, time_steps::Vector{T}) where {T<:Real}
    N, _, M = size(rates_tensor)

    M == length(time_steps) || throw(ArgumentError("More rate matrices than " *
                                                   "time steps were provided."))
    M >= 1 || throw(ArgumentError("At least 2 time steps are needed for the analysis"))


    #qi in the paper is this, but with a minus sign. 
    #Though we never use the negative value so I didn't bother to invert this.
    # FIXME: Given that this is a rate matrix, does this just reduce to the diagonal of the matrix?
    qi = reduce(hcat, sum(Rk .- Diagonal(Rk); dims=2) for Rk = eachslice(rates_tensor; dims=3))

    # qt = copy(list_of_rate_matrices)
    qt = similar(rates_tensor)

    # Conclusion: This is dividing the cols of Rk, when its supposed to dive rowwise.
    for k in 1:M
        qt[:, :, k] = diagm(invert_or_zero.(qi[:, k])) * rates_tensor[:, :, k]
        for i in 1:N
            qt[i, i, k] = iszero(qi[i, k]) ? one(T) : zero(T)
        end
    end

    s = exp.(-time_steps' .* qi)

    # Preallocating vectors II, JJ, VV that make the sparse matrix J
    # They all have size N^2 * binom(N+1, 2) (triangular numbers)
    sz = N^2 * binomial(M+1, 2)
    II = Vector{Int}(undef, sz)
    JJ = Vector{Int}(undef, sz)
    VV = Vector{T}(undef, sz)

    # Fill II, JJ & VV and then construct the sparse matrix with them.
    entry = 1
    for i in 1:N, j in 1:N, l in 1:M
        for k in 1:l-1
            II[entry] = i+(k-1)*N
            JJ[entry] = j+(l-1)*N
            VV[entry] = *(inv(time_steps[k]), qt[i, j, l], inv(qi[i, k]),
                (1 - s[i, k]) * (1 - s[i, l]),
                prod(s[i, m] for m in k+1:l-1; init = one(T))
            )

            entry += 1
        end
        k = l

        II[entry] = i+(k-1)*N
        JJ[entry] = j+(l-1)*N
        VV[entry] = *(inv(time_steps[k]), qt[i, j, k], invert_or_zero(qi[i, k]),
            (s[i, k] + time_steps[k] * qi[i, k] - 1)
        )

        entry += 1
        end
    # Create J directly as sparse, and drop subnormal numbers
    J = sparse(II, JJ, VV)
    droptol!(J, eps(T))
    return J
end