using LinearAlgebra, SparseArrays

#Used in one line in function below
function invert_or_zero(x::Real)
    if x == 0
        return 0
    else
        return 1 / x
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
function augmentedRateMatrix(rates_tensor::Array{T,3}, time_steps::Vector{T}) where {T<:Real}
    N, _, M = size(rates_tensor)

    M == length(time_steps) || throw(ArgumentError("More rate matrices than " *
                                                   "time steps were provided."))
    M >= 1 || throw(ArgumentError("At least 2 time steps are needed for the analysis"))


    # Pre-allocating the final matrix J
    J = spzeros(T, N * M, N * M)

    #q in the paper is this, but with a minus sign. 
    #Though we never use the negative value so I didn't bother to invert this.
    # FIXME: Given that this is a rate matrix, does this just reduce to the diagonal of the matrix?
    q_positive = reduce(hcat, sum(Rk .- Diagonal(Rk); dims=2) for Rk = eachslice(rates_tensor; dims=3))

    # q_tilde_positive = copy(list_of_rate_matrices)
    q_tilde_positive = Array{T, 3}(undef, N, N, M)

    for k in 1:M
        q_tilde_positive[:, :, k] = rates_tensor[:, :, k] * diagm(invert_or_zero.(q_positive[:, k]))
        for i in 1:N
            q_tilde_positive[i, i, k] = iszero(q_positive[i, k]) ? 1 : 0
        end
    end

    #Quick way to compute s, as denoted in the paper.
    #Since we took q_positive there is no need for the minus sign from the paper.
    # The paper references s_{ik} = exp(-ΔT_k * q_i (t)), does this correspond?
<<<<<<< HEAD
    s = exp.(-time_steps' .* q_positive)
=======
    s = exp.(-1 .* time_steps' .* q_positive)
>>>>>>> 2ae0b4f (Fixing jump chain operator)

    #Optimize me! (Perhaps a GPU kernel could do this quickly o.O)
    for i in 1:N
        for j in 1:N
            for l in 1:M
                for k in 1:l-1
                    J[i+(k-1)*N, j+(l-1)*N] = *(time_steps[k]^(-1),
                        q_tilde_positive[i, j, l],
                        invert_or_zero(q_positive[i, k]),
                        (1 - s[i, k]) * (1 - s[i, l]),
                        prod(s[i, m] for m in k:l)
                    )
                end
                k = l
                J[i+(k-1)*N, j+(l-1)*N] = *(time_steps[k]^(-1),
                    q_tilde_positive[i, j, k],
                    invert_or_zero(q_positive[i, k]),
                    # FIXME: This has an extra minus in ΔT_k ...
                    (s[i, k] + time_steps[k] * q_positive[i, k] - 1)
                )
            end
        end
    end
    return J, q_positive, q_tilde_positive, s
end