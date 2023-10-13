using LinearAlgebra

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
    J = zeros(T, N * M, N * M)

    #q in the paper is this, but with a minus sign. 
    #Though we never use the negative value so I didn't bother to invert this.
    q_positive = Array{Array{Real}}(undef, M)
    for k in 1:M
        q_positive[k] = [sum(rates_tensor[i, j, k] for j = setdiff(1:N, i)) for i = 1:N]
    end
    return q_positive
    # q_positive = reduce(hcat, sum(Rk .- Diagonal(Rk); dims=2) for Rk = eachslice(rates_tensor; dims=3))
    q_tilde_positive = copy(list_of_rate_matrices)

    for k in 1:M
        # FIXME: Why are we referencing q_tilde on the lhs & rhs? Is this just q_positive?
        q_tilde_positive[:, k] = q_tilde_positive[:, k] * diagm(invert_or_zero.(q_positive[:, k]))
        for i in 1:N
            # FIXME: Why two indices to access q_tilde[k]? its 1-dimensional
            q_tilde_positive[k][i, i] = q_positive[k][i] == 0 ? 1 : 0
        end
    end

    #Quick way to compute s, as denoted in the paper.
    #Since we took q_positive there is no need for the minus sign from the paper.
    # The paper references s_{ik} = exp(-ΔT_k * q_i (t)), does this correspond?
    s = exp.(hcat(q_positive .* list_of_time_steps...))

    #Optimize me! (Perhaps a GPU kernel could do this quickly o.O)
    for i in 1:number_of_states
        for j in 1:number_of_states
            for l in 1:number_of_times
                for k in 1:l-1
                    J[i+(k-1)*number_of_states, j+(l-1)*number_of_states] = *(list_of_time_steps[k]^(-1),
                        q_tilde_positive[l][i, j],
                        q_positive[k][i]^(-1),
                        (1 - s[i, k]) * (1 - s[i, l]),
                        prod(s[i, m] for m in k:l)
                    )
                end
                k = l
                J[i+(k-1)*number_of_states, j+(l-1)*number_of_states] = *(list_of_time_steps[k]^(-1),
                    q_tilde_positive[l][i, j],
                    q_positive[k][i]^(-1),
                    (s[i, k] - list_of_time_steps[k] * q_positive[k][i] - 1)
                )
            end
        end
    end
    return J
end