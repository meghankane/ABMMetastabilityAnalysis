using LinearAlgebra
using Arpack
using Statistics
using SparseArrays
using LightGraphs
using Optim

function optimal_num_clusters(eigenvalues::AbstractVector)
    # find index where the gap between successive eigenvalues is maximized
    gaps = diff(eigenvalues)
    _, index = findmax(abs.(gaps))
    # add 1 due to 0-indexing to get optimal number of clusters 
    opt_num_clusters = index + 1
    return opt_num_clusters
end

function pcca_simple(transition_matrix::AbstractMatrix)
    n_states = size(transition_matrix, 1)

    # verify transition_matrix is row stochastic (sums to 1)
    for i in 1:n_states
        if abs(sum(transition_matrix[i, :]) - 1) >= 1e-10
            throw(ArgumentError("Transition matrix isn't row stochastic. Row $i doesn't sum to 1"))
        end
    end

    # compute eigenvalues & corresponding eigenvectors
    eigenvalues, eigenvectors = eigen(transition_matrix)

    # sort in decreasing order
    # TODO: how to handle complex eigenvalues? sorting will fail
    indices = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    num_clusters = optimal_num_clusters(eigenvalues)
    assignments = assign_cluster_membership(n_states, num_clusters, eigenvectors)

    return assignments, num_clusters
end

# compute membership distributions probabilities that the microstates belong to the same metastable state
# by using the properties of slow processes in eigenvector space
function assign_cluster_membership(n_states, num_clusters, eigenvectors)
    assignments = zeros(Int, n_states)
    for i in 1:n_states
        # assign state i to cluster with largest component in eigenvector matrix
        _, index = findmax(abs.(eigenvectors[i, 2:num_clusters+1]))
        assignments[i] = index
    end
    return assignments
end

# Reimplementation of DeepTime PCCA in Julia
# https://github.com/deeptime-ml/deeptime/blob/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/_pcca.py#L9
function pcca(P::Matrix{Float64}, num_clusters::Int)
    pi = compute_stationary_distribution(P)
    memberships = pcca_memberships(P, num_clusters, pi)
    pi_coarse_grained = memberships' * pi  # coarse-grained stationary distribution

    # HMM output matrix
    HMM_output_mat = Diagonal(1.0 ./ pi_coarse_grained) * memberships' * Diagonal(pi)

    # renormalize B to make it row-stochastic
    HMM_output_mat .= HMM_output_mat ./ sum(HMM_output_mat, dims=2)

    # coarse-grained transition matrix
    W = inv(memberships' * memberships)
    A = memberships' * P * memberships
    P_coarse_grained = W * A

    # symmetrize & renormalize (eliminates numerical errors)
    X = Diagonal(pi_coarse_grained) * P_coarse_grained

    # normalize to get the coarse-grained transition matrix
    P_coarse_grained = X ./ sum(X, dims=2)

    return PCCAModel(P_coarse_grained, pi_coarse_grained, memberships, HMM_output_mat)
end

# Reimplementation of DeepTime PCCAModel in Julia
# https://github.com/deeptime-ml/deeptime/blob/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/_pcca.py#L71
abstract type Model end

struct PCCAModel{T} <: Model
    transition_matrix_coarse::Matrix{T}
    pi_coarse::Vector{T}
    memberships::Matrix{T}
    metastable_distributions::Matrix{T}
    m::Int
end

function PCCAModel(transition_matrix_coarse::Matrix{Float64}, pi_coarse::Vector{Float64}, memberships::Matrix{Float64}, metastable_distributions::Matrix{Float64})
    m = size(memberships, 2)
    return PCCAModel(transition_matrix_coarse, pi_coarse, memberships, metastable_distributions, m)
end

n_metastable(model::PCCAModel) = model.m

function memberships(model::PCCAModel)
    return model.memberships
end

function metastable_distributions(model::PCCAModel)
    return model.metastable_distributions
end

function coarse_grained_transition_matrix(model::PCCAModel)
    return model.transition_matrix_coarse
end

function coarse_grained_stationary_probability(model::PCCAModel)
    return model.pi_coarse
end

function assignments(model::PCCAModel)
    return argmax(model.memberships, dims=2)
end

function sets(model::PCCAModel)
    assignment = assignments(model)
    return [findall(x -> x == i, assignment) for i in 1:n_metastable(model)]
end
