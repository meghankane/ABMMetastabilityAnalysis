using LinearAlgebra
using Arpack

function optimal_num_clusters(eigenvalues::Vector{Float64})
    # find index where the gap between successive eigenvalues is maximized
    gaps = diff(eigenvalues)
    _, index = findmax(gaps)
    # add 1 due to 0-indexing to get optimal number of clusters 
    opt_num_clusters = index + 1 
    return opt_num_clusters
end

function pcca(transition_matrix::Matrix{Float64})
    n_states = size(transition_matrix, 1)
    
    # verify transition_matrix is row stochastic (sums to 1)
    for i in 1:n_states
        @assert abs(sum(transition_matrix[i, :]) - 1) < 1e-10 "Row $i of transition matrix doesn't sum to 1"
    end
    
    # compute eigenvalues & corresponding eigenvectors
    eigenvalues, eigenvectors = eigen(transition_matrix)
    
    # sort in decreasing order
    # TODO: how to handle complex eigenvalues? sorting will fail
    indices = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    
    # determine optimal # of clusters
    n_clusters = optimal_num_clusters(eigenvalues)
    
    # assign to clusters
    assignments = zeros(Int, n_states)
    for i in 1:n_states
        # assign state i to cluster with largest component in eigenvector matrix
        _, index = findmax(abs.(eigenvectors[i, 2:n_clusters+1]))
        assignments[i] = index
    end
    
    return assignments, n_clusters
end

# If computeEigen performance isn't sufficient, use optimized eigen decomposition on sparse matrix with Arpack eigs
# num_eigenvalues: and only compute a certain number of eigenvalues & eigenvectors
# Cmputes eigenvalues of largest magnitude by default. See documentation: https://arpack.julialinearalgebra.org/stable/eigs/
function compute_sparse_eigen(matrix::SparseMatrixCSC, num_eigenvalues::Int=3)
    # converts to a sparse matrix (SparseMatrixCSC)
    sparse_matrix = sparse(matrix)
    eigen = eigs(sparse_matrix, nev=num_eigenvalues)
    eigenvalues = eigen[1]
    eigenvectors = eigen[2]
    return eigenvalues, eigenvectors
end
