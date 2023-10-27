using LinearAlgebra
using LightGraphs
using SparseArrays

# Various helper functions needed for clustering

# compute connected components for a directed graph, weights represented by C
function connected_sets(C, directed=true)
    M, _ = size(C)
    g = SimpleGraph(M)
    for i in 1:M
        for j in 1:M
            if C[i, j] != 0
                add_edge!(g, i, j)
            end
        end
    end

    components = connected_components(g)
    
    if !directed
        components = [sort(comp) for comp in components]
        sort!(components, by=length, rev=true)
    end
    
    return components
end

function fill_matrix(rot_crop_matrix, eigvectors)
    x, y = size(rot_crop_matrix)
    row_sums = sum(rot_crop_matrix, dims=2)
    row_sums = reshape(row_sums, x, 1)
    
    # add -row_sums as the leftmost column to rot_crop_matrix
    rot_crop_matrix = hcat(-row_sums, rot_crop_matrix)
    
    tmp = -(eigvectors[:, 2:end] * rot_crop_matrix)
    tmp_col_max = maximum(tmp, dims=1)
    tmp_col_max = reshape(tmp_col_max, 1, y + 1)
    tmp_col_max_sum = sum(tmp_col_max)
    
    # add tmp_col_max as the top row to rot_crop_matrix and normalize
    rot_matrix = vcat(tmp_col_max, rot_crop_matrix)
    rot_matrix ./= tmp_col_max_sum
    
    return rot_matrix
end

function compute_stationary_distribution(P::Matrix{Float64})
    n = size(P, 1)
    A = P' - I
    A[n, :] = ones(1, n)
    b = zeros(n)
    b[n] = 1.0
    stationary_distribution = A \ b
    return stationary_distribution
end

function is_transition_matrix(T, tol=1e-12)
    if ndims(T) != 2 || size(T, 1) != size(T, 2)
        return false
    end

    if issparse(T)
        T_csr = convert(SparseMatrixCSC{eltype(T), Int}, T)
        values = nonzeros(T_csr)
    else
        values = T
    end

    is_positive = isapprox(values, abs.(values), rtol=tol)
    is_normed = isapprox(sum(T, dims=2), ones(size(T, 1)), rtol=tol)

    return is_positive && is_normed
end

# computes hitting probabilities for all states to target states
function hitting_probability(T, target::AbstractVector{<:Integer})
    if issparse(T)
        error("Needs to be sparse (no sparse implementation for now)")
    else
        n = size(T, 1)
        nontarget = setdiff(1:n, target)
        stable = findall(x -> isapprox(x, 1.0), diag(T))
        origin = setdiff(nontarget, stable)
        A = T[origin, origin] - I(size(origin, 1))
        b = sum(-T[origin, target], dims=2)
        x = A \ b
        
        xfull = ones(n)
        xfull[origin] = x
        xfull[target] .= 1
        xfull[stable] .= 0
        
        return xfull
    end
end
