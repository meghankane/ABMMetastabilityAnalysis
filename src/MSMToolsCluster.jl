using LinearAlgebra
using Arpack
using Statistics
using SparseArrays
using LightGraphs
using Optim

# NOTE: Reimplementation in Julia for select functions from MSM Tools (referenced in DeepTime)
# https://github.com/markovmodel/msmtools
# https://github.com/deeptime-ml/deeptime/tree/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/tools

# Reimplementation of PCCA+ spectral clustering method with optimized memberships
# https://github.com/deeptime-ml/deeptime/blob/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/tools/analysis/dense/_pcca.py#L288
function pcca_memberships(P::Matrix{Float64}, m::Int, π::Vector{Float64}=nothing)
    n = size(P, 1)
    if m > n
        throw(ArgumentError("Number of metastable states m=$m exceeds number of states of transition matrix n=$n"))
    end
    println("P: $P")
    if !is_transition_matrix(P)
        throw(ArgumentError("Input matrix is not a transition matrix."))
    end
    if isnothing(π)
        if ndims(π) > 1
            throw(ArgumentError("Expected 1D stationary distribution array"))
        elseif length(π) != n
            throw(ArgumentError("Stationary distribution must be defined on entire space, piecewise if the transition matrix " *
            "has multiple connected components. It covered $(length(π)) != $n states."))
        end
    end

    chi = zeros(n, m)
    components = connected_sets(P)
    closed_components = []
    transition_states = []
    for component in components
        rest = setdiff(1:n, component)
        # if component is closed, store with positive equilibrium distribution
        if isapprox(sum(P[component, rest]), 0, atol=0.0001)
            push!(closed_components, component)
        # otherwise store as transition states with vanishing equilibrium distribution
        else
            push!(transition_states, component)
        end
    end

    n_closed_components = length(closed_components)
    closed_states = reduce(vcat, closed_components)
    if isempty(transition_states)
        transition_states = Int[]
    else
        transition_states = vcat(transition_states...)
    end

    # check if we have enough clusters to support the disconnected sets
    if m < n_closed_components
        throw(error("Number of metastable states m=$m is too small. Transition matrix " *
              "has $n_closed_components disconnected components."))
    end

    closed_components_Psub = []
    closed_components_ev = []
    closed_components_enum = []
    for i in 1:n_closed_components
        component = closed_components[i]
        Psub = P[component, component] # compute eigenvalues in submatrix
        push!(closed_components_Psub, Psub)
        push!(closed_components_ev, eigen(Psub).values)
        push!(closed_components_enum, fill(i, length(component)))
    end

    closed_components_ev_flat = vcat(closed_components_ev...)
    closed_components_enum_flat = vcat(closed_components_enum...)

    # cluster each component
    component_indexes = closed_components_enum_flat[sortperm(closed_components_ev_flat)][1:m]
    ipcca = 1
    for i in 1:n_closed_components
        component = closed_components[i]
        m_by_component = count(x -> x == i, component_indexes) # number of PCCA states per component

        if m_by_component == 1 # trivial case
            chi[component, ipcca] .= 1.0
            ipcca += 1
        elseif m_by_component > 1
            chi[component, ipcca:(ipcca + m_by_component - 1)] .= pcca_connected(
                closed_components_Psub[i], m_by_component, π == nothing ? nothing : π[component]
            )
            ipcca += m_by_component
        else
            throw(error("Did not expect component $i to have $m_by_component."))
        end
    end

    # assign transition states
    if !isempty(transition_states)
        # make all closed states absorbing, so we can see which closed state we hit first
        Pabs = copy(P)
        Pabs[closed_states, :] .= 0.0
        Pabs[closed_states, closed_states] .= 1.0
        for i in closed_states
            h = hitting_probability(Pabs, i) # hitting probability to each closed state
            for j in transition_states
                chi[j, :] .+= h[j] .* chi[i, :]
            end
        end
    end

    nmeta = count(x -> x != 0, sum(chi, dims=1))
    if nmeta < m
        throw(error("Requested $m metastable states, but transition matrix only has $nmeta. " *
                "Try requesting fewer metastable states."))
    end

    return chi
end

# Reimplementation of PCCA+ spectral clustering method with optimized memberships (assumes fully connected transition matrix)
# https://github.com/deeptime-ml/deeptime/blob/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/tools/analysis/dense/_pcca.py#L197
function pcca_connected(P, n, pi=nothing)
    labels = connected_sets(P)
    n_components = length(labels)
    if n_components > 1
        throw(error("Unable to use pcca_connected, transition matrix is not fully connected."))
    end

    if pi === nothing
        pi = stationary_distribution(P)
    else
        if size(pi, 1) != size(P, 1)
            throw(error("Stationary distribution must span the entire state space but got $(size(pi, 1)) states " *
                                "instead of $(size(P, 1))."))
        end
        pi ./= sum(pi)  # normalization
    end

    eigenvalues, eigenvectors = eigen(P)
    indices = sortperm(eigenvalues, rev=true)
    evecs = eigenvectors[:, indices]

    # orthonormalize
    for i in 1:n
        evecs[:, i] /= sqrt(dot(evecs[:, i] .* pi, evecs[:, i]))
    end

    # make 1st eigenvector positive
    evecs[:, 1] = abs.(evecs[:, 1])

    if any(x -> !isreal(x), evecs)
        println("Transition matrix has complex eigenvectors. Forcing eigenvectors to be real (see DeepTime note).")
    end
    evecs = real.(evecs)

    # start with initial PCCA+ solution (can contain negative memberships)
    chi, rot_matrix = pcca_connected_isa(evecs, n)

    # optimizes PCCA+ rotation matrix s.t. memberships are always nonnegative
    rot_matrix = opt_soft(evecs, rot_matrix, n)
    memberships = evecs[:, :] * rot_matrix

    # force memberships to be in range [0,1] (due to numerical errors)
    memberships = clamp.(memberships, 0.0, 1.0)

    for i in 1:size(memberships, 1)
        memberships[i, :] ./= sum(memberships[i, :])
    end

    return memberships
end

# Reimplementation of PCCA+ spectral clustering method using the inner simplex algorithm.
# https://github.com/deeptime-ml/deeptime/blob/a6ac0b93a55d688fe8f3af119680105763366220/deeptime/markov/tools/analysis/dense/_pcca.py#L14
function pcca_connected_isa(eigenvectors, num_clusters, tol=1e-6)
    n, m = size(eigenvectors)
    if num_clusters > m
        throw(ArgumentError("Cannot cluster the ($n x $m) eigenvector matrix to $num_clusters clusters."))
    end

    # Check if (ONLY) 1st eigenvector is constant
    diffs = abs.(maximum(eigenvectors, dims=1) .- minimum(eigenvectors, dims=1))
    if diffs[1] >= tol
        throw(error("Unable to do PCCA. 1st eigenvector is not constant. The transition matrix is not connected or the eigenvectors are incorrectly sorted."))
    end
    if diffs[2] <= tol
        throw(error("Unable to do PCCA. An eigenvector after 1st eigenvector is constant. Potentially sorting of eigenvectors issue."))
    end

    c = eigenvectors[:, 1:num_clusters] # copy of the eigenvectors
    ortho_sys = copy(c)
    max_dist = 0.0

    ind = zeros(Int, num_clusters) # representative states

    # select 1st representative as most distant point
    for i in 1:num_clusters
        for j in 1:size(c, 1)
            if norm(c[j, :]) > max_dist
                max_dist = norm(c[j, :])
                ind[1] = j
            end
        end
    end

    # translate coordinates to make the first representative the origin
    ortho_sys .-= c[:, ind[1] + 1]

    # select remaining num_clusters - 1 representatives as orthogonal to each other as possible (using Gram-Schmidt)
    for k in 2:num_clusters
        max_dist = 0.0
        temp = copy(ortho_sys[ind[k - 1], :])
        # select next farthest point that is not yet a representative
        for i in 1:size(ortho_sys, 1)
            row = copy(ortho_sys[i, :])
            row .-= (temp' * row) .* temp
            distt = norm(row, 2)
            if distt > max_dist && i ∉ ind[1:k-1]
                max_dist = distt
                ind[k] = i
            end
        end
        ortho_sys[ind[k], :] ./= norm(ortho_sys[ind[k], :], 2)
    end

    # transformation matrix of eigenvectors to the membership matrix
    rot_mat = inv(c[ind, :])
    # membership matrix
    chi = c * rot_mat

    return chi, rot_mat
end

function susanna_func(rot_crop_vec, num_clusters, eigvectors, x, y)
    rot_crop_matrix = reshape(rot_crop_vec, x, y)
    rot_matrix = fill_matrix(rot_crop_matrix, eigvectors)
    
    result = 0.0
    for i in 1:num_clusters
        for j in 1:num_clusters
            result += rot_matrix[j, i]^2 / rot_matrix[1, i]
        end
    end
    return -result
end

function opt_soft(eigenvectors, rot_matrix, num_clusters)
    eigenvectors = eigenvectors[:, 1:num_clusters]
    rot_crop_matrix = rot_matrix[2:end, 2:end]
    
    x, y = size(rot_crop_matrix)
    rot_crop_vec = reshape(rot_crop_matrix, x * y)

    result = optimize(opt -> susanna_func(opt, num_clusters, eigenvectors, x, y), rot_crop_vec, LBFGS(), Optim.Options(show_trace=false))

    rot_crop_matrix = reshape(Optim.minimizer(result), x, y)
    rot_matrix = fill_matrix(rot_crop_matrix, eigenvectors)

    return rot_matrix
end
