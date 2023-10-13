using LinearAlgebra

#Used in one line in function below
function invert_or_zero(x :: Real)
    if x == 0
        return 0
    else
        return 1/x
    end
end

#Code to compute augmented rate matrix
function augmentedRateMatrix(list_of_rate_matrices :: Array{B}, list_of_time_steps :: Array{T}) where {T <: Real, B <: Matrix{T}}
    number_of_times = length(list_of_time_steps)
    @assert length(list_of_rate_matrices) == number_of_times
    @assert length(list_of_rate_matrices) ≥ 1
    number_of_states = size(list_of_rate_matrices[1])[1]

    #Big in terms of memory.
    J = zeros(number_of_states*number_of_times, number_of_states*number_of_times)

    #q in the paper is this, but with a minus sign. 
    #Though we never use the negative value so I didn't bother to invert this.
    q_positive = Array{Array{Real}}(undef,number_of_times)
    for k in 1:number_of_times
        q_positive[k] = [ sum(list_of_rate_matrices[k][i,j] for j in vcat(1:i-1,i+1:number_of_states)) for i in 1:number_of_states]
    end
    q_tilde_positive = copy(list_of_rate_matrices)
    for k in 1:number_of_times
        q_tilde_positive[k] = q_tilde_positive[k]*diagm(invert_or_zero.(q_positive[k]))
        for i in 1:number_of_states
            q_tilde_positive[k][i,i] = q_positive[k][i] == 0 ? 1 : 0 
        end
    end

    #Quick way to compute s, as denoted in the paper.
    #Since we took q_positive there is no need for the minus sign from the paper.
    s = exp.(hcat(q_positive.*list_of_time_steps...))

    #Optimize me! (Perhaps a GPU kernel could do this quickly o.O)
    for i in 1:number_of_states
        for j in 1:number_of_states
            for l in 1:number_of_times
                for k in 1:l-1
                    J[i+(k-1)*number_of_times,j+(l-1)*number_of_times] = *(list_of_time_steps[k]^(-1),
                        q_tilde_positive[l][i,j],
                        q_positive[k][i]^(-1),
                        (1-s[i,k])*(1-s[i,l]),
                        prod(s[i,m] for m in k:l)
                    )
                end
                k = l
                J[i+(k-1)*number_of_times,j+(l-1)*number_of_times] = *(list_of_time_steps[k]^(-1),
                q_tilde_positive[l][i,j],
                q_positive[k][i]^(-1),
                (s[i,k]-list_of_time_steps[k]*q_positive[k][i]-1)
                )
            end
        end
    end
    return J
end 