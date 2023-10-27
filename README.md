# ABMMetastabilityAnalysis

Implementation of the augmented jump chain ([Sikorski, Weber, Schütte 2021](https://onlinelibrary.wiley.com/doi/full/10.1002/adts.202000274)) and spectral clustering using PCCA+ to identify metastable sets. 

After building a simple clustering algorithm, we utilized more sophisticated existing PCCA+ implementations. We ported select functions from the Python packages [DeepTime](https://github.com/deeptime-ml/deeptime/tree/a6ac0b93a55d688fe8f3af119680105763366220) and [MSMTools](https://github.com/markovmodel/msmtools) to Julia so that they could be adapted to our use case and benefit from Julia's performance improvements over Python.

This was developed for the application of studying opinion dynamics where we aim to identify the metastable sets of agents in an agent based model. Our ongoing project notes can be found in our publicly viewable [Overleaf](https://www.overleaf.com/read/mkdzngsprcbd#6735ef).

## Usage

### Entrypoints
To compute the augmented rate matrix, reference `AugmentedMatrix.jl`: 
```julia
augmentedRateMatrix(rates_tensor::Array{T,3}, time_steps::Vector{T})
```

To use PCCA+ on the transition matrix, $P$. With the `PCCAModel` returned from the `pcca` function, access the metastable sets with `sets(model)`. Reference `Cluster.jl`: 
```julia
pcca(P::Matrix{Float64}, num_clusters::Int)
```

## Test
Tests can be executed by running `runtests.jl` directly in an IDE (e.g. Visual Studio Code).

Alternatively, from the command line:
```julia
julia> ]
pkg>   test
```

