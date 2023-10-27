# ABMMetastabilityAnalysis

Implementation of the augmented jump chain ([Sikorski, Weber, Schütte 2021](https://onlinelibrary.wiley.com/doi/full/10.1002/adts.202000274)) and spectral clustering using PCCA+ to identify metastable sets. 

After building a simple clustering algorithm, we utilized more sophisticated existing PCCA+ implementations. We ported select functions from the Python packages [DeepTime](https://github.com/deeptime-ml/deeptime/tree/a6ac0b93a55d688fe8f3af119680105763366220) and [MSMTools](https://github.com/markovmodel/msmtools) to Julia so that they could be adapted to our use case and benefit from Julia's performance improvements over Python.

Developed for the application of studying opinion dynamics by identifying the metastable sets of agents in an agent based model.
We then do eigenvalue decomposition on the augmented matrix and utilize spectral clustering (PCCA) to identify metastable sets. Ongoing project notes can be found in our publicly viewable [Overleaf](https://www.overleaf.com/read/mkdzngsprcbd#6735ef).

## Usage

`AugmentedMatrix.jl`: 
```julia
augmentedRateMatrix(rates_tensor::Array{T,3}, time_steps::Vector{T})
```

`Cluster.jl`: 
```julia
pcca(transition_matrix::Matrix{Float64})
```

## Test
Tests can be executed by running `runtests.jl` directly in an IDE (e.g. Visual Studio Code).

Alternatively, from the command line:
```julia
julia> ]
pkg>   test
```

