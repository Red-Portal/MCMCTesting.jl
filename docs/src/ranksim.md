
# [Visualizing Ranks](@id ranksim)

## Introduction
This section documents how to visualize ranks sampled using Algorithm 2 by Gandy & Scott (2021)[^GS2021].
For more information, refer to the documentation for the [exact rank test](@ref exactrank).

## Simulating Ranks
```@docs
simulate_ranks
```

## Visualizing Ranks with `Plots`
We provide a `Plots` recipe for visualizing the ranks:

```docs
rankplot
```

This can be used as follows:
```julia
using Plots
using MCMCTesting

# Set up the simulation

ranks = simulate_ranks(test, subject)
rankplot(test, ranks; param_names)
```
Also refer to the [tutorial](@ref tutorial) for a working example.

## References
[^GS2021]: Gandy, A., & Scott, J. (2020). Unit testing for MCMC and other Monte Carlo methods. arXiv preprint arXiv:2001.06465.
