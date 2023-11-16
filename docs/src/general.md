
# General Usage
## Introduction

The tests provided by `MCMCTesting` are frequentist hypothesis tests for testing the correctness of MCMC kernels.
In particular, it compute the p-value for the null hypothesis that the MCMC kernel has the correct stationary distribution against the alternative hypothesis that it doesn't.

Currently, `MCMCTesting` provide three different tests originally proposed by Gandy and Scott[^gandyandscott2021]: 
1. [Simple Two-Sample Test](@ref twosample)
2. [Two-Sample Test with an Additional Gibbs Step](@ref twosamplegibbs)
3. [Exact Rank Test](@ref exactrank)

The two-sample tests are generally applicable. 
On the other hand, the exact rank test assumes that the MCMC kernel is reversible.
Therefore, it can specifically be used to test reversibility.

## Interface
The user needs to implement the following function specializations for the `model` and `kernel` subject to the test.
```@docs
sample_joint
markovchain_transition
```
Some tests might be require additional interfaces to be implemented.
For an overview of how to implement these interfaces, refer to the [tutorial](@ref tutorial).

The `model` and `kernel` are then passed to `MCMCTesting` through the following struct:
```@docs
TestSubject
```

## Simulating a P-Value Through `mcmctest`
Each of the test internally run simulations and compute a single p-value through the following routine:
```@docs
mcmctest
```

## Increasing Power Through `seqmcmctest`
`seqmcmctest` (Algorithm 3[^gandyandscott2021]) sequentially calls `mcmctest` to increase the power and ensure a low false rejection rate.
Furthermore, the p-values from each component of the statistics are combined through multiple hypothesis adjustment.

```@docs
seqmcmctest
```

## References
[^gandyandscott2021]: Gandy, A., & Scott, J. (2020). Unit testing for MCMC and other Monte Carlo methods. arXiv preprint arXiv:2001.06465.
