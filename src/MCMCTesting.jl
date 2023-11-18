
module MCMCTesting

export
    TestSubject,
    TwoSampleTest,
    TwoSampleGibbsTest,
    ExactRankTest,
    sample_predictive,
    sample_joint,
    markovchain_transition,
    mcmctest,
    seqmcmctest,
    simulate_ranks,
    rankplot

"""
    sample_joint(rng, model)

Sample from the joint distribution of the prior and the predictive distribution of `model`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model`: Model subject to test.

# Returns
- `θ`: Model parameter sampled from the prior `p(θ)`.
- `y`: Data generated from conditionally on `θ` from `p(y|θ)`
"""
function sample_joint            end

"""
    sample_predictive(rng, model, θ)

Sample from the predictive distribution of `model` conditionally on `θ`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model`: Model subject to test.
- `θ`: Model parameters to condition on.

# Returns
- `y`: Data generated from conditionally on `θ` from `p(y|θ)`
"""
function sample_predictive       end

"""
    markovchain_transition(rng, model, kernel, θ, y)

Perform a single Markov chain transition of `kernel` on the previous state `θ` targeting the posterior of `model` conditioned on `y`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `model`: Model forming the posterior `p(θ|y)` conditioned on `y`.
- `θ`: Previous state of the Markov chain.
- `y`: Data to condition on.

# Returns
- `θ′`: Next state of the Markov chain.
"""
function markovchain_transition  end

"""
    mcmctest([rng,] test, subject; kwargs...)

Sample a p-value according to `test` for `subject`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator. (Default: `Random.default_rng()`.)
- `test::AbstractMCMCTest`: Test strategy.
- `subject::TestSubject`: MCMC algorithm and model subject to test.

# Keyword Arguments
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `statistics`: Function for computing test statistics from samples generated from the tests. (See section below for additional description.)
- Check the documentation for the respective test strategy for additional keyword arugments.

# Custom Test Statistics
The statistics used for the hypothesis tests can be modified by passing a custom funciton to `statistics`.
The default statistics are the first and second moments computed as below.
```julia
statistics = params -> vcat(params, params.^2)
```
The cross-interaction can also be tested by adding an additional entry as below.
```julia
statistics = params -> vcat(params, params.^2, reshape(params*params',:))
```
But naturally, adding more statistics increase the computational cost of computing the tests.

Also, different tests may result in different statistics being computed through the same `statistics` function.
For example, the two-sample test strategies generate both model parameters `θ` and data `y`.
Therefore, `params = vcat(θ, y)`.
On the other hand, the exac rank test only generates model parameters `θ`.
Therefore, `params = θ`.
Naturally, `statistics` can also be used to `select` a subset of parameters used for testing.
For example, for the two-sample test strategies, if we only want to use `θ` for the tests, where `d = length(θ) > 0`, one can do the following:
```julia
statistics = params -> θ[1:d]
```
"""
function mcmctest end

"""
    TestSubject(model, kernel)

Model and MCMC kernel obejct subject to test.

# Arguments
- `model`: Model subject to test.
- `kernel`: MCMC kernel subject to test.
"""
struct TestSubject{M, K}
    model ::M
    kernel::K
end

abstract type AbstractMCMCTest end

function markovchain_multiple_transition(
    rng::Random.AbstractRNG, model, kernel, n_steps::Int, θ, y
)
    for _ = 1:n_steps
        θ = markovchain_transition(rng, model, kernel, θ, y)
    end
    θ
end

include("defaults.jl")
include("twosampletest.jl")
include("exactranktest.jl")
include("seqtest.jl")

struct RankSimulationResult{Ranks}
    ranks::Ranks
    test::ExactRankTest
end

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
            include("../ext/MCMCTestingPlotsExt.jl")
        end
    end
end
end
