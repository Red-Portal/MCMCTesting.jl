
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
    simulate_ranks

using StatsBase
using Random
using HypothesisTests
using ProgressMeter
using MultipleTesting

function sample_joint            end
function sample_predictive       end
function markovchain_transition  end
function mcmctest                end

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

end
