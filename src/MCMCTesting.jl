
module MCMCTesting

export
    TestSubject,
    TwoSampleTest,
    TwoSampleGibbsTest,
    ExactRankTest,
    sample_predictive,
    sample_joint,
    sample_markov_chain,
    mcmctest,
    seqmcmctest

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

include("twosample.jl")
include("seqtest.jl")

end
