
module MCMCTesting

export
    TestSubject,
    TwoSampleTest,
    TwoSampleGibbsTest,
    ExactRankTest,
    sample_predictive,
    sample_joint,
    sample_markov_chain,
    mcmctest

using Random
using HypothesisTests
using ProgressMeter

function sample_joint            end
function sample_predictive       end
function markovchain_transition  end
function mcmctest                end

struct TestSubject{M, K}
    model ::M
    kernel::K
end

include("twosample.jl")

end
