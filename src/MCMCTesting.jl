
module MCMCTesting

using Random

function sample_prior_predictive end
function sample_markov_chain end

struct TestSubject{M, K}
    model ::M
    kernel::K
end

include("twosample.jl")

end
