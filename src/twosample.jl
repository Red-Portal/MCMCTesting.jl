
struct TwoSampleTest
    n_control   ::Int
    n_treatment ::Int
    n_mcmc_steps::Int
end

function markovchain_multiple_transition(
    rng::Random.AbstractRNG, subject::TestSubject, n_steps::Int, θ, y
)
    for _ = 1:n_steps
        θ = markovchain_transition(rng, subject, θ, y)
    end
    θ
end

"""
# Two-Sample Test 

Algorithm 1 in Gandy & Scott 2021.
"""
function simulate(rng::Random.AbtractRNG, test::TwoSampleTest, subject::TestSubject)
    # Treatment Group
    trtm = map(hcat, 1:test.n_treatment) do _
        θ, y   = sample_prior_predictive(rng, subject)
        θ_trtm = markovchain_multiple_transition(rng, subject, test.n_mcmc_steps, θ, y)
        vcat(θ_trtm, y)
    end

    # Control Group
    ctrl = map(hcat, 1:test.n_control) do _
        θ_ctrl, y = sample_prior_predictive(rng, subject)
        vcat(θ_ctrl, y)
    end
    trtm, ctrl
end

function default_two_sample_test(x, y)
    @error("The default two-sample test only supports real vectors. Please supply your own compute_pvalue function.")
end

function default_two_sample_test(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    HypothesisTests.pvalue(
        HypothesisTests.ApproximateTwoSampleKSTest(x, y)
    )
end

function test(
    rng    ::Random.AbtractRNG,
    test   ::TwoSampleTest,
    subject::TestSubject;
    compute_pvalue = default_two_sample_test,
)
    trtm, ctrl = simulate(rng, test, subject)
    pvals      = map(trtm, ctrl) do param_trtm, param_ctrl  
        compute_pvalue(param_trtm, param_ctrl)
    end
end
