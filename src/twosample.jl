
function markovchain_multiple_transition(
    rng::Random.AbstractRNG, model, kernel, n_steps::Int, θ, y
)
    for _ = 1:n_steps
        θ = markovchain_transition(rng, model, kernel, θ, y)
    end
    θ
end

struct TwoSampleTest
    n_control   ::Int
    n_treatment ::Int
    n_mcmc_steps::Int
end

struct TwoSampleGibbsTest
    n_control   ::Int
    n_treatment ::Int
    n_mcmc_steps::Int
end

"""
# Two-Sample Test 

Algorithm 1 in Gandy & Scott 2021.
"""
function simulate(
    rng          ::Random.AbstractRNG,
    test         ::TwoSampleTest,
    subject      ::TestSubject,
    show_progress::Bool
)
    model  = subject.model
    kernel = subject.kernel
    n_trtm = test.n_treatment
    n_ctrl = test.n_control
    prog   = ProgressMeter.Progress(
        n_ctrl + n_trtm; barlen = 31, showspeed = true, enabled = show_progress
    )

    # Treatment Group
    trtm   = map(1:n_trtm) do n
        θ, y   = sample_joint(rng, model)
        θ_trtm = markovchain_multiple_transition(rng, model, kernel, test.n_mcmc_steps, θ, y)
        next!(prog, showvalues=[(:treatment_group, "$(n)/$(n_trtm)")])
        vcat(θ_trtm, y)
    end

    # Control Group
    ctrl = map(1:n_ctrl) do n
        θ_ctrl, y = sample_joint(rng, model)
        next!(prog, showvalues=[(:control_group, "$(n)/$(n_ctrl)")])
        vcat(θ_ctrl, y)
    end
    trtm, ctrl
end

"""
# Two-Sample Test with Gibbs-like Sample Generation

Modified Algorithm 1 in Gandy & Scott 2021.
"""
function simulate(
    rng          ::Random.AbstractRNG,
    test         ::TwoSampleGibbsTest,
    subject      ::TestSubject,
    show_progress::Bool
)
    model  = subject.model
    kernel = subject.kernel
    n_trtm = test.n_treatment
    n_ctrl = test.n_control
    prog   = ProgressMeter.Progress(
        n_ctrl + n_trtm; barlen = 31, showspeed = true, enabled = show_progress
    )

    # Treatment Group
    trtm = map(1:n_trtm) do n
        θ, y   = sample_joint(rng, model)
        θ_trtm = markovchain_multiple_transition(rng, model, kernel, test.n_mcmc_steps, θ, y)
        y_trtm = sample_predictive(rng, model, θ_trtm)
        next!(prog, showvalues=[(:treatment_group, "$(n)/$(n_trtm)")])
        vcat(θ_trtm, y_trtm)
    end

    # Control Group
    ctrl = map(1:n_ctrl) do n
        θ_ctrl, y_ctrl = sample_joint(rng, model)
        next!(prog, showvalues=[(:control_group, "$(n)/$(n_ctrl)")])
        vcat(θ_ctrl, y_ctrl)
    end
    trtm, ctrl
end

function default_two_sample_test_pvalue(x, y)
    @error("The default two-sample test only supports real vectors. " *
           "Please supply your own two_sample_test_pvalue function.")
end

function default_two_sample_test_pvalue(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    HypothesisTests.pvalue(
        HypothesisTests.ApproximateTwoSampleKSTest(x, y)
    )
end

function default_statistics(x::AbstractVector{<:Real})
    vcat(x, x.^2)
end

function mcmctest(
    rng    ::Random.AbstractRNG,
    test   ::Union{TwoSampleTest, TwoSampleGibbsTest},
    subject::TestSubject;
    two_sample_test_pvalue = default_two_sample_test_pvalue,
    statistics             = default_statistics,
    show_progress::Bool    = true,
)
    trtm, ctrl = simulate(rng, test, subject, show_progress)
    stats_trtm = map(statistics, trtm)
    stats_ctrl = map(statistics, ctrl)

    stats_trtm_cat = hcat(stats_trtm...)
    stats_ctrl_cat = hcat(stats_ctrl...)

    pvals = map(eachrow(stats_trtm_cat), eachrow(stats_ctrl_cat)) do param_trtm, param_ctrl  
        two_sample_test_pvalue(param_trtm, param_ctrl)
    end
end

function mcmctest(
    test   ::Union{TwoSampleTest, TwoSampleGibbsTest},
    subject::TestSubject;
    two_sample_test_pvalue = default_two_sample_test_pvalue,
    statistics             = default_statistics,
    show_progress::Bool    = true,
)
    mcmctest(Random.default_rng(), test, subject;
             two_sample_test_pvalue, statistics, show_progress)
end
