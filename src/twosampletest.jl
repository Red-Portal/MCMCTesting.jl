
struct TwoSampleTest <: AbstractMCMCTest
    n_control   ::Int
    n_treatment ::Int
    n_mcmc_steps::Int
    n_mcmc_thin ::Int
end

function TwoSampleTest(
    n_samples   ::Int,
    n_mcmc_steps::Int;
    n_control   ::Int = n_samples,
    n_treatment ::Int = n_samples,
    n_mcmc_thin ::Int = 1,
)
    @assert n_control    > 1
    @assert n_treatment  > 1
    @assert n_mcmc_steps ≥ 1
    @assert n_mcmc_thin  ≥ 1
    TwoSampleTest(n_control, n_treatment, n_mcmc_steps, n_mcmc_thin)
end

struct TwoSampleGibbsTest <: AbstractMCMCTest
    n_control   ::Int
    n_treatment ::Int
    n_mcmc_steps::Int
    n_mcmc_thin ::Int
end

function TwoSampleGibbsTest(
    n_samples   ::Int,
    n_mcmc_steps::Int;
    n_control   ::Int = n_samples,
    n_treatment ::Int = n_samples,
    n_mcmc_thin ::Int = 1,
)
    @assert n_control    > 1
    @assert n_treatment  > 1
    @assert n_mcmc_steps ≥ 1
    @assert n_mcmc_thin  ≥ 1
    TwoSampleGibbsTest(n_control, n_treatment, n_mcmc_steps, n_mcmc_thin)
end

"""
# Two-Sample Test 

Algorithm 1 in Gandy & Scott 2021.
"""
function simulate_two_sample_groups(
    rng          ::Random.AbstractRNG,
    test         ::TwoSampleTest,
    subject      ::TestSubject,
    show_progress::Bool
)
    model        = subject.model
    kernel       = subject.kernel
    n_trtm       = test.n_treatment
    n_ctrl       = test.n_control
    n_mcmc_steps = test.n_mcmc_steps
    n_mcmc_thin  = test.n_mcmc_thin

    prog   = ProgressMeter.Progress(
        n_ctrl + n_trtm; barlen = 31, showspeed = true, enabled = show_progress
    )

    # Treatment Group
    trtm   = map(1:n_trtm) do n
        θ, y   = sample_joint(rng, model)
        θ_trtm = markovchain_multiple_transition(
            rng, model, kernel, n_mcmc_steps*n_mcmc_thin, θ, y
        )
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

function simulate_two_sample_groups(
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

    n_mcmc_steps = test.n_mcmc_steps
    n_mcmc_thin  = test.n_mcmc_thin

    # Treatment Group
    θs_trtm = map(1:n_trtm) do n
        θ, y   = sample_joint(rng, model)
        θ_trtm = markovchain_multiple_transition(
            rng, model, kernel, n_mcmc_steps*n_mcmc_thin, θ, y
        )
        y_trtm = sample_predictive(rng, model, θ_trtm)
        next!(prog, showvalues=[(:treatment_group, "$(n)/$(n_trtm)")])
        vcat(θ_trtm, y_trtm)
    end

    # Control Group
    θs_ctrl = map(1:n_ctrl) do n
        θ_ctrl, y_ctrl = sample_joint(rng, model)
        next!(prog, showvalues=[(:control_group, "$(n)/$(n_ctrl)")])
        vcat(θ_ctrl, y_ctrl)
    end
    θs_trtm, θs_ctrl
end

"""
# Two-Sample Test with Gibbs-like Sample Generation

Modified Algorithm 1 in Gandy & Scott 2021.
"""
function mcmctest(
    rng    ::Random.AbstractRNG,
    test   ::Union{TwoSampleTest, TwoSampleGibbsTest},
    subject::TestSubject;
    two_sample_test_pvalue = default_two_sample_test_pvalue,
    statistics             = default_statistics,
    show_progress::Bool    = true,
)
    θs_trtm, θs_ctrl = simulate_two_sample_groups(rng, test, subject, show_progress)
    stats_trtm = map(statistics, θs_trtm)
    stats_ctrl = map(statistics, θs_ctrl)

    stats_trtm_cat = hcat(stats_trtm...)
    stats_ctrl_cat = hcat(stats_ctrl...)

    map(eachrow(stats_trtm_cat), eachrow(stats_ctrl_cat)) do param_trtm, param_ctrl  
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
