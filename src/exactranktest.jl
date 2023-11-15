
struct ExactRankTest <: AbstractMCMCTest
    n_samples   ::Int
    n_mcmc_steps::Int
    n_mcmc_thin ::Int
end

function ExactRankTest(
    n_samples   ::Int,
    n_mcmc_steps::Int;
    n_mcmc_thin ::Int = 1,
)
    @assert n_samples    > 1
    @assert n_mcmc_steps > 1
    @assert n_mcmc_thin  ≥ 1
    ExactRankTest(n_samples, n_mcmc_steps, n_mcmc_thin)
end

function simulate_rank(
    rng          ::Random.AbstractRNG,
    test         ::ExactRankTest,
    subject      ::TestSubject,
    statistics,
    tie_epsilon
)
    model        = subject.model
    kernel       = subject.kernel
    n_mcmc_steps = test.n_mcmc_steps
    n_mcmc_thin  = test.n_mcmc_thin

    idx_mid    = sample(rng, 1:n_mcmc_steps)
    n_mcmc_fwd = n_mcmc_steps - idx_mid
    n_mcmc_bwd = idx_mid - 1

    θ_mid, y  = sample_joint(rng, model)
    stat_mid  = statistics(θ_mid)
    rank_wins = ones( Int, length(stat_mid))
    rank_ties = zeros(Int, length(stat_mid))

    # Forward Transitions
    θ_fwd  = copy(θ_mid)
    for _ in 1:n_mcmc_fwd
        θ_fwd = markovchain_multiple_transition(
            rng, model, kernel, n_mcmc_thin, θ_fwd, y
        )
        rank_wins += statistics(θ_fwd) .< (stat_mid .- tie_epsilon) 
        rank_ties += abs.(stat_mid - statistics(θ_fwd)) .≤ tie_epsilon
    end

    # Backward Transition
    θ_bwd  = copy(θ_mid)
    for _ in 1:n_mcmc_bwd
        θ_bwd = markovchain_multiple_transition(
            rng, model, kernel, n_mcmc_thin, θ_bwd, y
        )
        rank_wins += statistics(θ_bwd) .< (stat_mid .- tie_epsilon) 
        rank_ties += abs.(stat_mid - statistics(θ_bwd)) .≤ tie_epsilon
    end

    # Tie Resolution
    map(rank_wins, rank_ties) do rank_wins_param, rank_ties_param
        rank_wins_param + sample(rng, 1:rank_ties_param+1)
    end
end

function compute_rank_count(ranks::AbstractVector{Int}, maxrank::Int)
    param_freq = zeros(Int, maxrank)
    for rank in ranks
        param_freq[rank] += 1
    end
    param_freq
end

function simulate_ranks(
    rng          ::Random.AbstractRNG,
    test         ::ExactRankTest,
    subject      ::TestSubject;
    statistics          = default_statistics,
    show_progress::Bool = true,
    tie_epsilon         = eps(Float64)
)
    n_samples    = test.n_samples

    prog = ProgressMeter.Progress(
        n_samples; barlen = 31, showspeed = true, enabled = show_progress
    )

    mapreduce(hcat, 1:n_samples) do n
        next!(prog, showvalues=[(:simulated_ranks, "$(n)/$(n_samples)")])
        simulate_rank(rng, test, subject, statistics, tie_epsilon)
    end
end

function simulate_ranks(
    test         ::ExactRankTest,
    subject      ::TestSubject;
    statistics          = default_statistics,
    show_progress::Bool = true,
)
    simulate_ranks(Random.default_rng(), test,subject; statistics, show_progress)
end

"""
# Exact Rank Test 

Algorithm 2 in Gandy & Scott 2021.

Assumes that the mcmc kernel is reversible.
"""
function mcmctest(
    rng    ::Random.AbstractRNG,
    test   ::ExactRankTest,
    subject::TestSubject;
    statistics             = default_statistics,
    uniformity_test_pvalue = default_uniformity_test_pvalue,
    show_progress::Bool    = true,
)
    n_mcmc_steps = test.n_mcmc_steps
    ranks = simulate_ranks(rng, test, subject; statistics, show_progress)
    map(eachrow(ranks)) do param_ranks
        param_freqs = compute_rank_count(param_ranks, n_mcmc_steps)
        uniformity_test_pvalue(param_freqs)
    end
end

function mcmctest(
    test   ::ExactRankTest,
    subject::TestSubject;
    statistics             = default_statistics,
    show_progress::Bool    = true,
)
    mcmctest(Random.default_rng(), test, subject; statistics, show_progress)
end
