
"""
  Example in Section 4 of Gandy and Scott 2021.
"""

using MCMCTesting
using Random
using Statistics
using Distributions

struct Model
    sigma    ::Float64
    sigma_eps::Float64
end

struct GibbsRandScan end

struct GibbsRandScanWrongMean end

struct GibbsRandScanWrongVar end

function MCMCTesting.sample_predictive(rng::Random.AbstractRNG, model::Model, θ)
    # y ∼ θ₁ + θ₂ + ϵ
    rand(rng, Normal(θ[1] + θ[2], model.sigma_eps))
end

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::Model)
    θ₁ = rand(rng, Normal(0, model.sigma))
    θ₂ = rand(rng, Normal(0, model.sigma))
    θ  = [θ₁, θ₂]
    y  = MCMCTesting.sample_predictive(rng, model, θ)
    θ, y
end

function complete_conditional(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y - θ)
    σ = 1/sqrt(1/σ²_ϵ + 1/σ²)
    Normal(μ, σ)
end

function complete_conditional_wrongmean(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y + θ)
    σ = 1/sqrt(1/σ²_ϵ + 1/σ²)
    Normal(μ, σ)
end

function complete_conditional_wrongvar(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y - θ)
    Normal(μ, sqrt(σ²))
end

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::Model, kernel::GibbsRandScan, θ, y
)
    θ    = copy(θ)
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2

    if rand(Bernoulli(0.5))
        θ[1] = rand(rng, complete_conditional(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional(θ[1], σ², σ²_ϵ, y))
    else
        θ[2] = rand(rng, complete_conditional(θ[1], σ², σ²_ϵ, y))
        θ[1] = rand(rng, complete_conditional(θ[2], σ², σ²_ϵ, y))
    end
    θ
end

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::Model, kernel::GibbsRandScanWrongMean, θ, y
)
    θ    = copy(θ)
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2

    if rand(Bernoulli(0.5))
        θ[1] = rand(rng, complete_conditional_wrongmean(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional(          θ[1], σ², σ²_ϵ, y))
    else
        θ[2] = rand(rng, complete_conditional_wrongmean(θ[1], σ², σ²_ϵ, y))
        θ[1] = rand(rng, complete_conditional(          θ[2], σ², σ²_ϵ, y))
    end
    θ
end

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::Model, kernel::GibbsRandScanWrongVar, θ, y
)
    θ    = copy(θ)
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2

    if rand(Bernoulli(0.5))
        θ[1] = rand(rng, complete_conditional_wrongvar(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional(         θ[1], σ², σ²_ϵ, y))
    else
        θ[2] = rand(rng, complete_conditional_wrongvar(θ[1], σ², σ²_ϵ, y))
        θ[1] = rand(rng, complete_conditional(         θ[2], σ², σ²_ϵ, y))
    end
    θ
end

function main()
    model = Model(sqrt(10), sqrt(0.1))

    for test in [
        TwoSampleTest(     100, 100),
        TwoSampleGibbsTest(100, 100),
        ExactRankTest(     100, 100),
        ]
        @info(test)
        mcmctest(test, TestSubject(model, GibbsRandScan())) |> display
        mcmctest(test, TestSubject(model, GibbsRandScanWrongMean())) |> display
        mcmctest(test, TestSubject(model, GibbsRandScanWrongVar())) |> display
        
        seqmcmctest(test, TestSubject(model, GibbsRandScan()),          0.001, 32) |> display
        seqmcmctest(test, TestSubject(model, GibbsRandScanWrongMean()), 0.001, 32) |> display
        seqmcmctest(test, TestSubject(model, GibbsRandScanWrongVar()),  0.001, 32) |> display
    end
end
