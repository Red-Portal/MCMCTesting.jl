
# [Tutorial](@id tutorial)

## Problem Setup
Let's consider a simple Normal-Normal model:
```math
\begin{aligned}
\theta_1 &\sim \mathrm{normal}\left(0,   \sigma^2\right) \\
\theta_2 &\sim \mathrm{normal}\left(0,   \sigma^2\right) \\
y        &\sim \mathrm{normal}\left(\theta_1 + \theta_2, \sigma_{\epsilon}^2\right).
\end{aligned}
```

The joint log-likelihood can be implemented as follows:
```@example started
using Random
using Distributions

struct Model
    sigma    ::Float64
    sigma_eps::Float64
    y        ::Float64
end
nothing
```
For sampling from the posterior, a simple Gibbs sampling strategy is possible:
```@example started
struct Gibbs end

function complete_conditional(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y - θ)
    σ = 1/sqrt(1/σ²_ϵ + 1/σ²)
    Normal(μ, σ)
end

function step(rng::Random.AbstractRNG, model::Model, ::Gibbs, θ)
    θ    = copy(θ)
    y    = model.y
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2
    θ[1] = rand(rng, complete_conditional(θ[2], σ², σ²_ϵ, y))
    θ[2] = rand(rng, complete_conditional(θ[1], σ², σ²_ϵ, y))
    θ
end
nothing
```

## Testing the Gibbs Sampler
All of the functionalities of `MCMCTesting` assume that we can sample from the joint distribution $p(\theta, y)$.
This is done by as follows:
```@example started
using MCMCTesting

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::Model)
    θ₁ = rand(rng, Normal(0, model.sigma))
    θ₂ = rand(rng, Normal(0, model.sigma))
    θ  = [θ₁, θ₂]
    y  = rand(rng, Normal(θ[1] + θ[2], model.sigma_eps))
    θ, y
end
nothing
```

The Gibbs samplers can be connected to `MCMCTesting` by implementing the following:
```@example started
using Accessors

function MCMCTesting.markovchain_transition(
    rng::Random.AbstractRNG, model::Model, kernel, θ, y
)
    model′ = @set model.y = only(y)
    step(rng, model′, kernel, θ)
end
nothing
```

Let's check that the implementation is correct by 
```@example started
model = Model(1., .5, randn())
kernel = Gibbs()
test = TwoSampleTest(100, 100)
subject = TestSubject(model, kernel)
seqmcmctest(test, subject, 0.0001, 100; show_progress=false)
```
`true` means that the tests have passed.

Now, let's consider two erroneous implementations:
```@example started
struct GibbsWrongMean end

function complete_conditional_wrongmean(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y + θ)
    σ = 1/sqrt(1/σ²_ϵ + 1/σ²)
    Normal(μ, σ)
end

function step(rng::Random.AbstractRNG, model::Model, ::GibbsWrongMean, θ)
    θ    = copy(θ)
    y    = model.y
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2

    θ[1] = rand(rng, complete_conditional_wrongmean(θ[2], σ², σ²_ϵ, y))
    θ[2] = rand(rng, complete_conditional_wrongmean(θ[1], σ², σ²_ϵ, y))
    θ
end

struct GibbsWrongVar end

function complete_conditional_wrongvar(θ::Real, σ²::Real, σ²_ϵ::Real, y::Real)
    μ = σ²/(σ²_ϵ + σ²)*(y - θ)
    Normal(μ, sqrt(σ²))
end

function step(rng::Random.AbstractRNG, model::Model, ::GibbsWrongVar, θ)
    θ    = copy(θ)
    y    = model.y
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2

    if rand(Bernoulli(0.5))
        θ[1] = rand(rng, complete_conditional_wrongvar(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional_wrongvar(θ[1], σ², σ²_ϵ, y))
    else
        θ[2] = rand(rng, complete_conditional_wrongvar(θ[1], σ², σ²_ϵ, y))
        θ[1] = rand(rng, complete_conditional_wrongvar(θ[2], σ², σ²_ϵ, y))
    end
    θ
end
nothing
```
The kernel with a wrong mean fails:
```@example started
kernel = GibbsWrongMean()
subject = TestSubject(model, kernel)
seqmcmctest(test, subject, 0.0001, 100; show_progress=false)
```
and so does the one with the wrong variance:
```@example started
kernel = GibbsWrongVar()
subject = TestSubject(model, kernel)
seqmcmctest(test, subject, 0.0001, 100; show_progress=false)
```

## Visualizing Simulated Ranks
`MCMCTesting` also provides some basic plot recipes for visualizing the simulated rank for the [exact rank test](@ref exactrank).
For this, `Plots` must be imported *before* `MCMCTesting`.

```@example started
using Plots
gr()
using MCMCTesting
nothing
```

Also, since the exact rank test explicitly requires the MCMC kernel to be reversible, we modify the previous Gibbs sampler to use a random scan order.
```@example started
function step(rng::Random.AbstractRNG, model::Model, ::Gibbs, θ)
    θ    = copy(θ)
    y    = model.y
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

function step(rng::Random.AbstractRNG, model::Model, ::GibbsWrongVar, θ)
    θ    = copy(θ)
    y    = model.y
    σ²   = model.sigma^2
    σ²_ϵ = model.sigma_eps^2
    if rand(Bernoulli(0.5))
        θ[1] = rand(rng, complete_conditional_wrongvar(θ[2], σ², σ²_ϵ, y))
        θ[2] = rand(rng, complete_conditional_wrongvar(θ[1], σ², σ²_ϵ, y))
    else
        θ[2] = rand(rng, complete_conditional_wrongvar(θ[1], σ², σ²_ϵ, y))
        θ[1] = rand(rng, complete_conditional_wrongvar(θ[2], σ², σ²_ϵ, y))
    end
    θ
end
nothing
```
Then, we can simulate the ranks and then plot them using `Plots.
```@example started
test = ExactRankTest(1000, 30, 10)

rank_correct = simulate_ranks(test, TestSubject(model, Gibbs()); show_progress=false)
rank_wrong = simulate_ranks(test, TestSubject(model, GibbsWrongVar()); show_progress=false)

param_names = ["θ1 mean", "θ2 mean", "θ1 var", "θ2 var"]
plot(rank_wrong,    test; param_names)
plot!(rank_correct, test; param_names)
savefig("rankplot.svg")
nothing
```
![](rankplot.svg)

We can see that the ranks of the erroneous kernel are not uniform.
