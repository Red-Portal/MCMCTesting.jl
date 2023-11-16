
# [Getting Started](@id tutorial)

## Problem Setup
Let's consider a simple Normal-Normal model with a shared scale:
```math
\begin{aligned}
\theta &\sim \mathrm{normal}\left(0,   \sigma^2\right) \\
y      &\sim \mathrm{normal}\left(\mu, \sigma^2\right)
\end{aligned}
```

The joint log-likelihood can be implemented as follows:
```@example started
using Random
using Distributions

struct Model
    σ::Float64
    y::Float64
end

function logdensity(model::Model, θ)
    σ, y = model.σ, model.y
    logpdf(Normal(0, σ), only(θ)) + logpdf(Normal(0, σ), y)
end
nothing
```
For sampling from the posterior, a simple Gibbs sampling strategy is possible:
```@example started
struct Gibbs end

function step(rng::Random.AbstractRNG, model::Model, ::Gibbs, θ)
    y = model.y
    σ = model.σ
    rand(rng, MvNormal([y]/2, σ))
end
nothing
```
We could also use the classic symmetric random walk Metropolis-Hastings sampler:
```@example started
struct RWMH
    σ::Float64
end

function step(rng::Random.AbstractRNG, model::Model, kernel::RWMH, θ)
    σ = kernel.σ
    θ′ = rand(rng, MvNormal(θ, σ))
    ℓπ = logdensity(model, θ)
    ℓπ′ = logdensity(model, θ′)
    ℓα = ℓπ′ - ℓπ
    if log(rand(rng)) ≤ ℓα
        θ′
    else
        θ
    end
end
nothing
```

## Testing the Gibbs Sampler
All of the functionalities of `MCMCTesting` assume that we can sample from the joint distribution $p(\theta, y)$.
This is done by as follows:
```@example started
using MCMCTesting

function MCMCTesting.sample_joint(rng::Random.AbstractRNG, model::Model)
    σ = model.σ
    θ = rand(rng, Normal(0, σ))
    y = rand(rng, Normal(θ, σ))
    [θ], [y]
end
nothing
```

The Gibbs sampler can be connected to `MCMCTesting` by implementing the following:

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
model = Model(1., 1.)
kernel = Gibbs()
test = TwoSampleTest(100, 100)
subject = TestSubject(model, kernel)
seqmcmctest(test, subject, 0.0001, 100; show_progress=false)
```
`true` means that the tests have passed.
Now, let's consider two erroneous implementations:

```@example started
struct GibbsWrongMean end

function step(rng::Random.AbstractRNG, model::Model, ::GibbsWrongMean, θ)
    y = model.y
    σ = model.σ
    rand(rng, MvNormal([y], σ/2))
end

struct GibbsWrongVar end

function step(rng::Random.AbstractRNG, model::Model, ::GibbsWrongVar, θ)
    y = model.y
    σ = model.σ
    rand(rng, MvNormal([y/2], 2*σ))
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

## Visualizing the ranks
