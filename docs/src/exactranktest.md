
# [Exact Rank Hypothesis Tests](@id exactrank)

## Introduction

The exact rank hypothesis testing strategy is based on Algorithm 2 by Gandy & Scott (2021)[^gandyandscott2021].

## `ExactRankTest`
The ranks are computed by simulating a single Markov chain backwards and forward.
First, a random midpoint is simulated as
```math
\begin{aligned}
   M \sim \mathrm{Uniform}(1,\;L),
\end{aligned}
```
where $L = \texttt{n\_mcmc\_steps}$.
Then, we simulate the Markov chain forward and backward as
```math
\begin{alignat*}{3}
  \theta_{M-l} &\sim K\left(\theta_{L-l+1}, \cdot \right) \qquad &&\text{for}\; l = 1, \ldots, M-1 \\
  \theta_{l}   &\sim K\left(\theta_{l-1}, \cdot \right) \qquad &&\text{for}\; l = M+1, \ldots, L
\end{alignat*}
```
forming the chain
```math
\theta_1,\; \ldots, \; \theta_{M},\; \ldots, \; \theta_{L}.
```
The *rank* is the ranking of the statistics of $\theta_{M}$.
If the sampler and the model are correct, the rank has an uniform distribution as long as the midpoint is independently sampled.

```@docs
ExactRankTest
```

# References
[^gandyandscott2021]: Gandy, A., & Scott, J. (2020). Unit testing for MCMC and other Monte Carlo methods. arXiv preprint arXiv:2001.06465.
