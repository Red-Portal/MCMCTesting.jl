
# [Two Sample Hypothesis Tests](@id twosample)

## Introduction

The two-sample hypothesis testing strategies are based on Algorithm 1 by Gandy & Scott (2021)[^gandyandscott2021].

## `TwoSampleTest`
The first basic strategy generates the **treatment group**
```math
\begin{aligned}
  \theta,\; y_{\text{trtm}} &\sim p\left(\theta, y\right) \\
  \theta_{\text{trtm}} &\sim K\left(\theta, \cdot\right),
\end{aligned}
```
and the **control group** as
```math
\begin{aligned}
  \theta_{\text{ctrl}},\; y_{\text{ctrl}} &\sim p\left(\theta, y\right), \\
\end{aligned}
```
where the test compares 
```math
(\theta_{\text{trtm}}, \, y_{\text{trtm}})
\quad\text{versus}\quad 
(\theta_{\text{ctrl}}, \, y_{\text{ctrl}}).
```

```@docs
TwoSampleTest
```

## `TwoSampleGibbsTest`
The second strategy performs applies an additional Gibbs sampling step when generating the **treatment group**  as
```math
\begin{aligned}
  \theta,\; y &\sim p\left(\theta, y\right) \\
  \theta_{\text{trtm}} &\sim K\left(\theta, \cdot\right), \\
  y_{\text{trtm}} &\sim p\left(y \mid \theta_{\text{trtm}}\right)
\end{aligned}
```
resulting in the treatment group 
```math
(\theta_{\text{trtm}}, \, y_{\text{trtm}}).
```
The control group is generated the same as `TwoSampleTest`.

```@docs
TwoSampleGibbsTest
```

# References
[^gandyandscott2021]: Gandy, A., & Scott, J. (2020). Unit testing for MCMC and other Monte Carlo methods. arXiv preprint arXiv:2001.06465.
