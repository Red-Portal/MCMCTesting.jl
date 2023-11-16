```@meta
CurrentModule = MCMCTesting
```

# MCMCTesting

Documentation for [MCMCTesting](https://github.com/Red-Portal/MCMCTesting.jl).

`MCMCTesting` provides the MCMC testing algorithms developed by Gandy & Scott (2021)[^gandyandscott2021].
These tests can be seen as an improvement of the hypothesis testing approach proposed by Geweke [^geweke2004].
Unlike simulation-based calibration (SBC; [^talts2018][^yaoanddomke2023][^modrak2022]), these tests are more appropriate for testing the exactness of the MCMC algorithm rather than the identifiability of the models.
This is because the tests focus on maximizing the power for verify the validity of *individual Markov transitions* instead of a *set of samples*.
Furthermore, unlike SBC, the approach of Gandy & Scott[^gandyandscott2021] is able to exactly satisfy the assumptions required for the theoretical guarantees.

## References
[^gandyandscott2021]: Gandy, A., & Scott, J. (2020). Unit testing for MCMC and other Monte Carlo methods. arXiv preprint arXiv:2001.06465.
[^geweke2004]: Geweke, J. (2004). Getting it right: Joint distribution tests of posterior simulators. Journal of the American Statistical Association, 99(467), 799-804.
[^talts2018]: Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv preprint arXiv:1804.06788.
[^yaoanddomke2023]: Yao, Y., & Domke, J. (2023, November). Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier. In Thirty-seventh Conference on Neural Information Processing Systems.
[^modrak2022]: Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N., Faltejsková, K., ... & Vehtari, A. (2022). Simulation-based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. arXiv preprint arXiv:2211.02383.

