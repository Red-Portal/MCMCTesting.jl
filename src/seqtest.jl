
"""
    seqmcmctest([rng,] test, subject, false_rejection_rate, samplesize; kwargs...)

Sequential run multiple hypothesis tests to guarantee `false_rejection_rate`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `test::AbstractMCMCTest`: Test strategy.
- `subject::TestSubject`: MCMC algorithm and model subject to test.
- `false_rejection_rate::Real`: Desired false rejection rate.
- `samplesize::Int`: The number of p-values used at each test iteration.

# Keyword Arguments
- `samplesize_increase`: Factor of increase for the samplsize after the first test iteration turns out inconclusive. (Default: `2.0`)
- `show_progress::Bool`: Whether to show progress. (Default: `true`)
- `pvalue_adjustmeht::MultipleTesting.PValueAdjustment`: P-value adjustment for multiple testing over the elements of the statistic. (Default: `MultipleTesting.Bonferroni()`)
Additional keyword arguments are passed to internal calls to `mcmctest`.

# Returns
- `test_result::Bool`: `true` if the null-hypothesis (the MCMC algorithm has the correct stationary distribution) wasn't rejected, `false` otherwise.
"""
function seqmcmctest(
    rng                 ::Random.AbstractRNG,
    test                ::AbstractMCMCTest,
    subject             ::TestSubject,
    false_rejection_rate::Real,
    samplesize          ::Int,
    max_iter            ::Real = 3,
    samplesize_increase ::Real = 2.;
    show_progress = true,
    pvalue_adjustment::MultipleTesting.PValueAdjustment = MultipleTesting.Bonferroni(),
    kwargs...
)
    α  = false_rejection_rate
    k  = max_iter
    β  = α / k
    γ  = β^(1/k)
    Δ  = samplesize_increase

    for i = 1:k
        prog   = ProgressMeter.Progress(
            samplesize;
            barlen    = 31,
            showspeed = true,
            enabled   = show_progress
        )
        pvals_all = mapreduce(hcat, 1:samplesize) do n
            pval = mcmctest(rng, test, subject; show_progress=false, kwargs...)
            next!(prog,
                  showvalues = [
                      (:test_iteration, i),
                      (:pvalue_sampling, "$(n)/$(samplesize)")
                  ])
            pval
        end

        pvals_adjusted = mapreduce(vcat, eachcol(pvals_all)) do pvals_paramwise
            adjust(Vector(pvals_paramwise), pvalue_adjustment)
        end
        
        q = minimum(pvals_adjusted)*length(pvals_adjusted)

        if q ≤ β
            return false
        elseif q > γ + β
            break
        end

        β /= γ

        if i == 1
            samplesize = ceil(Int, samplesize*Δ)
        end
    end
    true
end

function seqmcmctest(
    test                ::AbstractMCMCTest,
    subject             ::TestSubject,
    false_rejection_rate::Real,
    samplesize          ::Int,
    max_iter            ::Real = 3,
    samplesize_increase ::Real = 2.;
    show_progress = true,
    pvalue_adjustment::MultipleTesting.PValueAdjustment = MultipleTesting.Bonferroni(),
    kwargs...
)
    seqmcmctest(Random.default_rng(), test, subject, false_rejection_rate,
                samplesize, max_iter, samplesize_increase;
                show_progress, pvalue_adjustment, kwargs...)
end
