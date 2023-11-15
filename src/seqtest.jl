
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
