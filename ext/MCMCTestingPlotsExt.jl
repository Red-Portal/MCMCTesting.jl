
module MCMCTestingPlotsExt

if isdefined(Base, :get_extension)
    using Plots
    using MCMCTesting
else
    using ..Plots
    using ..MCMCTesting
end

@userplot RankPlot
@recipe function plot(ranks, test::ExactRankTest)
    n_max_rank = test.n_mcmc_steps
    n_samples  = test.n_samples
    binprob    = 1/n_max_rank
    binstd     = sqrt((1 - binprob)*(binprob)*n_samples)
    xguide --> "Rank"
    yguide --> "Count"
    xlims  --> [1,n_max_rank]
    bins   --> 1:1:n_max_rank
    @series begin
        seriestype := :histogram
        ranks'
    end
    @series begin
        seriestype := :hline
        ribbon --> [2*binstd,2*binstd]
        [n_samples/n_max_rank]
    end
end
end
