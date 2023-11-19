
module MCMCTestingPlotsExt

if isdefined(Base, :get_extension)
    using Plots
    using MCMCTesting
else
    using ..Plots
    using ..MCMCTesting
end

"""
    plotranks(ranks, test; kwargs...)

`Plots` recipe for plotting the simulated ranks from `simulate_ranks`.
`Plots` must be imported *before* `MCMCTesting` to use this plot recipe.

# Arguments
- ranks: The output of `simulate_rank`.
- test::ExactRankTest: The exact rank test object used to simulate the ranks.

# Keyword Arguments
- stats_names: The name for the statistics used in the rank simulation. The default argument automatically assign default names. (Default: :auto). 
"""
@recipe function plotranks(ranks, test::ExactRankTest; stat_names=:auto)
    n_max_rank = test.n_mcmc_steps
    n_samples  = test.n_samples
    binprob    = 1/n_max_rank
    binstd     = sqrt((1 - binprob)*(binprob)*n_samples)
    xguide     --> "Rank"
    yguide     --> "Count"
    xlims      --> [1,n_max_rank]
    bins       --> 1:1:n_max_rank
    fillalpha  --> 0.2

    # default two-column layout
    n_params   = size(ranks,1)
    n_rows     = ceil(Int,n_params/2)
    size       --> (300*2, 200*n_rows)
    layout     --> (n_rows,2)

    for (idx, ranks_param) in enumerate(eachrow(ranks))
        stat_name = if stat_names isa Symbol && stat_names == :auto
            "θ$(idx)"
        elseif stat_names isa AbstractVector && length(stat_names) == size(ranks,1)
            string.(stat_names[idx])
        else
            error("A custom list of parameter names must be an <: AbstractVector " *
                  "and have the same length as size(rank,1), the number of statistics used in the rank simulation.")
        end
        @series begin
            label       := stat_name
            subplot     := idx
            fill        := true
            fillcolor   := :match
            linecolor   := :match
            linealpha   := 1.0
            seriestype  := :stephist
            ranks_param
        end
        # 1σ confidence interval
        @series begin
            label      := nothing
            subplot    := idx
            seriestype := :hline
            linealpha  := 1.0
            fillalpha  := 0.1
            color      := :black
            ribbon --> [binstd,binstd]
            [n_samples/n_max_rank]
        end
        # 2σ confidence interval
        @series begin
            label      := nothing
            subplot    := idx
            seriestype := :hline
            linealpha  := 0.0
            fillalpha  := 0.05
            color      := :black
            ribbon --> [2*binstd,2*binstd]
            [n_samples/n_max_rank]
        end
        # 2σ confidence interval
        @series begin
            label      := nothing
            subplot    := idx
            seriestype := :hline
            linealpha  := 0.0
            fillalpha  := 0.025
            color      := :black
            ribbon --> [3*binstd,3*binstd]
            [n_samples/n_max_rank]
        end
    end
end
end
