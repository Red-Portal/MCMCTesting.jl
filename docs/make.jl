using MCMCTesting
using Documenter
using Plots

DocMeta.setdocmeta!(MCMCTesting, :DocTestSetup, :(using MCMCTesting); recursive=true)

makedocs(;
    modules=[MCMCTesting,],
    repo="https://github.com/Red-Portal/MCMCTesting.jl/blob/{commit}{path}#{line}",
    sitename="MCMCTesting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/MCMCTesting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home"              => "index.md",
        "Tutorial"          => "example.md",
        "General Usage"     => "general.md",
        "Two-Sample Tests"  => "twosampletest.md",
        "Exact Rank Tests"  => "exactranktest.md",
        "Visualizing Ranks" => "ranksim.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/MCMCTesting.jl",
    push_preview=true
)
