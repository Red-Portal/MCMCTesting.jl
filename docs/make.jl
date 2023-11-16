using MCMCTesting
using Documenter

DocMeta.setdocmeta!(MCMCTesting, :DocTestSetup, :(using MCMCTesting); recursive=true)

makedocs(;
    modules=[MCMCTesting],
    authors="Kyurae Kim <kyrkim@seas.upenn.edu> and contributors",
    repo="https://github.com/Red-Portal/MCMCTesting.jl/blob/{commit}{path}#{line}",
    sitename="MCMCTesting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/MCMCTesting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "MCMCTesting"       => "introduction.md",
        "Getting Started"   => "example.md",
        "Two-Sample Tests"  => "twosampletest.md",
        "Exact Rank Tests"  => "exactranktest.md",
        "API"               => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/MCMCTesting.jl",
)
