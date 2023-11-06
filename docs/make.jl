using MCMCTesting
using Documenter

DocMeta.setdocmeta!(MCMCTesting, :DocTestSetup, :(using MCMCTesting); recursive=true)

makedocs(;
    modules=[MCMCTesting],
    authors="Ray Kim <msca8h@naver.com> and contributors",
    repo="https://github.com/Red-Portal/MCMCTesting.jl/blob/{commit}{path}#{line}",
    sitename="MCMCTesting.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Red-Portal.github.io/MCMCTesting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Red-Portal/MCMCTesting.jl",
    devbranch="main",
)
