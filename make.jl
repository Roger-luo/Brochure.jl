using Documenter

make() = makedocs(
    format=Documenter.HTML(prettyurls=false, assets = ["assets/chainrules.css"]),
    sitename="Julia编程指南",
    authors="罗秀哲 Xiuzhe (Roger) Luo",
    pages=[
        "简介" => "index.md",
    ],
)

make()

deploydocs(
    repo = "github.com/Roger-luo/JuliaGuide.jl.git",
    target = "build",
)
