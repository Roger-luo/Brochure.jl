using Documenter, LiveServer
using LiveServer: SimpleWatcher, WatchedFile, set_callback!, file_changed_callback

const deploy = "deploy" in ARGS

make() = makedocs(
    format=Documenter.HTML(
        prettyurls = deploy,
        canonical = ("deploy" in ARGS) ? "https://rogerluo.me/Brochure.jl/dev/" : nothing,
    ),
    sitename="Julia编程指南",
    authors="罗秀哲 Xiuzhe (Roger) Luo",
    pages=[
        "简介" => "index.md",
        "在开始之前" => "before-we-start.md",
        "程序的编写流程" => "workflow.md",
        "快速入门" => "quick-start.md",
        "实现你自己的稀疏矩阵" => "define-your-own-matrix.md",
        "实现你自己的自动微分" => "automatic-differentiation.md",
    ],
)

function scan_files!(dw::SimpleWatcher)
    for (root, _, files) in walkdir("src"), file in files
        push!(dw.watchedfiles, WatchedFile(joinpath(root, file)))
    end
end

function update_book_callback(fp::AbstractString)
    if splitext(fp)[2] == ".md"
        make()
    end
    file_changed_callback(fp)
    return nothing
end

function serve_book(verbose=false)
    watcher = SimpleWatcher()
    scan_files!(watcher)
    set_callback!(watcher, update_book_callback)
    make()
    serve(watcher, dir="build", verbose=verbose)
    return nothing
end

if ("s" in ARGS) || ("serve" in ARGS)
    serve_book("verbose" in ARGS)
end

if deploy
    make()
    deploydocs(
        repo = "github.com/Roger-luo/Brochure.jl.git",
        target = "build",
    )
end
