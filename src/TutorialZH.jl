module TutorialZH

using IJulia, Pkg

export tutorial, notebooks

function tutorial()
    cmd = IJulia.find_jupyter_subcommand("notebook")
    push!(cmd.exec, joinpath(@__DIR__, "..", "notebooks", "tutorial.ipynb"))
    return IJulia.launch(cmd, joinpath(@__DIR__, "..", "notebooks"), false)
end

function notebooks()
    return IJulia.notebook(dir=joinpath(@__DIR__, "..", "notebooks"))
end

REQUIRE = [
    "GR",
    "PyCall",
    "IJulia",
    "Revise",
    "Interact",
]

function __init__()
    for each in REQUIRE
        if each in keys(Pkg.installed())
            continue
        else
            Pkg.add(each)
        end
    end
end

end # module
