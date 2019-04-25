module TutorialZH

using IJulia, Pkg

export tutorial

function tutorial()
    cmd = IJulia.find_jupyter_subcommand("notebook")
    push!(cmd.exec, joinpath(@__DIR__, "..", "notebooks", "tutorial.ipynb"))
    return IJulia.launch(cmd, joinpath(@__DIR__, "..", "notebooks"), false)
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
