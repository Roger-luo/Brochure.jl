module TutorialZH

using IJulia

start() = notebook(dir=joinpath(@__DIR__, "..", "notebooks"))

end # module
