module PyVnnlib

using PyCall, JLD2

const vnnlib = PyNULL()
const compat = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")["path"]), @__DIR__)
    copy!(vnnlib, pyimport("vnnlib"))
    # why can't i just use vnnlib.compat, but have to specifically pyimport that submodule?
    copy!(compat, pyimport("vnnlib.compat"))
end


"""
Generate matrix of properties given input file.

The output is a matrix of shape n × 2 with one row for each of the n properties.
Each row is a vector [bounds, output_specs]
"""
function parse_properties(infile; verbosity=0)
    verbosity == 1 && println("parsing file $infile") 
    t_parse = @elapsed parsed = vnnlib.parse_file(infile, strict=false)

    verbosity == 1 && println("extracting properties")
    t_transform = @elapsed properties = compat.CompatTransformer("X","Y").transform(parsed)

    verbosity == 1 && println("parsing: $t_parse, transforming: $t_transform")
    return properties
end


"""
Generate list of specifications [(l, u, [(A, b)])] of
- input bounds l, u for each input 
- polytopes Ax ≤ b over each common input space (s.t. property is SAT, if one of the polytopes is SAT)
"""
function get_speclist(props; dtype=nothing)
    specs = []
    for i in axes(props,1)
        bounds = props[i,1]
        lbs = bounds[:,1]
        ubs = bounds[:,2]

        output_specs = props[i,2]

        # list of (A, b) s.t. Ax ≤ b
        out_specs = []
        for (A, b) in output_specs
            A = isnothing(dtype) ? A : dtype.(A)
            b = isnothing(dtype) ? b : dtype.(b)

            if ndims(A) != 2
                println("Warning! Output constraints have to be polytopes, found A with size $(size(A)) - try flattening")
                A = reshape(A, :, size(A)[end])
            end

            if ndims(b) != 1
                println("Warning! Output constraints have to be polytopes, found b with size $(size(b)) - trying vec(b)")
                b = vec(b)
            end

            push!(out_specs, (A, b))
        end
        
        push!(specs, (lbs, ubs, out_specs))
    end

    return specs
end


"""
Generate list of specifications [(l, u, [(A, b)])] from a vnnlib file of
- input bounds l, u for each input 
- polytopes A₁x ≤ b₁ ∨ A₂x ≤ b₂ ∨ ... over each common input space
"""
function generate_specs(vnnlib_file; verbosity=0, dtype=nothing)
    props = parse_properties(vnnlib_file, verbosity=verbosity)
    speclist = get_speclist(props, dtype=dtype)

    return speclist
end


"""
Generates a list of specifications and saves it as outfile.

The list of specifications has format [(l, u, [(A, b)])] from a vnnlib file of
- input bounds l, u for each input 
- polytopes A₁x ≤ b₁ ∨ A₂x ≤ b₂ ∨ ... over each common input space

The list of specifications can be loaded from the outfile by calling

speclist = load(outfile, "speclist")

args:
    vnnlib_file - either a .vnnlib or a .vnnlib.gz file
    outfile - name of output file has to end in .jld2
"""
function save_specs(vnnlib_file, outfile; verbosity=0, dtype=nothing)
    speclist = generate_specs(vnnlib_file, verbosity=verbosity, dtype=dtype)

    save(outfile, "speclist", speclist)
end


export generate_specs, save_specs



end # module PyVnnlib
