using Pkg

# Set your wrapper path and your network file
wrapper_path = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper"
network_file = string(@__DIR__, "/Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet")

# Activate your MIPVerifyWrapper directory and include MIPVerify and the util files
Pkg.activate(wrapper_path)
using Parameters, LazySets, Interpolations, NPZ, JuMP, ConditionalJuMP, LinearAlgebra, MathProgBase
using CPUTime, Memento, AutoHashEquals, DocStringExtensions, ProgressMeter, MAT, GLPKMathProgInterface
include(wrapper_path*"/MIPVerify.jl/src/MIPVerify.jl")
include(wrapper_path*"/activation.jl")
include(wrapper_path*"/network.jl")
include(wrapper_path*"/problem.jl")
include(wrapper_path*"/util.jl")

# You can use your backend optimizer of choice. Gurobi has a free student license and is quite fast typically
using Gurobi

# Read in your network and convert to MIPVerify's format
network = read_nnet(network_file)
mipverify_network = network_to_mipverify_network(network, "NA", MIPVerify.mip)

# Create the backend optimizers and set your hyperparameters
threads = 8
timeout_per_node = 1.0
main_solve_timeout = 60.0
tightening_solver = GurobiSolver(OutputFlag = 0, TimeLimit=timeout_per_node, Threads=threads)
main_solver = GurobiSolver(Threads=threads, TimeLimit=main_solve_timeout)

# Preprocessing
problem = get_optimization_problem(
        (128,),
        mipverify_network,
        main_solver,
        lower_bounds=0.4*ones(128),
        upper_bounds=0.6*ones(128),
        tightening_solver=tightening_solver
        )

# Add anobjective and/or constraints then solve the problem
@objective(problem.model, Max, 1.0 * problem.output_variable[1])
solve(problem.model)
println("Maximum control: ", getobjectivevalue(problem.model))