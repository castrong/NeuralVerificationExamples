using NeuralVerification
using LazySets

network_file = string(@__DIR__, "/Networks/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet")
network = read_nnet(network_file)
input_set = Hyperrectangle(0.5 * ones(128), 0.1*ones(128)) # center, radius
output_set = HalfSpace([1.0], 2.0)
problem = Problem(network, input_set, output_set)

result = solve(Ai2z(), problem)
println("Result: ", result.status)