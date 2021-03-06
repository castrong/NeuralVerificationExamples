# NeuralVerificationExamples


This repository contains examples for how to use [NeuralVerification.jl](https://github.com/sisl/NeuralVerification.jl), [Marabou](https://github.com/NeuralNetworkVerification/Marabou), and [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) through a [wrapper](https://github.com/castrong/MIPVerifyWrapper) that allows you to perform optimization. It was written for a reading group on April 9th, 2021. The presentation can be found [here](https://docs.google.com/presentation/d/1gzerfBzOTBf8j8gN7-ojrCu0UYf_FcCYqz6TDfbn9Tw/edit?usp=sharing)

In each of the three tools demonstrated here, we try to verify a property of a network designed to map from images of a runway to control effort for a taxiing airplane meant to steer it to stay along the centerline. the following property is verified: will the control effort produced by the autotaxi network always be below 2.0 when the image has each pixel between 0.4 and 0.6 (a bit unrealistic of an input set, typically you could make something like a small hypercube around a known image to test robustness to perturbations). Some networks in [.nnet](https://github.com/sisl/NNet) file format can be found in the `Networks` folder of the repository.

The `nvjl_example.jl` file demonstrates how to use NeuralVerification.jl to accomplish this. It is written in Julia and requires the packages NeuralVerification and LazySets.

The `marabou_shell_example.sh` demonstrates how to use Marabou to accomplish this. A working build of Marabou must be installed, and then you must update the paths in the shell script to correspond to (1) the path to the marabou build, (2) your path to the network, and (3) your path to the property file before running. 

The `mipverify_example.jl` script demonstrates how to use the MIPVerify wrapper to solve this problem by performing an maximization on the control effort. Unlike the other two solvers it provides an actual maximum control effort instead of just telling us a yes or no answer of whether it can be above 2.0. You could additionally impose the output constraint (control effort <= 2.0) which may or may not speed up computation. 

