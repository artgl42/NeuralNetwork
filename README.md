# NeuralNetwork

This library, that implements computation neural networks (multilayer perceptron) with customizable layers, built from scratch with C# and is under development. The library doesn't use GPU for computing (only CPU). The library exposes easy to use classes and methods (APIs) to create a new neural network and train it. The following settings are currently available (via the APIs):

* Activation functions: Sigmoid, TanH, Identity, ReLU, Gaussian
* Algorithm for initialization neurons weights: Zeros, Ones, Random, XavierUniform
* Algorithm for initialization bias neurons: Zeros, Ones, Random
* Algorithm for optimization: NAG
