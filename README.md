# C-neural-nets_MLP
Multi-layer perceptron with OpenMP CPU multi-threading

A small fast lightweight neural network package in C. Currently uses only the Sigmoid transfer function. Based on the pseudo-code found on page 98 of "Machine Learning" by Tom M. Mitchell. Created for TCSS 570 Intro to Parallel Computing in the summer of 2016.
The demo executable shows the creation of the neural net and dataset structures as well as the feedforward, backpropagation, and weights adjustment stages.
The OpenMP functions were used to test parallelization performance gains.

This is purely intended as a code sample for prospective employers or partners at this time.

Neural Nets running in Unreal Engine learning the XOR function using same tiny XOR demo data set and learning parameters as found in nn_demo_1. Epochs look high because there are only 4 samples in the data set, but the RMSE does get extremely small:
![C_NeuralNetsInUnrealActorInWorld](https://user-images.githubusercontent.com/16049374/115104011-b62fb000-9f0a-11eb-9d9d-dd60fb3bb850.png)
