// Copyright 2016 - 2021 David Kaplan

#ifndef NN_FUNCTIONS_H
#define NN_FUNCTIONS_H

#include <math.h>
#include <omp.h>
#include "NN_utils.h"

#ifdef __cplusplus
extern “C”{
#endif

 /*
  * Starting at the first layer past the input nodes,
  * applies the weights to the nodes in the layer above
  * and sets the output value of each node, layer by layer,
  * until the output layer has its value updated.
  * This version uses the sigmoid (logistic) function
  * for thresholding the outputs.
  */
void feed_forward(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example);

/*
 * Back propagate the errors from the outputs to all perceptrons
 * in the network.
 * Returns the sum of the squared errors for all the outputs,
 * on this training example
 * Expects an error matrix of the exact same format
 * as the neural network.
 */
double backPropagateError(NN_parameters *np, double **NN, 
                        double **errorMatrix, int example_num,
                        NN_data_set *dataset);

void updateWeights(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example);

void updateWeights_momentum(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example);

void feed_forwardOMP1(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example, int num_threads);

void feed_forwardOMP2(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example, int num_threads);

void updateWeights_momentumOMP1(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads);

void updateWeights_momentumOMP2(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads);

#ifdef __cplusplus
}
#endif

#endif

