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
 * Multi-layer perceptron code. Based on pseudo-code
 * found on page 98-100 of "Machine Learning" by Tom M. Mitchell.
 * Uses a sigmoid transfer for now.
 * This code was written for UW MSCC TCSS 570
 * Parallel Computing, Summer 2016, as test code for parallelization.
 * The basic functions AND, OR, and XOR can be tested
 * by uncommenting various sections and simple datasets.
 *  
 */

void feed_forward(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example);

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

