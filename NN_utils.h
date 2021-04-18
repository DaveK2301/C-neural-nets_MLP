#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
 * Copyright 2016-2021 David Kaplan 
 */

typedef struct NN_params NN_parameters;
struct NN_params {
    int num_layers;
    int num_inputs;
    int num_outputs;
    double Eta;  //the learning rate
    double alpha; //the momentum
    double ***layer_weight_matrices;
    int* num_units_in_layer;
    double ***layer_weight_deltas;
};

/*
 * This struct is intended to store a 2d matrix
 * of row vectors, 1 vector per training
 * or test example.
 * The first num_inputs entries in the row
 * are the inputs for that example,
 * the next num_outputs entries for that row
 * are the outputs for that example
 */
typedef struct NN_dataset NN_data_set;
struct NN_dataset {
    int num_inputs;
    int num_outputs;
    int num_examples;     
    double **data;
};

/*****************************************************************************/
// Neural Net utilities

 /*
  * Creates the neural network and the weights array.
  *
  * The NN is ragged 2d array representing the
  * neural network, as one continuously
  * allocated chunk of memory, with pointers
  * to the starts of the rows.
  * The weights array is described in the
  * allocateWeights() function.
  * Caller must check for NULL return value.
  */
double** createNeuralNet(NN_parameters* np);

 /*
  * Creates the jagged array, one row
  * per layer of the neural network.
  * Caller must check for NULL return.
  */
double **allocateNN(NN_parameters *np);
/*
 * Allocates the various rectangular
 * matrices (or vectors for single neuron layers)
 * representing the weigths to layer L of the neural net
 *
 * Returns 0 if allocation is succesful, 1 otherwise.
 */
int allocateWeights(NN_parameters *np);
/*
 * This creates the parallel matrices to the
 * neural net matrices
 * to hold the delta values when using
 * momentum.
 */
int allocateDeltas(NN_parameters *np );
void clearNeuralNet(NN_parameters* np, double** NN);

/*****************************************************************************/
// Neural Net storage

/*
 * Loading and saving NN's, weights and parameters
 * very lightly tested!
 */
int saveWeights(NN_parameters* np, char* path);
int saveParams(NN_parameters* np, char* params_path, char* weights_path, int save_weights);

/*
 * Caller check for NULL return values
 * does not create the error matrix
 * or deltas.
 */
double** loadNN(NN_parameters* np,
    char* param_path, char* weight_path, int load_weights);

// IO utilities
void print_matrix(double** A, int m, int n);
void printNeuralNet(NN_parameters* np, double** NN);
void printWeights(NN_parameters* np);
int readCSV(double** data, int r, int c, char* path);

/*****************************************************************************/
// data set utilites

/*
 * Created purely random inputs and outputs.
 * Purely for testing parallelization.
 */
NN_data_set createRandomDataSet(int num_examples, int num_inputs,
    int num_outputs);
/*
 * Creates an empty data set of the format described in
 * the NN_data_set struct.
 * Returns 0 if succesfull, 1 otherwise.
 */
int createDataSet(NN_data_set* nn);
/*
 * This will not work if the file contains a
 * header row (TODO).
 * Strictly works with commas only for now (TODO)!
 */
NN_data_set createDataSetFromCSV(char* path, int num_inputs,
    int num_outputs, int num_examples);

/*****************************************************************************/
// compact matrix utilities

/*
 * Caller must check for NULL
 */
double** allocateMatrix(int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif
