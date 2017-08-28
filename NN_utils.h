#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/*
 * Multi-layer perceptron code. Based on pseudo-code
 * found on page 98-100 of "Machine Learning" by Tom M. Mitchell.
 * Uses a sigmoid transfer for now.
 * This code was written for UW MSCC TCSS 570
 * Parallel Computing, Summer 2016, as test code for parallelization.
 * The basic functions AND, OR, and XOR can be tested
 * by uncommenting various sections and simple datasets.
 * 
 * Copyright 2016-2017 David Kaplan 
 */

typedef struct NN_params NN_parameters;
struct NN_params {
    int num_layers;
    int num_inputs;
    int num_outputs;
    double Eta;  //the learning rate
    double alpha; //the momentum
    int *num_units_in_layer;
    double ***layer_weight_matrices;
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

void print_matrix(double **A, int m, int n);
void printNeuralNet(NN_parameters *np, double **NN);
void printWeights(NN_parameters *np);
void clearNeuralNet(NN_parameters *np, double **NN);
int readCSV(double **data, int r, int c, char *path);
int createDataSet(NN_data_set *nn);
NN_data_set createDataSetFromCSV(char * path, int num_inputs,
				int num_outputs, int num_examples);
NN_data_set createRandomDataSet(int num_examples, int num_inputs,
                                int num_outputs);
double **allocateMatrix(int rows, int cols);
double **allocateNN(NN_parameters *np);
int allocateWeights(NN_parameters *np);
double **createNeuralNet(NN_parameters *np);
int allocateDeltas(NN_parameters *np );
double **createNeuralNet(NN_parameters *np);
int saveWeights(NN_parameters *np, char *path);
int saveParams(NN_parameters *np, char *params_path, char *weights_path, int save_weights);
double ** loadNN(NN_parameters *np, 
            char *param_path, char *weight_path, int load_weights);
int ms_diff(struct timespec start, struct timespec stop);
void destroyDataSet(NN_data_set *nn);
