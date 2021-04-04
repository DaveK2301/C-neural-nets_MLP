#include "NN_utils.h"

/*
 * Multi-layer perceptron code. Based on pseudo-code
 * found on page 98-100 of "Machine Learning" by Tom M. Mitchell.
 * Uses a sigmoid transfer for now.
 * These are the functions that were tested
 * for multi-CPU parallelization.
 * This code was written for UW MSCC TCSS 570
 * Parallel Computing, Summer 2016, as test code for parallelization.
 * The basic functions AND, OR, and XOR can be tested
 * by uncommenting various sections and simple datasets.
 * 
 * Copyright 2016-2021 David Kaplan 
 */ 

void feed_forwardOMP1(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example, int num_threads) {
    //can use openMP to parallelize this loop
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(static)
    for (int nL1 = 0; nL1 < np->num_units_in_layer[0]; nL1++) {           
            neural_net[0][nL1] = np->layer_weight_matrices[0][nL1][0];
            
            for (int k = 0; k < np->num_inputs; k++) {
                neural_net[0][nL1] += np->layer_weight_matrices[0][nL1][k + 1] *
                                        dataset->data[example][k]; 
            }
            //apply the thresholding function here
            neural_net[0][nL1] = 1 / (1 + exp(-1.0 * neural_net[0][nL1]));    
    }
    //Start at the first hidden layer (or output layer
    //if no hidden). Go layer by layer until output
    //This loop cannot be executed out of order
    for (int layer = 1; layer < np->num_layers; layer++) {
        //this loop can be parallelized
        #pragma omp parallel num_threads(num_threads)  
        #pragma omp for schedule(static)
        for (int j = 0; j < np->num_units_in_layer[layer]; j++) {
            //add-in the bias value first. The multiplication by 1.0
            //is implicit           
            neural_net[layer][j] = np->layer_weight_matrices[layer][j][0];
            
            for (int k = 0; k < np->num_units_in_layer[layer - 1]; k++) {
                neural_net[layer][j] += np->layer_weight_matrices[layer][j][k + 1] * neural_net[layer- 1][k] ; 
            }
            //apply the thresholding function here
            neural_net[layer][j] = 1 / (1 + exp(-1.0 * neural_net[layer][j]));
        }
    }    
}
void feed_forwardOMP2(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example, int num_threads) {
    //can use openMP to parallelize this loop
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(dynamic, 1)
    for (int nL1 = 0; nL1 < np->num_units_in_layer[0]; nL1++) {           
            neural_net[0][nL1] = np->layer_weight_matrices[0][nL1][0];
            
            for (int k = 0; k < np->num_inputs; k++) {
                neural_net[0][nL1] += np->layer_weight_matrices[0][nL1][k + 1] *
                                        dataset->data[example][k]; 
            }
            //apply the thresholding function here
            neural_net[0][nL1] = 1 / (1 + exp(-1.0 * neural_net[0][nL1]));    
    }
    //Start at the first hidden layer (or output layer
    //if no hidden). Go layer by layer until output
    //This loop cannot be executed out of order
    for (int layer = 1; layer < np->num_layers; layer++) {
        //this loop can be parallelized
        #pragma omp parallel num_threads(num_threads)  
        #pragma omp for schedule(dynamic, 1)
        for (int j = 0; j < np->num_units_in_layer[layer]; j++) {
            //add-in the bias value first. The multiplication by 1.0
            //is implicit           
            neural_net[layer][j] = np->layer_weight_matrices[layer][j][0];
            
            for (int k = 0; k < np->num_units_in_layer[layer - 1]; k++) {
                neural_net[layer][j] += np->layer_weight_matrices[layer][j][k + 1] * neural_net[layer- 1][k] ; 
            }
            //apply the thresholding function here
            neural_net[layer][j] = 1 / (1 + exp(-1.0 * neural_net[layer][j]));
        }
    }    
}


void updateWeightsOMP1(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    int ncols = np->num_inputs + 1; //+1 for bias
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(static)
    for (int neuron = 0; neuron < np->num_units_in_layer[0];
                                neuron++) {
        //do the delta for the bias weight
        double neuron_err = errorMatrix[0][neuron];
        double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
        weights[neuron][0]+= delta0;
        for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * dataset->data[data_example][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta;
        }
    }
     
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(static)
    for(int layer = 0; layer < np->num_layers; layer++) {
        double **weights = np->layer_weight_matrices[layer];
        int ncols = np->num_units_in_layer[layer - 1] + 1; //+1 for bias
        for (int neuron = 0; neuron < np->num_units_in_layer[layer];
                                neuron++) {
            //do the delta for the bias weight
            double neuron_err = errorMatrix[layer][neuron];
            double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
            weights[neuron][0]+= delta0;
            for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err *  NN[layer -1][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta;
            }
        }
    }
}

void updateWeightsOMP2(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    int ncols = np->num_inputs + 1; //+1 for bias
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(dynamic, 1)
    for (int neuron = 0; neuron < np->num_units_in_layer[0];
                                neuron++) {
        //do the delta for the bias weight
        double neuron_err = errorMatrix[0][neuron];
        double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
        weights[neuron][0]+= delta0;
        for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * dataset->data[data_example][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta;
        }
    }
     
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(dynamic, 1)
    for(int layer = 0; layer < np->num_layers; layer++) {
        double **weights = np->layer_weight_matrices[layer];
        int ncols = np->num_units_in_layer[layer - 1] + 1; //+1 for bias
        for (int neuron = 0; neuron < np->num_units_in_layer[layer];
                                neuron++) {
            //do the delta for the bias weight
            double neuron_err = errorMatrix[layer][neuron];
            double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
            weights[neuron][0]+= delta0;
            for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err *  NN[layer -1][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta;
            }
        }
    }
}

void updateWeights_momentumOMP1(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    double **old_deltas = np->layer_weight_deltas[0];
    int ncols = np->num_inputs + 1; //+1 for bias
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(static)
    for (int neuron = 0; neuron < np->num_units_in_layer[0];
                                neuron++) {
        //do the delta for the bias weight
        double neuron_err = errorMatrix[0][neuron];
        double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
        weights[neuron][0]+= delta0 + np->alpha * old_deltas[neuron][0];
        old_deltas[neuron][0] = delta0;
        for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * dataset->data[data_example][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta + np->alpha * old_deltas[neuron][w];
                old_deltas[neuron][w] = delta;
            }
        }
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(static) 
    for(int layer = 1; layer < np->num_layers; layer++) {
        old_deltas = np->layer_weight_deltas[layer];
        double **weights = np->layer_weight_matrices[layer];
        int ncols = np->num_units_in_layer[layer - 1] + 1; //+1 for bias
        for (int neuron = 0; neuron < np->num_units_in_layer[layer];
                                neuron++) {
            //do the delta for the bias weight
            double neuron_err = errorMatrix[layer][neuron];
            double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
            weights[neuron][0]+= delta0 + np->alpha * old_deltas[neuron][0];
            old_deltas[neuron][0] = delta0;
            for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * NN[layer -1][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta + np->alpha * old_deltas[neuron][w];
                old_deltas[neuron][w] = delta;
            }
        }
    }
}

void updateWeights_momentumOMP2(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example, int num_threads) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    double **old_deltas = np->layer_weight_deltas[0];
    int ncols = np->num_inputs + 1; //+1 for bias
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(dynamic, 1)
    for (int neuron = 0; neuron < np->num_units_in_layer[0];
                                neuron++) {
        //do the delta for the bias weight
        double neuron_err = errorMatrix[0][neuron];
        double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
        weights[neuron][0]+= delta0 + np->alpha * old_deltas[neuron][0];
        old_deltas[neuron][0] = delta0;
        for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * dataset->data[data_example][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta + np->alpha * old_deltas[neuron][w];
                old_deltas[neuron][w] = delta;
            }
        }
    #pragma omp parallel num_threads(num_threads)  
    #pragma omp for schedule(dynamic, 1) 
    for(int layer = 1; layer < np->num_layers; layer++) {
        old_deltas = np->layer_weight_deltas[layer];
        double **weights = np->layer_weight_matrices[layer];
        int ncols = np->num_units_in_layer[layer - 1] + 1; //+1 for bias
        for (int neuron = 0; neuron < np->num_units_in_layer[layer];
                                neuron++) {
            //do the delta for the bias weight
            double neuron_err = errorMatrix[layer][neuron];
            double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
            weights[neuron][0]+= delta0 + np->alpha * old_deltas[neuron][0];
            old_deltas[neuron][0] = delta0;
            for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * NN[layer -1][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta + np->alpha * old_deltas[neuron][w];
                old_deltas[neuron][w] = delta;
            }
        }
    }
}
