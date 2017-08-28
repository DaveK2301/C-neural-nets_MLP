#include <math.h>
#include "NN_functions.h"

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


/*
 * Starting at the first layer past the input nodes,
 * applies the weights to the nodes in the layer above
 * and sets the output value of each node, layer by layer,
 * until the output layer has its value updated.
 * This version uses the sigmoid (logistic) function
 * for thresholding the outputs.
 */
void feed_forward(NN_parameters *np, double **neural_net, 
                                NN_data_set *dataset, int example) {
    //can use openMP to parallelize this loop
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
                        NN_data_set *dataset) {
    //hold the sum of the squared errors for this example
    double E = 0.0;
    
    //calculate the errors for the output units
    int output_layer = np->num_layers - 1;
    int outputs = np->num_units_in_layer[output_layer];
    int col_offset = dataset->num_inputs;
    for (int i = 0; i < outputs; i++) {
        double o = NN[output_layer][i];
        double target = dataset->data[example_num][col_offset + i];
        double t_minus_o = target - o;
        E += t_minus_o * t_minus_o;
        errorMatrix[output_layer][i] = o * (1.0 - o) * t_minus_o;
    }
    //now do the calcs for all the hidden layers
    //loop backwards through the hidden layers
    for (int row = output_layer - 1; row >= 0; row--) {
        int num_neurons = np->num_units_in_layer[row];
        //we want the weights to the layer BELOW this layer
        int num_weights = np->num_units_in_layer[row + 1];
        double **weights = np->layer_weight_matrices[row + 1];
        
        for(int n = 0; n < num_neurons; n++) {
            double sum = 0.0;
            for(int k = 0; k < num_weights; k++) {
                 sum+= weights[k][1 + n] * errorMatrix[row + 1][k];
            }
            double o = NN[row][n];
            errorMatrix[row][n] = o *(1.0 - o) * sum;
        }
    }
    //printf("Total squared error for example %d : %5.10f\n", example_num, E);
    return E;
}
void updateWeights(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    int ncols = np->num_inputs + 1; //+1 for bias
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
     
    for(int layer = 1; layer < np->num_layers; layer++) {
        double **weights = np->layer_weight_matrices[layer];
        int ncols = np->num_units_in_layer[layer - 1] + 1; //+1 for bias
        for (int neuron = 0; neuron < np->num_units_in_layer[layer];
                                neuron++) {
            //do the delta for the bias weight
            double neuron_err = errorMatrix[layer][neuron];
            double delta0 = np->Eta * neuron_err; //x_j_i = 1.0 for the bias
            weights[neuron][0]+= delta0;
            for (int w = 1; w < ncols; w++) {
                double delta = np->Eta * neuron_err * NN[layer -1][w - 1]; //w - 1 because of bias offset
                weights[neuron][w]+= delta;
            }
        }
    }
}


void updateWeights_momentum(NN_parameters *np, double **NN, 
                    double **errorMatrix, NN_data_set *dataset, int data_example) {
    /* First take care of layer 0, which 
     * has x_j_i's coming from the data set.
     */
    double **weights = np->layer_weight_matrices[0];
    double **old_deltas = np->layer_weight_deltas[0];
    int ncols = np->num_inputs + 1; //+1 for bias
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





