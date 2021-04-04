#include "NN_utils.h"

/*
 * Utility functions.
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

void print_matrix(double **A, int m, int n) {
    int i, j;
	for (i = 0; i < m; i++) {
    	for (j = 0; j < n; j++) {
        	printf("\t%5.10f",A[i][j]);
    	}
    	printf("\n");
	}
	printf("\n");
}


/*
 * Caller must check for NULL
 */
double **allocateMatrix(int rows, int cols) {
	double ** A;
    int r_index=0;
    A = (double **)malloc(sizeof(double *) * rows);
    if (A == NULL)
        return NULL;
    double * rowptr = A[0] = (double *)malloc(sizeof(double) * rows * cols);
    if (A[0] == NULL)
        return NULL;
        
    while (r_index < rows) {
		A[r_index++] = rowptr;
		rowptr += cols;
	}
    return A;
}
/*
 * Creates the jagged array, one row
 * per layer of the neural network.
 * Caller must check for NULL return.
 */
double **allocateNN(NN_parameters *np) {
    double **NN;
    int total_entries = 0;
    for (int i = 0; i < np-> num_layers; i++) {
        total_entries += np->num_units_in_layer[i];
    }
    //printf("\ntotal entries = %d\n", total_entries);
    NN = (double **)malloc(sizeof(double *) * np->num_layers);
    if(NN == NULL)
        return NULL;
    NN[0] = (double *)malloc(sizeof(double) * total_entries);
    if(NN[0] == NULL)
        return NULL;
    //printf("allocated \n");
    
    //the index for start of row 1
    //should be last index of row + 1
    int index = np->num_units_in_layer[0];
    
    //layer 0 pointer is already correct
    for (int i = 1; i < np->num_layers; i++) {
        NN[i] = (*NN + index);
        //printf("layer %d starting index = %d\n", i, index); 
        index+= np->num_units_in_layer[i];
    }
    return NN;
}

/*
 * Allocates the various rectangular
 * matrices (or vectors for single neuron layers)
 * representing the weigths to layer L of the neural net
 * 
 * Returns 0 if allocation is succesful, 1 otherwise.
 * 
 */ 
int allocateWeights(NN_parameters *np ) {
    srand(time(NULL));
    //for i = 0, the number of weights is num_inputs + 1
    np->layer_weight_matrices[0] = 
        allocateMatrix(np->num_units_in_layer[0], np->num_inputs + 1);
    for (int Wrow = 0; Wrow < np->num_units_in_layer[0]; Wrow++) {
        for (int Wcol = 0; Wcol < np->num_inputs + 1; Wcol++) {
           np->layer_weight_matrices[0][Wrow][Wcol] = -0.5 + (rand()/(double) RAND_MAX); 
        }
    }
    //for the rest of the layers, its the number of units
    //in the layer above, +1 for the bias
    for (int i = 1; i < np->num_layers; i++) {
     np->layer_weight_matrices[i] = 
        allocateMatrix(np->num_units_in_layer[i], np->num_units_in_layer[i - 1] + 1);
    if (np->layer_weight_matrices[i] == NULL)
        return 1;
        for (int Wrow = 0; Wrow < np->num_units_in_layer[i]; Wrow++) {
            for (int Wcol = 0; Wcol < np->num_units_in_layer[i - 1] + 1; Wcol++) {
                np->layer_weight_matrices[i][Wrow][Wcol] = 2*(-0.5 + (rand()/(double) RAND_MAX)); 
            }
        }        
    }
    return 0;
}

/*
 * This creates the parallel matrices to the 
 * neural net matrices
 * to hold the delta values when using
 * momentum.
 */
int allocateDeltas(NN_parameters *np ) {
    np->layer_weight_deltas[0] = 
        allocateMatrix(np->num_units_in_layer[0], np->num_inputs + 1);
    for (int Wrow = 0; Wrow < np->num_units_in_layer[0]; Wrow++) {
        for (int Wcol = 0; Wcol < np->num_inputs + 1; Wcol++) {
           np->layer_weight_deltas[0][Wrow][Wcol] = 0; 
        }
    }
    for (int i = 1; i < np->num_layers; i++) {
     np->layer_weight_deltas[i] = 
        allocateMatrix(np->num_units_in_layer[i], np->num_units_in_layer[i - 1] + 1);
    if (np->layer_weight_deltas[i] == NULL)
        return 1;
        for (int Wrow = 0; Wrow < np->num_units_in_layer[i]; Wrow++) {
            for (int Wcol = 0; Wcol < np->num_units_in_layer[i - 1] + 1; Wcol++) {
                np->layer_weight_deltas[i][Wrow][Wcol] = 0; 
            }
        }        
    }
    return 0;
}

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
double **createNeuralNet(NN_parameters *np) {
    double **NN;
    np->layer_weight_matrices =
                malloc((np->num_layers) * sizeof(double**));
    NN = allocateNN(np);
    if (NN == NULL)
        return NULL;
    np->layer_weight_deltas =
        malloc((np->num_layers) * sizeof(double**));
    if(allocateWeights(np) == 0) {
        return NN;
    } else {
        return NULL;
    }
}






void printNeuralNet(NN_parameters *np, double **NN) {
    for (int p = 0; p < np->num_layers; p++)
        print_matrix(&NN[p], 1, np->num_units_in_layer[p]); 
}

void clearNeuralNet(NN_parameters *np, double **NN) {
    for (int r = 0; r < np->num_layers; r++) {
        int num_cols = np->num_units_in_layer[r];
        for (int c = 0; c < num_cols; c++) {
            NN[r][c] = 0.0;
        }
    }
}

void printWeights(NN_parameters *np) {
    print_matrix(np->layer_weight_matrices[0], np->num_units_in_layer[0], np->num_inputs + 1);
    for (int i = 1; i < np->num_layers; i++)  {
         print_matrix(np->layer_weight_matrices[i], 
                        np->num_units_in_layer[i], np->num_units_in_layer[i - 1] + 1);
    }
}

/*
 * Creates an empty data set of the format described in
 * the NN_data_set struct.
 * Returns 0 if succesfull, 1 otherwise. 
 */    
int createDataSet(NN_data_set *nn) {
    nn->data = allocateMatrix(nn->num_examples , nn->num_inputs + nn->num_outputs);
    if (nn->data == NULL) {
        printf("Error allocating data set.\n");
        return 1;
    } else {
        return 0;
    }
}
/*
 * Created purely random inputs and outputs.
 * Purely for testing parallelization.
 */
NN_data_set createRandomDataSet(int num_examples, int num_inputs,
                                int num_outputs) {
    NN_data_set nn;
    nn.num_examples = num_examples;
    nn.num_inputs = num_inputs;
    nn.num_outputs = num_outputs;
    if (1 - createDataSet(&nn)) {
        //random inputs
        for (int i = 0; i < nn.num_examples; i++) {
            //first the random inputs between 0.0 and 1.0
            for (int j = 0; j < nn.num_inputs; j++) {
                nn.data[i][j] = rand()/(double) RAND_MAX;
            }
            //next the random outputs between 0.1 and 0.9
            for (int j = nn.num_inputs; j < nn.num_inputs + nn.num_outputs; j++) {
                nn.data[i][j] = 0.1 + 0.8 * rand()/(double) RAND_MAX;
            }
        }        
        return nn;
    }
    //return the data set with null matrix, error message 
    //printed in createDataSet already
    return nn;
}

void destroyDataSet(NN_data_set *nn) {
    free(nn->data[0]);
    free(nn->data);
}
/*
 * if DAT.data == NULL, something went wrong
 * Data should be normalized.
 * This will not work if the file contains a
 * header row (TODO).
 * Strictly works with commas only for now (TODO)!
 */
NN_data_set createDataSetFromCSV(char * path, int num_inputs, int num_outputs,
					int num_examples) {
    NN_data_set DAT;
    DAT.num_examples = num_examples;
    DAT.num_inputs = num_inputs;
    DAT.num_outputs = num_outputs;
    int success = 1 - createDataSet(&DAT);
    if (success) {
        success = 1 - readCSV(DAT.data, num_examples, 
            num_inputs + num_outputs, path);
        if (success)
           return DAT; 
    }
    printf("Error loading or creating data set!");
    DAT.data = NULL;
    return DAT;
}
/*
 * This will not work if the file contains a
 * header row (TODO).
 * Strictly works with commas only for now (TODO)!
 */
int readCSV(double **data, int r, int c, char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        printf("readCSV : Null file");
        return 1;
    }
    for (int i = 0; i < r; i++) {
        for(int j = 0; j < c - 1; j++) {
            fscanf(f, "%lf,", &(data[i][j]));
        }
        //read the last value without comma after it
        fscanf(f, "%lf", &(data[i][c -1]));
    }
    fclose(f);
    return 0;
}
/*
 * Loading and saving weights and parameters
 * very lightly tested!
 */
int saveWeights(NN_parameters *np, char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        printf("saveWeights : Null file");
        return 1;
    }
    //first do layer 0
    double **weights = np->layer_weight_matrices[0];
    for (int i = 0; i < np->num_units_in_layer[0]; i++) {
        for (int j = 0; j < np->num_inputs + 1; j++) {
            //maybe some loss of precision here
            fprintf(f, "%17.17lf%s", weights[i][j],
                (j < np->num_inputs) ? "," : "\n");    
        }
    }
    
    //next do the rest of the layers
    for (int L = 1; L < np->num_layers; L++)  {
         //print_matrix(np->layer_weight_matrices[i], 
         //               np->num_units_in_layer[i], np->num_units_in_layer[i - 1] + 1);
        weights = np->layer_weight_matrices[L];
        for (int i = 0; i < np->num_units_in_layer[L]; i++) {
            for (int j = 0; j < np->num_units_in_layer[L - 1] + 1; j++) {
                //maybe some loss of precision here
                fprintf(f, "%17.17lf%s", weights[i][j],
                    (j < np->num_units_in_layer[L - 1]) ? "," : "\n");    
            }
        }
    }
    fclose(f);
    return 0;
}

/*
 * Loading and saving weights and parameters
 * very lightly tested!
 */
int saveParams(NN_parameters *np, char *params_path, char *weights_path, int save_weights) {
    if (save_weights) {
        if (saveWeights(np, weights_path)) {
            printf("Error saving weights\n");
            return 1;
        }
            
    }
    FILE *fp = fopen(params_path, "w");    
    if (fp == NULL) {
        printf("saveParams : Error creating param file\n");
        return 1;
    }
    //saves all params in single column text file
    //last n lines are the number of neurons in
    //layer n, however many layers there are
    fprintf(fp, "%d\n", np->num_inputs);
    fprintf(fp, "%d\n", np->num_outputs);
    fprintf(fp, "%d\n", np->num_layers);
    //dont save the last layer, its num_outputs wide!
    for (int i = 0; i < np->num_layers - 1; i++) {
        fprintf(fp, "%d\n", np->num_units_in_layer[i]);
    }
    fclose(fp);
    return 0;
}
/*
 * Caller check for NULL return values
 * does not create the error matrix 
 * or deltas.
 * 
 * Loading and saving weights and parameters
 * very lightly tested!
 */
double ** loadNN(NN_parameters *np, 
            char *param_path, char *weight_path, int load_weights) {
    double ** NN = NULL;
    FILE *fp = fopen(param_path, "r");
    if (fp == NULL) {
        printf("loadNN error opening param file!\n");
        return NN;
    }
    //int num_inputs, num_outputs, num_layers;
    fscanf(fp, "%d", &np->num_inputs);
    fscanf(fp, "%d", &np->num_outputs);
    fscanf(fp, "%d", &np->num_layers);
    printf("got params\n");
    np->num_units_in_layer = 
                (int *)malloc(sizeof(int) * np->num_layers);
    for (int i = 0; i < np->num_layers - 1; i++) {
        fscanf(fp, "%d", &np->num_units_in_layer[i]);
        printf("layer %d: num units:%d\n", i, np->num_units_in_layer[i]);
    }
    np->num_units_in_layer[np->num_layers - 1] = np->num_outputs;
    printf("layer %d: num units:%d\n",  np->num_layers -1,  np->num_units_in_layer[np->num_layers - 1]);
    printf("got all layers\n");
    printf("allocating neural network\n");
    NN = createNeuralNet(np);
    clearNeuralNet(np, NN);
    allocateWeights(np);
    fclose(fp);
    printf("got here");
    //load the weights
    FILE *fw = fopen(weight_path, "r");
    for (int n = 0; n < np->num_units_in_layer[0]; n++) {
        for (int w = 0; w < np->num_inputs + 1; w++) {
            fscanf(fw, (w < np->num_inputs) ? "%lf," : "%lf", &np->layer_weight_matrices[0][n][w]);
        }
    }
    for (int layer = 1; layer < np->num_layers; layer++) {
        for (int n = 0; n < np->num_units_in_layer[layer]; n++) {
            for (int w = 0; w < np->num_units_in_layer[layer -1] + 1; w++) {
                fscanf(fw, (w < np->num_units_in_layer[layer -1]) ? "%lf," : "%lf", &np->layer_weight_matrices[layer][n][w]);
            }
        }
        
    }
    fclose(fw);
    return NN;
    
}

/*
 * This timing code was adopted from
 * Matt Alden's starter code for TCSS 372,
 * and Dave Kaplan's Homework 5 Wi 14
 * submission on Catalyst
 */
int ms_diff(struct timespec start, struct timespec stop) {

    return 1000 * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec) / 1000000;

}
