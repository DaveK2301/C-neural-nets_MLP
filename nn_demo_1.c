// Copyright 2016 - 2021 David Kaplan

/*
 * Simple demonstration code for multi-layer perceptron
 * code. This code was written for UW MSCC TCSS 570
 * Parallel Computing, Summer 2016, as test code for parallelization.
 * The basic functions AND, OR, and XOR can be tested
 * by uncommenting various sections and simple datasets.
 * 
 */


#include <stdio.h>
#include <math.h>
#include "NN_functions.h"

int main(int argc, char *argv[])
{
    
	/*if (argc != 1) {
    	printf("Usage: ");
	}*/
    //create a simple data set, 2 inputs, 1 output
    NN_data_set DAT;
    DAT.num_examples = 4;
    DAT.num_inputs = 2;
    DAT.num_outputs = 1;
    createDataSet(&DAT);
    
    //clock_gettime(CLOCK_REALTIME, &start);    
    //Stop tracking execution time.
    //clock_gettime(CLOCK_REALTIME, &stop);
    
    //NN_data_set DAT = createDataSetFromCSV("training_data_gray.csv", 36000, 2, 683);
    
    //NN_data_set DAT = createRandomDataSet(40, 20, 4);
    
    //populate the data set for the AND function
    /*DAT.data[0][0] = 0.1; //x0
    DAT.data[0][1] = 0.1; //x1
    DAT.data[0][2] = 0.1; //output1
    
    DAT.data[1][0] = 0.1; //x0
    DAT.data[1][1] = 0.9; //x1
    DAT.data[1][2] = 0.1; //output1
    
    DAT.data[2][0] = 0.9; //x0
    DAT.data[2][1] = 0.1; //x1
    DAT.data[2][2] = 0.1; //output1
    
    DAT.data[3][0] = 0.9; //x0
    DAT.data[3][1] = 0.9; //x1
    DAT.data[3][2] = 0.9; //output1*/
    
    //populate the data set for the OR function
    /*DAT.data[0][0] = 0.0; //x0
    DAT.data[0][1] = 0.0; //x1
    DAT.data[0][2] = 0.05; //output1
    
    DAT.data[1][0] = 0.0; //x0
    DAT.data[1][1] = 1.0; //x1
    DAT.data[1][2] = 0.95; //output1
    
    DAT.data[2][0] = 1.0; //x0
    DAT.data[2][1] = 0.0; //x1
    DAT.data[2][2] = 0.95; //output1
    
    DAT.data[3][0] = 1.0; //x0
    DAT.data[3][1] = 1.0; //x1
    DAT.data[3][2] = 0.95; //output1*/
    
    //populate the data set for the XOR function
    DAT.data[0][0] = 0.0; //x0
    DAT.data[0][1] = 0.0; //x1
    DAT.data[0][2] = 0.05; //output1
    
    DAT.data[1][0] = 0.0; //x0
    DAT.data[1][1] = 1.0; //x1
    DAT.data[1][2] = 0.95; //output1
    
    DAT.data[2][0] = 1.0; //x0
    DAT.data[2][1] = 0.0; //x1
    DAT.data[2][2] = 0.95; //output1
    
    DAT.data[3][0] = 1.0; //x0
    DAT.data[3][1] = 1.0; //x1
    DAT.data[3][2] = 0.05; //output1
    
    
    //printf("data set DAT:\n");
    //print_matrix(DAT.data, 4, 3);
    
    NN_parameters NNparams;
    NNparams.num_inputs = DAT.num_inputs;
    NNparams.num_outputs = DAT.num_outputs;
    NNparams.Eta = 0.2;
    NNparams.num_layers = 3;
    NNparams.num_units_in_layer = 
                (int *)malloc(sizeof(int) * NNparams.num_layers);
    NNparams.num_units_in_layer[0] = 2;    
    NNparams.num_units_in_layer[1] = 7;
    NNparams.num_units_in_layer[2] = 1;
    //NNparams.num_units_in_layer[3] = 1;
    //NNparams.num_units_in_layer[4] = 2;
    
    //creates the NN as well as the weights array
    //contained in the struct
    double **NN = createNeuralNet(&NNparams);
    //printf("**Initial neural network config**:\n");
    //printNeuralNet(&NNparams, NN);
    //the array to hold errors 
     
    double **err = allocateNN(&NNparams);

    // *** these steps must be taken to use momentum ****
    //initialize the extra weights arrays
    //for the momentum term
    
    allocateDeltas(&NNparams);
    
    //set momentum
    
    NNparams.alpha = 0.1;
    
    
    // **************************************************

    //clock_gettime(CLOCK_REALTIME, &start); 
    for (int i = 0; i < 50000; i++) {
        double sum = 0.0;
        for(int j = 0; j < DAT.num_examples; j++) {
            feed_forward(&NNparams, NN, &DAT, j);
            sum+= backPropagateError(&NNparams, NN, err, j, &DAT);
            updateWeights_momentum(&NNparams, NN, err, &DAT, j);
        }
        //get the average squared error, 
        double avg_err = sum /(double)DAT.num_examples;
        double rmse = sqrt(avg_err);
        //this is good for importing to excel
        if (i % 100 == 0)
            printf("%d, %2.10f\n", i, rmse);
    }
    //clock_gettime(CLOCK_REALTIME, &stop);

    //printf("layer weights matrices after run:\n");
    //printWeights(&NNparams);
             
    //destroy data set
    destroyDataSet(&DAT);	

}
