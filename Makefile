nn_demo_1: nn_demo_1.c
	gcc -o nn_demo_1 nn_test1.c NN_functions.c NN_utils.c -lm -fopenmp
