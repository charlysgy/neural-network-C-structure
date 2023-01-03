#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "neuralNetwork.h"


/// @brief Create a struct network object
/// @param inputSize The number of inputs
/// @param hiddenSize The number of hidden nodes
/// @param outputSize The number of outputs
/// @return A pointer to the network object
network* createNetwork(int inputSize, int hiddenSize, int outputSize) {
    network* net = calloc(1, sizeof(network));
    net->inputLayerSize = inputSize;
    net->hiddenLayerSize = hiddenSize;
    net->outputLayerSize = outputSize;

    net->inputs = calloc(inputSize, sizeof(double));
    net->hiddens = calloc(hiddenSize, sizeof(double));
    net->outputs = calloc(outputSize, sizeof(double));
    net->bias = calloc(hiddenSize, sizeof(double));

    // Allocate weights memory space in a three dimensional array
    net->weights = calloc(2, sizeof(double**));
    net->weights[INPUT_LAYER] = calloc(inputSize, sizeof(double*));
    net->weights[OUTPUT_LAYER] = calloc(outputSize, sizeof(double*));

    for (int i = 0; i < inputSize; i++) {
        net->weights[INPUT_LAYER][i] = calloc(hiddenSize, sizeof(double));
    }
    for (int i = 0; i < outputSize; i++) {
        net->weights[OUTPUT_LAYER][i] = calloc(hiddenSize, sizeof(double));
    }

    return net;
}

/// @brief Initialise the weights of the network
/// @param net The network to initialise
/// @return void
void initWeights(network* net) {
    srand(time(NULL));
    for (int i = 0; i < net->inputLayerSize; i++) {
        for (int j = 0; j < net->hiddenLayerSize; j++) {
            net->weights[INPUT_LAYER][i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
    for (int i = 0; i < net->outputLayerSize; i++) {
        for (int j = 0; j < net->hiddenLayerSize; j++) {
            net->weights[OUTPUT_LAYER][i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

/// @brief Initialise the biases of the network
/// @param net The network to initialise
/// @return void
void initBiases(network* net) {
    for (int i = 0; i < net->hiddenLayerSize; i++) {
        net->bias[i] = (double)rand() / (double)RAND_MAX;
    }
}

/// @brief Initialise the network
/// @param net The network to initialise
/// @param inputs The inputs to the network
/// @return void
void initNetwork(network* net, int* inputs) {
    initWeights(net);
    initBiases(net);
    for (int i = 0; i < net->inputLayerSize; i++) {
        net->inputs[i] = inputs[i];
    }
}

/// @brief Train the network
/// @param net The network to train
/// @param epochs The number of epochs to train for
/// @param learningRate The learning rate
/// @return void
void train(network* net, int epochs, double learningRate) {
    if(net->targets == NULL || net->trainingSets == NULL){
        printf("Error, network has no targets set or training sets for training\n");
        return;
    }
    if(net->func == NULL || net->funcPrime == NULL){
        printf("Error, network has no activation function set or one is missing\n");
        return;
    }
    double (*func)(double) = net->func;
    double (*funcPrime)(double) = net->funcPrime;
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < net->numTrainingSets; j++) {
            // Sets the inputs following the training sets
            for (int k = 0; k < net->inputLayerSize; k++) {
                net->inputs[k] = net->trainingSets[j][k];
            }

            // Forward propagation (computes each hidden node)
            for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                double sum = 0;
                for(int input = 0; input < net->inputLayerSize; input++){
                    sum += net->inputs[input] * net->weights[INPUT_LAYER][input][hidden];
                }
                net->hiddens[hidden] = func(sum + net->bias[hidden]);
            }

            // Forward propagation (computes each output node)
            for(int outputs = 0; outputs < net->outputLayerSize; outputs++){
                double sum = 0;
                for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                    sum += net->hiddens[hidden] * net->weights[OUTPUT_LAYER][outputs][hidden];
                }
                net->outputs[outputs] = func(sum);
            }

            // Back propagation (computes the error for each output node)
            double* outputErrors = calloc(net->outputLayerSize, sizeof(double));
            for(int outputs = 0; outputs < net->outputLayerSize; outputs++){
                outputErrors[outputs] = net->targets[j][outputs] - net->outputs[outputs];
            }

            // Back propagation (computes the error for each hidden node)
            double* hiddenErrors = calloc(net->hiddenLayerSize, sizeof(double));
            for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                double sum = 0;
                for(int outputs = 0; outputs < net->outputLayerSize; outputs++){
                    sum += outputErrors[outputs] * net->weights[OUTPUT_LAYER][outputs][hidden];
                }
                hiddenErrors[hidden] = sum;
            }

            // Back propagation (updates the weights between the hidden and output layers)
            for(int outputs = 0; outputs < net->outputLayerSize; outputs++){
                for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                    net->weights[OUTPUT_LAYER][outputs][hidden] += \
                        learningRate * outputErrors[outputs] * \
                        funcPrime(net->outputs[outputs]) * net->hiddens[hidden];
                }
            }

            // Back propagation (updates the weights between the input and hidden layers)
            for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                for(int input = 0; input < net->inputLayerSize; input++){
                    net->weights[INPUT_LAYER][input][hidden] += \
                        learningRate * hiddenErrors[hidden] * \
                        funcPrime(net->hiddens[hidden]) * net->inputs[input];
                }
            }

            // Back propagation (updates the bias)
            for(int hidden = 0; hidden < net->hiddenLayerSize; hidden++){
                net->bias[hidden] += learningRate * hiddenErrors[hidden] * \
                    funcPrime(net->hiddens[hidden]);
            }

            free(outputErrors);
            free(hiddenErrors);

            if(i == epochs - 1)
                printf("Training set %d, output: %f, target: %d\n", j, net->outputs[0], net->targets[j][0]);
        }
    }
}

/// @brief Free the memory allocated to the network
/// @param net The network to free
/// @return void
void freeNetwork(network* net) {
    // Free 1 dimensional arrays
    free(net->inputs);
    free(net->hiddens);
    free(net->outputs);
    free(net->bias);

    // Free 2 dimensional arrays
    for (int i = 0; i < net->numTrainingSets; i++) {
        free(net->trainingSets[i]);
    }
    free(net->trainingSets);

    for(int i = 0; i < net->numTargets; i++){
        free(net->targets[i]);
    }
    free(net->targets);

    // Free 3 dimensional arrays
    for (int i = 0; i < net->inputLayerSize; i++) {
        free(net->weights[INPUT_LAYER][i]);
    }
    for (int i = 0; i < net->outputLayerSize; i++) {
        free(net->weights[OUTPUT_LAYER][i]);
    }
    free(net->weights[INPUT_LAYER]);
    free(net->weights[OUTPUT_LAYER]); // Works only with 1 lay
    free(net->weights);


    free(net);
}