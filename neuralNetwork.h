#include <math.h>

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H /*NEURAL_NETWORK_H*/

#define INPUT_LAYER 0
#define OUTPUT_LAYER 1

// Activation functions and their derivatives
// Keep in mind that each functions and its derivative
// must be used together and are made for specific purposes
// Please, search for the proper activation function for your
// problem on the internet
static double Sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

static double SigmoidPrime(double x) {
    return x * (1 - x);
}

static double Softmax(double x) {
    return exp(x) / (exp(x) + exp(-x));
}

static double SoftmaxPrime(double x) {
    return (1 - x) * x;
}

static double Tanh(double x) {
    return tanh(x);
}

static double TanhPrime(double x) {
    return 1 - (x * x);
}

static double ReLU(double x) {
    return x > 0 ? x : 0;
}

static double ReLUPrime(double x) {
    return x > 0 ? 1 : 0;
}

static double LeakyReLU(double x) {
    return x > 0 ? x : 0.01 * x;
}

static double LeakyReLUPrime(double x) {
    return x > 0 ? 1 : 0.01;
}

static double Identity(double x) {
    return x;
}

static double IdentityPrime(double x) {
    return 1;
}

// The network struct holds all the information about the network
struct network {
    // The number of nodes in each layer
    int inputLayerSize;
    int hiddenLayerSize;
    int outputLayerSize;

    // The number of training sets
    int numTrainingSets;
    int numTargets;

    // Training sets
    int** trainingSets;

    // Target used for training
    int** targets;

    // Arrays to hold the values of the 
    // inputs, hiddens, outputs and biases
    double* inputs;
    double* hiddens;
    double* outputs;
    double* bias;

    // Three dimensional array to hold the weights. Use layer constants 
    // to index into the array
    //      weights[layer][node][weight]
    // layer 0 is the input layer and layer 1 is the output layer
    // node is the node in the layer
    // weight is the weight between input node and hidden node 
    // or hidden node and output node
    double*** weights;

    // The activation function used in the network
    double (*func)(double);
    double (*funcPrime)(double);

}typedef network;

network* createNetwork(int inputLayerSize, int hiddenLayerSize, int outputLayerSize);
void initWeights(network* net);
void initBiases(network* net);
void train(network* net, int epochs, double learningRate);
void freeNetwork(network* net);

#endif /*NEURAL_NETWORK_H*/