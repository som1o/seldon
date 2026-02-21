#include "NeuralNet.h"
#include <cmath>
#include <cstdlib>

// Basic Sigmoid for prototype predictive normalization
double NeuralNet::Neuron::activationFunction(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

NeuralNet::NeuralNet(std::vector<size_t> topology) {
    size_t numLayers = topology.size();
    for (size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        layers.push_back(std::vector<Neuron>());
        
        // Num outputs for current neuron is the number of neurons in the NEXT layer
        // (0 if it's the output layer)
        size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Add neurons to layer
        for (size_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            Neuron n;
            n.outputVal = 0.0;
            n.bias = ((rand() / double(RAND_MAX)) * 2.0) - 1.0; // Random [-1, 1]

            for(size_t c = 0; c < numOutputs; ++c){
                 n.outputWeights.push_back(((rand() / double(RAND_MAX)) * 2.0) - 1.0);
            }
            layers.back().push_back(n);
        }
        
        // Force the bias node's output to 1.0 (the last neuron created above is the bias node)
        layers.back().back().outputVal = 1.0;
    }
}

void NeuralNet::feedForward(const std::vector<double>& inputValues) {
    // Assign inputs to the input layer neurons
    // Note: layers[0].size() includes the bias node, so -1
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0][i].outputVal = inputValues[i];
    }

    // Forward propagate
    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        std::vector<Neuron>& prevLayer = layers[layerNum - 1];

        for (size_t n = 0; n < layers[layerNum].size() - 1; ++n) { // Exclude bias
            double sum = 0.0;
            
            // Sum previous layer outputs * weights
            for (size_t pn = 0; pn < prevLayer.size(); ++pn) {
                sum += prevLayer[pn].outputVal * prevLayer[pn].outputWeights[n];
            }
            
            layers[layerNum][n].outputVal = Neuron::activationFunction(sum + layers[layerNum][n].bias);
        }
    }
}

std::vector<double> NeuralNet::getResults() const {
    std::vector<double> resultVals;
    for (size_t n = 0; n < layers.back().size() - 1; ++n) {
        resultVals.push_back(layers.back()[n].outputVal);
    }
    return resultVals;
}
