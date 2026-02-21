#pragma once
#include <vector>

// A highly lightweight, dependency-free Dense Feed-Forward Neural Network for conceptual prototype
class NeuralNet {
public:
    NeuralNet(std::vector<size_t> topology);

    // Forward pass
    void feedForward(const std::vector<double>& inputValues);

    // Get final output layer predictions
    std::vector<double> getResults() const;

private:
    struct Neuron {
        double outputVal;
        std::vector<double> outputWeights;
        double bias;
        
        // Activation
        static double activationFunction(double x); // ReLU or Sigmoid
    };

    std::vector<std::vector<Neuron>> layers; // layers[layerNum][neuronNum]
};
