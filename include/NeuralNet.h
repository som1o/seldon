#pragma once
#include <vector>

// A lightweight, dependency-free Dense Feed-Forward Neural Network for predictive modeling
class NeuralNet {
public:
    NeuralNet(std::vector<size_t> topology);

    // Train the network using stochastic gradient descent
    void train(const std::vector<std::vector<double>>& X_train, const std::vector<std::vector<double>>& Y_train, double learningRate, size_t epochs);

    // Forward pass prediction
    std::vector<double> predict(const std::vector<double>& inputValues);

private:
    void feedForward(const std::vector<double>& inputValues);
    void backpropagate(const std::vector<double>& targetValues, double learningRate);

    struct Neuron {
        double outputVal;
        std::vector<double> outputWeights;
        double bias;
        double gradient;
        
        // Activation
        static double activationFunction(double x); // Sigmoid
        static double activationDerivative(double x); // Derivative of Sigmoid assuming x is already activation output
    };

    std::vector<std::vector<Neuron>> layers; // layers[layerNum][neuronNum]
};
