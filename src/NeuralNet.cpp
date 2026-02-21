#include "NeuralNet.h"
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>
#include <thread>

double NeuralNet::Neuron::activationFunction(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNet::Neuron::activationDerivative(double x) {
    return x * (1.0 - x); // Assuming x is already the sigmoid output
}

NeuralNet::NeuralNet(std::vector<size_t> topology) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    size_t numLayers = topology.size();
    for (size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        layers.push_back(std::vector<Neuron>());
        
        size_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Decoupled explicit bias node per layer to simplify sizing
        for (size_t neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
            Neuron n;
            n.outputVal = 0.0;
            n.gradient = 0.0;
            n.bias = dis(gen);

            for (size_t c = 0; c < numOutputs; ++c) {
                 n.outputWeights.push_back(dis(gen));
            }
            layers.back().push_back(n);
        }
    }
}

void NeuralNet::feedForward(const std::vector<double>& inputValues) {
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0][i].outputVal = inputValues[i];
    }

    for (size_t layerNum = 1; layerNum < layers.size(); ++layerNum) {
        std::vector<Neuron>& prevLayer = layers[layerNum - 1];

        for (size_t n = 0; n < layers[layerNum].size(); ++n) {
            double sum = 0.0;
            for (size_t pn = 0; pn < prevLayer.size(); ++pn) {
                sum += prevLayer[pn].outputVal * prevLayer[pn].outputWeights[n];
            }
            layers[layerNum][n].outputVal = Neuron::activationFunction(sum + layers[layerNum][n].bias);
        }
    }
}

void NeuralNet::backpropagate(const std::vector<double>& targetValues, double learningRate) {
    // Output layer gradients
    std::vector<Neuron>& outputLayer = layers.back();
    for (size_t n = 0; n < outputLayer.size(); ++n) {
        double delta = targetValues[n] - outputLayer[n].outputVal;
        outputLayer[n].gradient = delta * Neuron::activationDerivative(outputLayer[n].outputVal);
    }

    // Hidden layer gradients
    for (size_t layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
        std::vector<Neuron>& hiddenLayer = layers[layerNum];
        std::vector<Neuron>& nextLayer = layers[layerNum + 1];

        for (size_t n = 0; n < hiddenLayer.size(); ++n) {
            double dow = 0.0;
            for (size_t nn = 0; nn < nextLayer.size(); ++nn) {
                dow += hiddenLayer[n].outputWeights[nn] * nextLayer[nn].gradient;
            }
            hiddenLayer[n].gradient = dow * Neuron::activationDerivative(hiddenLayer[n].outputVal);
        }
    }

    // Update weights and biases
    for (size_t layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
        std::vector<Neuron>& currentLayer = layers[layerNum];
        std::vector<Neuron>& prevLayer = layers[layerNum - 1];

        for (size_t n = 0; n < currentLayer.size(); ++n) {
            // Update bias
            currentLayer[n].bias += learningRate * currentLayer[n].gradient;
            
            // Update weights from previous layer
            for (size_t pn = 0; pn < prevLayer.size(); ++pn) {
                prevLayer[pn].outputWeights[n] += learningRate * currentLayer[n].gradient * prevLayer[pn].outputVal;
            }
        }
    }
}

void NeuralNet::train(const std::vector<std::vector<double>>& X_train, const std::vector<std::vector<double>>& Y_train, double learningRate, size_t epochs) {
    if (X_train.empty() || Y_train.empty() || X_train.size() != Y_train.size()) return;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double epochLoss = 0.0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            feedForward(X_train[i]);
            backpropagate(Y_train[i], learningRate);
            
            // Calculate Mean Squared Error loss
            double loss = 0.0;
            for (size_t j = 0; j < Y_train[i].size(); ++j) {
                double diff = Y_train[i][j] - layers.back()[j].outputVal;
                loss += diff * diff;
            }
            epochLoss += loss;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> active_duration = end_time - start_time;
        
        // Target 80% CPU utilization by sleeping for 25% of the active time
        // If active time is 80%, sleep time should be 20% -> sleep = active * (0.2 / 0.8) = active * 0.25
        double sleep_seconds = active_duration.count() * 0.25;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep_seconds));
        
        if (epoch % (epochs / 10 == 0 ? 1 : epochs / 10) == 0 || epoch == epochs - 1) {
            std::cout << "        [Agent Synthesis] Epoch " << epoch << " | MSE Loss: " << (epochLoss / X_train.size()) << "\n";
        }
    }
}

std::vector<double> NeuralNet::predict(const std::vector<double>& inputValues) {
    feedForward(inputValues);
    std::vector<double> resultVals;
    for (size_t n = 0; n < layers.back().size(); ++n) {
        resultVals.push_back(layers.back()[n].outputVal);
    }
    return resultVals;
}
