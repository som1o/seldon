#include "NeuralNet.h"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>

NeuralNet::NeuralNet(std::vector<size_t> topology) : topology(topology) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (size_t l = 0; l < topology.size(); ++l) {
        layers.push_back(std::vector<Neuron>());
        size_t numOutputs = (l == topology.size() - 1) ? 0 : topology[l + 1];

        for (size_t n = 0; n < topology[l]; ++n) {
            Neuron neuron;
            neuron.outputVal = 0.0;
            neuron.bias = dis(gen);
            neuron.gradient = 0.0;
            
            for (size_t w = 0; w < numOutputs; ++w) {
                neuron.weights.push_back(dis(gen));
                neuron.m_weights.push_back(0.0);
                neuron.v_weights.push_back(0.0);
            }
            layers.back().push_back(neuron);
        }
    }
}

double NeuralNet::activate(double x, Activation act) {
    switch (act) {
        case Activation::RELU: return std::max(0.0, x);
        case Activation::TANH: return std::tanh(x);
        case Activation::SIGMOID: return 1.0 / (1.0 + std::exp(-x));
        default: return x;
    }
}

double NeuralNet::activateDerivative(double out, Activation act) {
    switch (act) {
        case Activation::RELU: return out > 0 ? 1.0 : 0.0;
        case Activation::TANH: return 1.0 - out * out;
        case Activation::SIGMOID: return out * (1.0 - out);
        default: return 1.0;
    }
}

void NeuralNet::feedForward(const std::vector<double>& inputValues, Activation act, Activation outputAct, bool isTraining, double dropoutRate) {
    for (size_t i = 0; i < inputValues.size(); ++i) {
        layers[0][i].outputVal = inputValues[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t l = 1; l < layers.size(); ++l) {
        std::vector<Neuron>& prevLayer = layers[l - 1];
        bool isOutputLayer = (l == layers.size() - 1);

        for (size_t n = 0; n < layers[l].size(); ++n) {
            double sum = 0.0;
            for (size_t pn = 0; pn < prevLayer.size(); ++pn) {
                sum += prevLayer[pn].outputVal * prevLayer[pn].weights[n];
            }
            
            double out = activate(sum + layers[l][n].bias, isOutputLayer ? outputAct : act);

            // Apply Dropout only on hidden layers during training
            if (isTraining && !isOutputLayer && dropoutRate > 0.0) {
                if (dis(gen) < dropoutRate) {
                    layers[l][n].outputVal = 0.0;
                    layers[l][n].dropMask = true;
                } else {
                    layers[l][n].outputVal = out / (1.0 - dropoutRate); // Inverted Dropout scaling
                    layers[l][n].dropMask = false;
                }
            } else {
                layers[l][n].outputVal = out;
                layers[l][n].dropMask = false;
            }
        }
    }
}

void NeuralNet::backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step) {
    std::vector<Neuron>& outputLayer = layers.back();
    for (size_t n = 0; n < outputLayer.size(); ++n) {
        double delta = targetValues[n] - outputLayer[n].outputVal;
        outputLayer[n].gradient = delta * activateDerivative(outputLayer[n].outputVal, hp.outputActivation);
    }

    for (size_t l = layers.size() - 2; l > 0; --l) {
        std::vector<Neuron>& hiddenLayer = layers[l];
        std::vector<Neuron>& nextLayer = layers[l + 1];

        for (size_t n = 0; n < hiddenLayer.size(); ++n) {
            if (hiddenLayer[n].dropMask) {
                hiddenLayer[n].gradient = 0.0;
                continue;
            }
            double dow = 0.0;
            for (size_t nn = 0; nn < nextLayer.size(); ++nn) {
                dow += hiddenLayer[n].weights[nn] * nextLayer[nn].gradient;
            }
            hiddenLayer[n].gradient = dow * activateDerivative(hiddenLayer[n].outputVal, hp.activation);
        }
    }

    // Adam Optimizer Constants
    const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    for (size_t l = layers.size() - 1; l > 0; --l) {
        std::vector<Neuron>& currentLayer = layers[l];
        std::vector<Neuron>& prevLayer = layers[l - 1];

        for (size_t n = 0; n < currentLayer.size(); ++n) {
            // Update Bias
            double grad_bias = currentLayer[n].gradient;
            if (hp.optimizer == Optimizer::ADAM) {
                currentLayer[n].m_bias = beta1 * currentLayer[n].m_bias + (1.0 - beta1) * grad_bias;
                currentLayer[n].v_bias = beta2 * currentLayer[n].v_bias + (1.0 - beta2) * grad_bias * grad_bias;
                double m_hat = currentLayer[n].m_bias / (1.0 - std::pow(beta1, t_step));
                double v_hat = currentLayer[n].v_bias / (1.0 - std::pow(beta2, t_step));
                currentLayer[n].bias += hp.learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            } else {
                currentLayer[n].bias += hp.learningRate * grad_bias;
            }

            for (size_t pn = 0; pn < prevLayer.size(); ++pn) {
                double grad_w = currentLayer[n].gradient * prevLayer[pn].outputVal;
                // L2 Regularization (Weight Decay)
                grad_w += hp.l2Lambda * prevLayer[pn].weights[n];

                if (hp.optimizer == Optimizer::ADAM) {
                    prevLayer[pn].m_weights[n] = beta1 * prevLayer[pn].m_weights[n] + (1.0 - beta1) * grad_w;
                    prevLayer[pn].v_weights[n] = beta2 * prevLayer[pn].v_weights[n] + (1.0 - beta2) * grad_w * grad_w;
                    double m_hat = prevLayer[pn].m_weights[n] / (1.0 - std::pow(beta1, t_step));
                    double v_hat = prevLayer[pn].v_weights[n] / (1.0 - std::pow(beta2, t_step));
                    prevLayer[pn].weights[n] += hp.learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                } else {
                    prevLayer[pn].weights[n] += hp.learningRate * grad_w;
                }
            }
        }
    }
}

void NeuralNet::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, const Hyperparameters& hp) {
    if (X.empty()) return;

    size_t valSize = static_cast<size_t>(X.size() * hp.valSplit);
    size_t trainSize = X.size() - valSize;

    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    double bestValLoss = 1e30;
    int patienceCounter = 0;
    size_t t_step = 0;

    for (size_t epoch = 0; epoch < hp.epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), std::random_device());

        double trainLoss = 0.0;
        for (size_t i = 0; i < trainSize; i += hp.batchSize) {
            size_t currentBatchEnd = std::min(i + hp.batchSize, trainSize);
            t_step++;

            for (size_t b = i; b < currentBatchEnd; ++b) {
                feedForward(X[indices[b]], hp.activation, hp.outputActivation, true, hp.dropoutRate);
                backpropagate(Y[indices[b]], hp, t_step);

                for (size_t j = 0; j < Y[indices[b]].size(); ++j) {
                    double diff = Y[indices[b]][j] - layers.back()[j].outputVal;
                    trainLoss += diff * diff;
                }
            }
        }

        // Validation phase
        double valLoss = 0.0;
        if (valSize > 0) {
            for (size_t i = trainSize; i < X.size(); ++i) {
                feedForward(X[indices[i]], hp.activation, hp.outputActivation, false);
                for (size_t j = 0; j < Y[indices[i]].size(); ++j) {
                    double diff = Y[indices[i]][j] - layers.back()[j].outputVal;
                    valLoss += diff * diff;
                }
            }
            valLoss /= valSize;
        }

        if (hp.verbose && (epoch % std::max(size_t(1), hp.epochs / 10) == 0 || epoch == hp.epochs - 1)) {
            std::cout << "[Epoch " << epoch << "] Train Loss: " << (trainLoss / trainSize) 
                      << " | Val Loss: " << valLoss << std::endl;
        }

        // Early Stopping
        if (valSize > 0) {
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= hp.earlyStoppingPatience) {
                    if (hp.verbose) std::cout << "Early stopping triggered at epoch " << epoch << std::endl;
                    break;
                }
            }
        }
    }
}

std::vector<double> NeuralNet::predict(const std::vector<double>& inputValues) {
    feedForward(inputValues, Activation::RELU, Activation::SIGMOID, false); // Default RELU but usually load from saved model state
    std::vector<double> result;
    for (auto& n : layers.back()) result.push_back(n.outputVal);
    return result;
}

bool NeuralNet::saveModel(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) return false;
    out << topology.size() << "\n";
    for (size_t t : topology) out << t << " ";
    out << "\n";

    for (const auto& layer : layers) {
        for (const auto& neuron : layer) {
            out << neuron.bias << " ";
            for (double w : neuron.weights) out << w << " ";
            out << "\n";
        }
    }
    return true;
}

bool NeuralNet::loadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return false;
    size_t topSize;
    in >> topSize;
    topology.clear();
    for (size_t i = 0; i < topSize; ++i) {
        size_t t; in >> t; topology.push_back(t);
    }
    
    // Re-init with correct topology
    *this = NeuralNet(topology);

    for (auto& layer : layers) {
        for (auto& neuron : layer) {
            in >> neuron.bias;
            for (double& w : neuron.weights) in >> w;
        }
    }
    return true;
}
