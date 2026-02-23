#pragma once

#include <cstdint>
#include <random>
#include <vector>

enum class NeuralActivation { SIGMOID, RELU, TANH, LINEAR };
enum class NeuralOptimizer { SGD, ADAM };

class DenseLayer {
public:
    DenseLayer() = default;
    DenseLayer(size_t size, size_t prevSize, NeuralActivation activation, std::mt19937& rng);

    size_t size() const { return m_size; }
    size_t prevSize() const { return m_prevSize; }

    NeuralActivation activation() const { return m_activation; }
    void setActivation(NeuralActivation activation) { m_activation = activation; }

    std::vector<double>& outputs() { return m_outputs; }
    const std::vector<double>& outputs() const { return m_outputs; }

    std::vector<double>& gradients() { return m_gradients; }
    const std::vector<double>& gradients() const { return m_gradients; }

    std::vector<double>& biases() { return m_biases; }
    const std::vector<double>& biases() const { return m_biases; }

    std::vector<double>& weights() { return m_weights; }
    const std::vector<double>& weights() const { return m_weights; }

    std::vector<uint8_t>& dropMask() { return m_dropMask; }
    const std::vector<uint8_t>& dropMask() const { return m_dropMask; }

    void forward(const DenseLayer& prev,
                 bool isTraining,
                 double dropoutRate,
                 uint32_t seedState,
                 uint64_t forwardCounter);

    void computeOutputGradients(const std::vector<double>& targetValues, bool crossEntropyWithSigmoid);
    void accumulateHiddenGradients(const DenseLayer& next);
    void applyActivationDerivativeAndDropout();

    void updateParameters(const DenseLayer& prev,
                          double learningRate,
                          double l2Lambda,
                          NeuralOptimizer optimizer,
                          size_t tStep);

private:
    static double activate(double x, NeuralActivation activation);
    static double activateDerivative(double out, NeuralActivation activation);

    size_t m_size = 0;
    size_t m_prevSize = 0;
    std::vector<double> m_outputs;
    std::vector<double> m_biases;
    std::vector<double> m_weights;
    std::vector<double> m_gradients;

    std::vector<double> m_mWeights;
    std::vector<double> m_vWeights;
    std::vector<double> m_mBiases;
    std::vector<double> m_vBiases;

    std::vector<uint8_t> m_dropMask;
    NeuralActivation m_activation = NeuralActivation::RELU;
};
