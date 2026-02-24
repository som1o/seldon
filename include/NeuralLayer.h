#pragma once

#include <cstdint>
#include <random>
#include <vector>

enum class NeuralActivation { SIGMOID, RELU, TANH, LINEAR, GELU };
enum class NeuralOptimizer { SGD, ADAM, LOOKAHEAD };

class DenseLayer {
public:
    DenseLayer() = default;
    DenseLayer(size_t size, size_t prevSize, NeuralActivation activation, std::mt19937& rng);

    size_t size() const noexcept { return m_size; }
    size_t prevSize() const noexcept { return m_prevSize; }

    NeuralActivation activation() const noexcept { return m_activation; }
    void setActivation(NeuralActivation activation) noexcept { m_activation = activation; }

    std::vector<double>& outputs() noexcept { return m_outputs; }
    const std::vector<double>& outputs() const noexcept { return m_outputs; }

    std::vector<double>& gradients() noexcept { return m_gradients; }
    const std::vector<double>& gradients() const noexcept { return m_gradients; }

    std::vector<double>& biases() noexcept { return m_biases; }
    const std::vector<double>& biases() const noexcept { return m_biases; }

    std::vector<double>& weights() noexcept { return m_weights; }
    const std::vector<double>& weights() const noexcept { return m_weights; }

    std::vector<uint8_t>& dropMask() noexcept { return m_dropMask; }
    const std::vector<uint8_t>& dropMask() const noexcept { return m_dropMask; }

    void forward(const DenseLayer& prev,
                 bool isTraining,
                 double dropoutRate,
                 bool useBatchNorm,
                 double batchNormMomentum,
                 double batchNormEpsilon,
                 bool useLayerNorm,
                 double layerNormEpsilon,
                 uint32_t seedState,
                 uint64_t forwardCounter);

    void computeOutputGradients(const std::vector<double>& targetValues, bool crossEntropyWithSigmoid);
    void accumulateHiddenGradients(const DenseLayer& next);
    void applyActivationDerivativeAndDropout();

    void updateParameters(const DenseLayer& prev,
                          double learningRate,
                          double l2Lambda,
                          NeuralOptimizer optimizer,
                          size_t tStep,
                          const std::vector<double>* inputL2Scales = nullptr);

private:
    static double activate(double x, NeuralActivation activation);
    static double activateDerivativeFromInput(double x, NeuralActivation activation);

    size_t m_size = 0;
    size_t m_prevSize = 0;
    std::vector<double> m_outputs;
    std::vector<double> m_biases;
    std::vector<double> m_weights;
    std::vector<double> m_gradients;
    std::vector<double> m_activationInputs;

    std::vector<double> m_mWeights;
    std::vector<double> m_vWeights;
    std::vector<double> m_mBiases;
    std::vector<double> m_vBiases;

    std::vector<uint8_t> m_dropMask;
    std::vector<double> m_dropoutScale;

    std::vector<double> m_bnRunningMean;
    std::vector<double> m_bnRunningVar;
    std::vector<double> m_bnGamma;
    std::vector<double> m_bnBeta;
    std::vector<double> m_bnBackpropScale;
    std::vector<double> m_lnBackpropScale;

    NeuralActivation m_activation = NeuralActivation::RELU;
};
