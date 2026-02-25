#pragma once

#include <cstdint>
#include <random>
#include <vector>

enum class NeuralActivation { SIGMOID, RELU, TANH, LINEAR, GELU };
enum class NeuralOptimizer { SGD, ADAM, LOOKAHEAD };

#ifdef SELDON_NEURAL_FLOAT32
using NeuralScalar = float;
#else
using NeuralScalar = double;
#endif

class DenseLayer {
public:
    DenseLayer() = default;
    DenseLayer(size_t size, size_t prevSize, NeuralActivation activation, std::mt19937& rng);

    size_t size() const noexcept { return m_size; }
    size_t prevSize() const noexcept { return m_prevSize; }

    NeuralActivation activation() const noexcept { return m_activation; }
    void setActivation(NeuralActivation activation) noexcept { m_activation = activation; }

    std::vector<NeuralScalar>& outputs() noexcept { return m_outputs; }
    const std::vector<NeuralScalar>& outputs() const noexcept { return m_outputs; }

    std::vector<NeuralScalar>& gradients() noexcept { return m_gradients; }
    const std::vector<NeuralScalar>& gradients() const noexcept { return m_gradients; }

    std::vector<NeuralScalar>& biases() noexcept { return m_biases; }
    const std::vector<NeuralScalar>& biases() const noexcept { return m_biases; }

    std::vector<NeuralScalar>& weights() noexcept { return m_weights; }
    const std::vector<NeuralScalar>& weights() const noexcept { return m_weights; }

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
    void updateParametersAccumulated(double learningRate,
                                     double l2Lambda,
                                     NeuralOptimizer optimizer,
                                     size_t tStep,
                                     const std::vector<NeuralScalar>& gradBiasAccum,
                                     const std::vector<NeuralScalar>& gradWeightAccum,
                                     const std::vector<double>* inputL2Scales = nullptr);

private:
    static NeuralScalar activate(NeuralScalar x, NeuralActivation activation);
    static NeuralScalar activateDerivativeFromInput(NeuralScalar x, NeuralActivation activation);

    size_t m_size = 0;
    size_t m_prevSize = 0;
    std::vector<NeuralScalar> m_outputs;
    std::vector<NeuralScalar> m_biases;
    std::vector<NeuralScalar> m_weights;
    std::vector<NeuralScalar> m_gradients;
    std::vector<NeuralScalar> m_activationInputs;

    std::vector<NeuralScalar> m_mWeights;
    std::vector<NeuralScalar> m_vWeights;
    std::vector<NeuralScalar> m_mBiases;
    std::vector<NeuralScalar> m_vBiases;

    std::vector<uint8_t> m_dropMask;
    std::vector<NeuralScalar> m_dropoutScale;

    std::vector<NeuralScalar> m_bnRunningMean;
    std::vector<NeuralScalar> m_bnRunningVar;
    std::vector<NeuralScalar> m_bnGamma;
    std::vector<NeuralScalar> m_bnBeta;
    std::vector<NeuralScalar> m_bnBackpropScale;
    std::vector<NeuralScalar> m_lnBackpropScale;

    NeuralActivation m_activation = NeuralActivation::RELU;
};
