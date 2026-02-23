#include "NeuralLayer.h"

#include <algorithm>
#include <cmath>

namespace {
uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
}

DenseLayer::DenseLayer(size_t size, size_t prevSize, NeuralActivation activation, std::mt19937& rng)
    : m_size(size), m_prevSize(prevSize), m_activation(activation) {
    m_outputs.assign(m_size, 0.0);
    m_biases.assign(m_size, 0.0);
    m_gradients.assign(m_size, 0.0);
    m_activationInputs.assign(m_size, 0.0);
    m_mBiases.assign(m_size, 0.0);
    m_vBiases.assign(m_size, 0.0);
    m_dropMask.assign(m_size, static_cast<uint8_t>(0));
    m_dropoutScale.assign(m_size, 1.0);
    m_bnRunningMean.assign(m_size, 0.0);
    m_bnRunningVar.assign(m_size, 1.0);
    m_bnGamma.assign(m_size, 1.0);
    m_bnBeta.assign(m_size, 0.0);
    m_bnBackpropScale.assign(m_size, 1.0);
    m_lnBackpropScale.assign(m_size, 1.0);

    if (m_prevSize == 0) return;

    const size_t weightCount = m_size * m_prevSize;
    double stddev = 0.1;
    if (m_activation == NeuralActivation::RELU || m_activation == NeuralActivation::GELU) {
        stddev = std::sqrt(2.0 / static_cast<double>(m_prevSize));
    } else {
        stddev = std::sqrt(1.0 / static_cast<double>(m_prevSize));
    }

    std::normal_distribution<> weightDis(0.0, stddev);
    m_weights.resize(weightCount, 0.0);
    for (double& w : m_weights) {
        w = weightDis(rng);
    }

    m_mWeights.assign(weightCount, 0.0);
    m_vWeights.assign(weightCount, 0.0);
    std::fill(m_biases.begin(), m_biases.end(), 0.01);
}

double DenseLayer::activate(double x, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::RELU: return std::max(0.0, x);
        case NeuralActivation::GELU: {
            constexpr double kInvSqrtPi = 0.7978845608028654; // sqrt(2/pi)
            const double c = kInvSqrtPi;
            const double x3 = x * x * x;
            const double inner = c * (x + 0.044715 * x3);
            return 0.5 * x * (1.0 + std::tanh(inner));
        }
        case NeuralActivation::TANH: return std::tanh(x);
        case NeuralActivation::SIGMOID: {
            double clipped = std::clamp(x, -60.0, 60.0);
            return 1.0 / (1.0 + std::exp(-clipped));
        }
        case NeuralActivation::LINEAR: return x;
        default: return x;
    }
}

double DenseLayer::activateDerivativeFromInput(double x, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::RELU: return x > 0.0 ? 1.0 : 0.0;
        case NeuralActivation::GELU: {
            constexpr double kInvSqrtPi = 0.7978845608028654; // sqrt(2/pi)
            const double c = kInvSqrtPi;
            const double x2 = x * x;
            const double x3 = x2 * x;
            const double inner = c * (x + 0.044715 * x3);
            const double t = std::tanh(inner);
            const double sech2 = 1.0 - t * t;
            const double innerPrime = c * (1.0 + 0.134145 * x2);
            return 0.5 * (1.0 + t) + 0.5 * x * sech2 * innerPrime;
        }
        case NeuralActivation::TANH: {
            const double out = std::tanh(x);
            return 1.0 - out * out;
        }
        case NeuralActivation::SIGMOID: {
            const double clipped = std::clamp(x, -60.0, 60.0);
            const double out = 1.0 / (1.0 + std::exp(-clipped));
            return out * (1.0 - out);
        }
        case NeuralActivation::LINEAR: return 1.0;
        default: return 1.0;
    }
}

void DenseLayer::forward(const DenseLayer& prev,
                         bool isTraining,
                         double dropoutRate,
                         bool useBatchNorm,
                         double batchNormMomentum,
                         double batchNormEpsilon,
                         bool useLayerNorm,
                         double layerNormEpsilon,
                         uint32_t seedState,
                         uint64_t forwardCounter) {
    if (m_prevSize == 0) return;

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < m_size; ++n) {
        double sum = m_biases[n];
        const size_t weightOffset = n * m_prevSize;
        #ifdef USE_OPENMP
        #pragma omp simd reduction(+:sum)
        #endif
        for (size_t pn = 0; pn < m_prevSize; ++pn) {
            sum += prev.outputs()[pn] * m_weights[weightOffset + pn];
        }

        double activationInput = sum;
        if (useBatchNorm) {
            const double clippedMomentum = std::clamp(batchNormMomentum, 0.0, 0.9999);
            const double eps = std::max(batchNormEpsilon, 1e-12);
            if (isTraining) {
                const double oldRunningMean = m_bnRunningMean[n];
                m_bnRunningMean[n] = clippedMomentum * oldRunningMean + (1.0 - clippedMomentum) * activationInput;
                const double centered = activationInput - oldRunningMean;
                m_bnRunningVar[n] = clippedMomentum * m_bnRunningVar[n] + (1.0 - clippedMomentum) * (centered * centered);
            }

            const double var = std::max(m_bnRunningVar[n], 0.0);
            const double invStd = 1.0 / std::sqrt(var + eps);
            const double normalized = (activationInput - m_bnRunningMean[n]) * invStd;
            activationInput = m_bnGamma[n] * normalized + m_bnBeta[n];
            m_bnBackpropScale[n] = m_bnGamma[n] * invStd;
        } else {
            m_bnBackpropScale[n] = 1.0;
        }

        m_activationInputs[n] = activationInput;
        m_outputs[n] = activate(activationInput, m_activation);
    }

    if (useLayerNorm && m_size > 0) {
        const double eps = std::max(layerNormEpsilon, 1e-12);
        double mean = 0.0;
        for (double out : m_outputs) mean += out;
        mean /= static_cast<double>(m_size);

        double var = 0.0;
        for (double out : m_outputs) {
            const double d = out - mean;
            var += d * d;
        }
        var /= static_cast<double>(m_size);

        const double invStd = 1.0 / std::sqrt(var + eps);
        for (size_t n = 0; n < m_size; ++n) {
            m_outputs[n] = (m_outputs[n] - mean) * invStd;
            m_lnBackpropScale[n] = invStd;
        }
    } else {
        std::fill(m_lnBackpropScale.begin(), m_lnBackpropScale.end(), 1.0);
    }

    if (isTraining && dropoutRate > 0.0) {
        for (size_t n = 0; n < m_size; ++n) {
            uint64_t key = static_cast<uint64_t>(seedState);
            key ^= (static_cast<uint64_t>(n) << 24);
            key ^= forwardCounter;
            const double u = (splitmix64(key) & 0xFFFFFF) / static_cast<double>(0x1000000);
            if (u < dropoutRate) {
                m_outputs[n] = 0.0;
                m_dropMask[n] = static_cast<uint8_t>(1);
                m_dropoutScale[n] = 0.0;
            } else {
                m_outputs[n] /= (1.0 - dropoutRate);
                m_dropMask[n] = static_cast<uint8_t>(0);
                m_dropoutScale[n] = 1.0 / (1.0 - dropoutRate);
            }
        }
    } else {
        std::fill(m_dropMask.begin(), m_dropMask.end(), static_cast<uint8_t>(0));
        std::fill(m_dropoutScale.begin(), m_dropoutScale.end(), 1.0);
    }
}

void DenseLayer::computeOutputGradients(const std::vector<double>& targetValues, bool crossEntropyWithSigmoid) {
    for (size_t n = 0; n < m_size; ++n) {
        if (crossEntropyWithSigmoid) {
            m_gradients[n] = m_outputs[n] - targetValues[n];
        } else {
            const double delta = m_outputs[n] - targetValues[n];
            m_gradients[n] = delta * activateDerivativeFromInput(m_activationInputs[n], m_activation);
        }
    }
}

void DenseLayer::accumulateHiddenGradients(const DenseLayer& next) {
    std::fill(m_gradients.begin(), m_gradients.end(), 0.0);

    for (size_t nn = 0; nn < next.size(); ++nn) {
        const double nextGrad = next.gradients()[nn];
        const size_t weightOffset = nn * m_size;
        #ifdef USE_OPENMP
        #pragma omp simd
        #endif
        for (size_t n = 0; n < m_size; ++n) {
            m_gradients[n] += next.weights()[weightOffset + n] * nextGrad;
        }
    }
}

void DenseLayer::applyActivationDerivativeAndDropout() {
    for (size_t n = 0; n < m_size; ++n) {
        if (m_dropMask[n]) {
            m_gradients[n] = 0.0;
            continue;
        }
        m_gradients[n] *= m_dropoutScale[n];
        m_gradients[n] *= m_lnBackpropScale[n];
        m_gradients[n] *= activateDerivativeFromInput(m_activationInputs[n], m_activation);
        m_gradients[n] *= m_bnBackpropScale[n];
    }
}

void DenseLayer::updateParameters(const DenseLayer& prev,
                                  double learningRate,
                                  double l2Lambda,
                                  NeuralOptimizer optimizer,
                                  size_t tStep,
                                  const std::vector<double>* inputL2Scales) {
    if (m_prevSize == 0) return;

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    const double beta1_t = 1.0 - std::pow(beta1, static_cast<double>(tStep));
    const double beta2_t = 1.0 - std::pow(beta2, static_cast<double>(tStep));

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < m_size; ++n) {
        const double gradBias = m_gradients[n];
        if (optimizer == NeuralOptimizer::ADAM) {
            m_mBiases[n] = beta1 * m_mBiases[n] + (1.0 - beta1) * gradBias;
            m_vBiases[n] = beta2 * m_vBiases[n] + (1.0 - beta2) * gradBias * gradBias;
            const double mHat = m_mBiases[n] / beta1_t;
            const double vHat = m_vBiases[n] / beta2_t;
            m_biases[n] -= learningRate * (mHat / (std::sqrt(vHat) + epsilon));
        } else {
            m_biases[n] -= learningRate * gradBias;
        }

        const size_t weightOffset = n * m_prevSize;
        #ifdef USE_OPENMP
        #pragma omp simd
        #endif
        for (size_t pn = 0; pn < m_prevSize; ++pn) {
            double gradW = m_gradients[n] * prev.outputs()[pn];
            const double l2Scale = (inputL2Scales != nullptr && pn < inputL2Scales->size())
                ? std::max(0.0, (*inputL2Scales)[pn])
                : 1.0;
            gradW += (l2Lambda * l2Scale) * m_weights[weightOffset + pn];

            if (optimizer == NeuralOptimizer::ADAM) {
                m_mWeights[weightOffset + pn] = beta1 * m_mWeights[weightOffset + pn] + (1.0 - beta1) * gradW;
                m_vWeights[weightOffset + pn] = beta2 * m_vWeights[weightOffset + pn] + (1.0 - beta2) * gradW * gradW;
                const double mHat = m_mWeights[weightOffset + pn] / beta1_t;
                const double vHat = m_vWeights[weightOffset + pn] / beta2_t;
                m_weights[weightOffset + pn] -= learningRate * (mHat / (std::sqrt(vHat) + epsilon));
            } else {
                m_weights[weightOffset + pn] -= learningRate * gradW;
            }
        }
    }
}
