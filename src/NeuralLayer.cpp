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
    const double fanIn = static_cast<double>(std::max<size_t>(1, m_prevSize));
    const double fanOut = static_cast<double>(std::max<size_t>(1, m_size));
    double stddev = 0.1;
    if (m_activation == NeuralActivation::RELU || m_activation == NeuralActivation::GELU) {
        stddev = std::sqrt(2.0 / fanIn);
    } else {
        stddev = std::sqrt(2.0 / (fanIn + fanOut));
    }

    std::normal_distribution<double> weightDis(0.0, stddev);
    const double clip = 2.5 * stddev;
    m_weights.resize(weightCount, 0.0);
    for (NeuralScalar& w : m_weights) {
        const double sampled = std::clamp(weightDis(rng), -clip, clip);
        w = static_cast<NeuralScalar>(sampled);
    }

    m_mWeights.assign(weightCount, 0.0);
    m_vWeights.assign(weightCount, 0.0);
    std::fill(m_biases.begin(), m_biases.end(), static_cast<NeuralScalar>(0.0));
}

NeuralScalar DenseLayer::activate(NeuralScalar x, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::RELU: return std::max(static_cast<NeuralScalar>(0.0), x);
        case NeuralActivation::GELU: {
            constexpr NeuralScalar kInvSqrtPi = static_cast<NeuralScalar>(0.7978845608028654); // sqrt(2/pi)
            const NeuralScalar c = kInvSqrtPi;
            const NeuralScalar x3 = x * x * x;
            const NeuralScalar inner = c * (x + static_cast<NeuralScalar>(0.044715) * x3);
            return static_cast<NeuralScalar>(0.5) * x * (static_cast<NeuralScalar>(1.0) + static_cast<NeuralScalar>(std::tanh(inner)));
        }
        case NeuralActivation::TANH: return static_cast<NeuralScalar>(std::tanh(x));
        case NeuralActivation::SIGMOID: {
            NeuralScalar clipped = std::clamp(x, static_cast<NeuralScalar>(-60.0), static_cast<NeuralScalar>(60.0));
            return static_cast<NeuralScalar>(1.0) / (static_cast<NeuralScalar>(1.0) + static_cast<NeuralScalar>(std::exp(-clipped)));
        }
        case NeuralActivation::LINEAR: return x;
        default: return x;
    }
}

NeuralScalar DenseLayer::activateDerivativeFromInput(NeuralScalar x, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::RELU: return x > 0.0 ? 1.0 : 0.0;
        case NeuralActivation::GELU: {
            constexpr NeuralScalar kInvSqrtPi = static_cast<NeuralScalar>(0.7978845608028654); // sqrt(2/pi)
            const NeuralScalar c = kInvSqrtPi;
            const NeuralScalar x2 = x * x;
            const NeuralScalar x3 = x2 * x;
            const NeuralScalar inner = c * (x + static_cast<NeuralScalar>(0.044715) * x3);
            const NeuralScalar t = static_cast<NeuralScalar>(std::tanh(inner));
            const NeuralScalar sech2 = static_cast<NeuralScalar>(1.0) - t * t;
            const NeuralScalar innerPrime = c * (static_cast<NeuralScalar>(1.0) + static_cast<NeuralScalar>(0.134145) * x2);
            return static_cast<NeuralScalar>(0.5) * (static_cast<NeuralScalar>(1.0) + t)
                + static_cast<NeuralScalar>(0.5) * x * sech2 * innerPrime;
        }
        case NeuralActivation::TANH: {
            const NeuralScalar out = static_cast<NeuralScalar>(std::tanh(x));
            return static_cast<NeuralScalar>(1.0) - out * out;
        }
        case NeuralActivation::SIGMOID: {
            const NeuralScalar clipped = std::clamp(x, static_cast<NeuralScalar>(-60.0), static_cast<NeuralScalar>(60.0));
            const NeuralScalar out = static_cast<NeuralScalar>(1.0) / (static_cast<NeuralScalar>(1.0) + static_cast<NeuralScalar>(std::exp(-clipped)));
            return out * (static_cast<NeuralScalar>(1.0) - out);
        }
        case NeuralActivation::LINEAR: return static_cast<NeuralScalar>(1.0);
        default: return static_cast<NeuralScalar>(1.0);
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
        NeuralScalar sum = m_biases[n];
        const size_t weightOffset = n * m_prevSize;
        constexpr size_t kTile = 32;
        for (size_t t0 = 0; t0 < m_prevSize; t0 += kTile) {
            const size_t tMax = std::min(m_prevSize, t0 + kTile);
            #ifdef USE_OPENMP
            #pragma omp simd reduction(+:sum)
            #endif
            for (size_t pn = t0; pn < tMax; ++pn) {
                #if defined(__GNUC__) || defined(__clang__)
                if (pn + 16 < m_prevSize) {
                    __builtin_prefetch(&prev.outputs()[pn + 16], 0, 1);
                    __builtin_prefetch(&m_weights[weightOffset + pn + 16], 0, 1);
                }
                #endif
                sum += prev.outputs()[pn] * m_weights[weightOffset + pn];
            }
        }

        NeuralScalar activationInput = sum;
        if (useBatchNorm) {
            const NeuralScalar clippedMomentum = static_cast<NeuralScalar>(std::clamp(batchNormMomentum, 0.0, 0.9999));
            const NeuralScalar eps = static_cast<NeuralScalar>(std::max(batchNormEpsilon, 1e-12));
            if (isTraining) {
                const NeuralScalar oldRunningMean = m_bnRunningMean[n];
                m_bnRunningMean[n] = clippedMomentum * oldRunningMean + (static_cast<NeuralScalar>(1.0) - clippedMomentum) * activationInput;
                const NeuralScalar centered = activationInput - oldRunningMean;
                m_bnRunningVar[n] = clippedMomentum * m_bnRunningVar[n] + (static_cast<NeuralScalar>(1.0) - clippedMomentum) * (centered * centered);
            }

            const NeuralScalar var = std::max(m_bnRunningVar[n], static_cast<NeuralScalar>(0.0));
            const NeuralScalar invStd = static_cast<NeuralScalar>(1.0 / std::sqrt(static_cast<double>(var + eps)));
            const NeuralScalar normalized = (activationInput - m_bnRunningMean[n]) * invStd;
            activationInput = m_bnGamma[n] * normalized + m_bnBeta[n];
            m_bnBackpropScale[n] = m_bnGamma[n] * invStd;
        } else {
            m_bnBackpropScale[n] = static_cast<NeuralScalar>(1.0);
        }

        m_activationInputs[n] = activationInput;
        m_outputs[n] = activate(activationInput, m_activation);
    }

    if (useLayerNorm && m_size > 0) {
        const NeuralScalar eps = static_cast<NeuralScalar>(std::max(layerNormEpsilon, 1e-12));
        NeuralScalar mean = 0.0;
        for (NeuralScalar out : m_outputs) mean += out;
        mean /= static_cast<NeuralScalar>(m_size);

        NeuralScalar var = 0.0;
        for (NeuralScalar out : m_outputs) {
            const NeuralScalar d = out - mean;
            var += d * d;
        }
        var /= static_cast<NeuralScalar>(m_size);

        const NeuralScalar invStd = static_cast<NeuralScalar>(1.0 / std::sqrt(static_cast<double>(var + eps)));
        for (size_t n = 0; n < m_size; ++n) {
            m_outputs[n] = (m_outputs[n] - mean) * invStd;
            m_lnBackpropScale[n] = invStd;
        }
    } else {
        std::fill(m_lnBackpropScale.begin(), m_lnBackpropScale.end(), static_cast<NeuralScalar>(1.0));
    }

    if (isTraining && dropoutRate > 0.0) {
        for (size_t n = 0; n < m_size; ++n) {
            uint64_t key = static_cast<uint64_t>(seedState);
            key ^= (static_cast<uint64_t>(n) << 24);
            key ^= forwardCounter;
            const double u = (splitmix64(key) & 0xFFFFFF) / static_cast<double>(0x1000000);
            if (u < dropoutRate) {
                m_outputs[n] = static_cast<NeuralScalar>(0.0);
                m_dropMask[n] = static_cast<uint8_t>(1);
                m_dropoutScale[n] = static_cast<NeuralScalar>(0.0);
            } else {
                m_outputs[n] /= static_cast<NeuralScalar>(1.0 - dropoutRate);
                m_dropMask[n] = static_cast<uint8_t>(0);
                m_dropoutScale[n] = static_cast<NeuralScalar>(1.0 / (1.0 - dropoutRate));
            }
        }
    } else {
        std::fill(m_dropMask.begin(), m_dropMask.end(), static_cast<uint8_t>(0));
        std::fill(m_dropoutScale.begin(), m_dropoutScale.end(), static_cast<NeuralScalar>(1.0));
    }
}

void DenseLayer::computeOutputGradients(const std::vector<double>& targetValues, bool crossEntropyWithSigmoid) {
    for (size_t n = 0; n < m_size; ++n) {
        if (crossEntropyWithSigmoid) {
            m_gradients[n] = m_outputs[n] - targetValues[n];
        } else {
            const NeuralScalar delta = m_outputs[n] - static_cast<NeuralScalar>(targetValues[n]);
            m_gradients[n] = delta * activateDerivativeFromInput(m_activationInputs[n], m_activation);
        }
    }
}

void DenseLayer::accumulateHiddenGradients(const DenseLayer& next) {
    std::fill(m_gradients.begin(), m_gradients.end(), static_cast<NeuralScalar>(0.0));

    for (size_t nn = 0; nn < next.size(); ++nn) {
        const NeuralScalar nextGrad = next.gradients()[nn];
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
            m_gradients[n] = static_cast<NeuralScalar>(0.0);
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

    const NeuralScalar beta1 = static_cast<NeuralScalar>(0.9);
    const NeuralScalar beta2 = static_cast<NeuralScalar>(0.999);
    const NeuralScalar epsilon = static_cast<NeuralScalar>(1e-8);
    const NeuralScalar beta1_t = static_cast<NeuralScalar>(1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(tStep)));
    const NeuralScalar beta2_t = static_cast<NeuralScalar>(1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(tStep)));

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < m_size; ++n) {
        const NeuralScalar gradBias = m_gradients[n];
        if (optimizer == NeuralOptimizer::ADAM) {
            m_mBiases[n] = beta1 * m_mBiases[n] + (1.0 - beta1) * gradBias;
            m_vBiases[n] = beta2 * m_vBiases[n] + (1.0 - beta2) * gradBias * gradBias;
            const NeuralScalar mHat = m_mBiases[n] / beta1_t;
            const NeuralScalar vHat = m_vBiases[n] / beta2_t;
            m_biases[n] -= static_cast<NeuralScalar>(learningRate) * (mHat / (static_cast<NeuralScalar>(std::sqrt(static_cast<double>(vHat))) + epsilon));
        } else {
            m_biases[n] -= static_cast<NeuralScalar>(learningRate) * gradBias;
        }

        const size_t weightOffset = n * m_prevSize;
        #ifdef USE_OPENMP
        #pragma omp simd
        #endif
        for (size_t pn = 0; pn < m_prevSize; ++pn) {
            NeuralScalar gradW = m_gradients[n] * prev.outputs()[pn];
            const NeuralScalar l2Scale = (inputL2Scales != nullptr && pn < inputL2Scales->size())
                ? std::max(0.0, (*inputL2Scales)[pn])
                : 1.0;
            gradW += static_cast<NeuralScalar>(l2Lambda * static_cast<double>(l2Scale)) * m_weights[weightOffset + pn];

            if (optimizer == NeuralOptimizer::ADAM) {
                m_mWeights[weightOffset + pn] = beta1 * m_mWeights[weightOffset + pn] + (1.0 - beta1) * gradW;
                m_vWeights[weightOffset + pn] = beta2 * m_vWeights[weightOffset + pn] + (1.0 - beta2) * gradW * gradW;
                const NeuralScalar mHat = m_mWeights[weightOffset + pn] / beta1_t;
                const NeuralScalar vHat = m_vWeights[weightOffset + pn] / beta2_t;
                m_weights[weightOffset + pn] -= static_cast<NeuralScalar>(learningRate) * (mHat / (static_cast<NeuralScalar>(std::sqrt(static_cast<double>(vHat))) + epsilon));
            } else {
                m_weights[weightOffset + pn] -= static_cast<NeuralScalar>(learningRate) * gradW;
            }
        }
    }
}

void DenseLayer::updateParametersAccumulated(double learningRate,
                                             double l2Lambda,
                                             NeuralOptimizer optimizer,
                                             size_t tStep,
                                             const std::vector<NeuralScalar>& gradBiasAccum,
                                             const std::vector<NeuralScalar>& gradWeightAccum,
                                             const std::vector<double>* inputL2Scales) {
    if (m_prevSize == 0) return;
    if (gradBiasAccum.size() != m_size || gradWeightAccum.size() != m_weights.size()) return;

    const NeuralScalar beta1 = static_cast<NeuralScalar>(0.9);
    const NeuralScalar beta2 = static_cast<NeuralScalar>(0.999);
    const NeuralScalar epsilon = static_cast<NeuralScalar>(1e-8);
    const NeuralScalar beta1_t = static_cast<NeuralScalar>(1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(tStep)));
    const NeuralScalar beta2_t = static_cast<NeuralScalar>(1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(tStep)));

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t n = 0; n < m_size; ++n) {
        const NeuralScalar gradBias = gradBiasAccum[n];
        if (optimizer == NeuralOptimizer::ADAM) {
            m_mBiases[n] = beta1 * m_mBiases[n] + (1.0 - beta1) * gradBias;
            m_vBiases[n] = beta2 * m_vBiases[n] + (1.0 - beta2) * gradBias * gradBias;
            const NeuralScalar mHat = m_mBiases[n] / std::max(static_cast<NeuralScalar>(1e-12), beta1_t);
            const NeuralScalar vHat = m_vBiases[n] / std::max(static_cast<NeuralScalar>(1e-12), beta2_t);
            m_biases[n] -= static_cast<NeuralScalar>(learningRate) * (mHat / (static_cast<NeuralScalar>(std::sqrt(static_cast<double>(vHat))) + epsilon));
        } else {
            m_biases[n] -= static_cast<NeuralScalar>(learningRate) * gradBias;
        }

        const size_t weightOffset = n * m_prevSize;
        #ifdef USE_OPENMP
        #pragma omp simd
        #endif
        for (size_t pn = 0; pn < m_prevSize; ++pn) {
            NeuralScalar gradW = gradWeightAccum[weightOffset + pn];
            const NeuralScalar l2Scale = (inputL2Scales != nullptr && pn < inputL2Scales->size())
                ? std::max(0.0, (*inputL2Scales)[pn])
                : 1.0;
            gradW += static_cast<NeuralScalar>(l2Lambda * static_cast<double>(l2Scale)) * m_weights[weightOffset + pn];

            if (optimizer == NeuralOptimizer::ADAM) {
                m_mWeights[weightOffset + pn] = beta1 * m_mWeights[weightOffset + pn] + (1.0 - beta1) * gradW;
                m_vWeights[weightOffset + pn] = beta2 * m_vWeights[weightOffset + pn] + (1.0 - beta2) * gradW * gradW;
                const NeuralScalar mHat = m_mWeights[weightOffset + pn] / std::max(static_cast<NeuralScalar>(1e-12), beta1_t);
                const NeuralScalar vHat = m_vWeights[weightOffset + pn] / std::max(static_cast<NeuralScalar>(1e-12), beta2_t);
                m_weights[weightOffset + pn] -= static_cast<NeuralScalar>(learningRate) * (mHat / (static_cast<NeuralScalar>(std::sqrt(static_cast<double>(vHat))) + epsilon));
            } else {
                m_weights[weightOffset + pn] -= static_cast<NeuralScalar>(learningRate) * gradW;
            }
        }
    }
}
