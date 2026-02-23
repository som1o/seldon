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
    m_mBiases.assign(m_size, 0.0);
    m_vBiases.assign(m_size, 0.0);
    m_dropMask.assign(m_size, static_cast<uint8_t>(0));

    if (m_prevSize == 0) return;

    const size_t weightCount = m_size * m_prevSize;
    double stddev = 0.1;
    if (m_activation == NeuralActivation::RELU) {
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
        case NeuralActivation::TANH: return std::tanh(x);
        case NeuralActivation::SIGMOID: {
            double clipped = std::clamp(x, -60.0, 60.0);
            return 1.0 / (1.0 + std::exp(-clipped));
        }
        case NeuralActivation::LINEAR: return x;
        default: return x;
    }
}

double DenseLayer::activateDerivative(double out, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::RELU: return out > 0.0 ? 1.0 : 0.0;
        case NeuralActivation::TANH: return 1.0 - out * out;
        case NeuralActivation::SIGMOID: return out * (1.0 - out);
        case NeuralActivation::LINEAR: return 1.0;
        default: return 1.0;
    }
}

void DenseLayer::forward(const DenseLayer& prev,
                         bool isTraining,
                         double dropoutRate,
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

        const double out = activate(sum, m_activation);
        if (isTraining && dropoutRate > 0.0) {
            uint64_t key = static_cast<uint64_t>(seedState);
            key ^= (static_cast<uint64_t>(n) << 24);
            key ^= forwardCounter;
            const double u = (splitmix64(key) & 0xFFFFFF) / static_cast<double>(0x1000000);
            if (u < dropoutRate) {
                m_outputs[n] = 0.0;
                m_dropMask[n] = static_cast<uint8_t>(1);
            } else {
                m_outputs[n] = out / (1.0 - dropoutRate);
                m_dropMask[n] = static_cast<uint8_t>(0);
            }
        } else {
            m_outputs[n] = out;
            m_dropMask[n] = static_cast<uint8_t>(0);
        }
    }
}

void DenseLayer::computeOutputGradients(const std::vector<double>& targetValues, bool crossEntropyWithSigmoid) {
    for (size_t n = 0; n < m_size; ++n) {
        if (crossEntropyWithSigmoid) {
            m_gradients[n] = m_outputs[n] - targetValues[n];
        } else {
            const double delta = m_outputs[n] - targetValues[n];
            m_gradients[n] = delta * activateDerivative(m_outputs[n], m_activation);
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
        m_gradients[n] *= activateDerivative(m_outputs[n], m_activation);
    }
}

void DenseLayer::updateParameters(const DenseLayer& prev,
                                  double learningRate,
                                  double l2Lambda,
                                  NeuralOptimizer optimizer,
                                  size_t tStep) {
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
            gradW += l2Lambda * m_weights[weightOffset + pn];

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
