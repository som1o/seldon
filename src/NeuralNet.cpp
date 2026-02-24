#include "NeuralNet.h"

#include "SeldonExceptions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
constexpr uint32_t kModelFormatVersion = 3;
constexpr uint64_t kChecksumOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kChecksumPrime = 1099511628211ULL;
constexpr size_t kHardMaxTopologyNodes = 65536;
constexpr size_t kHardMaxTrainableParams = 100000000;
constexpr double kNumericEps = 1e-12;

bool isLittleEndian() {
    uint16_t number = 0x1;
    const auto* bytes = reinterpret_cast<const char*>(&number);
    return bytes[0] == 1;
}

template <typename T>
void swapEndian(T& val) {
    auto* first = reinterpret_cast<unsigned char*>(&val);
    std::reverse(first, first + sizeof(T));
}

template <typename T>
void updateChecksum(uint64_t& checksum, const T& value) {
    T copy = value;
    if (!isLittleEndian()) swapEndian(copy);
    const auto* bytes = reinterpret_cast<const unsigned char*>(&copy);
    for (size_t i = 0; i < sizeof(T); ++i) {
        checksum ^= static_cast<uint64_t>(bytes[i]);
        checksum *= kChecksumPrime;
    }
}

template <typename T>
void writeLE(std::ostream& out, T value) {
    if (!isLittleEndian()) swapEndian(value);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw Seldon::IOException("Binary write failed");
}

template <typename T>
void readLE(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw Seldon::NeuralNetException("Binary read failed or model file is truncated");
    if (!isLittleEndian()) swapEndian(value);
}

double meanSquaredError(const std::vector<double>& prediction,
                        const std::vector<double>& target,
                        size_t outDim) {
    double err = 0.0;
    for (size_t j = 0; j < outDim; ++j) {
        const double p = (j < prediction.size()) ? prediction[j] : 0.0;
        const double t = (j < target.size()) ? target[j] : 0.0;
        const double d = p - t;
        err += d * d;
    }
    return err;
}

} // namespace

NeuralNet::NeuralNet(std::vector<size_t> topologyConfig)
    : topology(std::move(topologyConfig)) {
    if (topology.size() < 2) {
        throw Seldon::NeuralNetException("Topology must include at least input and output layers");
    }

    size_t totalNodes = 0;
    size_t totalParams = 0;
    for (size_t i = 0; i < topology.size(); ++i) {
        const size_t layerSize = topology[i];
        if (layerSize == 0) {
            throw Seldon::NeuralNetException("Layer size cannot be zero");
        }
        totalNodes += layerSize;
        if (i > 0) {
            totalParams += topology[i - 1] * topology[i];
            totalParams += topology[i];
        }
    }

    if (totalNodes > kHardMaxTopologyNodes) {
        throw Seldon::NeuralNetException("Topology node count exceeds hard safety limit");
    }
    if (totalParams > kHardMaxTrainableParams) {
        throw Seldon::NeuralNetException("Topology parameter count exceeds hard safety limit");
    }

    rng.seed(seedState);
    m_layers.reserve(topology.size());
    for (size_t l = 0; l < topology.size(); ++l) {
        const Activation act = (l + 1 == topology.size()) ? Activation::SIGMOID : Activation::GELU;
        const size_t prev = (l == 0) ? 0 : topology[l - 1];
        m_layers.emplace_back(topology[l], prev, act, rng);
    }
}

void NeuralNet::setSeed(uint32_t seed) {
    seedState = seed;
    rng.seed(seedState);
}

void NeuralNet::setInputL2Scales(const std::vector<double>& scales) {
    if (m_layers.size() > 1 && !scales.empty() && scales.size() != m_layers[1].prevSize()) {
        throw Seldon::NeuralNetException("Input L2 scales must match input layer width");
    }
    m_inputL2Scales = scales;
}

void NeuralNet::updateEmaWeights(double decay) {
    const double d = std::clamp(decay, 0.0, 0.999999);
    if (!m_emaInitialized) {
        m_emaWeights.clear();
        m_emaBiases.clear();
        m_emaWeights.reserve(m_layers.size());
        m_emaBiases.reserve(m_layers.size());
        for (const auto& layer : m_layers) {
            m_emaWeights.push_back(layer.weights());
            m_emaBiases.push_back(layer.biases());
        }
        m_emaInitialized = true;
        return;
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        auto& ew = m_emaWeights[l];
        auto& eb = m_emaBiases[l];
        const auto& w = m_layers[l].weights();
        const auto& b = m_layers[l].biases();
        for (size_t i = 0; i < ew.size() && i < w.size(); ++i) {
            ew[i] = d * ew[i] + (1.0 - d) * w[i];
        }
        for (size_t i = 0; i < eb.size() && i < b.size(); ++i) {
            eb[i] = d * eb[i] + (1.0 - d) * b[i];
        }
    }
}

void NeuralNet::applyEmaWeights() {
    if (!m_emaInitialized) return;
    for (size_t l = 1; l < m_layers.size(); ++l) {
        if (l < m_emaWeights.size() && m_emaWeights[l].size() == m_layers[l].weights().size()) {
            m_layers[l].weights() = m_emaWeights[l];
        }
        if (l < m_emaBiases.size() && m_emaBiases[l].size() == m_layers[l].biases().size()) {
            m_layers[l].biases() = m_emaBiases[l];
        }
    }
}

void NeuralNet::feedForward(const std::vector<double>& inputValues, bool isTraining, double dropoutRate) {
    if (m_layers.empty()) throw Seldon::NeuralNetException("Network has no layers");

    auto& inputLayer = m_layers.front();
    const size_t n = std::min(inputValues.size(), inputLayer.size());
    for (size_t i = 0; i < n; ++i) inputLayer.outputs()[i] = inputValues[i];
    for (size_t i = n; i < inputLayer.size(); ++i) inputLayer.outputs()[i] = 0.0;

    uint64_t runCounter = 0;
    if (isTraining) {
        runCounter = ++forwardCounter;
    }
    for (size_t l = 1; l < m_layers.size(); ++l) {
        const bool hidden = l + 1 < m_layers.size();
        m_layers[l].forward(
            m_layers[l - 1],
            isTraining && hidden,
            dropoutRate,
            m_useBatchNorm && hidden,
            m_batchNormMomentum,
            m_batchNormEpsilon,
            m_useLayerNorm && hidden,
            m_layerNormEpsilon,
            seedState,
            runCounter + l * 1315423911ULL);
    }
}

void NeuralNet::computeGradients(const std::vector<double>& targetValues, LossFunction loss) {
    if (m_layers.size() < 2 || targetValues.size() != m_layers.back().size()) {
        throw Seldon::NeuralNetException("Target dimensions do not match output layer");
    }

    auto& outputLayer = m_layers.back();
    if (loss == LossFunction::CROSS_ENTROPY && outputLayer.activation() != Activation::SIGMOID) {
        throw Seldon::NeuralNetException("Cross-Entropy loss currently requires SIGMOID output activation");
    }

    outputLayer.computeOutputGradients(targetValues, loss == LossFunction::CROSS_ENTROPY);

    for (size_t l = m_layers.size() - 1; l-- > 1;) {
        m_layers[l].accumulateHiddenGradients(m_layers[l + 1]);
        m_layers[l].applyActivationDerivativeAndDropout();
    }
}

void NeuralNet::applyOptimization(const Hyperparameters& hp, size_t t_step) {
    Optimizer fastOptimizer = hp.optimizer;
    if (hp.optimizer == Optimizer::LOOKAHEAD) {
        fastOptimizer = hp.lookaheadFastOptimizer;
        if (fastOptimizer == Optimizer::LOOKAHEAD) fastOptimizer = Optimizer::ADAM;
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        const std::vector<double>* l2Scales = nullptr;
        if (l == 1 && !m_inputL2Scales.empty()) l2Scales = &m_inputL2Scales;
        m_layers[l].updateParameters(m_layers[l - 1], hp.learningRate, hp.l2Lambda, fastOptimizer, t_step, l2Scales);
    }

    if (hp.optimizer != Optimizer::LOOKAHEAD) return;

    if (!m_lookaheadInitialized) {
        m_lookaheadSlowWeights.clear();
        m_lookaheadSlowBiases.clear();
        m_lookaheadSlowWeights.reserve(m_layers.size());
        m_lookaheadSlowBiases.reserve(m_layers.size());
        for (const auto& layer : m_layers) {
            m_lookaheadSlowWeights.push_back(layer.weights());
            m_lookaheadSlowBiases.push_back(layer.biases());
        }
        m_lookaheadInitialized = true;
    }

    const size_t k = std::max<size_t>(1, hp.lookaheadSyncPeriod);
    const double alpha = std::clamp(hp.lookaheadAlpha, 0.0, 1.0);
    if (t_step % k != 0) return;

    for (size_t l = 1; l < m_layers.size(); ++l) {
        auto& w = m_layers[l].weights();
        auto& b = m_layers[l].biases();
        auto& sw = m_lookaheadSlowWeights[l];
        auto& sb = m_lookaheadSlowBiases[l];

        for (size_t i = 0; i < w.size() && i < sw.size(); ++i) {
            sw[i] += alpha * (w[i] - sw[i]);
            w[i] = sw[i];
        }
        for (size_t i = 0; i < b.size() && i < sb.size(); ++i) {
            sb[i] += alpha * (b[i] - sb[i]);
            b[i] = sb[i];
        }
    }
}

void NeuralNet::backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step) {
    computeGradients(targetValues, hp.loss);

    if (hp.gradientClipNorm > 0.0) {
        double sqNorm = 0.0;
        for (size_t l = 1; l < m_layers.size(); ++l) {
            for (double g : m_layers[l].gradients()) sqNorm += g * g;
        }

        const double norm = std::sqrt(std::max(0.0, sqNorm));
        if (norm > hp.gradientClipNorm && norm > kNumericEps) {
            const double scale = hp.gradientClipNorm / norm;
            for (size_t l = 1; l < m_layers.size(); ++l) {
                for (double& g : m_layers[l].gradients()) g *= scale;
            }
        }
    }

    applyOptimization(hp, t_step);
}

double NeuralNet::runBatch(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y,
                           const std::vector<size_t>& indices,
                           size_t batchStart,
                           size_t batchEnd,
                           const Hyperparameters& hp,
                           size_t& t_step) {
    double batchLoss = 0.0;
    const size_t outDim = m_layers.back().size();
    const size_t accumSteps = std::max<size_t>(1, hp.gradientAccumulationSteps);

    Optimizer fastOptimizer = hp.optimizer;
    if (hp.optimizer == Optimizer::LOOKAHEAD) {
        fastOptimizer = hp.lookaheadFastOptimizer;
        if (fastOptimizer == Optimizer::LOOKAHEAD) fastOptimizer = Optimizer::ADAM;
    }

    std::vector<std::vector<double>> gradBAccum(m_layers.size());
    std::vector<std::vector<double>> gradWAccum(m_layers.size());
    std::vector<std::vector<double>> gradBWork(m_layers.size());
    std::vector<std::vector<double>> gradWWork(m_layers.size());
    for (size_t l = 1; l < m_layers.size(); ++l) {
        gradBAccum[l].assign(m_layers[l].size(), 0.0);
        gradWAccum[l].assign(m_layers[l].weights().size(), 0.0);
        gradBWork[l].assign(m_layers[l].size(), 0.0);
        gradWWork[l].assign(m_layers[l].weights().size(), 0.0);
    }

    std::vector<double> targetWork(outDim, 0.0);
    size_t microCount = 0;

    for (size_t b = batchStart; b < batchEnd; ++b) {
        const size_t idx = indices[b];
        feedForward(X[idx], true, hp.dropoutRate);

        const double smoothing = std::clamp(hp.labelSmoothing, 0.0, 0.25);
        if (hp.loss == LossFunction::CROSS_ENTROPY && smoothing > 0.0) {
            const size_t yDim = std::min(outDim, Y[idx].size());
            const double smoothTarget = 0.5 * smoothing;
            for (size_t j = 0; j < yDim; ++j) {
                const double y = std::clamp(Y[idx][j], 0.0, 1.0);
                targetWork[j] = y * (1.0 - smoothing) + smoothTarget;
            }
            for (size_t j = yDim; j < outDim; ++j) targetWork[j] = smoothTarget;
            computeGradients(targetWork, hp.loss);
        } else {
            computeGradients(Y[idx], hp.loss);
        }

        if (hp.gradientNoiseStd > 0.0) {
            const double decay = std::clamp(hp.gradientNoiseDecay, 0.0, 1.0);
            const double noiseStd = hp.gradientNoiseStd * std::pow(decay, static_cast<double>(t_step));
            if (noiseStd > 0.0) {
                std::normal_distribution<double> noiseDist(0.0, noiseStd);
                for (size_t l = 1; l < m_layers.size(); ++l) {
                    auto& g = m_layers[l].gradients();
                    for (double& v : g) v += noiseDist(rng);
                }
            }
        }

        for (size_t l = 1; l < m_layers.size(); ++l) {
            const auto& prevOut = m_layers[l - 1].outputs();
            const auto& g = m_layers[l].gradients();
            auto& gb = gradBAccum[l];
            auto& gw = gradWAccum[l];
            for (size_t n = 0; n < g.size(); ++n) {
                gb[n] += g[n];
                const size_t offset = n * prevOut.size();
                for (size_t pn = 0; pn < prevOut.size(); ++pn) {
                    gw[offset + pn] += g[n] * prevOut[pn];
                }
            }
        }

        ++microCount;

        const auto& out = m_layers.back().outputs();
        if (hp.loss == LossFunction::CROSS_ENTROPY) {
            for (size_t j = 0; j < out.size() && j < Y[idx].size(); ++j) {
                const double p = std::clamp(out[j], 1e-15, 1.0 - 1e-15);
                const double y = Y[idx][j];
                batchLoss -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
            }
        } else {
            batchLoss += meanSquaredError(out, Y[idx], std::min(out.size(), Y[idx].size()));
        }

        const bool shouldStep = (microCount >= accumSteps) || (b + 1 == batchEnd);
        if (!shouldStep) continue;

        const double invMicro = 1.0 / static_cast<double>(std::max<size_t>(1, microCount));
        double sqNorm = 0.0;
        for (size_t l = 1; l < m_layers.size(); ++l) {
            for (double v : gradBAccum[l]) {
                const double g = v * invMicro;
                sqNorm += g * g;
            }
            for (double v : gradWAccum[l]) {
                const double g = v * invMicro;
                sqNorm += g * g;
            }
        }

        const double gradNorm = std::sqrt(std::max(0.0, sqNorm));
        double clipThreshold = hp.gradientClipNorm;
        if (hp.adaptiveGradientClipping) {
            const double beta = std::clamp(hp.adaptiveClipBeta, 0.0, 0.9999);
            if (!m_runningGradNormReady) {
                m_runningGradNormEma = gradNorm;
                m_runningGradNormReady = true;
            } else {
                m_runningGradNormEma = beta * m_runningGradNormEma + (1.0 - beta) * gradNorm;
            }
            const double adaptive = std::max(hp.adaptiveClipMin, m_runningGradNormEma * hp.adaptiveClipMultiplier);
            clipThreshold = (clipThreshold > 0.0) ? std::max(clipThreshold, adaptive) : adaptive;
        }

        double clipScale = 1.0;
        if (clipThreshold > 0.0 && gradNorm > clipThreshold && gradNorm > kNumericEps) {
            clipScale = clipThreshold / gradNorm;
        }

        ++t_step;
        for (size_t l = 1; l < m_layers.size(); ++l) {
            auto& gb = gradBWork[l];
            auto& gw = gradWWork[l];
            for (size_t i = 0; i < gb.size(); ++i) gb[i] = gradBAccum[l][i] * invMicro * clipScale;
            for (size_t i = 0; i < gw.size(); ++i) gw[i] = gradWAccum[l][i] * invMicro * clipScale;

            const std::vector<double>* l2Scales = nullptr;
            if (l == 1 && !m_inputL2Scales.empty()) l2Scales = &m_inputL2Scales;
            m_layers[l].updateParametersAccumulated(hp.learningRate,
                                                    hp.l2Lambda,
                                                    fastOptimizer,
                                                    t_step,
                                                    gb,
                                                    gw,
                                                    l2Scales);

            std::fill(gradBAccum[l].begin(), gradBAccum[l].end(), 0.0);
            std::fill(gradWAccum[l].begin(), gradWAccum[l].end(), 0.0);
        }

        if (hp.optimizer == Optimizer::LOOKAHEAD) {
            if (!m_lookaheadInitialized) {
                m_lookaheadSlowWeights.clear();
                m_lookaheadSlowBiases.clear();
                m_lookaheadSlowWeights.reserve(m_layers.size());
                m_lookaheadSlowBiases.reserve(m_layers.size());
                for (const auto& layer : m_layers) {
                    m_lookaheadSlowWeights.push_back(layer.weights());
                    m_lookaheadSlowBiases.push_back(layer.biases());
                }
                m_lookaheadInitialized = true;
            }

            const size_t k = std::max<size_t>(1, hp.lookaheadSyncPeriod);
            const double alpha = std::clamp(hp.lookaheadAlpha, 0.0, 1.0);
            if (t_step % k == 0) {
                for (size_t l = 1; l < m_layers.size(); ++l) {
                    auto& w = m_layers[l].weights();
                    auto& b = m_layers[l].biases();
                    auto& sw = m_lookaheadSlowWeights[l];
                    auto& sb = m_lookaheadSlowBiases[l];

                    for (size_t i = 0; i < w.size() && i < sw.size(); ++i) {
                        sw[i] += alpha * (w[i] - sw[i]);
                        w[i] = sw[i];
                    }
                    for (size_t i = 0; i < b.size() && i < sb.size(); ++i) {
                        sb[i] += alpha * (b[i] - sb[i]);
                        b[i] = sb[i];
                    }
                }
            }
        }
        if (hp.useEmaWeights) {
            updateEmaWeights(hp.emaDecay);
        }
        microCount = 0;
    }

    return batchLoss;
}

double NeuralNet::runEpoch(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y,
                           const std::vector<size_t>& indices,
                           size_t trainSize,
                           const Hyperparameters& hp,
                           size_t& t_step) {
    double epochLoss = 0.0;
    for (size_t i = 0; i < trainSize; i += hp.batchSize) {
        const size_t end = std::min(i + hp.batchSize, trainSize);
        epochLoss += runBatch(X, Y, indices, i, end, hp, t_step);
    }

    double gradNorm = 0.0;
    double wAbs = 0.0;
    double wSq = 0.0;
    size_t gCount = 0;
    size_t wCount = 0;
    for (size_t l = 1; l < m_layers.size(); ++l) {
        for (double g : m_layers[l].gradients()) {
            gradNorm += g * g;
            ++gCount;
        }
        for (double w : m_layers[l].weights()) {
            wAbs += std::abs(w);
            wSq += w * w;
            ++wCount;
        }
    }

    gradientNormHistory.push_back((gCount > 0) ? std::sqrt(gradNorm / static_cast<double>(gCount)) : 0.0);
    weightMeanAbsHistory.push_back((wCount > 0) ? (wAbs / static_cast<double>(wCount)) : 0.0);
    weightStdHistory.push_back((wCount > 0) ? std::sqrt(wSq / static_cast<double>(wCount)) : 0.0);

    const double denom = static_cast<double>(std::max<size_t>(1, trainSize) * m_layers.back().size());
    return epochLoss / std::max(denom, 1.0);
}

double NeuralNet::validate(const std::vector<std::vector<double>>& X,
                           const std::vector<std::vector<double>>& Y,
                           const std::vector<size_t>& indices,
                           size_t trainSize,
                           const Hyperparameters& hp) {
    if (X.size() <= trainSize) return 0.0;

    double loss = 0.0;
    const size_t outDim = m_layers.back().size();
    for (size_t i = trainSize; i < X.size(); ++i) {
        const size_t idx = indices[i];
        feedForward(X[idx], false);
        const auto& out = m_layers.back().outputs();

        if (hp.loss == LossFunction::CROSS_ENTROPY) {
            for (size_t j = 0; j < out.size() && j < Y[idx].size(); ++j) {
                const double p = std::clamp(out[j], 1e-15, 1.0 - 1e-15);
                const double y = Y[idx][j];
                loss -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
            }
        } else {
            loss += meanSquaredError(out, Y[idx], std::min(out.size(), Y[idx].size()));
        }
    }

    const size_t valSize = X.size() - trainSize;
    const double denom = static_cast<double>(std::max<size_t>(1, valSize) * std::max<size_t>(1, outDim));
    return loss / denom;
}

void NeuralNet::train(const std::vector<std::vector<double>>& X,
                      const std::vector<std::vector<double>>& Y,
                      const Hyperparameters& hp,
                      const std::vector<ScaleInfo>& inScales,
                      const std::vector<ScaleInfo>& outScales) {
    if (X.empty() || Y.empty()) throw Seldon::NeuralNetException("Training data cannot be empty");
    if (X.size() != Y.size()) throw Seldon::NeuralNetException("Input and target sample counts must match");
    if (hp.batchSize == 0) throw Seldon::NeuralNetException("Batch size must be greater than zero");

    const size_t expectedInput = m_layers.front().size();
    const size_t expectedOutput = m_layers.back().size();
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != expectedInput) {
            throw Seldon::NeuralNetException("Input row size does not match network input layer size");
        }
        if (Y[i].size() != expectedOutput) {
            throw Seldon::NeuralNetException("Target row size does not match network output layer size");
        }
    }

    inputScales = inScales;
    outputScales = outScales;
    m_useBatchNorm = hp.useBatchNorm;
    m_batchNormMomentum = hp.batchNormMomentum;
    m_batchNormEpsilon = hp.batchNormEpsilon;
    m_useLayerNorm = hp.useLayerNorm;
    m_layerNormEpsilon = hp.layerNormEpsilon;

    if (!hp.incrementalMode) {
        setSeed(hp.seed);
        forwardCounter = 0;
        trainLossHistory.clear();
        valLossHistory.clear();
        gradientNormHistory.clear();
        weightStdHistory.clear();
        weightMeanAbsHistory.clear();
    }

    if (!hp.incrementalMode) {
        m_emaInitialized = false;
        m_emaWeights.clear();
        m_emaBiases.clear();
        m_runningGradNormEma = 0.0;
        m_runningGradNormReady = false;
    }

    m_lookaheadInitialized = false;
    m_lookaheadSlowWeights.clear();
    m_lookaheadSlowBiases.clear();
    lastTrainingDropoutRate = hp.dropoutRate;

    for (size_t l = 1; l < m_layers.size(); ++l) {
        m_layers[l].setActivation((l + 1 == m_layers.size()) ? hp.outputActivation : hp.activation);
    }

    size_t valSize = static_cast<size_t>(std::llround(static_cast<double>(X.size()) * hp.valSplit));
    if (X.size() >= 16) {
        const size_t minVal = std::min<size_t>(8, X.size() / 2);
        valSize = std::max(valSize, minVal);
    }
    if (X.size() >= 8) {
        valSize = std::min(valSize, X.size() - 8);
    }
    size_t trainSize = X.size() - valSize;
    if (trainSize == 0) {
        trainSize = X.size();
        valSize = 0;
    }

    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    double bestVal = std::numeric_limits<double>::infinity();
    bool hasBest = false;
    int patience = 0;
    int plateau = 0;
    int cooldown = 0;
    int reductions = 0;
    size_t t_step = 0;
    double lr = hp.learningRate;

    bool emaReady = false;
    double valEma = 0.0;

    std::vector<std::vector<double>> bestW(m_layers.size());
    std::vector<std::vector<double>> bestB(m_layers.size());

    for (size_t epoch = 0; epoch < hp.epochs; ++epoch) {
        if (hp.verbose && (epoch % std::max<size_t>(1, hp.epochs / 10) == 0)) {
            std::cout << "\r[Agent] Training Neural Lattice: ["
                      << (epoch * 100 / std::max<size_t>(1, hp.epochs)) << "%] " << std::flush;
        }

        std::shuffle(indices.begin(), indices.end(), rng);

        Hyperparameters h = hp;
        const size_t warmupEpochs = std::min(hp.lrWarmupEpochs, hp.epochs);
        double scheduledLr = lr;
        if (warmupEpochs > 0 && epoch < warmupEpochs) {
            const double warm = static_cast<double>(epoch + 1) / static_cast<double>(warmupEpochs);
            scheduledLr *= std::clamp(warm, 0.0, 1.0);
        }

        const double lrFloorFactor = std::clamp(hp.lrScheduleMinFactor, 0.0, 1.0);
        const double lrFloor = std::max(hp.minLearningRate, lr * lrFloorFactor);
        if (epoch >= warmupEpochs) {
            if (hp.useCosineAnnealing) {
                const size_t span = std::max<size_t>(1, hp.epochs - warmupEpochs);
                const double t = static_cast<double>(epoch - warmupEpochs) / static_cast<double>(span);
                const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * std::clamp(t, 0.0, 1.0)));
                scheduledLr = lrFloor + (lr - lrFloor) * cosine;
            } else if (hp.useCyclicalLr) {
                const size_t cycle = std::max<size_t>(2, hp.lrCycleEpochs);
                const size_t cyclePos = (epoch - warmupEpochs) % cycle;
                const double phase = static_cast<double>(cyclePos) / static_cast<double>(cycle - 1);
                const double tri = 1.0 - std::abs(2.0 * phase - 1.0);
                scheduledLr = lrFloor + (lr - lrFloor) * tri;
            }
        }

        h.learningRate = std::max(hp.minLearningRate, scheduledLr);
        const double trainLoss = runEpoch(X, Y, indices, trainSize, h, t_step);
        const double rawVal = (valSize > 0) ? validate(X, Y, indices, trainSize, hp) : 0.0;

        double monitoredVal = rawVal;
        if (valSize > 0 && hp.useValidationLossEma) {
            const double beta = std::clamp(hp.validationLossEmaBeta, 0.0, 0.999);
            if (!emaReady) {
                valEma = rawVal;
                emaReady = true;
            } else {
                valEma = beta * valEma + (1.0 - beta) * rawVal;
            }
            monitoredVal = valEma;
        }

        trainLossHistory.push_back(trainLoss);
        valLossHistory.push_back(rawVal);

        if (hp.verbose) {
            std::cout << "[Epoch " << (epoch + 1) << "/" << hp.epochs << "] Train Loss: "
                      << trainLoss << " | Val Loss: " << rawVal << std::endl;
        }

        if (valSize == 0) continue;

        if (monitoredVal < (bestVal - hp.minDelta)) {
            bestVal = monitoredVal;
            patience = 0;
            plateau = 0;
            hasBest = true;
            for (size_t l = 1; l < m_layers.size(); ++l) {
                bestW[l] = m_layers[l].weights();
                bestB[l] = m_layers[l].biases();
            }
        } else {
            ++patience;
            if (cooldown > 0) {
                --cooldown;
            } else {
                ++plateau;
            }

            if (hp.lrDecay > 0.0 && hp.lrDecay < 1.0 && hp.lrPlateauPatience > 0 &&
                plateau >= hp.lrPlateauPatience && reductions < hp.maxLrReductions) {
                const double nextLr = std::max(hp.minLearningRate, lr * hp.lrDecay);
                if (nextLr < lr) {
                    lr = nextLr;
                    ++reductions;
                    if (hp.verbose) {
                        std::cout << "[Agent] Plateau LR scheduler: reducing learning rate to " << lr << std::endl;
                    }
                }
                plateau = 0;
                cooldown = std::max(0, hp.lrCooldownEpochs);
            }

            if (patience >= hp.earlyStoppingPatience) {
                if (hp.verbose) {
                    std::cout << "\n[Agent] Early stopping triggered at epoch " << epoch << std::endl;
                }
                break;
            }
        }
    }

    if (hasBest) {
        for (size_t l = 1; l < m_layers.size(); ++l) {
            m_layers[l].weights() = bestW[l];
            m_layers[l].biases() = bestB[l];
        }
    }

    if (hp.useEmaWeights && m_emaInitialized) {
        applyEmaWeights();
    }

    std::cout << "\r[Agent] Training Neural Lattice: [100%] Complete.          " << std::endl;
}

void NeuralNet::trainIncremental(const std::vector<std::vector<double>>& X,
                                 const std::vector<std::vector<double>>& Y,
                                 const Hyperparameters& hp,
                                 size_t chunkRows,
                                 const std::vector<ScaleInfo>& inScales,
                                 const std::vector<ScaleInfo>& outScales) {
    if (chunkRows == 0) throw Seldon::NeuralNetException("chunkRows must be > 0 for incremental training");
    if (X.size() != Y.size()) throw Seldon::NeuralNetException("Input and target sample counts must match");

    Hyperparameters h = hp;
    h.incrementalMode = true;

    const size_t n = X.size();
    for (size_t start = 0; start < n; start += chunkRows) {
        const size_t end = std::min(start + chunkRows, n);
        std::vector<std::vector<double>> chunkX;
        std::vector<std::vector<double>> chunkY;
        chunkX.reserve(end - start);
        chunkY.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            chunkX.push_back(X[i]);
            chunkY.push_back(Y[i]);
        }
        train(chunkX, chunkY, h, inScales, outScales);
    }
}

std::vector<double> NeuralNet::predict(const std::vector<double>& inputValues) {
    std::vector<double> scaled = inputValues;
    if (!inputScales.empty() && inputScales.size() == inputValues.size()) {
        for (size_t i = 0; i < scaled.size(); ++i) {
            double range = inputScales[i].max - inputScales[i].min;
            if (std::abs(range) <= kNumericEps) range = 1.0;
            scaled[i] = (scaled[i] - inputScales[i].min) / range;
        }
    }

    feedForward(scaled, false);

    std::vector<double> out(m_layers.back().size(), 0.0);
    for (size_t i = 0; i < out.size(); ++i) {
        double v = m_layers.back().outputs()[i];
        if (!outputScales.empty() && i < outputScales.size()) {
            double range = outputScales[i].max - outputScales[i].min;
            if (std::abs(range) <= kNumericEps) range = 1.0;
            v = v * range + outputScales[i].min;
        }
        out[i] = v;
    }
    return out;
}

NeuralNet::UncertaintyEstimate NeuralNet::predictWithUncertainty(const std::vector<double>& inputValues,
                                                                 size_t samples,
                                                                 double dropoutRate) {
    if (samples == 0) samples = 1;

    const size_t outDim = m_layers.back().size();
    UncertaintyEstimate out;
    out.mean.assign(outDim, 0.0);
    out.stddev.assign(outDim, 0.0);
    out.ciLow.assign(outDim, 0.0);
    out.ciHigh.assign(outDim, 0.0);

    std::vector<std::vector<double>> draws(samples, std::vector<double>(outDim, 0.0));
    for (size_t s = 0; s < samples; ++s) {
        feedForward(inputValues, true, dropoutRate);
        for (size_t j = 0; j < outDim; ++j) {
            draws[s][j] = m_layers.back().outputs()[j];
        }
    }

    for (size_t j = 0; j < outDim; ++j) {
        double sum = 0.0;
        for (size_t s = 0; s < samples; ++s) sum += draws[s][j];
        const double mu = sum / static_cast<double>(samples);

        double var = 0.0;
        for (size_t s = 0; s < samples; ++s) {
            const double d = draws[s][j] - mu;
            var += d * d;
        }
        const double sd = std::sqrt(var / static_cast<double>(std::max<size_t>(1, samples - 1)));

        out.mean[j] = mu;
        out.stddev[j] = sd;
        out.ciLow[j] = mu - 1.96 * sd;
        out.ciHigh[j] = mu + 1.96 * sd;
    }

    return out;
}

std::vector<double> NeuralNet::calculateFeatureImportance(const std::vector<std::vector<double>>& X,
                                                          const std::vector<std::vector<double>>& Y,
                                                          size_t trials,
                                                          size_t maxRows,
                                                          bool parallel) {
    (void)trials;

    if (X.empty() || Y.empty() || X.size() != Y.size()) return {};
    const size_t numFeatures = X[0].size();
    const size_t outDim = Y[0].size();
    if (numFeatures == 0 || outDim == 0) return {};

    std::vector<size_t> rows(X.size());
    std::iota(rows.begin(), rows.end(), 0);

    size_t rowCap = (maxRows > 0) ? maxRows : rows.size();
    rowCap = std::min(rowCap, static_cast<size_t>(256));
    if (rows.size() > rowCap) {
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(rowCap);
    }

    std::vector<size_t> features(numFeatures);
    std::iota(features.begin(), features.end(), 0);

    const size_t featureCap = 64;
    if (numFeatures > featureCap) {
        std::vector<std::pair<double, size_t>> score;
        score.reserve(numFeatures);
        for (size_t f = 0; f < numFeatures; ++f) {
            double mean = 0.0;
            for (size_t idx : rows) mean += X[idx][f];
            mean /= static_cast<double>(std::max<size_t>(1, rows.size()));

            double var = 0.0;
            for (size_t idx : rows) {
                const double d = X[idx][f] - mean;
                var += d * d;
            }
            score.push_back({var, f});
        }

        std::nth_element(
            score.begin(),
            score.begin() + static_cast<long>(featureCap),
            score.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        features.clear();
        features.reserve(featureCap);
        for (size_t i = 0; i < featureCap; ++i) features.push_back(score[i].second);
    }

    std::vector<double> importance(numFeatures, 0.0);
    const bool canUseParallel = parallel && rows.size() >= 32 && features.size() >= 8;

#ifdef USE_OPENMP
    if (canUseParallel) {
        const int threadCount = std::max(1, omp_get_max_threads());
        std::vector<std::vector<double>> partial(static_cast<size_t>(threadCount), std::vector<double>(numFeatures, 0.0));

        #pragma omp parallel default(none) shared(rows, X, Y, features, outDim, partial, numFeatures)
        {
            const int tid = omp_get_thread_num();
            NeuralNet localNet = *this;
            std::vector<double> work;
            work.reserve(numFeatures);

            #pragma omp for schedule(static)
            for (int rowPos = 0; rowPos < static_cast<int>(rows.size()); ++rowPos) {
                const size_t idx = rows[static_cast<size_t>(rowPos)];
                const std::vector<double>& x = X[idx];
                const std::vector<double>& y = Y[idx];
                const std::vector<double> base = localNet.predict(x);

                double rowErr = 0.0;
                for (size_t j = 0; j < outDim; ++j) {
                    const double p = (j < base.size()) ? base[j] : 0.0;
                    const double t = (j < y.size()) ? y[j] : 0.0;
                    rowErr += std::abs(p - t);
                }
                const double rowWeight = 1.0 + rowErr / static_cast<double>(outDim);

                work = x;
                for (size_t f : features) {
                    const double baseX = work[f];
                    const double eps = std::max(1e-4, 1e-3 * std::max(1.0, std::abs(baseX)));

                    work[f] = baseX + eps;
                    const std::vector<double> plus = localNet.predict(work);
                    work[f] = baseX;

                    double sens = 0.0;
                    for (size_t j = 0; j < outDim; ++j) {
                        const double p0 = (j < base.size()) ? base[j] : 0.0;
                        const double p1 = (j < plus.size()) ? plus[j] : 0.0;
                        sens += std::abs((p1 - p0) / eps);
                    }
                    partial[static_cast<size_t>(tid)][f] += rowWeight * (sens / static_cast<double>(outDim));
                }
            }
        }

        for (const auto& local : partial) {
            for (size_t f : features) {
                importance[f] += local[f];
            }
        }
    } else
#endif
    {
        std::vector<double> work;
        work.reserve(numFeatures);

        for (size_t idx : rows) {
            const std::vector<double>& x = X[idx];
            const std::vector<double>& y = Y[idx];
            const std::vector<double> base = predict(x);

            double rowErr = 0.0;
            for (size_t j = 0; j < outDim; ++j) {
                const double p = (j < base.size()) ? base[j] : 0.0;
                const double t = (j < y.size()) ? y[j] : 0.0;
                rowErr += std::abs(p - t);
            }
            const double rowWeight = 1.0 + rowErr / static_cast<double>(outDim);

            work = x;
            for (size_t f : features) {
                const double baseX = work[f];
                const double eps = std::max(1e-4, 1e-3 * std::max(1.0, std::abs(baseX)));

                work[f] = baseX + eps;
                const std::vector<double> plus = predict(work);
                work[f] = baseX;

                double sens = 0.0;
                for (size_t j = 0; j < outDim; ++j) {
                    const double p0 = (j < base.size()) ? base[j] : 0.0;
                    const double p1 = (j < plus.size()) ? plus[j] : 0.0;
                    sens += std::abs((p1 - p0) / eps);
                }
                importance[f] += rowWeight * (sens / static_cast<double>(outDim));
            }
        }
    }

    for (size_t f : features) {
        importance[f] /= static_cast<double>(std::max<size_t>(1, rows.size()));
    }

    const double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (sum > kNumericEps) {
        for (double& v : importance) v /= sum;
    } else {
        const double uniform = 1.0 / static_cast<double>(importance.size());
        for (double& v : importance) v = uniform;
    }

    return importance;
}

std::vector<double> NeuralNet::calculateIntegratedGradients(const std::vector<std::vector<double>>& X,
                                                            size_t steps,
                                                            size_t maxRows) {
    if (X.empty() || X[0].empty()) return {};
    steps = std::max<size_t>(4, steps);

    const size_t numFeatures = X[0].size();
    std::vector<size_t> rows(X.size());
    std::iota(rows.begin(), rows.end(), 0);

    if (maxRows > 0 && rows.size() > maxRows) {
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(maxRows);
    }

    std::vector<double> baseline(numFeatures, 0.0);
    for (size_t f = 0; f < numFeatures; ++f) {
        double s = 0.0;
        for (size_t idx : rows) s += X[idx][f];
        baseline[f] = s / static_cast<double>(std::max<size_t>(1, rows.size()));
    }

    std::vector<double> attr(numFeatures, 0.0);
    constexpr double eps = 1e-3;

    std::vector<double> interp(numFeatures, 0.0);
    std::vector<double> plus(numFeatures, 0.0);

    for (size_t idx : rows) {
        const std::vector<double>& row = X[idx];
        std::vector<double> gradAccum(numFeatures, 0.0);

        for (size_t s = 1; s <= steps; ++s) {
            const double alpha = static_cast<double>(s) / static_cast<double>(steps);
            for (size_t f = 0; f < numFeatures; ++f) {
                interp[f] = baseline[f] + alpha * (row[f] - baseline[f]);
            }

            const std::vector<double> base = predict(interp);
            const double baseOut = base.empty() ? 0.0 : base[0];

            for (size_t f = 0; f < numFeatures; ++f) {
                plus = interp;
                plus[f] += eps;
                const std::vector<double> pp = predict(plus);
                const double plusOut = pp.empty() ? 0.0 : pp[0];
                gradAccum[f] += (plusOut - baseOut) / eps;
            }
        }

        for (size_t f = 0; f < numFeatures; ++f) {
            attr[f] += std::abs((row[f] - baseline[f]) * (gradAccum[f] / static_cast<double>(steps)));
        }
    }

    const double c = static_cast<double>(std::max<size_t>(1, rows.size()));
    for (double& v : attr) v /= c;

    const double sum = std::accumulate(attr.begin(), attr.end(), 0.0);
    if (sum > kNumericEps) {
        for (double& v : attr) v /= sum;
    }
    return attr;
}

void NeuralNet::saveModelBinary(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw Seldon::IOException("Could not open " + filename + " for writing");

    const char signature[] = "SELDON_BIN_V2";
    out.write(signature, sizeof(signature));

    writeLE(out, kModelFormatVersion);

    uint64_t checksum = kChecksumOffsetBasis;
    updateChecksum(checksum, kModelFormatVersion);

    const uint64_t topSize = static_cast<uint64_t>(topology.size());
    writeLE(out, topSize);
    updateChecksum(checksum, topSize);

    for (size_t layer : topology) {
        const uint64_t width = static_cast<uint64_t>(layer);
        writeLE(out, width);
        updateChecksum(checksum, width);
    }

    for (const auto& layer : m_layers) {
        const int32_t act = static_cast<int32_t>(layer.activation());
        writeLE(out, act);
        updateChecksum(checksum, act);
    }

    const uint64_t inS = static_cast<uint64_t>(inputScales.size());
    const uint64_t outS = static_cast<uint64_t>(outputScales.size());
    writeLE(out, inS);
    writeLE(out, outS);
    updateChecksum(checksum, inS);
    updateChecksum(checksum, outS);

    for (const auto& s : inputScales) {
        writeLE(out, s.min);
        writeLE(out, s.max);
        updateChecksum(checksum, s.min);
        updateChecksum(checksum, s.max);
    }
    for (const auto& s : outputScales) {
        writeLE(out, s.min);
        writeLE(out, s.max);
        updateChecksum(checksum, s.min);
        updateChecksum(checksum, s.max);
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        const auto& layer = m_layers[l];
        for (double b : layer.biases()) {
            writeLE(out, b);
            updateChecksum(checksum, b);
        }
        for (double w : layer.weights()) {
            writeLE(out, w);
            updateChecksum(checksum, w);
        }
    }

    writeLE(out, checksum);
}

void NeuralNet::loadModelBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw Seldon::IOException("Could not open " + filename + " for reading");

    char signature[sizeof("SELDON_BIN_V2")];
    in.read(signature, sizeof(signature));
    if (!in) throw Seldon::NeuralNetException("Failed to read model signature from " + filename);

    if (std::string(signature) != "SELDON_BIN_V2") {
        throw Seldon::NeuralNetException("Unsupported or invalid binary model signature in " + filename);
    }

    uint32_t version = 0;
    readLE(in, version);
    if (version != kModelFormatVersion) {
        throw Seldon::NeuralNetException("Unsupported model version in file: " + filename);
    }

    uint64_t checksum = kChecksumOffsetBasis;
    updateChecksum(checksum, version);

    uint64_t topSize = 0;
    readLE(in, topSize);
    updateChecksum(checksum, topSize);
    if (topSize < 2 || topSize > 1000000ULL) {
        throw Seldon::NeuralNetException("Invalid topology size in model file: " + filename);
    }

    std::vector<size_t> top;
    top.reserve(static_cast<size_t>(topSize));
    for (uint64_t i = 0; i < topSize; ++i) {
        uint64_t width = 0;
        readLE(in, width);
        updateChecksum(checksum, width);
        if (width == 0 || width > 1000000ULL) {
            throw Seldon::NeuralNetException("Invalid layer width in model file: " + filename);
        }
        top.push_back(static_cast<size_t>(width));
    }

    *this = NeuralNet(top);

    for (size_t i = 0; i < m_layers.size(); ++i) {
        int32_t act = 0;
        readLE(in, act);
        updateChecksum(checksum, act);
        m_layers[i].setActivation(static_cast<Activation>(act));
    }

    uint64_t inS = 0;
    uint64_t outS = 0;
    readLE(in, inS);
    readLE(in, outS);
    updateChecksum(checksum, inS);
    updateChecksum(checksum, outS);

    inputScales.resize(static_cast<size_t>(inS));
    outputScales.resize(static_cast<size_t>(outS));

    for (uint64_t i = 0; i < inS; ++i) {
        readLE(in, inputScales[static_cast<size_t>(i)].min);
        readLE(in, inputScales[static_cast<size_t>(i)].max);
        updateChecksum(checksum, inputScales[static_cast<size_t>(i)].min);
        updateChecksum(checksum, inputScales[static_cast<size_t>(i)].max);
    }
    for (uint64_t i = 0; i < outS; ++i) {
        readLE(in, outputScales[static_cast<size_t>(i)].min);
        readLE(in, outputScales[static_cast<size_t>(i)].max);
        updateChecksum(checksum, outputScales[static_cast<size_t>(i)].min);
        updateChecksum(checksum, outputScales[static_cast<size_t>(i)].max);
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        auto& layer = m_layers[l];
        for (double& b : layer.biases()) {
            readLE(in, b);
            updateChecksum(checksum, b);
        }
        for (double& w : layer.weights()) {
            readLE(in, w);
            updateChecksum(checksum, w);
        }
    }

    uint64_t storedChecksum = 0;
    readLE(in, storedChecksum);
    if (storedChecksum != checksum) {
        throw Seldon::NeuralNetException("Model checksum mismatch (corrupt file): " + filename);
    }
}
