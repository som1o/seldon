#include "NeuralNet.h"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include "SeldonExceptions.h"

namespace {
    constexpr uint32_t kModelFormatVersion = 3;
    constexpr uint64_t kChecksumOffsetBasis = 1469598103934665603ULL;
    constexpr uint64_t kChecksumPrime = 1099511628211ULL;

    template<typename T>
    void swapEndian(T &val);

    bool isLittleEndian() {
        uint16_t number = 0x1;
        char *numPtr = reinterpret_cast<char*>(&number);
        return (numPtr[0] == 1);
    }

    template<typename T>
    void updateChecksum(uint64_t& checksum, const T& value) {
        T copy = value;
        if (!isLittleEndian()) {
            swapEndian(copy);
        }
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&copy);
        for (size_t i = 0; i < sizeof(T); ++i) {
            checksum ^= static_cast<uint64_t>(bytes[i]);
            checksum *= kChecksumPrime;
        }
    }

    template<typename T>
    void swapEndian(T &val) {
        auto* first = reinterpret_cast<unsigned char*>(&val);
        std::reverse(first, first + sizeof(T));
    }

    template<typename T>
    void writeLE(std::ostream& out, T val) {
        if (!isLittleEndian()) {
            swapEndian(val);
        }
        out.write(reinterpret_cast<const char*>(&val), sizeof(T));
        if (!out) {
            throw Seldon::IOException("Binary write failed");
        }
    }

    template<typename T>
    void readLE(std::istream& in, T& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!in) {
            throw Seldon::NeuralNetException("Binary read failed or model file is truncated");
        }
        if (!isLittleEndian()) {
            swapEndian(val);
        }
    }

}

NeuralNet::NeuralNet(std::vector<size_t> topology) : topology(topology) {
    if (topology.size() < 2) {
        throw Seldon::NeuralNetException("Topology must include at least input and output layers");
    }
    for (size_t layerSize : topology) {
        if (layerSize == 0) {
            throw Seldon::NeuralNetException("Layer size cannot be zero");
        }
    }

    rng.seed(seedState);

    for (size_t l = 0; l < topology.size(); ++l) {
        const Activation activation = (l == topology.size() - 1) ? Activation::SIGMOID : Activation::GELU;
        const size_t prevSize = (l > 0) ? topology[l - 1] : 0;
        m_layers.emplace_back(topology[l], prevSize, activation, rng);
    }
}

void NeuralNet::setSeed(uint32_t seed) {
    seedState = seed;
    rng.seed(seedState);
}

void NeuralNet::setInputL2Scales(const std::vector<double>& scales) {
    if (!m_layers.empty() && m_layers.size() > 1 && !scales.empty() && scales.size() != m_layers[1].prevSize()) {
        throw Seldon::NeuralNetException("Input L2 scales must match input layer width");
    }
    m_inputL2Scales = scales;
}

void NeuralNet::feedForward(const std::vector<double>& inputValues, bool isTraining, double dropoutRate) {
    if (m_layers.empty()) {
        throw Seldon::NeuralNetException("Network has no layers");
    }

    size_t inputSize = std::min(inputValues.size(), m_layers[0].size());
    for (size_t i = 0; i < inputSize; ++i) {
        m_layers[0].outputs()[i] = inputValues[i];
    }
    for (size_t i = inputSize; i < m_layers[0].size(); ++i) {
        m_layers[0].outputs()[i] = 0.0;
    }

    ++forwardCounter;
    const bool useBatchNorm = m_useBatchNorm;
    const double batchNormMomentum = m_batchNormMomentum;
    const double batchNormEpsilon = m_batchNormEpsilon;
    const bool useLayerNorm = m_useLayerNorm;
    const double layerNormEpsilon = m_layerNormEpsilon;
    for (size_t l = 1; l < m_layers.size(); ++l) {
        const bool dropoutEnabled = isTraining && l < m_layers.size() - 1;
        m_layers[l].forward(m_layers[l - 1],
                            dropoutEnabled,
                            dropoutRate,
                            useBatchNorm && l < m_layers.size() - 1,
                            batchNormMomentum,
                            batchNormEpsilon,
                            useLayerNorm && l < m_layers.size() - 1,
                            layerNormEpsilon,
                            seedState,
                            forwardCounter + l * 1315423911ULL);
    }
}

void NeuralNet::computeGradients(const std::vector<double>& targetValues, LossFunction loss) {
    if (m_layers.size() < 2 || targetValues.size() != m_layers.back().size()) {
        throw Seldon::NeuralNetException("Target dimensions do not match output layer");
    }

    DenseLayer& outputLayer = m_layers.back();
    if (loss == LossFunction::CROSS_ENTROPY && outputLayer.activation() != Activation::SIGMOID) {
        throw Seldon::NeuralNetException("Cross-Entropy loss currently requires SIGMOID output activation");
    }
    outputLayer.computeOutputGradients(targetValues, loss == LossFunction::CROSS_ENTROPY);

    for (size_t l = m_layers.size() - 1; l-- > 1;) {
        DenseLayer& hidden = m_layers[l];
        DenseLayer& next = m_layers[l + 1];
        hidden.accumulateHiddenGradients(next);
        hidden.applyActivationDerivativeAndDropout();
    }
}

void NeuralNet::applyOptimization(const Hyperparameters& hp, size_t t_step) {
    NeuralOptimizer fastOptimizer = hp.optimizer;
    if (hp.optimizer == Optimizer::LOOKAHEAD) {
        fastOptimizer = hp.lookaheadFastOptimizer;
        if (fastOptimizer == Optimizer::LOOKAHEAD) {
            fastOptimizer = Optimizer::ADAM;
        }
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        const std::vector<double>* layerInputL2 = nullptr;
        if (l == 1 && !m_inputL2Scales.empty()) {
            layerInputL2 = &m_inputL2Scales;
        }
        m_layers[l].updateParameters(m_layers[l - 1], hp.learningRate, hp.l2Lambda, fastOptimizer, t_step, layerInputL2);
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

        const size_t syncPeriod = std::max<size_t>(1, hp.lookaheadSyncPeriod);
        const double alpha = std::clamp(hp.lookaheadAlpha, 0.0, 1.0);
        if (t_step % syncPeriod == 0) {
            for (size_t l = 1; l < m_layers.size(); ++l) {
                auto& weights = m_layers[l].weights();
                auto& biases = m_layers[l].biases();
                auto& slowW = m_lookaheadSlowWeights[l];
                auto& slowB = m_lookaheadSlowBiases[l];

                for (size_t i = 0; i < weights.size() && i < slowW.size(); ++i) {
                    slowW[i] += alpha * (weights[i] - slowW[i]);
                    weights[i] = slowW[i];
                }
                for (size_t i = 0; i < biases.size() && i < slowB.size(); ++i) {
                    slowB[i] += alpha * (biases[i] - slowB[i]);
                    biases[i] = slowB[i];
                }
            }
        }
    }
}

void NeuralNet::backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step) {
    computeGradients(targetValues, hp.loss);

    if (hp.gradientClipNorm > 0.0) {
        for (size_t l = 1; l < m_layers.size(); ++l) {
            for (double& g : m_layers[l].gradients()) {
                g = std::clamp(g, -hp.gradientClipNorm, hp.gradientClipNorm);
            }
        }

        double sqNorm = 0.0;
        for (size_t l = 1; l < m_layers.size(); ++l) {
            for (double g : m_layers[l].gradients()) sqNorm += g * g;
        }
        double norm = std::sqrt(sqNorm);
        if (norm > hp.gradientClipNorm && norm > 1e-12) {
            double scale = hp.gradientClipNorm / norm;
            for (size_t l = 1; l < m_layers.size(); ++l) {
                for (double& g : m_layers[l].gradients()) g *= scale;
            }
        }
    }

    applyOptimization(hp, t_step);
}

void NeuralNet::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                      const Hyperparameters& hp,
                      const std::vector<ScaleInfo>& inScales,
                      const std::vector<ScaleInfo>& outScales) 
{
    if (X.empty() || Y.empty()) {
        throw Seldon::NeuralNetException("Training data cannot be empty");
    }
    if (X.size() != Y.size()) {
        throw Seldon::NeuralNetException("Input and target sample counts must match");
    }
    if (hp.batchSize == 0) {
        throw Seldon::NeuralNetException("Batch size must be greater than zero");
    }

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

    this->inputScales = inScales;
    this->outputScales = outScales;
    m_useBatchNorm = hp.useBatchNorm;
    m_batchNormMomentum = hp.batchNormMomentum;
    m_batchNormEpsilon = hp.batchNormEpsilon;
    m_useLayerNorm = hp.useLayerNorm;
    m_layerNormEpsilon = hp.layerNormEpsilon;
    m_lookaheadInitialized = false;
    m_lookaheadSlowWeights.clear();
    m_lookaheadSlowBiases.clear();
    setSeed(hp.seed);
    forwardCounter = 0;
    trainLossHistory.clear();
    valLossHistory.clear();

    for (size_t l = 1; l < m_layers.size(); ++l) {
        m_layers[l].setActivation((l == m_layers.size() - 1) ? hp.outputActivation : hp.activation);
    }

    size_t valSize = static_cast<size_t>(X.size() * hp.valSplit);
    size_t trainSize = X.size() - valSize;
    
    // Safety guard for extremely small datasets
    if (trainSize == 0) {
        trainSize = X.size();
        valSize = 0;
    }

    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    double bestValLoss = 1e30;
    int patienceCounter = 0;
    size_t t_step = 0;
    double currentLR = hp.learningRate;
    int lrPlateauCounter = 0;
    int lrCooldownCounter = 0;
    int lrReductionCount = 0;
    bool emaInitialized = false;
    double valLossEma = 0.0;
    bool hasBestCheckpoint = false;
    std::vector<std::vector<double>> bestWeights;
    std::vector<std::vector<double>> bestBiases;
    bestWeights.resize(m_layers.size());
    bestBiases.resize(m_layers.size());

    for (size_t epoch = 0; epoch < hp.epochs; ++epoch) {
        if (epoch % std::max(size_t(1), hp.epochs / 10) == 0) {
            std::cout << "\r[Agent] Training Neural Lattice: [" << (epoch * 100 / hp.epochs) << "%] " << std::flush;
        }
        std::shuffle(indices.begin(), indices.end(), rng);

        Hyperparameters currentHp = hp;
        currentHp.learningRate = currentLR;
        double trainLoss = runEpoch(X, Y, indices, trainSize, currentHp, t_step);
        double valLoss = valSize > 0 ? validate(X, Y, indices, trainSize, hp) : 0.0;
        double monitoredValLoss = valLoss;
        if (valSize > 0 && hp.useValidationLossEma) {
            const double beta = std::clamp(hp.validationLossEmaBeta, 0.0, 0.999);
            if (!emaInitialized) {
                valLossEma = valLoss;
                emaInitialized = true;
            } else {
                valLossEma = beta * valLossEma + (1.0 - beta) * valLoss;
            }
            monitoredValLoss = valLossEma;
        }
        trainLossHistory.push_back(trainLoss);
        valLossHistory.push_back(valLoss);

        if (hp.verbose) {
            std::cout << "[Epoch " << (epoch + 1) << "/" << hp.epochs << "] Train Loss: " << trainLoss << " | Val Loss: " << valLoss << std::endl;
        }

        if (valSize > 0) {
            if (monitoredValLoss < (bestValLoss - hp.minDelta)) {
                bestValLoss = monitoredValLoss;
                patienceCounter = 0;
                lrPlateauCounter = 0;
                hasBestCheckpoint = true;
                for (size_t l = 1; l < m_layers.size(); ++l) {
                    bestWeights[l] = m_layers[l].weights();
                    bestBiases[l] = m_layers[l].biases();
                }
            } else {
                patienceCounter++;
                if (lrCooldownCounter > 0) {
                    lrCooldownCounter--;
                } else {
                    lrPlateauCounter++;
                }

                if (hp.lrDecay > 0.0 && hp.lrDecay < 1.0 && hp.lrPlateauPatience > 0 &&
                    lrPlateauCounter >= hp.lrPlateauPatience &&
                    lrReductionCount < hp.maxLrReductions) {
                    const double nextLR = std::max(hp.minLearningRate, currentLR * hp.lrDecay);
                    const bool changed = nextLR < currentLR;
                    currentLR = nextLR;
                    lrPlateauCounter = 0;
                    lrCooldownCounter = std::max(0, hp.lrCooldownEpochs);
                    if (changed) {
                        lrReductionCount++;
                    }
                    if (hp.verbose && changed) {
                        std::cout << "[Agent] Plateau LR scheduler: reducing learning rate to " << currentLR << std::endl;
                    }
                }

                if (patienceCounter >= hp.earlyStoppingPatience) {
                    if (hp.verbose) std::cout << "\n[Agent] Early stopping triggered at epoch " << epoch << std::endl;
                    break;
                }
            }
        }
    }

    if (hasBestCheckpoint) {
        for (size_t l = 1; l < m_layers.size(); ++l) {
            m_layers[l].weights() = bestWeights[l];
            m_layers[l].biases() = bestBiases[l];
        }
    }

    std::cout << "\r[Agent] Training Neural Lattice: [100%] Complete.          " << std::endl;
}

double NeuralNet::runEpoch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                           const std::vector<size_t>& indices, size_t trainSize, 
                           const Hyperparameters& hp, size_t& t_step) {

    double epochLoss = 0.0;
    for (size_t i = 0; i < trainSize; i += hp.batchSize) {
        size_t currentBatchEnd = std::min(i + hp.batchSize, trainSize);
        epochLoss += runBatch(X, Y, indices, i, currentBatchEnd, hp, t_step);
    }
    double denom = static_cast<double>(trainSize * m_layers.back().size());
    return denom > 0 ? epochLoss / denom : 0.0;
}

double NeuralNet::runBatch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                            const std::vector<size_t>& indices, size_t batchStart, size_t batchEnd, 
                            const Hyperparameters& hp, size_t& t_step) {
    double batchLoss = 0.0;
    t_step++;
    for (size_t b = batchStart; b < batchEnd; ++b) {
        feedForward(X[indices[b]], true, hp.dropoutRate);
        backpropagate(Y[indices[b]], hp, t_step);

        size_t outputSize = std::min(Y[indices[b]].size(), m_layers.back().outputs().size());
        for (size_t j = 0; j < outputSize; ++j) {
            double actual = Y[indices[b]][j];
            double pred = m_layers.back().outputs()[j];
            if (hp.loss == LossFunction::CROSS_ENTROPY) {
                pred = std::clamp(pred, 1e-15, 1.0 - 1e-15);
                batchLoss -= (actual * std::log(pred) + (1.0 - actual) * std::log(1.0 - pred));
            } else {
                batchLoss += (actual - pred) * (actual - pred);
            }
        }
    }
    return batchLoss;
}

double NeuralNet::validate(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                           const std::vector<size_t>& indices, size_t trainSize, const Hyperparameters& hp) {
    double valLossSum = 0.0;
    size_t valSize = X.size() - trainSize;
    for (size_t i = trainSize; i < X.size(); ++i) {
        feedForward(X[indices[i]], false);
        for (size_t j = 0; j < Y[indices[i]].size(); ++j) {
            double actual = Y[indices[i]][j];
            double pred = m_layers.back().outputs()[j];
            if (hp.loss == LossFunction::CROSS_ENTROPY) {
                pred = std::clamp(pred, 1e-15, 1.0 - 1e-15);
                valLossSum -= (actual * std::log(pred) + (1.0 - actual) * std::log(1.0 - pred));
            } else {
                valLossSum += (actual - pred) * (actual - pred);
            }
        }
    }
    double denom = static_cast<double>(valSize * m_layers.back().size());
    return denom > 0 ? valLossSum / denom : 0.0;
}

std::vector<double> NeuralNet::predict(const std::vector<double>& inputValues) {
    std::vector<double> scaledInput = inputValues;
    if (!inputScales.empty() && inputScales.size() == inputValues.size()) {
        for (size_t i = 0; i < inputValues.size(); ++i) {
            double range = inputScales[i].max - inputScales[i].min;
            if (range == 0) range = 1.0;
            scaledInput[i] = (inputValues[i] - inputScales[i].min) / range;
        }
    }

    feedForward(scaledInput, false);
    
    std::vector<double> result;
    for (size_t i = 0; i < m_layers.back().size(); ++i) {
        double val = m_layers.back().outputs()[i];
        if (!outputScales.empty() && i < outputScales.size()) {
            double range = outputScales[i].max - outputScales[i].min;
            if (range == 0) range = 1.0;
            val = val * range + outputScales[i].min;
        }
        result.push_back(val);
    }
    return result;
}
// Textual save/load logic removed for V3. Refer to binary persistence.

std::vector<double> NeuralNet::calculateFeatureImportance(const std::vector<std::vector<double>>& X, 
                                               const std::vector<std::vector<double>>& Y,
                                               size_t trials) {
    if (X.empty() || Y.empty() || X.size() != Y.size()) return {};
    size_t numFeatures = X[0].size();
    size_t outDim = Y[0].size();
    std::vector<double> importances(numFeatures, 0.0);
    std::vector<double> fallbackSensitivity(numFeatures, 0.0);

    if (trials == 0) trials = 1;

    // Calculate baseline error (MSE)
    auto getError = [&](const std::vector<std::vector<double>>& dataX,
                        std::vector<std::vector<double>>* predsOut = nullptr) {
        double totalError = 0.0;
        if (predsOut) predsOut->assign(dataX.size(), std::vector<double>(outDim, 0.0));
        for (size_t i = 0; i < dataX.size(); ++i) {
            auto p = predict(dataX[i]);
            for (size_t j = 0; j < outDim && j < p.size(); ++j) {
                double diff = p[j] - Y[i][j];
                totalError += diff * diff;
                if (predsOut) {
                    (*predsOut)[i][j] = p[j];
                }
            }
        }
        return totalError / (dataX.size() * outDim);
    };

    std::vector<std::vector<double>> baselinePred;
    double baselineError = getError(X, &baselinePred);

    for (size_t f = 0; f < numFeatures; ++f) {
        double featureErrorSum = 0.0;
        double sensitivitySum = 0.0;
        for (size_t t = 0; t < trials; ++t) {
            std::vector<std::vector<double>> shuffledX = X;
            std::vector<double> featureVals;
            for (const auto& row : X) featureVals.push_back(row[f]);
            std::shuffle(featureVals.begin(), featureVals.end(), rng);
            for (size_t i = 0; i < X.size(); ++i) shuffledX[i][f] = featureVals[i];

            std::vector<std::vector<double>> shuffledPred;
            featureErrorSum += getError(shuffledX, &shuffledPred);
            for (size_t i = 0; i < shuffledPred.size(); ++i) {
                for (size_t j = 0; j < outDim; ++j) {
                    sensitivitySum += std::abs(shuffledPred[i][j] - baselinePred[i][j]);
                }
            }
        }

        // Importance = how much error increased when feature was destroyed
        importances[f] = (featureErrorSum / (double)trials) - baselineError;
        if (importances[f] < 0) importances[f] = 0; // Clip noise
        fallbackSensitivity[f] = sensitivitySum / static_cast<double>(trials * X.size() * outDim);
    }

    // Normalize and blend loss delta with prediction sensitivity.
    double sumLoss = std::accumulate(importances.begin(), importances.end(), 0.0);
    double sumSens = std::accumulate(fallbackSensitivity.begin(), fallbackSensitivity.end(), 0.0);

    std::vector<double> lossNorm = importances;
    std::vector<double> sensNorm = fallbackSensitivity;
    if (sumLoss > 1e-12) {
        for (double& v : lossNorm) v /= sumLoss;
    } else {
        std::fill(lossNorm.begin(), lossNorm.end(), 0.0);
    }
    if (sumSens > 1e-12) {
        for (double& v : sensNorm) v /= sumSens;
    } else {
        std::fill(sensNorm.begin(), sensNorm.end(), 0.0);
    }

    if (sumLoss <= 1e-12 && sumSens > 1e-12) {
        importances = sensNorm;
    } else if (!importances.empty()) {
        importances.assign(importances.size(), 0.0);
        for (size_t i = 0; i < importances.size(); ++i) {
            const double lossWeight = (sumLoss > 1e-12) ? 0.8 : 0.0;
            const double sensWeight = (sumSens > 1e-12) ? (1.0 - lossWeight) : 0.0;
            importances[i] = lossWeight * lossNorm[i] + sensWeight * sensNorm[i];
        }
    }

    const double sum = std::accumulate(importances.begin(), importances.end(), 0.0);
    if (sum > 1e-12) {
        for (double& v : importances) v /= sum;
    } else if (!importances.empty()) {
        const double uniform = 1.0 / static_cast<double>(importances.size());
        for (double& v : importances) v = uniform;
    }

    return importances;
}

void NeuralNet::saveModelBinary(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw Seldon::IOException("Could not open " + filename + " for writing");

    const char sig[] = "SELDON_BIN_V2";
    out.write(sig, sizeof(sig));

    writeLE(out, kModelFormatVersion);
    uint64_t checksum = kChecksumOffsetBasis;
    updateChecksum(checksum, kModelFormatVersion);

    uint64_t topSize = static_cast<uint64_t>(topology.size());
    writeLE(out, topSize);
    updateChecksum(checksum, topSize);
    for (size_t t : topology) {
        uint64_t layerWidth = static_cast<uint64_t>(t);
        writeLE(out, layerWidth);
        updateChecksum(checksum, layerWidth);
    }

    for (const auto& layer : m_layers) {
        int32_t act = static_cast<int32_t>(layer.activation());
        writeLE(out, act);
        updateChecksum(checksum, act);
    }

    uint64_t inS = static_cast<uint64_t>(inputScales.size());
    uint64_t outS = static_cast<uint64_t>(outputScales.size());
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

    char sigV2[sizeof("SELDON_BIN_V2")];
    in.read(sigV2, sizeof(sigV2));
    if (!in) {
        throw Seldon::NeuralNetException("Failed to read model signature from " + filename);
    }
    if (std::string(sigV2) == "SELDON_BIN_V2") {
        uint32_t version = 0;
        readLE(in, version);
        if (version != kModelFormatVersion) {
            throw Seldon::NeuralNetException("Unsupported model version in file: " + filename);
        }

        uint64_t computedChecksum = kChecksumOffsetBasis;
        updateChecksum(computedChecksum, version);

        uint64_t topSize;
        readLE(in, topSize);
        updateChecksum(computedChecksum, topSize);
        if (topSize < 2 || topSize > 1000000) {
            throw Seldon::NeuralNetException("Invalid topology size in model file: " + filename);
        }
        std::vector<size_t> top(topSize);
        for (uint64_t i = 0; i < topSize; ++i) {
            uint64_t val;
            readLE(in, val);
            updateChecksum(computedChecksum, val);
            if (val == 0 || val > 1000000) {
                throw Seldon::NeuralNetException("Invalid layer width in model file: " + filename);
            }
            top[i] = static_cast<size_t>(val);
        }

        *this = NeuralNet(top);

        for (size_t i = 0; i < m_layers.size(); ++i) {
            int32_t act;
            readLE(in, act);
            updateChecksum(computedChecksum, act);
            m_layers[i].setActivation(static_cast<Activation>(act));
        }

        uint64_t inS, outS;
        readLE(in, inS);
        readLE(in, outS);
        updateChecksum(computedChecksum, inS);
        updateChecksum(computedChecksum, outS);
        
        inputScales.resize(inS);
        for (uint64_t i = 0; i < inS; ++i) {
            readLE(in, inputScales[i].min);
            readLE(in, inputScales[i].max);
            updateChecksum(computedChecksum, inputScales[i].min);
            updateChecksum(computedChecksum, inputScales[i].max);
        }
        outputScales.resize(outS);
        for (uint64_t i = 0; i < outS; ++i) {
            readLE(in, outputScales[i].min);
            readLE(in, outputScales[i].max);
            updateChecksum(computedChecksum, outputScales[i].min);
            updateChecksum(computedChecksum, outputScales[i].max);
        }

        for (size_t l = 1; l < m_layers.size(); ++l) {
            auto& layer = m_layers[l];
            for (double& b : layer.biases()) {
                readLE(in, b);
                updateChecksum(computedChecksum, b);
            }
            for (double& w : layer.weights()) {
                readLE(in, w);
                updateChecksum(computedChecksum, w);
            }
        }

        uint64_t storedChecksum = 0;
        readLE(in, storedChecksum);
        if (storedChecksum != computedChecksum) {
            throw Seldon::NeuralNetException("Model checksum mismatch (corrupt file): " + filename);
        }
        return;
    } 
    
    throw Seldon::NeuralNetException("Unsupported or invalid binary model signature in " + filename);
}
