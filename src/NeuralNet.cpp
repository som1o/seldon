#include "NeuralNet.h"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include "SeldonExceptions.h"

namespace {
    bool isLittleEndian() {
        uint16_t number = 0x1;
        char *numPtr = reinterpret_cast<char*>(&number);
        return (numPtr[0] == 1);
    }

    template<typename T>
    void swapEndian(T &val) {
        char *ptr = reinterpret_cast<char*>(&val);
        int size = sizeof(T);
        for (int i = 0; i < size / 2; i++) {
            std::swap(ptr[i], ptr[size - 1 - i]);
        }
    }

    template<typename T>
    void writeLE(std::ostream& out, T val) {
        if (!isLittleEndian()) {
            swapEndian(val);
        }
        out.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    template<typename T>
    void readLE(std::istream& in, T& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(T));
        if (!isLittleEndian()) {
            swapEndian(val);
        }
    }
}

NeuralNet::NeuralNet(std::vector<size_t> topology) : topology(topology) {
    std::random_device rd;
    rng.seed(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (size_t l = 0; l < topology.size(); ++l) {
        Layer layer;
        layer.size = topology[l];
        layer.outputs.assign(layer.size, 0.0);
        layer.biases.assign(layer.size, 0.0);
        layer.gradients.assign(layer.size, 0.0);
        layer.m_biases.assign(layer.size, 0.0);
        layer.v_biases.assign(layer.size, 0.0);
        layer.dropMask.assign(layer.size, false);
        layer.activation = (l == topology.size() - 1) ? Activation::SIGMOID : Activation::RELU;

        if (l > 0) {
            size_t prevSize = topology[l - 1];
            size_t weightCount = layer.size * prevSize;
            
            // Xavier/He initialization based on activation
            double stddev = 0.1;
            if (layer.activation == Activation::RELU) {
                stddev = std::sqrt(2.0 / prevSize); // He initialization
            } else {
                stddev = std::sqrt(1.0 / prevSize); // Xavier initialization
            }
            
            std::normal_distribution<> weightDis(0.0, stddev);
            layer.weights.reserve(weightCount);
            for (size_t i = 0; i < weightCount; ++i) {
                layer.weights.push_back(weightDis(rng));
            }

            layer.m_weights.assign(weightCount, 0.0);
            layer.v_weights.assign(weightCount, 0.0);
            for (size_t i = 0; i < layer.size; ++i) {
                layer.biases[i] = 0.01; // Small positive bias to prevent dead neurons
            }
        }
        m_layers.push_back(layer);
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
    size_t inputSize = std::min(inputValues.size(), m_layers[0].size);
    for (size_t i = 0; i < inputSize; ++i) {
        m_layers[0].outputs[i] = inputValues[i];
    }

    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t l = 1; l < m_layers.size(); ++l) {
        Layer& current = m_layers[l];
        Layer& prev = m_layers[l - 1];
        current.activation = (l == m_layers.size() - 1) ? outputAct : act;

        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t n = 0; n < current.size; ++n) {
            double sum = current.biases[n];
            size_t weightOffset = n * prev.size;
            #ifdef USE_OPENMP
            #pragma omp simd reduction(+:sum)
            #endif
            for (size_t pn = 0; pn < prev.size; ++pn) {
                sum += prev.outputs[pn] * current.weights[weightOffset + pn];
            }
            
            double out = activate(sum, current.activation);

            if (isTraining && l < m_layers.size() - 1 && dropoutRate > 0.0) {
                // Thread-local RNG to avoid critical section contention
                static thread_local std::mt19937 t_rng(std::random_device{}());
                std::uniform_real_distribution<> t_dis(0.0, 1.0);

                if (t_dis(t_rng) < dropoutRate) {
                    current.outputs[n] = 0.0;
                    current.dropMask[n] = true;
                } else {
                    current.outputs[n] = out / (1.0 - dropoutRate);
                    current.dropMask[n] = false;
                }
            } else {
                current.outputs[n] = out;
                current.dropMask[n] = false;
            }
        }
    }
}

void NeuralNet::computeGradients(const std::vector<double>& targetValues, LossFunction loss) {
    Layer& outputLayer = m_layers.back();
    for (size_t n = 0; n < outputLayer.size; ++n) {
        if (loss == LossFunction::CROSS_ENTROPY && outputLayer.activation == Activation::SIGMOID) {
            // Simplified gradient for Cross-Entropy + Sigmoid
            outputLayer.gradients[n] = outputLayer.outputs[n] - targetValues[n];
        } else {
            double delta = outputLayer.outputs[n] - targetValues[n]; // MSE Gradient
            outputLayer.gradients[n] = delta * activateDerivative(outputLayer.outputs[n], outputLayer.activation);
        }
    }

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t l = m_layers.size() - 2; l > 0; --l) {
        Layer& hidden = m_layers[l];
        Layer& next = m_layers[l + 1];

        for (size_t n = 0; n < hidden.size; ++n) {
            if (hidden.dropMask[n]) {
                hidden.gradients[n] = 0.0;
                continue;
            }
            double dow = 0.0;
            #ifdef USE_OPENMP
            #pragma omp simd reduction(+:dow)
            #endif
            for (size_t nn = 0; nn < next.size; ++nn) {
                dow += next.weights[nn * hidden.size + n] * next.gradients[nn];
            }
            hidden.gradients[n] = dow * activateDerivative(hidden.outputs[n], hidden.activation);
        }
    }
}

void NeuralNet::applyOptimization(const Hyperparameters& hp, size_t t_step) {
    const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    for (size_t l = 1; l < m_layers.size(); ++l) {
        Layer& current = m_layers[l];
        Layer& prev = m_layers[l - 1];

        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t n = 0; n < current.size; ++n) {
            // Update Bias
            double grad_bias = current.gradients[n];
            if (hp.optimizer == Optimizer::ADAM) {
                current.m_biases[n] = beta1 * current.m_biases[n] + (1.0 - beta1) * grad_bias;
                current.v_biases[n] = beta2 * current.v_biases[n] + (1.0 - beta2) * grad_bias * grad_bias;
                double m_hat = current.m_biases[n] / (1.0 - std::pow(beta1, t_step));
                double v_hat = current.v_biases[n] / (1.0 - std::pow(beta2, t_step));
                current.biases[n] -= hp.learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
            } else {
                current.biases[n] -= hp.learningRate * grad_bias;
            }

            size_t weightOffset = n * prev.size;
            #ifdef USE_OPENMP
            #pragma omp simd
            #endif
            for (size_t pn = 0; pn < prev.size; ++pn) {
                double grad_w = current.gradients[n] * prev.outputs[pn];
                grad_w += hp.l2Lambda * current.weights[weightOffset + pn]; // Weight decay scaling

                if (hp.optimizer == Optimizer::ADAM) {
                    current.m_weights[weightOffset + pn] = beta1 * current.m_weights[weightOffset + pn] + (1.0 - beta1) * grad_w;
                    current.v_weights[weightOffset + pn] = beta2 * current.v_weights[weightOffset + pn] + (1.0 - beta2) * grad_w * grad_w;
                    double m_hat = current.m_weights[weightOffset + pn] / (1.0 - std::pow(beta1, t_step));
                    double v_hat = current.v_weights[weightOffset + pn] / (1.0 - std::pow(beta2, t_step));
                    current.weights[weightOffset + pn] -= hp.learningRate * (m_hat / (std::sqrt(v_hat) + epsilon));
                } else {
                    current.weights[weightOffset + pn] -= hp.learningRate * grad_w;
                }
            }
        }
    }
}

void NeuralNet::backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step) {
    computeGradients(targetValues, hp.loss);
    applyOptimization(hp, t_step);
}

void NeuralNet::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                      const Hyperparameters& hp,
                      const std::vector<ScaleInfo>& inScales,
                      const std::vector<ScaleInfo>& outScales) 
{
    if (X.empty()) return;
    this->inputScales = inScales;
    this->outputScales = outScales;

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

    for (size_t epoch = 0; epoch < hp.epochs; ++epoch) {
        if (epoch % std::max(size_t(1), hp.epochs / 10) == 0) {
            std::cout << "\r[Agent] Training Neural Lattice: [" << (epoch * 100 / hp.epochs) << "%] " << std::flush;
        }
        std::shuffle(indices.begin(), indices.end(), rng);

        double trainLoss = runEpoch(X, Y, indices, trainSize, hp, currentLR, t_step);
        double valLoss = valSize > 0 ? validate(X, Y, indices, trainSize, hp) : 0.0;

        if (hp.verbose && (epoch % std::max(size_t(1), hp.epochs / 10) == 0 || epoch == hp.epochs - 1)) {
            std::cout << "[Epoch " << epoch << "] Train Loss: " << trainLoss << " | Val Loss: " << valLoss << std::endl;
        }

        if (valSize > 0) {
            if (valLoss < (bestValLoss - hp.minDelta)) {
                bestValLoss = valLoss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= hp.earlyStoppingPatience) {
                    if (hp.verbose) std::cout << "\n[Agent] Early stopping triggered at epoch " << epoch << std::endl;
                    break;
                }
            }
        }
    }
    std::cout << "\r[Agent] Training Neural Lattice: [100%] Complete.          " << std::endl;
}

double NeuralNet::runEpoch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                           const std::vector<size_t>& indices, size_t trainSize, 
                           const Hyperparameters& hp, double& currentLR, size_t& t_step) {
    // Learning Rate Scheduling
    size_t batchesPerEpoch = std::max(size_t(1), trainSize / hp.batchSize);
    size_t epoch = t_step / batchesPerEpoch;
    if (epoch > 0 && epoch % hp.lrStepSize == 0 && t_step % batchesPerEpoch == 0) {
        currentLR *= hp.lrDecay;
        if (hp.verbose) std::cout << "\n[Agent] Learning rate decayed to " << currentLR << "\n";
    }

    Hyperparameters currentHp = hp;
    currentHp.learningRate = currentLR;

    double epochLoss = 0.0;
    for (size_t i = 0; i < trainSize; i += hp.batchSize) {
        size_t currentBatchEnd = std::min(i + hp.batchSize, trainSize);
        epochLoss += runBatch(X, Y, indices, i, currentBatchEnd, currentHp, t_step);
    }
    double denom = static_cast<double>(trainSize * m_layers.back().size);
    return denom > 0 ? epochLoss / denom : 0.0;
}

double NeuralNet::runBatch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                            const std::vector<size_t>& indices, size_t batchStart, size_t batchEnd, 
                            const Hyperparameters& hp, size_t& t_step) {
    double batchLoss = 0.0;
    t_step++;
    for (size_t b = batchStart; b < batchEnd; ++b) {
        feedForward(X[indices[b]], hp.activation, hp.outputActivation, true, hp.dropoutRate);
        backpropagate(Y[indices[b]], hp, t_step);

        for (size_t j = 0; j < Y[indices[b]].size(); ++j) {
            double actual = Y[indices[b]][j];
            double pred = m_layers.back().outputs[j];
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
        feedForward(X[indices[i]], hp.activation, hp.outputActivation, false);
        for (size_t j = 0; j < Y[indices[i]].size(); ++j) {
            double actual = Y[indices[i]][j];
            double pred = m_layers.back().outputs[j];
            if (hp.loss == LossFunction::CROSS_ENTROPY) {
                pred = std::clamp(pred, 1e-15, 1.0 - 1e-15);
                valLossSum -= (actual * std::log(pred) + (1.0 - actual) * std::log(1.0 - pred));
            } else {
                valLossSum += (actual - pred) * (actual - pred);
            }
        }
    }
    double denom = static_cast<double>(valSize * m_layers.back().size);
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

    feedForward(scaledInput, m_layers[1].activation, m_layers.back().activation, false);
    
    std::vector<double> result;
    for (size_t i = 0; i < m_layers.back().size; ++i) {
        double val = m_layers.back().outputs[i];
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
    std::vector<double> importances(numFeatures, 0.0);

    // Calculate baseline error (MSE)
    auto getError = [&](const std::vector<std::vector<double>>& dataX) {
        double totalError = 0.0;
        for (size_t i = 0; i < dataX.size(); ++i) {
            auto p = predict(dataX[i]);
            for (size_t j = 0; j < p.size(); ++j) {
                double diff = p[j] - Y[i][j];
                totalError += diff * diff;
            }
        }
        return totalError / (dataX.size() * Y[0].size());
    };

    double baselineError = getError(X);

    for (size_t f = 0; f < numFeatures; ++f) {
        double featureErrorSum = 0.0;
        for (size_t t = 0; t < trials; ++t) {
            std::vector<std::vector<double>> shuffledX = X;
            std::vector<double> featureVals;
            for (const auto& row : X) featureVals.push_back(row[f]);
            std::shuffle(featureVals.begin(), featureVals.end(), rng);
            for (size_t i = 0; i < X.size(); ++i) shuffledX[i][f] = featureVals[i];
            
            featureErrorSum += getError(shuffledX);
        }
        // Importance = how much error increased when feature was destroyed
        importances[f] = (featureErrorSum / (double)trials) - baselineError;
        if (importances[f] < 0) importances[f] = 0; // Clip noise
    }

    // Normalize to percentages
    double sum = 0;
    for (double v : importances) sum += v;
    if (sum > 0) {
        for (double& v : importances) v /= sum;
    }

    return importances;
}

void NeuralNet::saveModelBinary(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw Seldon::IOException("Could not open " + filename + " for writing");

    const char sig[] = "SELDON_BIN_V2";
    out.write(sig, sizeof(sig));

    uint64_t topSize = static_cast<uint64_t>(topology.size());
    writeLE(out, topSize);
    for (size_t t : topology) {
        writeLE(out, static_cast<uint64_t>(t));
    }

    for (const auto& layer : m_layers) {
        int32_t act = static_cast<int32_t>(layer.activation);
        writeLE(out, act);
    }

    uint64_t inS = static_cast<uint64_t>(inputScales.size());
    uint64_t outS = static_cast<uint64_t>(outputScales.size());
    writeLE(out, inS);
    writeLE(out, outS);
    
    for (const auto& s : inputScales) {
        writeLE(out, s.min);
        writeLE(out, s.max);
    }
    for (const auto& s : outputScales) {
        writeLE(out, s.min);
        writeLE(out, s.max);
    }

    for (size_t l = 1; l < m_layers.size(); ++l) {
        const auto& layer = m_layers[l];
        for (double b : layer.biases) writeLE(out, b);
        for (double w : layer.weights) writeLE(out, w);
    }
}

void NeuralNet::loadModelBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw Seldon::IOException("Could not open " + filename + " for reading");

    char sigV2[sizeof("SELDON_BIN_V2")];
    in.read(sigV2, sizeof(sigV2));
    if (std::string(sigV2) == "SELDON_BIN_V2") {
        uint64_t topSize;
        readLE(in, topSize);
        std::vector<size_t> top(topSize);
        for (uint64_t i = 0; i < topSize; ++i) {
            uint64_t val;
            readLE(in, val);
            top[i] = static_cast<size_t>(val);
        }

        *this = NeuralNet(top);

        for (size_t i = 0; i < m_layers.size(); ++i) {
            int32_t act;
            readLE(in, act);
            m_layers[i].activation = static_cast<Activation>(act);
        }

        uint64_t inS, outS;
        readLE(in, inS);
        readLE(in, outS);
        
        inputScales.resize(inS);
        for (uint64_t i = 0; i < inS; ++i) {
            readLE(in, inputScales[i].min);
            readLE(in, inputScales[i].max);
        }
        outputScales.resize(outS);
        for (uint64_t i = 0; i < outS; ++i) {
            readLE(in, outputScales[i].min);
            readLE(in, outputScales[i].max);
        }

        for (size_t l = 1; l < m_layers.size(); ++l) {
            auto& layer = m_layers[l];
            for (double& b : layer.biases) readLE(in, b);
            for (double& w : layer.weights) readLE(in, w);
        }
        return;
    } 
    
    throw Seldon::NeuralNetException("Unsupported or invalid binary model signature in " + filename);
}
