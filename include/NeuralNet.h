#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include "NeuralLayer.h"

// Advanced, dependency-free Dense Feed-Forward Neural Network
class NeuralNet {
public:
    using Scalar = NeuralScalar;
    using Activation = NeuralActivation;
    enum class LossFunction { MSE, CROSS_ENTROPY };
    using Optimizer = NeuralOptimizer;

    struct ScaleInfo {
        double min;
        double max;
    };

    struct Hyperparameters {
        double learningRate = 0.001;
        double lrDecay = 0.5;
        int lrPlateauPatience = 5;
        int lrCooldownEpochs = 2;
        int maxLrReductions = 8;
        double minLearningRate = 1e-6;
        size_t lrWarmupEpochs = 5;
        bool useCosineAnnealing = true;
        bool useCyclicalLr = false;
        size_t lrCycleEpochs = 24;
        double lrScheduleMinFactor = 0.15;
        size_t epochs = 100;
        size_t batchSize = 32;
        Activation activation = Activation::GELU;
        Activation outputActivation = Activation::SIGMOID; 
        LossFunction loss = LossFunction::MSE;
        Optimizer optimizer = Optimizer::ADAM;
        Optimizer lookaheadFastOptimizer = Optimizer::ADAM;
        size_t lookaheadSyncPeriod = 5;
        double lookaheadAlpha = 0.5;
        double l2Lambda = 0.001;
        double categoricalInputL2Boost = 3.0;
        double dropoutRate = 0.0;
        bool useBatchNorm = true;
        double batchNormMomentum = 0.95;
        double batchNormEpsilon = 1e-5;
        bool useLayerNorm = true;
        double layerNormEpsilon = 1e-5;
        double valSplit = 0.2;
        bool useValidationLossEma = true;
        double validationLossEmaBeta = 0.6;
        int earlyStoppingPatience = 10;
        double minDelta = 1e-4;       // Minimum improvement for early stopping
        double gradientClipNorm = 5.0;
        bool adaptiveGradientClipping = true;
        double adaptiveClipBeta = 0.90;
        double adaptiveClipMultiplier = 1.6;
        double adaptiveClipMin = 1.0;
        double gradientNoiseStd = 0.01;
        double gradientNoiseDecay = 0.995;
        bool useEmaWeights = true;
        double emaDecay = 0.995;
        double labelSmoothing = 0.02;
        size_t gradientAccumulationSteps = 2;
        bool incrementalMode = false;
        bool useDiskStreaming = false;
        std::string inputBinaryPath;
        std::string targetBinaryPath;
        size_t streamingRows = 0;
        size_t streamingInputDim = 0;
        size_t streamingOutputDim = 0;
        bool useMemoryMappedInput = true;
        size_t streamingChunkRows = 2048;
        size_t importanceMaxRows = 5000;
        bool importanceParallel = true;
        uint32_t seed = 1337;
        bool verbose = true;
    };

    struct UncertaintyEstimate {
        std::vector<double> mean;
        std::vector<double> stddev;
        std::vector<double> ciLow;
        std::vector<double> ciHigh;
    };

    NeuralNet(std::vector<size_t> topology);
    void setSeed(uint32_t seed);
    void setInputL2Scales(const std::vector<double>& scales);
    
    // Train the network using advanced features
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
               const Hyperparameters& hp,
               const std::vector<ScaleInfo>& inputScales = {},
               const std::vector<ScaleInfo>& outputScales = {});
    void trainIncremental(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y,
                          const Hyperparameters& hp,
                          size_t chunkRows,
                          const std::vector<ScaleInfo>& inputScales = {},
                          const std::vector<ScaleInfo>& outputScales = {});
    void trainIncrementalFromBinary(const std::string& inputBinaryPath,
                                    const std::string& targetBinaryPath,
                                    size_t rows,
                                    size_t inputDim,
                                    size_t outputDim,
                                    const Hyperparameters& hp,
                                    size_t chunkRows,
                                    bool useMemoryMap,
                                    const std::vector<ScaleInfo>& inputScales = {},
                                    const std::vector<ScaleInfo>& outputScales = {});

    // Forward pass prediction (handles internal scaling if scales are loaded)
    std::vector<double> predict(const std::vector<double>& inputValues);
    UncertaintyEstimate predictWithUncertainty(const std::vector<double>& inputValues,
                                               size_t samples,
                                               double dropoutRate);
    const std::vector<double>& getTrainLossHistory() const noexcept { return trainLossHistory; }
    const std::vector<double>& getValLossHistory() const noexcept { return valLossHistory; }
    const std::vector<double>& getGradientNormHistory() const noexcept { return gradientNormHistory; }
    const std::vector<double>& getWeightStdHistory() const noexcept { return weightStdHistory; }
    const std::vector<double>& getWeightMeanAbsHistory() const noexcept { return weightMeanAbsHistory; }

    // Explainability
    std::vector<double> calculateFeatureImportance(const std::vector<std::vector<double>>& X, 
                                                   const std::vector<std::vector<double>>& Y,
                                                   size_t trials = 5,
                                                   size_t maxRows = 5000,
                                                   bool parallel = true);
    std::vector<double> calculateIntegratedGradients(const std::vector<std::vector<double>>& X,
                                                     size_t steps = 16,
                                                     size_t maxRows = 2000);

    /**
     * @brief Saves the current model state (weights, biases, and scale info) to a binary file.
     * @param filename Path to the output file.
     */
    void saveModelBinary(const std::string& filename) const;

    /**
     * @brief Loads a model state from a binary file.
     * @param filename Path to the Seldon binary model file.
     */
    void loadModelBinary(const std::string& filename);

private:
    void ensureBatchWorkspace();
    void feedForward(const std::vector<double>& inputValues, bool isTraining, double dropoutRate = 0.0);
    void backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step);
    
    void computeGradients(const std::vector<double>& targetValues, LossFunction loss);
    void applyOptimization(const Hyperparameters& hp, size_t t_step);
    
    // Decomposition helpers for train
    double runEpoch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                    const std::vector<size_t>& indices, size_t trainSize, 
                    const Hyperparameters& hp, size_t& t_step);
    double runBatch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                     const std::vector<size_t>& indices, size_t batchStart, size_t batchEnd, 
                     const Hyperparameters& hp, size_t& t_step);
    double validate(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                    const std::vector<size_t>& indices, size_t trainSize, const Hyperparameters& hp);
    void updateEmaWeights(double decay);
    void applyEmaWeights();

    std::vector<DenseLayer> m_layers;
    std::vector<size_t> topology;
    std::vector<ScaleInfo> inputScales;
    std::vector<ScaleInfo> outputScales;
    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;
    std::vector<double> gradientNormHistory;
    std::vector<double> weightStdHistory;
    std::vector<double> weightMeanAbsHistory;
    uint32_t seedState = 1337;
    uint64_t forwardCounter = 0;
    double lastTrainingDropoutRate = 0.0;
    std::vector<double> m_inputL2Scales;
    bool m_useBatchNorm = true;
    double m_batchNormMomentum = 0.95;
    double m_batchNormEpsilon = 1e-5;
    bool m_useLayerNorm = true;
    double m_layerNormEpsilon = 1e-5;
    bool m_lookaheadInitialized = false;
    std::vector<std::vector<Scalar>> m_lookaheadSlowWeights;
    std::vector<std::vector<Scalar>> m_lookaheadSlowBiases;
    bool m_emaInitialized = false;
    std::vector<std::vector<Scalar>> m_emaWeights;
    std::vector<std::vector<Scalar>> m_emaBiases;
    double m_runningGradNormEma = 0.0;
    bool m_runningGradNormReady = false;
    std::vector<std::vector<Scalar>> m_gradBAccumWorkspace;
    std::vector<std::vector<Scalar>> m_gradWAccumWorkspace;
    std::vector<std::vector<Scalar>> m_gradBWorkWorkspace;
    std::vector<std::vector<Scalar>> m_gradWWorkWorkspace;
    std::vector<double> m_targetWork;

    // Persistent RNG for performance and consistency
    std::mt19937 rng;
};
