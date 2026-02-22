#pragma once
#include <vector>
#include <string>
#include <random>
#include <map>

// Advanced, dependency-free Dense Feed-Forward Neural Network
class NeuralNet {
public:
    enum class Activation { SIGMOID, RELU, TANH, LINEAR };
    enum class LossFunction { MSE, CROSS_ENTROPY };
    enum class Optimizer { SGD, ADAM };

    struct ScaleInfo {
        double min;
        double max;
    };

    struct Hyperparameters {
        double learningRate = 0.001;
        double lrDecay = 0.5;         // Decay factor
        int lrStepSize = 100;          // Decay every N epochs
        size_t epochs = 100;
        size_t batchSize = 32;
        Activation activation = Activation::RELU;
        Activation outputActivation = Activation::SIGMOID; 
        LossFunction loss = LossFunction::MSE;
        Optimizer optimizer = Optimizer::ADAM;
        double l2Lambda = 0.001;
        double dropoutRate = 0.0;
        double valSplit = 0.2;
        int earlyStoppingPatience = 10;
        double minDelta = 1e-4;       // Minimum improvement for early stopping
        double gradientClipNorm = 5.0;
        uint32_t seed = 1337;
        bool verbose = true;
    };

    NeuralNet(std::vector<size_t> topology);
    void setSeed(uint32_t seed);
    
    // Train the network using advanced features
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
               const Hyperparameters& hp,
               const std::vector<ScaleInfo>& inputScales = {},
               const std::vector<ScaleInfo>& outputScales = {});

    // Forward pass prediction (handles internal scaling if scales are loaded)
    std::vector<double> predict(const std::vector<double>& inputValues);
    const std::vector<double>& getTrainLossHistory() const { return trainLossHistory; }
    const std::vector<double>& getValLossHistory() const { return valLossHistory; }

    // Explainability
    std::vector<double> calculateFeatureImportance(const std::vector<std::vector<double>>& X, 
                                                   const std::vector<std::vector<double>>& Y,
                                                   size_t trials = 5);

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
    struct Layer {
        size_t size;
        std::vector<double> outputs;
        std::vector<double> biases;
        std::vector<double> weights; // Contiguous: weights for all neurons in this layer
        std::vector<double> gradients;
        
        // Adam buffers
        std::vector<double> m_weights, v_weights;
        std::vector<double> m_biases, v_biases;
        
        std::vector<bool> dropMask;
        
        Activation activation; // Per-layer activation
    };

    void feedForward(const std::vector<double>& inputValues, bool isTraining, double dropoutRate = 0.0);
    void backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step);
    
    void computeGradients(const std::vector<double>& targetValues, LossFunction loss);
    void applyOptimization(const Hyperparameters& hp, size_t t_step);
    
    double activate(double x, Activation act);
    double activateDerivative(double out, Activation act);

    // Decomposition helpers for train
    double runEpoch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                    const std::vector<size_t>& indices, size_t trainSize, 
                    const Hyperparameters& hp, double& currentLR, size_t& t_step);
    double runBatch(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                     const std::vector<size_t>& indices, size_t batchStart, size_t batchEnd, 
                     const Hyperparameters& hp, size_t& t_step);
    double validate(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
                    const std::vector<size_t>& indices, size_t trainSize, const Hyperparameters& hp);

    std::vector<Layer> m_layers;
    std::vector<size_t> topology;
    std::vector<ScaleInfo> inputScales;
    std::vector<ScaleInfo> outputScales;
    std::vector<double> trainLossHistory;
    std::vector<double> valLossHistory;
    uint32_t seedState = 1337;
    uint64_t forwardCounter = 0;

    // Persistent RNG for performance and consistency
    std::mt19937 rng;
};
