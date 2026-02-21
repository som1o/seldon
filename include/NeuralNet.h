#include <vector>
#include <string>

// Advanced, dependency-free Dense Feed-Forward Neural Network
class NeuralNet {
public:
    enum class Activation { SIGMOID, RELU, TANH };
    enum class Optimizer { SGD, ADAM };

    struct Hyperparameters {
        double learningRate = 0.001;
        size_t epochs = 100;
        size_t batchSize = 32;
        Activation activation = Activation::RELU;
        Activation outputActivation = Activation::SIGMOID; // Flexible output activation
        Optimizer optimizer = Optimizer::ADAM;
        double l2Lambda = 0.001;
        double dropoutRate = 0.0; // 0.0 to 1.0
        double valSplit = 0.2;    // 20% for validation
        int earlyStoppingPatience = 10;
        bool verbose = true;
    };

    NeuralNet(std::vector<size_t> topology);
    
    // Train the network using advanced features
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, const Hyperparameters& hp);

    // Forward pass prediction
    std::vector<double> predict(const std::vector<double>& inputValues);

    // Model Persistence
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);

private:
    struct Neuron {
        double outputVal;
        double bias;
        double gradient;
        std::vector<double> weights;
        
        // Adam optimizer buffers
        std::vector<double> m_weights, v_weights;
        double m_bias = 0.0, v_bias = 0.0;
        
        bool dropMask = false; // For Dropout
    };

    void feedForward(const std::vector<double>& inputValues, Activation act, Activation outputAct, bool isTraining, double dropoutRate = 0.0);
    void backpropagate(const std::vector<double>& targetValues, const Hyperparameters& hp, size_t t_step);
    
    // Activation functions
    double activate(double x, Activation act);
    double activateDerivative(double x, Activation act);

    std::vector<std::vector<Neuron>> layers;
    std::vector<size_t> topology;
};
