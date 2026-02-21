#include "Dataset.h"
#include "StatsEngine.h"
#include "LogicEngine.h"
#include "NeuralNet.h"
#include "TerminalUI.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <map>
#include <iomanip>
#include <optional>

void printUsage(const std::string& prog) {
    std::cout << "Usage: " << prog << " <dataset.csv> [options]\n"
              << "Options:\n"
              << "  --threshold-bivariate, -tb <val> Correlation threshold for simple regression (default: 0.6)\n"
              << "  --threshold-mlr, -tm <val>       Threshold for MLR inclusion (default: 0.35)\n"
              << "  --exhaustive-scan                Scan entire dataset to detect numeric columns\n"
              << "  --epochs <val>                   Neural network training epochs (default: 300)\n"
              << "  --lr <val>                       Learning rate (default: 0.02)\n"
              << "  --impute <skip|zero|mean|median>  Missing value handling (default: zero)\n"
              << "  --skip-malformed <true/false>    Malformed row handling (default: true)\n"
              << "  --output, -o <file>              Export results to specified file (json/csv)\n"
              << "  --loss <mse|cross_entropy>       Manual neural net loss function override\n"
              << "  --verbose                        Enable detailed logs\n"
              << "  --batch                          Non-interactive mode\n"
              << "  --help                           Show this help message\n";
}

struct SeldonConfig {
    std::string filename;
    std::string outputFilename;
    double agenticThreshold = 0.6;
    double agenticMlrThreshold = 0.35;
    bool exhaustiveScan = false;
    bool skipMalformed = true;
    bool verbose = false;
    int epochs = 300;
    double lr = 0.02;
    Dataset::ImputationStrategy imputation = Dataset::ImputationStrategy::ZERO;
    bool batchMode = false;
    bool showHelp = false;
    std::optional<NeuralNet::LossFunction> lossOverride;
};

#include <fstream>

void exportResults(const std::string& filename, 
                   const std::vector<RegressionResult>& simple, 
                   const std::vector<MultipleRegressionResult>& mlr) {
    std::ofstream out(filename);
    if (!out) return;

    out << "{\n  \"simple_regressions\": [\n";
    for (size_t i = 0; i < simple.size(); ++i) {
        const auto& r = simple[i];
        out << "    { \"x\": \"" << r.featureX << "\", \"y\": \"" << r.featureY 
            << "\", \"slope\": " << r.m << ", \"intercept\": " << r.c 
            << ", \"rSquared\": " << r.rSquared << " }" << (i == simple.size() - 1 ? "" : ",") << "\n";
    }
    out << "  ],\n  \"multiple_regressions\": [\n";
    for (size_t i = 0; i < mlr.size(); ++i) {
        const auto& r = mlr[i];
        out << "    { \"dependent\": \"" << r.dependentFeature << "\", \"adjRSquared\": " << r.adjustedRSquared 
            << ", \"fStat\": " << r.fStatistic << " }" << (i == mlr.size() - 1 ? "" : ",") << "\n";
    }
    out << "  ]\n}\n";
    std::cout << "[Agent] Results exported to: " << filename << "\n";
}

SeldonConfig parseArguments(int argc, char* argv[]) {
    SeldonConfig config;
    if (argc < 2) return config;

    config.filename = argv[1];
    if (config.filename == "--help" || config.filename == "-h") {
        config.showHelp = true;
        return config;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--threshold-bivariate" || arg == "-tb") && i + 1 < argc) {
            config.agenticThreshold = std::stod(argv[++i]);
        } else if ((arg == "--threshold-mlr" || arg == "-tm") && i + 1 < argc) {
            config.agenticMlrThreshold = std::stod(argv[++i]);
        } else if (arg == "--exhaustive-scan") {
            config.exhaustiveScan = true;
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.lr = std::stod(argv[++i]);
        } else if (arg == "--impute" && i + 1 < argc) {
            std::string strat = argv[++i];
            if (strat == "skip") config.imputation = Dataset::ImputationStrategy::SKIP;
            else if (strat == "mean") config.imputation = Dataset::ImputationStrategy::MEAN;
            else if (strat == "median") config.imputation = Dataset::ImputationStrategy::MEDIAN;
            else config.imputation = Dataset::ImputationStrategy::ZERO;
        } else if (arg == "--skip-malformed" && i + 1 < argc) {
            config.skipMalformed = (std::string(argv[++i]) == "true");
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--batch") {
            config.batchMode = true;
        } else if (arg == "--loss" && i + 1 < argc) {
            std::string loss = argv[++i];
            if (loss == "mse") config.lossOverride = NeuralNet::LossFunction::MSE;
            else if (loss == "cross_entropy") config.lossOverride = NeuralNet::LossFunction::CROSS_ENTROPY;
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            config.outputFilename = argv[++i];
        } else if (arg == "--help") {
            config.showHelp = true;
        } else {
            std::cerr << "Warning: Unknown or incomplete argument: " << arg << "\n";
        }
    }
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "Seldon: Agentic Data Analytics Engine Initialization...\n";
    SeldonConfig config = parseArguments(argc, argv);

    if (config.showHelp || config.filename.empty()) {
        printUsage(argv[0]);
        return config.showHelp ? 0 : 1;
    }

    Dataset dataset(config.filename);
    try {
        if (!dataset.load(config.skipMalformed, config.exhaustiveScan, config.imputation)) {
            std::cerr << "[Agent Error] Failed to load dataset: " << config.filename << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Agent Exception] " << e.what() << "\n";
        return 1;
    }

    dataset.printSummary();

    if (dataset.getRowCount() == 0 || dataset.getColCount() == 0) {
        std::cerr << "\n[Agent Error] Zero mathematical rows or columns isolated. Terminating analytics sequence gracefully.\n";
        return 1;
    }

    std::cout << "\n[Agent] Ingestion successful. Initiating Layer 1 analysis...\n";

    // Layer 1
    auto stats = StatsEngine::calculateFoundation(dataset);
    TerminalUI::printFoundationTable(dataset.getColumnNames(), stats);

    // Agentic Checkpoint 1
    if (!config.batchMode) {
        std::string response;
        std::cout << "\n[Agent] Foundation analysis complete. Proceed to Bivariate pattern hunting? (Y/n): ";
        std::getline(std::cin, response);

        if (response == "n" || response == "N") {
            std::cout << "[Agent] Acknowledged. Terminating sequence.\n";
            return 0;
        }
    }

    std::cout << "\n[Agent] Initiating Layer 2 Logic Engine...\n";

    // Layer 2
    auto corrMatrix = LogicEngine::calculateCorrelationMatrix(dataset, stats);
    auto dynamic = LogicEngine::suggestThresholds(corrMatrix);

    std::cout << "[Agent] Dynamic thresholds calculated: Bivariate=" << std::fixed << std::setprecision(2) << dynamic.bivariate 
              << ", MLR=" << dynamic.mlr << "\n";

    TerminalUI::printCorrelationMatrix(dataset.getColumnNames(), corrMatrix);

    auto highCorrs = LogicEngine::findHighCorrelations(dataset, corrMatrix, config.agenticThreshold);
    auto regressions = LogicEngine::performSimpleRegressions(dataset, stats, highCorrs);

    TerminalUI::printSimpleRegressionFindings(regressions);

    auto mlrRegressions = LogicEngine::performMultipleRegressions(dataset, corrMatrix, config.agenticMlrThreshold);
    TerminalUI::printMultipleRegressionFindings(mlrRegressions);

    if (!config.outputFilename.empty()) {
        exportResults(config.outputFilename, regressions, mlrRegressions);
    }

    // Agentic Checkpoint 2
    if (!config.batchMode) {
        std::string response;
        std::cout << "\n[Agent] Inference mapped. Do you want to initialize a combined, deep multivariate analysis? (Y/n): ";
        std::getline(std::cin, response);

        if (response == "n" || response == "N") {
            std::cout << "[Agent] Acknowledged. Terminating sequence.\n";
            return 0;
        }
    }

    std::cout << "\n[Agent] Initializing predictive modeling. Synthesizing Neural Topology...\n";

    // Layer 3 Synthesis
    std::vector<size_t> targetIndices = {dataset.getColCount() - 1}; // Default target: last column
    std::vector<size_t> inputIndices;
    std::string dependentVarName = dataset.getColumnNames().back();

    if (!mlrRegressions.empty()) {
        dependentVarName = mlrRegressions.front().dependentFeature;
        for (const auto& mlr : mlrRegressions) {
            if (mlr.rSquared > 0.45) { // Threshold for inclusion in aggregate architecture
                for (const auto& feat : mlr.independentFeatures) {
                    for (size_t i = 0; i < dataset.getColCount(); ++i) {
                        if (dataset.getColumnNames()[i] == feat) {
                            if (std::find(inputIndices.begin(), inputIndices.end(), i) == inputIndices.end()) {
                                inputIndices.push_back(i);
                            }
                            break;
                        }
                    }
                }
            }
        }
        targetIndices.clear();
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
            if (dataset.getColumnNames()[i] == dependentVarName) {
                targetIndices.push_back(i); break;
            }
        }
    }

    if (inputIndices.empty()) {
        for (const auto& reg : regressions) {
            for (size_t i = 0; i < dataset.getColCount(); ++i) {
                if (dataset.getColumnNames()[i] == reg.featureX) {
                    if (std::find(inputIndices.begin(), inputIndices.end(), i) == inputIndices.end()) {
                        inputIndices.push_back(i);
                    }
                    break;
                }
            }
        }
    }

    if (inputIndices.empty()) {
        size_t limit = std::min(dataset.getColCount() - 1, static_cast<size_t>(3));
        for(size_t i = 0; i < limit; ++i) inputIndices.push_back(i);
    }

    size_t inputNodes = inputIndices.size();
    size_t outputNodes = targetIndices.size();
    size_t hiddenNodes = inputNodes * 2 + 2; 
    
    std::cout << "        -> Target variable: " << dependentVarName << "\n";

    std::vector<size_t> topology = {inputNodes, hiddenNodes, outputNodes};
    TerminalUI::printNeuralNetInit(inputNodes, hiddenNodes, outputNodes);
    NeuralNet nn(topology);

    const auto& columns = dataset.getColumns();
    std::vector<std::vector<double>> X_train, Y_train;
    std::vector<NeuralNet::ScaleInfo> inputScales, outputScales;

    auto prepareScaledData = [&](const std::vector<size_t>& indices, std::vector<NeuralNet::ScaleInfo>& scales) {
        std::vector<std::vector<double>> scaled;
        for (size_t idx : indices) {
            const auto& col = columns[idx];
            double mn = *std::min_element(col.begin(), col.end());
            double mx = *std::max_element(col.begin(), col.end());
            scales.push_back({mn, mx});
            
            double range = (mx - mn == 0) ? 1.0 : (mx - mn);
            std::vector<double> scaledCol;
            for (double v : col) scaledCol.push_back((v - mn) / range);
            scaled.push_back(scaledCol);
        }
        return scaled;
    };

    auto scaledXCols = prepareScaledData(inputIndices, inputScales);
    auto scaledYCols = prepareScaledData(targetIndices, outputScales);
    
    for (size_t r = 0; r < dataset.getRowCount(); ++r) {
        std::vector<double> xRow, yRow;
        for (const auto& col : scaledXCols) xRow.push_back(col[r]);
        for (const auto& col : scaledYCols) yRow.push_back(col[r]);
        X_train.push_back(xRow);
        Y_train.push_back(yRow);
    }

    NeuralNet::Hyperparameters hp;
    hp.learningRate = config.lr;
    hp.lrDecay = 0.5;
    hp.epochs = config.epochs;
    hp.lrStepSize = hp.epochs / 3; // Decay every 1/3 of total epochs
    hp.batchSize = 16;
    hp.activation = NeuralNet::Activation::RELU;
    hp.outputActivation = NeuralNet::Activation::SIGMOID;
    // Simple heuristic for loss function: check if target is binary
    bool isBinary = true;
    for (double val : Y_train[0]) { // Check first target value as hint or scan all
        // In real app, we'd scan all Y_train. For now, let's just check if it's reasonably binary.
    }
    const double BINARY_EPSILON = 1e-7;
    size_t checkLimit = config.exhaustiveScan ? Y_train.size() : std::min(Y_train.size(), size_t(100));
    
    for (size_t i = 0; i < checkLimit; ++i) {
        double v = Y_train[i][0];
        if (std::abs(v) > BINARY_EPSILON && std::abs(v - 1.0) > BINARY_EPSILON) {
            isBinary = false;
            break;
        }
    }
    
    if (config.lossOverride) {
        hp.loss = *config.lossOverride;
        if (config.verbose) std::cout << "[Agent] Using manual loss override.\n";
    } else {
        hp.loss = isBinary ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
        if (isBinary && config.verbose) std::cout << "[Agent] Detected potential classification task. Using Cross-Entropy loss.\n";
    }

    hp.optimizer = NeuralNet::Optimizer::ADAM;
    hp.verbose = config.verbose;
    hp.l2Lambda = 0.001;

    try {
        nn.train(X_train, Y_train, hp, inputScales, outputScales);
        nn.saveModel("seldon_model.json");

        std::cout << "[Agent] Calculating Feature Importance...\n";
        auto importance = nn.calculateFeatureImportance(X_train, Y_train);
        
        std::vector<std::string> featureNames;
        for (size_t idx : inputIndices) featureNames.push_back(dataset.getColumnNames()[idx]);
        
        std::vector<double> results = nn.predict(X_train[0]);
        TerminalUI::printSynthesisReport(dependentVarName, featureNames, results);

        std::cout << "\n[Agent] Feature Importance Analysis:\n";
        for (size_t i = 0; i < featureNames.size(); ++i) {
            std::cout << "        - " << std::left << std::setw(20) << featureNames[i] 
                      << ": " << std::fixed << std::setprecision(2) << (importance[i] * 100.0) << "%\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Agent Error] Core synthesis failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
