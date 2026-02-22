#include "Dataset.h"
#include "StatsEngine.h"
#include "LogicEngine.h"
#include "NeuralNet.h"
#include "TerminalUI.h"
#include "SeldonExceptions.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <map>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <cmath>

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
              << "  --delimiter <char>                CSV delimiter character (default: ,)\n"
              << "  --output, -o <file>              Export results to specified file (json/csv)\n"
              << "  --loss <mse|cross_entropy>       Manual neural net loss function override\n"
              << "  --verbose                        Enable detailed logs\n"
              << "  --batch                          Non-interactive mode\n"
              << "  --target <col>                   Set target feature for prediction (default: last column)\n"
              << "  --exclude <cols>                 Exclude columns from neural network inputs/targets\n"
              << "  --hidden-nodes <val>             Override neural network hidden layer size\n"
              << "  --chunked                        Enable out-of-core processing for large files\n"
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
    bool chunked = false;
    char delimiter = ',';
    std::string targetCol;
    std::vector<std::string> excludedCols;
    int hiddenNodesOverride = -1;
    std::optional<NeuralNet::LossFunction> lossOverride;
};

#include <fstream>

namespace {
constexpr double kMlrAggregateThreshold = 0.45;
constexpr size_t kBinaryCheckSampleRows = 100;
constexpr double kBinaryEpsilon = 1e-7;

std::string escapeJsonString(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    escaped += "?";
                } else {
                    escaped += ch;
                }
                break;
        }
    }
    return escaped;
}

bool isFiniteNumber(double value) {
    return std::isfinite(value);
}
}

void exportResults(const std::string& filename, 
                   const std::vector<RegressionResult>& simple, 
                   const std::vector<MultipleRegressionResult>& mlr) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "[Agent Warning] Failed to open output file: " << filename << "\n";
        return;
    }

    out << "{\n  \"simple_regressions\": [\n";
    for (size_t i = 0; i < simple.size(); ++i) {
        const auto& r = simple[i];
        out << "    { \"x\": \"" << escapeJsonString(r.featureX) << "\", \"y\": \"" << escapeJsonString(r.featureY)
            << "\", \"slope\": " << (isFiniteNumber(r.m) ? r.m : 0.0)
            << ", \"intercept\": " << (isFiniteNumber(r.c) ? r.c : 0.0)
            << ", \"rSquared\": " << (isFiniteNumber(r.rSquared) ? r.rSquared : 0.0)
            << " }" << (i == simple.size() - 1 ? "" : ",") << "\n";
    }
    out << "  ],\n  \"multiple_regressions\": [\n";
    for (size_t i = 0; i < mlr.size(); ++i) {
        const auto& r = mlr[i];
        out << "    { \"dependent\": \"" << escapeJsonString(r.dependentFeature)
            << "\", \"adjRSquared\": " << (isFiniteNumber(r.adjustedRSquared) ? r.adjustedRSquared : 0.0)
            << ", \"fStat\": " << (isFiniteNumber(r.fStatistic) ? r.fStatistic : 0.0)
            << " }" << (i == mlr.size() - 1 ? "" : ",") << "\n";
    }
    out << "  ]\n}\n";
    if (!out.good()) {
        std::cerr << "[Agent Warning] Failed while writing output file: " << filename << "\n";
        return;
    }
    std::cout << "[Agent] Results exported to: " << filename << "\n";
}

SeldonConfig parseArguments(int argc, char* argv[]) {
    SeldonConfig config;
    if (argc < 2) return config;

    auto parseDouble = [](const std::string& value, const std::string& flag) -> double {
        try {
            size_t pos = 0;
            double parsed = std::stod(value, &pos);
            if (pos != value.size()) {
                throw std::invalid_argument("contains trailing characters");
            }
            if (!std::isfinite(parsed)) {
                throw std::invalid_argument("must be finite");
            }
            return parsed;
        } catch (const std::exception&) {
            throw Seldon::ConfigurationException("Invalid numeric value for " + flag + ": " + value);
        }
    };

    auto parseInt = [](const std::string& value, const std::string& flag) -> int {
        try {
            size_t pos = 0;
            int parsed = std::stoi(value, &pos);
            if (pos != value.size()) {
                throw std::invalid_argument("contains trailing characters");
            }
            return parsed;
        } catch (const std::exception&) {
            throw Seldon::ConfigurationException("Invalid integer value for " + flag + ": " + value);
        }
    };

    config.filename = argv[1];
    if (config.filename == "--help" || config.filename == "-h") {
        config.showHelp = true;
        return config;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--threshold-bivariate" || arg == "-tb") && i + 1 < argc) {
            config.agenticThreshold = parseDouble(argv[++i], arg);
        } else if ((arg == "--threshold-mlr" || arg == "-tm") && i + 1 < argc) {
            config.agenticMlrThreshold = parseDouble(argv[++i], arg);
        } else if (arg == "--exhaustive-scan") {
            config.exhaustiveScan = true;
        } else if (arg == "--epochs" && i + 1 < argc) {
            config.epochs = parseInt(argv[++i], arg);
        } else if (arg == "--lr" && i + 1 < argc) {
            config.lr = parseDouble(argv[++i], arg);
        } else if (arg == "--impute" && i + 1 < argc) {
            std::string strat = argv[++i];
            if (strat == "skip") config.imputation = Dataset::ImputationStrategy::SKIP;
            else if (strat == "mean") config.imputation = Dataset::ImputationStrategy::MEAN;
            else if (strat == "median") config.imputation = Dataset::ImputationStrategy::MEDIAN;
            else if (strat == "zero") config.imputation = Dataset::ImputationStrategy::ZERO;
            else throw Seldon::ConfigurationException("Invalid value for --impute: " + strat + " (expected skip|zero|mean|median)");
        } else if (arg == "--skip-malformed" && i + 1 < argc) {
            std::string value = argv[++i];
            if (value == "true") config.skipMalformed = true;
            else if (value == "false") config.skipMalformed = false;
            else throw Seldon::ConfigurationException("Invalid value for --skip-malformed: " + value + " (expected true/false)");
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--batch") {
            config.batchMode = true;
        } else if (arg == "--loss" && i + 1 < argc) {
            std::string loss = argv[++i];
            if (loss == "mse") config.lossOverride = NeuralNet::LossFunction::MSE;
            else if (loss == "cross_entropy") config.lossOverride = NeuralNet::LossFunction::CROSS_ENTROPY;
            else throw Seldon::ConfigurationException("Invalid value for --loss: " + loss + " (expected mse|cross_entropy)");
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            config.outputFilename = argv[++i];
        } else if (arg == "--chunked") {
            config.chunked = true;
        } else if (arg == "--delimiter" && i + 1 < argc) {
            std::string delim = argv[++i];
            if (delim.size() != 1) {
                throw Seldon::ConfigurationException("--delimiter expects a single character");
            }
            config.delimiter = delim[0];
        } else if (arg == "--target" && i + 1 < argc) {
            config.targetCol = argv[++i];
        } else if (arg == "--exclude" && i + 1 < argc) {
            std::string exList = argv[++i];
            size_t pos = 0;
            while ((pos = exList.find(',')) != std::string::npos) {
                config.excludedCols.push_back(exList.substr(0, pos));
                exList.erase(0, pos + 1);
            }
            if (!exList.empty()) config.excludedCols.push_back(exList);
        } else if (arg == "--hidden-nodes" && i + 1 < argc) {
            config.hiddenNodesOverride = parseInt(argv[++i], arg);
        } else if (arg == "--help") {
            config.showHelp = true;
        } else {
            std::cerr << "Warning: Unknown or incomplete argument: " << arg << "\n";
        }
    }

    config.agenticThreshold = std::clamp(config.agenticThreshold, 0.0, 1.0);
    config.agenticMlrThreshold = std::clamp(config.agenticMlrThreshold, 0.0, 1.0);
    if (config.epochs < 1) config.epochs = 1;
    if (config.lr <= 0.0) {
        throw Seldon::ConfigurationException("Learning rate must be > 0");
    }
    if (config.hiddenNodesOverride == 0 || config.hiddenNodesOverride < -1) {
        throw Seldon::ConfigurationException("--hidden-nodes must be positive");
    }
    if (config.delimiter == '"' || config.delimiter == '\n' || config.delimiter == '\r' || config.delimiter == '\0') {
        throw Seldon::ConfigurationException("Invalid delimiter character");
    }

    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "Seldon: Agentic Data Analytics Engine Initialization...\n";
    SeldonConfig config;
    try {
        config = parseArguments(argc, argv);
    } catch (const Seldon::SeldonException& e) {
        std::cerr << "[Agent Error] " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    if (config.showHelp || config.filename.empty()) {
        printUsage(argv[0]);
        return config.showHelp ? 0 : 1;
    }

    Dataset dataset(config.filename);
    dataset.setDelimiter(config.delimiter);
    try {
        if (config.chunked) {
             std::cout << "[Agent] Operating in Chunked Mode. Ingesting first 10k rows for metadata...\n";
             dataset.loadChunk(0, 10000);
        } else {
             dataset.loadAuto(config.skipMalformed, config.exhaustiveScan, config.imputation);
        }
    } catch (const Seldon::SeldonException& e) {
        std::cerr << "[Agent Error] " << e.what() << "\n";
        return 1;
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

    auto allCols = dataset.getColumnNames();
    std::vector<std::string> filteredCols;
    std::vector<size_t> filteredIndices;

    for (const auto& excluded : config.excludedCols) {
        if (std::find(allCols.begin(), allCols.end(), excluded) == allCols.end()) {
            std::cerr << "\n[Agent Error] Excluded column not found: " << excluded << "\n";
            return 1;
        }
    }

    for (size_t i = 0; i < allCols.size(); ++i) {
        if (std::find(config.excludedCols.begin(), config.excludedCols.end(), allCols[i]) == config.excludedCols.end()) {
            filteredCols.push_back(allCols[i]);
            filteredIndices.push_back(i);
        }
    }

    if (filteredCols.empty()) {
        std::cerr << "\n[Agent Error] All numeric columns were excluded. Terminating analytics sequence gracefully.\n";
        return 1;
    }

    if (config.excludedCols.size() > 0 && config.verbose) {
        std::cout << "[Agent] Filtered out " << config.excludedCols.size() << " column(s).\n";
    }

    // Since Seldon passes Dataset object, we'll keep the exclude check in StatsEngine and LogicEngine
    // Wait, StatsEngine doesn't know about exclusions unless we filter it here or pass the indices.
    // Instead, let's just make a sub-matrix of dataset columns for these operations. 
    // Actually, Seldon's existing architecture passes Dataset& which doesn't support col subsetting easily.
    // Let's filter the exclude list against correlation matrices and inputs below.
    
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
    std::vector<size_t> targetIndices;
    std::vector<size_t> inputIndices;
    std::string dependentVarName;

    if (!config.targetCol.empty()) {
        dependentVarName = config.targetCol;
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
            if (dataset.getColumnNames()[i] == dependentVarName) {
                targetIndices.push_back(i); break;
            }
        }
        if (targetIndices.empty()) {
            std::cerr << "[Agent Error] Target column '" << config.targetCol << "' not found.\n";
            return 1;
        }
    } else {
        // Find last column that is NOT excluded
        for (int i = dataset.getColCount() - 1; i >= 0; --i) {
            if (std::find(config.excludedCols.begin(), config.excludedCols.end(), dataset.getColumnNames()[i]) == config.excludedCols.end()) {
                dependentVarName = dataset.getColumnNames()[i];
                targetIndices.push_back(i);
                break;
            }
        }
    }

    if (!mlrRegressions.empty() && config.targetCol.empty()) {
        dependentVarName = mlrRegressions.front().dependentFeature;
        for (const auto& mlr : mlrRegressions) {
            if (mlr.rSquared > kMlrAggregateThreshold) { // Threshold for inclusion in aggregate architecture
                for (const auto& feat : mlr.independentFeatures) {
                    if (std::find(config.excludedCols.begin(), config.excludedCols.end(), feat) != config.excludedCols.end()) continue;
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
            if (std::find(config.excludedCols.begin(), config.excludedCols.end(), reg.featureX) != config.excludedCols.end()) continue;
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
        size_t limit = std::min(dataset.getColCount(), static_cast<size_t>(3));
        for(size_t i = 0; i < dataset.getColCount() && inputIndices.size() < limit; ++i) {
            if (std::find(config.excludedCols.begin(), config.excludedCols.end(), dataset.getColumnNames()[i]) == config.excludedCols.end() && 
                std::find(targetIndices.begin(), targetIndices.end(), i) == targetIndices.end()) {
                inputIndices.push_back(i);
            }
        }
    }

    size_t inputNodes = inputIndices.size();
    size_t outputNodes = targetIndices.size();
    if (inputNodes == 0 || outputNodes == 0) {
        std::cerr << "[Agent Error] Could not resolve a valid input/target configuration for neural modeling.\n";
        return 1;
    }
    
    // Advanced Topology Automation
    size_t hiddenNodes;
    if (config.hiddenNodesOverride > 0) {
        hiddenNodes = static_cast<size_t>(config.hiddenNodesOverride);
    } else {
        if (inputNodes == 1) {
            hiddenNodes = inputNodes + 2; 
        } else {
            // Scaled heuristic for complex feature sets: Geometric mean representation with a bias factor
            size_t geoMean = static_cast<size_t>(std::round(std::sqrt(inputNodes * outputNodes)));
            hiddenNodes = std::max(geoMean + 2, inputNodes + 1);
            // Cap to prevent explosion on very large feature sets
            if (hiddenNodes > 100) hiddenNodes = 100;
        }
    }
    
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

    bool isBinary = true;
    size_t checkLimitRaw = config.exhaustiveScan ? dataset.getRowCount() : std::min(dataset.getRowCount(), kBinaryCheckSampleRows);
    for (size_t i = 0; i < checkLimitRaw; ++i) {
        double value = columns[targetIndices[0]][i];
        if (std::abs(value) > kBinaryEpsilon && std::abs(value - 1.0) > kBinaryEpsilon) {
            isBinary = false;
            break;
        }
    }
    
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
    hp.lrStepSize = std::max(size_t(1), hp.epochs / 3); // Decay every 1/3 of total epochs
    hp.batchSize = 16;
    hp.activation = NeuralNet::Activation::RELU;
    hp.outputActivation = NeuralNet::Activation::SIGMOID;
    if (Y_train.empty() || Y_train[0].empty()) {
        std::cerr << "[Agent Error] No valid target rows available for neural training.\n";
        return 1;
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
        if (!config.chunked) {
            nn.train(X_train, Y_train, hp, inputScales, outputScales);
        } else {
            std::cout << "[Agent] Starting Out-of-Core Chunked Training...\n";
            dataset.openStream();
            size_t chunkCount = 0;
            // Train on initial metadata chunk
            nn.train(X_train, Y_train, hp, inputScales, outputScales);
            
            // fetchNextChunk will read subsequent data
            while (dataset.fetchNextChunk(10000)) {
                chunkCount++;
                std::cout << "\n[Agent] Processing Chunk " << chunkCount << "...\n";
                // Prepare scaled data for this chunk
                std::vector<std::vector<double>> chunk_X, chunk_Y;
                const auto& cols = dataset.getColumns();
                for (size_t r = 0; r < dataset.getRowCount(); ++r) {
                    std::vector<double> xRow, yRow;
                    for (size_t i = 0; i < inputIndices.size(); ++i) {
                        double val = cols[inputIndices[i]][r];
                        double range = inputScales[i].max - inputScales[i].min;
                        if (range == 0) range = 1;
                        xRow.push_back((val - inputScales[i].min) / range);
                    }
                    for (size_t i = 0; i < targetIndices.size(); ++i) {
                        double val = cols[targetIndices[i]][r];
                        double range = outputScales[i].max - outputScales[i].min;
                        if (range == 0) range = 1;
                        yRow.push_back((val - outputScales[i].min) / range);
                    }
                    chunk_X.push_back(xRow);
                    chunk_Y.push_back(yRow);
                }
                
                // Fine-tune on this chunk
                NeuralNet::Hyperparameters chunkHp = hp;
                chunkHp.epochs = std::max(size_t(1), hp.epochs / 10);
                chunkHp.earlyStoppingPatience = 3;
                
                nn.train(chunk_X, chunk_Y, chunkHp, inputScales, outputScales);
            }
            dataset.closeStream();
            std::cout << "\n[Agent] Out-of-Core Training Complete. Processed " << chunkCount << " additional chunks.\n";
        }
        std::cout << "[Agent] Saving trained matrices to binary file...\n";
        nn.saveModelBinary("seldon_model.seldon");
        std::cout << "[Agent] Neural structure preserved at 'seldon_model.seldon'.\n";
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
    } catch (const Seldon::SeldonException& e) {
        std::cerr << "[Agent Error] Core synthesis failed: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Agent Exception] " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
