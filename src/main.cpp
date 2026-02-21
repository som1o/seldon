#include "Dataset.h"
#include "StatsEngine.h"
#include "LogicEngine.h"
#include "NeuralNet.h"
#include "TerminalUI.h"
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char* argv[]) {
    std::cout << "Seldon: Agentic Data Analytics Engine Initialization...\n";
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset.csv>\n";
        return 1;
    }

    std::string filename = argv[1];
    Dataset dataset(filename);

    if (!dataset.load()) {
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
    std::cout << "\n[Agent] Foundation analysis complete. Proceed to Bivariate pattern hunting? (Y/n): ";
    std::string response;
    std::getline(std::cin, response);

    if (response == "n" || response == "N") {
        std::cout << "[Agent] Acknowledged. Terminating sequence.\n";
        return 0;
    }

    std::cout << "\n[Agent] Initiating Layer 2 Logic Engine...\n";

    // Layer 2
    auto corrMatrix = LogicEngine::calculateCorrelationMatrix(dataset, stats);
    TerminalUI::printCorrelationMatrix(dataset.getColumnNames(), corrMatrix);

    // Agentic Threshold: 0.6 is a deliberate design choice. 
    // It purposefully filters out weak signals to protect the user's cognitive load,
    // ensuring Seldon only tells statistically "meaningful" stories before moving to regression.
    double agenticThreshold = 0.6; 
    auto highCorrs = LogicEngine::findHighCorrelations(dataset, corrMatrix, agenticThreshold);
    auto regressions = LogicEngine::performSimpleRegressions(dataset, stats, highCorrs);

    TerminalUI::printSimpleRegressionFindings(regressions);

    // Multiple Linear Regression mapping on clustered columns
    double agenticMlrThreshold = 0.35; // Looser threshold to find multi-variable clusters
    auto mlrRegressions = LogicEngine::performMultipleRegressions(dataset, corrMatrix, agenticMlrThreshold);
    TerminalUI::printMultipleRegressionFindings(mlrRegressions);

    // Agentic Checkpoint 2
    std::cout << "\n[Agent] Inference mapped. Do you want to initialize a combined, deep multivariate analysis? (Y/n): ";
    std::getline(std::cin, response);

    if (response == "n" || response == "N") {
        std::cout << "[Agent] Acknowledged. Terminating sequence.\n";
        return 0;
    }

    std::cout << "\n[Agent] Initializing predictive modeling. Synthesizing Neural Topology...\n";

    // Layer 3 Synthesis
    // Dynamic Topology: Input size dynamically scales based on MLR independent variable counts.
    size_t targetIdx = dataset.getColCount() - 1; // Default to last column
    std::vector<size_t> inputIndices;
    
    if (!mlrRegressions.empty()) {
        for (const auto& indepFeature : mlrRegressions.front().independentFeatures) {
             for (size_t i = 0; i < dataset.getColCount(); ++i) {
                 if (dataset.getColumnNames()[i] == indepFeature) {
                     inputIndices.push_back(i); break;
                 }
             }
        }
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
             if (dataset.getColumnNames()[i] == mlrRegressions.front().dependentFeature) {
                 targetIdx = i; break;
             }
        }
    } else if (!regressions.empty()) {
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
             if (dataset.getColumnNames()[i] == regressions.front().featureX) {
                 inputIndices.push_back(i); break;
             }
        }
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
             if (dataset.getColumnNames()[i] == regressions.front().featureY) {
                 targetIdx = i; break;
             }
        }
    } else {
        // Fallback: Use first up to 3 columns to predict the last
        size_t limit = std::min(dataset.getColCount() - 1, static_cast<size_t>(3));
        for(size_t i = 0; i < limit; ++i) inputIndices.push_back(i);
    }

    size_t inputNodes = inputIndices.empty() ? 1 : inputIndices.size();
    size_t hiddenNodes = inputNodes * 2; // Simple heuristic

    std::cout << "        -> Agent configured dynamic architecture: [" 
              << inputNodes << " Input] -> [" << hiddenNodes << " Hidden] -> [1 Output]\n";

    std::vector<size_t> topology = {inputNodes, hiddenNodes, 1};
    NeuralNet nn(topology);

    const auto& columns = dataset.getColumns();
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> Y_train;
    
    // Scale features to [0, 1] for stable sigmoids
    auto getScaledCol = [&](size_t colIdx) {
        std::vector<double> col = columns[colIdx];
        double minVal = *std::min_element(col.begin(), col.end());
        double maxVal = *std::max_element(col.begin(), col.end());
        double range = maxVal - minVal;
        if (range == 0) range = 1.0;
        for (auto& val : col) val = (val - minVal) / range;
        return col;
    };
    
    std::vector<std::vector<double>> scaledXCols;
    for (size_t idx : inputIndices) {
        scaledXCols.push_back(getScaledCol(idx));
    }
    std::vector<double> scaledYCol = getScaledCol(targetIdx);
    
    for (size_t r = 0; r < dataset.getRowCount(); ++r) {
        std::vector<double> xRow;
        for (const auto& col : scaledXCols) xRow.push_back(col[r]);
        X_train.push_back(xRow);
        Y_train.push_back({scaledYCol[r]});
    }

    nn.train(X_train, Y_train, 0.1, 100);

    // Run inference on the first row of data for conceptual validation
    std::vector<double> inferenceInput;
    for (size_t i = 0; i < inputNodes; ++i) {
        inferenceInput.push_back(X_train[0][i]);
    }
    
    std::vector<double> results = nn.predict(inferenceInput);

    TerminalUI::printSynthesisReport(results);
    
    return 0;
}
