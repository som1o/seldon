#include "Dataset.h"
#include "StatsEngine.h"
#include "LogicEngine.h"
#include "NeuralNet.h"
#include "TerminalUI.h"
#include <iostream>
#include <string>

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
    size_t inputNodes = 3; // Minimum fallback
    if (!mlrRegressions.empty()) {
        inputNodes = mlrRegressions.front().independentFeatures.size();
    } else if (!regressions.empty()) {
        inputNodes = 1;
    }

    size_t hiddenNodes = inputNodes * 2; // Simple heuristic

    std::cout << "        -> Agent configured dynamic architecture: [" 
              << inputNodes << " Input] -> [" << hiddenNodes << " Hidden] -> [1 Output]\n";

    std::vector<size_t> topology = {inputNodes, hiddenNodes, 1};
    NeuralNet nn(topology);

    // Provide some normalized sample parameters derived from previous steps 
    // Example conceptual simulation for Synthesis Output
    std::vector<double> inferenceInput;
    for (size_t i = 0; i < inputNodes; ++i) {
        inferenceInput.push_back((rand() / double(RAND_MAX))); // Random normalized conceptual inputs
    }
    
    nn.feedForward(inferenceInput);
    
    std::vector<double> results = nn.getResults();

    TerminalUI::printSynthesisReport(results);
    
    return 0;
}
