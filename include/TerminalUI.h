#pragma once
#include "StatsEngine.h"
#include "LogicEngine.h" // For RegressionResult
#include <vector>
#include <string>

class TerminalUI {
public:
    static void printFoundationTable(const std::vector<std::string>& columnNames, const std::vector<ColumnStats>& stats);
    
    // Phase 2 display
    static void printCorrelationMatrix(const std::vector<std::string>& columnNames, const std::vector<std::vector<double>>& matrix);
    static void printSimpleRegressionFindings(const std::vector<RegressionResult>& correlations);
    static void printMultipleRegressionFindings(const std::vector<MultipleRegressionResult>& mlrCorrelations);

    // Phase 3 display
    static void printNeuralNetInit(size_t inputs, size_t hidden, size_t outputs);
    static void printSynthesisReport(const std::string& targetName, const std::vector<std::string>& inputNames, const std::vector<double>& syntheticOutput);
};
