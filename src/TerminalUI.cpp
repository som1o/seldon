#include "TerminalUI.h"
#include <iostream>
#include <iomanip>

void TerminalUI::printFoundationTable(const std::vector<std::string>& columnNames, const std::vector<ColumnStats>& stats) {
    size_t maxNameLen = 15;
    for (const auto& name : columnNames) maxNameLen = std::max(maxNameLen, name.length());
    
    int w = static_cast<int>(maxNameLen) + 2;
    std::cout << "\n============================================ FOUNDATION SUMMARY ============================================\n";
    std::cout << std::left 
              << std::setw(w) << "Feature" 
              << std::setw(12) << "Mean" 
              << std::setw(12) << "Median" 
              << std::setw(12) << "Variance" 
              << std::setw(12) << "StdDev" 
              << std::setw(12) << "Skewness" 
              << std::setw(12) << "Kurtosis\n";
    std::cout << std::string(w + 12 * 6, '-') << "\n";

    for (size_t i = 0; i < columnNames.size(); ++i) {
        std::cout << std::left << std::setw(w) << columnNames[i]
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(10) << stats[i].mean << "  "
                  << std::setw(10) << stats[i].median << "  "
                  << std::setw(10) << stats[i].variance << "  "
                  << std::setw(10) << stats[i].stddev << "  "
                  << std::setw(10) << stats[i].skewness << "  "
                  << std::setw(10) << stats[i].kurtosis << "\n";
    }
    std::cout << "============================================================================================================\n";
}

void TerminalUI::printCorrelationMatrix(const std::vector<std::string>& columnNames, const std::vector<std::vector<double>>& matrix) {
    std::cout << "\n============================================ CORRELATION MATRIX ============================================\n";
    std::cout << std::setw(12) << " ";
    for (const auto& name : columnNames) {
        std::cout << std::setw(10) << name;
    }
    std::cout << "\n------------------------------------------------------------------------------------------------------------\n";
    for (size_t i = 0; i < columnNames.size(); ++i) {
        std::cout << std::left << std::setw(12) << columnNames[i] << std::right;
        for (size_t j = 0; j < columnNames.size(); ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << matrix[i][j];
        }
        std::cout << "\n";
    }
    std::cout << "============================================================================================================\n";
}

void TerminalUI::printSimpleRegressionFindings(const std::vector<RegressionResult>& correlations) {
    std::cout << "\n[Agent] Significant relationships detected. Filtering noise and constructing linear models...\n";
    if (correlations.empty()) {
        std::cout << "        -> No significant bivariate correlations found above the threshold.\n";
        return;
    }
    for (const auto& res : correlations) {
        std::cout << "        -> " << std::left << std::setw(15) << res.featureY << " ~ " << res.featureX << "\n"
                  << "           Equation: y = " << std::fixed << std::setprecision(4) << res.m << "x + " << res.c << "\n"
                  << "           Fit Metrics: r=" << std::setprecision(3) << res.r << " | r²=" << res.rSquared << "\n"
                  << "           Slope CI: [" << res.confLowM << ", " << res.confHighM << "] | p-value: " << res.pValueM << (res.pValueM < 0.05 ? " (*)" : "") << "\n";
    }
}

void TerminalUI::printMultipleRegressionFindings(const std::vector<MultipleRegressionResult>& mlrCorrelations) {
    std::cout << "\n[Agent] Multi-variable clusters detected. Constructing Multiple Linear Regression equations...\n";
    if (mlrCorrelations.empty()) {
        std::cout << "        -> No significant multi-variable clusters found above the threshold.\n";
        return;
    }
    for (const auto& res : mlrCorrelations) {
        std::cout << "        -> " << res.dependentFeature << " ~ ";
        for (const auto& indep : res.independentFeatures) std::cout << indep << " ";
        std::cout << "\n           Equation: y = " << std::fixed << std::setprecision(3) << res.coefficients[0];
        for (size_t i = 0; i < res.independentFeatures.size(); ++i) {
             std::cout << " + (" << res.coefficients[i+1] << "*" << res.independentFeatures[i] << ")";
        }
        std::cout << "\n           [Model Fit] Adj. r²: " << std::fixed << std::setprecision(4) << res.adjustedRSquared 
                  << " | F-Stat: " << std::setprecision(2) << res.fStatistic << " (p=" << std::setprecision(4) << res.modelPValue << ")\n";
        
        std::cout << "           [Coefficients]:\n";
        for (size_t i = 1; i < res.coefficients.size(); ++i) {
            std::cout << "             " << std::left << std::setw(15) << res.independentFeatures[i-1] 
                      << ": beta=" << std::right << std::setw(8) << std::fixed << std::setprecision(3) << res.coefficients[i]
                      << " | t=" << std::setw(6) << res.tStats[i] 
                      << " | p=" << std::setw(6) << std::setprecision(4) << res.pValues[i] << (res.pValues[i] < 0.05 ? " (*)" : "") << "\n";
        }
    }
}

void TerminalUI::printNeuralNetInit(size_t inputs, size_t hidden, size_t outputs) {
    std::cout << "        -> Agent configured dynamic architecture: [" 
              << inputs << " Input] -> [" << hidden << " Hidden] -> [" << outputs << " Output]\n";
}

void TerminalUI::printSynthesisReport(const std::string& targetName, const std::vector<std::string>& inputNames, const std::vector<double>& syntheticOutput) {
    std::cout << "\n============================================ NARRATIVE SYNTHESIS ===========================================\n";
    std::cout << "[Agent] Deep prediction matrices complete. Analysis of feature intersections finalized.\n\n";
    
    std::cout << "    [Result] Predicted Value for \"" << targetName << "\": " 
              << std::fixed << std::setprecision(4) << syntheticOutput[0] << "\n\n";

    std::cout << "    [Context] This projection is synthesized based on the following feature set:\n";
    for (const auto& name : inputNames) {
        std::cout << "        - " << name << "\n";
    }

    if (syntheticOutput.size() > 1) {
        std::cout << "\n    [Auxiliary Signals]:\n";
        for (size_t i = 1; i < syntheticOutput.size(); ++i) {
             std::cout << "        Signal " << i << ": " << syntheticOutput[i] << "\n";
        }
    }

    std::cout << "\n[Agent] Synthesis finalized. Ready for human presentation.\n";
    std::cout << "============================================================================================================\n";
}
