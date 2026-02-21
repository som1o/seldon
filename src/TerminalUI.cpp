#include "TerminalUI.h"
#include <iostream>
#include <iomanip>

void TerminalUI::printFoundationTable(const std::vector<std::string>& columnNames, const std::vector<ColumnStats>& stats) {
    std::cout << "\n============================================ FOUNDATION SUMMARY ============================================\n";
    std::cout << std::left 
              << std::setw(15) << "Feature" 
              << std::setw(12) << "Mean" 
              << std::setw(12) << "Median" 
              << std::setw(12) << "Variance" 
              << std::setw(12) << "StdDev" 
              << std::setw(12) << "Skewness" 
              << std::setw(12) << "Kurtosis\n";
    std::cout << "------------------------------------------------------------------------------------------------------------\n";

    for (size_t i = 0; i < columnNames.size(); ++i) {
        std::cout << std::left << std::setw(15) << columnNames[i]
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
        std::cout << "        -> " << res.featureY << " ~ " << res.featureX 
                  << " (r=" << std::fixed << std::setprecision(2) << res.r << ") | "
                  << "Equation: y = " << res.m << "x + " << res.c << "\n";
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
        std::cout << "\n           Equation: y = " << std::fixed << std::setprecision(2) << res.coefficients[0];
        for (size_t i = 0; i < res.independentFeatures.size(); ++i) {
             std::cout << " + (" << res.coefficients[i+1] << "*" << res.independentFeatures[i] << ")";
        }
        std::cout << "\n";
    }
}

void TerminalUI::printSynthesisReport(const std::vector<double>& syntheticOutput) {
    std::cout << "\n=========================================== SYNTHESIS REPORT ===========================================\n";
    std::cout << "[Agent] Deep prediction matrices complete. Final simulated inference outputs mapped below:\n\n";
    for (size_t i = 0; i < syntheticOutput.size(); ++i) {
        std::cout << "        Neural Node [" << i << "] Output Signal: " 
                  << std::fixed << std::setprecision(4) << syntheticOutput[i] << "\n";
    }
    std::cout << "\n[Agent] Synthesis finalized. Ready for human presentation.\n";
    std::cout << "============================================================================================================\n";
}
