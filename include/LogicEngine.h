#pragma once
#include "Dataset.h"
#include "StatsEngine.h"
#include <vector>
#include <string>
#include <utility>

struct CorrelationResult {
    std::string feature1;
    std::string feature2;
    double r;
    double p_value;
    bool is_significant;
};

struct RegressionResult {
    std::string featureX;
    std::string featureY;
    double m; // slope
    double c; // intercept
    double r; // r-score
};

struct MultipleRegressionResult {
    std::string dependentFeature;
    std::vector<std::string> independentFeatures;
    std::vector<double> coefficients; // [Intercept, b1, b2, ...]
};

class LogicEngine {
public:
    static std::vector<std::vector<double>> calculateCorrelationMatrix(
        const Dataset& dataset, 
        const std::vector<ColumnStats>& stats);

    static std::vector<CorrelationResult> findHighCorrelations(
        const Dataset& dataset, 
        const std::vector<std::vector<double>>& matrix, 
        double threshold = 0.6); // Agentic cognitive protection threshold

    static std::vector<RegressionResult> performSimpleRegressions(
        const Dataset& dataset, 
        const std::vector<ColumnStats>& stats,
        const std::vector<CorrelationResult>& highCorrelations);

    static std::vector<MultipleRegressionResult> performMultipleRegressions(
        const Dataset& dataset,
        const std::vector<std::vector<double>>& matrix,
        double threshold = 0.5); // MLR inclusion threshold
};
