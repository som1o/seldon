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
    double m;             // slope
    double c;             // intercept
    double r;             // Pearson r
    double rSquared;
    double stdErrorM;     // Standard Error for slope
    double tStatM;        // t-statistic for slope
    double pValueM;       // p-value for slope
    double confLowM;      // 95% Confidence Interval Low
    double confHighM;     // 95% Confidence Interval High
};

struct MultipleRegressionResult {
    std::string dependentFeature;
    std::vector<std::string> independentFeatures;
    std::vector<double> coefficients; // [Intercept, b1, b2, ...]
    std::vector<double> stdErrors;    // Standard errors for coefficients
    std::vector<double> tStats;       // t-statistics for coefficients
    std::vector<double> pValues;      // p-values for coefficients
    double rSquared;
    double adjustedRSquared;
    double fStatistic;
    double modelPValue;
};

struct Thresholds {
    double bivariate;
    double mlr;
};

class LogicEngine {
public:
    static std::vector<std::vector<double>> calculateCorrelationMatrix(
        const Dataset& dataset, 
        const std::vector<ColumnStats>& stats);

    static Thresholds suggestThresholds(const std::vector<std::vector<double>>& matrix);

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
