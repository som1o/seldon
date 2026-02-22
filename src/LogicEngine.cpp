#include "LogicEngine.h"
#include "MathUtils.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#ifdef USE_OPENMP
#include <omp.h>
#endif

std::vector<std::vector<double>> LogicEngine::calculateCorrelationMatrix(
    const Dataset& dataset, const std::vector<ColumnStats>& stats) 
{
    size_t cols = dataset.getColCount();
    std::vector<std::vector<double>> matrix(cols, std::vector<double>(cols, 1.0)); // diagonal is 1.0

    const auto& columns = dataset.getColumns();

    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (i >= j) continue; // Diagonal and symmetric handled
            auto r_opt = MathUtils::calculatePearson(columns[i], columns[j], stats[i], stats[j]);
            double r = r_opt.value_or(std::numeric_limits<double>::quiet_NaN());
            matrix[i][j] = r;
            matrix[j][i] = r; // symmetric
        }
    }
    return matrix;
}

Thresholds LogicEngine::suggestThresholds(const std::vector<std::vector<double>>& matrix) {
    size_t cols = matrix.size();
    std::vector<double> rValues;
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            rValues.push_back(std::abs(matrix[i][j]));
        }
    }

    if (rValues.empty()) return {0.6, 0.5};

    double sum = 0;
    for (double v : rValues) sum += v;
    double mean = sum / rValues.size();

    double sq_sum = 0;
    for (double v : rValues) sq_sum += (v - mean) * (v - mean);
    double stdev = std::sqrt(sq_sum / rValues.size());

    // Heuristic: Use Mean + 1 StdDev for Bivariate, Mean + 0.5 StdDev for MLR
    // Clamp to reasonable ranges
    double biv = std::clamp(mean + stdev, 0.3, 0.85);
    double mlr = std::clamp(mean + 0.5 * stdev, 0.25, 0.7);

    return {biv, mlr};
}

std::vector<CorrelationResult> LogicEngine::findHighCorrelations(
    const Dataset& dataset, const std::vector<std::vector<double>>& matrix, double threshold) 
{
    std::vector<CorrelationResult> results;
    const auto& names = dataset.getColumnNames();
    size_t cols = names.size();
    size_t rows = dataset.getRowCount();

    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            if (std::abs(matrix[i][j]) >= threshold) {
                auto sig = MathUtils::calculateSignificance(matrix[i][j], rows);
                if (sig.is_significant) {
                    results.push_back({names[i], names[j], matrix[i][j], sig.p_value, sig.is_significant});
                } else {
                    std::cout << "        [Agent Protection] Correlation " << names[i] << " ~ " << names[j] 
                              << " (|r|=" << std::abs(matrix[i][j]) << ") dropped. Not statistically significant (p="
                              << sig.p_value << " > 0.05).\n";
                }
            }
        }
    }
    return results;
}

std::vector<RegressionResult> LogicEngine::performSimpleRegressions(
    const Dataset& dataset, 
    const std::vector<ColumnStats>& stats,
    const std::vector<CorrelationResult>& highCorrelations) 
{
    const auto& columns = dataset.getColumns();
    std::vector<RegressionResult> results(highCorrelations.size());
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t k = 0; k < highCorrelations.size(); ++k) {
        const auto& corr = highCorrelations[k];
        size_t idxX = dataset.getColCount(), idxY = dataset.getColCount();
        for (size_t i = 0; i < dataset.getColCount(); ++i) {
            if (dataset.getColumnNames()[i] == corr.feature1) idxX = i;
            if (dataset.getColumnNames()[i] == corr.feature2) idxY = i;
        }

        RegressionResult res{};
        res.featureX = corr.feature1;
        res.featureY = corr.feature2;
        res.r = corr.r;
        res.rSquared = corr.r * corr.r;
        res.m = 0.0;
        res.c = 0.0;
        res.stdErrorM = 0.0;
        res.tStatM = 0.0;
        res.pValueM = 1.0;
        res.confLowM = 0.0;
        res.confHighM = 0.0;

        if (idxX >= dataset.getColCount() || idxY >= dataset.getColCount()) {
            results[k] = res;
            continue;
        }

        auto params = MathUtils::simpleLinearRegression(columns[idxX], columns[idxY], stats[idxX], stats[idxY], corr.r);

        res.m = params.first;
        res.c = params.second;

        // Diagnostics
        size_t n = dataset.getRowCount();
        if (n > 2) {
            double rss = 0;
            for (size_t i = 0; i < n; ++i) {
                double y_pred = res.m * columns[idxX][i] + res.c;
                double diff = columns[idxY][i] - y_pred;
                rss += diff * diff;
            }
            
            double s2 = rss / (n - 2);
            double varX_sum = (n - 1) * stats[idxX].variance;
            
            if (varX_sum > 0) {
                res.stdErrorM = std::sqrt(s2 / varX_sum);
                if (res.stdErrorM > 0) {
                    res.tStatM = res.m / res.stdErrorM;
                    res.pValueM = MathUtils::getPValueFromT(res.tStatM, n - 2);
                } else {
                    res.tStatM = 0.0;
                    res.pValueM = 1.0;
                }
                
                double tCrit = MathUtils::getCriticalT(0.05, n - 2);
                res.confLowM = res.m - tCrit * res.stdErrorM;
                res.confHighM = res.m + tCrit * res.stdErrorM;
            } else {
                res.stdErrorM = 0; res.tStatM = 0; res.pValueM = 1.0;
                res.confLowM = res.m; res.confHighM = res.m;
            }
        }
        results[k] = res;
    }
    return results;
}

std::vector<MultipleRegressionResult> LogicEngine::performMultipleRegressions(
    const Dataset& dataset,
    const std::vector<std::vector<double>>& matrix,
    double threshold) 
{
    std::vector<MultipleRegressionResult> results;
    size_t cols = dataset.getColCount();
    const auto& names = dataset.getColumnNames();
    const auto& columns = dataset.getColumns();

    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < cols; ++i) {
        std::vector<size_t> independentIndices;
        for (size_t j = 0; j < cols; ++j) {
            if (i == j) continue;
            if (std::abs(matrix[i][j]) >= threshold) {
                independentIndices.push_back(j);
            }
        }

        if (independentIndices.empty()) continue;

        size_t n = dataset.getRowCount();
        MathUtils::Matrix X(n, independentIndices.size() + 1);
        MathUtils::Matrix Y(n, 1);

        for (size_t r = 0; r < n; ++r) {
            X.data[r][0] = 1.0; // Intercept
            for (size_t c = 0; c < independentIndices.size(); ++c) {
                X.data[r][c + 1] = columns[independentIndices[c]][r];
            }
            Y.data[r][0] = columns[i][r];
        }

        auto diag = MathUtils::performMLRWithDiagnostics(X, Y);
        if (diag.success) {
            MultipleRegressionResult res;
            res.dependentFeature = names[i];
            for (size_t idx : independentIndices) res.independentFeatures.push_back(names[idx]);
            res.coefficients = diag.coefficients;
            res.stdErrors = diag.stdErrors;
            res.tStats = diag.tStats;
            res.pValues = diag.pValues;
            res.rSquared = diag.rSquared;
            res.adjustedRSquared = diag.adjustedRSquared;
            res.fStatistic = diag.fStatistic;
            res.modelPValue = diag.modelPValue;
            
            #pragma omp critical
            results.push_back(res);
        } else {
            // Log a message if MLR fails for a particular combination
            #pragma omp critical
            std::cout << "        [Agent Protection] Multi-variable cluster for " << names[i] 
                      << " contains collinear/singular matrices. MLR aborted gracefully.\n";
        }
    }

    // Sort by Adjusted R-Squared descending
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.adjustedRSquared > b.adjustedRSquared;
    });

    return results;
}
