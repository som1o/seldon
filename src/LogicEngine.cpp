#include "LogicEngine.h"
#include "MathUtils.h"
#include <cmath>
#include <iostream>

std::vector<std::vector<double>> LogicEngine::calculateCorrelationMatrix(
    const Dataset& dataset, const std::vector<ColumnStats>& stats) 
{
    size_t cols = dataset.getColCount();
    size_t rows = dataset.getRowCount();
    std::vector<std::vector<double>> matrix(cols, std::vector<double>(cols, 1.0)); // diagonal is 1.0

    // Extract columns temporarily for processing efficiency
    std::vector<std::vector<double>> columns(cols, std::vector<double>(rows));
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            columns[c][r] = dataset.getData()[r][c];
        }
    }

    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            double r = MathUtils::calculatePearson(columns[i], columns[j], stats[i], stats[j]);
            matrix[i][j] = r;
            matrix[j][i] = r; // symmetric
        }
    }
    return matrix;
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
    const Dataset& dataset, const std::vector<ColumnStats>& stats, const std::vector<CorrelationResult>& highCorrelations)
{
    std::vector<RegressionResult> results;
    const auto& names = dataset.getColumnNames();
    size_t cols = names.size();
    size_t rows = dataset.getRowCount();

    // Map feature name to index (inefficient but readable for prototype)
    auto getIndex = [&names](const std::string& name) -> size_t {
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == name) return i;
        }
        return 0; // should throw ideally
    };

    // Extract columns
    std::vector<std::vector<double>> columns(cols, std::vector<double>(rows));
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            columns[c][r] = dataset.getData()[r][c];
        }
    }

    for (const auto& corr : highCorrelations) {
        size_t idxX = getIndex(corr.feature1);
        size_t idxY = getIndex(corr.feature2);

        auto line = MathUtils::simpleLinearRegression(columns[idxX], columns[idxY], stats[idxX], stats[idxY], corr.r);
        results.push_back({corr.feature1, corr.feature2, line.first, line.second, corr.r});
    }

    return results;
}

std::vector<MultipleRegressionResult> LogicEngine::performMultipleRegressions(
    const Dataset& dataset, const std::vector<std::vector<double>>& matrix, double threshold)
{
    std::vector<MultipleRegressionResult> mlr_results;
    const auto& names = dataset.getColumnNames();
    size_t cols = names.size();
    size_t rows = dataset.getRowCount();

    // Naive Agentic Selection for prototype: 
    // Pick the most correlated variable as Y, use others > threshold as X.
    for (size_t yIdx = 0; yIdx < cols; ++yIdx) {
        std::vector<size_t> xIndices;
        for (size_t xIdx = 0; xIdx < cols; ++xIdx) {
            if (xIdx != yIdx && std::abs(matrix[yIdx][xIdx]) >= threshold) {
                xIndices.push_back(xIdx);
            }
        }

        if (xIndices.size() > 1) { // Needs at least 2 independents for MLR
            MathUtils::Matrix X(rows, xIndices.size() + 1); // +1 for intercept
            MathUtils::Matrix Y(rows, 1);

            std::vector<std::string> indepNames;
            for (auto xI : xIndices) indepNames.push_back(names[xI]);

            for (size_t r = 0; r < rows; ++r) {
                Y.data[r][0] = dataset.getData()[r][yIdx];
                X.data[r][0] = 1.0; // Intercept column
                for (size_t c = 0; c < xIndices.size(); ++c) {
                    X.data[r][c + 1] = dataset.getData()[r][xIndices[c]];
                }
            }

            std::vector<double> beta = MathUtils::multipleLinearRegression(X, Y);
            if (!beta.empty()) {
                mlr_results.push_back({names[yIdx], indepNames, beta});
            } else {
                std::cout << "        [Agent Protection] Multi-variable cluster for " << names[yIdx] 
                          << " contains collinear/singular matrices. MLR aborted gracefully.\n";
            }
        }
    }
    return mlr_results;
}
