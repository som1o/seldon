#include "StatsEngine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>

ColumnStats StatsEngine::calculateStats(const std::vector<double>& col) {
    ColumnStats stats{0, 0, 0, 0, 0, 0};
    if (col.empty()) return stats;

    size_t n = col.size();
    
    // Mean
    double sum = std::accumulate(col.begin(), col.end(), 0.0);
    stats.mean = sum / n;

    // Median
    std::vector<double> sortedCol = col;
    std::sort(sortedCol.begin(), sortedCol.end());
    if (n % 2 == 0) {
        stats.median = (sortedCol[n / 2 - 1] + sortedCol[n / 2]) / 2.0;
    } else {
        stats.median = sortedCol[n / 2];
    }

    // Variance & StdDev (Sample)
    double varianceSum = 0;
    for (double val : col) {
        varianceSum += (val - stats.mean) * (val - stats.mean);
    }
    // Using (n-1) for unbiased sample variance
    stats.variance = (n > 1) ? varianceSum / (n - 1) : 0;
    stats.stddev = std::sqrt(stats.variance);

    // Skewness and Kurtosis (Unbiased Sample Estimators)
    if (n > 2 && stats.stddev > 0) {
        double m3 = 0, m4 = 0;
        for (double val : col) {
            double diff = val - stats.mean;
            m3 += std::pow(diff, 3);
            m4 += std::pow(diff, 4);
        }
        
        // Fisher-Pearson standardized moment coefficient (sample skewness)
        // G1 = [n / ((n-1)(n-2))] * sum((xi - mean)^3 / s^3)
        double term1 = static_cast<double>(n) / ((n - 1) * (n - 2));
        stats.skewness = term1 * (m3 / std::pow(stats.stddev, 3));

        // Excess Kurtosis (Unbiased Sample)
        // G2 = [n(n+1) / ((n-1)(n-2)(n-3))] * sum((xi - mean)^4 / s^4) - [3(n-1)^2 / ((n-2)(n-3))]
        if (n > 3) {
            double termK1 = (static_cast<double>(n) * (n + 1)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
            double termK2 = (3.0 * std::pow(n - 1.0, 2)) / ((n - 2.0) * (n - 3.0));
            stats.kurtosis = termK1 * (m4 / std::pow(stats.stddev, 4)) - termK2;
        }
    }

    return stats;
}

std::vector<ColumnStats> StatsEngine::calculateFoundation(const Dataset& dataset) {
    size_t cols = dataset.getColCount();
    std::vector<ColumnStats> allStats(cols);
    
    #pragma omp parallel for
    for (size_t c = 0; c < cols; ++c) {
        allStats[c] = calculateStats(dataset.getColumns()[c]);
    }

    return allStats;
}
