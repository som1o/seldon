#include "StatsEngine.h"
#include <cmath>
#include <algorithm>
#include <numeric>

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

    // Variance & StdDev
    double varianceSum = 0;
    for (double val : col) {
        varianceSum += (val - stats.mean) * (val - stats.mean);
    }
    stats.variance = varianceSum / (n - 1); // Sample variance
    stats.stddev = std::sqrt(stats.variance);

    // Skewness and Kurtosis
    double m3 = 0, m4 = 0;
    for (double val : col) {
        double diff = val - stats.mean;
        m3 += std::pow(diff, 3);
        m4 += std::pow(diff, 4);
    }
    m3 /= n;
    m4 /= n;

    if (stats.stddev > 0) {
        stats.skewness = m3 / std::pow(stats.stddev, 3);
        stats.kurtosis = (m4 / std::pow(stats.stddev, 4)) - 3.0; // Excess kurtosis
    }

    return stats;
}

std::vector<ColumnStats> StatsEngine::calculateFoundation(const Dataset& dataset) {
    std::vector<ColumnStats> allStats;
    size_t cols = dataset.getColCount();
    
    for (size_t c = 0; c < cols; ++c) {
        // Since Dataset is now column-major, we can directly pass the column vector
        // eliminating the heavy abstraction overhead of building isolated copies.
        allStats.push_back(calculateStats(dataset.getColumns()[c]));
    }

    return allStats;
}
