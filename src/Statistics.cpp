#include "Statistics.h"

#include <algorithm>
#include <cmath>

ColumnStats Statistics::calculateStats(const std::vector<double>& col) {
    ColumnStats stats{0, 0, 0, 0, 0, 0};
    if (col.empty()) return stats;

    std::vector<double> finite;
    finite.reserve(col.size());
    for (double value : col) {
        if (std::isfinite(value)) {
            finite.push_back(value);
        }
    }
    if (finite.empty()) return stats;

    const size_t n = finite.size();

    double mean = 0.0;
    double m2 = 0.0;
    size_t count = 0;
    for (double value : finite) {
        ++count;
        double delta = value - mean;
        mean += delta / static_cast<double>(count);
        double delta2 = value - mean;
        m2 += delta * delta2;
    }
    stats.mean = mean;
    stats.variance = (count > 1) ? (m2 / static_cast<double>(count - 1)) : 0.0;
    stats.stddev = std::sqrt(stats.variance);

    std::vector<double> medianWork = finite;
    size_t mid = n / 2;
    std::nth_element(medianWork.begin(), medianWork.begin() + mid, medianWork.end());
    double upper = medianWork[mid];
    if (n % 2 == 0) {
        std::nth_element(medianWork.begin(), medianWork.begin() + (mid - 1), medianWork.begin() + mid);
        stats.median = (medianWork[mid - 1] + upper) / 2.0;
    } else {
        stats.median = upper;
    }

    if (n > 2 && stats.stddev > 0) {
        double m3 = 0, m4 = 0;
        for (double val : finite) {
            double diff = val - stats.mean;
            double diff2 = diff * diff;
            m3 += diff2 * diff;
            m4 += diff2 * diff2;
        }

        double term1 = static_cast<double>(n) / ((n - 1) * (n - 2));
        double stddev2 = stats.stddev * stats.stddev;
        double stddev3 = stddev2 * stats.stddev;
        stats.skewness = term1 * (m3 / stddev3);

        if (n > 3) {
            double termK1 = (static_cast<double>(n) * (n + 1)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
            double nMinus1 = (n - 1.0);
            double termK2 = (3.0 * nMinus1 * nMinus1) / ((n - 2.0) * (n - 3.0));
            double stddev4 = stddev2 * stddev2;
            stats.kurtosis = termK1 * (m4 / stddev4) - termK2;
        }
    }

    return stats;
}
