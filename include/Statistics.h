#pragma once

#include <vector>

struct ColumnStats {
    double mean;
    double median;
    double variance;
    double stddev;
    double skewness;
    double kurtosis;
};

namespace Statistics {
ColumnStats calculateStats(const std::vector<double>& col);
}
