#pragma once
#include "Dataset.h"
#include <vector>

struct ColumnStats {
    double mean;
    double median;
    double variance;
    double stddev;
    double skewness;
    double kurtosis;
};

class StatsEngine {
public:
    static std::vector<ColumnStats> calculateFoundation(const Dataset& dataset);
    static ColumnStats calculateStats(const std::vector<double>& column);
};
