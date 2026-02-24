#pragma once

#include <vector>

namespace StatsUtils {
double runningMean(const std::vector<double>& values);
double percentileSorted(const std::vector<double>& sorted, double q);
}
