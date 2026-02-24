#pragma once

#include "AutoConfig.h"

#include <cstddef>
#include <vector>

namespace PlotHeuristics {
size_t approximateUniqueCount(const std::vector<double>& values);
size_t sturgesBinCount(size_t sampleSize);
bool shouldUseSturgesGrouping(const std::vector<double>& values, double epsilon = 1e-12);
bool shouldAvoidCategoryHeavyCharts(size_t categoryCount, size_t hardLimit = 18);
bool shouldAddOgive(const std::vector<double>& values, const HeuristicTuningConfig& tuning);
bool shouldAddBoxPlot(const std::vector<double>& values, const HeuristicTuningConfig& tuning, double eps);
bool shouldAddPieChart(const std::vector<double>& counts, const HeuristicTuningConfig& tuning);
bool shouldAddConfidenceBand(double r,
                             bool statSignificant,
                             size_t sampleSize,
                             const HeuristicTuningConfig& tuning);
bool shouldAddResidualPlot(double r,
                           bool selected,
                           size_t sampleSize,
                           const HeuristicTuningConfig& tuning);
bool shouldOverlayFittedLine(double r,
                             bool statSignificant,
                             const std::vector<double>& x,
                             const std::vector<double>& y,
                             double slope,
                             double intercept,
                             const HeuristicTuningConfig& tuning);
}
