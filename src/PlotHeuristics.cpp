#include "PlotHeuristics.h"

#include "MathUtils.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace PlotHeuristics {
size_t approximateUniqueCount(const std::vector<double>& values) {
    std::unordered_set<long long> uniq;
    uniq.reserve(values.size());
    for (double v : values) {
        if (!std::isfinite(v)) continue;
        long long q = static_cast<long long>(std::llround(v * 1000000.0));
        uniq.insert(q);
    }
    return uniq.size();
}

bool shouldAddOgive(const std::vector<double>& values, const HeuristicTuningConfig& tuning) {
    if (values.size() < tuning.ogiveMinPoints) return false;
    return approximateUniqueCount(values) >= tuning.ogiveMinUnique;
}

bool shouldAddBoxPlot(const std::vector<double>& values, const HeuristicTuningConfig& tuning, double eps) {
    if (values.size() < tuning.boxPlotMinPoints) return false;
    MathUtils::NumericSummary summary = MathUtils::summarizeNumeric(values);
    const double threshold = std::max(tuning.boxPlotMinIqr, eps);
    return std::isfinite(summary.iqr) && summary.iqr > threshold;
}

bool shouldAddPieChart(const std::vector<double>& counts, const HeuristicTuningConfig& tuning) {
    if (counts.size() < tuning.pieMinCategories || counts.size() > tuning.pieMaxCategories) return false;
    double total = 0.0;
    double top = 0.0;
    size_t nonZero = 0;
    for (double c : counts) {
        if (c > 0.0) {
            total += c;
            top = std::max(top, c);
            ++nonZero;
        }
    }
    if (total <= 0.0) return false;
    if (nonZero < 2) return false;

    double entropy = 0.0;
    for (double c : counts) {
        if (c <= 0.0) continue;
        const double p = c / total;
        entropy += -p * std::log(std::max(p, tuning.numericEpsilon));
    }
    const double maxEntropy = std::log(static_cast<double>(nonZero));
    const double normalizedEntropy = maxEntropy > tuning.numericEpsilon ? (entropy / maxEntropy) : 0.0;

    return (top / total) <= tuning.pieMaxDominanceRatio && normalizedEntropy >= 0.55;
}

bool shouldAddConfidenceBand(double r,
                             bool statSignificant,
                             size_t sampleSize,
                             const HeuristicTuningConfig& tuning) {
    if (!statSignificant) return false;
    if (sampleSize < tuning.scatterConfidenceMinSampleSize) return false;
    if (!std::isfinite(r)) return false;
    return std::abs(r) >= tuning.scatterConfidenceMinAbsCorr;
}

bool shouldAddResidualPlot(double r,
                           bool selected,
                           size_t sampleSize,
                           const HeuristicTuningConfig& tuning) {
    if (!selected) return false;
    if (sampleSize < tuning.residualPlotMinSampleSize) return false;
    if (!std::isfinite(r)) return false;
    return std::abs(r) >= tuning.residualPlotMinAbsCorr;
}

bool shouldOverlayFittedLine(double r,
                             bool statSignificant,
                             const std::vector<double>& x,
                             const std::vector<double>& y,
                             double slope,
                             double intercept,
                             const HeuristicTuningConfig& tuning) {
    if (!statSignificant) return false;
    if (!std::isfinite(r) || !std::isfinite(slope) || !std::isfinite(intercept)) return false;

    const size_t size = std::min(x.size(), y.size());
    std::vector<double> xValid;
    std::vector<double> yValid;
    xValid.reserve(size);
    yValid.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        const double xv = x[i];
        const double yv = y[i];
        if (!std::isfinite(xv) || !std::isfinite(yv)) continue;
        xValid.push_back(xv);
        yValid.push_back(yv);
    }

    const size_t sampleSize = xValid.size();
    if (sampleSize < tuning.scatterFitMinSampleSize) return false;
    if (std::abs(r) < tuning.scatterFitMinAbsCorr) return false;

    const size_t minUniqueX = std::max<size_t>(6, sampleSize / 10);
    if (approximateUniqueCount(xValid) < minUniqueX) return false;

    const MathUtils::NumericSummary ySummary = MathUtils::summarizeNumeric(yValid);
    if (!std::isfinite(ySummary.stddev) || ySummary.stddev <= tuning.numericEpsilon) return false;

    const double explained = r * r;
    const double minExplained = (sampleSize < 40) ? 0.18 : 0.12;
    if (explained < minExplained) return false;

    return true;
}
}
