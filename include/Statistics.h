#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct ColumnStats {
    double mean;
    double median;
    double variance;
    double stddev;
    double skewness;
    double kurtosis;
};

struct StationarityDiagnostic {
    std::string feature;
    std::string axis;
    size_t samples = 0;
    double gamma = 0.0;
    double tStatistic = 0.0;
    double pApprox = 1.0;
    double driftRatio = 0.0;
    bool nonStationary = false;
    std::string verdict = "insufficient";
};

struct AsymmetricDirectionScore {
    double xToY = 0.0;
    double yToX = 0.0;
    double asymmetry = 0.0;
    std::string suggestedDirection = "undirected";
};

namespace Statistics {
ColumnStats calculateStats(const std::vector<double>& col);
StationarityDiagnostic adfStyleDrift(const std::vector<double>& series,
                                     const std::vector<double>& axis,
                                     const std::string& featureName = "",
                                     const std::string& axisName = "");
AsymmetricDirectionScore asymmetricInformationGain(const std::vector<double>& x,
                                                   const std::vector<double>& y,
                                                   size_t bins = 8);
}
