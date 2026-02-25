#pragma once

#include "Preprocessor.h"
#include "TypedDataset.h"

#include <string>
#include <vector>

namespace DeterministicHeuristics {

struct Outcome {
    std::vector<int> filteredFeatures;
    std::vector<std::string> excludedReasonLines;
    std::vector<std::vector<std::string>> roleTagRows;

    bool lassoGateApplied = false;
    size_t lassoSelectedCount = 0;

    double rowsToFeatures = 0.0;
    bool lowRatioMode = false;
    bool highRatioMode = false;

    std::string residualNarrative;
    std::vector<std::string> badgeNarratives;
};

Outcome runAllPhases(const TypedDataset& data,
                     const PreprocessReport& prep,
                     int targetIdx,
                     const std::vector<int>& candidateFeatureIdx);

} // namespace DeterministicHeuristics
