#pragma once

#include "TypedDataset.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CausalDiscoveryOptions {
    size_t maxFeatures = 8;
    size_t maxConditionSet = 2;
    double alpha = 0.05;
    size_t bootstrapSamples = 100;
    uint32_t randomSeed = 1337;
    bool enableLiNGAM = true;
    bool enableFCI = true;
    bool enableGES = true;
    bool markExperimentalHeuristics = true;
    bool enableKernelCiFallback = true;
    bool enableGrangerValidation = true;
    bool enableIcpValidation = true;
};

struct CausalEdgeResult {
    size_t fromIdx = static_cast<size_t>(-1);
    size_t toIdx = static_cast<size_t>(-1);
    double confidence = 0.0;
    double bootstrapSupport = 0.0;
    std::string evidence;
    std::string interpretation;
};

struct CausalDiscoveryResult {
    std::vector<CausalEdgeResult> edges;
    std::vector<std::string> notes;
    size_t bootstrapRuns = 0;
    bool usedLiNGAM = false;
};

class CausalDiscovery {
public:
    static CausalDiscoveryResult discover(const TypedDataset& data,
                                          const std::vector<size_t>& candidateFeatures,
                                          size_t targetIdx,
                                          const CausalDiscoveryOptions& options = {});
};
