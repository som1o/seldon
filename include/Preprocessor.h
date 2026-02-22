#pragma once
#include "AutoConfig.h"
#include "TypedDataset.h"
#include <string>
#include <unordered_map>
#include <vector>

enum class ScalingMethod { NONE, ZSCORE, MINMAX };

struct ScalingParams {
    ScalingMethod method = ScalingMethod::NONE;
    double mean = 0.0;
    double stddev = 1.0;
    double min = 0.0;
    double max = 1.0;
};

struct PreprocessReport {
    size_t originalRowCount = 0;
    std::unordered_map<std::string, size_t> missingCounts;
    std::unordered_map<std::string, size_t> outlierCounts;
    std::unordered_map<std::string, std::vector<bool>> outlierFlags;
    std::unordered_map<std::string, ScalingParams> scaling;
};

class Preprocessor {
public:
    static PreprocessReport run(TypedDataset& data, const AutoConfig& config);
};
