#pragma once
#include "TypedDataset.h"
#include <cstdint>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string model;
    double rmse = 0.0;
    double r2 = 0.0;
    double accuracy = 0.0;
    bool hasAccuracy = false;
    std::vector<double> actual;
    std::vector<double> predicted;
    std::vector<double> featureImportance;
};

struct MultiTargetBenchmarkSummary {
    std::vector<int> targetIndices;
    std::vector<std::string> targetNames;
    std::vector<std::vector<BenchmarkResult>> perTargetResults;
    std::vector<BenchmarkResult> aggregateByModel;
};

class BenchmarkEngine {
public:
    static std::vector<BenchmarkResult> run(const TypedDataset& data, int targetIndex, const std::vector<int>& featureIndices, int kfold, uint32_t seed = 1337);
    static MultiTargetBenchmarkSummary runMultiTarget(const TypedDataset& data,
                                                      const std::vector<int>& targetIndices,
                                                      const std::vector<int>& featureIndices,
                                                      int kfold,
                                                      uint32_t seed = 1337);
};
