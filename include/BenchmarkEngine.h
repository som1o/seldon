#pragma once
#include "TypedDataset.h"
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

class BenchmarkEngine {
public:
    static std::vector<BenchmarkResult> run(const TypedDataset& data, int targetIndex, const std::vector<int>& featureIndices, int kfold);
};
