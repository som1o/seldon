#pragma once

#include "NeuralNet.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ModelMetadata {
    std::string modelId;
    std::string modelPath;
    std::string trainingTimestamp;
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, std::string> hyperparameters;
};

struct ModelRecord {
    ModelMetadata metadata;
    NeuralNet model;
    mutable std::mutex inferenceMutex;

    ModelRecord(ModelMetadata metadataValue, NeuralNet modelValue);
};

struct PredictionDistributionStats {
    uint64_t count = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double mean = 0.0;
};

struct MonitoringSnapshot {
    uint64_t totalRequests = 0;
    uint64_t predictRequests = 0;
    uint64_t batchRequests = 0;
    uint64_t errorRequests = 0;
    uint64_t totalPredictions = 0;
    double averageLatencyMs = 0.0;
    std::vector<PredictionDistributionStats> predictionDistributions;
};

class ModelRegistry {
public:
    void loadFromFile(const std::string& registryPath);
    std::shared_ptr<ModelRecord> getModel(const std::string& modelId) const;
    std::string defaultModelId() const;
    std::vector<ModelMetadata> metadataSnapshot() const;

private:
    mutable std::shared_mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<ModelRecord>> models;
    std::vector<std::string> insertionOrder;
};

class RequestMonitor {
public:
    void recordSuccess(const std::string& endpoint,
                       double latencyMs,
                       const std::vector<std::vector<double>>& predictions);
    void recordError(const std::string& endpoint, double latencyMs);
    MonitoringSnapshot snapshot() const;

private:
    struct MutableDistributionStats {
        uint64_t count = 0;
        double sum = 0.0;
        double min = std::numeric_limits<double>::infinity();
        double max = -std::numeric_limits<double>::infinity();
    };

    std::atomic<uint64_t> totalRequests{0};
    std::atomic<uint64_t> predictRequests{0};
    std::atomic<uint64_t> batchRequests{0};
    std::atomic<uint64_t> errorRequests{0};
    std::atomic<uint64_t> totalPredictions{0};
    std::atomic<uint64_t> totalLatencyMicros{0};

    mutable std::mutex distributionMutex;
    std::vector<MutableDistributionStats> predictionStats;
};

class PredictionService {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8080;
        size_t threadCount = 8;
    };

    PredictionService(ModelRegistry& registry, RequestMonitor& monitor);
    int start(const Config& config);

private:
    ModelRegistry& registry;
    RequestMonitor& monitor;
};
