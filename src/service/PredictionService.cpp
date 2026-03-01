#include "PredictionService.h"

#include "SeldonExceptions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <httplib.h>

namespace {
using Clock = std::chrono::steady_clock;

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool booleanValue = false;
    double numberValue = 0.0;
    std::string stringValue;
    std::vector<JsonValue> arrayValue;
    std::unordered_map<std::string, JsonValue> objectValue;

    bool isObject() const noexcept { return type == Type::Object; }
    bool isArray() const noexcept { return type == Type::Array; }
    bool isString() const noexcept { return type == Type::String; }
    bool isNumber() const noexcept { return type == Type::Number; }

    const JsonValue* find(const std::string& key) const {
        if (!isObject()) return nullptr;
        auto it = objectValue.find(key);
        if (it == objectValue.end()) return nullptr;
        return &it->second;
    }

    std::string dump() const {
        switch (type) {
            case Type::Null:
                return "null";
            case Type::Bool:
                return booleanValue ? "true" : "false";
            case Type::Number: {
                std::ostringstream out;
                out << std::setprecision(15) << numberValue;
                return out.str();
            }
            case Type::String: {
                std::ostringstream out;
                out << '"';
                for (char c : stringValue) {
                    switch (c) {
                        case '"': out << "\\\""; break;
                        case '\\': out << "\\\\"; break;
                        case '\n': out << "\\n"; break;
                        case '\r': out << "\\r"; break;
                        case '\t': out << "\\t"; break;
                        default: out << c; break;
                    }
                }
                out << '"';
                return out.str();
            }
            case Type::Array: {
                std::ostringstream out;
                out << '[';
                for (size_t i = 0; i < arrayValue.size(); ++i) {
                    if (i > 0) out << ',';
                    out << arrayValue[i].dump();
                }
                out << ']';
                return out.str();
            }
            case Type::Object: {
                std::ostringstream out;
                out << '{';
                bool first = true;
                for (const auto& kv : objectValue) {
                    if (!first) out << ',';
                    first = false;
                    JsonValue key;
                    key.type = Type::String;
                    key.stringValue = kv.first;
                    out << key.dump() << ':' << kv.second.dump();
                }
                out << '}';
                return out.str();
            }
        }
        return "null";
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& source) : text(source) {}

    JsonValue parse() {
        skipWhitespace();
        JsonValue value = parseValue();
        skipWhitespace();
        if (position != text.size()) {
            throw Seldon::ConfigurationException("Unexpected trailing JSON content");
        }
        return value;
    }

private:
    const std::string& text;
    size_t position = 0;

    void skipWhitespace() {
        while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])) != 0) {
            ++position;
        }
    }

    char peek() const {
        if (position >= text.size()) {
            throw Seldon::ConfigurationException("Unexpected end of JSON input");
        }
        return text[position];
    }

    char take() {
        if (position >= text.size()) {
            throw Seldon::ConfigurationException("Unexpected end of JSON input");
        }
        return text[position++];
    }

    void expect(char expected) {
        const char value = take();
        if (value != expected) {
            throw Seldon::ConfigurationException(std::string("Expected JSON character '") + expected + "'");
        }
    }

    JsonValue parseValue() {
        skipWhitespace();
        const char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBoolean();
        if (c == 'n') return parseNull();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0) return parseNumber();
        throw Seldon::ConfigurationException("Invalid JSON token");
    }

    JsonValue parseObject() {
        JsonValue object;
        object.type = JsonValue::Type::Object;

        expect('{');
        skipWhitespace();
        if (peek() == '}') {
            take();
            return object;
        }

        while (true) {
            JsonValue key = parseString();
            skipWhitespace();
            expect(':');
            skipWhitespace();
            JsonValue value = parseValue();
            object.objectValue.emplace(key.stringValue, std::move(value));

            skipWhitespace();
            const char next = take();
            if (next == '}') {
                break;
            }
            if (next != ',') {
                throw Seldon::ConfigurationException("Expected ',' or '}' in JSON object");
            }
            skipWhitespace();
        }

        return object;
    }

    JsonValue parseArray() {
        JsonValue array;
        array.type = JsonValue::Type::Array;

        expect('[');
        skipWhitespace();
        if (peek() == ']') {
            take();
            return array;
        }

        while (true) {
            array.arrayValue.push_back(parseValue());
            skipWhitespace();
            const char next = take();
            if (next == ']') {
                break;
            }
            if (next != ',') {
                throw Seldon::ConfigurationException("Expected ',' or ']' in JSON array");
            }
            skipWhitespace();
        }

        return array;
    }

    JsonValue parseString() {
        JsonValue str;
        str.type = JsonValue::Type::String;

        expect('"');
        while (true) {
            const char c = take();
            if (c == '"') break;
            if (c == '\\') {
                const char escaped = take();
                switch (escaped) {
                    case '"': str.stringValue.push_back('"'); break;
                    case '\\': str.stringValue.push_back('\\'); break;
                    case '/': str.stringValue.push_back('/'); break;
                    case 'b': str.stringValue.push_back('\b'); break;
                    case 'f': str.stringValue.push_back('\f'); break;
                    case 'n': str.stringValue.push_back('\n'); break;
                    case 'r': str.stringValue.push_back('\r'); break;
                    case 't': str.stringValue.push_back('\t'); break;
                    default:
                        throw Seldon::ConfigurationException("Unsupported escaped character in JSON string");
                }
                continue;
            }
            str.stringValue.push_back(c);
        }

        return str;
    }

    JsonValue parseBoolean() {
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        if (text.compare(position, 4, "true") == 0) {
            value.booleanValue = true;
            position += 4;
            return value;
        }
        if (text.compare(position, 5, "false") == 0) {
            value.booleanValue = false;
            position += 5;
            return value;
        }
        throw Seldon::ConfigurationException("Invalid JSON boolean value");
    }

    JsonValue parseNull() {
        if (text.compare(position, 4, "null") != 0) {
            throw Seldon::ConfigurationException("Invalid JSON null value");
        }
        position += 4;
        JsonValue value;
        value.type = JsonValue::Type::Null;
        return value;
    }

    JsonValue parseNumber() {
        const size_t start = position;
        if (peek() == '-') take();

        if (peek() == '0') {
            take();
        } else {
            while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
                ++position;
            }
        }

        if (position < text.size() && text[position] == '.') {
            ++position;
            while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
                ++position;
            }
        }

        if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
            ++position;
            if (position < text.size() && (text[position] == '+' || text[position] == '-')) {
                ++position;
            }
            while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])) != 0) {
                ++position;
            }
        }

        const std::string token = text.substr(start, position - start);
        if (token.empty() || token == "-") {
            throw Seldon::ConfigurationException("Invalid JSON number");
        }

        JsonValue number;
        number.type = JsonValue::Type::Number;
        try {
            number.numberValue = std::stod(token);
        } catch (const std::exception&) {
            throw Seldon::ConfigurationException("Failed to parse JSON number");
        }
        return number;
    }
};

std::string escapeJsonString(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

std::string formatDouble(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(15) << value;
    return out.str();
}

void ensureFiniteVector(const std::vector<double>& values, const std::string& label) {
    for (double v : values) {
        if (!std::isfinite(v)) {
            throw Seldon::ConfigurationException(label + " contains non-finite numeric values");
        }
    }
}

std::vector<double> parseFeatureVector(const JsonValue& value, const std::string& label) {
    if (!value.isArray()) {
        throw Seldon::ConfigurationException(label + " must be a numeric array");
    }

    std::vector<double> out;
    out.reserve(value.arrayValue.size());
    for (const auto& item : value.arrayValue) {
        if (!item.isNumber()) {
            throw Seldon::ConfigurationException(label + " must contain numbers only");
        }
        out.push_back(item.numberValue);
    }

    ensureFiniteVector(out, label);
    return out;
}

std::vector<std::vector<double>> parseBatchFeatureVectors(const JsonValue& value) {
    if (!value.isArray()) {
        throw Seldon::ConfigurationException("instances must be a 2D numeric array");
    }

    std::vector<std::vector<double>> out;
    out.reserve(value.arrayValue.size());
    for (size_t i = 0; i < value.arrayValue.size(); ++i) {
        out.push_back(parseFeatureVector(value.arrayValue[i], "instances[" + std::to_string(i) + "]"));
    }
    return out;
}

std::string modelIdFromRequest(const JsonValue& request, const ModelRegistry& registry) {
    if (const JsonValue* modelId = request.find("model_id"); modelId != nullptr && modelId->isString()) {
        if (!modelId->stringValue.empty()) {
            return modelId->stringValue;
        }
    }

    const std::string fallback = registry.defaultModelId();
    if (fallback.empty()) {
        throw Seldon::ConfigurationException("No models are registered");
    }
    return fallback;
}

JsonValue parseJsonText(const std::string& text) {
    JsonParser parser(text);
    return parser.parse();
}

void setJsonResponse(httplib::Response& response, int status, const std::string& payload) {
    response.status = status;
    response.set_content(payload, "application/json");
}

std::string serializeVector(const std::vector<double>& values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ',';
        out << formatDouble(values[i]);
    }
    out << ']';
    return out.str();
}

std::string serializeMatrix(const std::vector<std::vector<double>>& matrix) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < matrix.size(); ++i) {
        if (i > 0) out << ',';
        out << serializeVector(matrix[i]);
    }
    out << ']';
    return out.str();
}

long long toLatencyMicros(double latencyMs) {
    if (!std::isfinite(latencyMs) || latencyMs <= 0.0) return 0;
    return static_cast<long long>(std::llround(latencyMs * 1000.0));
}

void logMonitoringLine(const std::string& endpoint, double latencyMs, const MonitoringSnapshot& snapshot) {
    std::ostringstream line;
    line << "[SeldonService][Monitor] endpoint=" << endpoint
         << " total_requests=" << snapshot.totalRequests
         << " errors=" << snapshot.errorRequests
         << " latency_ms=" << latencyMs
         << " avg_latency_ms=" << snapshot.averageLatencyMs
         << " predictions=" << snapshot.totalPredictions;

    if (!snapshot.predictionDistributions.empty()) {
        const auto& d = snapshot.predictionDistributions.front();
        line << " first_dim_mean=" << d.mean
             << " first_dim_min=" << d.min
             << " first_dim_max=" << d.max;
    }

    std::cout << line.str() << "\n";
}

std::string makePredictSuccessResponse(const std::string& modelId,
                                       const std::vector<double>& prediction,
                                       double latencyMs,
                                       uint64_t totalRequests) {
    std::ostringstream out;
    out << "{"
        << "\"model_id\":\"" << escapeJsonString(modelId) << "\"," 
        << "\"prediction\":" << serializeVector(prediction) << ","
        << "\"latency_ms\":" << formatDouble(latencyMs) << ","
        << "\"monitoring\":{\"total_requests\":" << totalRequests << "}"
        << "}";
    return out.str();
}

std::string makeBatchSuccessResponse(const std::string& modelId,
                                     const std::vector<std::vector<double>>& predictions,
                                     double latencyMs,
                                     uint64_t totalRequests) {
    std::ostringstream out;
    out << "{"
        << "\"model_id\":\"" << escapeJsonString(modelId) << "\","
        << "\"count\":" << predictions.size() << ","
        << "\"predictions\":" << serializeMatrix(predictions) << ","
        << "\"latency_ms\":" << formatDouble(latencyMs) << ","
        << "\"monitoring\":{\"total_requests\":" << totalRequests << "}"
        << "}";
    return out.str();
}

std::string makeErrorResponse(const std::string& error, double latencyMs) {
    std::ostringstream out;
    out << "{"
        << "\"error\":\"" << escapeJsonString(error) << "\","
        << "\"latency_ms\":" << formatDouble(latencyMs)
        << "}";
    return out.str();
}
} // namespace

ModelRecord::ModelRecord(ModelMetadata metadataValue, NeuralNet modelValue)
    : metadata(std::move(metadataValue)), model(std::move(modelValue)) {}

void ModelRegistry::loadFromFile(const std::string& registryPath) {
    std::ifstream input(registryPath);
    if (!input) {
        throw Seldon::ConfigurationException("Failed to open model registry file: " + registryPath);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    const JsonValue root = parseJsonText(buffer.str());

    const JsonValue* modelsNode = root.find("models");
    if (!root.isObject() || modelsNode == nullptr || !modelsNode->isArray()) {
        throw Seldon::ConfigurationException("Registry must contain a top-level 'models' array");
    }

    std::unordered_map<std::string, std::shared_ptr<ModelRecord>> loadedModels;
    std::vector<std::string> loadedOrder;

    for (const auto& item : modelsNode->arrayValue) {
        if (!item.isObject()) {
            throw Seldon::ConfigurationException("Each entry in 'models' must be an object");
        }

        const JsonValue* modelIdNode = item.find("model_id");
        const JsonValue* modelPathNode = item.find("model_path");
        if (modelIdNode == nullptr || !modelIdNode->isString()) {
            throw Seldon::ConfigurationException("Each model entry requires string field 'model_id'");
        }
        if (modelPathNode == nullptr || !modelPathNode->isString()) {
            throw Seldon::ConfigurationException("Each model entry requires string field 'model_path'");
        }

        ModelMetadata metadata;
        metadata.modelId = modelIdNode->stringValue;
        metadata.modelPath = modelPathNode->stringValue;
        if (metadata.modelId.empty()) {
            throw Seldon::ConfigurationException("model_id cannot be empty");
        }
        if (loadedModels.find(metadata.modelId) != loadedModels.end()) {
            throw Seldon::ConfigurationException("Duplicate model_id in registry: " + metadata.modelId);
        }

        if (const JsonValue* timestampNode = item.find("training_timestamp");
            timestampNode != nullptr && timestampNode->isString()) {
            metadata.trainingTimestamp = timestampNode->stringValue;
        }

        if (const JsonValue* metricsNode = item.find("metrics");
            metricsNode != nullptr && metricsNode->isObject()) {
            if (const JsonValue* rmseNode = metricsNode->find("rmse"); rmseNode != nullptr && rmseNode->isNumber()) {
                metadata.rmse = rmseNode->numberValue;
            }
            if (const JsonValue* accuracyNode = metricsNode->find("accuracy");
                accuracyNode != nullptr && accuracyNode->isNumber()) {
                metadata.accuracy = accuracyNode->numberValue;
            }
        }

        if (const JsonValue* hyperNode = item.find("hyperparameters");
            hyperNode != nullptr && hyperNode->isObject()) {
            for (const auto& kv : hyperNode->objectValue) {
                if (kv.second.isString()) {
                    metadata.hyperparameters[kv.first] = kv.second.stringValue;
                } else {
                    metadata.hyperparameters[kv.first] = kv.second.dump();
                }
            }
        }

        NeuralNet model({1, 1});
        model.loadModelBinary(metadata.modelPath);

        loadedOrder.push_back(metadata.modelId);
        loadedModels.emplace(metadata.modelId,
                             std::make_shared<ModelRecord>(std::move(metadata), std::move(model)));
    }

    if (loadedModels.empty()) {
        throw Seldon::ConfigurationException("Registry has no model entries");
    }

    std::unique_lock<std::shared_mutex> lock(mutex);
    models = std::move(loadedModels);
    insertionOrder = std::move(loadedOrder);
}

std::shared_ptr<ModelRecord> ModelRegistry::getModel(const std::string& modelId) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = models.find(modelId);
    if (it == models.end()) {
        return nullptr;
    }
    return it->second;
}

std::string ModelRegistry::defaultModelId() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (insertionOrder.empty()) {
        return "";
    }
    return insertionOrder.front();
}

std::vector<ModelMetadata> ModelRegistry::metadataSnapshot() const {
    std::vector<ModelMetadata> out;
    std::shared_lock<std::shared_mutex> lock(mutex);
    out.reserve(insertionOrder.size());
    for (const auto& modelId : insertionOrder) {
        auto it = models.find(modelId);
        if (it != models.end() && it->second) {
            out.push_back(it->second->metadata);
        }
    }
    return out;
}

void RequestMonitor::recordSuccess(const std::string& endpoint,
                                   double latencyMs,
                                   const std::vector<std::vector<double>>& predictions) {
    totalRequests.fetch_add(1, std::memory_order_relaxed);
    if (endpoint == "/predict") {
        predictRequests.fetch_add(1, std::memory_order_relaxed);
    } else if (endpoint == "/batch_predict") {
        batchRequests.fetch_add(1, std::memory_order_relaxed);
    }

    totalLatencyMicros.fetch_add(static_cast<uint64_t>(std::max<long long>(0, toLatencyMicros(latencyMs))),
                                 std::memory_order_relaxed);

    uint64_t count = 0;
    for (const auto& prediction : predictions) {
        count += static_cast<uint64_t>(prediction.size());
    }
    totalPredictions.fetch_add(count, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(distributionMutex);
    for (const auto& prediction : predictions) {
        if (predictionStats.size() < prediction.size()) {
            predictionStats.resize(prediction.size());
        }
        for (size_t i = 0; i < prediction.size(); ++i) {
            const double value = prediction[i];
            if (!std::isfinite(value)) continue;
            auto& stat = predictionStats[i];
            stat.count += 1;
            stat.sum += value;
            stat.min = std::min(stat.min, value);
            stat.max = std::max(stat.max, value);
        }
    }
}

void RequestMonitor::recordError(const std::string& endpoint, double latencyMs) {
    totalRequests.fetch_add(1, std::memory_order_relaxed);
    errorRequests.fetch_add(1, std::memory_order_relaxed);
    if (endpoint == "/predict") {
        predictRequests.fetch_add(1, std::memory_order_relaxed);
    } else if (endpoint == "/batch_predict") {
        batchRequests.fetch_add(1, std::memory_order_relaxed);
    }
    totalLatencyMicros.fetch_add(static_cast<uint64_t>(std::max<long long>(0, toLatencyMicros(latencyMs))),
                                 std::memory_order_relaxed);
}

MonitoringSnapshot RequestMonitor::snapshot() const {
    MonitoringSnapshot out;
    out.totalRequests = totalRequests.load(std::memory_order_relaxed);
    out.predictRequests = predictRequests.load(std::memory_order_relaxed);
    out.batchRequests = batchRequests.load(std::memory_order_relaxed);
    out.errorRequests = errorRequests.load(std::memory_order_relaxed);
    out.totalPredictions = totalPredictions.load(std::memory_order_relaxed);

    const uint64_t latencyMicros = totalLatencyMicros.load(std::memory_order_relaxed);
    if (out.totalRequests > 0) {
        out.averageLatencyMs = static_cast<double>(latencyMicros) / static_cast<double>(out.totalRequests) / 1000.0;
    }

    std::lock_guard<std::mutex> lock(distributionMutex);
    out.predictionDistributions.reserve(predictionStats.size());
    for (const auto& stat : predictionStats) {
        PredictionDistributionStats view;
        view.count = stat.count;
        if (stat.count > 0) {
            view.mean = stat.sum / static_cast<double>(stat.count);
            view.min = stat.min;
            view.max = stat.max;
        }
        out.predictionDistributions.push_back(view);
    }
    return out;
}

PredictionService::PredictionService(ModelRegistry& registryRef, RequestMonitor& monitorRef)
    : registry(registryRef), monitor(monitorRef) {}

int PredictionService::start(const Config& config) {
    httplib::Server server;
    server.new_task_queue = [threadCount = std::max<size_t>(1, config.threadCount)] {
        return new httplib::ThreadPool(static_cast<int>(threadCount));
    };

    server.Post("/predict", [this](const httplib::Request& request, httplib::Response& response) {
        const auto started = Clock::now();
        try {
            const JsonValue payload = parseJsonText(request.body);
            if (!payload.isObject()) {
                throw Seldon::ConfigurationException("Request body must be a JSON object");
            }

            const std::string modelId = modelIdFromRequest(payload, registry);
            const std::shared_ptr<ModelRecord> modelRecord = registry.getModel(modelId);
            if (!modelRecord) {
                throw Seldon::ConfigurationException("Unknown model_id: " + modelId);
            }

            const JsonValue* featuresNode = payload.find("features");
            if (featuresNode == nullptr) {
                throw Seldon::ConfigurationException("Request requires 'features' array");
            }
            const std::vector<double> features = parseFeatureVector(*featuresNode, "features");

            std::vector<double> prediction;
            {
                std::lock_guard<std::mutex> guard(modelRecord->inferenceMutex);
                prediction = modelRecord->model.predict(features);
            }

            const auto ended = Clock::now();
            const double latencyMs = std::chrono::duration<double, std::milli>(ended - started).count();
            monitor.recordSuccess("/predict", latencyMs, {prediction});
            const MonitoringSnapshot snapshot = monitor.snapshot();

            setJsonResponse(response,
                            200,
                            makePredictSuccessResponse(modelId,
                                                       prediction,
                                                       latencyMs,
                                                       snapshot.totalRequests));
            logMonitoringLine("/predict", latencyMs, snapshot);
        } catch (const std::exception& e) {
            const auto ended = Clock::now();
            const double latencyMs = std::chrono::duration<double, std::milli>(ended - started).count();
            monitor.recordError("/predict", latencyMs);
            const MonitoringSnapshot snapshot = monitor.snapshot();
            setJsonResponse(response, 400, makeErrorResponse(e.what(), latencyMs));
            logMonitoringLine("/predict", latencyMs, snapshot);
        }
    });

    server.Post("/batch_predict", [this](const httplib::Request& request, httplib::Response& response) {
        const auto started = Clock::now();
        try {
            const JsonValue payload = parseJsonText(request.body);
            if (!payload.isObject()) {
                throw Seldon::ConfigurationException("Request body must be a JSON object");
            }

            const std::string modelId = modelIdFromRequest(payload, registry);
            const std::shared_ptr<ModelRecord> modelRecord = registry.getModel(modelId);
            if (!modelRecord) {
                throw Seldon::ConfigurationException("Unknown model_id: " + modelId);
            }

            const JsonValue* instancesNode = payload.find("instances");
            if (instancesNode == nullptr) {
                throw Seldon::ConfigurationException("Request requires 'instances' 2D array");
            }
            const std::vector<std::vector<double>> instances = parseBatchFeatureVectors(*instancesNode);

            std::vector<std::vector<double>> predictions;
            predictions.reserve(instances.size());
            {
                std::lock_guard<std::mutex> guard(modelRecord->inferenceMutex);
                for (const auto& features : instances) {
                    predictions.push_back(modelRecord->model.predict(features));
                }
            }

            const auto ended = Clock::now();
            const double latencyMs = std::chrono::duration<double, std::milli>(ended - started).count();
            monitor.recordSuccess("/batch_predict", latencyMs, predictions);
            const MonitoringSnapshot snapshot = monitor.snapshot();

            setJsonResponse(response,
                            200,
                            makeBatchSuccessResponse(modelId,
                                                     predictions,
                                                     latencyMs,
                                                     snapshot.totalRequests));
            logMonitoringLine("/batch_predict", latencyMs, snapshot);
        } catch (const std::exception& e) {
            const auto ended = Clock::now();
            const double latencyMs = std::chrono::duration<double, std::milli>(ended - started).count();
            monitor.recordError("/batch_predict", latencyMs);
            const MonitoringSnapshot snapshot = monitor.snapshot();
            setJsonResponse(response, 400, makeErrorResponse(e.what(), latencyMs));
            logMonitoringLine("/batch_predict", latencyMs, snapshot);
        }
    });

    const auto registrySnapshot = registry.metadataSnapshot();
    std::cout << "[SeldonService] loaded_models=" << registrySnapshot.size()
              << " host=" << config.host
              << " port=" << config.port
              << " threads=" << std::max<size_t>(1, config.threadCount)
              << "\n";
    for (const auto& model : registrySnapshot) {
        std::cout << "[SeldonService] model_id=" << model.modelId
                  << " model_path=" << model.modelPath
                  << " training_timestamp=" << model.trainingTimestamp
                  << " rmse=" << model.rmse
                  << " accuracy=" << model.accuracy
                  << " hyperparams=" << model.hyperparameters.size()
                  << "\n";
    }

    if (!server.listen(config.host.c_str(), config.port)) {
        std::cerr << "[SeldonService] failed_to_bind host=" << config.host << " port=" << config.port << "\n";
        return 1;
    }

    return 0;
}
