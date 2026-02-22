#include "AutoConfig.h"
#include "SeldonExceptions.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace {
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            std::string t = trim(cur);
            if (!t.empty()) out.push_back(t);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    std::string t = trim(cur);
    if (!t.empty()) out.push_back(t);
    return out;
}

int parseIntStrict(const std::string& value, const std::string& key, int minValue) {
    try {
        size_t pos = 0;
        int parsed = std::stoi(value, &pos);
        if (pos != value.size()) {
            throw Seldon::ConfigurationException("Invalid integer for " + key + ": " + value);
        }
        if (parsed < minValue) {
            throw Seldon::ConfigurationException("Value for " + key + " must be >= " + std::to_string(minValue));
        }
        return parsed;
    } catch (const Seldon::SeldonException&) {
        throw;
    } catch (const std::exception&) {
        throw Seldon::ConfigurationException("Invalid integer for " + key + ": " + value);
    }
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

void assignKeyValue(AutoConfig& config, const std::string& key, const std::string& value) {
    if (key == "target") config.targetColumn = value;
    else if (key == "dataset") config.datasetPath = value;
    else if (key == "report") config.reportFile = value;
    else if (key == "assets_dir") config.assetsDir = value;
    else if (key == "delimiter") {
        if (value.size() != 1) throw Seldon::ConfigurationException("delimiter expects a single character");
        config.delimiter = value[0];
    }
    else if (key == "outlier_method") config.outlierMethod = lower(value);
    else if (key == "outlier_action") config.outlierAction = lower(value);
    else if (key == "scaling") config.scalingMethod = lower(value);
    else if (key == "kfold") config.kfold = parseIntStrict(value, "kfold", 2);
    else if (key == "plot_format") config.plot.format = lower(value);
    else if (key == "plot_width") config.plot.width = parseIntStrict(value, "plot_width", 320);
    else if (key == "plot_height") config.plot.height = parseIntStrict(value, "plot_height", 240);
    else if (key == "exclude") config.excludedColumns = splitCSV(value);
}
}

AutoConfig AutoConfig::fromArgs(int argc, char* argv[]) {
    if (argc < 2) {
        throw Seldon::ConfigurationException("Usage: seldon <dataset.csv> [--config path] [--target col] [--delimiter ,]");
    }

    AutoConfig config;
    config.datasetPath = argv[1];

    std::string configPath;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            config.targetColumn = argv[++i];
        } else if (arg == "--delimiter" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v.size() != 1) throw Seldon::ConfigurationException("--delimiter expects a single character");
            config.delimiter = v[0];
        }
    }

    if (!configPath.empty()) {
        config = fromFile(configPath, config);
        if (config.datasetPath.empty()) config.datasetPath = argv[1];
    }

    if (config.datasetPath.empty()) {
        throw Seldon::ConfigurationException("Dataset path is required");
    }

    return config;
}

AutoConfig AutoConfig::fromFile(const std::string& configPath, const AutoConfig& base) {
    std::ifstream in(configPath);
    if (!in) throw Seldon::ConfigurationException("Could not open config file: " + configPath);

    AutoConfig config = base;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Support loose YAML (key: value) and loose JSON-ish ("key": "value",)
        line.erase(std::remove(line.begin(), line.end(), '{'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '}'), line.end());
        if (!line.empty() && line.back() == ',') line.pop_back();

        size_t sep = line.find(':');
        if (sep == std::string::npos) continue;

        std::string key = trim(line.substr(0, sep));
        std::string value = trim(line.substr(sep + 1));

        if (!key.empty() && key.front() == '"' && key.back() == '"') key = key.substr(1, key.size() - 2);
        if (!value.empty() && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);

        assignKeyValue(config, key, value);

        // per-column imputation syntax: impute.<column>: <strategy>
        if (key.rfind("impute.", 0) == 0) {
            config.columnImputation[key.substr(7)] = lower(value);
        }
    }

    if (config.plot.format != "png" && config.plot.format != "svg" && config.plot.format != "pdf") {
        throw Seldon::ConfigurationException("plot_format must be one of: png, svg, pdf");
    }

    if (config.outlierMethod != "iqr" && config.outlierMethod != "zscore") {
        throw Seldon::ConfigurationException("outlier_method must be iqr or zscore");
    }

    if (config.outlierAction != "flag" && config.outlierAction != "remove" && config.outlierAction != "cap") {
        throw Seldon::ConfigurationException("outlier_action must be flag, remove, or cap");
    }

    if (config.scalingMethod != "auto" && config.scalingMethod != "zscore" && config.scalingMethod != "minmax" && config.scalingMethod != "none") {
        throw Seldon::ConfigurationException("scaling must be auto, zscore, minmax, or none");
    }

    return config;
}
