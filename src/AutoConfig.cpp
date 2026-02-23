#include "AutoConfig.h"
#include "CommonUtils.h"
#include "SeldonExceptions.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace {
std::vector<std::string> splitCSV(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            std::string t = CommonUtils::trim(cur);
            if (!t.empty()) out.push_back(t);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    std::string t = CommonUtils::trim(cur);
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

uint32_t parseUIntStrict(const std::string& value, const std::string& key) {
    try {
        size_t pos = 0;
        unsigned long parsed = std::stoul(value, &pos);
        if (pos != value.size()) {
            throw Seldon::ConfigurationException("Invalid unsigned integer for " + key + ": " + value);
        }
        return static_cast<uint32_t>(parsed);
    } catch (const Seldon::SeldonException&) {
        throw;
    } catch (const std::exception&) {
        throw Seldon::ConfigurationException("Invalid unsigned integer for " + key + ": " + value);
    }
}

double parseDoubleStrict(const std::string& value, const std::string& key, double minValue) {
    try {
        size_t pos = 0;
        double parsed = std::stod(value, &pos);
        if (pos != value.size()) {
            throw Seldon::ConfigurationException("Invalid number for " + key + ": " + value);
        }
        if (parsed < minValue) {
            throw Seldon::ConfigurationException("Value for " + key + " must be >= " + std::to_string(minValue));
        }
        return parsed;
    } catch (const Seldon::SeldonException&) {
        throw;
    } catch (const std::exception&) {
        throw Seldon::ConfigurationException("Invalid number for " + key + ": " + value);
    }
}

bool parseBoolStrict(const std::string& value, const std::string& key) {
    std::string v = CommonUtils::toLower(CommonUtils::trim(value));
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw Seldon::ConfigurationException("Invalid boolean for " + key + ": " + value);
}

bool isValidImputationStrategy(const std::string& value) {
    static const std::unordered_set<std::string> allowed = {
        "auto", "mean", "median", "zero", "mode", "interpolate"
    };
    return allowed.find(CommonUtils::toLower(CommonUtils::trim(value))) != allowed.end();
}

void applyPlotModes(AutoConfig& config, const std::string& value) {
    config.plotUnivariate = false;
    config.plotOverall = false;
    config.plotBivariateSignificant = false;

    for (const auto& tokenRaw : splitCSV(value)) {
        const std::string token = CommonUtils::toLower(CommonUtils::trim(tokenRaw));
        if (token == "none") {
            config.plotUnivariate = false;
            config.plotOverall = false;
            config.plotBivariateSignificant = false;
        } else if (token == "all") {
            config.plotUnivariate = true;
            config.plotOverall = true;
            config.plotBivariateSignificant = true;
        } else if (token == "univariate") {
            config.plotUnivariate = true;
        } else if (token == "overall") {
            config.plotOverall = true;
        } else if (token == "bivariate" || token == "significant") {
            config.plotBivariateSignificant = true;
        }
    }
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
    else if (key == "outlier_method") config.outlierMethod = CommonUtils::toLower(value);
    else if (key == "outlier_action") config.outlierAction = CommonUtils::toLower(value);
    else if (key == "scaling") config.scalingMethod = CommonUtils::toLower(value);
    else if (key == "kfold") config.kfold = parseIntStrict(value, "kfold", 2);
    else if (key == "max_feature_missing_ratio") {
        config.maxFeatureMissingRatio = parseDoubleStrict(value, "max_feature_missing_ratio", -1.0);
        if (config.maxFeatureMissingRatio > 1.0 || config.maxFeatureMissingRatio < -1.0) {
            throw Seldon::ConfigurationException("max_feature_missing_ratio must be -1 or within [0,1]");
        }
    }
    else if (key == "plot_format") config.plot.format = CommonUtils::toLower(value);
    else if (key == "plot_width") config.plot.width = parseIntStrict(value, "plot_width", 320);
    else if (key == "plot_height") config.plot.height = parseIntStrict(value, "plot_height", 240);
    else if (key == "plot_univariate") config.plotUnivariate = parseBoolStrict(value, "plot_univariate");
    else if (key == "plot_overall") config.plotOverall = parseBoolStrict(value, "plot_overall");
    else if (key == "plot_bivariate_significant") config.plotBivariateSignificant = parseBoolStrict(value, "plot_bivariate_significant");
    else if (key == "generate_html") config.generateHtml = parseBoolStrict(value, "generate_html");
    else if (key == "verbose_analysis") config.verboseAnalysis = parseBoolStrict(value, "verbose_analysis");
    else if (key == "neural_seed") config.neuralSeed = parseUIntStrict(value, "neural_seed");
    else if (key == "benchmark_seed") config.benchmarkSeed = parseUIntStrict(value, "benchmark_seed");
    else if (key == "gradient_clip_norm") config.gradientClipNorm = parseDoubleStrict(value, "gradient_clip_norm", 0.0);
    else if (key == "plots") applyPlotModes(config, value);
    else if (key == "target_strategy") config.targetStrategy = CommonUtils::toLower(value);
    else if (key == "feature_strategy") config.featureStrategy = CommonUtils::toLower(value);
    else if (key == "neural_strategy") config.neuralStrategy = CommonUtils::toLower(value);
    else if (key == "bivariate_strategy") config.bivariateStrategy = CommonUtils::toLower(value);
    else if (key == "fast_mode") config.fastMode = parseBoolStrict(value, "fast_mode");
    else if (key == "fast_max_bivariate_pairs") config.fastMaxBivariatePairs = static_cast<size_t>(parseIntStrict(value, "fast_max_bivariate_pairs", 1));
    else if (key == "fast_neural_sample_rows") config.fastNeuralSampleRows = static_cast<size_t>(parseIntStrict(value, "fast_neural_sample_rows", 1));
    else if (key == "exclude") config.excludedColumns = splitCSV(value);
    else if (key == "feature_min_variance") config.tuning.featureMinVariance = parseDoubleStrict(value, "feature_min_variance", 0.0);
    else if (key == "significance_alpha") config.tuning.significanceAlpha = parseDoubleStrict(value, "significance_alpha", 0.0);
    else if (key == "outlier_iqr_multiplier") config.tuning.outlierIqrMultiplier = parseDoubleStrict(value, "outlier_iqr_multiplier", 0.0);
    else if (key == "outlier_z_threshold") config.tuning.outlierZThreshold = parseDoubleStrict(value, "outlier_z_threshold", 0.0);
    else if (key == "feature_leakage_corr_threshold") config.tuning.featureLeakageCorrThreshold = parseDoubleStrict(value, "feature_leakage_corr_threshold", 0.0);
    else if (key == "feature_missing_q3_offset") config.tuning.featureMissingQ3Offset = parseDoubleStrict(value, "feature_missing_q3_offset", 0.0);
    else if (key == "feature_missing_floor") config.tuning.featureMissingAdaptiveMin = parseDoubleStrict(value, "feature_missing_floor", 0.0);
    else if (key == "feature_missing_ceiling") config.tuning.featureMissingAdaptiveMax = parseDoubleStrict(value, "feature_missing_ceiling", 0.0);
    else if (key == "feature_aggressive_delta") config.tuning.featureAggressiveDelta = parseDoubleStrict(value, "feature_aggressive_delta", 0.0);
    else if (key == "feature_lenient_delta") config.tuning.featureLenientDelta = parseDoubleStrict(value, "feature_lenient_delta", 0.0);
    else if (key == "bivariate_selection_quantile") config.tuning.bivariateSelectionQuantileOverride = parseDoubleStrict(value, "bivariate_selection_quantile", -1.0);
    else if (key == "coherence_weight_small_dataset") config.tuning.coherenceWeightSmallDataset = parseDoubleStrict(value, "coherence_weight_small_dataset", 0.0);
    else if (key == "coherence_weight_regular_dataset") config.tuning.coherenceWeightRegularDataset = parseDoubleStrict(value, "coherence_weight_regular_dataset", 0.0);
    else if (key == "coherence_overfit_penalty_train_ratio") config.tuning.coherenceOverfitPenaltyTrainRatio = parseDoubleStrict(value, "coherence_overfit_penalty_train_ratio", 0.0);
    else if (key == "coherence_benchmark_penalty_ratio") config.tuning.coherenceBenchmarkPenaltyRatio = parseDoubleStrict(value, "coherence_benchmark_penalty_ratio", 0.0);
    else if (key == "coherence_penalty_step") config.tuning.coherencePenaltyStep = parseDoubleStrict(value, "coherence_penalty_step", 0.0);
    else if (key == "coherence_weight_min") config.tuning.coherenceWeightMin = parseDoubleStrict(value, "coherence_weight_min", 0.0);
    else if (key == "coherence_weight_max") config.tuning.coherenceWeightMax = parseDoubleStrict(value, "coherence_weight_max", 0.0);
    else if (key == "corr_heavy_max_importance_threshold") config.tuning.corrHeavyMaxImportanceThreshold = parseDoubleStrict(value, "corr_heavy_max_importance_threshold", 0.0);
    else if (key == "corr_heavy_concentration_threshold") config.tuning.corrHeavyConcentrationThreshold = parseDoubleStrict(value, "corr_heavy_concentration_threshold", 0.0);
    else if (key == "importance_heavy_max_importance_threshold") config.tuning.importanceHeavyMaxImportanceThreshold = parseDoubleStrict(value, "importance_heavy_max_importance_threshold", 0.0);
    else if (key == "importance_heavy_concentration_threshold") config.tuning.importanceHeavyConcentrationThreshold = parseDoubleStrict(value, "importance_heavy_concentration_threshold", 0.0);
    else if (key == "numeric_epsilon") config.tuning.numericEpsilon = parseDoubleStrict(value, "numeric_epsilon", 0.0);
    else if (key == "beta_fallback_intervals_start") config.tuning.betaFallbackIntervalsStart = static_cast<size_t>(parseIntStrict(value, "beta_fallback_intervals_start", 256));
    else if (key == "beta_fallback_intervals_max") config.tuning.betaFallbackIntervalsMax = static_cast<size_t>(parseIntStrict(value, "beta_fallback_intervals_max", 512));
    else if (key == "beta_fallback_tolerance") config.tuning.betaFallbackTolerance = parseDoubleStrict(value, "beta_fallback_tolerance", 0.0);
}
}

AutoConfig AutoConfig::fromArgs(int argc, char* argv[]) {
    if (argc < 2) {
        throw Seldon::ConfigurationException("Usage: seldon <dataset.csv> [--config path] [--target col] [--delimiter ,] [--plots bivariate,univariate,overall] [--plot-univariate true|false] [--plot-overall true|false] [--plot-bivariate true|false] [--generate-html true|false] [--verbose-analysis true|false] [--neural-seed N] [--benchmark-seed N] [--gradient-clip N] [--max-feature-missing-ratio -1|0..1] [--target-strategy auto|quality|max_variance|last_numeric] [--feature-strategy auto|adaptive|aggressive|lenient] [--neural-strategy auto|none|fast|balanced|expressive] [--bivariate-strategy auto|balanced|corr_heavy|importance_heavy] [--fast true|false] [--fast-max-bivariate-pairs N] [--fast-neural-sample-rows N] [--feature-min-variance N] [--feature-leakage-corr-threshold 0..1] [--significance-alpha 0..1] [--outlier-iqr-multiplier N] [--outlier-z-threshold N] [--bivariate-selection-quantile 0..1]");
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
        } else if (arg == "--plots" && i + 1 < argc) {
            applyPlotModes(config, argv[++i]);
        } else if (arg == "--plot-univariate" && i + 1 < argc) {
            config.plotUnivariate = parseBoolStrict(argv[++i], "--plot-univariate");
        } else if (arg == "--plot-overall" && i + 1 < argc) {
            config.plotOverall = parseBoolStrict(argv[++i], "--plot-overall");
        } else if (arg == "--plot-bivariate" && i + 1 < argc) {
            config.plotBivariateSignificant = parseBoolStrict(argv[++i], "--plot-bivariate");
        } else if (arg == "--generate-html" && i + 1 < argc) {
            config.generateHtml = parseBoolStrict(argv[++i], "--generate-html");
        } else if (arg == "--verbose-analysis" && i + 1 < argc) {
            config.verboseAnalysis = parseBoolStrict(argv[++i], "--verbose-analysis");
        } else if (arg == "--neural-seed" && i + 1 < argc) {
            config.neuralSeed = parseUIntStrict(argv[++i], "--neural-seed");
        } else if (arg == "--benchmark-seed" && i + 1 < argc) {
            config.benchmarkSeed = parseUIntStrict(argv[++i], "--benchmark-seed");
        } else if (arg == "--gradient-clip" && i + 1 < argc) {
            config.gradientClipNorm = parseDoubleStrict(argv[++i], "--gradient-clip", 0.0);
        } else if (arg == "--max-feature-missing-ratio" && i + 1 < argc) {
            config.maxFeatureMissingRatio = parseDoubleStrict(argv[++i], "--max-feature-missing-ratio", -1.0);
            if (config.maxFeatureMissingRatio > 1.0 || config.maxFeatureMissingRatio < -1.0) {
                throw Seldon::ConfigurationException("--max-feature-missing-ratio must be -1 or within [0,1]");
            }
        } else if (arg == "--target-strategy" && i + 1 < argc) {
            config.targetStrategy = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--feature-strategy" && i + 1 < argc) {
            config.featureStrategy = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--neural-strategy" && i + 1 < argc) {
            config.neuralStrategy = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--bivariate-strategy" && i + 1 < argc) {
            config.bivariateStrategy = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--fast" && i + 1 < argc) {
            config.fastMode = parseBoolStrict(argv[++i], "--fast");
        } else if (arg == "--fast-max-bivariate-pairs" && i + 1 < argc) {
            config.fastMaxBivariatePairs = static_cast<size_t>(parseIntStrict(argv[++i], "--fast-max-bivariate-pairs", 1));
        } else if (arg == "--fast-neural-sample-rows" && i + 1 < argc) {
            config.fastNeuralSampleRows = static_cast<size_t>(parseIntStrict(argv[++i], "--fast-neural-sample-rows", 1));
        } else if (arg == "--feature-min-variance" && i + 1 < argc) {
            config.tuning.featureMinVariance = parseDoubleStrict(argv[++i], "--feature-min-variance", 0.0);
        } else if (arg == "--significance-alpha" && i + 1 < argc) {
            config.tuning.significanceAlpha = parseDoubleStrict(argv[++i], "--significance-alpha", 0.0);
        } else if (arg == "--outlier-iqr-multiplier" && i + 1 < argc) {
            config.tuning.outlierIqrMultiplier = parseDoubleStrict(argv[++i], "--outlier-iqr-multiplier", 0.0);
        } else if (arg == "--outlier-z-threshold" && i + 1 < argc) {
            config.tuning.outlierZThreshold = parseDoubleStrict(argv[++i], "--outlier-z-threshold", 0.0);
        } else if (arg == "--feature-leakage-corr-threshold" && i + 1 < argc) {
            config.tuning.featureLeakageCorrThreshold = parseDoubleStrict(argv[++i], "--feature-leakage-corr-threshold", 0.0);
        } else if (arg == "--bivariate-selection-quantile" && i + 1 < argc) {
            config.tuning.bivariateSelectionQuantileOverride = parseDoubleStrict(argv[++i], "--bivariate-selection-quantile", -1.0);
        }
    }

    if (!configPath.empty()) {
        config = fromFile(configPath, config);
        if (config.datasetPath.empty()) config.datasetPath = argv[1];
    }

    if (config.datasetPath.empty()) {
        throw Seldon::ConfigurationException("Dataset path is required");
    }

    const auto isIn = [](const std::string& value, const std::vector<std::string>& allowed) {
        return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
    };
    if (!isIn(config.targetStrategy, {"auto", "quality", "max_variance", "last_numeric"})) {
        throw Seldon::ConfigurationException("--target-strategy must be one of: auto, quality, max_variance, last_numeric");
    }
    if (!isIn(config.featureStrategy, {"auto", "adaptive", "aggressive", "lenient"})) {
        throw Seldon::ConfigurationException("--feature-strategy must be one of: auto, adaptive, aggressive, lenient");
    }
    if (!isIn(config.neuralStrategy, {"auto", "none", "fast", "balanced", "expressive"})) {
        throw Seldon::ConfigurationException("--neural-strategy must be one of: auto, none, fast, balanced, expressive");
    }
    if (!isIn(config.bivariateStrategy, {"auto", "balanced", "corr_heavy", "importance_heavy"})) {
        throw Seldon::ConfigurationException("--bivariate-strategy must be one of: auto, balanced, corr_heavy, importance_heavy");
    }
    if (config.tuning.featureLeakageCorrThreshold < 0.0 || config.tuning.featureLeakageCorrThreshold > 1.0) {
        throw Seldon::ConfigurationException("--feature-leakage-corr-threshold must be within [0,1]");
    }
    if (config.tuning.significanceAlpha <= 0.0 || config.tuning.significanceAlpha >= 1.0) {
        throw Seldon::ConfigurationException("--significance-alpha must be within (0,1)");
    }
    if (config.tuning.outlierIqrMultiplier <= 0.0) {
        throw Seldon::ConfigurationException("--outlier-iqr-multiplier must be > 0");
    }
    if (config.tuning.outlierZThreshold <= 0.0) {
        throw Seldon::ConfigurationException("--outlier-z-threshold must be > 0");
    }
    if (config.tuning.featureMissingAdaptiveMin > config.tuning.featureMissingAdaptiveMax) {
        throw Seldon::ConfigurationException("feature_missing_floor must be <= feature_missing_ceiling");
    }
    if (config.tuning.bivariateSelectionQuantileOverride != -1.0 &&
        (config.tuning.bivariateSelectionQuantileOverride < 0.0 || config.tuning.bivariateSelectionQuantileOverride > 1.0)) {
        throw Seldon::ConfigurationException("bivariate_selection_quantile must be -1 or within [0,1]");
    }
    if (config.tuning.coherenceWeightMin > config.tuning.coherenceWeightMax) {
        throw Seldon::ConfigurationException("coherence_weight_min must be <= coherence_weight_max");
    }
    if (config.tuning.betaFallbackIntervalsStart > config.tuning.betaFallbackIntervalsMax) {
        throw Seldon::ConfigurationException("beta_fallback_intervals_start must be <= beta_fallback_intervals_max");
    }
    if (config.tuning.numericEpsilon <= 0.0 || config.tuning.betaFallbackTolerance <= 0.0) {
        throw Seldon::ConfigurationException("numeric_epsilon and beta_fallback_tolerance must be > 0");
    }
    if (config.fastMaxBivariatePairs == 0 || config.fastNeuralSampleRows == 0) {
        throw Seldon::ConfigurationException("fast_max_bivariate_pairs and fast_neural_sample_rows must be > 0");
    }
    for (const auto& kv : config.columnImputation) {
        if (!isValidImputationStrategy(kv.second)) {
            throw Seldon::ConfigurationException(
                "Invalid imputation strategy for column '" + kv.first +
                "': '" + kv.second + "' (allowed: auto, mean, median, zero, mode, interpolate)");
        }
    }

    return config;
}

AutoConfig AutoConfig::fromFile(const std::string& configPath, const AutoConfig& base) {
    std::ifstream in(configPath);
    if (!in) throw Seldon::ConfigurationException("Could not open config file: " + configPath);

    AutoConfig config = base;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        line = CommonUtils::trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Support loose YAML (key: value) and loose JSON-ish ("key": "value",)
        line.erase(std::remove(line.begin(), line.end(), '{'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '}'), line.end());
        if (!line.empty() && line.back() == ',') line.pop_back();

        size_t sep = line.find(':');
        if (sep == std::string::npos) continue;

        std::string key = CommonUtils::trim(line.substr(0, sep));
        std::string value = CommonUtils::trim(line.substr(sep + 1));

        if (!key.empty() && key.front() == '"' && key.back() == '"') key = key.substr(1, key.size() - 2);
        if (!value.empty() && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);

        try {
            assignKeyValue(config, key, value);
        } catch (const Seldon::SeldonException& ex) {
            throw Seldon::ConfigurationException(
                "Config parse error at line " + std::to_string(lineNo) +
                ": '" + line + "' -> " + ex.what());
        }

        // per-column imputation syntax: impute.<column>: <strategy>
        if (key.rfind("impute.", 0) == 0) {
            const std::string normalized = CommonUtils::toLower(value);
            if (!isValidImputationStrategy(normalized)) {
                throw Seldon::ConfigurationException(
                    "Config parse error at line " + std::to_string(lineNo) +
                    ": invalid imputation strategy '" + value +
                    "' (allowed: auto, mean, median, zero, mode, interpolate)");
            }
            config.columnImputation[key.substr(7)] = normalized;
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

    if (config.tuning.significanceAlpha <= 0.0 || config.tuning.significanceAlpha >= 1.0) {
        throw Seldon::ConfigurationException("significance_alpha must be within (0,1)");
    }
    if (config.tuning.outlierIqrMultiplier <= 0.0) {
        throw Seldon::ConfigurationException("outlier_iqr_multiplier must be > 0");
    }
    if (config.tuning.outlierZThreshold <= 0.0) {
        throw Seldon::ConfigurationException("outlier_z_threshold must be > 0");
    }

    const auto isIn = [](const std::string& value, const std::vector<std::string>& allowed) {
        return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
    };
    if (!isIn(config.targetStrategy, {"auto", "quality", "max_variance", "last_numeric"})) {
        throw Seldon::ConfigurationException("target_strategy must be one of: auto, quality, max_variance, last_numeric");
    }
    if (!isIn(config.featureStrategy, {"auto", "adaptive", "aggressive", "lenient"})) {
        throw Seldon::ConfigurationException("feature_strategy must be one of: auto, adaptive, aggressive, lenient");
    }
    if (!isIn(config.neuralStrategy, {"auto", "none", "fast", "balanced", "expressive"})) {
        throw Seldon::ConfigurationException("neural_strategy must be one of: auto, none, fast, balanced, expressive");
    }
    if (!isIn(config.bivariateStrategy, {"auto", "balanced", "corr_heavy", "importance_heavy"})) {
        throw Seldon::ConfigurationException("bivariate_strategy must be one of: auto, balanced, corr_heavy, importance_heavy");
    }
    if (config.tuning.featureLeakageCorrThreshold < 0.0 || config.tuning.featureLeakageCorrThreshold > 1.0) {
        throw Seldon::ConfigurationException("feature_leakage_corr_threshold must be within [0,1]");
    }
    if (config.tuning.featureMissingAdaptiveMin > config.tuning.featureMissingAdaptiveMax) {
        throw Seldon::ConfigurationException("feature_missing_floor must be <= feature_missing_ceiling");
    }
    if (config.tuning.bivariateSelectionQuantileOverride != -1.0 &&
        (config.tuning.bivariateSelectionQuantileOverride < 0.0 || config.tuning.bivariateSelectionQuantileOverride > 1.0)) {
        throw Seldon::ConfigurationException("bivariate_selection_quantile must be -1 or within [0,1]");
    }
    if (config.tuning.coherenceWeightMin > config.tuning.coherenceWeightMax) {
        throw Seldon::ConfigurationException("coherence_weight_min must be <= coherence_weight_max");
    }
    if (config.tuning.betaFallbackIntervalsStart > config.tuning.betaFallbackIntervalsMax) {
        throw Seldon::ConfigurationException("beta_fallback_intervals_start must be <= beta_fallback_intervals_max");
    }
    if (config.tuning.numericEpsilon <= 0.0 || config.tuning.betaFallbackTolerance <= 0.0) {
        throw Seldon::ConfigurationException("numeric_epsilon and beta_fallback_tolerance must be > 0");
    }
    if (config.fastMaxBivariatePairs == 0 || config.fastNeuralSampleRows == 0) {
        throw Seldon::ConfigurationException("fast_max_bivariate_pairs and fast_neural_sample_rows must be > 0");
    }
    for (const auto& kv : config.columnImputation) {
        if (!isValidImputationStrategy(kv.second)) {
            throw Seldon::ConfigurationException(
                "Invalid imputation strategy for column '" + kv.first +
                "': '" + kv.second + "' (allowed: auto, mean, median, zero, mode, interpolate)");
        }
    }

    return config;
}
