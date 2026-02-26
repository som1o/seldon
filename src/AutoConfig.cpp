#include "AutoConfig.h"
#include "CommonUtils.h"
#include "SeldonExceptions.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
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

template <typename T, typename Parser>
T parseNumericStrict(const std::string& value,
                     const std::string& key,
                     const std::string& errorPrefix,
                     Parser parser) {
    try {
        size_t pos = 0;
        T parsed = parser(value, &pos);
        if (pos != value.size()) {
            throw Seldon::ConfigurationException(errorPrefix + key + ": " + value);
        }
        return parsed;
    } catch (const Seldon::SeldonException&) {
        throw;
    } catch (const std::exception& ex) {
        throw Seldon::ConfigurationException(errorPrefix + key + ": " + value + " (" + ex.what() + ")");
    }
}

std::string stripStructuralTokensOutsideQuotes(const std::string& line) {
    std::string out;
    out.reserve(line.size());

    bool inQuotes = false;
    bool escaped = false;
    for (char c : line) {
        if (escaped) {
            out.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            out.push_back(c);
            escaped = true;
            continue;
        }
        if (c == '"') {
            inQuotes = !inQuotes;
            out.push_back(c);
            continue;
        }
        if (!inQuotes && (c == '{' || c == '}')) {
            continue;
        }
        out.push_back(c);
    }

    size_t lastNonSpace = out.find_last_not_of(" \t\r\n");
    if (lastNonSpace != std::string::npos && out[lastNonSpace] == ',') {
        out.erase(lastNonSpace, 1);
    }
    return out;
}

size_t findSeparatorOutsideQuotes(const std::string& line, char sep) {
    bool inQuotes = false;
    bool escaped = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if (!inQuotes && c == sep) {
            return i;
        }
    }
    return std::string::npos;
}

std::string maybeUnquote(std::string value) {
    value = CommonUtils::trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::string normalizeConfigKey(std::string key) {
    key = CommonUtils::trim(key);
    const std::string lowered = CommonUtils::toLower(key);
    if (lowered.rfind("impute.", 0) == 0) {
        const size_t dotPos = key.find('.');
        if (dotPos != std::string::npos && dotPos + 1 < key.size()) {
            return "impute." + key.substr(dotPos + 1);
        }
        return "impute.";
    }
    if (lowered.rfind("type.", 0) == 0) {
        const size_t dotPos = key.find('.');
        if (dotPos != std::string::npos && dotPos + 1 < key.size()) {
            return "type." + key.substr(dotPos + 1);
        }
        return "type.";
    }

    std::string out = lowered;
    std::replace(out.begin(), out.end(), '-', '_');
    return out;
}

int parseIntStrict(const std::string& value, const std::string& key, int minValue) {
    int parsed = parseNumericStrict<int>(
        value,
        key,
        "Invalid integer for ",
        [](const std::string& v, size_t* pos) { return std::stoi(v, pos); });
    if (parsed < minValue) {
        throw Seldon::ConfigurationException("Value for " + key + " must be >= " + std::to_string(minValue));
    }
    return parsed;
}

uint32_t parseUIntStrict(const std::string& value, const std::string& key) {
    unsigned long parsed = parseNumericStrict<unsigned long>(
        value,
        key,
        "Invalid unsigned integer for ",
        [](const std::string& v, size_t* pos) { return std::stoul(v, pos); });
    if (parsed > static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())) {
        throw Seldon::ConfigurationException("Value for " + key + " exceeds uint32 range");
    }
    return static_cast<uint32_t>(parsed);
}

double parseDoubleStrict(const std::string& value, const std::string& key, double minValue) {
    double parsed = parseNumericStrict<double>(
        value,
        key,
        "Invalid number for ",
        [](const std::string& v, size_t* pos) { return std::stod(v, pos); });
    if (parsed < minValue) {
        throw Seldon::ConfigurationException("Value for " + key + " must be >= " + std::to_string(minValue));
    }
    return parsed;
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

bool isValidColumnTypeOverride(const std::string& value) {
    static const std::unordered_set<std::string> allowed = {
        "numeric", "categorical", "datetime"
    };
    return allowed.find(CommonUtils::toLower(CommonUtils::trim(value))) != allowed.end();
}

void applyProfile(AutoConfig& config, const std::string& profileRaw) {
    const std::string p = CommonUtils::toLower(CommonUtils::trim(profileRaw));
    if (p.empty() || p == "auto") {
        config.profile = "auto";
        return;
    }

    config.profile = p;
    if (p == "quick") {
        config.fastMode = true;
        config.plotUnivariate = false;
        config.plotOverall = false;
        config.plotBivariateSignificant = true;
        config.plotModesExplicit = true;
        config.generateHtml = false;
        config.fastMaxBivariatePairs = std::min<size_t>(config.fastMaxBivariatePairs, 1200);
        config.fastNeuralSampleRows = std::min<size_t>(config.fastNeuralSampleRows, 12000);
        config.neuralStrategy = "fast";
    } else if (p == "thorough") {
        config.fastMode = false;
        config.plotUnivariate = true;
        config.plotOverall = true;
        config.plotBivariateSignificant = true;
        config.plotModesExplicit = true;
        config.neuralStrategy = "expressive";
        config.neuralImportanceMaxRows = std::max<size_t>(config.neuralImportanceMaxRows, 1000);
        config.neuralIntegratedGradSteps = std::max<size_t>(config.neuralIntegratedGradSteps, 8);
    } else if (p == "minimal") {
        config.fastMode = true;
        config.plotUnivariate = false;
        config.plotOverall = false;
        config.plotBivariateSignificant = false;
        config.plotModesExplicit = true;
        config.generateHtml = false;
        config.neuralStrategy = "fast";
        config.fastMaxBivariatePairs = std::min<size_t>(config.fastMaxBivariatePairs, 500);
    }
}

void applyLowMemoryDefaults(AutoConfig& config) {
    config.lowMemoryMode = true;
    config.fastMode = true;
    config.profile = (config.profile == "auto") ? "quick" : config.profile;

    config.fastMaxBivariatePairs = std::min<size_t>(config.fastMaxBivariatePairs, 600);
    config.fastNeuralSampleRows = std::min<size_t>(config.fastNeuralSampleRows, 10000);
    config.neuralStrategy = (config.neuralStrategy == "auto") ? "fast" : config.neuralStrategy;
    config.neuralStreamingMode = true;
    config.neuralStreamingChunkRows = std::min<size_t>(std::max<size_t>(128, config.neuralStreamingChunkRows), 512);
    config.neuralMaxOneHotPerColumn = std::min<size_t>(config.neuralMaxOneHotPerColumn, 8);

    config.featureEngineeringEnablePoly = false;
    config.featureEngineeringEnableLog = false;
    config.featureEngineeringEnableRatioProductDiscovery = false;
    config.featureEngineeringMaxPairwiseDiscovery = std::min<size_t>(config.featureEngineeringMaxPairwiseDiscovery, 8);
    config.featureEngineeringMaxGeneratedColumns = std::min<size_t>(config.featureEngineeringMaxGeneratedColumns, 64);

    config.plotUnivariate = false;
    config.plotOverall = false;
    config.plotBivariateSignificant = true;
    config.plotModesExplicit = true;
    config.generateHtml = false;

    config.tuning.overallCorrHeatmapMaxColumns = std::min<size_t>(config.tuning.overallCorrHeatmapMaxColumns, 24);
}

AutoConfig runInteractiveWizard() {
    AutoConfig cfg;
    cfg.interactiveMode = true;

    auto ask = [](const std::string& prompt, const std::string& fallback = "") {
        std::cout << prompt;
        std::string value;
        std::getline(std::cin, value);
        value = CommonUtils::trim(value);
        return value.empty() ? fallback : value;
    };

    cfg.datasetPath = ask("Dataset path (.csv/.csv.gz/.csv.zip/.xlsx/.xls): ");
    cfg.targetColumn = ask("Target column (optional): ");
    applyProfile(cfg, ask("Profile [quick/thorough/minimal/auto] (auto): ", "auto"));
    cfg.plotUnivariate = parseBoolStrict(ask("Enable univariate plots? [true/false] (false): ", "false"), "interactive.plot_univariate");
    cfg.plotOverall = parseBoolStrict(ask("Enable overall plots? [true/false] (false): ", "false"), "interactive.plot_overall");
    cfg.plotBivariateSignificant = parseBoolStrict(ask("Enable significant bivariate plots? [true/false] (true): ", "true"), "interactive.plot_bivariate");
    cfg.plotModesExplicit = true;
    cfg.neuralStrategy = CommonUtils::toLower(ask("Neural strategy [auto/fast/balanced/expressive] (auto): ", "auto"));
    cfg.bivariateStrategy = CommonUtils::toLower(ask("Bivariate strategy [auto/balanced/corr_heavy/importance_heavy] (auto): ", "auto"));
    cfg.exportPreprocessed = CommonUtils::toLower(ask("Export preprocessed dataset [none/csv/parquet] (none): ", "none"));

    const std::string outputCfgPath = ask("Write config file path (seldon_interactive.yaml): ", "seldon_interactive.yaml");
    std::ofstream out(outputCfgPath);
    if (out) {
        out << "dataset: " << cfg.datasetPath << "\n";
        if (!cfg.targetColumn.empty()) out << "target: " << cfg.targetColumn << "\n";
        out << "profile: " << cfg.profile << "\n";
        out << "plot_univariate: " << (cfg.plotUnivariate ? "true" : "false") << "\n";
        out << "plot_overall: " << (cfg.plotOverall ? "true" : "false") << "\n";
        out << "plot_bivariate_significant: " << (cfg.plotBivariateSignificant ? "true" : "false") << "\n";
        out << "neural_strategy: " << cfg.neuralStrategy << "\n";
        out << "bivariate_strategy: " << cfg.bivariateStrategy << "\n";
        out << "export_preprocessed: " << cfg.exportPreprocessed << "\n";
        std::cout << "[Seldon] Interactive config saved to " << outputCfgPath << "\n";
    }

    return cfg;
}

void applyPlotModes(AutoConfig& config, const std::string& value) {
    config.plotModesExplicit = true;
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
    if (key == "delimiter") {
        if (value.size() != 1) throw Seldon::ConfigurationException("delimiter expects a single character");
        config.delimiter = value[0];
        return;
    }
    if (key == "max_feature_missing_ratio") {
        config.maxFeatureMissingRatio = parseDoubleStrict(value, "max_feature_missing_ratio", -1.0);
        if (config.maxFeatureMissingRatio > 1.0 || config.maxFeatureMissingRatio < -1.0) {
            throw Seldon::ConfigurationException("max_feature_missing_ratio must be -1 or within [0,1]");
        }
        return;
    }
    if (key == "plots") {
        applyPlotModes(config, value);
        return;
    }
    if (key == "plot_univariate" || key == "plot_overall" || key == "plot_bivariate_significant") {
        config.plotModesExplicit = true;
    }
    if (key == "exclude") {
        config.excludedColumns = splitCSV(value);
        return;
    }

    struct IntRule {
        int AutoConfig::*member;
        int minValue;
    };
    struct SizeRule {
        size_t AutoConfig::*member;
        int minValue;
    };
    struct DoubleRule {
        double AutoConfig::*member;
        double minValue;
    };
    struct PlotIntRule {
        int PlotConfig::*member;
        int minValue;
    };
    struct PlotDoubleRule {
        double PlotConfig::*member;
        double minValue;
    };
    struct TuningDoubleRule {
        double HeuristicTuningConfig::*member;
        double minValue;
    };
    struct TuningSizeRule {
        size_t HeuristicTuningConfig::*member;
        int minValue;
    };

    static const std::unordered_map<std::string, std::string AutoConfig::*> rawStringFields = {
        {"target", &AutoConfig::targetColumn},
        {"dataset", &AutoConfig::datasetPath},
        {"report", &AutoConfig::reportFile},
        {"assets_dir", &AutoConfig::assetsDir},
        {"output_dir", &AutoConfig::outputDir},
        {"export_preprocessed_path", &AutoConfig::exportPreprocessedPath}
    };
    static const std::unordered_map<std::string, std::string AutoConfig::*> lowerStringFields = {
        {"outlier_method", &AutoConfig::outlierMethod},
        {"outlier_action", &AutoConfig::outlierAction},
        {"scaling", &AutoConfig::scalingMethod},
        {"profile", &AutoConfig::profile},
        {"datetime_locale_hint", &AutoConfig::datetimeLocaleHint},
        {"numeric_locale_hint", &AutoConfig::numericLocaleHint},
        {"export_preprocessed", &AutoConfig::exportPreprocessed},
        {"neural_optimizer", &AutoConfig::neuralOptimizer},
        {"neural_lookahead_fast_optimizer", &AutoConfig::neuralLookaheadFastOptimizer},
        {"neural_explainability", &AutoConfig::neuralExplainability},
        {"target_strategy", &AutoConfig::targetStrategy},
        {"feature_strategy", &AutoConfig::featureStrategy},
        {"neural_strategy", &AutoConfig::neuralStrategy},
        {"bivariate_strategy", &AutoConfig::bivariateStrategy}
    };
    static const std::unordered_map<std::string, bool AutoConfig::*> boolFields = {
        {"plot_univariate", &AutoConfig::plotUnivariate},
        {"plot_overall", &AutoConfig::plotOverall},
        {"plot_bivariate_significant", &AutoConfig::plotBivariateSignificant},
        {"generate_html", &AutoConfig::generateHtml},
        {"verbose_analysis", &AutoConfig::verboseAnalysis},
        {"neural_use_batch_norm", &AutoConfig::neuralUseBatchNorm},
        {"neural_use_cosine_annealing", &AutoConfig::neuralUseCosineAnnealing},
        {"neural_use_cyclical_lr", &AutoConfig::neuralUseCyclicalLr},
        {"neural_use_layer_norm", &AutoConfig::neuralUseLayerNorm},
        {"neural_use_validation_loss_ema", &AutoConfig::neuralUseValidationLossEma},
        {"neural_use_adaptive_gradient_clipping", &AutoConfig::neuralUseAdaptiveGradientClipping},
        {"neural_use_ema_weights", &AutoConfig::neuralUseEmaWeights},
        {"fast_mode", &AutoConfig::fastMode},
        {"low_memory_mode", &AutoConfig::lowMemoryMode},
        {"neural_streaming_mode", &AutoConfig::neuralStreamingMode},
        {"neural_multi_output", &AutoConfig::neuralMultiOutput},
        {"neural_importance_parallel", &AutoConfig::neuralImportanceParallel},
        {"feature_engineering_enable_poly", &AutoConfig::featureEngineeringEnablePoly},
        {"feature_engineering_enable_log", &AutoConfig::featureEngineeringEnableLog},
        {"feature_engineering_enable_ratio_product_discovery", &AutoConfig::featureEngineeringEnableRatioProductDiscovery},
        {"store_outlier_flags_in_report", &AutoConfig::storeOutlierFlagsInReport}
    };
    static const std::unordered_map<std::string, uint32_t AutoConfig::*> uintFields = {
        {"neural_seed", &AutoConfig::neuralSeed},
        {"benchmark_seed", &AutoConfig::benchmarkSeed}
    };
    static const std::unordered_map<std::string, IntRule> intFields = {
        {"kfold", {&AutoConfig::kfold, 2}},
        {"neural_lookahead_sync_period", {&AutoConfig::neuralLookaheadSyncPeriod, 1}},
        {"neural_lr_plateau_patience", {&AutoConfig::neuralLrPlateauPatience, 1}},
        {"neural_lr_cooldown_epochs", {&AutoConfig::neuralLrCooldownEpochs, 0}},
        {"neural_max_lr_reductions", {&AutoConfig::neuralMaxLrReductions, 0}},
        {"neural_lr_warmup_epochs", {&AutoConfig::neuralLrWarmupEpochs, 0}},
        {"neural_lr_cycle_epochs", {&AutoConfig::neuralLrCycleEpochs, 2}},
        {"neural_gradient_accumulation_steps", {&AutoConfig::neuralGradientAccumulationSteps, 1}},
        {"neural_min_layers", {&AutoConfig::neuralMinLayers, 1}},
        {"neural_max_layers", {&AutoConfig::neuralMaxLayers, 1}},
        {"neural_fixed_layers", {&AutoConfig::neuralFixedLayers, 0}},
        {"neural_fixed_hidden_nodes", {&AutoConfig::neuralFixedHiddenNodes, 0}},
        {"neural_max_hidden_nodes", {&AutoConfig::neuralMaxHiddenNodes, 4}},
        {"feature_engineering_degree", {&AutoConfig::featureEngineeringDegree, 1}}
    };
    static const std::unordered_map<std::string, SizeRule> sizeFields = {
        {"fast_max_bivariate_pairs", {&AutoConfig::fastMaxBivariatePairs, 1}},
        {"fast_neural_sample_rows", {&AutoConfig::fastNeuralSampleRows, 1}},
        {"neural_streaming_chunk_rows", {&AutoConfig::neuralStreamingChunkRows, 16}},
        {"neural_max_aux_targets", {&AutoConfig::neuralMaxAuxTargets, 0}},
        {"neural_integrated_grad_steps", {&AutoConfig::neuralIntegratedGradSteps, 4}},
        {"neural_uncertainty_samples", {&AutoConfig::neuralUncertaintySamples, 4}},
        {"neural_importance_max_rows", {&AutoConfig::neuralImportanceMaxRows, 128}},
        {"neural_importance_trials", {&AutoConfig::neuralImportanceTrials, 0}},
        {"feature_engineering_max_base", {&AutoConfig::featureEngineeringMaxBase, 2}},
        {"feature_engineering_max_pairwise_discovery", {&AutoConfig::featureEngineeringMaxPairwiseDiscovery, 2}},
        {"neural_max_one_hot_per_column", {&AutoConfig::neuralMaxOneHotPerColumn, 1}},
        {"neural_max_topology_nodes", {&AutoConfig::neuralMaxTopologyNodes, 16}},
        {"neural_max_trainable_params", {&AutoConfig::neuralMaxTrainableParams, 1024}},
        {"feature_engineering_max_generated_columns", {&AutoConfig::featureEngineeringMaxGeneratedColumns, 16}}
    };
    static const std::unordered_map<std::string, DoubleRule> doubleFields = {
        {"gradient_clip_norm", {&AutoConfig::gradientClipNorm, 0.0}},
        {"neural_learning_rate", {&AutoConfig::neuralLearningRate, 1e-8}},
        {"neural_lookahead_alpha", {&AutoConfig::neuralLookaheadAlpha, 0.0}},
        {"neural_batch_norm_momentum", {&AutoConfig::neuralBatchNormMomentum, 0.0}},
        {"neural_batch_norm_epsilon", {&AutoConfig::neuralBatchNormEpsilon, 1e-12}},
        {"neural_layer_norm_epsilon", {&AutoConfig::neuralLayerNormEpsilon, 1e-12}},
        {"neural_lr_decay", {&AutoConfig::neuralLrDecay, 0.0}},
        {"neural_lr_schedule_min_factor", {&AutoConfig::neuralLrScheduleMinFactor, 0.0}},
        {"neural_min_learning_rate", {&AutoConfig::neuralMinLearningRate, 0.0}},
        {"neural_validation_loss_ema_beta", {&AutoConfig::neuralValidationLossEmaBeta, 0.0}},
        {"neural_categorical_input_l2_boost", {&AutoConfig::neuralCategoricalInputL2Boost, 0.0}},
        {"neural_adaptive_clip_beta", {&AutoConfig::neuralAdaptiveClipBeta, 0.0}},
        {"neural_adaptive_clip_multiplier", {&AutoConfig::neuralAdaptiveClipMultiplier, 0.0}},
        {"neural_adaptive_clip_min", {&AutoConfig::neuralAdaptiveClipMin, 0.0}},
        {"neural_gradient_noise_std", {&AutoConfig::neuralGradientNoiseStd, 0.0}},
        {"neural_gradient_noise_decay", {&AutoConfig::neuralGradientNoiseDecay, 0.0}},
        {"neural_ema_decay", {&AutoConfig::neuralEmaDecay, 0.0}},
        {"neural_label_smoothing", {&AutoConfig::neuralLabelSmoothing, 0.0}}
    };
    static const std::unordered_map<std::string, PlotIntRule> plotIntFields = {
        {"plot_width", {&PlotConfig::width, 320}},
        {"plot_height", {&PlotConfig::height, 240}}
    };
    static const std::unordered_map<std::string, PlotDoubleRule> plotDoubleFields = {
        {"plot_point_size", {&PlotConfig::pointSize, 0.1}},
        {"plot_line_width", {&PlotConfig::lineWidth, 0.1}}
    };
    static const std::unordered_map<std::string, TuningDoubleRule> tuningDoubleFields = {
        {"feature_min_variance", {&HeuristicTuningConfig::featureMinVariance, 0.0}},
        {"significance_alpha", {&HeuristicTuningConfig::significanceAlpha, 0.0}},
        {"outlier_iqr_multiplier", {&HeuristicTuningConfig::outlierIqrMultiplier, 0.0}},
        {"outlier_z_threshold", {&HeuristicTuningConfig::outlierZThreshold, 0.0}},
        {"feature_leakage_corr_threshold", {&HeuristicTuningConfig::featureLeakageCorrThreshold, 0.0}},
        {"feature_missing_q3_offset", {&HeuristicTuningConfig::featureMissingQ3Offset, 0.0}},
        {"feature_missing_floor", {&HeuristicTuningConfig::featureMissingAdaptiveMin, 0.0}},
        {"feature_missing_ceiling", {&HeuristicTuningConfig::featureMissingAdaptiveMax, 0.0}},
        {"feature_aggressive_delta", {&HeuristicTuningConfig::featureAggressiveDelta, 0.0}},
        {"feature_lenient_delta", {&HeuristicTuningConfig::featureLenientDelta, 0.0}},
        {"coherence_weight_small_dataset", {&HeuristicTuningConfig::coherenceWeightSmallDataset, 0.0}},
        {"coherence_weight_regular_dataset", {&HeuristicTuningConfig::coherenceWeightRegularDataset, 0.0}},
        {"coherence_overfit_penalty_train_ratio", {&HeuristicTuningConfig::coherenceOverfitPenaltyTrainRatio, 0.0}},
        {"coherence_benchmark_penalty_ratio", {&HeuristicTuningConfig::coherenceBenchmarkPenaltyRatio, 0.0}},
        {"coherence_penalty_step", {&HeuristicTuningConfig::coherencePenaltyStep, 0.0}},
        {"coherence_weight_min", {&HeuristicTuningConfig::coherenceWeightMin, 0.0}},
        {"coherence_weight_max", {&HeuristicTuningConfig::coherenceWeightMax, 0.0}},
        {"corr_heavy_max_importance_threshold", {&HeuristicTuningConfig::corrHeavyMaxImportanceThreshold, 0.0}},
        {"corr_heavy_concentration_threshold", {&HeuristicTuningConfig::corrHeavyConcentrationThreshold, 0.0}},
        {"importance_heavy_max_importance_threshold", {&HeuristicTuningConfig::importanceHeavyMaxImportanceThreshold, 0.0}},
        {"importance_heavy_concentration_threshold", {&HeuristicTuningConfig::importanceHeavyConcentrationThreshold, 0.0}},
        {"numeric_epsilon", {&HeuristicTuningConfig::numericEpsilon, 0.0}},
        {"beta_fallback_tolerance", {&HeuristicTuningConfig::betaFallbackTolerance, 0.0}},
        {"bivariate_selection_quantile", {&HeuristicTuningConfig::bivariateSelectionQuantileOverride, -1.0}},
        {"bivariate_tier3_fallback_aggressiveness", {&HeuristicTuningConfig::bivariateTier3FallbackAggressiveness, 0.0}},
        {"box_plot_min_iqr", {&HeuristicTuningConfig::boxPlotMinIqr, 0.0}},
        {"pie_max_dominance_ratio", {&HeuristicTuningConfig::pieMaxDominanceRatio, 0.0}},
        {"scatter_fit_min_abs_corr", {&HeuristicTuningConfig::scatterFitMinAbsCorr, 0.0}},
        {"hybrid_explainability_weight_permutation", {&HeuristicTuningConfig::hybridExplainabilityWeightPermutation, 0.0}},
        {"hybrid_explainability_weight_integrated_gradients", {&HeuristicTuningConfig::hybridExplainabilityWeightIntegratedGradients, 0.0}},
        {"gantt_duration_hours_threshold", {&HeuristicTuningConfig::ganttDurationHoursThreshold, 0.0}},
        {"lof_fallback_modified_z_threshold", {&HeuristicTuningConfig::lofFallbackModifiedZThreshold, 0.0}},
        {"lof_threshold_floor", {&HeuristicTuningConfig::lofThresholdFloor, 0.0}}
    };
    static const std::unordered_map<std::string, TuningSizeRule> tuningSizeFields = {
        {"beta_fallback_intervals_start", {&HeuristicTuningConfig::betaFallbackIntervalsStart, 256}},
        {"beta_fallback_intervals_max", {&HeuristicTuningConfig::betaFallbackIntervalsMax, 512}},
        {"overall_corr_heatmap_max_columns", {&HeuristicTuningConfig::overallCorrHeatmapMaxColumns, 2}},
        {"ogive_min_points", {&HeuristicTuningConfig::ogiveMinPoints, 2}},
        {"ogive_min_unique", {&HeuristicTuningConfig::ogiveMinUnique, 2}},
        {"box_plot_min_points", {&HeuristicTuningConfig::boxPlotMinPoints, 3}},
        {"pie_min_categories", {&HeuristicTuningConfig::pieMinCategories, 2}},
        {"pie_max_categories", {&HeuristicTuningConfig::pieMaxCategories, 2}},
        {"scatter_fit_min_sample_size", {&HeuristicTuningConfig::scatterFitMinSampleSize, 3}},
        {"time_series_season_period", {&HeuristicTuningConfig::timeSeriesSeasonPeriod, 2}},
        {"lof_max_rows", {&HeuristicTuningConfig::lofMaxRows, 100}},
        {"gantt_min_tasks", {&HeuristicTuningConfig::ganttMinTasks, 1}},
        {"gantt_max_tasks", {&HeuristicTuningConfig::ganttMaxTasks, 1}}
    };

    if (key == "plot_format") {
        config.plot.format = CommonUtils::toLower(value);
        return;
    }
    if (key == "plot_theme") {
        config.plot.theme = CommonUtils::toLower(value);
        return;
    }
    if (key == "plot_grid") {
        config.plot.showGrid = parseBoolStrict(value, key);
        return;
    }
    if (key == "gantt_auto_enabled") {
        config.tuning.ganttAutoEnabled = parseBoolStrict(value, key);
        return;
    }

    if (const auto it = rawStringFields.find(key); it != rawStringFields.end()) {
        config.*(it->second) = value;
        return;
    }
    if (const auto it = lowerStringFields.find(key); it != lowerStringFields.end()) {
        config.*(it->second) = CommonUtils::toLower(value);
        return;
    }
    if (const auto it = boolFields.find(key); it != boolFields.end()) {
        config.*(it->second) = parseBoolStrict(value, key);
        return;
    }
    if (const auto it = uintFields.find(key); it != uintFields.end()) {
        config.*(it->second) = parseUIntStrict(value, key);
        return;
    }
    if (const auto it = intFields.find(key); it != intFields.end()) {
        config.*(it->second.member) = parseIntStrict(value, key, it->second.minValue);
        return;
    }
    if (const auto it = sizeFields.find(key); it != sizeFields.end()) {
        config.*(it->second.member) = static_cast<size_t>(parseIntStrict(value, key, it->second.minValue));
        return;
    }
    if (const auto it = doubleFields.find(key); it != doubleFields.end()) {
        config.*(it->second.member) = parseDoubleStrict(value, key, it->second.minValue);
        return;
    }
    if (const auto it = plotIntFields.find(key); it != plotIntFields.end()) {
        config.plot.*(it->second.member) = parseIntStrict(value, key, it->second.minValue);
        return;
    }
    if (const auto it = plotDoubleFields.find(key); it != plotDoubleFields.end()) {
        config.plot.*(it->second.member) = parseDoubleStrict(value, key, it->second.minValue);
        return;
    }
    if (const auto it = tuningDoubleFields.find(key); it != tuningDoubleFields.end()) {
        config.tuning.*(it->second.member) = parseDoubleStrict(value, key, it->second.minValue);
        return;
    }
    if (const auto it = tuningSizeFields.find(key); it != tuningSizeFields.end()) {
        config.tuning.*(it->second.member) = static_cast<size_t>(parseIntStrict(value, key, it->second.minValue));
        return;
    }
}
}

AutoConfig AutoConfig::fromArgs(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--interactive") {
        AutoConfig cfg = runInteractiveWizard();
        cfg.validate();
        return cfg;
    }

    if (argc < 2) {
        throw Seldon::ConfigurationException("Usage: seldon <dataset.csv> [--config path] [--target col] [--delimiter ,] [--plots bivariate,univariate,overall] [--plot-univariate true|false] [--plot-overall true|false] [--plot-bivariate true|false] [--plot-theme auto|light|dark] [--plot-grid true|false] [--plot-point-size >0] [--plot-line-width >0] [--generate-html true|false] [--verbose-analysis true|false] [--neural-seed N] [--benchmark-seed N] [--gradient-clip N] [--neural-optimizer sgd|adam|lookahead] [--neural-lookahead-fast-optimizer sgd|adam] [--neural-lookahead-sync-period N] [--neural-lookahead-alpha 0..1] [--neural-use-batch-norm true|false] [--neural-batch-norm-momentum 0..1) [--neural-batch-norm-epsilon >0] [--neural-use-layer-norm true|false] [--neural-layer-norm-epsilon >0] [--neural-lr-decay 0..1] [--neural-lr-plateau-patience N] [--neural-lr-cooldown-epochs N] [--neural-max-lr-reductions N] [--neural-min-learning-rate >=0] [--neural-use-validation-loss-ema true|false] [--neural-validation-loss-ema-beta 0..1] [--neural-categorical-input-l2-boost >=0] [--max-feature-missing-ratio -1|0..1] [--target-strategy auto|quality|max_variance|last_numeric] [--feature-strategy auto|adaptive|aggressive|lenient] [--neural-strategy auto|none|fast|balanced|expressive] [--bivariate-strategy auto|balanced|corr_heavy|importance_heavy] [--fast true|false] [--fast-max-bivariate-pairs N] [--fast-neural-sample-rows N] [--feature-min-variance N] [--feature-leakage-corr-threshold 0..1] [--significance-alpha 0..1] [--outlier-iqr-multiplier N] [--outlier-z-threshold N] [--bivariate-selection-quantile 0..1] [--tier3-fallback-aggressiveness 0..3] [--ogive-min-points N] [--ogive-min-unique N] [--box-min-points N] [--box-min-iqr >=0] [--pie-min-categories N] [--pie-max-categories N] [--pie-max-dominance 0..1] [--fit-min-corr 0..1] [--fit-min-samples N] [--time-series-season-period N] [--lof-max-rows N] [--lof-fallback-modified-z-threshold N] [--lof-threshold-floor N] [--gantt-auto true|false] [--gantt-min-tasks N] [--gantt-max-tasks N] [--gantt-duration-hours-threshold >0]");
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
        } else if (arg == "--profile" && i + 1 < argc) {
            applyProfile(config, argv[++i]);
        } else if (arg == "--datetime-locale-hint" && i + 1 < argc) {
            config.datetimeLocaleHint = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--numeric-locale-hint" && i + 1 < argc) {
            config.numericLocaleHint = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.outputDir = argv[++i];
        } else if (arg == "--export-preprocessed" && i + 1 < argc) {
            config.exportPreprocessed = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--export-preprocessed-path" && i + 1 < argc) {
            config.exportPreprocessedPath = argv[++i];
        } else if (arg == "--delimiter" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v.size() != 1) throw Seldon::ConfigurationException("--delimiter expects a single character");
            config.delimiter = v[0];
        } else if (arg == "--plots" && i + 1 < argc) {
            applyPlotModes(config, argv[++i]);
        } else if (arg == "--plot-univariate" && i + 1 < argc) {
            config.plotModesExplicit = true;
            config.plotUnivariate = parseBoolStrict(argv[++i], "--plot-univariate");
        } else if (arg == "--plot-overall" && i + 1 < argc) {
            config.plotModesExplicit = true;
            config.plotOverall = parseBoolStrict(argv[++i], "--plot-overall");
        } else if (arg == "--plot-bivariate" && i + 1 < argc) {
            config.plotModesExplicit = true;
            config.plotBivariateSignificant = parseBoolStrict(argv[++i], "--plot-bivariate");
        } else if (arg == "--plot-theme" && i + 1 < argc) {
            config.plot.theme = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--plot-grid" && i + 1 < argc) {
            config.plot.showGrid = parseBoolStrict(argv[++i], "--plot-grid");
        } else if (arg == "--plot-point-size" && i + 1 < argc) {
            config.plot.pointSize = parseDoubleStrict(argv[++i], "--plot-point-size", 0.1);
        } else if (arg == "--plot-line-width" && i + 1 < argc) {
            config.plot.lineWidth = parseDoubleStrict(argv[++i], "--plot-line-width", 0.1);
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
        } else if (arg == "--neural-optimizer" && i + 1 < argc) {
            config.neuralOptimizer = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--neural-lookahead-fast-optimizer" && i + 1 < argc) {
            config.neuralLookaheadFastOptimizer = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--neural-lookahead-sync-period" && i + 1 < argc) {
            config.neuralLookaheadSyncPeriod = parseIntStrict(argv[++i], "--neural-lookahead-sync-period", 1);
        } else if (arg == "--neural-lookahead-alpha" && i + 1 < argc) {
            config.neuralLookaheadAlpha = parseDoubleStrict(argv[++i], "--neural-lookahead-alpha", 0.0);
        } else if (arg == "--neural-use-batch-norm" && i + 1 < argc) {
            config.neuralUseBatchNorm = parseBoolStrict(argv[++i], "--neural-use-batch-norm");
        } else if (arg == "--neural-batch-norm-momentum" && i + 1 < argc) {
            config.neuralBatchNormMomentum = parseDoubleStrict(argv[++i], "--neural-batch-norm-momentum", 0.0);
        } else if (arg == "--neural-batch-norm-epsilon" && i + 1 < argc) {
            config.neuralBatchNormEpsilon = parseDoubleStrict(argv[++i], "--neural-batch-norm-epsilon", 1e-12);
        } else if (arg == "--neural-use-layer-norm" && i + 1 < argc) {
            config.neuralUseLayerNorm = parseBoolStrict(argv[++i], "--neural-use-layer-norm");
        } else if (arg == "--neural-layer-norm-epsilon" && i + 1 < argc) {
            config.neuralLayerNormEpsilon = parseDoubleStrict(argv[++i], "--neural-layer-norm-epsilon", 1e-12);
        } else if (arg == "--neural-lr-decay" && i + 1 < argc) {
            config.neuralLrDecay = parseDoubleStrict(argv[++i], "--neural-lr-decay", 0.0);
        } else if (arg == "--neural-lr-warmup-epochs" && i + 1 < argc) {
            config.neuralLrWarmupEpochs = parseIntStrict(argv[++i], "--neural-lr-warmup-epochs", 0);
        } else if (arg == "--neural-use-cosine-annealing" && i + 1 < argc) {
            config.neuralUseCosineAnnealing = parseBoolStrict(argv[++i], "--neural-use-cosine-annealing");
        } else if (arg == "--neural-use-cyclical-lr" && i + 1 < argc) {
            config.neuralUseCyclicalLr = parseBoolStrict(argv[++i], "--neural-use-cyclical-lr");
        } else if (arg == "--neural-lr-cycle-epochs" && i + 1 < argc) {
            config.neuralLrCycleEpochs = parseIntStrict(argv[++i], "--neural-lr-cycle-epochs", 2);
        } else if (arg == "--neural-lr-schedule-min-factor" && i + 1 < argc) {
            config.neuralLrScheduleMinFactor = parseDoubleStrict(argv[++i], "--neural-lr-schedule-min-factor", 0.0);
        } else if (arg == "--neural-lr-plateau-patience" && i + 1 < argc) {
            config.neuralLrPlateauPatience = parseIntStrict(argv[++i], "--neural-lr-plateau-patience", 1);
        } else if (arg == "--neural-lr-cooldown-epochs" && i + 1 < argc) {
            config.neuralLrCooldownEpochs = parseIntStrict(argv[++i], "--neural-lr-cooldown-epochs", 0);
        } else if (arg == "--neural-max-lr-reductions" && i + 1 < argc) {
            config.neuralMaxLrReductions = parseIntStrict(argv[++i], "--neural-max-lr-reductions", 0);
        } else if (arg == "--neural-min-learning-rate" && i + 1 < argc) {
            config.neuralMinLearningRate = parseDoubleStrict(argv[++i], "--neural-min-learning-rate", 0.0);
        } else if (arg == "--neural-use-validation-loss-ema" && i + 1 < argc) {
            config.neuralUseValidationLossEma = parseBoolStrict(argv[++i], "--neural-use-validation-loss-ema");
        } else if (arg == "--neural-validation-loss-ema-beta" && i + 1 < argc) {
            config.neuralValidationLossEmaBeta = parseDoubleStrict(argv[++i], "--neural-validation-loss-ema-beta", 0.0);
        } else if (arg == "--neural-categorical-input-l2-boost" && i + 1 < argc) {
            config.neuralCategoricalInputL2Boost = parseDoubleStrict(argv[++i], "--neural-categorical-input-l2-boost", 0.0);
        } else if (arg == "--neural-use-adaptive-gradient-clipping" && i + 1 < argc) {
            config.neuralUseAdaptiveGradientClipping = parseBoolStrict(argv[++i], "--neural-use-adaptive-gradient-clipping");
        } else if (arg == "--neural-adaptive-clip-beta" && i + 1 < argc) {
            config.neuralAdaptiveClipBeta = parseDoubleStrict(argv[++i], "--neural-adaptive-clip-beta", 0.0);
        } else if (arg == "--neural-adaptive-clip-multiplier" && i + 1 < argc) {
            config.neuralAdaptiveClipMultiplier = parseDoubleStrict(argv[++i], "--neural-adaptive-clip-multiplier", 0.0);
        } else if (arg == "--neural-adaptive-clip-min" && i + 1 < argc) {
            config.neuralAdaptiveClipMin = parseDoubleStrict(argv[++i], "--neural-adaptive-clip-min", 0.0);
        } else if (arg == "--neural-gradient-noise-std" && i + 1 < argc) {
            config.neuralGradientNoiseStd = parseDoubleStrict(argv[++i], "--neural-gradient-noise-std", 0.0);
        } else if (arg == "--neural-gradient-noise-decay" && i + 1 < argc) {
            config.neuralGradientNoiseDecay = parseDoubleStrict(argv[++i], "--neural-gradient-noise-decay", 0.0);
        } else if (arg == "--neural-use-ema-weights" && i + 1 < argc) {
            config.neuralUseEmaWeights = parseBoolStrict(argv[++i], "--neural-use-ema-weights");
        } else if (arg == "--neural-ema-decay" && i + 1 < argc) {
            config.neuralEmaDecay = parseDoubleStrict(argv[++i], "--neural-ema-decay", 0.0);
        } else if (arg == "--neural-label-smoothing" && i + 1 < argc) {
            config.neuralLabelSmoothing = parseDoubleStrict(argv[++i], "--neural-label-smoothing", 0.0);
        } else if (arg == "--neural-gradient-accumulation-steps" && i + 1 < argc) {
            config.neuralGradientAccumulationSteps = parseIntStrict(argv[++i], "--neural-gradient-accumulation-steps", 1);
        } else if (arg == "--neural-learning-rate" && i + 1 < argc) {
            config.neuralLearningRate = parseDoubleStrict(argv[++i], "--neural-learning-rate", 1e-8);
        } else if (arg == "--neural-min-layers" && i + 1 < argc) {
            config.neuralMinLayers = parseIntStrict(argv[++i], "--neural-min-layers", 1);
        } else if (arg == "--neural-max-layers" && i + 1 < argc) {
            config.neuralMaxLayers = parseIntStrict(argv[++i], "--neural-max-layers", 1);
        } else if (arg == "--neural-fixed-layers" && i + 1 < argc) {
            config.neuralFixedLayers = parseIntStrict(argv[++i], "--neural-fixed-layers", 0);
        } else if (arg == "--neural-fixed-hidden-nodes" && i + 1 < argc) {
            config.neuralFixedHiddenNodes = parseIntStrict(argv[++i], "--neural-fixed-hidden-nodes", 0);
        } else if (arg == "--neural-max-hidden-nodes" && i + 1 < argc) {
            config.neuralMaxHiddenNodes = parseIntStrict(argv[++i], "--neural-max-hidden-nodes", 4);
        } else if (arg == "--neural-streaming-mode" && i + 1 < argc) {
            config.neuralStreamingMode = parseBoolStrict(argv[++i], "--neural-streaming-mode");
        } else if (arg == "--neural-streaming-chunk-rows" && i + 1 < argc) {
            config.neuralStreamingChunkRows = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-streaming-chunk-rows", 16));
        } else if (arg == "--neural-multi-output" && i + 1 < argc) {
            config.neuralMultiOutput = parseBoolStrict(argv[++i], "--neural-multi-output");
        } else if (arg == "--neural-max-aux-targets" && i + 1 < argc) {
            config.neuralMaxAuxTargets = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-max-aux-targets", 0));
        } else if (arg == "--neural-explainability" && i + 1 < argc) {
            config.neuralExplainability = CommonUtils::toLower(argv[++i]);
        } else if (arg == "--neural-integrated-grad-steps" && i + 1 < argc) {
            config.neuralIntegratedGradSteps = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-integrated-grad-steps", 4));
        } else if (arg == "--neural-uncertainty-samples" && i + 1 < argc) {
            config.neuralUncertaintySamples = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-uncertainty-samples", 4));
        } else if (arg == "--neural-importance-parallel" && i + 1 < argc) {
            config.neuralImportanceParallel = parseBoolStrict(argv[++i], "--neural-importance-parallel");
        } else if (arg == "--neural-importance-max-rows" && i + 1 < argc) {
            config.neuralImportanceMaxRows = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-importance-max-rows", 128));
        } else if (arg == "--neural-importance-trials" && i + 1 < argc) {
            config.neuralImportanceTrials = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-importance-trials", 0));
        } else if (arg == "--neural-max-one-hot-per-column" && i + 1 < argc) {
            config.neuralMaxOneHotPerColumn = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-max-one-hot-per-column", 1));
        } else if (arg == "--neural-max-topology-nodes" && i + 1 < argc) {
            config.neuralMaxTopologyNodes = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-max-topology-nodes", 16));
        } else if (arg == "--neural-max-trainable-params" && i + 1 < argc) {
            config.neuralMaxTrainableParams = static_cast<size_t>(parseIntStrict(argv[++i], "--neural-max-trainable-params", 1024));
        } else if (arg == "--store-outlier-flags-in-report" && i + 1 < argc) {
            config.storeOutlierFlagsInReport = parseBoolStrict(argv[++i], "--store-outlier-flags-in-report");
        } else if (arg == "--feature-engineering-enable-poly" && i + 1 < argc) {
            config.featureEngineeringEnablePoly = parseBoolStrict(argv[++i], "--feature-engineering-enable-poly");
        } else if (arg == "--feature-engineering-enable-log" && i + 1 < argc) {
            config.featureEngineeringEnableLog = parseBoolStrict(argv[++i], "--feature-engineering-enable-log");
        } else if (arg == "--feature-engineering-enable-ratio-product-discovery" && i + 1 < argc) {
            config.featureEngineeringEnableRatioProductDiscovery = parseBoolStrict(argv[++i], "--feature-engineering-enable-ratio-product-discovery");
        } else if (arg == "--feature-engineering-degree" && i + 1 < argc) {
            config.featureEngineeringDegree = parseIntStrict(argv[++i], "--feature-engineering-degree", 1);
        } else if (arg == "--feature-engineering-max-base" && i + 1 < argc) {
            config.featureEngineeringMaxBase = static_cast<size_t>(parseIntStrict(argv[++i], "--feature-engineering-max-base", 2));
        } else if (arg == "--feature-engineering-max-pairwise-discovery" && i + 1 < argc) {
            config.featureEngineeringMaxPairwiseDiscovery = static_cast<size_t>(parseIntStrict(argv[++i], "--feature-engineering-max-pairwise-discovery", 2));
        } else if (arg == "--feature-engineering-max-generated-columns" && i + 1 < argc) {
            config.featureEngineeringMaxGeneratedColumns = static_cast<size_t>(parseIntStrict(argv[++i], "--feature-engineering-max-generated-columns", 16));
        } else if (arg == "--type" && i + 1 < argc) {
            const std::string raw = argv[++i];
            const size_t sep = raw.find(':');
            if (sep == std::string::npos || sep == 0 || sep + 1 >= raw.size()) {
                throw Seldon::ConfigurationException("--type expects <column>:<numeric|categorical|datetime>");
            }
            const std::string columnName = CommonUtils::trim(raw.substr(0, sep));
            const std::string typeName = CommonUtils::toLower(CommonUtils::trim(raw.substr(sep + 1)));
            if (!isValidColumnTypeOverride(typeName)) {
                throw Seldon::ConfigurationException("Invalid --type override for column '" + columnName + "': " + typeName);
            }
            config.columnTypeOverrides[CommonUtils::toLower(columnName)] = typeName;
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
        } else if (arg == "--low-memory" && i + 1 < argc) {
            config.lowMemoryMode = parseBoolStrict(argv[++i], "--low-memory");
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
        } else if (arg == "--tier3-fallback-aggressiveness" && i + 1 < argc) {
            config.tuning.bivariateTier3FallbackAggressiveness = parseDoubleStrict(argv[++i], "--tier3-fallback-aggressiveness", 0.0);
        } else if (arg == "--overall-corr-heatmap-max-columns" && i + 1 < argc) {
            config.tuning.overallCorrHeatmapMaxColumns = static_cast<size_t>(parseIntStrict(argv[++i], "--overall-corr-heatmap-max-columns", 2));
        } else if (arg == "--ogive-min-points" && i + 1 < argc) {
            config.tuning.ogiveMinPoints = static_cast<size_t>(parseIntStrict(argv[++i], "--ogive-min-points", 2));
        } else if (arg == "--ogive-min-unique" && i + 1 < argc) {
            config.tuning.ogiveMinUnique = static_cast<size_t>(parseIntStrict(argv[++i], "--ogive-min-unique", 2));
        } else if (arg == "--box-min-points" && i + 1 < argc) {
            config.tuning.boxPlotMinPoints = static_cast<size_t>(parseIntStrict(argv[++i], "--box-min-points", 3));
        } else if (arg == "--box-min-iqr" && i + 1 < argc) {
            config.tuning.boxPlotMinIqr = parseDoubleStrict(argv[++i], "--box-min-iqr", 0.0);
        } else if (arg == "--pie-min-categories" && i + 1 < argc) {
            config.tuning.pieMinCategories = static_cast<size_t>(parseIntStrict(argv[++i], "--pie-min-categories", 2));
        } else if (arg == "--pie-max-categories" && i + 1 < argc) {
            config.tuning.pieMaxCategories = static_cast<size_t>(parseIntStrict(argv[++i], "--pie-max-categories", 2));
        } else if (arg == "--pie-max-dominance" && i + 1 < argc) {
            config.tuning.pieMaxDominanceRatio = parseDoubleStrict(argv[++i], "--pie-max-dominance", 0.0);
        } else if (arg == "--fit-min-corr" && i + 1 < argc) {
            config.tuning.scatterFitMinAbsCorr = parseDoubleStrict(argv[++i], "--fit-min-corr", 0.0);
        } else if (arg == "--fit-min-samples" && i + 1 < argc) {
            config.tuning.scatterFitMinSampleSize = static_cast<size_t>(parseIntStrict(argv[++i], "--fit-min-samples", 3));
        } else if (arg == "--time-series-season-period" && i + 1 < argc) {
            config.tuning.timeSeriesSeasonPeriod = static_cast<size_t>(parseIntStrict(argv[++i], "--time-series-season-period", 2));
        } else if (arg == "--lof-max-rows" && i + 1 < argc) {
            config.tuning.lofMaxRows = static_cast<size_t>(parseIntStrict(argv[++i], "--lof-max-rows", 100));
        } else if (arg == "--lof-fallback-modified-z-threshold" && i + 1 < argc) {
            config.tuning.lofFallbackModifiedZThreshold = parseDoubleStrict(argv[++i], "--lof-fallback-modified-z-threshold", 0.0);
        } else if (arg == "--lof-threshold-floor" && i + 1 < argc) {
            config.tuning.lofThresholdFloor = parseDoubleStrict(argv[++i], "--lof-threshold-floor", 0.0);
        } else if (arg == "--gantt-auto" && i + 1 < argc) {
            config.tuning.ganttAutoEnabled = parseBoolStrict(argv[++i], "--gantt-auto");
        } else if (arg == "--gantt-min-tasks" && i + 1 < argc) {
            config.tuning.ganttMinTasks = static_cast<size_t>(parseIntStrict(argv[++i], "--gantt-min-tasks", 1));
        } else if (arg == "--gantt-max-tasks" && i + 1 < argc) {
            config.tuning.ganttMaxTasks = static_cast<size_t>(parseIntStrict(argv[++i], "--gantt-max-tasks", 1));
        } else if (arg == "--gantt-duration-hours-threshold" && i + 1 < argc) {
            config.tuning.ganttDurationHoursThreshold = parseDoubleStrict(argv[++i], "--gantt-duration-hours-threshold", 0.1);
        }
    }

    if (!configPath.empty()) {
        config = fromFile(configPath, config);
        if (config.datasetPath.empty()) config.datasetPath = argv[1];
    }

    applyProfile(config, config.profile);
    if (config.lowMemoryMode) {
        applyLowMemoryDefaults(config);
    }

    config.validate();

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
        line = CommonUtils::trim(stripStructuralTokensOutsideQuotes(line));
        if (line.empty()) continue;

        size_t sep = findSeparatorOutsideQuotes(line, ':');
        if (sep == std::string::npos) continue;

        std::string key = normalizeConfigKey(maybeUnquote(line.substr(0, sep)));
        std::string value = maybeUnquote(line.substr(sep + 1));

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
            const std::string columnName = CommonUtils::trim(key.substr(7));
            if (columnName.empty()) {
                throw Seldon::ConfigurationException(
                    "Config parse error at line " + std::to_string(lineNo) +
                    ": impute.<column> requires a non-empty column name");
            }
            config.columnImputation[columnName] = normalized;
        }
        if (key.rfind("type.", 0) == 0) {
            const std::string normalized = CommonUtils::toLower(value);
            if (!isValidColumnTypeOverride(normalized)) {
                throw Seldon::ConfigurationException(
                    "Config parse error at line " + std::to_string(lineNo) +
                    ": invalid column type override '" + value +
                    "' (allowed: numeric, categorical, datetime)");
            }
            config.columnTypeOverrides[CommonUtils::toLower(key.substr(5))] = normalized;
        }
    }
    config.validate();

    return config;
}

void HeuristicTuningConfig::validate() const {
    if (featureLeakageCorrThreshold < 0.0 || featureLeakageCorrThreshold > 1.0) {
        throw Seldon::ConfigurationException("feature_leakage_corr_threshold must be within [0,1]");
    }
    if (significanceAlpha <= 0.0 || significanceAlpha >= 1.0) {
        throw Seldon::ConfigurationException("significance_alpha must be within (0,1)");
    }
    if (outlierIqrMultiplier <= 0.0) {
        throw Seldon::ConfigurationException("outlier_iqr_multiplier must be > 0");
    }
    if (outlierZThreshold <= 0.0) {
        throw Seldon::ConfigurationException("outlier_z_threshold must be > 0");
    }
    if (featureMissingAdaptiveMin > featureMissingAdaptiveMax) {
        throw Seldon::ConfigurationException("feature_missing_floor must be <= feature_missing_ceiling");
    }
    if (featureMissingQ3Offset < 0.0) {
        throw Seldon::ConfigurationException("feature_missing_q3_offset must be >= 0");
    }
    if (bivariateSelectionQuantileOverride != -1.0 &&
        (bivariateSelectionQuantileOverride < 0.0 || bivariateSelectionQuantileOverride > 1.0)) {
        throw Seldon::ConfigurationException("bivariate_selection_quantile must be -1 or within [0,1]");
    }
    if (bivariateTier3FallbackAggressiveness < 0.0 || bivariateTier3FallbackAggressiveness > 3.0) {
        throw Seldon::ConfigurationException("bivariate_tier3_fallback_aggressiveness must be within [0,3]");
    }
    if (coherenceWeightMin > coherenceWeightMax) {
        throw Seldon::ConfigurationException("coherence_weight_min must be <= coherence_weight_max");
    }
    const double hybridWeightSum = hybridExplainabilityWeightPermutation +
                                   hybridExplainabilityWeightIntegratedGradients;
    if (hybridWeightSum <= 0.0) {
        throw Seldon::ConfigurationException("hybrid explainability weights must sum to > 0");
    }
    if (betaFallbackIntervalsStart > betaFallbackIntervalsMax) {
        throw Seldon::ConfigurationException("beta_fallback_intervals_start must be <= beta_fallback_intervals_max");
    }
    if (numericEpsilon <= 0.0 || betaFallbackTolerance <= 0.0) {
        throw Seldon::ConfigurationException("numeric_epsilon and beta_fallback_tolerance must be > 0");
    }
    if (overallCorrHeatmapMaxColumns < 2) {
        throw Seldon::ConfigurationException("overall_corr_heatmap_max_columns must be >= 2");
    }
    if (ogiveMinPoints < 2 || ogiveMinUnique < 2) {
        throw Seldon::ConfigurationException("ogive_min_points and ogive_min_unique must be >= 2");
    }
    if (boxPlotMinPoints < 3) {
        throw Seldon::ConfigurationException("box_plot_min_points must be >= 3");
    }
    if (boxPlotMinIqr < 0.0) {
        throw Seldon::ConfigurationException("box_plot_min_iqr must be >= 0");
    }
    if (pieMinCategories < 2 || pieMaxCategories < pieMinCategories) {
        throw Seldon::ConfigurationException("pie_min_categories must be >= 2 and <= pie_max_categories");
    }
    if (pieMaxDominanceRatio <= 0.0 || pieMaxDominanceRatio > 1.0) {
        throw Seldon::ConfigurationException("pie_max_dominance_ratio must be within (0,1]");
    }
    if (scatterFitMinAbsCorr < 0.0 || scatterFitMinAbsCorr > 1.0) {
        throw Seldon::ConfigurationException("scatter_fit_min_abs_corr must be within [0,1]");
    }
    if (scatterFitMinSampleSize < 3) {
        throw Seldon::ConfigurationException("scatter_fit_min_sample_size must be >= 3");
    }
    if (timeSeriesSeasonPeriod < 2) {
        throw Seldon::ConfigurationException("time_series_season_period must be >= 2");
    }
    if (lofMaxRows < 100) {
        throw Seldon::ConfigurationException("lof_max_rows must be >= 100");
    }
    if (lofFallbackModifiedZThreshold <= 0.0) {
        throw Seldon::ConfigurationException("lof_fallback_modified_z_threshold must be > 0");
    }
    if (lofThresholdFloor <= 0.0) {
        throw Seldon::ConfigurationException("lof_threshold_floor must be > 0");
    }
    if (ganttMinTasks < 1 || ganttMaxTasks < ganttMinTasks) {
        throw Seldon::ConfigurationException("gantt_min_tasks must be >= 1 and <= gantt_max_tasks");
    }
    if (ganttDurationHoursThreshold <= 0.0) {
        throw Seldon::ConfigurationException("gantt_duration_hours_threshold must be > 0");
    }
}

void AutoConfig::validate() const {
    if (datasetPath.empty()) {
        throw Seldon::ConfigurationException("dataset path is required");
    }

    const auto isIn = [](const std::string& value, const std::vector<std::string>& allowed) {
        return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
    };

    if (!isIn(plot.format, {"png", "svg", "pdf"})) {
        throw Seldon::ConfigurationException("plot_format must be one of: png, svg, pdf");
    }
    if (!isIn(profile, {"auto", "quick", "thorough", "minimal"})) {
        throw Seldon::ConfigurationException("profile must be one of: auto, quick, thorough, minimal");
    }
    if (!isIn(datetimeLocaleHint, {"auto", "dmy", "mdy"})) {
        throw Seldon::ConfigurationException("datetime_locale_hint must be one of: auto, dmy, mdy");
    }
    if (!isIn(numericLocaleHint, {"auto", "us", "eu"})) {
        throw Seldon::ConfigurationException("numeric_locale_hint must be one of: auto, us, eu");
    }
    if (!isIn(exportPreprocessed, {"none", "csv", "parquet"})) {
        throw Seldon::ConfigurationException("export_preprocessed must be one of: none, csv, parquet");
    }
    if (!isIn(plot.theme, {"auto", "light", "dark"})) {
        throw Seldon::ConfigurationException("plot_theme must be one of: auto, light, dark");
    }
    if (plot.pointSize <= 0.0 || plot.lineWidth <= 0.0) {
        throw Seldon::ConfigurationException("plot_point_size and plot_line_width must be > 0");
    }
    if (!isIn(outlierMethod, {"iqr", "zscore", "modified_zscore", "adjusted_boxplot", "lof"})) {
        throw Seldon::ConfigurationException("outlier_method must be one of: iqr, zscore, modified_zscore, adjusted_boxplot, lof");
    }
    if (!isIn(outlierAction, {"flag", "remove", "cap"})) {
        throw Seldon::ConfigurationException("outlier_action must be flag, remove, or cap");
    }
    if (!isIn(scalingMethod, {"auto", "zscore", "minmax", "none"})) {
        throw Seldon::ConfigurationException("scaling must be auto, zscore, minmax, or none");
    }

    if (!isIn(targetStrategy, {"auto", "quality", "max_variance", "last_numeric"})) {
        throw Seldon::ConfigurationException("target_strategy must be one of: auto, quality, max_variance, last_numeric");
    }
    if (!isIn(featureStrategy, {"auto", "adaptive", "aggressive", "lenient"})) {
        throw Seldon::ConfigurationException("feature_strategy must be one of: auto, adaptive, aggressive, lenient");
    }
    if (!isIn(neuralStrategy, {"auto", "none", "fast", "balanced", "expressive"})) {
        throw Seldon::ConfigurationException("neural_strategy must be one of: auto, none, fast, balanced, expressive");
    }
    if (!isIn(bivariateStrategy, {"auto", "balanced", "corr_heavy", "importance_heavy"})) {
        throw Seldon::ConfigurationException("bivariate_strategy must be one of: auto, balanced, corr_heavy, importance_heavy");
    }
    if (!isIn(neuralOptimizer, {"sgd", "adam", "lookahead"})) {
        throw Seldon::ConfigurationException("neural_optimizer must be one of: sgd, adam, lookahead");
    }
    if (!isIn(neuralLookaheadFastOptimizer, {"sgd", "adam"})) {
        throw Seldon::ConfigurationException("neural_lookahead_fast_optimizer must be one of: sgd, adam");
    }
    if (!isIn(neuralExplainability, {"permutation", "integrated_gradients", "hybrid"})) {
        throw Seldon::ConfigurationException("neural_explainability must be one of: permutation, integrated_gradients, hybrid");
    }

    if (neuralLookaheadAlpha < 0.0 || neuralLookaheadAlpha > 1.0) {
        throw Seldon::ConfigurationException("neural_lookahead_alpha must be within [0,1]");
    }
    if (neuralBatchNormMomentum < 0.0 || neuralBatchNormMomentum >= 1.0) {
        throw Seldon::ConfigurationException("neural_batch_norm_momentum must be within [0,1)");
    }
    if (neuralLrDecay <= 0.0 || neuralLrDecay >= 1.0) {
        throw Seldon::ConfigurationException("neural_lr_decay must be within (0,1)");
    }
    if (neuralLrWarmupEpochs < 0) {
        throw Seldon::ConfigurationException("neural_lr_warmup_epochs must be >= 0");
    }
    if (neuralLrCycleEpochs < 2) {
        throw Seldon::ConfigurationException("neural_lr_cycle_epochs must be >= 2");
    }
    if (neuralLrScheduleMinFactor < 0.0 || neuralLrScheduleMinFactor > 1.0) {
        throw Seldon::ConfigurationException("neural_lr_schedule_min_factor must be within [0,1]");
    }
    if (neuralValidationLossEmaBeta < 0.0 || neuralValidationLossEmaBeta >= 1.0) {
        throw Seldon::ConfigurationException("neural_validation_loss_ema_beta must be within [0,1)");
    }
    if (neuralAdaptiveClipBeta < 0.0 || neuralAdaptiveClipBeta >= 1.0) {
        throw Seldon::ConfigurationException("neural_adaptive_clip_beta must be within [0,1)");
    }
    if (neuralAdaptiveClipMultiplier <= 0.0) {
        throw Seldon::ConfigurationException("neural_adaptive_clip_multiplier must be > 0");
    }
    if (neuralGradientNoiseDecay < 0.0 || neuralGradientNoiseDecay > 1.0) {
        throw Seldon::ConfigurationException("neural_gradient_noise_decay must be within [0,1]");
    }
    if (neuralEmaDecay < 0.0 || neuralEmaDecay >= 1.0) {
        throw Seldon::ConfigurationException("neural_ema_decay must be within [0,1)");
    }
    if (neuralLabelSmoothing < 0.0 || neuralLabelSmoothing > 0.25) {
        throw Seldon::ConfigurationException("neural_label_smoothing must be within [0,0.25]");
    }
    if (neuralGradientAccumulationSteps < 1) {
        throw Seldon::ConfigurationException("neural_gradient_accumulation_steps must be >= 1");
    }
    if (neuralLearningRate <= 0.0) {
        throw Seldon::ConfigurationException("neural_learning_rate must be > 0");
    }
    if (neuralMinLayers > neuralMaxLayers) {
        throw Seldon::ConfigurationException("neural_min_layers must be <= neural_max_layers");
    }
    if (neuralFixedLayers < 0 || (neuralFixedLayers > 0 && (neuralFixedLayers < neuralMinLayers || neuralFixedLayers > neuralMaxLayers))) {
        throw Seldon::ConfigurationException("neural_fixed_layers must be 0 or within [neural_min_layers, neural_max_layers]");
    }
    if (neuralFixedHiddenNodes < 0) {
        throw Seldon::ConfigurationException("neural_fixed_hidden_nodes must be >= 0");
    }
    if (neuralMaxHiddenNodes < 4) {
        throw Seldon::ConfigurationException("neural_max_hidden_nodes must be >= 4");
    }
    if (neuralStreamingChunkRows < 16) {
        throw Seldon::ConfigurationException("neural_streaming_chunk_rows must be >= 16");
    }
    if (neuralIntegratedGradSteps < 4) {
        throw Seldon::ConfigurationException("neural_integrated_grad_steps must be >= 4");
    }
    if (neuralUncertaintySamples < 4) {
        throw Seldon::ConfigurationException("neural_uncertainty_samples must be >= 4");
    }
    if (neuralImportanceMaxRows < 128) {
        throw Seldon::ConfigurationException("neural_importance_max_rows must be >= 128");
    }
    if (featureEngineeringDegree < 1) {
        throw Seldon::ConfigurationException("feature_engineering_degree must be >= 1");
    }
    if (featureEngineeringMaxBase < 2) {
        throw Seldon::ConfigurationException("feature_engineering_max_base must be >= 2");
    }
    if (featureEngineeringMaxPairwiseDiscovery < 2) {
        throw Seldon::ConfigurationException("feature_engineering_max_pairwise_discovery must be >= 2");
    }
    if (featureEngineeringMaxGeneratedColumns < 16) {
        throw Seldon::ConfigurationException("feature_engineering_max_generated_columns must be >= 16");
    }
    if (neuralMaxOneHotPerColumn < 1) {
        throw Seldon::ConfigurationException("neural_max_one_hot_per_column must be >= 1");
    }
    if (neuralMaxTopologyNodes < 16) {
        throw Seldon::ConfigurationException("neural_max_topology_nodes must be >= 16");
    }
    if (neuralMaxTrainableParams < 1024) {
        throw Seldon::ConfigurationException("neural_max_trainable_params must be >= 1024");
    }

    tuning.validate();

    if (fastMaxBivariatePairs == 0 || fastNeuralSampleRows == 0) {
        throw Seldon::ConfigurationException("fast_max_bivariate_pairs and fast_neural_sample_rows must be > 0");
    }

    for (const auto& kv : columnImputation) {
        if (CommonUtils::trim(kv.first).empty()) {
            throw Seldon::ConfigurationException("Invalid imputation override: column name cannot be empty");
        }
        if (!isValidImputationStrategy(kv.second)) {
            throw Seldon::ConfigurationException(
                "Invalid imputation strategy for column '" + kv.first +
                "': '" + kv.second + "' (allowed: auto, mean, median, zero, mode, interpolate)");
        }
    }

    for (const auto& kv : columnTypeOverrides) {
        if (!isValidColumnTypeOverride(kv.second)) {
            throw Seldon::ConfigurationException(
                "Invalid type override for column '" + kv.first +
                "': '" + kv.second + "' (allowed: numeric, categorical, datetime)");
        }
    }
}
