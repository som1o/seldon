#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct PlotConfig {
    std::string format = "png";
    int width = 1280;
    int height = 720;
};

struct HeuristicTuningConfig {
    // Global significance threshold used in pairwise significance checks.
    double significanceAlpha = 0.05;
    // IQR multiplier for Tukey-style outlier fence (lower/higher => more/less sensitive).
    double outlierIqrMultiplier = 1.5;
    // Absolute z-score threshold for z-score outlier detection.
    double outlierZThreshold = 3.0;

    // Numerical floor for retaining low-variance numeric features.
    double featureMinVariance = 1e-10;
    // Leakage guard: drop feature when abs(feature,target corr) exceeds this value.
    double featureLeakageCorrThreshold = 0.995;

    // Adaptive missingness threshold = Q3(missing ratios) + offset.
    double featureMissingQ3Offset = 0.15;
    // Clamp for adaptive missingness threshold.
    double featureMissingAdaptiveMin = 0.35;
    double featureMissingAdaptiveMax = 0.95;

    // Strategy-specific deltas applied around adaptive threshold.
    double featureAggressiveDelta = 0.20;
    double featureAggressiveMin = 0.20;
    double featureAggressiveMax = 0.80;
    double featureLenientDelta = 0.20;
    double featureLenientMin = 0.40;
    double featureLenientMax = 0.98;

    // Optional override for pair selection quantile (-1 means strategy default).
    double bivariateSelectionQuantileOverride = -1.0; // -1 => policy default

    // Coherent-importance blending heuristics.
    double coherenceWeightSmallDataset = 0.55;
    double coherenceWeightRegularDataset = 0.70;
    double coherenceOverfitPenaltyTrainRatio = 1.5;
    double coherenceBenchmarkPenaltyRatio = 1.5;
    double coherencePenaltyStep = 0.20;
    double coherenceWeightMin = 0.20;
    double coherenceWeightMax = 0.85;

    // Auto bivariate policy switching triggers.
    double corrHeavyMaxImportanceThreshold = 0.65;
    double corrHeavyConcentrationThreshold = 0.55;
    double importanceHeavyMaxImportanceThreshold = 0.30;
    double importanceHeavyConcentrationThreshold = 0.40;

    // Runtime tolerances used across numeric routines.
    double numericEpsilon = 1e-12;
    size_t betaFallbackIntervalsStart = 4096;
    size_t betaFallbackIntervalsMax = 65536;
    double betaFallbackTolerance = 1e-8;

    // Maximum numeric columns used to build overall correlation heatmap (caps O(n^2) work).
    size_t overallCorrHeatmapMaxColumns = 50;
};

struct AutoConfig {
    std::string datasetPath;
    std::string reportFile = "neural_synthesis.md";
    std::string assetsDir = "seldon_report_assets";
    std::string targetColumn;
    char delimiter = ',';
    bool exhaustiveScan = false;

    std::vector<std::string> excludedColumns;
    std::unordered_map<std::string, std::string> columnImputation;

    std::string outlierMethod = "iqr";      // iqr|zscore
    std::string outlierAction = "flag";     // flag|remove|cap

    std::string scalingMethod = "auto";     // auto|zscore|minmax|none
    int kfold = 5;
    double maxFeatureMissingRatio = -1.0; // -1 => auto
    std::string targetStrategy = "auto";    // auto|quality|max_variance|last_numeric
    std::string featureStrategy = "auto";   // auto|adaptive|aggressive|lenient
    std::string neuralStrategy = "auto";    // auto|none|fast|balanced|expressive
    std::string bivariateStrategy = "auto"; // auto|balanced|corr_heavy|importance_heavy
    bool fastMode = false;
    size_t fastMaxBivariatePairs = 2500;
    size_t fastNeuralSampleRows = 25000;

    bool plotUnivariate = false;
    bool plotOverall = false;
    bool plotBivariateSignificant = true;
    bool generateHtml = false;
    bool verboseAnalysis = true;
    uint32_t neuralSeed = 1337;
    uint32_t benchmarkSeed = 1337;
    double gradientClipNorm = 5.0;

    // Neural stability overrides (universal runtime knobs)
    std::string neuralOptimizer = "lookahead";            // sgd|adam|lookahead
    std::string neuralLookaheadFastOptimizer = "adam";    // sgd|adam
    int neuralLookaheadSyncPeriod = 5;
    double neuralLookaheadAlpha = 0.5;
    bool neuralUseBatchNorm = true;
    double neuralBatchNormMomentum = 0.95;
    double neuralBatchNormEpsilon = 1e-5;
    bool neuralUseLayerNorm = true;
    double neuralLayerNormEpsilon = 1e-5;
    double neuralLrDecay = 0.5;
    int neuralLrPlateauPatience = 5;
    int neuralLrCooldownEpochs = 2;
    int neuralMaxLrReductions = 8;
    double neuralMinLearningRate = 1e-6;
    bool neuralUseValidationLossEma = true;
    double neuralValidationLossEmaBeta = 0.6;
    double neuralCategoricalInputL2Boost = 3.0;

    PlotConfig plot;
    HeuristicTuningConfig tuning;

    /**
     * @brief Builds config from CLI args and optional config file override.
     * @pre argc/argv contain at least dataset path in argv[1].
     * @post Returns a validated config object.
     * @throws Seldon::ConfigurationException on invalid arguments or values.
     */
    static AutoConfig fromArgs(int argc, char* argv[]);

    /**
     * @brief Loads config values from a lightweight YAML/JSON-like key:value file.
     * @pre configPath points to a readable text file.
     * @post Returns merged config using `base` as defaults.
     * @throws Seldon::ConfigurationException on parse/validation failures.
     */
    static AutoConfig fromFile(const std::string& configPath, const AutoConfig& base);

    /**
     * @brief Validates merged configuration invariants and enum-like fields.
     * @throws Seldon::ConfigurationException on invalid values.
     */
    void validate() const;
};
