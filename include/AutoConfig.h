#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct PlotConfig {
    std::string format = "png";
    int width = 1280;
    int height = 720;
    std::string theme = "auto"; // auto|light|dark
    bool showGrid = true;
    double pointSize = 0.8;
    double lineWidth = 2.0;
};

struct HeuristicTuningConfig {
    // Supported config keys (file/CLI aliases):
    // significance_alpha, outlier_iqr_multiplier, outlier_z_threshold,
    // feature_min_variance, feature_leakage_corr_threshold,
    // feature_missing_q3_offset, feature_missing_floor, feature_missing_ceiling,
    // feature_aggressive_delta, feature_lenient_delta,
    // bivariate_selection_quantile,
    // coherence_weight_small_dataset, coherence_weight_regular_dataset,
    // coherence_overfit_penalty_train_ratio, coherence_benchmark_penalty_ratio,
    // coherence_penalty_step, coherence_weight_min, coherence_weight_max,
    // corr_heavy_max_importance_threshold, corr_heavy_concentration_threshold,
    // importance_heavy_max_importance_threshold, importance_heavy_concentration_threshold,
    // numeric_epsilon, beta_fallback_intervals_start, beta_fallback_intervals_max, beta_fallback_tolerance,
    // overall_corr_heatmap_max_columns,
    // ogive_min_points, ogive_min_unique,
    // box_plot_min_points, box_plot_min_iqr,
    // pie_min_categories, pie_max_categories, pie_max_dominance_ratio,
    // scatter_fit_min_abs_corr, scatter_fit_min_sample_size,
    // time_series_season_period,
    // lof_max_rows, lof_fallback_modified_z_threshold, lof_threshold_floor,
    // gantt_auto_enabled, gantt_min_tasks, gantt_max_tasks, gantt_duration_hours_threshold.
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
    // Tier-3 fallback aggressiveness for sparse neural yield (0 disables fallback promotion).
    double bivariateTier3FallbackAggressiveness = 1.0;

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

    // Univariate suitability tuning
    size_t ogiveMinPoints = 8;
    size_t ogiveMinUnique = 6;
    size_t boxPlotMinPoints = 5;
    double boxPlotMinIqr = 1e-9;

    // Categorical pie-chart suitability tuning
    size_t pieMinCategories = 2;
    size_t pieMaxCategories = 10;
    double pieMaxDominanceRatio = 0.90;

    // Categorical-vs-numeric split/facetting tuning
    size_t facetMinRows = 30;
    size_t facetMaxCategories = 6;
    double facetMinCategoryShare = 0.05;

    // Scatter downsampling + confidence interval tuning
    size_t scatterDownsampleThreshold = 10000;
    double scatterConfidenceMinAbsCorr = 0.45;
    size_t scatterConfidenceMinSampleSize = 30;

    // Residual plot gating for selected regressions
    double residualPlotMinAbsCorr = 0.55;
    size_t residualPlotMinSampleSize = 24;

    // Distribution plot auto-mode tuning
    size_t categoryNumericDistributionMinRows = 40;
    size_t categoryNumericDistributionMaxPairs = 6;

    // Density-overlay minimum sample size
    size_t histogramDensityMinSample = 80;

    // Parallel coordinates tuning
    size_t parallelCoordinatesMinRows = 40;
    size_t parallelCoordinatesMaxRows = 220;
    size_t parallelCoordinatesMinDims = 4;
    size_t parallelCoordinatesMaxDims = 10;

    // Automatic time-series trend overlay tuning
    size_t timeSeriesTrendMinRows = 18;
    size_t timeSeriesSeasonPeriod = 12;

    // LOF outlier detector tuning
    size_t lofMaxRows = 120000;
    double lofFallbackModifiedZThreshold = 3.5;
    double lofThresholdFloor = 1.5;

    // Scatter fitted-line suitability tuning
    double scatterFitMinAbsCorr = 0.35;
    size_t scatterFitMinSampleSize = 12;

    // Hybrid explainability blending (permutation / integrated gradients)
    double hybridExplainabilityWeightPermutation = 0.50;
    double hybridExplainabilityWeightIntegratedGradients = 0.50;

    // Project-timeline (Gantt) auto-detection tuning
    bool ganttAutoEnabled = true;
    size_t ganttMinTasks = 3;
    size_t ganttMaxTasks = 25;
    double ganttDurationHoursThreshold = 72.0;

    void validate() const;
};

struct AutoConfig {
    // Supported top-level config keys:
    // dataset, target, report, assets_dir, delimiter, exclude,
    // outlier_method, outlier_action, scaling,
    // kfold, max_feature_missing_ratio,
    // target_strategy, feature_strategy, neural_strategy, bivariate_strategy,
    // fast_mode, fast_max_bivariate_pairs, fast_neural_sample_rows,
    // plot_univariate, plot_overall, plot_bivariate_significant, plots,
    // plot_format, plot_theme, plot_grid, plot_width, plot_height, plot_point_size, plot_line_width,
    // generate_html, verbose_analysis,
    // neural_seed, benchmark_seed, gradient_clip_norm,
    // neural_optimizer, neural_lookahead_fast_optimizer, neural_lookahead_sync_period, neural_lookahead_alpha,
    // neural_use_batch_norm, neural_batch_norm_momentum, neural_batch_norm_epsilon,
    // neural_use_layer_norm, neural_layer_norm_epsilon,
    // neural_lr_decay, neural_lr_plateau_patience, neural_lr_cooldown_epochs,
    // neural_max_lr_reductions, neural_min_learning_rate,
    // neural_use_validation_loss_ema, neural_validation_loss_ema_beta,
    // neural_categorical_input_l2_boost,
    // neural_ensemble_members, neural_ensemble_probe_rows, neural_ensemble_probe_epochs,
    // neural_ood_enabled, neural_ood_z_threshold, neural_ood_distance_threshold,
    // neural_drift_psi_warning, neural_drift_psi_critical,
    // impute.<column_name> (per-column imputation strategy).
    std::string datasetPath;
    std::string reportFile = "neural_synthesis.md";
    std::string assetsDir = "seldon_report_assets";
    std::string outputDir;
    std::string targetColumn;
    char delimiter = ',';
    bool exhaustiveScan = false;
    bool interactiveMode = false;
    std::string profile = "auto"; // auto|quick|thorough|minimal
    std::string datetimeLocaleHint = "auto"; // auto|dmy|mdy
    std::string numericLocaleHint = "auto"; // auto|us|eu
    std::string exportPreprocessed = "none"; // none|csv|parquet
    std::string exportPreprocessedPath;

    std::vector<std::string> excludedColumns;
    std::unordered_map<std::string, std::string> columnImputation;
    std::unordered_map<std::string, std::string> columnTypeOverrides;

    std::string outlierMethod = "iqr";      // iqr|zscore|modified_zscore|adjusted_boxplot|lof
    std::string outlierAction = "flag";     // flag|remove|cap

    std::string scalingMethod = "auto";     // auto|zscore|minmax|none
    int kfold = 5;
    double maxFeatureMissingRatio = -1.0; // -1 => auto
    std::string targetStrategy = "auto";    // auto|quality|max_variance|last_numeric
    std::string featureStrategy = "auto";   // auto|adaptive|aggressive|lenient
    std::string neuralStrategy = "auto";    // auto|none|fast|balanced|expressive
    std::string bivariateStrategy = "auto"; // auto|balanced|corr_heavy|importance_heavy
    bool fastMode = false;
    bool lowMemoryMode = false;
    size_t fastMaxBivariatePairs = 2500;
    size_t fastNeuralSampleRows = 25000;

    bool plotUnivariate = false;
    bool plotOverall = false;
    bool plotBivariateSignificant = true;
    bool plotModesExplicit = false;
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
    int neuralLrWarmupEpochs = 5;
    bool neuralUseCosineAnnealing = true;
    bool neuralUseCyclicalLr = false;
    int neuralLrCycleEpochs = 24;
    double neuralLrScheduleMinFactor = 0.15;
    bool neuralUseValidationLossEma = true;
    double neuralValidationLossEmaBeta = 0.6;
    double neuralCategoricalInputL2Boost = 3.0;
    bool neuralUseAdaptiveGradientClipping = true;
    double neuralAdaptiveClipBeta = 0.90;
    double neuralAdaptiveClipMultiplier = 1.6;
    double neuralAdaptiveClipMin = 1.0;
    double neuralGradientNoiseStd = 0.01;
    double neuralGradientNoiseDecay = 0.995;
    bool neuralUseEmaWeights = true;
    double neuralEmaDecay = 0.995;
    double neuralLabelSmoothing = 0.02;
    int neuralGradientAccumulationSteps = 2;

    // Advanced neural controls
    double neuralLearningRate = 0.001;
    int neuralMinLayers = 1;
    int neuralMaxLayers = 3;
    int neuralFixedLayers = 0; // 0 => auto
    int neuralFixedHiddenNodes = 0; // 0 => auto
    int neuralMaxHiddenNodes = 128;
    bool neuralStreamingMode = false;
    size_t neuralStreamingChunkRows = 2048;
    bool neuralMultiOutput = true;
    size_t neuralMaxAuxTargets = 2;
    std::string neuralExplainability = "hybrid"; // permutation|integrated_gradients|hybrid
    size_t neuralIntegratedGradSteps = 8;
    size_t neuralUncertaintySamples = 24;
    size_t neuralEnsembleMembers = 3;
    size_t neuralEnsembleProbeRows = 256;
    size_t neuralEnsembleProbeEpochs = 48;
    bool neuralOodEnabled = true;
    double neuralOodZThreshold = 3.5;
    double neuralOodDistanceThreshold = 2.5;
    double neuralDriftPsiWarning = 0.15;
    double neuralDriftPsiCritical = 0.25;
    bool neuralImportanceParallel = true;
    size_t neuralImportanceMaxRows = 1000;
    size_t neuralImportanceTrials = 0; // 0 => auto
    size_t neuralMaxOneHotPerColumn = 24;
    size_t neuralMaxTopologyNodes = 4096;
    size_t neuralMaxTrainableParams = 20000000;

    // Auto feature engineering controls
    bool featureEngineeringEnablePoly = true;
    bool featureEngineeringEnableLog = true;
    bool featureEngineeringEnableRatioProductDiscovery = true;
    int featureEngineeringDegree = 2;
    size_t featureEngineeringMaxBase = 8;
    size_t featureEngineeringMaxPairwiseDiscovery = 24;
    size_t featureEngineeringMaxGeneratedColumns = 512;

    bool storeOutlierFlagsInReport = false;

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
