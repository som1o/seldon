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
    double featureMinVariance = 1e-10;
    double featureLeakageCorrThreshold = 0.995;
    double featureMissingQ3Offset = 0.15;
    double featureMissingAdaptiveMin = 0.35;
    double featureMissingAdaptiveMax = 0.95;
    double featureAggressiveDelta = 0.20;
    double featureAggressiveMin = 0.20;
    double featureAggressiveMax = 0.80;
    double featureLenientDelta = 0.20;
    double featureLenientMin = 0.40;
    double featureLenientMax = 0.98;
    double bivariateSelectionQuantileOverride = -1.0; // -1 => policy default
};

struct AutoConfig {
    std::string datasetPath;
    std::string reportFile = "neural_synthesis.txt";
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
    std::string neuralStrategy = "auto";    // auto|fast|balanced|expressive
    std::string bivariateStrategy = "auto"; // auto|balanced|corr_heavy|importance_heavy

    bool plotUnivariate = false;
    bool plotOverall = false;
    bool plotBivariateSignificant = true;
    bool verboseAnalysis = true;
    uint32_t neuralSeed = 1337;
    double gradientClipNorm = 5.0;

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
};
