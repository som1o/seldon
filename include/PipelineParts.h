#pragma once

#include "AutomationPipeline.h"
#include "AutoConfig.h"
#include "BenchmarkEngine.h"
#include "CausalDiscovery.h"
#include "CommonUtils.h"
#include "GnuplotEngine.h"
#include "MathUtils.h"
#include "NeuralNet.h"
#include "Preprocessor.h"
#include "ReportEngine.h"
#include "SeldonExceptions.h"
#include "Statistics.h"
#include "TypedDataset.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace seldon_pipeline {

struct DomainRuleBundle {
    std::unordered_set<std::string> suppressCausalColumns;
    std::unordered_set<std::string> downweightImportanceColumns;
    std::vector<std::string> loadedFrom;
};

struct BivariateScoringPolicy {
    std::string name = std::string(StrategyKeys::kBalanced);
    double wImportance = 0.50;
    double wCorrelation = 0.35;
    double wSignificance = 0.15;
    double selectionQuantile = 0.60;
    double coverageBoth = 1.0;
    double coverageOne = 0.60;
    double coverageNone = 0.25;
};

struct TargetChoice {
    int index = -1;
    std::string strategyUsed;
};

struct TargetSemantics {
    bool isBinary = false;
    bool isOrdinal = false;
    double lowLabel = 0.0;
    double highLabel = 1.0;
    size_t cardinality = 0;
    std::string inferredTask = "regression";
};

struct FeatureSelectionResult {
    std::vector<int> included;
    std::vector<std::string> droppedByMissingness;
    double missingThresholdUsed = 0.0;
    std::string strategyUsed;
};

struct PairInsight {
    size_t idxA = 0;
    size_t idxB = 0;
    std::string featureA;
    std::string featureB;
    double r = 0.0;
    double r2 = 0.0;
    double slope = 0.0;
    double intercept = 0.0;
    double spearman = 0.0;
    double kendallTau = 0.0;
    double tStat = 0.0;
    double pValue = 1.0;
    bool statSignificant = false;
    double neuralScore = 0.0;
    double effectSize = 0.0;
    double foldStability = 0.0;
    bool selected = false;
    int significanceTier = 0;
    std::string selectionReason;
    bool filteredAsRedundant = false;
    bool filteredAsStructural = false;
    bool leakageRisk = false;
    std::string relationLabel;
    std::string redundancyGroup;
    bool fitLineAdded = false;
    bool confidenceBandAdded = false;
    std::string plotPath;
    std::string residualPlotPath;
    std::string facetedPlotPath;
    std::string stackedPlotPath;
};

struct NeuralAnalysis {
    size_t inputNodes = 0;
    size_t numericInputNodes = 0;
    size_t categoricalEncodedNodes = 0;
    size_t categoricalColumnsUsed = 0;
    size_t hiddenNodes = 0;
    size_t outputNodes = 0;
    bool binaryTarget = false;
    bool classificationTarget = false;
    std::string hiddenActivation;
    std::string outputActivation;
    std::string lossName = "mse";
    size_t epochs = 0;
    size_t batchSize = 0;
    double valSplit = 0.0;
    double l2Lambda = 0.0;
    double dropoutRate = 0.0;
    int earlyStoppingPatience = 0;
    std::string policyUsed;
    std::string explainabilityMethod;
    std::string topology;
    size_t trainingRowsUsed = 0;
    size_t trainingRowsTotal = 0;
    size_t outputAuxTargets = 0;
    std::vector<double> trainLoss;
    std::vector<double> valLoss;
    std::vector<double> gradientNorm;
    std::vector<double> weightStd;
    std::vector<double> weightMeanAbs;
    std::vector<double> featureImportance;
    double categoricalImportanceShare = 0.0;
    std::vector<double> uncertaintyStd;
    std::vector<double> uncertaintyCiWidth;
    std::vector<double> ensembleStd;
    double confidenceScore = 0.0;
    double oodRate = 0.0;
    double oodMeanDistance = 0.0;
    double oodMaxDistance = 0.0;
    size_t oodReferenceRows = 0;
    size_t oodMonitorRows = 0;
    double driftPsiMean = 0.0;
    double driftPsiMax = 0.0;
    std::string driftBand = "stable";
    bool driftWarning = false;
    bool strictPruningApplied = false;
    size_t strictPrunedColumns = 0;
};

struct OodDriftDiagnostics {
    double oodRate = 0.0;
    double meanDistance = 0.0;
    double maxDistance = 0.0;
    size_t referenceRows = 0;
    size_t monitorRows = 0;
    double psiMean = 0.0;
    double psiMax = 0.0;
    std::string driftBand = "stable";
    bool warning = false;
};

struct BivariateStackedBarData {
    std::vector<std::string> categories;
    std::vector<double> lowCounts;
    std::vector<double> highCounts;
    bool valid = false;
};

struct FastRng {
    uint64_t state;
    explicit FastRng(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t nextU64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717ULL;
    }
    size_t uniformIndex(size_t upperExclusive) {
        if (upperExclusive == 0) return 0;
        return static_cast<size_t>(nextU64() % static_cast<uint64_t>(upperExclusive));
    }
};

struct DataHealthSummary {
    double score = 0.0;
    std::string band = "limited";
    double completeness = 0.0;
    double numericCoverage = 0.0;
    double featureRetention = 0.0;
    double statYield = 0.0;
    double selectedYield = 0.0;
    double trainingStability = 0.5;
    double driftPsiMean = 0.0;
    double driftPenalty = 0.0;
};

struct EncodedNeuralMatrix {
    std::vector<std::vector<double>> X;
    std::vector<int> sourceNumericFeaturePos;
    size_t categoricalEncodedNodes = 0;
    size_t categoricalColumnsUsed = 0;
};

struct NeuralCategoryPlan {
    size_t columnIdx = 0;
    std::vector<std::string_view> keptLabels;
    bool includeOther = false;
};

struct NeuralEncodingPlan {
    std::vector<int> sourceNumericFeaturePos;
    std::vector<size_t> numericColumns;
    std::vector<NeuralCategoryPlan> categoryPlans;
    size_t encodedWidth = 0;
    size_t categoricalEncodedNodes = 0;
    size_t categoricalColumnsUsed = 0;
};

using NumericStatsCache = std::unordered_map<size_t, ColumnStats>;

struct PreflightCullSummary {
    double threshold = 0.95;
    size_t dropped = 0;
    std::vector<std::string> droppedColumns;
};

struct TargetContext {
    bool userProvidedTarget = false;
    TargetChoice choice;
    int targetIdx = -1;
    TargetSemantics semantics;
    bool encodedFromCategorical = false;
    size_t encodedCardinality = 0;
};

struct ReportSaveSummary {
    std::string reportPath;
    size_t combinedBytes = 0;
    size_t finalBytes = 0;
};

struct ContingencyInsight {
    std::string catA;
    std::string catB;
    double chi2 = 0.0;
    double pValue = 1.0;
    double cramerV = 0.0;
    double oddsRatio = 1.0;
    double oddsCiLow = 1.0;
    double oddsCiHigh = 1.0;
};

struct AnovaInsight {
    std::string categorical;
    std::string numeric;
    double fStat = 0.0;
    double pValue = 1.0;
    double eta2 = 0.0;
    std::string tukeySummary;
};

struct AdvancedAnalyticsOutputs {
    std::vector<std::vector<std::string>> orderedRows;
    std::vector<std::vector<std::string>> mahalanobisRows;
    std::vector<std::vector<std::string>> pdpRows;
    std::vector<std::vector<std::string>> causalDagRows;
    std::vector<std::vector<std::string>> globalConditionalRows;
    std::vector<std::vector<std::string>> temporalDriftRows;
    std::vector<std::vector<std::string>> contextualDeadZoneRows;
    std::vector<std::string> narrativeRows;
    std::vector<std::string> priorityTakeaways;
    std::optional<std::string> interactionEvidence;
    std::optional<std::string> causalDagMermaid;
    std::unordered_map<size_t, double> mahalanobisByRow;
    double mahalanobisThreshold = 0.0;
    std::optional<std::string> executiveSummary;
};

struct ConditionalDriftAssessment {
    bool signFlip = false;
    bool magnitudeCollapse = false;
    double collapseRatio = 1.0;
    std::string label = "stable";
};

struct TemporalAxisDescriptor {
    std::vector<double> axis;
    std::string name;
};

struct ContextualDeadZoneInsight {
    std::string feature;
    std::string strongCluster;
    std::string weakCluster;
    double strongCorr = 0.0;
    double weakCorr = 0.0;
    double dropRatio = 1.0;
    size_t support = 0;
};

struct StratifiedPopulationInsight {
    std::string segmentColumn;
    std::string numericColumn;
    size_t groups = 0;
    size_t rows = 0;
    double eta2 = 0.0;
    double separation = 0.0;
    std::string groupMeans;
};

std::string toFixed(double v, int prec = 4);
std::string scoreBar100(double score);
void addExecutiveDashboard(ReportEngine& report,
                           const std::string& title,
                           const std::vector<std::pair<std::string, std::string>>& metrics,
                           const std::vector<std::string>& highlights,
                           const std::string& note = "");
bool containsToken(const std::string& text, const std::vector<std::string>& tokens);

bool isAdministrativeColumnName(const std::string& name);
bool isEngineeredFeatureName(const std::string& name);
bool isEngineeredLineagePair(const std::string& a, const std::string& b);
std::string canonicalEngineeredBaseName(const std::string& name);
bool sharesEngineeredRootIdentity(const std::string& a, const std::string& b);
std::string normalizeRuleName(const std::string& value);
DomainRuleBundle loadDomainRules(const AutoConfig& config);
bool isCausalEdgeSuppressedByRule(const std::string& from,
                                  const std::string& to,
                                  const DomainRuleBundle& rules);

std::string activationToString(NeuralNet::Activation activation);
void printProgressBar(const std::string& label, size_t current, size_t total);

bool shouldAddOgive(const std::vector<double>& values, const HeuristicTuningConfig& tuning);
bool shouldAddBoxPlot(const std::vector<double>& values, const HeuristicTuningConfig& tuning, double eps);
bool shouldAddConfidenceBand(double r,
                             bool statSignificant,
                             size_t sampleSize,
                             const HeuristicTuningConfig& tuning);
bool shouldAddResidualPlot(double r,
                           bool selected,
                           size_t sampleSize,
                           const HeuristicTuningConfig& tuning);
bool shouldOverlayFittedLine(double r,
                             bool statSignificant,
                             const std::vector<double>& x,
                             const std::vector<double>& y,
                             double slope,
                             double intercept,
                             const HeuristicTuningConfig& tuning);

std::vector<size_t> buildNormalizedTopology(size_t inputNodes,
                                            size_t outputNodes,
                                            size_t hiddenLayers,
                                            size_t firstHidden,
                                            size_t maxHiddenNodes);
std::optional<size_t> chooseFacetingColumn(const TypedDataset& data,
                                           size_t idxA,
                                           size_t idxB,
                                           const HeuristicTuningConfig& tuning);
BivariateStackedBarData buildBivariateStackedBar(const std::vector<double>& x,
                                                 const std::vector<double>& y);

void normalizeBinaryTarget(TypedDataset& data, int targetIdx, const TargetSemantics& semantics);
FeatureSelectionResult collectFeatureIndices(const TypedDataset& data,
                                            int targetIdx,
                                            const AutoConfig& config,
                                            const PreprocessReport& prep);
NumericStatsCache buildNumericStatsCache(const TypedDataset& data);
std::vector<double> computeFeatureTargetAbsCorr(const TypedDataset& data,
                                                int targetIdx,
                                                const std::vector<int>& featureIdx,
                                                const NumericStatsCache& statsCache);
EncodedNeuralMatrix buildEncodedNeuralInputs(const TypedDataset& data,
                                             int targetIdx,
                                             const std::vector<int>& numericFeatureIdx,
                                             const AutoConfig& config);
std::vector<double> buildCoherentImportance(const TypedDataset& data,
                                            int targetIdx,
                                            const std::vector<int>& featureIdx,
                                            const NeuralAnalysis& neural,
                                            const std::vector<BenchmarkResult>& benchmarks,
                                            const AutoConfig& config,
                                            const NumericStatsCache& statsCache);
std::vector<int> selectAuxiliaryNumericTargets(const TypedDataset& data,
                                               int primaryTargetIdx,
                                               size_t maxAuxTargets,
                                               const NumericStatsCache& statsCache);
NeuralEncodingPlan buildNeuralEncodingPlan(const TypedDataset& data,
                                           int targetIdx,
                                           const std::vector<int>& numericFeatureIdx,
                                           const AutoConfig& config);
void encodeNeuralRows(const TypedDataset& data,
                      const NeuralEncodingPlan& plan,
                      size_t rowStart,
                      size_t rowEnd,
                      std::vector<std::vector<double>>& out);
std::optional<std::string> writeCausalDagEditor(const AutoConfig& cfg,
                                                const std::vector<std::vector<std::string>>& causalRows);

std::string plotSubdir(const AutoConfig& cfg, const std::string& name);
void cleanupOutputs(const AutoConfig& config);
void cleanupPlotCacheArtifacts(const AutoConfig& config);
void validateExcludedColumns(const TypedDataset& data, const AutoConfig& config);
PreflightCullSummary applyPreflightSparseColumnCull(TypedDataset& data,
                                                    const std::optional<std::string>& protectedColumn,
                                                    bool verbose,
                                                    double threshold = 0.95);
TargetContext resolveTargetContext(TypedDataset& data, const AutoConfig& config, AutoConfig& runCfg);
void applyDynamicPlotDefaultsIfUnset(AutoConfig& runCfg, const TypedDataset& data);
bool configurePlotAvailability(AutoConfig& runCfg, ReportEngine& univariate, const GnuplotEngine& plotterBivariate);
void addUnivariatePlots(ReportEngine& univariate,
                        const TypedDataset& data,
                        const AutoConfig& runCfg,
                        bool canPlot,
                        GnuplotEngine& plotterUnivariate,
                        const std::unordered_set<size_t>& neuralApprovedNumericFeatures);
void exportPreprocessedDatasetIfRequested(const TypedDataset& data, const AutoConfig& runCfg);
void addUnivariateDetailedSection(ReportEngine& report,
                                  const TypedDataset& data,
                                  const PreprocessReport& prep,
                                  bool verbose,
                                  const NumericStatsCache& statsCache);
void addBenchmarkSection(ReportEngine& report, const std::vector<BenchmarkResult>& benchmarks);
void addDatasetHealthTable(ReportEngine& report,
                           const TypedDataset& data,
                           const PreprocessReport& prep,
                           const DataHealthSummary& health);
void addNeuralLossSummaryTable(ReportEngine& report, const NeuralAnalysis& neural);
double estimateTrainingStability(const std::vector<double>& trainLoss,
                                 const std::vector<double>& valLoss);
OodDriftDiagnostics computeOodDriftDiagnostics(const std::vector<std::vector<double>>& X,
                                               const AutoConfig& config);
double computeConfidenceScore(const std::vector<double>& uncertaintyStd,
                              const std::vector<double>& ensembleStd,
                              double oodRate,
                              const std::string& driftBand);
DataHealthSummary computeDataHealthSummary(const TypedDataset& data,
                                           const PreprocessReport& prep,
                                           const NeuralAnalysis& neural,
                                           size_t retainedFeatureCount,
                                           size_t pairsEvaluated,
                                           size_t statSigCount,
                                           size_t selectedPairCount);
ReportSaveSummary saveGeneratedReports(const AutoConfig& runCfg,
                                       const ReportEngine& univariate,
                                       const ReportEngine& bivariate,
                                       const ReportEngine& neuralReport,
                                       const ReportEngine& finalAnalysis,
                                       const ReportEngine& heuristicsReport);
void printPipelineCompletion(const AutoConfig& runCfg,
                             const ReportSaveSummary& saveSummary);

NeuralAnalysis runNeuralAnalysis(const TypedDataset& data,
                                 int targetIdx,
                                 const std::vector<int>& featureIdx,
                                 bool classificationTarget,
                                 bool ordinalTarget,
                                 size_t targetCardinality,
                                 bool verbose,
                                 const AutoConfig& config,
                                 bool fastModeEnabled,
                                 size_t fastSampleRows);
BivariateScoringPolicy chooseBivariatePolicy(const AutoConfig& config, const NeuralAnalysis& neural);
std::unordered_set<size_t> computeNeuralApprovedNumericFeatures(const TypedDataset& data,
                                                                int targetIdx,
                                                                const std::vector<int>& featureIdx,
                                                                const NeuralAnalysis& neural,
                                                                const NumericStatsCache& statsCache);
std::vector<PairInsight> analyzeBivariatePairs(const TypedDataset& data,
                                               const std::unordered_map<size_t, double>& importanceByIndex,
                                               const std::unordered_set<size_t>& modeledIndices,
                                               const BivariateScoringPolicy& policy,
                                               GnuplotEngine* plotter,
                                               bool verbose,
                                               const NumericStatsCache& statsCache,
                                               double significanceAlpha,
                                               const HeuristicTuningConfig& tuning,
                                               size_t maxPairs,
                                               size_t fastMaxPairs);
std::vector<ContingencyInsight> analyzeContingencyPairs(const TypedDataset& data);
std::vector<AnovaInsight> analyzeAnovaPairs(const TypedDataset& data);
ConditionalDriftAssessment assessGlobalConditionalDrift(double globalR,
                                                        double conditionalR,
                                                        double minGlobalAbs = 0.15,
                                                        double collapseRatioThreshold = 0.50,
                                                        double collapseAbsDrop = 0.12);
std::string driftPatternLabel(const std::string& raw);
std::string cramerStrengthLabel(double v);
TemporalAxisDescriptor detectTemporalAxis(const TypedDataset& data);
std::vector<ContextualDeadZoneInsight> detectContextualDeadZones(const TypedDataset& data,
                                                                 size_t targetIdx,
                                                                 const std::vector<size_t>& candidateFeatures,
                                                                 size_t maxRows = 10);

AdvancedAnalyticsOutputs buildAdvancedAnalyticsOutputs(const TypedDataset& data,
                                                      int targetIdx,
                                                      const std::vector<int>& featureIdx,
                                                      const std::vector<double>& featureImportance,
                                                      const std::vector<PairInsight>& bivariatePairs,
                                                      const std::vector<AnovaInsight>& anovaRows,
                                                      const std::vector<ContingencyInsight>& contingency,
                                                      const NumericStatsCache& statsCache);
std::vector<StratifiedPopulationInsight> detectStratifiedPopulations(const TypedDataset& data,
                                                                     size_t maxInsights = 12);
void addOverallSections(ReportEngine& report,
                        const TypedDataset& data,
                        const PreprocessReport& prep,
                        const std::vector<BenchmarkResult>& benchmarks,
                        const NeuralAnalysis& neural,
                        const DataHealthSummary& health,
                        const AutoConfig& config,
                        GnuplotEngine* overallPlotter,
                        bool canPlotOverall,
                        bool verbose,
                        const NumericStatsCache& statsCache,
                        const std::vector<int>& featureIdx,
                        const std::vector<PairInsight>& bivariatePairs);

std::optional<std::string> buildResidualDiscoveryNarrative(const TypedDataset& data,
                                                           int targetIdx,
                                                           const std::vector<int>& featureIdx,
                                                           const NumericStatsCache& statsCache);
std::vector<std::vector<std::string>> buildOutlierContextRows(const TypedDataset& data,
                                                              const PreprocessReport& prep,
                                                              const std::unordered_map<size_t, double>& mahalByRow = {},
                                                              double mahalThreshold = 0.0,
                                                              size_t maxRows = 8);

} // namespace seldon_pipeline
