#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
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
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cstdlib>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
std::string toFixed(double v, int prec = 4) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string activationToString(NeuralNet::Activation activation) {
    switch (activation) {
        case NeuralNet::Activation::SIGMOID: return "sigmoid";
        case NeuralNet::Activation::RELU: return "relu";
        case NeuralNet::Activation::TANH: return "tanh";
        case NeuralNet::Activation::LINEAR: return "linear";
        case NeuralNet::Activation::GELU: return "gelu";
    }
    return "unknown";
}

struct BivariateScoringPolicy {
    std::string name = "balanced";
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

TargetChoice resolveTargetChoice(const TypedDataset& data, const AutoConfig& config) {
    TargetChoice choice;

    if (!config.targetColumn.empty()) {
        int idx = data.findColumnIndex(config.targetColumn);
        if (idx < 0) {
            throw Seldon::ConfigurationException("Target column not found: " + config.targetColumn);
        }
        if (data.columns()[idx].type != ColumnType::NUMERIC) {
            throw Seldon::ConfigurationException("Target column must be numeric: " + config.targetColumn);
        }
        choice.index = idx;
        choice.strategyUsed = "user";
        return choice;
    }

    auto numeric = data.numericColumnIndices();
    if (numeric.empty()) {
        throw Seldon::ConfigurationException("Could not resolve a numeric target column");
    }

    auto scoreColumnQuality = [&](size_t idx) {
        const auto& col = data.columns()[idx];
        const auto& values = std::get<std::vector<double>>(col.values);
        size_t finiteCount = 0;
        size_t nonZero = 0;
        double mean = 0.0;
        double m2 = 0.0;
        for (double v : values) {
            if (!std::isfinite(v)) continue;
            ++finiteCount;
            const double delta = v - mean;
            mean += delta / static_cast<double>(finiteCount);
            const double delta2 = v - mean;
            m2 += delta * delta2;
            if (std::abs(v) > 1e-12) nonZero++;
        }
        if (finiteCount < 3) return -1e9;

        const double var = m2 / static_cast<double>(std::max<size_t>(1, finiteCount - 1));

        double missingRatio = 1.0 - (static_cast<double>(finiteCount) / static_cast<double>(std::max<size_t>(1, data.rowCount())));
        double nonZeroRatio = static_cast<double>(nonZero) / static_cast<double>(finiteCount);

        std::string lname = col.name;
        std::transform(lname.begin(), lname.end(), lname.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        double namePenalty = (lname.find("id") != std::string::npos || lname.find("index") != std::string::npos) ? 0.25 : 0.0;

        double varianceScore = std::clamp(std::log1p(std::max(0.0, var)) / 4.0, 0.0, 1.0);
        return 0.45 * (1.0 - missingRatio) + 0.35 * varianceScore + 0.20 * nonZeroRatio - namePenalty;
    };

    auto pickByQuality = [&]() {
        size_t best = numeric.front();
        double bestScore = -1e9;
        for (size_t idx : numeric) {
            double s = scoreColumnQuality(idx);
            if (s > bestScore) {
                bestScore = s;
                best = idx;
            }
        }
        return best;
    };

    auto pickByVariance = [&]() {
        size_t best = numeric.front();
        double bestVar = -1.0;
        for (size_t idx : numeric) {
            const auto& values = std::get<std::vector<double>>(data.columns()[idx].values);
            ColumnStats st = Statistics::calculateStats(values);
            if (std::isfinite(st.variance) && st.variance > bestVar) {
                bestVar = st.variance;
                best = idx;
            }
        }
        return best;
    };

    auto pickLastNumeric = [&]() { return numeric.back(); };

    std::string mode = CommonUtils::toLower(config.targetStrategy);
    if (mode == "quality") {
        choice.index = static_cast<int>(pickByQuality());
        choice.strategyUsed = "quality";
        return choice;
    }
    if (mode == "max_variance") {
        choice.index = static_cast<int>(pickByVariance());
        choice.strategyUsed = "max_variance";
        return choice;
    }
    if (mode == "last_numeric") {
        choice.index = static_cast<int>(pickLastNumeric());
        choice.strategyUsed = "last_numeric";
        return choice;
    }

    std::vector<size_t> votes = {pickByQuality(), pickByVariance(), pickLastNumeric()};
    std::unordered_map<size_t, int> count;
    for (size_t v : votes) count[v]++;

    size_t best = votes.front();
    int bestVotes = -1;
    double bestScore = -1e9;
    for (const auto& kv : count) {
        double s = scoreColumnQuality(kv.first);
        if (kv.second > bestVotes || (kv.second == bestVotes && s > bestScore)) {
            bestVotes = kv.second;
            bestScore = s;
            best = kv.first;
        }
    }

    choice.index = static_cast<int>(best);
    choice.strategyUsed = "auto_vote";
    return choice;
}

struct TargetSemantics {
    bool isBinary = false;
    double lowLabel = 0.0;
    double highLabel = 1.0;
};

TargetSemantics inferTargetSemanticsRaw(const TypedDataset& ds, int targetIdx) {
    TargetSemantics out;
    if (targetIdx < 0 || ds.columns()[targetIdx].type != ColumnType::NUMERIC) return out;

    const auto& y = std::get<std::vector<double>>(ds.columns()[targetIdx].values);
    std::set<double> uniq;
    for (double v : y) {
        if (!std::isfinite(v)) continue;
        uniq.insert(v);
        if (uniq.size() > 2) return out;
    }
    if (uniq.empty() || uniq.size() > 2) return out;

    if (uniq.size() == 1) {
        out.isBinary = true;
        out.lowLabel = *uniq.begin();
        out.highLabel = *uniq.begin();
        return out;
    }

    auto it = uniq.begin();
    out.lowLabel = *it;
    ++it;
    out.highLabel = *it;
    out.isBinary = true;
    return out;
}

struct FeatureSelectionResult {
    std::vector<int> included;
    std::vector<std::string> droppedByMissingness;
    double missingThresholdUsed = 0.0;
    std::string strategyUsed;
};

void normalizeBinaryTarget(TypedDataset& data, int targetIdx, const TargetSemantics& semantics) {
    if (!semantics.isBinary || targetIdx < 0 || data.columns()[targetIdx].type != ColumnType::NUMERIC) return;
    auto& y = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    const double low = semantics.lowLabel;
    const double high = semantics.highLabel;
    const double threshold = (low + high) / 2.0;

    for (double& v : y) {
        if (std::abs(v - low) <= 1e-9) {
            v = 0.0;
        } else if (std::abs(v - high) <= 1e-9) {
            v = 1.0;
        } else {
            v = (v > threshold) ? 1.0 : 0.0;
        }
    }
}

FeatureSelectionResult collectFeatureIndices(const TypedDataset& data,
                                            int targetIdx,
                                            const AutoConfig& config,
                                            const PreprocessReport& prep) {
    auto numeric = data.numericColumnIndices();
    FeatureSelectionResult out;
    const double minVar = config.tuning.featureMinVariance;
    const double leakageCorrThreshold = config.tuning.featureLeakageCorrThreshold;

    struct Candidate {
        size_t idx;
        std::string name;
        double missingRatio;
        double variance;
        double absTargetCorr;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(numeric.size());
    std::unordered_set<std::string> excluded(config.excludedColumns.begin(), config.excludedColumns.end());
    const size_t denominatorRows = (prep.originalRowCount > 0) ? prep.originalRowCount : data.rowCount();
    const auto& targetVals = std::get<std::vector<double>>(data.columns()[targetIdx].values);
    ColumnStats targetStats = Statistics::calculateStats(targetVals);

    for (size_t idx : numeric) {
        if (static_cast<int>(idx) == targetIdx) continue;
        const std::string& name = data.columns()[idx].name;
        if (excluded.find(name) != excluded.end()) continue;

        double missingRatio = 0.0;
        auto mit = prep.missingCounts.find(name);
        if (mit != prep.missingCounts.end() && denominatorRows > 0) {
            missingRatio = static_cast<double>(mit->second) / static_cast<double>(denominatorRows);
        }
        missingRatio = std::clamp(missingRatio, 0.0, 1.0);

        const auto& values = std::get<std::vector<double>>(data.columns()[idx].values);
        ColumnStats st = Statistics::calculateStats(values);
        if (!std::isfinite(st.variance) || st.variance < minVar) continue;
        double r = std::abs(MathUtils::calculatePearson(values, targetVals, st, targetStats).value_or(0.0));

        if (r >= leakageCorrThreshold) {
            out.droppedByMissingness.push_back(name + " (auto_leak_guard abs_corr=" + toFixed(r, 4) + ")");
            continue;
        }

        candidates.push_back({idx, name, missingRatio, st.variance, r});
    }

    if (candidates.empty()) {
        throw Seldon::ConfigurationException("No numeric input features available after exclusions");
    }

    std::vector<double> ratios;
    ratios.reserve(candidates.size());
    for (const auto& c : candidates) ratios.push_back(c.missingRatio);
    std::sort(ratios.begin(), ratios.end());
    size_t q3Pos = static_cast<size_t>(std::floor(0.75 * static_cast<double>(ratios.size() - 1)));
    double q3 = ratios[q3Pos];
    double adaptiveThreshold = std::clamp(
        q3 + config.tuning.featureMissingQ3Offset,
        config.tuning.featureMissingAdaptiveMin,
        config.tuning.featureMissingAdaptiveMax);
    auto aggressiveThreshold = [&]() {
        return std::clamp(adaptiveThreshold - config.tuning.featureAggressiveDelta,
                          config.tuning.featureAggressiveMin,
                          config.tuning.featureAggressiveMax);
    };
    auto lenientThreshold = [&]() {
        return std::clamp(adaptiveThreshold + config.tuning.featureLenientDelta,
                          config.tuning.featureLenientMin,
                          config.tuning.featureLenientMax);
    };

    struct FeaturePolicy {
        std::string name;
        double threshold;
    };

    std::vector<FeaturePolicy> policies;
    if (config.maxFeatureMissingRatio >= 0.0) {
        policies.push_back({"fixed", config.maxFeatureMissingRatio});
    } else {
        std::string mode = CommonUtils::toLower(config.featureStrategy);
        if (mode == "adaptive") {
            policies.push_back({"adaptive", adaptiveThreshold});
        } else if (mode == "aggressive") {
            policies.push_back({"aggressive", aggressiveThreshold()});
        } else if (mode == "lenient") {
            policies.push_back({"lenient", lenientThreshold()});
        } else {
            policies.push_back({"adaptive", adaptiveThreshold});
            policies.push_back({"aggressive", aggressiveThreshold()});
            policies.push_back({"lenient", lenientThreshold()});
        }
    }

    std::unordered_map<int, const Candidate*> candidateByIndex;
    candidateByIndex.reserve(candidates.size());
    for (const auto& c : candidates) {
        candidateByIndex[static_cast<int>(c.idx)] = &c;
    }

    auto evaluatePolicy = [&](const FeaturePolicy& policy) {
        FeatureSelectionResult res;
        res.strategyUsed = policy.name;
        res.missingThresholdUsed = policy.threshold;

        std::vector<Candidate> deferred;
        for (const auto& c : candidates) {
            if (c.missingRatio > policy.threshold) {
                res.droppedByMissingness.push_back(c.name + " (missing_ratio=" + toFixed(c.missingRatio, 4) + ")");
                deferred.push_back(c);
                continue;
            }
            res.included.push_back(static_cast<int>(c.idx));
        }

        if (res.included.size() < 3 && !deferred.empty()) {
            std::sort(deferred.begin(), deferred.end(), [](const Candidate& a, const Candidate& b) {
                return a.missingRatio < b.missingRatio;
            });
            for (const auto& c : deferred) {
                if (res.included.size() >= 3) break;
                res.included.push_back(static_cast<int>(c.idx));
            }
        }

        double corrSum = 0.0;
        double missSum = 0.0;
        for (int idx : res.included) {
            auto cit = candidateByIndex.find(idx);
            if (cit != candidateByIndex.end() && cit->second != nullptr) {
                corrSum += cit->second->absTargetCorr;
                missSum += cit->second->missingRatio;
            }
        }
        double meanCorr = res.included.empty() ? 0.0 : corrSum / static_cast<double>(res.included.size());
        double meanMissing = res.included.empty() ? 1.0 : missSum / static_cast<double>(res.included.size());
        double coverageScore = 1.0 - meanMissing;
        double countScore = std::clamp(static_cast<double>(res.included.size()) / 8.0, 0.0, 1.0);
        double utility = 0.55 * meanCorr + 0.25 * coverageScore + 0.20 * countScore;
        return std::make_pair(res, utility);
    };

    FeatureSelectionResult best;
    double bestUtility = -1e9;
    for (const auto& p : policies) {
        auto [candidateRes, utility] = evaluatePolicy(p);
        if (!candidateRes.included.empty() && utility > bestUtility) {
            bestUtility = utility;
            best = std::move(candidateRes);
        }
    }

    if (best.included.empty() && !candidates.empty()) {
        auto keep = std::max_element(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.absTargetCorr == b.absTargetCorr) return a.variance < b.variance;
            return a.absTargetCorr < b.absTargetCorr;
        });
        if (keep != candidates.end()) {
            best.included.push_back(static_cast<int>(keep->idx));
            if (best.strategyUsed.empty()) best.strategyUsed = "fallback_single_feature";
            if (best.missingThresholdUsed <= 0.0) best.missingThresholdUsed = 1.0;
            best.droppedByMissingness.push_back("fallback retained: " + keep->name + " (no other usable numeric features)");
        }
    }
    return best;
}

struct PairInsight {
    size_t idxA = 0;
    size_t idxB = 0;
    std::string featureA;
    std::string featureB;
    double r = 0.0;
    double r2 = 0.0;
    double slope = 0.0;
    double intercept = 0.0;
    double tStat = 0.0;
    double pValue = 1.0;
    bool statSignificant = false;
    double neuralScore = 0.0;
    bool selected = false;
    std::string plotPath;
};

struct NeuralAnalysis {
    size_t inputNodes = 0;
    size_t numericInputNodes = 0;
    size_t categoricalEncodedNodes = 0;
    size_t categoricalColumnsUsed = 0;
    size_t hiddenNodes = 0;
    size_t outputNodes = 0;
    bool binaryTarget = false;
    std::string hiddenActivation;
    std::string outputActivation;
    size_t epochs = 0;
    size_t batchSize = 0;
    double valSplit = 0.0;
    double l2Lambda = 0.0;
    double dropoutRate = 0.0;
    int earlyStoppingPatience = 0;
    std::string policyUsed;
    size_t trainingRowsUsed = 0;
    size_t trainingRowsTotal = 0;
    std::vector<double> trainLoss;
    std::vector<double> valLoss;
    std::vector<double> featureImportance;
    double categoricalImportanceShare = 0.0;
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
};

double safeRatio(size_t numerator, size_t denominator) {
    if (denominator == 0) return 0.0;
    return std::clamp(static_cast<double>(numerator) / static_cast<double>(denominator), 0.0, 1.0);
}

std::string healthBandFromScore(double score) {
    if (score >= 85.0) return "excellent";
    if (score >= 70.0) return "strong";
    if (score >= 55.0) return "moderate";
    if (score >= 40.0) return "weak";
    return "limited";
}

DataHealthSummary computeDataHealthSummary(const TypedDataset& data,
                                           const PreprocessReport& prep,
                                           const NeuralAnalysis& neural,
                                           size_t retainedFeatureCount,
                                           size_t pairsEvaluated,
                                           size_t statSigCount,
                                           size_t selectedPairCount) {
    DataHealthSummary out;

    size_t totalMissing = 0;
    for (const auto& kv : prep.missingCounts) totalMissing += kv.second;

    const size_t rows = std::max<size_t>(1, data.rowCount());
    const size_t cols = std::max<size_t>(1, data.colCount());
    const size_t numericCols = data.numericColumnIndices().size();
    const size_t totalCells = rows * cols;

    out.completeness = 1.0 - safeRatio(totalMissing, totalCells);
    out.numericCoverage = safeRatio(numericCols, cols);

    const size_t possibleFeatures = (numericCols > 0) ? (numericCols - 1) : 0;
    out.featureRetention = (possibleFeatures == 0)
        ? 0.0
        : safeRatio(retainedFeatureCount, possibleFeatures);

    if (pairsEvaluated > 0) {
        const double expectedStat = std::log1p(static_cast<double>(std::max<size_t>(1, pairsEvaluated / 4)));
        const double expectedSelected = std::log1p(static_cast<double>(std::max<size_t>(1, pairsEvaluated / 10)));
        out.statYield = expectedStat <= 0.0
            ? 0.0
            : std::clamp(std::log1p(static_cast<double>(statSigCount)) / expectedStat, 0.0, 1.0);
        out.selectedYield = expectedSelected <= 0.0
            ? 0.0
            : std::clamp(std::log1p(static_cast<double>(selectedPairCount)) / expectedSelected, 0.0, 1.0);
    }

    out.trainingStability = 0.5;
    if (!neural.trainLoss.empty()) {
        double trainStart = neural.trainLoss.front();
        double trainEnd = neural.trainLoss.back();
        double baseline = std::max(1e-9, std::abs(trainStart));
        double improvement = (trainStart - trainEnd) / baseline;
        double convergence = std::clamp(0.5 + 0.5 * improvement, 0.0, 1.0);

        if (!neural.valLoss.empty() && neural.valLoss.size() == neural.trainLoss.size()) {
            double valEnd = neural.valLoss.back();
            double gap = std::abs(valEnd - trainEnd) / std::max(1e-9, std::abs(trainEnd) + 1e-9);
            double generalization = std::clamp(1.0 - gap, 0.0, 1.0);
            out.trainingStability = 0.7 * convergence + 0.3 * generalization;
        } else {
            out.trainingStability = convergence;
        }
    }

    const double score01 =
        0.30 * out.completeness +
        0.10 * out.numericCoverage +
        0.15 * out.featureRetention +
        0.20 * out.statYield +
        0.15 * out.selectedYield +
        0.10 * out.trainingStability;

    out.score = std::clamp(score01 * 100.0, 0.0, 100.0);
    out.band = healthBandFromScore(out.score);
    return out;
}

struct EncodedNeuralMatrix {
    std::vector<std::vector<double>> X;
    std::vector<int> sourceNumericFeaturePos;
    size_t categoricalEncodedNodes = 0;
    size_t categoricalColumnsUsed = 0;
};

using NumericStatsCache = std::unordered_map<size_t, ColumnStats>;

NumericStatsCache buildNumericStatsCache(const TypedDataset& data) {
    NumericStatsCache cache;
    const auto numericIdx = data.numericColumnIndices();
    cache.reserve(numericIdx.size());
    for (size_t idx : numericIdx) {
        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        cache.emplace(idx, Statistics::calculateStats(vals));
    }
    return cache;
}

std::vector<double> normalizeNonNegative(const std::vector<double>& values) {
    std::vector<double> out(values.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = std::max(0.0, values[i]);
        sum += out[i];
    }
    if (sum > 1e-12) {
        for (double& v : out) v /= sum;
    } else if (!out.empty()) {
        const double u = 1.0 / static_cast<double>(out.size());
        for (double& v : out) v = u;
    }
    return out;
}

std::vector<double> computeFeatureTargetAbsCorr(const TypedDataset& data,
                                                int targetIdx,
                                                const std::vector<int>& featureIdx,
                                                const NumericStatsCache& statsCache) {
    std::vector<double> out(featureIdx.size(), 0.0);
    if (targetIdx < 0 || static_cast<size_t>(targetIdx) >= data.columns().size()) return out;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
    const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
    const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        int idx = featureIdx[i];
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        auto xIt = statsCache.find(static_cast<size_t>(idx));
        const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
        out[i] = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
    }
    return out;
}

EncodedNeuralMatrix buildEncodedNeuralInputs(const TypedDataset& data,
                                             int targetIdx,
                                             const std::vector<int>& numericFeatureIdx,
                                             const AutoConfig& config) {
    EncodedNeuralMatrix out;
    out.X.assign(data.rowCount(), std::vector<double>{});

    const std::unordered_set<std::string> excluded(config.excludedColumns.begin(), config.excludedColumns.end());

    for (size_t featurePos = 0; featurePos < numericFeatureIdx.size(); ++featurePos) {
        int idx = numericFeatureIdx[featurePos];
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        const auto& values = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        for (size_t r = 0; r < out.X.size() && r < values.size(); ++r) {
            out.X[r].push_back(values[r]);
        }
        out.sourceNumericFeaturePos.push_back(static_cast<int>(featurePos));
    }

    constexpr size_t kMaxOneHotPerColumn = 24;
    const auto categoricalIdx = data.categoricalColumnIndices();
    for (size_t idx : categoricalIdx) {
        if (static_cast<int>(idx) == targetIdx) continue;
        const std::string& columnName = data.columns()[idx].name;
        if (excluded.find(columnName) != excluded.end()) continue;

        const auto& values = std::get<std::vector<std::string>>(data.columns()[idx].values);
        if (values.empty()) continue;

        std::unordered_map<std::string, size_t> freq;
        for (const auto& v : values) {
            if (!v.empty()) freq[v]++;
        }
        if (freq.empty()) continue;

        std::vector<std::pair<std::string, size_t>> categories(freq.begin(), freq.end());
        std::sort(categories.begin(), categories.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });

        const size_t keepCount = std::min(kMaxOneHotPerColumn, categories.size());
        if (keepCount == 0) continue;

        std::unordered_set<std::string> kept;
        kept.reserve(keepCount);
        for (size_t i = 0; i < keepCount; ++i) {
            const std::string& label = categories[i].first;
            kept.insert(label);
            for (size_t r = 0; r < out.X.size() && r < values.size(); ++r) {
                out.X[r].push_back(values[r] == label ? 1.0 : 0.0);
            }
            out.sourceNumericFeaturePos.push_back(-1);
            out.categoricalEncodedNodes++;
        }

        if (categories.size() > keepCount) {
            for (size_t r = 0; r < out.X.size() && r < values.size(); ++r) {
                out.X[r].push_back(kept.find(values[r]) == kept.end() ? 1.0 : 0.0);
            }
            out.sourceNumericFeaturePos.push_back(-1);
            out.categoricalEncodedNodes++;
        }

        out.categoricalColumnsUsed++;
    }

    return out;
}

double bestBenchmarkRmse(const std::vector<BenchmarkResult>& benchmarks) {
    if (benchmarks.empty()) return std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    for (const auto& b : benchmarks) {
        if (std::isfinite(b.rmse) && b.rmse >= 0.0) best = std::min(best, b.rmse);
    }
    return best;
}

std::vector<double> buildCoherentImportance(const TypedDataset& data,
                                            int targetIdx,
                                            const std::vector<int>& featureIdx,
                                            const NeuralAnalysis& neural,
                                            const std::vector<BenchmarkResult>& benchmarks,
                                            const AutoConfig& config,
                                            const NumericStatsCache& statsCache) {
    std::vector<double> neuralNorm = normalizeNonNegative(neural.featureImportance);
    std::vector<double> corrNorm = normalizeNonNegative(computeFeatureTargetAbsCorr(data, targetIdx, featureIdx, statsCache));

    if (neuralNorm.size() != corrNorm.size()) {
        return corrNorm.empty() ? neuralNorm : corrNorm;
    }

    double neuralWeight = (data.rowCount() < 80)
        ? config.tuning.coherenceWeightSmallDataset
        : config.tuning.coherenceWeightRegularDataset;
    if (!neural.trainLoss.empty() && !neural.valLoss.empty()) {
        double trainLast = std::max(neural.trainLoss.back(), config.tuning.numericEpsilon);
        double valLast = std::max(neural.valLoss.back(), 0.0);
        if (valLast > (config.tuning.coherenceOverfitPenaltyTrainRatio * trainLast)) {
            neuralWeight -= config.tuning.coherencePenaltyStep;
        }
    }

    const double bestRmse = bestBenchmarkRmse(benchmarks);
    if (std::isfinite(bestRmse) && bestRmse >= 0.0 && !neural.valLoss.empty()) {
        double neuralValRmseApprox = std::sqrt(std::max(neural.valLoss.back(), 0.0));
        if (neuralValRmseApprox > (config.tuning.coherenceBenchmarkPenaltyRatio * bestRmse)) {
            neuralWeight -= config.tuning.coherencePenaltyStep;
        }
    }

    neuralWeight = std::clamp(neuralWeight, config.tuning.coherenceWeightMin, config.tuning.coherenceWeightMax);
    const double corrWeight = 1.0 - neuralWeight;

    std::vector<double> blended(neuralNorm.size(), 0.0);
    for (size_t i = 0; i < blended.size(); ++i) {
        blended[i] = neuralWeight * neuralNorm[i] + corrWeight * corrNorm[i];
    }
    return normalizeNonNegative(blended);
}

using NumericDetailedStats = MathUtils::NumericSummary;

NumericDetailedStats detailedNumeric(const std::vector<double>& values, const ColumnStats* precomputedStats = nullptr) {
    return MathUtils::summarizeNumeric(values, precomputedStats);
}

std::string plotSubdir(const AutoConfig& cfg, const std::string& name) {
    return cfg.assetsDir + "/" + name;
}

void cleanupOutputs(const AutoConfig& config) {
    namespace fs = std::filesystem;

    const std::vector<std::string> rootFiles = {
        "seldon_report.html",
        "univariate.html",
        "bivariate.html",
        "seldon_model.seldon",
        "univariate.md",
        "bivariate.md",
        "neural_synthesis.md",
        "final_analysis.md"
    };

    std::error_code ec;
    for (const auto& f : rootFiles) {
        fs::remove(f, ec);
    }
    fs::remove(config.reportFile, ec);

    fs::remove_all(config.assetsDir, ec);
    fs::create_directories(config.assetsDir, ec);
}

void validateExcludedColumns(const TypedDataset& data, const AutoConfig& config) {
    for (const auto& ex : config.excludedColumns) {
        if (data.findColumnIndex(ex) < 0) {
            throw Seldon::ConfigurationException("Excluded column not found: " + ex);
        }
    }
}

struct TargetContext {
    bool userProvidedTarget = false;
    TargetChoice choice;
    int targetIdx = -1;
    TargetSemantics semantics;
};

TargetContext resolveTargetContext(const TypedDataset& data, const AutoConfig& config, AutoConfig& runCfg) {
    TargetContext context;
    context.userProvidedTarget = !config.targetColumn.empty();
    context.choice = resolveTargetChoice(data, config);
    context.targetIdx = context.choice.index;
    context.semantics = inferTargetSemanticsRaw(data, context.targetIdx);
    runCfg.targetColumn = data.columns()[context.targetIdx].name;

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Target] "
                  << (context.userProvidedTarget ? "Using user-selected target: " : "Auto-selected target: ")
                  << runCfg.targetColumn
                  << " (strategy=" << context.choice.strategyUsed << ")\n";
    }

    return context;
}

bool configurePlotAvailability(AutoConfig& runCfg, ReportEngine& univariate, const GnuplotEngine& plotterBivariate) {
    const bool canPlot = plotterBivariate.isAvailable();
    univariate.addParagraph(canPlot ? "Gnuplot detected." : "Gnuplot not available: plot generation skipped.");
    if (!canPlot && (runCfg.plotUnivariate || runCfg.plotOverall || runCfg.plotBivariateSignificant)) {
        runCfg.plotUnivariate = false;
        runCfg.plotOverall = false;
        runCfg.plotBivariateSignificant = false;
        univariate.addParagraph("Requested plotting features were disabled because gnuplot is unavailable in PATH.");
    }
    return canPlot;
}

void addUnivariatePlots(ReportEngine& univariate,
                        const TypedDataset& data,
                        const AutoConfig& runCfg,
                        bool canPlot,
                        GnuplotEngine& plotterUnivariate) {
    if (!(runCfg.plotUnivariate && canPlot)) {
        univariate.addParagraph("Supervised setting disabled: univariate plots skipped.");
        return;
    }

    univariate.addParagraph("Supervised setting enabled: univariate plots generated in a dedicated folder.");
    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Generating supervised univariate plots...\n";
    }

    auto logGeneratedPlot = [&](const std::string& img) {
        if (runCfg.verboseAnalysis) std::cout << "[Seldon][Univariate] Plot generated: " << img << "\n";
    };

    for (size_t idx : data.numericColumnIndices()) {
        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        std::string img = plotterUnivariate.histogram("uni_hist_" + std::to_string(idx), vals, "Histogram: " + data.columns()[idx].name);
        if (!img.empty()) {
            univariate.addImage("Histogram: " + data.columns()[idx].name, img);
            logGeneratedPlot(img);
        }
    }

    for (size_t idx : data.categoricalColumnIndices()) {
        const auto& vals = std::get<std::vector<std::string>>(data.columns()[idx].values);
        std::map<std::string, size_t> freq;
        for (const auto& v : vals) freq[v]++;
        std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        if (top.size() > 12) top.resize(12);
        std::vector<std::string> labels;
        std::vector<double> counts;
        for (const auto& kv : top) {
            labels.push_back(kv.first);
            counts.push_back(static_cast<double>(kv.second));
        }
        std::string img = plotterUnivariate.bar("uni_cat_" + std::to_string(idx), labels, counts, "Category Frequencies: " + data.columns()[idx].name);
        if (!img.empty()) {
            univariate.addImage("Category Plot: " + data.columns()[idx].name, img);
            logGeneratedPlot(img);
        }
    }
}

void saveGeneratedReports(const AutoConfig& runCfg,
                          const ReportEngine& univariate,
                          const ReportEngine& bivariate,
                          const ReportEngine& neuralReport,
                          const ReportEngine& finalAnalysis) {
    univariate.save("univariate.md");
    bivariate.save("bivariate.md");
    neuralReport.save(runCfg.reportFile);
    finalAnalysis.save("final_analysis.md");

    if (runCfg.generateHtml) {
        const std::vector<std::pair<std::string, std::string>> conversions = {
            {"univariate.md", "univariate.html"},
            {"bivariate.md", "bivariate.html"},
            {runCfg.reportFile, "neural_synthesis.html"},
            {"final_analysis.md", "final_analysis.html"}
        };
        for (const auto& [src, dst] : conversions) {
            const std::string cmd = "pandoc \"" + src + "\" -o \"" + dst + "\" --standalone --self-contained > /dev/null 2>&1";
            std::system(cmd.c_str());
        }
    }
}

void printPipelineCompletion(const AutoConfig& runCfg) {
    std::cout << "[Seldon] Automated pipeline complete.\n";
    std::cout << "[Seldon] Reports: univariate.md, bivariate.md, " << runCfg.reportFile << ", final_analysis.md\n";
    if (runCfg.generateHtml) {
        std::cout << "[Seldon] HTML reports (self-contained): univariate.html, bivariate.html, neural_synthesis.html, final_analysis.html\n";
    }
    std::cout << "[Seldon] Plot folders: "
              << plotSubdir(runCfg, "univariate") << ", "
              << plotSubdir(runCfg, "bivariate") << ", "
              << plotSubdir(runCfg, "overall") << "\n";
}

void addUnivariateDetailedSection(ReportEngine& report,
                                  const TypedDataset& data,
                                  const PreprocessReport& prep,
                                  bool verbose,
                                  const NumericStatsCache& statsCache) {
    std::vector<std::vector<std::string>> summaryRows;
    std::vector<std::vector<std::string>> categoricalRows;
    summaryRows.reserve(data.colCount());

    if (verbose) {
        std::cout << "[Seldon][Univariate] Starting per-column statistical profiling...\n";
    }

    auto logVerbose = [&](const std::string& line) {
        if (verbose) std::cout << line << "\n";
    };

    for (const auto& col : data.columns()) {
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : 0;
        const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;

        if (col.type == ColumnType::NUMERIC) {
            const auto& vals = std::get<std::vector<double>>(col.values);
            int idx = data.findColumnIndex(col.name);
            const auto it = (idx >= 0) ? statsCache.find(static_cast<size_t>(idx)) : statsCache.end();
            const ColumnStats* precomputed = (it != statsCache.end()) ? &it->second : nullptr;
            NumericDetailedStats st = detailedNumeric(vals, precomputed);

            summaryRows.push_back({
                col.name,
                "numeric",
                toFixed(st.mean),
                toFixed(st.median),
                toFixed(st.min),
                toFixed(st.max),
                toFixed(st.range),
                toFixed(st.q1),
                toFixed(st.q3),
                toFixed(st.iqr),
                toFixed(st.p05),
                toFixed(st.p95),
                toFixed(st.mad),
                toFixed(st.variance),
                toFixed(st.stddev),
                toFixed(st.skewness),
                toFixed(st.kurtosis),
                toFixed(st.coeffVar),
                toFixed(st.sum),
                std::to_string(st.nonZero),
                std::to_string(missing),
                std::to_string(outliers)
            });
            logVerbose("[Seldon][Univariate] Numeric " + col.name
                       + " | n=" + std::to_string(vals.size())
                       + " mean=" + toFixed(st.mean)
                       + " std=" + toFixed(st.stddev)
                       + " iqr=" + toFixed(st.iqr)
                       + " missing=" + std::to_string(missing)
                       + " outliers=" + std::to_string(outliers));
        } else if (col.type == ColumnType::CATEGORICAL) {
            const auto& vals = std::get<std::vector<std::string>>(col.values);
            std::map<std::string, size_t> freq;
            for (const auto& v : vals) freq[v]++;

            std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string mode = top.empty() ? "-" : top.front().first;
            size_t modeCount = top.empty() ? 0 : top.front().second;
            double modeRatio = vals.empty() ? 0.0 : static_cast<double>(modeCount) / static_cast<double>(vals.size());

            summaryRows.push_back({
                col.name,
                "categorical",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                std::to_string(freq.size()),
                std::to_string(missing),
                "-"
            });

            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                categoricalRows.push_back({col.name, top[i].first, std::to_string(top[i].second), toFixed(static_cast<double>(top[i].second) / static_cast<double>(std::max<size_t>(1, vals.size())), 6)});
            }

            logVerbose("[Seldon][Univariate] Categorical " + col.name
                       + " | unique=" + std::to_string(freq.size())
                       + " mode='" + mode + "'"
                       + " mode_ratio=" + toFixed(modeRatio, 6)
                       + " missing=" + std::to_string(missing));
        } else {
            const auto& vals = std::get<std::vector<int64_t>>(col.values);
            int64_t minTs = vals.empty() ? 0 : *std::min_element(vals.begin(), vals.end());
            int64_t maxTs = vals.empty() ? 0 : *std::max_element(vals.begin(), vals.end());
            summaryRows.push_back({
                col.name,
                "datetime",
                "-",
                "-",
                std::to_string(minTs),
                std::to_string(maxTs),
                std::to_string(maxTs - minTs),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                std::to_string(vals.size()),
                std::to_string(missing),
                "-"
            });
            logVerbose("[Seldon][Univariate] Datetime " + col.name
                       + " | min_ts=" + std::to_string(minTs)
                       + " max_ts=" + std::to_string(maxTs)
                       + " missing=" + std::to_string(missing));
        }
    }

    report.addTable("Column Statistical Super-Summary", {
        "Column", "Type", "Mean", "Median", "Min", "Max", "Range", "Q1", "Q3", "IQR",
        "P05", "P95", "MAD", "Variance", "StdDev", "Skew", "Kurtosis", "CoeffVar", "Sum", "NonZero/Unique", "Missing", "Outliers"
    }, summaryRows);

    if (!categoricalRows.empty()) {
        report.addTable("Categorical Top Frequencies", {"Column", "Category", "Count", "Share"}, categoricalRows);
    }
}

void addBenchmarkSection(ReportEngine& report, const std::vector<BenchmarkResult>& benchmarks) {
    std::vector<BenchmarkResult> ranked = benchmarks;
    bool classificationMode = std::any_of(ranked.begin(), ranked.end(), [](const BenchmarkResult& b) { return b.hasAccuracy; });
    std::sort(ranked.begin(), ranked.end(), [&](const BenchmarkResult& a, const BenchmarkResult& b) {
        if (classificationMode) {
            if (a.hasAccuracy != b.hasAccuracy) return a.hasAccuracy > b.hasAccuracy;
            if (a.accuracy != b.accuracy) return a.accuracy > b.accuracy;
            return a.rmse < b.rmse;
        }
        if (a.rmse != b.rmse) return a.rmse < b.rmse;
        return a.r2 > b.r2;
    });

    std::vector<std::vector<std::string>> benchRows;
    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto& b = ranked[i];
        benchRows.push_back({std::to_string(i + 1), b.model, toFixed(b.rmse), toFixed(b.r2), b.hasAccuracy ? toFixed(b.accuracy) : "N/A"});
    }
    report.addTable("Baseline Benchmark (k-fold)", {"Rank", "Model", "RMSE", "R2", "Accuracy"}, benchRows);
}

void addDatasetHealthTable(ReportEngine& report,
                           const TypedDataset& data,
                           const PreprocessReport& prep,
                           const DataHealthSummary& health) {
    size_t totalMissing = 0;
    size_t totalOutliers = 0;
    for (const auto& kv : prep.missingCounts) totalMissing += kv.second;
    for (const auto& kv : prep.outlierCounts) totalOutliers += kv.second;

    report.addTable("Dataset Health", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Columns", std::to_string(data.colCount())},
        {"Numeric Columns", std::to_string(data.numericColumnIndices().size())},
        {"Categorical Columns", std::to_string(data.categoricalColumnIndices().size())},
        {"Datetime Columns", std::to_string(data.datetimeColumnIndices().size())},
        {"Total Missing Cells", std::to_string(totalMissing)},
        {"Total Outlier Flags", std::to_string(totalOutliers)},
        {"Data Health Score (0-100)", toFixed(health.score, 1)},
        {"Data Health Band", health.band},
        {"Signal Yield (selected/evaluated)", toFixed(100.0 * health.selectedYield, 1) + "%"}
    });
}

void addNeuralLossSummaryTable(ReportEngine& report, const NeuralAnalysis& neural) {
    std::vector<std::vector<std::string>> lossSummary;
    if (!neural.trainLoss.empty()) {
        double minTrain = *std::min_element(neural.trainLoss.begin(), neural.trainLoss.end());
        double maxTrain = *std::max_element(neural.trainLoss.begin(), neural.trainLoss.end());
        double endTrain = neural.trainLoss.back();
        lossSummary.push_back({"train", toFixed(minTrain, 6), toFixed(maxTrain, 6), toFixed(endTrain, 6)});
    }
    if (!neural.valLoss.empty()) {
        double minVal = *std::min_element(neural.valLoss.begin(), neural.valLoss.end());
        double maxVal = *std::max_element(neural.valLoss.begin(), neural.valLoss.end());
        double endVal = neural.valLoss.back();
        lossSummary.push_back({"validation", toFixed(minVal, 6), toFixed(maxVal, 6), toFixed(endVal, 6)});
    }
    if (!lossSummary.empty()) {
        report.addTable("Neural Loss Summary", {"Series", "Min", "Max", "Final"}, lossSummary);
    }
}

NeuralAnalysis runNeuralAnalysis(const TypedDataset& data,
                                 int targetIdx,
                                 const std::vector<int>& featureIdx,
                                 bool binaryTarget,
                                 bool verbose,
                                 const AutoConfig& config,
                                 bool fastModeEnabled,
                                 size_t fastSampleRows) {
    NeuralAnalysis analysis;
    analysis.numericInputNodes = featureIdx.size();
    analysis.outputNodes = 1;
    analysis.binaryTarget = binaryTarget;
    analysis.hiddenActivation = "n/a";
    analysis.outputActivation = activationToString(binaryTarget ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR);
    analysis.trainingRowsTotal = data.rowCount();

    if (featureIdx.empty()) {
        analysis.policyUsed = "none_no_features";
        return analysis;
    }
    if (data.rowCount() < 2) {
        analysis.policyUsed = "none_too_few_rows";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }

    const auto& y = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    EncodedNeuralMatrix encoded = buildEncodedNeuralInputs(data, targetIdx, featureIdx, config);
    std::vector<std::vector<double>> Xnn = std::move(encoded.X);
    std::vector<std::vector<double>> Ynn(data.rowCount(), std::vector<double>(1, 0.0));

    for (size_t r = 0; r < data.rowCount(); ++r) {
        Ynn[r][0] = y[r];
    }

    size_t inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
    analysis.inputNodes = inputNodes;
    analysis.categoricalColumnsUsed = encoded.categoricalColumnsUsed;
    analysis.categoricalEncodedNodes = encoded.categoricalEncodedNodes;

    if (inputNodes == 0) {
        analysis.policyUsed = "none_no_encoded_features";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }

    size_t outputNodes = 1;
    size_t baseHidden = std::clamp<size_t>(
        static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes) * std::max<size_t>(8, data.rowCount() / 6))))),
        4,
        64);

    size_t baseBatch = 16;
    if (data.rowCount() >= 512) baseBatch = 64;
    else if (data.rowCount() >= 192) baseBatch = 32;
    else if (data.rowCount() < 64) baseBatch = 8;

    size_t baseEpochs = std::clamp<size_t>(120 + data.rowCount() / 2, 120, 320);
    int basePatience = static_cast<int>(std::clamp<size_t>(baseEpochs / 12, 8, 24));
    double baseDropout = (inputNodes >= 12) ? 0.12 : (inputNodes >= 6 ? 0.08 : 0.04);

    struct NeuralPolicy {
        std::string name;
        double hiddenMultiplier;
        double epochMultiplier;
        double dropoutDelta;
        double batchMultiplier;
        int patienceDelta;
    };
    const std::unordered_map<std::string, NeuralPolicy> registry = {
        {"fast", {"fast", 0.80, 0.65, -0.02, 1.40, -3}},
        {"balanced", {"balanced", 1.00, 1.00, 0.00, 1.00, 0}},
        {"expressive", {"expressive", 1.45, 1.45, 0.04, 0.75, 5}},
    };

    std::string requested = CommonUtils::toLower(config.neuralStrategy);
    if (requested == "none") {
        analysis.hiddenNodes = 0;
        analysis.policyUsed = "none";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }
    std::string policyName = requested;
    if (requested == "auto") {
        double complexity = std::log1p(static_cast<double>(data.rowCount())) * std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes)));
        if (complexity < 10.0) policyName = "fast";
        else if (complexity > 22.0) policyName = "expressive";
        else policyName = "balanced";
    }
    if (registry.find(policyName) == registry.end()) policyName = "balanced";
    if (fastModeEnabled) policyName = "fast";
    const NeuralPolicy& policy = registry.at(policyName);

    if (fastModeEnabled && Xnn.size() > fastSampleRows) {
        std::vector<size_t> order(Xnn.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(config.neuralSeed ^ 0x9e3779b9U);
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<std::vector<double>> sampledX;
        std::vector<std::vector<double>> sampledY;
        sampledX.reserve(fastSampleRows);
        sampledY.reserve(fastSampleRows);
        for (size_t i = 0; i < fastSampleRows; ++i) {
            sampledX.push_back(std::move(Xnn[order[i]]));
            sampledY.push_back(std::move(Ynn[order[i]]));
        }
        Xnn = std::move(sampledX);
        Ynn = std::move(sampledY);
    }

    size_t hidden = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseHidden) * policy.hiddenMultiplier)), 4, 96);
    size_t dynamicEpochs = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseEpochs) * policy.epochMultiplier)), 80, 420);
    size_t dynamicBatch = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseBatch) * policy.batchMultiplier)), 8, 128);
    int dynamicPatience = std::clamp(basePatience + policy.patienceDelta, 6, 40);
    double dynamicDropout = std::clamp(baseDropout + policy.dropoutDelta, 0.0, 0.35);

    if (Xnn.size() < 80) {
        hidden = std::min<size_t>(hidden, 24);
        dynamicEpochs = std::min<size_t>(dynamicEpochs, 140);
        dynamicPatience = std::min(dynamicPatience, 8);
        dynamicDropout = std::max(dynamicDropout, 0.10);
    }

    NeuralNet nn({inputNodes, hidden, outputNodes});
    NeuralNet::Hyperparameters hp;

    auto toOptimizer = [](const std::string& name) {
        const std::string n = CommonUtils::toLower(name);
        if (n == "sgd") return NeuralNet::Optimizer::SGD;
        if (n == "adam") return NeuralNet::Optimizer::ADAM;
        return NeuralNet::Optimizer::LOOKAHEAD;
    };

    hp.epochs = dynamicEpochs;
    hp.batchSize = dynamicBatch;
    hp.earlyStoppingPatience = dynamicPatience;
    hp.lrDecay = config.neuralLrDecay;
    hp.lrPlateauPatience = config.neuralLrPlateauPatience;
    hp.lrCooldownEpochs = config.neuralLrCooldownEpochs;
    hp.maxLrReductions = config.neuralMaxLrReductions;
    hp.minLearningRate = config.neuralMinLearningRate;
    hp.useValidationLossEma = config.neuralUseValidationLossEma;
    hp.validationLossEmaBeta = config.neuralValidationLossEmaBeta;
    hp.dropoutRate = dynamicDropout;
    hp.valSplit = (Xnn.size() < 80) ? 0.30 : 0.20;
    hp.l2Lambda = (Xnn.size() < 80) ? 0.010 : 0.001;
    hp.categoricalInputL2Boost = config.neuralCategoricalInputL2Boost;
    hp.activation = NeuralNet::Activation::GELU;
    hp.outputActivation = binaryTarget ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
    hp.useBatchNorm = config.neuralUseBatchNorm;
    hp.batchNormMomentum = config.neuralBatchNormMomentum;
    hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
    hp.useLayerNorm = config.neuralUseLayerNorm;
    hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
    hp.optimizer = toOptimizer(config.neuralOptimizer);
    hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
    hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
    hp.lookaheadAlpha = config.neuralLookaheadAlpha;
    hp.loss = binaryTarget ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
    hp.verbose = verbose;
    hp.seed = config.neuralSeed;
    hp.gradientClipNorm = config.gradientClipNorm;

    std::vector<double> inputL2Scales(inputNodes, 1.0);
    for (size_t i = 0; i < inputL2Scales.size() && i < encoded.sourceNumericFeaturePos.size(); ++i) {
        if (encoded.sourceNumericFeaturePos[i] < 0) {
            inputL2Scales[i] = hp.categoricalInputL2Boost;
        }
    }
    nn.setInputL2Scales(inputL2Scales);

    nn.train(Xnn, Ynn, hp);

    analysis.inputNodes = inputNodes;
    analysis.hiddenNodes = hidden;
    analysis.outputNodes = outputNodes;
    analysis.binaryTarget = binaryTarget;
    analysis.hiddenActivation = activationToString(hp.activation);
    analysis.outputActivation = activationToString(hp.outputActivation);
    analysis.epochs = hp.epochs;
    analysis.batchSize = hp.batchSize;
    analysis.valSplit = hp.valSplit;
    analysis.l2Lambda = hp.l2Lambda;
    analysis.dropoutRate = hp.dropoutRate;
    analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
    analysis.policyUsed = policy.name;
    analysis.trainingRowsUsed = Xnn.size();
    analysis.trainLoss = nn.getTrainLossHistory();
    analysis.valLoss = nn.getValLossHistory();
    const std::vector<double> rawImportance = nn.calculateFeatureImportance(Xnn, Ynn, inputNodes > 10 ? 7 : 5);
    analysis.featureImportance.assign(featureIdx.size(), 0.0);

    double totalImportance = 0.0;
    double categoricalImportance = 0.0;
    for (size_t i = 0; i < rawImportance.size(); ++i) {
        const double imp = std::max(0.0, std::isfinite(rawImportance[i]) ? rawImportance[i] : 0.0);
        totalImportance += imp;

        if (i >= encoded.sourceNumericFeaturePos.size()) {
            categoricalImportance += imp;
            continue;
        }

        const int numericPos = encoded.sourceNumericFeaturePos[i];
        if (numericPos >= 0) {
            const size_t numericPosU = static_cast<size_t>(numericPos);
            if (numericPosU < analysis.featureImportance.size()) {
                analysis.featureImportance[numericPosU] += imp;
            }
        } else {
            categoricalImportance += imp;
        }
    }
    analysis.categoricalImportanceShare = (totalImportance > config.tuning.numericEpsilon)
        ? (categoricalImportance / totalImportance)
        : 0.0;

    return analysis;
}

BivariateScoringPolicy chooseBivariatePolicy(const AutoConfig& config, const NeuralAnalysis& neural) {
    const std::unordered_map<std::string, BivariateScoringPolicy> registry = {
        {"balanced", {"balanced", 0.50, 0.35, 0.15, 0.60, 1.0, 0.60, 0.25}},
        {"corr_heavy", {"corr_heavy", 0.30, 0.55, 0.15, 0.55, 1.0, 0.55, 0.20}},
        {"importance_heavy", {"importance_heavy", 0.65, 0.20, 0.15, 0.65, 1.0, 0.70, 0.35}},
    };

    std::string mode = CommonUtils::toLower(config.bivariateStrategy);
    auto applySelectionQuantileOverride = [&](BivariateScoringPolicy policy) {
        if (config.tuning.bivariateSelectionQuantileOverride >= 0.0 &&
            config.tuning.bivariateSelectionQuantileOverride <= 1.0) {
            policy.selectionQuantile = config.tuning.bivariateSelectionQuantileOverride;
        }
        return policy;
    };
    if (mode != "auto") {
        auto it = registry.find(mode);
        if (it != registry.end()) return applySelectionQuantileOverride(it->second);
        return applySelectionQuantileOverride(registry.at("balanced"));
    }

    double maxImp = 0.0;
    double sumImpSq = 0.0;
    for (double v : neural.featureImportance) {
        maxImp = std::max(maxImp, v);
        sumImpSq += v * v;
    }
    double concentration = std::sqrt(sumImpSq);

    if (maxImp > config.tuning.corrHeavyMaxImportanceThreshold ||
        concentration > config.tuning.corrHeavyConcentrationThreshold) {
        return applySelectionQuantileOverride(registry.at("corr_heavy"));
    }
    if (maxImp < config.tuning.importanceHeavyMaxImportanceThreshold &&
        concentration < config.tuning.importanceHeavyConcentrationThreshold) {
        return applySelectionQuantileOverride(registry.at("importance_heavy"));
    }
    return applySelectionQuantileOverride(registry.at("balanced"));
}

std::vector<PairInsight> analyzeBivariatePairs(const TypedDataset& data,
                                               const std::unordered_map<size_t, double>& importanceByIndex,
                                               const std::unordered_set<size_t>& modeledIndices,
                                               const BivariateScoringPolicy& policy,
                                               GnuplotEngine* plotter,
                                               bool verbose,
                                               const NumericStatsCache& statsCache,
                                               double numericEpsilon,
                                               size_t maxPairs) {
    std::vector<PairInsight> pairs;
    const auto numericIdx = data.numericColumnIndices();
    const size_t n = data.rowCount();
    std::vector<size_t> evalNumericIdx = numericIdx;

    const size_t totalPairs = numericIdx.size() < 2 ? 0 : (numericIdx.size() * (numericIdx.size() - 1)) / 2;
    if (maxPairs > 0 && totalPairs > maxPairs) {
        std::vector<std::pair<size_t, double>> ranked;
        ranked.reserve(numericIdx.size());
        for (size_t idx : numericIdx) {
            double imp = 0.0;
            auto it = importanceByIndex.find(idx);
            if (it != importanceByIndex.end()) imp = it->second;
            double modeledBoost = (modeledIndices.find(idx) != modeledIndices.end()) ? 0.25 : 0.0;
            ranked.push_back({idx, imp + modeledBoost});
        }
        std::sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
            if (a.second == b.second) return data.columns()[a.first].name < data.columns()[b.first].name;
            return a.second > b.second;
        });

        const size_t keepColumns = std::clamp<size_t>(
            static_cast<size_t>(std::floor((1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(maxPairs))) / 2.0)),
            2,
            ranked.size());
        evalNumericIdx.clear();
        evalNumericIdx.reserve(keepColumns);
        for (size_t i = 0; i < keepColumns; ++i) {
            evalNumericIdx.push_back(ranked[i].first);
        }
    }

    if (verbose) {
        size_t activePairs = evalNumericIdx.size() < 2 ? 0 : (evalNumericIdx.size() * (evalNumericIdx.size() - 1)) / 2;
        std::cout << "[Seldon][Bivariate] Evaluating " << activePairs << " pair combinations with policy '" << policy.name << "'";
        if (activePairs < totalPairs) {
            std::cout << " (fast cap from " << totalPairs << ")";
        }
        std::cout << "...\n";
    }

    auto buildPair = [&](size_t aPos, size_t bPos) {
            size_t ia = evalNumericIdx[aPos];
            size_t ib = evalNumericIdx[bPos];
            const auto& va = std::get<std::vector<double>>(data.columns()[ia].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[ib].values);

            PairInsight p;
            p.idxA = ia;
            p.idxB = ib;
            p.featureA = data.columns()[ia].name;
            p.featureB = data.columns()[ib].name;
            auto aIt = statsCache.find(ia);
            auto bIt = statsCache.find(ib);
            ColumnStats statsAFallback = Statistics::calculateStats(va);
            ColumnStats statsBFallback = Statistics::calculateStats(vb);
            const ColumnStats& statsA = (aIt != statsCache.end()) ? aIt->second : statsAFallback;
            const ColumnStats& statsB = (bIt != statsCache.end()) ? bIt->second : statsBFallback;
            p.r = MathUtils::calculatePearson(va, vb, statsA, statsB).value_or(0.0);
            p.r2 = p.r * p.r;
            // Keep regression parameter derivation centralized in MathUtils.
            auto fit = MathUtils::simpleLinearRegression(va, vb, statsA, statsB, p.r);
            p.slope = fit.first;
            p.intercept = fit.second;

            auto sig = MathUtils::calculateSignificance(p.r, n);
            p.pValue = sig.p_value;
            p.tStat = sig.t_stat;
            p.statSignificant = sig.is_significant;

            double impA = 0.0;
            double impB = 0.0;
            auto ita = importanceByIndex.find(ia);
            auto itb = importanceByIndex.find(ib);
            if (ita != importanceByIndex.end()) impA = ita->second;
            if (itb != importanceByIndex.end()) impB = itb->second;
            double impScore = std::clamp((impA + impB) / 2.0, 0.0, 1.0);
            double corrScore = std::clamp(std::abs(p.r), 0.0, 1.0);
            double sigScore = std::clamp(-std::log10(std::max(p.pValue, numericEpsilon)) / 12.0, 0.0, 1.0);
            bool aModeled = modeledIndices.find(ia) != modeledIndices.end();
            bool bModeled = modeledIndices.find(ib) != modeledIndices.end();
            double coverageFactor = (aModeled && bModeled) ? policy.coverageBoth : ((aModeled || bModeled) ? policy.coverageOne : policy.coverageNone);
            p.neuralScore = (policy.wImportance * impScore + policy.wCorrelation * corrScore + policy.wSignificance * sigScore) * coverageFactor;

            return p;
    };

    auto appendPairsForRange = [&](size_t startAPos, size_t endAPos, std::vector<PairInsight>& out, bool emitVerboseRows) {
        for (size_t aPos = startAPos; aPos < endAPos; ++aPos) {
            for (size_t bPos = aPos + 1; bPos < evalNumericIdx.size(); ++bPos) {
                PairInsight p = buildPair(aPos, bPos);
                if (emitVerboseRows) {
                    std::cout << "[Seldon][Bivariate] " << p.featureA << " vs " << p.featureB
                              << " | r=" << toFixed(p.r, 6)
                              << " p=" << toFixed(p.pValue, 8)
                              << " t=" << toFixed(p.tStat, 6)
                              << " neural=" << toFixed(p.neuralScore, 8) << "\n";
                }
                out.push_back(std::move(p));
            }
        }
    };

    if (verbose) {
        appendPairsForRange(0, evalNumericIdx.size(), pairs, true);
    } else {
        #ifdef USE_OPENMP
        const int threadCount = std::max(1, omp_get_max_threads());
        std::vector<std::vector<PairInsight>> threadPairs(static_cast<size_t>(threadCount));

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<PairInsight>& localPairs = threadPairs[static_cast<size_t>(tid)];

            #pragma omp for schedule(dynamic)
            for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
                appendPairsForRange(aPos, aPos + 1, localPairs, false);
            }
        }

        for (auto& local : threadPairs) {
            pairs.insert(pairs.end(), local.begin(), local.end());
        }
        #else
        appendPairsForRange(0, evalNumericIdx.size(), pairs, false);
        #endif
    }

    std::vector<double> significantScores;
    for (const auto& p : pairs) {
        if (p.statSignificant) significantScores.push_back(p.neuralScore);
    }

    // Dynamic cutoff couples statistical significance with neural relevance.
    double dynamicCutoff = 0.0;
    if (!significantScores.empty()) {
        std::sort(significantScores.begin(), significantScores.end());
        size_t pos = static_cast<size_t>(std::floor(policy.selectionQuantile * static_cast<double>(significantScores.size() - 1)));
        dynamicCutoff = significantScores[pos];
    }

    for (auto& p : pairs) {
        p.selected = p.statSignificant && (p.neuralScore >= dynamicCutoff);
        if (p.selected && plotter) {
            const auto& va = std::get<std::vector<double>>(data.columns()[p.idxA].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[p.idxB].values);
            std::string id = "sig_" + p.featureA + "_" + p.featureB;
            p.plotPath = plotter->scatter(id, va, vb, p.featureA + " vs " + p.featureB);
        }
        if (verbose) {
            std::cout << "[Seldon][Bivariate] Decision " << p.featureA << " vs " << p.featureB
                      << " => " << (p.selected ? "SELECTED" : "REJECTED")
                      << " (cutoff=" << toFixed(dynamicCutoff, 8) << ")\n";
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairInsight& lhs, const PairInsight& rhs) {
        if (lhs.selected != rhs.selected) return lhs.selected > rhs.selected;
        return std::abs(lhs.r) > std::abs(rhs.r);
    });

    return pairs;
}

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
                        const NumericStatsCache& statsCache) {
    report.addParagraph("Overall analysis aggregates dataset health, model baselines, neural training behavior, and optional global visual diagnostics.");
    report.addParagraph("Data Health summarizes how much usable signal the engine found after preprocessing, feature selection, statistical filtering, and neural convergence checks.");
    addDatasetHealthTable(report, data, prep, health);

    addBenchmarkSection(report, benchmarks);
    addNeuralLossSummaryTable(report, neural);

    if (overallPlotter && canPlotOverall) {
        std::vector<std::string> labels;
        std::vector<double> missingCounts;
        for (const auto& col : data.columns()) {
            labels.push_back(col.name);
            auto it = prep.missingCounts.find(col.name);
            missingCounts.push_back(it == prep.missingCounts.end() ? 0.0 : static_cast<double>(it->second));
        }
        std::string missingImg = overallPlotter->bar("overall_missingness", labels, missingCounts, "Overall Missingness by Column");
        if (!missingImg.empty()) report.addImage("Overall Missingness", missingImg);

        auto numericIdx = data.numericColumnIndices();
        if (numericIdx.size() >= 2) {
            const size_t maxCols = std::max<size_t>(2, config.tuning.overallCorrHeatmapMaxColumns);
            if (numericIdx.size() > maxCols) {
                std::stable_sort(numericIdx.begin(), numericIdx.end(), [&](size_t lhs, size_t rhs) {
                    auto lIt = statsCache.find(lhs);
                    auto rIt = statsCache.find(rhs);
                    const double lv = (lIt != statsCache.end() && std::isfinite(lIt->second.variance)) ? lIt->second.variance : 0.0;
                    const double rv = (rIt != statsCache.end() && std::isfinite(rIt->second.variance)) ? rIt->second.variance : 0.0;
                    return lv > rv;
                });
                numericIdx.resize(maxCols);
            }

            std::vector<std::vector<double>> corr(numericIdx.size(), std::vector<double>(numericIdx.size(), 1.0));
            for (size_t i = 0; i < numericIdx.size(); ++i) {
                for (size_t j = i + 1; j < numericIdx.size(); ++j) {
                    const auto idxI = numericIdx[i];
                    const auto idxJ = numericIdx[j];
                    const auto& a = std::get<std::vector<double>>(data.columns()[idxI].values);
                    const auto& b = std::get<std::vector<double>>(data.columns()[idxJ].values);
                    auto ai = statsCache.find(idxI);
                    auto bi = statsCache.find(idxJ);
                    ColumnStats aFallback = Statistics::calculateStats(a);
                    ColumnStats bFallback = Statistics::calculateStats(b);
                    const ColumnStats& sa = (ai != statsCache.end()) ? ai->second : aFallback;
                    const ColumnStats& sb = (bi != statsCache.end()) ? bi->second : bFallback;
                    double r = MathUtils::calculatePearson(a, b, sa, sb).value_or(0.0);
                    corr[i][j] = r;
                    corr[j][i] = r;
                }
            }
            std::string img = overallPlotter->heatmap("overall_corr_heatmap", corr, "Overall Correlation Heatmap");
            if (!img.empty()) report.addImage("Overall Correlation Heatmap", img);
        }

        if (!neural.trainLoss.empty()) {
            std::vector<double> epochs(neural.trainLoss.size(), 0.0);
            for (size_t i = 0; i < epochs.size(); ++i) epochs[i] = static_cast<double>(i + 1);
            std::string trainImg = overallPlotter->line("overall_nn_train_loss", epochs, neural.trainLoss, "NN Train Loss");
            if (!trainImg.empty()) report.addImage("Overall NN Train Loss", trainImg);

            if (!neural.valLoss.empty() && neural.valLoss.size() == neural.trainLoss.size()) {
                std::string valImg = overallPlotter->line("overall_nn_val_loss", epochs, neural.valLoss, "NN Validation Loss");
                if (!valImg.empty()) report.addImage("Overall NN Validation Loss", valImg);
            }
        }

        if (verbose) {
            std::cout << "[Seldon][Overall] Optional overall plots generated in dedicated folder.\n";
        }
    } else {
        report.addParagraph("Overall plots were skipped (supervised mode disabled or gnuplot unavailable).");
    }
}
} // namespace

int AutomationPipeline::run(const AutoConfig& config) {
    AutoConfig runCfg = config;
    runCfg.plot.format = "png";

    cleanupOutputs(runCfg);
    MathUtils::setSignificanceAlpha(runCfg.tuning.significanceAlpha);
    MathUtils::setNumericTuning(runCfg.tuning.numericEpsilon,
                                runCfg.tuning.betaFallbackIntervalsStart,
                                runCfg.tuning.betaFallbackIntervalsMax,
                                runCfg.tuning.betaFallbackTolerance);

    TypedDataset data(config.datasetPath, config.delimiter);
    data.load();
    if (data.rowCount() == 0 || data.colCount() == 0) {
        throw Seldon::DatasetException("Dataset has no usable rows/columns");
    }

    validateExcludedColumns(data, config);
    const TargetContext targetContext = resolveTargetContext(data, config, runCfg);
    const int targetIdx = targetContext.targetIdx;

    const bool autoFastMode = (data.rowCount() > 100000) || (data.numericColumnIndices().size() > 50);
    const bool fastModeEnabled = runCfg.fastMode || autoFastMode;
    if (fastModeEnabled && CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
        runCfg.neuralStrategy = "fast";
    }

    PreprocessReport prep = Preprocessor::run(data, runCfg);
    normalizeBinaryTarget(data, targetIdx, targetContext.semantics);
    const NumericStatsCache statsCache = buildNumericStatsCache(data);

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Preparing deeply detailed univariate analysis...\n";
    }

    ReportEngine univariate;
    univariate.addTitle("Univariate Analysis");
    univariate.addParagraph("Dataset: " + runCfg.datasetPath);
    univariate.addParagraph("Rows: " + std::to_string(data.rowCount()) + " | Columns: " + std::to_string(data.colCount()));
    addUnivariateDetailedSection(univariate, data, prep, runCfg.verboseAnalysis, statsCache);

    GnuplotEngine plotterBivariate(plotSubdir(runCfg, "bivariate"), runCfg.plot);
    GnuplotEngine plotterUnivariate(plotSubdir(runCfg, "univariate"), runCfg.plot);
    GnuplotEngine plotterOverall(plotSubdir(runCfg, "overall"), runCfg.plot);
    const bool canPlot = configurePlotAvailability(runCfg, univariate, plotterBivariate);
    addUnivariatePlots(univariate, data, runCfg, canPlot, plotterUnivariate);

    FeatureSelectionResult selectedFeatures = collectFeatureIndices(data, targetIdx, runCfg, prep);
    std::vector<int> featureIdx = selectedFeatures.included;

    if (runCfg.verboseAnalysis && !selectedFeatures.droppedByMissingness.empty()) {
        std::cout << "[Seldon][Features] Dropped sparse features (>"
                  << toFixed(selectedFeatures.missingThresholdUsed, 2)
                  << " missing ratio, strategy=" << selectedFeatures.strategyUsed << "):\n";
        for (const auto& item : selectedFeatures.droppedByMissingness) {
            std::cout << "  - " << item << "\n";
        }
    }

    auto benchmarks = BenchmarkEngine::run(data, targetIdx, featureIdx, runCfg.kfold, runCfg.benchmarkSeed);

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Neural] Starting neural lattice training with verbose trace...\n";
    }
    NeuralAnalysis neural = runNeuralAnalysis(data,
                                              targetIdx,
                                              featureIdx,
                                              targetContext.semantics.isBinary,
                                              runCfg.verboseAnalysis,
                                              runCfg,
                                              fastModeEnabled,
                                              runCfg.fastNeuralSampleRows);
    neural.featureImportance = buildCoherentImportance(data, targetIdx, featureIdx, neural, benchmarks, runCfg, statsCache);
    BivariateScoringPolicy bivariatePolicy = chooseBivariatePolicy(runCfg, neural);

    std::unordered_map<size_t, double> importanceByIndex;
    double fallbackImportance = 0.0;
    for (double v : neural.featureImportance) fallbackImportance += v;
    if (!neural.featureImportance.empty()) fallbackImportance /= neural.featureImportance.size();
    for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size(); ++i) {
        importanceByIndex[static_cast<size_t>(featureIdx[i])] = neural.featureImportance[i];
    }
    importanceByIndex[static_cast<size_t>(targetIdx)] = fallbackImportance;

    const auto numeric = data.numericColumnIndices();
    for (size_t idx : numeric) {
        if (importanceByIndex.find(idx) == importanceByIndex.end()) {
            importanceByIndex[idx] = 0.0;
        }
    }

    std::unordered_set<size_t> modeledIndices;
    modeledIndices.insert(static_cast<size_t>(targetIdx));
    for (int idx : featureIdx) modeledIndices.insert(static_cast<size_t>(idx));

    GnuplotEngine* bivariatePlotter = (canPlot && runCfg.plotBivariateSignificant) ? &plotterBivariate : nullptr;
    const size_t totalPossiblePairs = numeric.size() < 2
        ? 0
        : (numeric.size() * (numeric.size() - 1)) / 2;
    auto bivariatePairs = analyzeBivariatePairs(data,
                                                importanceByIndex,
                                                modeledIndices,
                                                bivariatePolicy,
                                                bivariatePlotter,
                                                runCfg.verboseAnalysis,
                                                statsCache,
                                                runCfg.tuning.numericEpsilon,
                                                fastModeEnabled ? runCfg.fastMaxBivariatePairs : 0);

    ReportEngine bivariate;
    bivariate.addTitle("Bivariate Analysis");
    if (totalPossiblePairs > bivariatePairs.size()) {
        bivariate.addParagraph("Fast mode active: pair evaluation was capped for runtime safety. Results below cover the highest-priority numeric columns only.");
    } else {
        bivariate.addParagraph("All numeric pair combinations are included below (nC2). Significant table is dynamically filtered using statistical significance + neural relevance score.");
    }
    bivariate.addParagraph("Neural-lattice significance score drives final pair selection; only selected pairs are considered significant findings.");

    std::vector<std::vector<std::string>> allRows;
    std::vector<std::vector<std::string>> sigRows;
    size_t statSigCount = 0;
    for (const auto& p : bivariatePairs) {
        if (p.statSignificant) statSigCount++;
        allRows.push_back({
            p.featureA,
            p.featureB,
            toFixed(p.r),
            toFixed(p.r2),
            toFixed(p.slope),
            toFixed(p.intercept),
            toFixed(p.tStat, 6),
            toFixed(p.pValue, 6),
            p.statSignificant ? "yes" : "no",
            toFixed(p.neuralScore, 6),
            p.selected ? "yes" : "no",
            p.plotPath.empty() ? "-" : p.plotPath
        });

        if (p.selected) {
            sigRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.r2),
                toFixed(p.slope),
                toFixed(p.intercept),
                toFixed(p.tStat, 6),
                toFixed(p.pValue, 6),
                toFixed(p.neuralScore, 6),
                p.plotPath.empty() ? "-" : p.plotPath
            });
            if (!p.plotPath.empty()) {
                bivariate.addImage("Significant Pair: " + p.featureA + " vs " + p.featureB, p.plotPath);
            }
        }
    }

    bivariate.addParagraph("Total pairs evaluated: " + std::to_string(allRows.size()));
    bivariate.addParagraph("Statistically significant pairs (p<" + toFixed(MathUtils::getSignificanceAlpha(), 4) + "): " + std::to_string(statSigCount));
    bivariate.addParagraph("Final neural-selected significant pairs: " + std::to_string(sigRows.size()));
    bivariate.addTable("All Pairwise Results", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "stat_sig", "neural_score", "selected", "plot"}, allRows);
    bivariate.addTable("Final Significant Results", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "neural_score", "plot"}, sigRows);

    const DataHealthSummary dataHealth = computeDataHealthSummary(data,
                                                                  prep,
                                                                  neural,
                                                                  featureIdx.size(),
                                                                  allRows.size(),
                                                                  statSigCount,
                                                                  sigRows.size());

    ReportEngine neuralReport;
    neuralReport.addTitle("Neural Synthesis");
    neuralReport.addParagraph("This synthesis captures detailed lattice training traces and how neural relevance influenced bivariate selection.");
    neuralReport.addParagraph(std::string("Task type inferred from target: ") + (targetContext.semantics.isBinary ? "binary_classification" : "regression"));
    neuralReport.addTable("Auto Decision Log", {"Decision", "Value"}, {
        {"Target Selection", targetContext.userProvidedTarget ? "user-specified" : "auto"},
        {"Target Strategy", targetContext.choice.strategyUsed},
        {"Target Column", runCfg.targetColumn},
        {"Feature Strategy", selectedFeatures.strategyUsed},
        {"Feature Missingness Threshold", toFixed(selectedFeatures.missingThresholdUsed, 4)},
        {"Features Retained", std::to_string(featureIdx.size())},
        {"Categorical Columns Encoded", std::to_string(neural.categoricalColumnsUsed)},
        {"Categorical One-Hot Nodes", std::to_string(neural.categoricalEncodedNodes)},
        {"Sparse Features Dropped", std::to_string(selectedFeatures.droppedByMissingness.size())},
        {"Neural Strategy", neural.policyUsed},
        {"Fast Mode", fastModeEnabled ? "enabled" : "disabled"},
        {"Neural Training Rows Used", std::to_string(neural.trainingRowsUsed) + " / " + std::to_string(neural.trainingRowsTotal)},
        {"Bivariate Strategy", bivariatePolicy.name}
    });
    if (!selectedFeatures.droppedByMissingness.empty()) {
        neuralReport.addParagraph("Sparse numeric features dropped before modeling: " + std::to_string(selectedFeatures.droppedByMissingness.size()));
    }
    neuralReport.addTable("Neural Network Auto-Defaults", {"Setting", "Value"}, {
        {"Neural Policy", neural.policyUsed},
        {"Input Nodes", std::to_string(neural.inputNodes)},
        {"Numeric Input Nodes", std::to_string(neural.numericInputNodes)},
        {"Categorical Encoded Nodes", std::to_string(neural.categoricalEncodedNodes)},
        {"Hidden Nodes", std::to_string(neural.hiddenNodes)},
        {"Output Nodes", std::to_string(neural.outputNodes)},
        {"Epochs", std::to_string(neural.epochs)},
        {"Batch Size", std::to_string(neural.batchSize)},
        {"Validation Split", toFixed(neural.valSplit, 2)},
        {"L2 Lambda", toFixed(neural.l2Lambda, 4)},
        {"Dropout", toFixed(neural.dropoutRate, 4)},
        {"Categorical Importance Share", toFixed(neural.categoricalImportanceShare, 4)},
        {"Early Stop Patience", std::to_string(neural.earlyStoppingPatience)},
        {"Loss", neural.binaryTarget ? "cross_entropy" : "mse"},
        {"Hidden Activation", neural.hiddenActivation},
        {"Output Activation", neural.outputActivation}
    });

    std::vector<std::vector<std::string>> fiRows;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        fiRows.push_back({name, toFixed(imp, 6)});
    }
    neuralReport.addTable("Feature Importance (Permutation, Neural)", {"Feature", "Importance"}, fiRows);

    std::vector<std::vector<std::string>> lossRows;
    for (size_t e = 0; e < neural.trainLoss.size(); ++e) {
        double val = (e < neural.valLoss.size() ? neural.valLoss[e] : 0.0);
        lossRows.push_back({
            std::to_string(e + 1),
            toFixed(neural.trainLoss[e], 6),
            toFixed(val, 6)
        });
    }
    neuralReport.addTable("Neural Lattice Training Trace", {"Epoch", "Train Loss", "Validation Loss"}, lossRows);

    addOverallSections(neuralReport,
                       data,
                       prep,
                       benchmarks,
                       neural,
                       dataHealth,
                       runCfg,
                       &plotterOverall,
                       canPlot && runCfg.plotOverall,
                       runCfg.verboseAnalysis,
                       statsCache);

    ReportEngine finalAnalysis;
    finalAnalysis.addTitle("Final Analysis - Significant Findings Only");
    finalAnalysis.addParagraph("This report contains only neural-net-approved significant findings (dynamic decision engine). Non-selected findings are excluded by design.");
    finalAnalysis.addParagraph("Data Health Score: " + toFixed(dataHealth.score, 1) + "/100 (" + dataHealth.band + "). This score estimates discovered signal strength using completeness, retained feature coverage, significant-pair yield, selected-pair yield, and neural training stability.");
    finalAnalysis.addTable("Neural-Selected Significant Bivariate Findings", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "neural_score", "plot"}, sigRows);

    finalAnalysis.addTable("Data Health Signal Card", {"Component", "Value"}, {
        {"Score (0-100)", toFixed(dataHealth.score, 1)},
        {"Band", dataHealth.band},
        {"Completeness", toFixed(100.0 * dataHealth.completeness, 1) + "%"},
        {"Numeric Coverage", toFixed(100.0 * dataHealth.numericCoverage, 1) + "%"},
        {"Feature Retention", toFixed(100.0 * dataHealth.featureRetention, 1) + "%"},
        {"Statistical Yield", toFixed(100.0 * dataHealth.statYield, 1) + "%"},
        {"Neural-Selected Yield", toFixed(100.0 * dataHealth.selectedYield, 1) + "%"},
        {"Training Stability", toFixed(100.0 * dataHealth.trainingStability, 1) + "%"}
    });

    std::vector<std::vector<std::string>> topFeatures;
    std::vector<std::pair<std::string, double>> fiPairs;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        fiPairs.push_back({name, imp});
    }
    std::sort(fiPairs.begin(), fiPairs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& kv : fiPairs) {
        topFeatures.push_back({kv.first, toFixed(kv.second, 6)});
    }
    finalAnalysis.addTable("Neural Feature Importance Ranking", {"Feature", "Importance"}, topFeatures);

    finalAnalysis.addTable("Executive Statistics", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Columns", std::to_string(data.colCount())},
        {"Pairs Evaluated", std::to_string(allRows.size())},
        {"Pairs Statistically Significant", std::to_string(statSigCount)},
        {"Pairs Neural-Selected", std::to_string(sigRows.size())},
        {"Training Epochs Executed", std::to_string(neural.trainLoss.size())},
        {"Data Health Score", toFixed(dataHealth.score, 1) + " (" + dataHealth.band + ")"}
    });

    saveGeneratedReports(runCfg, univariate, bivariate, neuralReport, finalAnalysis);
    printPipelineCompletion(runCfg);

    return 0;
}
