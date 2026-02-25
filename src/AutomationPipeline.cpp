#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
#include "CommonUtils.h"
#include "GnuplotEngine.h"
#include "MathUtils.h"
#include "NeuralNet.h"
#include "PlotHeuristics.h"
#include "Preprocessor.h"
#include "ReportEngine.h"
#include "SeldonExceptions.h"
#include "Statistics.h"
#include "TypedDataset.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <set>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <cstdlib>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
std::string findExecutableInPath(const std::string& command) {
    if (command.empty()) return "";
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return "";

    std::stringstream ss{std::string(pathEnv)};
    std::string token;
    while (std::getline(ss, token, ':')) {
        if (token.empty()) token = ".";
        std::filesystem::path candidate = std::filesystem::path(token) / command;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec && ::access(candidate.c_str(), X_OK) == 0) {
            return candidate.string();
        }
    }
    return "";
}

int spawnProcessSilenced(const std::string& executable, const std::vector<std::string>& args) {
    pid_t pid = ::fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        const int devNull = ::open("/dev/null", O_WRONLY);
        if (devNull >= 0) {
            ::dup2(devNull, STDOUT_FILENO);
            ::dup2(devNull, STDERR_FILENO);
            ::close(devNull);
        }

        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execv(executable.c_str(), argv.data());
        _exit(127);
    }

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::string toFixed(double v, int prec = 4) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

void printProgressBar(const std::string& label, size_t current, size_t total) {
    const size_t width = 24;
    const double ratio = (total == 0) ? 0.0 : std::clamp(static_cast<double>(current) / static_cast<double>(total), 0.0, 1.0);
    const size_t fill = static_cast<size_t>(std::floor(ratio * static_cast<double>(width)));
    std::cout << "\r[Seldon] " << label << " [";
    for (size_t i = 0; i < width; ++i) {
        if (i < fill) std::cout << '=';
        else if (i == fill && fill < width) std::cout << '>';
        else std::cout << ' ';
    }
    std::cout << "] " << static_cast<int>(ratio * 100.0) << "%" << std::flush;
    if (current >= total) {
        std::cout << "\n";
    }
}

bool containsToken(const std::string& text, const std::vector<std::string>& tokens) {
    const std::string lower = CommonUtils::toLower(text);
    for (const auto& token : tokens) {
        if (lower.find(token) != std::string::npos) return true;
    }
    return false;
}

bool shouldAddOgive(const std::vector<double>& values, const HeuristicTuningConfig& tuning) {
    return PlotHeuristics::shouldAddOgive(values, tuning);
}

bool shouldAddBoxPlot(const std::vector<double>& values, const HeuristicTuningConfig& tuning, double eps) {
    return PlotHeuristics::shouldAddBoxPlot(values, tuning, eps);
}

bool shouldAddConfidenceBand(double r,
                            bool statSignificant,
                            size_t sampleSize,
                            const HeuristicTuningConfig& tuning) {
    return PlotHeuristics::shouldAddConfidenceBand(r, statSignificant, sampleSize, tuning);
}

bool shouldAddResidualPlot(double r,
                           bool selected,
                           size_t sampleSize,
                           const HeuristicTuningConfig& tuning) {
    return PlotHeuristics::shouldAddResidualPlot(r, selected, sampleSize, tuning);
}

std::vector<size_t> clusteredOrderFromCorrelation(const std::vector<std::vector<double>>& corr) {
    const size_t n = corr.size();
    if (n <= 2) {
        std::vector<size_t> trivial(n);
        std::iota(trivial.begin(), trivial.end(), 0);
        return trivial;
    }

    std::vector<std::vector<size_t>> clusters;
    clusters.reserve(n);
    for (size_t i = 0; i < n; ++i) clusters.push_back({i});

    auto avgDistance = [&](const std::vector<size_t>& a, const std::vector<size_t>& b) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t ia : a) {
            for (size_t ib : b) {
                double c = 0.0;
                if (ia < corr.size() && ib < corr[ia].size() && std::isfinite(corr[ia][ib])) c = corr[ia][ib];
                sum += (1.0 - std::abs(c));
                ++count;
            }
        }
        return count == 0 ? 1.0 : (sum / static_cast<double>(count));
    };

    while (clusters.size() > 1) {
        size_t bestA = 0;
        size_t bestB = 1;
        double bestD = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                const double d = avgDistance(clusters[i], clusters[j]);
                if (d < bestD) {
                    bestD = d;
                    bestA = i;
                    bestB = j;
                }
            }
        }

        std::vector<size_t> merged = clusters[bestA];
        merged.insert(merged.end(), clusters[bestB].begin(), clusters[bestB].end());
        if (bestA > bestB) std::swap(bestA, bestB);
        clusters.erase(clusters.begin() + static_cast<long>(bestB));
        clusters.erase(clusters.begin() + static_cast<long>(bestA));
        clusters.push_back(std::move(merged));
    }

    return clusters.front();
}

std::optional<size_t> chooseFacetingColumn(const TypedDataset& data,
                                           size_t idxA,
                                           size_t idxB,
                                           const HeuristicTuningConfig& tuning) {
    const auto categorical = data.categoricalColumnIndices();
    if (categorical.empty()) return std::nullopt;

    const auto& x = std::get<std::vector<double>>(data.columns()[idxA].values);
    const auto& y = std::get<std::vector<double>>(data.columns()[idxB].values);
    const size_t n = std::min(x.size(), y.size());
    if (n < tuning.facetMinRows) return std::nullopt;

    double bestScore = 0.0;
    std::optional<size_t> best;
    auto avg = [](const std::vector<double>& vals) {
        if (vals.empty()) return 0.0;
        double s = 0.0;
        for (double v : vals) s += v;
        return s / static_cast<double>(vals.size());
    };

    for (size_t cidx : categorical) {
        const auto& cat = std::get<std::vector<std::string>>(data.columns()[cidx].values);
        const size_t rows = std::min(n, cat.size());
        std::unordered_map<std::string, std::vector<double>> groupMean;
        size_t usable = 0;
        for (size_t i = 0; i < rows; ++i) {
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
            if (data.columns()[cidx].missing[i]) continue;
            const std::string key = cat[i];
            if (key.empty()) continue;
            groupMean[key].push_back(y[i]);
            ++usable;
        }
        if (usable < tuning.facetMinRows) continue;
        if (groupMean.size() < 2 || groupMean.size() > tuning.facetMaxCategories * 2) continue;

        std::vector<size_t> counts;
        counts.reserve(groupMean.size());
        double weightedMean = 0.0;
        size_t total = 0;
        for (const auto& kv : groupMean) {
            if (kv.second.empty()) continue;
            counts.push_back(kv.second.size());
            weightedMean += avg(kv.second) * static_cast<double>(kv.second.size());
            total += kv.second.size();
        }
        if (counts.size() < 2 || total == 0) continue;
        weightedMean /= static_cast<double>(total);

        size_t minCount = *std::min_element(counts.begin(), counts.end());
        const double minShare = static_cast<double>(minCount) / static_cast<double>(total);
        if (minShare < tuning.facetMinCategoryShare) continue;

        double between = 0.0;
        for (const auto& kv : groupMean) {
            if (kv.second.empty()) continue;
            const double mu = avg(kv.second);
            const double d = mu - weightedMean;
            between += static_cast<double>(kv.second.size()) * d * d;
        }
        const double score = between / static_cast<double>(total);
        if (score > bestScore) {
            bestScore = score;
            best = cidx;
        }
    }

    return best;
}

bool shouldOverlayFittedLine(double r,
                             bool statSignificant,
                             const std::vector<double>& x,
                             const std::vector<double>& y,
                             double slope,
                             double intercept,
                             const HeuristicTuningConfig& tuning) {
    return PlotHeuristics::shouldOverlayFittedLine(r, statSignificant, x, y, slope, intercept, tuning);
}

struct BivariateStackedBarData {
    std::vector<std::string> categories;
    std::vector<double> lowCounts;
    std::vector<double> highCounts;
    bool valid = false;
};

struct CategoryFrequencyProfile {
    std::vector<std::string> labels;
    std::vector<double> counts;
};

CategoryFrequencyProfile buildCategoryFrequencyProfile(const std::vector<std::string>& values,
                                                      const MissingMask& missing,
                                                      size_t maxCategories = 12) {
    CategoryFrequencyProfile out;
    std::unordered_map<std::string_view, double> freq;
    const size_t n = std::min(values.size(), missing.size());
    for (size_t i = 0; i < n; ++i) {
        if (missing[i]) continue;
        if (values[i].empty()) continue;
        freq[values[i]] += 1.0;
    }
    if (freq.size() < 2) return out;

    std::vector<std::pair<std::string_view, double>> rows(freq.begin(), freq.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    const size_t keep = rows.size() > maxCategories ? maxCategories - 1 : rows.size();
    double other = 0.0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i < keep) {
            out.labels.push_back(std::string(rows[i].first));
            out.counts.push_back(rows[i].second);
        } else {
            other += rows[i].second;
        }
    }
    if (other > 0.0) {
        out.labels.push_back("Other");
        out.counts.push_back(other);
    }
    return out;
}

BivariateStackedBarData buildBivariateStackedBar(const std::vector<double>& x,
                                                 const std::vector<double>& y) {
    BivariateStackedBarData out;

    const size_t n = std::min(x.size(), y.size());
    if (n < 24) return out;

    std::vector<std::pair<double, double>> rows;
    rows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        rows.push_back({x[i], y[i]});
    }
    if (rows.size() < 24) return out;

    std::vector<double> xSorted;
    std::vector<double> ySorted;
    xSorted.reserve(rows.size());
    ySorted.reserve(rows.size());
    for (const auto& row : rows) {
        xSorted.push_back(row.first);
        ySorted.push_back(row.second);
    }
    std::sort(xSorted.begin(), xSorted.end());
    std::sort(ySorted.begin(), ySorted.end());

    const size_t binCount = std::clamp<size_t>((rows.size() >= 200) ? 5 : (rows.size() >= 80 ? 4 : 3), 3, 5);
    std::vector<double> edges;
    edges.reserve(binCount - 1);
    for (size_t q = 1; q < binCount; ++q) {
        size_t pos = static_cast<size_t>(std::floor((static_cast<double>(q) * static_cast<double>(xSorted.size() - 1)) / static_cast<double>(binCount)));
        edges.push_back(xSorted[pos]);
    }

    const double yMedian = ySorted[ySorted.size() / 2];
    out.categories.reserve(binCount);
    for (size_t i = 0; i < binCount; ++i) {
        out.categories.push_back("Q" + std::to_string(i + 1));
    }
    out.lowCounts.assign(binCount, 0.0);
    out.highCounts.assign(binCount, 0.0);

    for (const auto& row : rows) {
        const double xv = row.first;
        const double yv = row.second;
        size_t bin = static_cast<size_t>(std::distance(edges.begin(), std::upper_bound(edges.begin(), edges.end(), xv)));
        if (bin >= binCount) bin = binCount - 1;
        if (yv <= yMedian) {
            out.lowCounts[bin] += 1.0;
        } else {
            out.highCounts[bin] += 1.0;
        }
    }

    size_t nonEmptyBins = 0;
    double lowTotal = 0.0;
    double highTotal = 0.0;
    for (size_t i = 0; i < binCount; ++i) {
        const double total = out.lowCounts[i] + out.highCounts[i];
        if (total > 0.0) ++nonEmptyBins;
        lowTotal += out.lowCounts[i];
        highTotal += out.highCounts[i];
    }

    if (nonEmptyBins < 2 || lowTotal <= 0.0 || highTotal <= 0.0) return out;

    const double dominance = std::max(lowTotal, highTotal) / std::max(1.0, lowTotal + highTotal);
    if (dominance > 0.97) return out;

    out.valid = true;
    return out;
}

struct ProjectTimeline {
    std::vector<std::string> taskNames;
    std::vector<int64_t> start;
    std::vector<int64_t> end;
    std::vector<std::string> semantics;
};

class CliProgressSpinner {
public:
    explicit CliProgressSpinner(bool enabled)
        : enabled_(enabled) {}

    void update(const std::string& label, size_t step, size_t total) {
        if (!enabled_) return;
        static const char frames[] = {'|', '/', '-', '\\'};
        const char frame = frames[tick_++ % 4];
        const size_t pct = (total > 0) ? static_cast<size_t>((100.0 * static_cast<double>(step)) / static_cast<double>(total)) : 0;
        std::cout << "\r[Seldon] " << frame << " [" << step << "/" << total << "] " << pct << "% " << label << std::flush;
    }

    void done(const std::string& label) {
        if (!enabled_) return;
        std::cout << "\r[Seldon] ✓ " << label << "                                \n";
    }

private:
    bool enabled_ = false;
    size_t tick_ = 0;
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

std::optional<size_t> findDatetimeBySemanticName(const TypedDataset& data, const std::vector<std::string>& hints) {
    for (size_t idx : data.datetimeColumnIndices()) {
        if (containsToken(data.columns()[idx].name, hints)) return idx;
    }
    return std::nullopt;
}

std::optional<size_t> findNumericBySemanticName(const TypedDataset& data, const std::vector<std::string>& hints) {
    for (size_t idx : data.numericColumnIndices()) {
        if (containsToken(data.columns()[idx].name, hints)) return idx;
    }
    return std::nullopt;
}

std::optional<size_t> findCategoricalBySemanticName(const TypedDataset& data, const std::vector<std::string>& hints) {
    for (size_t idx : data.categoricalColumnIndices()) {
        if (containsToken(data.columns()[idx].name, hints)) return idx;
    }
    return std::nullopt;
}

std::optional<size_t> findTaskLabelColumn(const TypedDataset& data) {
    const auto categorical = data.categoricalColumnIndices();
    for (size_t idx : categorical) {
        if (containsToken(data.columns()[idx].name, {"task", "activity", "milestone", "story", "work", "phase", "item"})) {
            return idx;
        }
    }
    if (!categorical.empty()) return categorical.front();
    return std::nullopt;
}

std::optional<ProjectTimeline> detectProjectTimeline(const TypedDataset& data, const HeuristicTuningConfig& tuning) {
    if (!tuning.ganttAutoEnabled) return std::nullopt;

    const auto taskIdx = findTaskLabelColumn(data);
    const auto startIdx = findDatetimeBySemanticName(data, {"start", "begin", "kickoff", "open"});
    const auto endIdx = findDatetimeBySemanticName(data, {"end", "finish", "due", "complete", "deadline", "close"});
    const auto durationIdx = findNumericBySemanticName(data, {"duration", "days", "day", "hours", "hour", "effort"});
    const auto statusIdx = findCategoricalBySemanticName(data, {"status", "state", "phase"});
    const auto priorityIdx = findCategoricalBySemanticName(data, {"priority", "severity", "critical"});

    if (!taskIdx || !startIdx) return std::nullopt;

    ProjectTimeline out;
    const auto& taskNames = std::get<std::vector<std::string>>(data.columns()[*taskIdx].values);
    const auto& starts = std::get<std::vector<int64_t>>(data.columns()[*startIdx].values);

    const std::vector<int64_t>* ends = nullptr;
    if (endIdx) {
        ends = &std::get<std::vector<int64_t>>(data.columns()[*endIdx].values);
    }

    const std::vector<double>* durations = nullptr;
    if (!ends && durationIdx) {
        durations = &std::get<std::vector<double>>(data.columns()[*durationIdx].values);
    }

    const std::vector<std::string>* statusValues = nullptr;
    const std::vector<std::string>* priorityValues = nullptr;
    if (statusIdx) statusValues = &std::get<std::vector<std::string>>(data.columns()[*statusIdx].values);
    if (priorityIdx) priorityValues = &std::get<std::vector<std::string>>(data.columns()[*priorityIdx].values);

    const size_t n = std::min(taskNames.size(), starts.size());
    for (size_t i = 0; i < n; ++i) {
        if (data.columns()[*taskIdx].missing[i] || data.columns()[*startIdx].missing[i]) continue;
        int64_t start = starts[i];
        int64_t end = start;
        bool valid = false;

        if (ends && i < ends->size() && !data.columns()[*endIdx].missing[i]) {
            end = (*ends)[i];
            valid = end > start;
        } else if (durations && i < durations->size() && !data.columns()[*durationIdx].missing[i]) {
            double dur = (*durations)[i];
            if (std::isfinite(dur) && dur > 0.0) {
                const bool looksLikeHours = dur <= tuning.ganttDurationHoursThreshold;
                const int64_t delta = static_cast<int64_t>(std::llround(dur * (looksLikeHours ? 3600.0 : 86400.0)));
                end = start + std::max<int64_t>(delta, 3600);
                valid = end > start;
            }
        }

        if (!valid) continue;
        const std::string label = CommonUtils::trim(taskNames[i]);
        out.taskNames.push_back(label.empty() ? ("Task " + std::to_string(i + 1)) : label);
        out.start.push_back(start);
        out.end.push_back(end);
        std::string semantic;
        if (priorityValues && i < priorityValues->size() && (!priorityIdx || !data.columns()[*priorityIdx].missing[i])) {
            semantic += CommonUtils::trim((*priorityValues)[i]);
        }
        if (statusValues && i < statusValues->size() && (!statusIdx || !data.columns()[*statusIdx].missing[i])) {
            if (!semantic.empty()) semantic += " | ";
            semantic += CommonUtils::trim((*statusValues)[i]);
        }
        out.semantics.push_back(semantic);
    }

    if (out.taskNames.size() < tuning.ganttMinTasks) return std::nullopt;

    std::vector<size_t> order(out.taskNames.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (out.start[lhs] == out.start[rhs]) return out.taskNames[lhs] < out.taskNames[rhs];
        return out.start[lhs] < out.start[rhs];
    });

    ProjectTimeline sorted;
    const size_t keep = std::min<size_t>(tuning.ganttMaxTasks, order.size());
    sorted.taskNames.reserve(keep);
    sorted.start.reserve(keep);
    sorted.end.reserve(keep);
    sorted.semantics.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        const size_t idx = order[i];
        sorted.taskNames.push_back(out.taskNames[idx]);
        sorted.start.push_back(out.start[idx]);
        sorted.end.push_back(out.end[idx]);
        sorted.semantics.push_back(out.semantics[idx]);
    }

    return sorted;
}

struct TimeSeriesSignal {
    std::string timeName;
    std::string valueName;
    std::vector<double> timeX;
    std::vector<double> values;
};

std::optional<TimeSeriesSignal> detectTimeSeriesSignal(const TypedDataset& data,
                                                       const HeuristicTuningConfig& tuning) {
    const auto dtCols = data.datetimeColumnIndices();
    const auto numCols = data.numericColumnIndices();
    if (dtCols.empty() || numCols.empty()) return std::nullopt;

    for (size_t dtIdx : dtCols) {
        const auto& t = std::get<std::vector<int64_t>>(data.columns()[dtIdx].values);
        for (size_t numIdx : numCols) {
            const auto& y = std::get<std::vector<double>>(data.columns()[numIdx].values);
            const size_t n = std::min(t.size(), y.size());
            std::vector<double> tx;
            std::vector<double> vy;
            tx.reserve(n);
            vy.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (data.columns()[dtIdx].missing[i] || data.columns()[numIdx].missing[i]) continue;
                if (!std::isfinite(y[i])) continue;
                tx.push_back(static_cast<double>(t[i]));
                vy.push_back(y[i]);
            }
            if (tx.size() < tuning.timeSeriesTrendMinRows) continue;

            std::vector<size_t> order(tx.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tx[a] < tx[b]; });

            TimeSeriesSignal out;
            out.timeName = data.columns()[dtIdx].name;
            out.valueName = data.columns()[numIdx].name;
            out.timeX.reserve(order.size());
            out.values.reserve(order.size());
            for (size_t idx : order) {
                out.timeX.push_back(tx[idx]);
                out.values.push_back(vy[idx]);
            }
            return out;
        }
    }
    return std::nullopt;
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

double safeEntropyFromCounts(const std::unordered_map<long long, size_t>& counts, size_t total);

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
        std::unordered_map<long long, size_t> roundedCardinality;
        for (double v : values) {
            if (!std::isfinite(v)) continue;
            ++finiteCount;
            const double delta = v - mean;
            mean += delta / static_cast<double>(finiteCount);
            const double delta2 = v - mean;
            m2 += delta * delta2;
            if (std::abs(v) > 1e-12) nonZero++;
            const long long key = static_cast<long long>(std::llround(v * 1000.0));
            roundedCardinality[key]++;
        }
        if (finiteCount < 3) return -1e9;

        const double var = m2 / static_cast<double>(std::max<size_t>(1, finiteCount - 1));

        double missingRatio = 1.0 - (static_cast<double>(finiteCount) / static_cast<double>(std::max<size_t>(1, data.rowCount())));
        double nonZeroRatio = static_cast<double>(nonZero) / static_cast<double>(finiteCount);

        std::string lname = col.name;
        std::transform(lname.begin(), lname.end(), lname.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        double namePenalty = (lname.find("id") != std::string::npos || lname.find("index") != std::string::npos) ? 0.25 : 0.0;
        if (containsToken(lname, {"timestamp", "created", "updated", "time", "date"})) {
            namePenalty += 0.15;
        }
        const double cardRatio = static_cast<double>(roundedCardinality.size()) / static_cast<double>(std::max<size_t>(1, finiteCount));
        if (cardRatio > 0.95) namePenalty += 0.20;

        double varianceScore = std::clamp(std::log1p(std::max(0.0, var)) / 4.0, 0.0, 1.0);
        const double entropyScore = safeEntropyFromCounts(roundedCardinality, finiteCount);
        return 0.30 * (1.0 - missingRatio)
             + 0.25 * varianceScore
             + 0.15 * nonZeroRatio
             + 0.30 * entropyScore
             - namePenalty;
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
    bool isOrdinal = false;
    double lowLabel = 0.0;
    double highLabel = 1.0;
    size_t cardinality = 0;
    std::string inferredTask = "regression";
};

double safeEntropyFromCounts(const std::unordered_map<long long, size_t>& counts, size_t total) {
    if (total == 0 || counts.empty()) return 0.0;
    double h = 0.0;
    for (const auto& kv : counts) {
        if (kv.second == 0) continue;
        const double p = static_cast<double>(kv.second) / static_cast<double>(total);
        if (p <= 0.0) continue;
        h -= p * std::log2(p);
    }
    const double hmax = std::log2(static_cast<double>(counts.size()));
    if (hmax <= 1e-12) return 0.0;
    return std::clamp(h / hmax, 0.0, 1.0);
}

TargetSemantics inferTargetSemanticsRaw(const TypedDataset& ds, int targetIdx) {
    TargetSemantics out;
    if (targetIdx < 0 || ds.columns()[targetIdx].type != ColumnType::NUMERIC) return out;

    const auto& y = std::get<std::vector<double>>(ds.columns()[targetIdx].values);
    std::set<double> uniq;
    for (double v : y) {
        if (!std::isfinite(v)) continue;
        uniq.insert(v);
        if (uniq.size() > 12) break;
    }
    out.cardinality = uniq.size();
    if (uniq.empty()) return out;

    if (uniq.size() == 1) {
        out.isBinary = true;
        out.lowLabel = *uniq.begin();
        out.highLabel = *uniq.begin();
        out.inferredTask = "binary_classification";
        return out;
    }

    if (uniq.size() == 2) {
        auto it = uniq.begin();
        out.lowLabel = *it;
        ++it;
        out.highLabel = *it;
        out.isBinary = true;
        out.inferredTask = "binary_classification";
        return out;
    }

    if (uniq.size() <= 8) {
        out.isOrdinal = true;
        out.lowLabel = *uniq.begin();
        out.highLabel = *uniq.rbegin();
        out.inferredTask = "ordinal_classification";
        return out;
    }

    out.inferredTask = "regression";
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

        const std::string lname = CommonUtils::toLower(name);
        const std::string tname = CommonUtils::toLower(data.columns()[static_cast<size_t>(targetIdx)].name);
        const bool temporalLeakHint = containsToken(lname, {"future", "post", "after", "resolved", "actual", "outcome", "label"});
        const bool targetEchoHint = (lname.find(tname) != std::string::npos && lname != tname);
        const bool highRiskLeakage = (r >= std::max(0.995, leakageCorrThreshold)) || (r >= 0.98 && (temporalLeakHint || targetEchoHint));

        if (highRiskLeakage) {
            std::string reason = "auto_leak_guard abs_corr=" + toFixed(r, 4);
            if (temporalLeakHint || targetEchoHint) {
                reason += " semantic_hint";
            }
            out.droppedByMissingness.push_back(name + " (" + reason + ")");
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
    double spearman = 0.0;
    double kendallTau = 0.0;
    double tStat = 0.0;
    double pValue = 1.0;
    bool statSignificant = false;
    double neuralScore = 0.0;
    double effectSize = 0.0;
    double foldStability = 0.0;
    bool selected = false;
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
    std::string hiddenActivation;
    std::string outputActivation;
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
    const size_t maxOneHotPerColumn = std::max<size_t>(1, config.neuralMaxOneHotPerColumn);

    struct CategoryPlan {
        size_t columnIdx = 0;
        std::vector<std::pair<std::string_view, size_t>> categories;
        size_t keepCount = 0;
        bool includeOther = false;
    };

    std::vector<CategoryPlan> categoryPlans;
    const auto categoricalIdx = data.categoricalColumnIndices();
    categoryPlans.reserve(categoricalIdx.size());

    size_t reservedWidth = 0;
    for (size_t featurePos = 0; featurePos < numericFeatureIdx.size(); ++featurePos) {
        int idx = numericFeatureIdx[featurePos];
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        reservedWidth++;
    }

    for (size_t idx : categoricalIdx) {
        if (static_cast<int>(idx) == targetIdx) continue;
        const std::string& columnName = data.columns()[idx].name;
        if (excluded.find(columnName) != excluded.end()) continue;

        const auto& values = std::get<std::vector<std::string>>(data.columns()[idx].values);
        if (values.empty()) continue;

        std::unordered_map<std::string_view, size_t> freq;
        for (const auto& v : values) {
            if (!v.empty()) freq[v]++;
        }
        if (freq.empty()) continue;

        CategoryPlan plan;
        plan.columnIdx = idx;
        plan.categories.assign(freq.begin(), freq.end());
        std::sort(plan.categories.begin(), plan.categories.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });
        plan.keepCount = std::min(maxOneHotPerColumn, plan.categories.size());
        if (plan.keepCount == 0) continue;
        plan.includeOther = plan.categories.size() > plan.keepCount;
        reservedWidth += plan.keepCount + (plan.includeOther ? 1 : 0);
        categoryPlans.push_back(std::move(plan));
    }

    for (auto& row : out.X) {
        row.reserve(reservedWidth);
    }

    for (size_t featurePos = 0; featurePos < numericFeatureIdx.size(); ++featurePos) {
        int idx = numericFeatureIdx[featurePos];
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        const auto& values = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        const size_t rowLimit = std::min(out.X.size(), values.size());
        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t r = 0; r < rowLimit; ++r) {
            out.X[r].push_back(values[r]);
        }
        out.sourceNumericFeaturePos.push_back(static_cast<int>(featurePos));
    }

    for (const auto& plan : categoryPlans) {
        const size_t idx = plan.columnIdx;
        const auto& values = std::get<std::vector<std::string>>(data.columns()[idx].values);
        const size_t keepCount = plan.keepCount;
        std::unordered_set<std::string> kept;
        kept.reserve(keepCount);
        for (size_t i = 0; i < keepCount; ++i) {
            const std::string label(plan.categories[i].first);
            kept.insert(label);
            const size_t rowLimit = std::min(out.X.size(), values.size());
            #ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (size_t r = 0; r < rowLimit; ++r) {
                out.X[r].push_back(values[r] == label ? 1.0 : 0.0);
            }
            out.sourceNumericFeaturePos.push_back(-1);
            out.categoricalEncodedNodes++;
        }

        if (plan.includeOther) {
            const size_t rowLimit = std::min(out.X.size(), values.size());
            #ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (size_t r = 0; r < rowLimit; ++r) {
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
    std::error_code ec;
    if (!config.outputDir.empty()) {
        fs::remove_all(config.outputDir, ec);
        fs::create_directories(config.outputDir, ec);
    }
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
                  << " (strategy=" << context.choice.strategyUsed
                  << ", task=" << context.semantics.inferredTask
                  << ", cardinality=" << context.semantics.cardinality << ")\n";
    }

    return context;
}

void applyDynamicPlotDefaultsIfUnset(AutoConfig& runCfg, const TypedDataset& data) {
    if (runCfg.plotModesExplicit) return;

    const size_t rows = data.rowCount();
    const size_t numericCols = data.numericColumnIndices().size();
    const size_t categoricalCols = data.categoricalColumnIndices().size();

    runCfg.plotUnivariate = rows > 0 && (numericCols + categoricalCols) > 0;
    runCfg.plotBivariateSignificant = (rows >= std::max<size_t>(8, runCfg.tuning.scatterFitMinSampleSize)) && numericCols >= 2;
    runCfg.plotOverall = rows > 0 && (numericCols >= 2 || categoricalCols >= 2 || runCfg.plotBivariateSignificant);

    if (!runCfg.plotUnivariate && !runCfg.plotBivariateSignificant && !runCfg.plotOverall) {
        runCfg.plotUnivariate = true;
    }

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Plots] Auto mode (no plot aliases): univariate="
                  << (runCfg.plotUnivariate ? "on" : "off")
                  << ", bivariate=" << (runCfg.plotBivariateSignificant ? "on" : "off")
                  << ", overall=" << (runCfg.plotOverall ? "on" : "off") << "\n";
    }
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
                        GnuplotEngine& plotterUnivariate,
                        const std::unordered_set<size_t>& neuralApprovedNumericFeatures) {
    if (!(runCfg.plotUnivariate && canPlot)) {
        univariate.addParagraph("Supervised setting disabled: univariate plots skipped.");
        return;
    }

    if (neuralApprovedNumericFeatures.empty()) {
        univariate.addParagraph("No neural-approved significant numeric features were found: univariate plots skipped.");
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
        if (neuralApprovedNumericFeatures.find(idx) == neuralApprovedNumericFeatures.end()) continue;
        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        std::string img = plotterUnivariate.histogram("uni_hist_" + std::to_string(idx), vals, "Histogram: " + data.columns()[idx].name);
        if (!img.empty()) {
            univariate.addImage("Histogram: " + data.columns()[idx].name, img);
            logGeneratedPlot(img);
        }

        if (shouldAddOgive(vals, runCfg.tuning)) {
            std::string ogive = plotterUnivariate.ogive("uni_ogive_" + std::to_string(idx), vals, "Ogive: " + data.columns()[idx].name);
            if (!ogive.empty()) {
                univariate.addImage("Ogive: " + data.columns()[idx].name, ogive);
                logGeneratedPlot(ogive);
            }
        }

        if (shouldAddBoxPlot(vals, runCfg.tuning, runCfg.tuning.numericEpsilon)) {
            std::string box = plotterUnivariate.box("uni_box_" + std::to_string(idx), vals, "Box Plot: " + data.columns()[idx].name);
            if (!box.empty()) {
                univariate.addImage("Box Plot: " + data.columns()[idx].name, box);
                logGeneratedPlot(box);
            }
        }

        if (vals.size() >= 20) {
            std::vector<double> sorted = vals;
            std::sort(sorted.begin(), sorted.end());
            const double mu = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
            double sd = 0.0;
            for (double v : sorted) { const double d = v - mu; sd += d * d; }
            sd = std::sqrt(sd / static_cast<double>(std::max<size_t>(1, sorted.size() - 1)));
            if (sd <= 1e-12) sd = 1.0;

            std::vector<double> qNorm(sorted.size(), 0.0);
            std::vector<double> qExp(sorted.size(), 0.0);
            const double lambda = (mu > runCfg.tuning.numericEpsilon) ? (1.0 / mu) : 1.0;
            for (size_t i = 0; i < sorted.size(); ++i) {
                const double p = (static_cast<double>(i) + 0.5) / static_cast<double>(sorted.size());
                qNorm[i] = mu + sd * std::sqrt(2.0) * std::erf(2.0 * p - 1.0);
                qExp[i] = -std::log(std::max(runCfg.tuning.numericEpsilon, 1.0 - p)) / lambda;
            }

            std::string qqNorm = plotterUnivariate.scatter("uni_qqnorm_" + std::to_string(idx),
                                                            qNorm,
                                                            sorted,
                                                            "Q-Q (Normal): " + data.columns()[idx].name,
                                                            false,
                                                            0.0,
                                                            0.0,
                                                            "",
                                                            false,
                                                            1.96,
                                                            5000);
            if (!qqNorm.empty()) {
                univariate.addImage("Q-Q Plot (Normal): " + data.columns()[idx].name, qqNorm);
                logGeneratedPlot(qqNorm);
            }

            std::string qqExp = plotterUnivariate.scatter("uni_qqexp_" + std::to_string(idx),
                                                           qExp,
                                                           sorted,
                                                           "Q-Q (Exponential): " + data.columns()[idx].name,
                                                           false,
                                                           0.0,
                                                           0.0,
                                                           "",
                                                           false,
                                                           1.96,
                                                           5000);
            if (!qqExp.empty()) {
                univariate.addImage("Q-Q Plot (Exponential): " + data.columns()[idx].name, qqExp);
                logGeneratedPlot(qqExp);
            }
        }
    }

    univariate.addParagraph("Categorical and category-numeric univariate plots were skipped because they do not have direct per-column neural significance attribution.");
}

void saveGeneratedReports(const AutoConfig& runCfg,
                          const ReportEngine& univariate,
                          const ReportEngine& bivariate,
                          const ReportEngine& neuralReport,
                          const ReportEngine& finalAnalysis) {
    const std::string uniMd = runCfg.outputDir + "/univariate.md";
    const std::string biMd = runCfg.outputDir + "/bivariate.md";
    const std::string finalMd = runCfg.outputDir + "/final_analysis.md";

    univariate.save(uniMd);
    bivariate.save(biMd);
    neuralReport.save(runCfg.reportFile);
    finalAnalysis.save(finalMd);

    if (runCfg.generateHtml) {
        const std::string pandocExe = findExecutableInPath("pandoc");
        if (pandocExe.empty()) {
            return;
        }
        const std::vector<std::pair<std::string, std::string>> conversions = {
            {uniMd, runCfg.outputDir + "/univariate.html"},
            {biMd, runCfg.outputDir + "/bivariate.html"},
            {runCfg.reportFile, runCfg.outputDir + "/neural_synthesis.html"},
            {finalMd, runCfg.outputDir + "/final_analysis.html"}
        };
        for (const auto& [src, dst] : conversions) {
            spawnProcessSilenced(pandocExe, {src, "-o", dst, "--standalone", "--self-contained"});
        }
    }
}

void printPipelineCompletion(const AutoConfig& runCfg) {
    std::cout << "[Seldon] Automated pipeline complete.\n";
    std::cout << "[Seldon] Report directory: " << runCfg.outputDir << "\n";
    std::cout << "[Seldon] Reports: "
              << runCfg.outputDir << "/univariate.md, "
              << runCfg.outputDir << "/bivariate.md, "
              << runCfg.reportFile << ", "
              << runCfg.outputDir << "/final_analysis.md\n";
    if (runCfg.generateHtml) {
        std::cout << "[Seldon] HTML reports (self-contained): "
                  << runCfg.outputDir << "/univariate.html, "
                  << runCfg.outputDir << "/bivariate.html, "
                  << runCfg.outputDir << "/neural_synthesis.html, "
                  << runCfg.outputDir << "/final_analysis.html\n";
    }
    std::cout << "[Seldon] Plot folders: "
              << plotSubdir(runCfg, "univariate") << ", "
              << plotSubdir(runCfg, "bivariate") << ", "
              << plotSubdir(runCfg, "overall") << "\n";
}

void exportPreprocessedDatasetIfRequested(const TypedDataset& data, const AutoConfig& runCfg) {
    if (runCfg.exportPreprocessed == "none") return;

    const std::string basePath = runCfg.exportPreprocessedPath.empty()
        ? (runCfg.outputDir + "/preprocessed")
        : runCfg.exportPreprocessedPath;
    const std::string csvPath = basePath + ".csv";

    std::ofstream out(csvPath);
    if (!out) {
        throw Seldon::IOException("Unable to write preprocessed export: " + csvPath);
    }

    for (size_t c = 0; c < data.columns().size(); ++c) {
        if (c) out << ',';
        out << '"' << data.columns()[c].name << '"';
    }
    out << '\n';

    for (size_t r = 0; r < data.rowCount(); ++r) {
        for (size_t c = 0; c < data.columns().size(); ++c) {
            if (c) out << ',';
            if (data.columns()[c].missing[r]) {
                out << "";
                continue;
            }
            if (data.columns()[c].type == ColumnType::NUMERIC) {
                const auto& vals = std::get<std::vector<double>>(data.columns()[c].values);
                out << vals[r];
            } else if (data.columns()[c].type == ColumnType::DATETIME) {
                const auto& vals = std::get<std::vector<int64_t>>(data.columns()[c].values);
                out << vals[r];
            } else {
                std::string s = std::get<std::vector<std::string>>(data.columns()[c].values)[r];
                std::replace(s.begin(), s.end(), '"', '\'');
                out << '"' << s << '"';
            }
        }
        out << '\n';
    }

    if (runCfg.exportPreprocessed == "parquet") {
        const std::string parquetPath = basePath + ".parquet";
        const std::string pythonExe = findExecutableInPath("python3");
        if (!pythonExe.empty()) {
            const std::string code =
                "import pandas as pd\n"
                "import sys\n"
                "df = pd.read_csv(sys.argv[1])\n"
                "df.to_parquet(sys.argv[2], index=False)\n";
            spawnProcessSilenced(pythonExe, {"-c", code, csvPath, parquetPath});
        }
    }
}

void addUnivariateDetailedSection(ReportEngine& report,
                                  const TypedDataset& data,
                                  const PreprocessReport& prep,
                                  bool verbose,
                                  const NumericStatsCache& statsCache) {
    std::vector<std::vector<std::string>> summaryRows;
    std::vector<std::vector<std::string>> categoricalRows;
    std::unordered_map<size_t, NumericDetailedStats> summaryCache;
    summaryCache.reserve(data.numericColumnIndices().size());
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
            NumericDetailedStats st;
            if (idx >= 0) {
                const size_t key = static_cast<size_t>(idx);
                const auto sit = summaryCache.find(key);
                if (sit != summaryCache.end()) {
                    st = sit->second;
                } else {
                    st = detailedNumeric(vals, precomputed);
                    summaryCache.emplace(key, st);
                }
            } else {
                st = detailedNumeric(vals, precomputed);
            }

            summaryRows.push_back({
                col.name,
                "numeric",
                toFixed(st.mean),
                toFixed(st.geometricMean),
                toFixed(st.harmonicMean),
                toFixed(st.trimmedMean),
                toFixed(st.mode),
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
        "Column", "Type", "Mean", "GeoMean", "HarmonicMean", "TrimmedMean", "Mode", "Median", "Min", "Max", "Range", "Q1", "Q3", "IQR",
        "P05", "P95", "MAD", "Variance", "StdDev", "Skew", "Kurtosis", "CoeffVar", "Sum", "NonZero/Unique", "Missing", "Outliers"
    }, summaryRows);

    std::vector<std::vector<std::string>> qualityRows;
    for (const auto& col : data.columns()) {
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : 0;
        const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;
        const std::string type = (col.type == ColumnType::NUMERIC) ? "numeric" : (col.type == ColumnType::CATEGORICAL ? "categorical" : "datetime");
        size_t unique = 0;
        if (col.type == ColumnType::NUMERIC) {
            const auto& v = std::get<std::vector<double>>(col.values);
            std::unordered_set<long long> s;
            for (double x : v) s.insert(static_cast<long long>(std::llround(x * 1e6)));
            unique = s.size();
        } else if (col.type == ColumnType::CATEGORICAL) {
            const auto& v = std::get<std::vector<std::string>>(col.values);
            std::unordered_set<std::string> s(v.begin(), v.end());
            unique = s.size();
        } else {
            const auto& v = std::get<std::vector<int64_t>>(col.values);
            std::unordered_set<int64_t> s(v.begin(), v.end());
            unique = s.size();
        }
        qualityRows.push_back({col.name, type, std::to_string(unique), std::to_string(missing), std::to_string(outliers)});
    }
    report.addTable("Data Quality Report", {"Column", "Type", "Unique", "Missing", "Outliers"}, qualityRows);

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

    std::vector<int> targetIndices;
    targetIndices.push_back(targetIdx);
    if (config.neuralMultiOutput && config.neuralMaxAuxTargets > 0) {
        const auto numericIdx = data.numericColumnIndices();
        std::vector<std::pair<int, double>> candidates;
        candidates.reserve(numericIdx.size());
        const ColumnStats yStats = Statistics::calculateStats(y);
        for (size_t idx : numericIdx) {
            if (static_cast<int>(idx) == targetIdx) continue;
            const auto& cand = std::get<std::vector<double>>(data.columns()[idx].values);
            const ColumnStats cStats = Statistics::calculateStats(cand);
            const double corr = std::abs(MathUtils::calculatePearson(cand, y, cStats, yStats).value_or(0.0));
            if (std::isfinite(corr)) {
                candidates.push_back({static_cast<int>(idx), corr});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });
        const size_t keep = std::min(config.neuralMaxAuxTargets, candidates.size());
        for (size_t i = 0; i < keep; ++i) {
            targetIndices.push_back(candidates[i].first);
        }
    }

    EncodedNeuralMatrix encoded = buildEncodedNeuralInputs(data, targetIdx, featureIdx, config);
    std::vector<std::vector<double>> Xnn = std::move(encoded.X);
    std::vector<std::vector<double>> Ynn(data.rowCount(), std::vector<double>(targetIndices.size(), 0.0));

    for (size_t r = 0; r < data.rowCount(); ++r) {
        for (size_t t = 0; t < targetIndices.size(); ++t) {
            const int idx = targetIndices[t];
            const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
            Ynn[r][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
        }
    }

    analysis.outputAuxTargets = (targetIndices.size() > 1) ? (targetIndices.size() - 1) : 0;

    size_t inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
    analysis.inputNodes = inputNodes;
    analysis.categoricalColumnsUsed = encoded.categoricalColumnsUsed;
    analysis.categoricalEncodedNodes = encoded.categoricalEncodedNodes;

    auto applyAdaptiveSparsity = [&](size_t maxColumns) {
        if (Xnn.empty() || Xnn.front().empty()) return;
        const size_t cols = Xnn.front().size();
        if (cols <= maxColumns || maxColumns == 0) return;

        struct ColScore {
            size_t idx = 0;
            double score = 0.0;
        };

        std::vector<ColScore> scores;
        scores.reserve(cols);

        const size_t rows = std::min(Xnn.size(), Ynn.size());
        double yMean = 0.0;
        for (size_t r = 0; r < rows; ++r) yMean += Ynn[r][0];
        yMean /= std::max<size_t>(1, rows);

        double yVar = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            const double d = Ynn[r][0] - yMean;
            yVar += d * d;
        }
        yVar = std::max(yVar, 1e-12);

        for (size_t c = 0; c < cols; ++c) {
            double xMean = 0.0;
            for (size_t r = 0; r < rows; ++r) xMean += Xnn[r][c];
            xMean /= std::max<size_t>(1, rows);

            double xVar = 0.0;
            double cov = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                const double dx = Xnn[r][c] - xMean;
                const double dy = Ynn[r][0] - yMean;
                xVar += dx * dx;
                cov += dx * dy;
            }

            double score = 0.0;
            if (xVar > 1e-12) {
                score = std::abs(cov) / std::sqrt(xVar * yVar);
            }
            if (!std::isfinite(score)) score = 0.0;
            scores.push_back({c, score});
        }

        std::sort(scores.begin(), scores.end(), [](const ColScore& a, const ColScore& b) {
            if (a.score == b.score) return a.idx < b.idx;
            return a.score > b.score;
        });
        scores.resize(maxColumns);
        std::sort(scores.begin(), scores.end(), [](const ColScore& a, const ColScore& b) { return a.idx < b.idx; });

        std::vector<std::vector<double>> squeezedX(Xnn.size(), std::vector<double>{});
        for (size_t r = 0; r < Xnn.size(); ++r) {
            squeezedX[r].reserve(scores.size());
            for (const auto& sc : scores) {
                squeezedX[r].push_back(Xnn[r][sc.idx]);
            }
        }

        std::vector<int> squeezedSource;
        squeezedSource.reserve(scores.size());
        for (const auto& sc : scores) {
            if (sc.idx < encoded.sourceNumericFeaturePos.size()) {
                squeezedSource.push_back(encoded.sourceNumericFeaturePos[sc.idx]);
            } else {
                squeezedSource.push_back(-1);
            }
        }

        Xnn = std::move(squeezedX);
        encoded.sourceNumericFeaturePos = std::move(squeezedSource);
    };

    if (inputNodes > 0) {
        const double dofRatio = static_cast<double>(data.rowCount()) / static_cast<double>(std::max<size_t>(1, inputNodes));
        if (dofRatio < 10.0) {
            const size_t squeezeCap = std::clamp<size_t>(data.rowCount() / 10, 4, std::max<size_t>(4, inputNodes));
            applyAdaptiveSparsity(squeezeCap);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }
    }

    analysis.categoricalEncodedNodes = static_cast<size_t>(std::count_if(
        encoded.sourceNumericFeaturePos.begin(),
        encoded.sourceNumericFeaturePos.end(),
        [](int v) { return v < 0; }));

    if (inputNodes == 0) {
        analysis.policyUsed = "none_no_encoded_features";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }

    size_t outputNodes = std::max<size_t>(1, Ynn.empty() ? 1 : Ynn.front().size());

    if (inputNodes >= 48 && Xnn.size() >= 64) {
        if (inputNodes > 96 && !Ynn.empty() && !Ynn.front().empty()) {
            const size_t preProbeCap = std::clamp<size_t>(std::max<size_t>(64, Xnn.size() / 3), 64, 192);
            if (preProbeCap < inputNodes) {
                std::vector<std::pair<size_t, double>> ranked;
                ranked.reserve(inputNodes);

                double yMean = 0.0;
                for (const auto& row : Ynn) yMean += row[0];
                yMean /= static_cast<double>(std::max<size_t>(1, Ynn.size()));
                double yVar = 0.0;
                for (const auto& row : Ynn) {
                    const double d = row[0] - yMean;
                    yVar += d * d;
                }
                yVar = std::max(1e-12, yVar);

                for (size_t c = 0; c < inputNodes; ++c) {
                    double xMean = 0.0;
                    for (const auto& row : Xnn) xMean += row[c];
                    xMean /= static_cast<double>(std::max<size_t>(1, Xnn.size()));
                    double xVar = 0.0;
                    double cov = 0.0;
                    for (size_t r = 0; r < Xnn.size(); ++r) {
                        const double dx = Xnn[r][c] - xMean;
                        const double dy = Ynn[r][0] - yMean;
                        xVar += dx * dx;
                        cov += dx * dy;
                    }
                    const double score = (xVar > 1e-12) ? std::abs(cov) / std::sqrt(std::max(1e-12, xVar * yVar)) : 0.0;
                    ranked.push_back({c, std::isfinite(score) ? score : 0.0});
                }

                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                    if (a.second == b.second) return a.first < b.first;
                    return a.second > b.second;
                });
                ranked.resize(preProbeCap);
                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

                std::vector<std::vector<double>> reducedX(Xnn.size(), std::vector<double>{});
                for (size_t r = 0; r < Xnn.size(); ++r) {
                    reducedX[r].reserve(ranked.size());
                    for (const auto& kv : ranked) reducedX[r].push_back(Xnn[r][kv.first]);
                }

                std::vector<int> reducedSource;
                reducedSource.reserve(ranked.size());
                for (const auto& kv : ranked) {
                    const size_t idx = kv.first;
                    reducedSource.push_back((idx < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[idx] : -1);
                }

                Xnn = std::move(reducedX);
                encoded.sourceNumericFeaturePos = std::move(reducedSource);
                inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
                analysis.inputNodes = inputNodes;
            }
        }

        const size_t probeRows = std::min<size_t>(Xnn.size(), std::max<size_t>(256, std::min<size_t>(config.neuralImportanceMaxRows, static_cast<size_t>(2000))));
        std::vector<size_t> probeOrder(Xnn.size());
        std::iota(probeOrder.begin(), probeOrder.end(), 0);
        std::mt19937 probeRng(config.neuralSeed ^ 0x85ebca6bU);
        std::shuffle(probeOrder.begin(), probeOrder.end(), probeRng);

        std::vector<std::vector<double>> probeX;
        std::vector<std::vector<double>> probeY;
        probeX.reserve(probeRows);
        probeY.reserve(probeRows);
        for (size_t i = 0; i < probeRows; ++i) {
            probeX.push_back(Xnn[probeOrder[i]]);
            probeY.push_back(Ynn[probeOrder[i]]);
        }

        const size_t probeHidden = std::clamp<size_t>(static_cast<size_t>(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes)))), 8, 48);
        NeuralNet probeNet({inputNodes, probeHidden, outputNodes});
        NeuralNet::Hyperparameters probeHp;
        probeHp.epochs = std::min<size_t>(60, std::max<size_t>(24, 8 + probeRows / 20));
        probeHp.batchSize = std::clamp<size_t>(probeRows / 16, 8, 64);
        probeHp.learningRate = config.neuralLearningRate;
        probeHp.dropoutRate = 0.05;
        probeHp.valSplit = 0.2;
        probeHp.earlyStoppingPatience = 6;
        probeHp.activation = NeuralNet::Activation::RELU;
        probeHp.outputActivation = (outputNodes == 1 && binaryTarget) ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
        probeHp.loss = (outputNodes == 1 && binaryTarget) ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
        probeHp.verbose = false;
        probeHp.seed = config.neuralSeed + 17;
        probeHp.importanceMaxRows = std::min<size_t>(config.neuralImportanceMaxRows, 2000);
        probeHp.importanceParallel = config.neuralImportanceParallel;
        probeNet.train(probeX, probeY, probeHp);

        const std::vector<double> probeImportance = probeNet.calculateFeatureImportance(
            probeX,
            probeY,
            3,
            std::min<size_t>(config.neuralImportanceMaxRows, 2000),
            config.neuralImportanceParallel);

        const size_t retainCap = std::clamp<size_t>(std::max<size_t>(24, data.rowCount() / 3), 24, inputNodes);
        if (retainCap < inputNodes && probeImportance.size() == inputNodes) {
            std::vector<std::pair<size_t, double>> ranked;
            ranked.reserve(inputNodes);
            for (size_t i = 0; i < inputNodes; ++i) ranked.push_back({i, probeImportance[i]});
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                if (a.second == b.second) return a.first < b.first;
                return a.second > b.second;
            });
            ranked.resize(retainCap);
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

            std::vector<std::vector<double>> prunedX(Xnn.size(), std::vector<double>{});
            for (size_t r = 0; r < Xnn.size(); ++r) {
                prunedX[r].reserve(retainCap);
                for (const auto& kv : ranked) prunedX[r].push_back(Xnn[r][kv.first]);
            }

            std::vector<int> prunedSource;
            prunedSource.reserve(retainCap);
            for (const auto& kv : ranked) {
                const size_t idx = kv.first;
                prunedSource.push_back((idx < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[idx] : -1);
            }

            Xnn = std::move(prunedX);
            encoded.sourceNumericFeaturePos = std::move(prunedSource);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }
    }

    if (inputNodes == 0) {
        analysis.policyUsed = "none_after_pruning";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }
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
        if (static_cast<double>(data.rowCount()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
            policyName = "fast";
        } else if (complexity < 10.0) policyName = "fast";
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

    size_t hidden = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseHidden) * policy.hiddenMultiplier)),
                                       4,
                                       static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));
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

    if (static_cast<double>(Xnn.size()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
        const size_t rowDrivenCap = std::clamp<size_t>(Xnn.size() / 3, 4, 20);
        hidden = std::min(hidden, rowDrivenCap);
        dynamicEpochs = std::min<size_t>(dynamicEpochs, 120);
        dynamicPatience = std::min(dynamicPatience, 7);
        dynamicDropout = std::max(dynamicDropout, 0.12);
    }

    size_t hiddenLayers = 1;
    if (config.neuralFixedLayers > 0) {
        hiddenLayers = static_cast<size_t>(config.neuralFixedLayers);
    } else {
        hiddenLayers = static_cast<size_t>(std::clamp<int>(
            1 + ((inputNodes > 24) ? 1 : 0) + ((outputNodes > 1) ? 1 : 0) + ((policyName == "expressive") ? 1 : 0),
            config.neuralMinLayers,
            config.neuralMaxLayers));
    }

    if (config.neuralFixedHiddenNodes > 0) {
        hidden = static_cast<size_t>(config.neuralFixedHiddenNodes);
    }

    std::vector<size_t> topology;
    topology.reserve(hiddenLayers + 2);
    topology.push_back(inputNodes);
    for (size_t l = 0; l < hiddenLayers; ++l) {
        const double decay = std::pow(0.75, static_cast<double>(l));
        const size_t width = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(hidden) * decay)), 4, static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));
        topology.push_back(width);
    }
    topology.push_back(outputNodes);

    auto estimateTrainableParams = [](const std::vector<size_t>& topo) {
        size_t params = 0;
        if (topo.size() < 2) return params;
        for (size_t i = 1; i < topo.size(); ++i) {
            params += topo[i - 1] * topo[i];
            params += topo[i];
        }
        return params;
    };

    const size_t topologyNodes = std::accumulate(topology.begin(), topology.end(), static_cast<size_t>(0));
    const size_t trainableParams = estimateTrainableParams(topology);
    if (topologyNodes > config.neuralMaxTopologyNodes) {
        throw Seldon::ConfigurationException(
            "Neural topology exceeds neural_max_topology_nodes (" +
            std::to_string(topologyNodes) + " > " + std::to_string(config.neuralMaxTopologyNodes) + ")");
    }
    if (trainableParams > config.neuralMaxTrainableParams) {
        throw Seldon::ConfigurationException(
            "Neural parameter count exceeds neural_max_trainable_params (" +
            std::to_string(trainableParams) + " > " + std::to_string(config.neuralMaxTrainableParams) + ")");
    }

    NeuralNet nn(topology);
    NeuralNet::Hyperparameters hp;

    auto toOptimizer = [](const std::string& name) {
        const std::string n = CommonUtils::toLower(name);
        if (n == "sgd") return NeuralNet::Optimizer::SGD;
        if (n == "adam") return NeuralNet::Optimizer::ADAM;
        return NeuralNet::Optimizer::LOOKAHEAD;
    };

    hp.epochs = dynamicEpochs;
    hp.batchSize = dynamicBatch;
    hp.learningRate = config.neuralLearningRate;
    hp.earlyStoppingPatience = dynamicPatience;
    hp.lrDecay = config.neuralLrDecay;
    hp.lrPlateauPatience = config.neuralLrPlateauPatience;
    hp.lrCooldownEpochs = config.neuralLrCooldownEpochs;
    hp.maxLrReductions = config.neuralMaxLrReductions;
    hp.minLearningRate = config.neuralMinLearningRate;
    hp.lrWarmupEpochs = static_cast<size_t>(std::max(0, config.neuralLrWarmupEpochs));
    hp.useCosineAnnealing = config.neuralUseCosineAnnealing;
    hp.useCyclicalLr = config.neuralUseCyclicalLr;
    hp.lrCycleEpochs = static_cast<size_t>(std::max(2, config.neuralLrCycleEpochs));
    hp.lrScheduleMinFactor = config.neuralLrScheduleMinFactor;
    hp.useValidationLossEma = config.neuralUseValidationLossEma;
    hp.validationLossEmaBeta = config.neuralValidationLossEmaBeta;
    hp.dropoutRate = dynamicDropout;
    double desiredValSplit = (Xnn.size() < 80) ? 0.30 : 0.20;
    if (Xnn.size() >= 16) {
        const size_t minValRows = std::min<size_t>(std::max<size_t>(8, Xnn.size() / 5), Xnn.size() - 8);
        desiredValSplit = std::max(desiredValSplit, static_cast<double>(minValRows) / static_cast<double>(Xnn.size()));
    }
    hp.valSplit = std::clamp(desiredValSplit, 0.10, 0.40);
    hp.l2Lambda = (Xnn.size() < 80) ? 0.010 : 0.001;
    hp.categoricalInputL2Boost = config.neuralCategoricalInputL2Boost;
    if (policyName == "fast") hp.activation = NeuralNet::Activation::RELU;
    else if (policyName == "expressive") hp.activation = NeuralNet::Activation::GELU;
    else hp.activation = (inputNodes < 10) ? NeuralNet::Activation::TANH : NeuralNet::Activation::GELU;
    hp.outputActivation = (outputNodes == 1 && binaryTarget) ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
    hp.useBatchNorm = config.neuralUseBatchNorm;
    hp.batchNormMomentum = config.neuralBatchNormMomentum;
    hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
    hp.useLayerNorm = config.neuralUseLayerNorm;
    hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
    hp.optimizer = toOptimizer(config.neuralOptimizer);
    hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
    hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
    hp.lookaheadAlpha = config.neuralLookaheadAlpha;
    hp.loss = (outputNodes == 1 && binaryTarget) ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
    hp.importanceMaxRows = config.neuralImportanceMaxRows;
    hp.importanceParallel = config.neuralImportanceParallel;
    hp.verbose = verbose;
    hp.seed = config.neuralSeed;
    hp.gradientClipNorm = config.gradientClipNorm;
    hp.adaptiveGradientClipping = config.neuralUseAdaptiveGradientClipping;
    hp.adaptiveClipBeta = config.neuralAdaptiveClipBeta;
    hp.adaptiveClipMultiplier = config.neuralAdaptiveClipMultiplier;
    hp.adaptiveClipMin = config.neuralAdaptiveClipMin;
    hp.gradientNoiseStd = config.neuralGradientNoiseStd;
    hp.gradientNoiseDecay = config.neuralGradientNoiseDecay;
    hp.useEmaWeights = config.neuralUseEmaWeights;
    hp.emaDecay = config.neuralEmaDecay;
    hp.labelSmoothing = config.neuralLabelSmoothing;
    hp.gradientAccumulationSteps = static_cast<size_t>(std::max(1, config.neuralGradientAccumulationSteps));

    std::vector<double> inputL2Scales(inputNodes, 1.0);
    for (size_t i = 0; i < inputL2Scales.size() && i < encoded.sourceNumericFeaturePos.size(); ++i) {
        if (encoded.sourceNumericFeaturePos[i] < 0) {
            inputL2Scales[i] = hp.categoricalInputL2Boost;
        }
    }

    auto buildTopology = [&](size_t layers, size_t firstHidden) {
        std::vector<size_t> topo;
        topo.reserve(layers + 2);
        topo.push_back(inputNodes);
        for (size_t l = 0; l < layers; ++l) {
            const double decay = std::pow(0.75, static_cast<double>(l));
            const size_t width = std::clamp<size_t>(
                static_cast<size_t>(std::llround(static_cast<double>(firstHidden) * decay)),
                4,
                static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));
            topo.push_back(width);
        }
        topo.push_back(outputNodes);
        return topo;
    };

    if (requested == "auto" && config.neuralFixedLayers == 0 && config.neuralFixedHiddenNodes == 0 && Xnn.size() >= 64) {
        struct ArchCandidate {
            size_t layers;
            size_t hidden;
            NeuralNet::Activation activation;
            double dropout;
        };

        std::vector<ArchCandidate> candidates = {
            {hiddenLayers, hidden, hp.activation, hp.dropoutRate},
            {std::clamp<size_t>(hiddenLayers + 1, static_cast<size_t>(config.neuralMinLayers), static_cast<size_t>(config.neuralMaxLayers)), std::min<size_t>(hidden + hidden / 4, static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes))), NeuralNet::Activation::GELU, std::clamp(hp.dropoutRate + 0.04, 0.0, 0.40)},
            {std::max<size_t>(1, hiddenLayers), std::max<size_t>(4, hidden - hidden / 5), NeuralNet::Activation::RELU, std::max(0.0, hp.dropoutRate - 0.03)}
        };

        const size_t searchRows = std::min<size_t>(Xnn.size(), 2000);
        std::vector<size_t> order(Xnn.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 searchRng(config.neuralSeed ^ 0xc2b2ae35U);
        std::shuffle(order.begin(), order.end(), searchRng);

        std::vector<std::vector<double>> searchX;
        std::vector<std::vector<double>> searchY;
        searchX.reserve(searchRows);
        searchY.reserve(searchRows);
        for (size_t i = 0; i < searchRows; ++i) {
            searchX.push_back(Xnn[order[i]]);
            searchY.push_back(Ynn[order[i]]);
        }

        double bestScore = std::numeric_limits<double>::infinity();
        ArchCandidate best = candidates.front();

        for (size_t c = 0; c < candidates.size(); ++c) {
            const auto& cand = candidates[c];
            const std::vector<size_t> candTopology = buildTopology(cand.layers, cand.hidden);
            const size_t candNodes = std::accumulate(candTopology.begin(), candTopology.end(), static_cast<size_t>(0));
            const size_t candParams = estimateTrainableParams(candTopology);
            if (candNodes > config.neuralMaxTopologyNodes || candParams > config.neuralMaxTrainableParams) {
                continue;
            }

            NeuralNet candidateNet(candTopology);
            candidateNet.setInputL2Scales(inputL2Scales);

            NeuralNet::Hyperparameters chp = hp;
            chp.epochs = std::min<size_t>(60, std::max<size_t>(24, hp.epochs / 4));
            chp.earlyStoppingPatience = std::min(10, hp.earlyStoppingPatience);
            chp.batchSize = std::min<size_t>(hp.batchSize, 64);
            chp.activation = cand.activation;
            chp.dropoutRate = cand.dropout;
            chp.verbose = false;
            chp.seed = config.neuralSeed + static_cast<uint32_t>(31 + c * 13);
            candidateNet.train(searchX, searchY, chp);

            const auto& vloss = candidateNet.getValLossHistory();
            const auto& tloss = candidateNet.getTrainLossHistory();
            const double score = !vloss.empty()
                ? vloss.back()
                : (!tloss.empty() ? tloss.back() : std::numeric_limits<double>::infinity());
            if (score < bestScore) {
                bestScore = score;
                best = cand;
            }
        }

        hiddenLayers = best.layers;
        hidden = best.hidden;
        hp.activation = best.activation;
        hp.dropoutRate = best.dropout;
        topology = buildTopology(hiddenLayers, hidden);
        nn = NeuralNet(topology);
    }

    nn.setInputL2Scales(inputL2Scales);

    if (config.neuralStreamingMode) {
        nn.trainIncremental(Xnn, Ynn, hp, std::max<size_t>(16, config.neuralStreamingChunkRows));
    } else {
        nn.train(Xnn, Ynn, hp);
    }

    analysis.inputNodes = inputNodes;
    analysis.hiddenNodes = hidden;
    analysis.outputNodes = outputNodes;
    analysis.binaryTarget = (outputNodes == 1 && binaryTarget);
    analysis.hiddenActivation = activationToString(hp.activation);
    analysis.outputActivation = activationToString(hp.outputActivation);
    analysis.epochs = hp.epochs;
    analysis.batchSize = hp.batchSize;
    analysis.valSplit = hp.valSplit;
    analysis.l2Lambda = hp.l2Lambda;
    analysis.dropoutRate = hp.dropoutRate;
    analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
    analysis.policyUsed = policy.name;
    analysis.explainabilityMethod = config.neuralExplainability;
    analysis.trainingRowsUsed = Xnn.size();
    {
        std::ostringstream topo;
        for (size_t i = 0; i < topology.size(); ++i) {
            if (i > 0) topo << " -> ";
            topo << topology[i];
        }
        analysis.topology = topo.str();
    }
    analysis.trainLoss = nn.getTrainLossHistory();
    analysis.valLoss = nn.getValLossHistory();
    analysis.gradientNorm = nn.getGradientNormHistory();
    analysis.weightStd = nn.getWeightStdHistory();
    analysis.weightMeanAbs = nn.getWeightMeanAbsHistory();

    const size_t importanceTrials = (config.neuralImportanceTrials > 0)
        ? config.neuralImportanceTrials
        : static_cast<size_t>(inputNodes > 10 ? 7 : 5);

    std::vector<double> rawImportance;
    const std::string explainability = CommonUtils::toLower(config.neuralExplainability);
    const size_t explainabilitySteps = (explainability == "hybrid") ? 2 : 1;
    size_t explainabilityStep = 0;
    printProgressBar("feature-extraction", explainabilityStep, explainabilitySteps);

    auto advanceExplainability = [&]() {
        explainabilityStep = std::min(explainabilityStep + 1, explainabilitySteps);
        printProgressBar("feature-extraction", explainabilityStep, explainabilitySteps);
    };

    if (explainability == "integrated_gradients") {
        rawImportance = nn.calculateIntegratedGradients(Xnn,
                                                        config.neuralIntegratedGradSteps,
                                                        config.neuralImportanceMaxRows);
        advanceExplainability();
    } else if (explainability == "hybrid") {
        const std::vector<double> perm = nn.calculateFeatureImportance(Xnn,
                                                                        Ynn,
                                                                        importanceTrials,
                                                                        config.neuralImportanceMaxRows,
                                                                        config.neuralImportanceParallel);
        advanceExplainability();
        const std::vector<double> ig = nn.calculateIntegratedGradients(Xnn,
                                                                        config.neuralIntegratedGradSteps,
                                                                        config.neuralImportanceMaxRows);
        advanceExplainability();
        const double wp = std::max(0.0, config.tuning.hybridExplainabilityWeightPermutation);
        const double wi = std::max(0.0, config.tuning.hybridExplainabilityWeightIntegratedGradients);
        const double wsum = std::max(config.tuning.numericEpsilon, wp + wi);
        rawImportance.assign(std::max(perm.size(), ig.size()), 0.0);
        for (size_t i = 0; i < rawImportance.size(); ++i) {
            const double p = (i < perm.size()) ? perm[i] : 0.0;
            const double g = (i < ig.size()) ? ig[i] : 0.0;
            rawImportance[i] = (wp * p + wi * g) / wsum;
        }
    } else {
        rawImportance = nn.calculateFeatureImportance(Xnn,
                                                      Ynn,
                                                      importanceTrials,
                                                      config.neuralImportanceMaxRows,
                                                      config.neuralImportanceParallel);
        advanceExplainability();
    }

    if (!Xnn.empty()) {
        const size_t uncertaintyRows = std::min<size_t>(std::min<size_t>(Xnn.size(), 128), static_cast<size_t>(512));
        std::vector<double> outStd(outputNodes, 0.0);
        std::vector<double> outCiWidth(outputNodes, 0.0);
        for (size_t r = 0; r < uncertaintyRows; ++r) {
            const auto unc = nn.predictWithUncertainty(Xnn[r], config.neuralUncertaintySamples, hp.dropoutRate);
            for (size_t j = 0; j < outputNodes && j < unc.stddev.size(); ++j) {
                outStd[j] += unc.stddev[j];
                const double width = (j < unc.ciLow.size() && j < unc.ciHigh.size()) ? (unc.ciHigh[j] - unc.ciLow[j]) : 0.0;
                outCiWidth[j] += width;
            }
        }
        if (uncertaintyRows > 0) {
            for (size_t j = 0; j < outputNodes; ++j) {
                outStd[j] /= static_cast<double>(uncertaintyRows);
                outCiWidth[j] /= static_cast<double>(uncertaintyRows);
            }
        }
        analysis.uncertaintyStd = std::move(outStd);
        analysis.uncertaintyCiWidth = std::move(outCiWidth);
    }

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

    auto isStrictMonotonicIndexLike = [&](const std::vector<double>& values) {
        if (values.size() < 6) return false;

        size_t finiteCount = 0;
        bool increasing = true;
        bool decreasing = true;
        std::vector<double> diffs;
        diffs.reserve(values.size() - 1);

        for (size_t i = 1; i < values.size(); ++i) {
            const double prev = values[i - 1];
            const double cur = values[i];
            if (!std::isfinite(prev) || !std::isfinite(cur)) return false;
            ++finiteCount;

            const double d = cur - prev;
            if (d <= 0.0) increasing = false;
            if (d >= 0.0) decreasing = false;
            diffs.push_back(std::abs(d));
        }

        if (finiteCount + 1 != values.size()) return false;
        if (!increasing && !decreasing) return false;

        const double meanStep = std::accumulate(diffs.begin(), diffs.end(), 0.0) / std::max<size_t>(1, diffs.size());
        if (meanStep <= 1e-12) return false;

        double stepVar = 0.0;
        for (double d : diffs) {
            const double dd = d - meanStep;
            stepVar += dd * dd;
        }
        stepVar /= std::max<size_t>(1, diffs.size() - 1);
        const double stepStd = std::sqrt(stepVar);

        const double relStepStd = stepStd / meanStep;
        return relStepStd <= 0.10;
    };

    bool reweightApplied = false;
    for (size_t featurePos = 0; featurePos < featureIdx.size() && featurePos < analysis.featureImportance.size(); ++featurePos) {
        const int colIdx = featureIdx[featurePos];
        if (colIdx < 0 || static_cast<size_t>(colIdx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(colIdx)].type != ColumnType::NUMERIC) continue;

        const auto& vals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(colIdx)].values);
        if (!isStrictMonotonicIndexLike(vals)) continue;

        analysis.featureImportance[featurePos] *= 0.10;
        reweightApplied = true;
    }

    if (reweightApplied) {
        const double adjustedSum = std::accumulate(analysis.featureImportance.begin(), analysis.featureImportance.end(), 0.0);
        if (adjustedSum > config.tuning.numericEpsilon) {
            for (double& imp : analysis.featureImportance) {
                imp /= adjustedSum;
            }
        }
    }

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

std::unordered_set<size_t> computeNeuralApprovedNumericFeatures(const TypedDataset& data,
                                                                int targetIdx,
                                                                const std::vector<int>& featureIdx,
                                                                const NeuralAnalysis& neural,
                                                                const NumericStatsCache& statsCache) {
    std::unordered_set<size_t> approved;
    if (featureIdx.empty() || neural.featureImportance.empty()) return approved;

    std::vector<double> absCorr = computeFeatureTargetAbsCorr(data, targetIdx, featureIdx, statsCache);

    std::vector<double> positiveImportance;
    positiveImportance.reserve(neural.featureImportance.size());
    for (double imp : neural.featureImportance) {
        if (std::isfinite(imp) && imp > 0.0) positiveImportance.push_back(imp);
    }

    double minImportance = 0.01;
    if (!positiveImportance.empty()) {
        std::sort(positiveImportance.begin(), positiveImportance.end());
        const size_t pos = static_cast<size_t>(std::floor(0.60 * static_cast<double>(positiveImportance.size() - 1)));
        minImportance = std::max(0.01, positiveImportance[pos]);
    }

    const size_t n = data.rowCount();
    for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size() && i < absCorr.size(); ++i) {
        const int col = featureIdx[i];
        if (col < 0) continue;

        const double imp = neural.featureImportance[i];
        const double corr = absCorr[i];
        const bool statSig = MathUtils::calculateSignificance(corr, n).is_significant;
        if (std::isfinite(imp) && imp >= minImportance && statSig) {
            approved.insert(static_cast<size_t>(col));
        }
    }

    if (approved.empty()) {
        double bestImp = -1.0;
        size_t bestIdx = 0;
        for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size(); ++i) {
            if (featureIdx[i] < 0) continue;
            if (neural.featureImportance[i] > bestImp) {
                bestImp = neural.featureImportance[i];
                bestIdx = static_cast<size_t>(featureIdx[i]);
            }
        }
        if (bestImp >= 0.0) approved.insert(bestIdx);
    }

    return approved;
}

struct StructuralRelation {
    size_t a = 0;
    size_t b = 0;
    size_t c = 0;
    std::string op;
};

std::vector<StructuralRelation> detectStructuralRelations(const TypedDataset& data,
                                                          const std::vector<size_t>& numericIdx,
                                                          size_t capColumns = 18,
                                                          size_t capRows = 512) {
    std::vector<StructuralRelation> out;
    if (numericIdx.size() < 3) return out;

    std::vector<size_t> cols = numericIdx;
    if (cols.size() > capColumns) cols.resize(capColumns);

    for (size_t ia = 0; ia < cols.size(); ++ia) {
        const auto& av = std::get<std::vector<double>>(data.columns()[cols[ia]].values);
        for (size_t ib = ia + 1; ib < cols.size(); ++ib) {
            const auto& bv = std::get<std::vector<double>>(data.columns()[cols[ib]].values);
            for (size_t ic = 0; ic < cols.size(); ++ic) {
                if (ic == ia || ic == ib) continue;
                const auto& cv = std::get<std::vector<double>>(data.columns()[cols[ic]].values);
                const size_t n = std::min({av.size(), bv.size(), cv.size(), data.rowCount(), capRows});
                if (n < 24) continue;

                size_t valid = 0;
                size_t addHits = 0;
                size_t mulHits = 0;
                for (size_t r = 0; r < n; ++r) {
                    if (data.columns()[cols[ia]].missing[r] || data.columns()[cols[ib]].missing[r] || data.columns()[cols[ic]].missing[r]) {
                        continue;
                    }
                    const double a = av[r];
                    const double b = bv[r];
                    const double c = cv[r];
                    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) continue;
                    const double addErr = std::abs((a + b) - c);
                    const double addScale = std::max(1.0, std::abs(c));
                    if (addErr <= (1e-5 * addScale + 1e-8)) addHits++;

                    const double mulErr = std::abs((a * b) - c);
                    const double mulScale = std::max(1.0, std::abs(c));
                    if (mulErr <= (1e-4 * mulScale + 1e-8)) mulHits++;
                    ++valid;
                }
                if (valid < 24) continue;

                const double addRate = static_cast<double>(addHits) / static_cast<double>(valid);
                const double mulRate = static_cast<double>(mulHits) / static_cast<double>(valid);
                if (addRate >= 0.985) {
                    out.push_back({cols[ia], cols[ib], cols[ic], "sum"});
                } else if (mulRate >= 0.985) {
                    out.push_back({cols[ia], cols[ib], cols[ic], "product"});
                }
            }
        }
    }

    return out;
}

std::unordered_map<size_t, size_t> buildInformationClusters(const TypedDataset& data,
                                                            const std::vector<size_t>& numericIdx,
                                                            const NumericStatsCache& statsCache,
                                                            const std::unordered_map<size_t, double>& importanceByIndex,
                                                            double absCorrThreshold = 0.97) {
    std::unordered_map<size_t, size_t> representativeByFeature;
    if (numericIdx.empty()) return representativeByFeature;

    std::unordered_map<size_t, size_t> parent;
    for (size_t idx : numericIdx) parent[idx] = idx;

    std::function<size_t(size_t)> findRoot = [&](size_t x) -> size_t {
        auto it = parent.find(x);
        if (it == parent.end()) return x;
        if (it->second == x) return x;
        it->second = findRoot(it->second);
        return it->second;
    };
    auto unite = [&](size_t a, size_t b) {
        const size_t ra = findRoot(a);
        const size_t rb = findRoot(b);
        if (ra == rb) return;
        parent[rb] = ra;
    };

    for (size_t i = 0; i < numericIdx.size(); ++i) {
        for (size_t j = i + 1; j < numericIdx.size(); ++j) {
            const size_t ia = numericIdx[i];
            const size_t ib = numericIdx[j];
            const auto& va = std::get<std::vector<double>>(data.columns()[ia].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[ib].values);
            auto aIt = statsCache.find(ia);
            auto bIt = statsCache.find(ib);
            ColumnStats aFallback = Statistics::calculateStats(va);
            ColumnStats bFallback = Statistics::calculateStats(vb);
            const ColumnStats& sa = (aIt != statsCache.end()) ? aIt->second : aFallback;
            const ColumnStats& sb = (bIt != statsCache.end()) ? bIt->second : bFallback;
            const double r = std::abs(MathUtils::calculatePearson(va, vb, sa, sb).value_or(0.0));
            if (r >= absCorrThreshold) unite(ia, ib);
        }
    }

    std::unordered_map<size_t, std::vector<size_t>> members;
    for (size_t idx : numericIdx) members[findRoot(idx)].push_back(idx);

    for (const auto& kv : members) {
        const auto& cluster = kv.second;
        size_t rep = cluster.front();
        double best = -1.0;
        for (size_t idx : cluster) {
            const double s = importanceByIndex.count(idx) ? importanceByIndex.at(idx) : 0.0;
            if (s > best) {
                best = s;
                rep = idx;
            }
        }
        for (size_t idx : cluster) representativeByFeature[idx] = rep;
    }
    return representativeByFeature;
}

double computeFoldStabilityFromCorrelation(const std::vector<double>& a,
                                           const std::vector<double>& b,
                                           size_t folds = 5) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 30 || folds < 2) return 0.0;
    const size_t foldCount = std::min(folds, n / 6);
    if (foldCount < 2) return 0.0;

    std::vector<double> foldScores;
    foldScores.reserve(foldCount);
    for (size_t f = 0; f < foldCount; ++f) {
        const size_t s = (f * n) / foldCount;
        const size_t e = ((f + 1) * n) / foldCount;
        std::vector<double> xa;
        std::vector<double> xb;
        xa.reserve(e - s);
        xb.reserve(e - s);
        for (size_t i = s; i < e; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) continue;
            xa.push_back(a[i]);
            xb.push_back(b[i]);
        }
        if (xa.size() < 8) continue;
        const ColumnStats sa = Statistics::calculateStats(xa);
        const ColumnStats sb = Statistics::calculateStats(xb);
        const double r = std::abs(MathUtils::calculatePearson(xa, xb, sa, sb).value_or(0.0));
        if (std::isfinite(r)) foldScores.push_back(r);
    }
    if (foldScores.size() < 2) return 0.0;

    const double mean = std::accumulate(foldScores.begin(), foldScores.end(), 0.0) / static_cast<double>(foldScores.size());
    if (mean <= 1e-12) return 0.0;
    double var = 0.0;
    for (double v : foldScores) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(std::max<size_t>(1, foldScores.size() - 1));
    const double cv = std::sqrt(var) / mean;
    return std::clamp(1.0 - cv, 0.0, 1.0);
}

std::vector<PairInsight> analyzeBivariatePairs(const TypedDataset& data,
                                               const std::unordered_map<size_t, double>& importanceByIndex,
                                               const std::unordered_set<size_t>& modeledIndices,
                                               const BivariateScoringPolicy& policy,
                                               GnuplotEngine* plotter,
                                               bool verbose,
                                               const NumericStatsCache& statsCache,
                                               double numericEpsilon,
                                               const HeuristicTuningConfig& tuning,
                                               size_t maxPairs,
                                               size_t maxSelectedPairs) {
    std::vector<PairInsight> pairs;
    const auto numericIdx = data.numericColumnIndices();
    const size_t n = data.rowCount();
    std::vector<size_t> evalNumericIdx = numericIdx;

    const auto representativeByFeature = buildInformationClusters(data, numericIdx, statsCache, importanceByIndex);
    const auto structuralRelations = detectStructuralRelations(data, evalNumericIdx);
    std::unordered_map<std::string, std::string> structuralPairLabel;
    for (const auto& sr : structuralRelations) {
        const size_t x = std::min(sr.a, sr.b);
        const size_t y = std::max(sr.a, sr.b);
        const std::string key = std::to_string(x) + "|" + std::to_string(y);
        const std::string label = (sr.op == "sum")
            ? (data.columns()[sr.a].name + " + " + data.columns()[sr.b].name + " ≈ " + data.columns()[sr.c].name)
            : (data.columns()[sr.a].name + " × " + data.columns()[sr.b].name + " ≈ " + data.columns()[sr.c].name);
        structuralPairLabel[key] = label;
    }

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
            p.spearman = 0.0;
            p.kendallTau = 0.0;
            p.r2 = p.r * p.r;
            // Keep regression parameter derivation centralized in MathUtils.
            auto fit = MathUtils::simpleLinearRegression(va, vb, statsA, statsB, p.r);
            p.slope = fit.first;
            p.intercept = fit.second;

            auto sig = MathUtils::calculateSignificance(p.r, n);
            p.pValue = sig.p_value;
            p.tStat = sig.t_stat;
            p.statSignificant = sig.is_significant;

            if (p.statSignificant || std::abs(p.r) >= 0.20) {
                p.spearman = MathUtils::calculateSpearman(va, vb).value_or(0.0);
                p.kendallTau = MathUtils::calculateKendallTau(va, vb).value_or(0.0);
            }

            const bool monotonicIdentity = (std::abs(p.spearman) >= 0.999 && std::abs(p.r) < 0.9985);
            const bool nearIdentity = std::abs(p.r) >= 0.9995;
            if (nearIdentity || monotonicIdentity) {
                p.filteredAsRedundant = true;
                p.relationLabel = nearIdentity ? "Near-identical signal" : "Monotonic transform proxy";
            }

            const size_t repA = representativeByFeature.count(ia) ? representativeByFeature.at(ia) : ia;
            const size_t repB = representativeByFeature.count(ib) ? representativeByFeature.at(ib) : ib;
            if (repA == repB) {
                p.filteredAsRedundant = true;
                p.redundancyGroup = data.columns()[repA].name;
                if (p.relationLabel.empty()) p.relationLabel = "Information cluster duplicate";
            }

            const size_t kx = std::min(ia, ib);
            const size_t ky = std::max(ia, ib);
            const std::string structuralKey = std::to_string(kx) + "|" + std::to_string(ky);
            auto sit = structuralPairLabel.find(structuralKey);
            if (sit != structuralPairLabel.end()) {
                p.filteredAsStructural = true;
                p.relationLabel = sit->second;
            }

            double impA = 0.0;
            double impB = 0.0;
            auto ita = importanceByIndex.find(ia);
            auto itb = importanceByIndex.find(ib);
            if (ita != importanceByIndex.end()) impA = ita->second;
            if (itb != importanceByIndex.end()) impB = itb->second;
            double impScore = std::clamp((impA + impB) / 2.0, 0.0, 1.0);
            p.effectSize = std::clamp(p.r2, 0.0, 1.0);
            p.foldStability = computeFoldStabilityFromCorrelation(va, vb, 5);
            bool aModeled = modeledIndices.find(ia) != modeledIndices.end();
            bool bModeled = modeledIndices.find(ib) != modeledIndices.end();
            double coverageFactor = (aModeled && bModeled) ? policy.coverageBoth : ((aModeled || bModeled) ? policy.coverageOne : policy.coverageNone);
            const double effectWeighted = 0.60 * p.effectSize + 0.40 * std::clamp(std::abs(p.r), 0.0, 1.0);
            p.neuralScore = (policy.wImportance * impScore + policy.wCorrelation * effectWeighted + policy.wSignificance * p.foldStability) * coverageFactor;

            const bool leakageNameHint =
                containsToken(CommonUtils::toLower(p.featureA), {"future", "post", "outcome", "label", "resolved", "actual"}) ||
                containsToken(CommonUtils::toLower(p.featureB), {"future", "post", "outcome", "label", "resolved", "actual"});
            p.leakageRisk = (std::abs(p.r) >= 0.995 && (leakageNameHint || p.filteredAsStructural));
            if (p.leakageRisk && p.relationLabel.empty()) {
                p.relationLabel = "Potential leakage proxy";
            }

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
        for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
            appendPairsForRange(aPos, aPos + 1, pairs, true);
            printProgressBar("pair-generation", aPos + 1, evalNumericIdx.size());
        }
    } else {
        #ifdef USE_OPENMP
        const int threadCount = std::max(1, omp_get_max_threads());
        std::vector<std::vector<PairInsight>> threadPairs(static_cast<size_t>(threadCount));

        #pragma omp parallel default(none) shared(threadPairs, evalNumericIdx, appendPairsForRange)
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
        printProgressBar("pair-generation", evalNumericIdx.size(), evalNumericIdx.size());
        #else
        for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
            appendPairsForRange(aPos, aPos + 1, pairs, false);
            if (((aPos + 1) % std::max<size_t>(1, evalNumericIdx.size() / 20)) == 0 || (aPos + 1) == evalNumericIdx.size()) {
                printProgressBar("pair-generation", aPos + 1, evalNumericIdx.size());
            }
        }
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

    std::vector<size_t> selectedCandidateIdx;
    selectedCandidateIdx.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].filteredAsRedundant || pairs[i].filteredAsStructural || pairs[i].leakageRisk) continue;
        if (pairs[i].statSignificant && pairs[i].neuralScore >= dynamicCutoff) {
            selectedCandidateIdx.push_back(i);
        }
    }

    std::sort(selectedCandidateIdx.begin(), selectedCandidateIdx.end(), [&](size_t lhs, size_t rhs) {
        if (pairs[lhs].neuralScore == pairs[rhs].neuralScore) {
            return pairs[lhs].effectSize > pairs[rhs].effectSize;
        }
        return pairs[lhs].neuralScore > pairs[rhs].neuralScore;
    });

    std::unordered_set<size_t> finalSelected;
    const size_t keepCount = std::min(selectedCandidateIdx.size(), maxSelectedPairs);
    for (size_t i = 0; i < keepCount; ++i) {
        finalSelected.insert(selectedCandidateIdx[i]);
    }

    for (size_t pairIdx = 0; pairIdx < pairs.size(); ++pairIdx) {
        auto& p = pairs[pairIdx];
        p.selected = finalSelected.find(pairIdx) != finalSelected.end();
        if (p.selected && plotter) {
            const auto& va = std::get<std::vector<double>>(data.columns()[p.idxA].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[p.idxB].values);
            const size_t sampleSize = std::min(va.size(), vb.size());
            std::string id = "sig_" + p.featureA + "_" + p.featureB;
            const bool addFit = shouldOverlayFittedLine(p.r,
                                                        p.statSignificant,
                                                        va,
                                                        vb,
                                                        p.slope,
                                                        p.intercept,
                                                        tuning);
            const bool addBand = shouldAddConfidenceBand(p.r,
                                                         p.statSignificant,
                                                         sampleSize,
                                                         tuning);
            p.fitLineAdded = addFit;
            p.confidenceBandAdded = addBand;
            p.plotPath = plotter->scatter(id,
                                          va,
                                          vb,
                                          p.featureA + " vs " + p.featureB,
                                          addFit,
                                          p.slope,
                                          p.intercept,
                                          "Fitted line",
                                          addBand,
                                          1.96,
                                          tuning.scatterDownsampleThreshold);

            if (shouldAddResidualPlot(p.r, p.selected, sampleSize, tuning)) {
                std::vector<double> fitted;
                std::vector<double> residuals;
                fitted.reserve(sampleSize);
                residuals.reserve(sampleSize);
                for (size_t i = 0; i < sampleSize; ++i) {
                    if (!std::isfinite(va[i]) || !std::isfinite(vb[i])) continue;
                    const double yHat = p.slope * va[i] + p.intercept;
                    fitted.push_back(yHat);
                    residuals.push_back(vb[i] - yHat);
                }
                p.residualPlotPath = plotter->residual("resid_" + p.featureA + "_" + p.featureB,
                                                       fitted,
                                                       residuals,
                                                       "Residuals: " + p.featureA + " -> " + p.featureB);
            }

            if (auto facetIdx = chooseFacetingColumn(data, p.idxA, p.idxB, tuning); facetIdx.has_value()) {
                const auto& facetVals = std::get<std::vector<std::string>>(data.columns()[*facetIdx].values);
                p.facetedPlotPath = plotter->facetedScatter("facet_" + p.featureA + "_" + p.featureB,
                                                            va,
                                                            vb,
                                                            facetVals,
                                                            p.featureA + " vs " + p.featureB + " by " + data.columns()[*facetIdx].name,
                                                            tuning.facetMaxCategories);
            }

            const BivariateStackedBarData stacked = buildBivariateStackedBar(va, vb);
            if (stacked.valid) {
                p.stackedPlotPath = plotter->stackedBar("sigstack_" + p.featureA + "_" + p.featureB,
                                                        stacked.categories,
                                                        {p.featureB + " <= median", p.featureB + " > median"},
                                                        {stacked.lowCounts, stacked.highCounts},
                                                        "Stacked Profile: " + p.featureA + " x " + p.featureB,
                                                        "Count");
            }
        }
        if (verbose) {
            std::cout << "[Seldon][Bivariate] Decision " << p.featureA << " vs " << p.featureB
                      << " => " << (p.selected ? "SELECTED" : "REJECTED")
                      << " (cutoff=" << toFixed(dynamicCutoff, 8) << ")\n";
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairInsight& lhs, const PairInsight& rhs) {
        if (lhs.selected != rhs.selected) return lhs.selected > rhs.selected;
        if (lhs.effectSize == rhs.effectSize) return std::abs(lhs.r) > std::abs(rhs.r);
        return lhs.effectSize > rhs.effectSize;
    });

    return pairs;
}

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

std::vector<ContingencyInsight> analyzeContingencyPairs(const TypedDataset& data) {
    std::vector<ContingencyInsight> out;
    const auto cats = data.categoricalColumnIndices();
    for (size_t i = 0; i < cats.size(); ++i) {
        for (size_t j = i + 1; j < cats.size(); ++j) {
            const auto& aCol = std::get<std::vector<std::string>>(data.columns()[cats[i]].values);
            const auto& bCol = std::get<std::vector<std::string>>(data.columns()[cats[j]].values);
            const size_t n = std::min(aCol.size(), bCol.size());
            if (n < 20) continue;

            std::map<std::string, size_t> aMap;
            std::map<std::string, size_t> bMap;
            std::vector<std::pair<std::string, std::string>> rows;
            rows.reserve(n);
            for (size_t r = 0; r < n; ++r) {
                if (data.columns()[cats[i]].missing[r] || data.columns()[cats[j]].missing[r]) continue;
                rows.push_back({aCol[r], bCol[r]});
                aMap.emplace(aCol[r], aMap.size());
                bMap.emplace(bCol[r], bMap.size());
            }
            if (rows.size() < 20 || aMap.size() < 2 || bMap.size() < 2) continue;

            const size_t R = aMap.size();
            const size_t C = bMap.size();
            std::vector<std::vector<double>> table(R, std::vector<double>(C, 0.0));
            for (const auto& row : rows) {
                table[aMap[row.first]][bMap[row.second]] += 1.0;
            }

            std::vector<double> rowSum(R, 0.0), colSum(C, 0.0);
            double total = 0.0;
            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    rowSum[r] += table[r][c];
                    colSum[c] += table[r][c];
                    total += table[r][c];
                }
            }
            if (total <= 0.0) continue;

            double chi2 = 0.0;
            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    const double expected = (rowSum[r] * colSum[c]) / total;
                    if (expected <= 1e-12) continue;
                    const double d = table[r][c] - expected;
                    chi2 += (d * d) / expected;
                }
            }
            const double p = std::exp(-0.5 * chi2); // simple approximation
            const double v = std::sqrt(std::max(0.0, chi2 / (total * static_cast<double>(std::min(R - 1, C - 1)))));

            double oratio = 1.0, lo = 1.0, hi = 1.0;
            if (R == 2 && C == 2) {
                const double a = table[0][0] + 0.5;
                const double b = table[0][1] + 0.5;
                const double c = table[1][0] + 0.5;
                const double d = table[1][1] + 0.5;
                oratio = (a * d) / (b * c);
                const double se = std::sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d);
                const double l = std::log(oratio);
                lo = std::exp(l - 1.96 * se);
                hi = std::exp(l + 1.96 * se);
            }

            out.push_back({data.columns()[cats[i]].name, data.columns()[cats[j]].name, chi2, p, v, oratio, lo, hi});
            if (out.size() >= 12) return out;
        }
    }
    return out;
}

struct AnovaInsight {
    std::string categorical;
    std::string numeric;
    double fStat = 0.0;
    double pValue = 1.0;
    double eta2 = 0.0;
    std::string tukeySummary;
};

std::vector<AnovaInsight> analyzeAnovaPairs(const TypedDataset& data) {
    std::vector<AnovaInsight> out;
    const auto cats = data.categoricalColumnIndices();
    const auto nums = data.numericColumnIndices();
    for (size_t cidx : cats) {
        const auto& cv = std::get<std::vector<std::string>>(data.columns()[cidx].values);
        for (size_t nidx : nums) {
            const auto& nv = std::get<std::vector<double>>(data.columns()[nidx].values);
            const size_t n = std::min(cv.size(), nv.size());
            if (n < 24) continue;

            std::map<std::string, std::vector<double>> groups;
            for (size_t i = 0; i < n; ++i) {
                if (data.columns()[cidx].missing[i] || data.columns()[nidx].missing[i] || !std::isfinite(nv[i])) continue;
                groups[cv[i]].push_back(nv[i]);
            }
            if (groups.size() < 2 || groups.size() > 10) continue;

            std::vector<double> all;
            for (const auto& kv : groups) all.insert(all.end(), kv.second.begin(), kv.second.end());
            if (all.size() < 24) continue;
            const double grand = std::accumulate(all.begin(), all.end(), 0.0) / static_cast<double>(all.size());

            double ssb = 0.0;
            double ssw = 0.0;
            std::vector<std::pair<std::string, double>> means;
            for (const auto& kv : groups) {
                if (kv.second.size() < 2) continue;
                const double mu = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / static_cast<double>(kv.second.size());
                means.push_back({kv.first, mu});
                const double dm = mu - grand;
                ssb += static_cast<double>(kv.second.size()) * dm * dm;
                for (double v : kv.second) {
                    const double d = v - mu;
                    ssw += d * d;
                }
            }
            const double dfb = static_cast<double>(groups.size() - 1);
            const double dfw = static_cast<double>(all.size() - groups.size());
            if (dfb <= 0.0 || dfw <= 0.0) continue;
            const double msb = ssb / dfb;
            const double msw = ssw / dfw;
            const double f = (msw <= 1e-12) ? 0.0 : (msb / msw);
            const double p = std::exp(-0.5 * f);
            const double eta2 = (ssb + ssw <= 1e-12) ? 0.0 : (ssb / (ssb + ssw));

            std::sort(means.begin(), means.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string tukey = "n/a";
            if (p < 0.05 && means.size() >= 2) {
                const double delta = std::abs(means[0].second - means[1].second);
                tukey = means[0].first + " vs " + means[1].first + " Δ=" + toFixed(delta, 4);
            }

            out.push_back({data.columns()[cidx].name, data.columns()[nidx].name, f, p, eta2, tukey});
            if (out.size() >= 12) return out;
        }
    }
    return out;
}

std::pair<double, double> bootstrapCI(const std::vector<double>& values,
                                      size_t rounds = 300,
                                      double alpha = 0.05,
                                      uint32_t seed = 1337,
                                      bool showProgress = false,
                                      const std::string& label = "bootstrap") {
    if (values.empty()) return {0.0, 0.0};
    FastRng rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
    std::vector<double> stats;
    stats.reserve(rounds);
    for (size_t b = 0; b < rounds; ++b) {
        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i) sum += values[rng.uniformIndex(values.size())];
        stats.push_back(sum / static_cast<double>(values.size()));
        if (showProgress && (((b + 1) % std::max<size_t>(1, rounds / 25)) == 0 || (b + 1) == rounds)) {
            printProgressBar(label, b + 1, rounds);
        }
    }
    std::sort(stats.begin(), stats.end());
    const double lo = CommonUtils::quantileByNth(stats, alpha / 2.0);
    const double hi = CommonUtils::quantileByNth(stats, 1.0 - alpha / 2.0);
    return {lo, hi};
}

struct PCAInsight {
    std::vector<double> pc1;
    std::vector<double> pc2;
    std::vector<double> explained;
    std::vector<std::string> labels;
};

PCAInsight runPCA2(const TypedDataset& data,
                   const std::vector<size_t>& numericIdx,
                   size_t maxRows = 300) {
    PCAInsight out;
    if (numericIdx.size() < 2) return out;

    std::vector<std::vector<double>> X;
    const size_t nRows = data.rowCount();
    for (size_t r = 0; r < nRows; ++r) {
        std::vector<double> row;
        row.reserve(numericIdx.size());
        bool ok = true;
        for (size_t idx : numericIdx) {
            const auto& v = std::get<std::vector<double>>(data.columns()[idx].values);
            if (r >= v.size() || data.columns()[idx].missing[r] || !std::isfinite(v[r])) {
                ok = false;
                break;
            }
            row.push_back(v[r]);
        }
        if (ok) X.push_back(std::move(row));
        if (X.size() >= maxRows) break;
    }
    if (X.size() < 12) return out;

    const size_t p = numericIdx.size();
    std::vector<double> mu(p, 0.0), sd(p, 0.0);
    for (const auto& row : X) for (size_t j = 0; j < p; ++j) mu[j] += row[j];
    for (size_t j = 0; j < p; ++j) mu[j] /= static_cast<double>(X.size());
    for (const auto& row : X) for (size_t j = 0; j < p; ++j) { double d = row[j] - mu[j]; sd[j] += d * d; }
    for (size_t j = 0; j < p; ++j) { sd[j] = std::sqrt(sd[j] / std::max<size_t>(1, X.size() - 1)); if (sd[j] <= 1e-12) sd[j] = 1.0; }
    for (auto& row : X) for (size_t j = 0; j < p; ++j) row[j] = (row[j] - mu[j]) / sd[j];

    std::vector<std::vector<double>> C(p, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i; j < p; ++j) {
            double s = 0.0;
            for (const auto& row : X) s += row[i] * row[j];
            s /= static_cast<double>(std::max<size_t>(1, X.size() - 1));
            C[i][j] = s;
            C[j][i] = s;
        }
    }

    auto power = [&](const std::vector<std::vector<double>>& M) {
        std::vector<double> v(p, 1.0 / std::sqrt(static_cast<double>(p)));
        for (int it = 0; it < 80; ++it) {
            std::vector<double> nv(p, 0.0);
            for (size_t i = 0; i < p; ++i) for (size_t j = 0; j < p; ++j) nv[i] += M[i][j] * v[j];
            double norm = 0.0;
            for (double q : nv) norm += q * q;
            norm = std::sqrt(std::max(norm, 1e-12));
            for (double& q : nv) q /= norm;
            v = std::move(nv);
        }
        double eig = 0.0;
        for (size_t i = 0; i < p; ++i) {
            double mv = 0.0;
            for (size_t j = 0; j < p; ++j) mv += M[i][j] * v[j];
            eig += v[i] * mv;
        }
        return std::make_pair(v, eig);
    };

    auto [v1, e1] = power(C);
    std::vector<std::vector<double>> C2 = C;
    for (size_t i = 0; i < p; ++i) for (size_t j = 0; j < p; ++j) C2[i][j] -= e1 * v1[i] * v1[j];
    auto [v2, e2] = power(C2);

    out.pc1.reserve(X.size());
    out.pc2.reserve(X.size());
    for (const auto& row : X) {
        double s1 = 0.0, s2 = 0.0;
        for (size_t j = 0; j < p; ++j) {
            s1 += row[j] * v1[j];
            s2 += row[j] * v2[j];
        }
        out.pc1.push_back(s1);
        out.pc2.push_back(s2);
    }
    out.explained = {std::max(0.0, e1), std::max(0.0, e2)};
    double total = 0.0;
    for (size_t j = 0; j < p; ++j) total += std::max(0.0, C[j][j]);
    if (total > 1e-12) {
        out.explained[0] /= total;
        out.explained[1] /= total;
    }
    for (size_t idx : numericIdx) out.labels.push_back(data.columns()[idx].name);
    return out;
}

struct KMeansInsight {
    size_t bestK = 0;
    double silhouette = 0.0;
    double gapStatistic = 0.0;
    std::vector<int> labels;
};

KMeansInsight runKMeans2D(const std::vector<double>& x, const std::vector<double>& y) {
    KMeansInsight out;
    const size_t n = std::min(x.size(), y.size());
    if (n < 20) return out;

    auto dist = [&](size_t a, size_t b) {
        const double dx = x[a] - x[b];
        const double dy = y[a] - y[b];
        return std::sqrt(dx * dx + dy * dy);
    };

    double bestSil = -1.0;
    double bestGap = 0.0;
    std::vector<int> bestLabels;
    size_t bestK = 0;
    for (size_t k = 2; k <= std::min<size_t>(6, n / 5); ++k) {
        std::vector<std::pair<double, double>> centers;
        for (size_t i = 0; i < k; ++i) centers.push_back({x[(i * n) / k], y[(i * n) / k]});
        std::vector<int> labels(n, 0);
        for (int it = 0; it < 20; ++it) {
            for (size_t i = 0; i < n; ++i) {
                double bd = std::numeric_limits<double>::infinity();
                int bc = 0;
                for (size_t c = 0; c < k; ++c) {
                    const double dx = x[i] - centers[c].first;
                    const double dy = y[i] - centers[c].second;
                    const double d = dx * dx + dy * dy;
                    if (d < bd) { bd = d; bc = static_cast<int>(c); }
                }
                labels[i] = bc;
            }
            std::vector<double> sx(k, 0.0), sy(k, 0.0), sc(k, 0.0);
            for (size_t i = 0; i < n; ++i) {
                const int c = labels[i];
                sx[c] += x[i]; sy[c] += y[i]; sc[c] += 1.0;
            }
            for (size_t c = 0; c < k; ++c) {
                if (sc[c] > 0.0) centers[c] = {sx[c] / sc[c], sy[c] / sc[c]};
            }
        }

        double sil = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double a = 0.0; size_t ac = 0;
            std::vector<double> bsum(k, 0.0);
            std::vector<size_t> bcnt(k, 0);
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const double d = dist(i, j);
                const int cj = labels[j];
                bsum[cj] += d;
                bcnt[cj]++;
                if (labels[i] == cj) { a += d; ac++; }
            }
            a = (ac > 0) ? (a / static_cast<double>(ac)) : 0.0;
            double b = std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < k; ++c) {
                if (static_cast<int>(c) == labels[i] || bcnt[c] == 0) continue;
                b = std::min(b, bsum[c] / static_cast<double>(bcnt[c]));
            }
            if (!std::isfinite(b)) b = a;
            const double den = std::max(a, b);
            if (den > 1e-12) sil += (b - a) / den;
        }
        sil /= static_cast<double>(n);

        double wk = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const auto& ctr = centers[static_cast<size_t>(labels[i])];
            const double dx = x[i] - ctr.first;
            const double dy = y[i] - ctr.second;
            wk += dx * dx + dy * dy;
        }

        const double minX = *std::min_element(x.begin(), x.end());
        const double maxX = *std::max_element(x.begin(), x.end());
        const double minY = *std::min_element(y.begin(), y.end());
        const double maxY = *std::max_element(y.begin(), y.end());
        std::mt19937 rg(1337U + static_cast<uint32_t>(k));
        std::uniform_real_distribution<double> ux(minX, maxX);
        std::uniform_real_distribution<double> uy(minY, maxY);
        double wkRef = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double rx = ux(rg);
            const double ry = uy(rg);
            double bd = std::numeric_limits<double>::infinity();
            for (const auto& ctr : centers) {
                const double dx = rx - ctr.first;
                const double dy = ry - ctr.second;
                bd = std::min(bd, dx * dx + dy * dy);
            }
            wkRef += bd;
        }
        const double gap = std::log(std::max(1e-12, wkRef)) - std::log(std::max(1e-12, wk));

        if (sil > bestSil) {
            bestSil = sil;
            bestGap = gap;
            bestLabels = labels;
            bestK = k;
        }
    }

    out.bestK = bestK;
    out.silhouette = std::max(0.0, bestSil);
    out.gapStatistic = bestGap;
    out.labels = std::move(bestLabels);
    return out;
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

    {
        std::vector<std::vector<std::string>> typeSummary = {
            {"numeric", std::to_string(data.numericColumnIndices().size())},
            {"categorical", std::to_string(data.categoricalColumnIndices().size())},
            {"datetime", std::to_string(data.datetimeColumnIndices().size())}
        };
        report.addTable("Data Type Summary", {"Type", "Columns"}, typeSummary);

        std::vector<std::vector<std::string>> qualityRows;
        qualityRows.reserve(data.columns().size());
        for (const auto& col : data.columns()) {
            const size_t miss = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : 0;
            const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;
            const double missPct = data.rowCount() == 0 ? 0.0 : (100.0 * static_cast<double>(miss) / static_cast<double>(data.rowCount()));
            qualityRows.push_back({
                col.name,
                col.type == ColumnType::NUMERIC ? "numeric" : (col.type == ColumnType::DATETIME ? "datetime" : "categorical"),
                std::to_string(miss),
                toFixed(missPct, 2) + "%",
                std::to_string(outliers)
            });
        }
        report.addTable("Column Quality Matrix", {"Column", "Type", "Missing", "Missing %", "Outliers"}, qualityRows);
    }

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
        const bool tooManyMissingBars = PlotHeuristics::shouldAvoidCategoryHeavyCharts(labels.size(), 18);
        if (!tooManyMissingBars) {
            std::string missingImg = overallPlotter->bar("overall_missingness", labels, missingCounts, "Overall Missingness by Column");
            if (!missingImg.empty()) report.addImage("Overall Missingness", missingImg);
        }

        if (PlotHeuristics::shouldAddPieChart(missingCounts, config.tuning)) {
            std::string missingPie = overallPlotter->pie("overall_missingness_pie",
                                                         labels,
                                                         missingCounts,
                                                         "Missingness Share by Column");
            if (!missingPie.empty()) report.addImage("Missingness Share (Pie)", missingPie);
        }

        std::vector<std::vector<double>> missHeat(1, std::vector<double>(missingCounts.size(), 0.0));
        for (size_t i = 0; i < missingCounts.size(); ++i) missHeat[0][i] = missingCounts[i];
        std::string missHeatImg = overallPlotter->heatmap("overall_missingness_heatmap", missHeat, "Missingness Heatmap", labels);
        if (!missHeatImg.empty()) report.addImage("Missingness Heatmap", missHeatImg);

        if (auto timeline = detectProjectTimeline(data, config.tuning); timeline.has_value()) {
            std::string gantt = overallPlotter->gantt("overall_project_timeline",
                                                      timeline->taskNames,
                                                      timeline->start,
                                                      timeline->end,
                                                      timeline->semantics,
                                                      "Project Timeline (Auto-Detected)");
            if (!gantt.empty()) {
                report.addImage("Project Timeline (Gantt)", gantt);
                report.addParagraph("Gantt chart generated automatically from project-like columns (task + start/end or start/duration). Timeline preview limit uses gantt_max_tasks tuning.");
            }
        }

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

            const std::vector<size_t> order = clusteredOrderFromCorrelation(corr);
            std::vector<std::vector<double>> clustered(corr.size(), std::vector<double>(corr.size(), 0.0));
            std::vector<std::string> heatLabels;
            heatLabels.reserve(order.size());
            for (size_t i = 0; i < order.size(); ++i) {
                heatLabels.push_back(data.columns()[numericIdx[order[i]]].name);
                for (size_t j = 0; j < order.size(); ++j) {
                    clustered[i][j] = corr[order[i]][order[j]];
                }
            }

            std::string img = overallPlotter->heatmap("overall_corr_heatmap", clustered, "Overall Correlation Heatmap (Clustered)", heatLabels);
            if (!img.empty()) report.addImage("Overall Correlation Heatmap", img);
        }

        if (!neural.trainLoss.empty()) {
            std::vector<double> epochs(neural.trainLoss.size(), 0.0);
            for (size_t i = 0; i < epochs.size(); ++i) epochs[i] = static_cast<double>(i + 1);
            if (!neural.valLoss.empty() && neural.valLoss.size() == neural.trainLoss.size()) {
                std::string lossImg = overallPlotter->multiLine("overall_nn_loss_multi",
                                                                epochs,
                                                                {neural.trainLoss, neural.valLoss},
                                                                {"Train", "Validation"},
                                                                "NN Loss Curves",
                                                                "Loss");
                if (!lossImg.empty()) {
                    report.addImage("Overall NN Loss (Train vs Validation)", lossImg);
                }
            } else {
                std::string trainImg = overallPlotter->line("overall_nn_train_loss", epochs, neural.trainLoss, "NN Train Loss");
                if (!trainImg.empty()) report.addImage("Overall NN Train Loss", trainImg);
            }
        }

        if (!neural.gradientNorm.empty()) {
            std::vector<double> x(neural.gradientNorm.size(), 0.0);
            for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i + 1);
            std::string gradImg = overallPlotter->line("overall_nn_gradient_norm", x, neural.gradientNorm, "NN Gradient Norm");
            if (!gradImg.empty()) report.addImage("Overall NN Gradient Norm", gradImg);
        }

        if (!neural.weightStd.empty() || !neural.weightMeanAbs.empty()) {
            const size_t n = std::max(neural.weightStd.size(), neural.weightMeanAbs.size());
            std::vector<double> x(n, 0.0);
            for (size_t i = 0; i < n; ++i) x[i] = static_cast<double>(i + 1);
            std::vector<double> weightStdSeries(n, 0.0);
            std::vector<double> weightAbsSeries(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                if (i < neural.weightStd.size()) weightStdSeries[i] = neural.weightStd[i];
                if (i < neural.weightMeanAbs.size()) weightAbsSeries[i] = neural.weightMeanAbs[i];
            }
            std::string weightDyn = overallPlotter->multiLine("overall_nn_weight_dynamics",
                                                               x,
                                                               {weightStdSeries, weightAbsSeries},
                                                               {"Weight RMS", "Weight Mean Abs"},
                                                               "NN Weight Dynamics",
                                                               "Value");
            if (!weightDyn.empty()) report.addImage("Overall NN Weight Dynamics", weightDyn);
        }

        auto numericIdxForParallel = data.numericColumnIndices();
        if (numericIdxForParallel.size() >= config.tuning.parallelCoordinatesMinDims &&
            numericIdxForParallel.size() <= config.tuning.parallelCoordinatesMaxDims &&
            data.rowCount() >= config.tuning.parallelCoordinatesMinRows) {
            std::vector<std::string> axisLabels;
            axisLabels.reserve(numericIdxForParallel.size());
            for (size_t idx : numericIdxForParallel) {
                axisLabels.push_back(data.columns()[idx].name);
            }

            std::vector<std::vector<double>> matrix;
            matrix.reserve(data.rowCount());
            for (size_t r = 0; r < data.rowCount(); ++r) {
                std::vector<double> row;
                row.reserve(numericIdxForParallel.size());
                bool ok = true;
                for (size_t idx : numericIdxForParallel) {
                    const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
                    if (r >= vals.size() || data.columns()[idx].missing[r] || !std::isfinite(vals[r])) {
                        ok = false;
                        break;
                    }
                    row.push_back(vals[r]);
                }
                if (ok) matrix.push_back(std::move(row));
            }

            std::string parallel = overallPlotter->parallelCoordinates("overall_parallel_coords",
                                                                        matrix,
                                                                        axisLabels,
                                                                        "Parallel Coordinates (Numeric Features)");
            if (!parallel.empty()) report.addImage("Parallel Coordinates", parallel);
        }

        if (auto ts = detectTimeSeriesSignal(data, config.tuning); ts.has_value()) {
            std::string trend = overallPlotter->timeSeriesTrend("overall_timeseries_trend",
                                                                ts->timeX,
                                                                ts->values,
                                                                ts->valueName + " over " + ts->timeName,
                                                                true);
            if (!trend.empty()) {
                report.addImage("Automatic Time-Series Trend", trend);
            }

            std::vector<double> trendComp = ts->values;
            const size_t w = std::clamp<size_t>(ts->values.size() / 12, 3, 24);
            for (size_t i = 0; i < ts->values.size(); ++i) {
                size_t lo = (i >= w) ? (i - w + 1) : 0;
                double s = 0.0;
                size_t c = 0;
                for (size_t j = lo; j <= i; ++j) {
                    s += ts->values[j];
                    ++c;
                }
                trendComp[i] = s / static_cast<double>(std::max<size_t>(1, c));
            }
            std::vector<double> seasonal(ts->values.size(), 0.0);
            std::vector<double> resid(ts->values.size(), 0.0);
            for (size_t i = 0; i < ts->values.size(); ++i) {
                seasonal[i] = ts->values[i] - trendComp[i];
                resid[i] = ts->values[i] - trendComp[i] - seasonal[i];
            }
            std::string decomp = overallPlotter->multiLine("overall_timeseries_decomposition",
                                                           ts->timeX,
                                                           {ts->values, trendComp, seasonal, resid},
                                                           {"Observed", "Trend", "Seasonal", "Residual"},
                                                           "Time-Series Decomposition (Additive)",
                                                           "Value");
            if (!decomp.empty()) {
                report.addImage("Time-Series Decomposition", decomp);
            }
        }

        {
            auto numIdx = data.numericColumnIndices();
            if (numIdx.size() >= 3) {
                const size_t cap = std::min<size_t>(numIdx.size(), config.tuning.parallelCoordinatesMaxDims);
                numIdx.resize(cap);
                const PCAInsight pca = runPCA2(data, numIdx, 300);
                if (!pca.pc1.empty() && pca.pc1.size() == pca.pc2.size()) {
                    std::vector<std::vector<std::string>> pcaRows;
                    pcaRows.push_back({"PC1", toFixed(100.0 * (pca.explained.empty() ? 0.0 : pca.explained[0]), 2) + "%"});
                    pcaRows.push_back({"PC2", toFixed(100.0 * (pca.explained.size() > 1 ? pca.explained[1] : 0.0), 2) + "%"});
                    report.addTable("PCA Explained Variance", {"Component", "Explained"}, pcaRows);

                    std::string pcaScatter = overallPlotter->scatter("overall_pca_biplot_like",
                                                                      pca.pc1,
                                                                      pca.pc2,
                                                                      "PCA Scores (PC1 vs PC2)",
                                                                      false,
                                                                      0.0,
                                                                      0.0,
                                                                      "",
                                                                      false,
                                                                      1.96,
                                                                      config.tuning.scatterDownsampleThreshold);
                    if (!pcaScatter.empty()) report.addImage("PCA Biplot (Score Projection)", pcaScatter);

                    std::vector<double> comps = {1.0, 2.0};
                    std::vector<double> exp = {pca.explained[0], pca.explained.size() > 1 ? pca.explained[1] : 0.0};
                    std::string scree = overallPlotter->line("overall_pca_scree", comps, exp, "PCA Scree (Top Components)");
                    if (!scree.empty()) report.addImage("PCA Scree Plot", scree);

                    KMeansInsight km = runKMeans2D(pca.pc1, pca.pc2);
                    if (km.bestK >= 2 && km.labels.size() == pca.pc1.size()) {
                        std::vector<std::string> fac;
                        fac.reserve(km.labels.size());
                        std::vector<double> counts(km.bestK, 0.0);
                        for (int label : km.labels) {
                            const size_t cl = static_cast<size_t>(std::max(0, label));
                            if (cl < counts.size()) counts[cl] += 1.0;
                            fac.push_back("cluster_" + std::to_string(label + 1));
                        }
                        std::vector<std::string> lbl;
                        for (size_t k = 0; k < km.bestK; ++k) lbl.push_back("Cluster " + std::to_string(k + 1));
                        std::string cbar = overallPlotter->bar("overall_kmeans_profile", lbl, counts, "K-Means Cluster Sizes");
                        if (!cbar.empty()) report.addImage("K-Means Cluster Profile", cbar);
                        std::string facPlot = overallPlotter->facetedScatter("overall_kmeans_faceted",
                                                                             pca.pc1,
                                                                             pca.pc2,
                                                                             fac,
                                                                             "PC Space by K-Means Cluster",
                                                                             std::min<size_t>(km.bestK, 6));
                        if (!facPlot.empty()) report.addImage("K-Means Cluster Visualization", facPlot);
                        report.addParagraph("K-Means auto-selection chose k=" + std::to_string(km.bestK) + " with silhouette=" + toFixed(km.silhouette, 4) + " and gap=" + toFixed(km.gapStatistic, 4) + ".");
                    }
                }
            }
        }

        {
            size_t categoricalPlotsAdded = 0;
            for (size_t cidx : data.categoricalColumnIndices()) {
                const auto& col = data.columns()[cidx];
                const auto& vals = std::get<std::vector<std::string>>(col.values);
                const CategoryFrequencyProfile profile = buildCategoryFrequencyProfile(vals, col.missing, 12);
                if (profile.labels.size() < 2) continue;

                const bool usePie = PlotHeuristics::shouldAddPieChart(profile.counts, config.tuning);
                const bool avoidBar = PlotHeuristics::shouldAvoidCategoryHeavyCharts(profile.labels.size(), 12);

                if (usePie) {
                    std::string pie = overallPlotter->pie("overall_cat_pie_" + std::to_string(cidx),
                                                          profile.labels,
                                                          profile.counts,
                                                          "Category Share: " + col.name);
                    if (!pie.empty()) {
                        report.addImage("Category Share (Pie): " + col.name, pie);
                        ++categoricalPlotsAdded;
                    }
                }

                if (!usePie && !avoidBar) {
                    std::string bar = overallPlotter->bar("overall_cat_bar_" + std::to_string(cidx),
                                                          profile.labels,
                                                          profile.counts,
                                                          "Category Frequency: " + col.name);
                    if (!bar.empty()) {
                        report.addImage("Category Frequency: " + col.name, bar);
                        ++categoricalPlotsAdded;
                    }
                }

                if (categoricalPlotsAdded >= 8) break;
            }
        }

        {
            std::vector<std::vector<std::string>> bootRows;
            for (size_t idx : data.numericColumnIndices()) {
                const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
                if (vals.size() < 20) continue;
                auto [lo, hi] = bootstrapCI(vals,
                                            250,
                                            0.05,
                                            config.neuralSeed ^ static_cast<uint32_t>(idx),
                                            verbose,
                                            "bootstrap-mean");
                const double mu = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
                bootRows.push_back({"mean(" + data.columns()[idx].name + ")", toFixed(mu, 6), toFixed(lo, 6), toFixed(hi, 6)});
                if (bootRows.size() >= 6) break;
            }
            if (!benchmarks.empty()) {
                std::vector<double> rmseVals;
                for (const auto& b : benchmarks) rmseVals.push_back(b.rmse);
                auto [lo, hi] = bootstrapCI(rmseVals,
                                            250,
                                            0.05,
                                            config.benchmarkSeed,
                                            verbose,
                                            "bootstrap-rmse");
                const double mu = std::accumulate(rmseVals.begin(), rmseVals.end(), 0.0) / static_cast<double>(std::max<size_t>(1, rmseVals.size()));
                bootRows.push_back({"benchmark_rmse", toFixed(mu, 6), toFixed(lo, 6), toFixed(hi, 6)});
            }

            const auto nums = data.numericColumnIndices();
            if (nums.size() >= 2) {
                const auto& x = std::get<std::vector<double>>(data.columns()[nums[0]].values);
                const auto& y = std::get<std::vector<double>>(data.columns()[nums[1]].values);
                const size_t n = std::min(x.size(), y.size());
                if (n >= 20) {
                    std::vector<double> corrBoot;
                    std::vector<double> slopeBoot;
                    corrBoot.reserve(220);
                    slopeBoot.reserve(220);
                    FastRng rng(static_cast<uint64_t>(config.neuralSeed) ^ 0x1234567ULL);
                    for (size_t b = 0; b < 220; ++b) {
                        std::vector<double> xs;
                        std::vector<double> ys;
                        xs.reserve(n);
                        ys.reserve(n);
                        for (size_t i = 0; i < n; ++i) {
                            const size_t idx = rng.uniformIndex(n);
                            xs.push_back(x[idx]);
                            ys.push_back(y[idx]);
                        }
                        const ColumnStats sx = Statistics::calculateStats(xs);
                        const ColumnStats sy = Statistics::calculateStats(ys);
                        const double r = MathUtils::calculatePearson(xs, ys, sx, sy).value_or(0.0);
                        const auto fit = MathUtils::simpleLinearRegression(xs, ys, sx, sy, r);
                        corrBoot.push_back(r);
                        slopeBoot.push_back(fit.first);
                        if (verbose && (((b + 1) % 11) == 0 || (b + 1) == 220)) {
                            printProgressBar("bootstrap-corr", b + 1, 220);
                        }
                    }
                    std::sort(corrBoot.begin(), corrBoot.end());
                    std::sort(slopeBoot.begin(), slopeBoot.end());
                    bootRows.push_back({"corr(" + data.columns()[nums[0]].name + "," + data.columns()[nums[1]].name + ")",
                                        toFixed(corrBoot[corrBoot.size() / 2], 6),
                                        toFixed(CommonUtils::quantileByNth(corrBoot, 0.025), 6),
                                        toFixed(CommonUtils::quantileByNth(corrBoot, 0.975), 6)});
                    bootRows.push_back({"slope(" + data.columns()[nums[1]].name + "~" + data.columns()[nums[0]].name + ")",
                                        toFixed(slopeBoot[slopeBoot.size() / 2], 6),
                                        toFixed(CommonUtils::quantileByNth(slopeBoot, 0.025), 6),
                                        toFixed(CommonUtils::quantileByNth(slopeBoot, 0.975), 6)});
                }
            }
            if (!bootRows.empty()) {
                report.addTable("Bootstrap Confidence Intervals", {"Metric", "Estimate", "CI Low", "CI High"}, bootRows);
            }
        }

        {
            const size_t rows = std::min<size_t>(data.rowCount(), 80);
            const size_t cols = std::min<size_t>(data.colCount(), 30);
            if (rows >= 5 && cols >= 2) {
                std::vector<std::vector<double>> miss(rows, std::vector<double>(cols, 0.0));
                std::vector<std::string> missLabels;
                for (size_t c = 0; c < cols; ++c) missLabels.push_back(data.columns()[c].name);
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t c = 0; c < cols; ++c) {
                        miss[r][c] = data.columns()[c].missing[r] ? 1.0 : 0.0;
                    }
                }
                std::string missMap = overallPlotter->heatmap("overall_missingness_matrix", miss, "Missingness Matrix", missLabels);
                if (!missMap.empty()) report.addImage("Missingness Matrix", missMap);

                std::vector<std::vector<std::string>> missCorrRows;
                for (size_t c = 0; c < cols && c < data.columns().size(); ++c) {
                    std::vector<double> missVec(rows, 0.0);
                    for (size_t r = 0; r < rows; ++r) missVec[r] = miss[r][c];
                    for (size_t nidx : data.numericColumnIndices()) {
                        const auto& vals = std::get<std::vector<double>>(data.columns()[nidx].values);
                        std::vector<double> y;
                        for (size_t r = 0; r < rows && r < vals.size(); ++r) y.push_back(vals[r]);
                        if (y.size() != missVec.size() || y.size() < 8) continue;
                        ColumnStats s1 = Statistics::calculateStats(missVec);
                        ColumnStats s2 = Statistics::calculateStats(y);
                        double r = MathUtils::calculatePearson(missVec, y, s1, s2).value_or(0.0);
                        if (std::abs(r) >= 0.15) {
                            missCorrRows.push_back({data.columns()[c].name, data.columns()[nidx].name, toFixed(r, 4)});
                        }
                    }
                }
                if (!missCorrRows.empty()) {
                    if (missCorrRows.size() > 20) missCorrRows.resize(20);
                    report.addTable("Missingness Correlation", {"Missing Column", "Numeric Column", "pearson_r"}, missCorrRows);
                }
            }
        }

        {
            auto it = std::find_if(benchmarks.begin(), benchmarks.end(), [](const BenchmarkResult& b) {
                return b.model == "LinearRegression";
            });
            if (it != benchmarks.end() && !it->actual.empty() && it->actual.size() == it->predicted.size()) {
                std::vector<double> residuals;
                residuals.reserve(it->actual.size());
                for (size_t i = 0; i < it->actual.size(); ++i) residuals.push_back(it->actual[i] - it->predicted[i]);

                std::string residPlot = overallPlotter->residual("overall_linear_residual",
                                                                 it->predicted,
                                                                 residuals,
                                                                 "Linear Regression Residuals vs Fitted");
                if (!residPlot.empty()) report.addImage("Linear Regression Residual Diagnostic", residPlot);

                std::vector<double> sortedRes = residuals;
                std::sort(sortedRes.begin(), sortedRes.end());
                const double mu = std::accumulate(sortedRes.begin(), sortedRes.end(), 0.0) / static_cast<double>(sortedRes.size());
                double sd = 0.0;
                for (double v : sortedRes) { const double d = v - mu; sd += d * d; }
                sd = std::sqrt(sd / static_cast<double>(std::max<size_t>(1, sortedRes.size() - 1)));
                if (sd <= 1e-12) sd = 1.0;
                std::vector<double> theo(sortedRes.size(), 0.0);
                for (size_t i = 0; i < sortedRes.size(); ++i) {
                    const double p = (static_cast<double>(i) + 0.5) / static_cast<double>(sortedRes.size());
                    theo[i] = mu + sd * std::sqrt(2.0) * std::erf(2.0 * p - 1.0);
                }
                std::string qqPlot = overallPlotter->scatter("overall_linear_qq",
                                                             theo,
                                                             sortedRes,
                                                             "Q-Q Plot (Approx Normal)",
                                                             false,
                                                             0.0,
                                                             0.0,
                                                             "",
                                                             false,
                                                             1.96,
                                                             5000);
                if (!qqPlot.empty()) report.addImage("Linear Residual Q-Q Plot", qqPlot);

                std::vector<std::vector<std::string>> diagRows;
                double mse = 0.0;
                for (double e : residuals) mse += e * e;
                const double mseDen = static_cast<double>(std::max<size_t>(1, residuals.size()));
                mse /= mseDen;
                std::vector<double> cooks;
                cooks.reserve(residuals.size());
                const double p = std::max(1.0, static_cast<double>(data.numericColumnIndices().size()));
                for (size_t i = 0; i < residuals.size(); ++i) {
                    const double h = std::min(0.99, 1.0 / mseDen + (static_cast<double>(i) / mseDen) * 0.01);
                    const double cd = (residuals[i] * residuals[i] / (p * std::max(mse, 1e-12))) * (h / std::pow(1.0 - h, 2.0));
                    cooks.push_back(cd);
                }
                const double maxCook = cooks.empty() ? 0.0 : *std::max_element(cooks.begin(), cooks.end());

                double maxVif = 0.0;
                const auto nums = data.numericColumnIndices();
                if (nums.size() >= 2) {
                    for (size_t i = 0; i < nums.size(); ++i) {
                        const auto& vi = std::get<std::vector<double>>(data.columns()[nums[i]].values);
                        double r2max = 0.0;
                        for (size_t j = 0; j < nums.size(); ++j) {
                            if (i == j) continue;
                            const auto& vj = std::get<std::vector<double>>(data.columns()[nums[j]].values);
                            ColumnStats si = Statistics::calculateStats(vi);
                            ColumnStats sj = Statistics::calculateStats(vj);
                            double r = MathUtils::calculatePearson(vi, vj, si, sj).value_or(0.0);
                            r2max = std::max(r2max, r * r);
                        }
                        const double vif = 1.0 / std::max(1e-6, 1.0 - r2max);
                        maxVif = std::max(maxVif, vif);
                    }
                }

                diagRows.push_back({"MSE", toFixed(mse, 6)});
                diagRows.push_back({"Max Cook Distance", toFixed(maxCook, 6)});
                diagRows.push_back({"Max Approx VIF", toFixed(maxVif, 6)});
                report.addTable("Regression Diagnostics (Linear Benchmark)", {"Metric", "Value"}, diagRows);
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
    {
        namespace fs = std::filesystem;
        const fs::path datasetPath(runCfg.datasetPath);
        fs::path baseDir = datasetPath.parent_path();
        if (baseDir.empty()) baseDir = fs::current_path();
        const std::string stem = datasetPath.stem().string().empty() ? "dataset" : datasetPath.stem().string();
        if (runCfg.outputDir.empty()) {
            runCfg.outputDir = (baseDir / (stem + "_seldon_outputs")).string();
        }
        runCfg.assetsDir = (fs::path(runCfg.outputDir) / "seldon_report_assets").string();
        runCfg.reportFile = (fs::path(runCfg.outputDir) / "neural_synthesis.md").string();
    }
    CliProgressSpinner progress(!runCfg.verboseAnalysis && ::isatty(STDOUT_FILENO));
    constexpr size_t totalSteps = 10;
    size_t currentStep = 0;

    auto advance = [&](const std::string& label) {
        ++currentStep;
        progress.update(label, currentStep, totalSteps);
    };

    cleanupOutputs(runCfg);
    MathUtils::setSignificanceAlpha(runCfg.tuning.significanceAlpha);
    MathUtils::setNumericTuning(runCfg.tuning.numericEpsilon,
                                runCfg.tuning.betaFallbackIntervalsStart,
                                runCfg.tuning.betaFallbackIntervalsMax,
                                runCfg.tuning.betaFallbackTolerance);

    TypedDataset data(config.datasetPath, config.delimiter);
    if (runCfg.numericLocaleHint == "us") {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::US_THOUSANDS);
    } else if (runCfg.numericLocaleHint == "eu") {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::EUROPEAN);
    } else {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::AUTO);
    }

    if (runCfg.datetimeLocaleHint == "dmy") {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::DMY);
    } else if (runCfg.datetimeLocaleHint == "mdy") {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::MDY);
    } else {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::AUTO);
    }

    if (!runCfg.columnTypeOverrides.empty()) {
        std::unordered_map<std::string, ColumnType> typeOverrides;
        typeOverrides.reserve(runCfg.columnTypeOverrides.size());
        for (const auto& kv : runCfg.columnTypeOverrides) {
            const std::string normalized = CommonUtils::toLower(CommonUtils::trim(kv.second));
            ColumnType type = ColumnType::CATEGORICAL;
            if (normalized == "numeric") {
                type = ColumnType::NUMERIC;
            } else if (normalized == "datetime") {
                type = ColumnType::DATETIME;
            }
            typeOverrides[CommonUtils::toLower(CommonUtils::trim(kv.first))] = type;
        }
        data.setColumnTypeOverrides(std::move(typeOverrides));
    }

    data.load();
    advance("Loaded dataset");
    if (data.rowCount() == 0 || data.colCount() == 0) {
        throw Seldon::DatasetException("Dataset has no usable rows/columns");
    }

    validateExcludedColumns(data, config);
    const TargetContext targetContext = resolveTargetContext(data, config, runCfg);
    const int targetIdx = targetContext.targetIdx;
    applyDynamicPlotDefaultsIfUnset(runCfg, data);

    const bool autoFastMode = (data.rowCount() > 100000) || (data.numericColumnIndices().size() > 50);
    const bool fastModeEnabled = runCfg.fastMode || autoFastMode;
    if (fastModeEnabled && CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
        runCfg.neuralStrategy = "fast";
    }

    const size_t rawColumnCount = data.colCount();
    PreprocessReport prep = Preprocessor::run(data, runCfg);
    advance("Preprocessed dataset");
    exportPreprocessedDatasetIfRequested(data, runCfg);
    normalizeBinaryTarget(data, targetIdx, targetContext.semantics);
    const NumericStatsCache statsCache = buildNumericStatsCache(data);
    advance("Prepared stats cache");

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Preparing deeply detailed univariate analysis...\n";
    }

    ReportEngine univariate;
    univariate.addTitle("Univariate Analysis");
    univariate.addParagraph("Dataset: " + runCfg.datasetPath);
    const size_t totalAnalysisDimensions = data.colCount();
    const size_t engineeredFeatureCount =
        (totalAnalysisDimensions >= rawColumnCount) ? (totalAnalysisDimensions - rawColumnCount) : 0;
    univariate.addParagraph("Dataset Stats:");
    univariate.addParagraph("Rows: " + std::to_string(data.rowCount()));
    univariate.addParagraph("Raw Columns: " + std::to_string(rawColumnCount));
    univariate.addParagraph("Engineered Features: " + std::to_string(engineeredFeatureCount));
    univariate.addParagraph("Total Analysis Dimensions: " + std::to_string(totalAnalysisDimensions));
    addUnivariateDetailedSection(univariate, data, prep, runCfg.verboseAnalysis, statsCache);
    advance("Built univariate tables");

    GnuplotEngine plotterBivariate(plotSubdir(runCfg, "bivariate"), runCfg.plot);
    GnuplotEngine plotterUnivariate(plotSubdir(runCfg, "univariate"), runCfg.plot);
    GnuplotEngine plotterOverall(plotSubdir(runCfg, "overall"), runCfg.plot);
    const bool canPlot = configurePlotAvailability(runCfg, univariate, plotterBivariate);

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
    advance("Finished benchmarks");

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
    advance("Completed neural analysis");
    neural.featureImportance = buildCoherentImportance(data, targetIdx, featureIdx, neural, benchmarks, runCfg, statsCache);
    const std::unordered_set<size_t> neuralApprovedNumericFeatures = computeNeuralApprovedNumericFeatures(data,
                                                                                                           targetIdx,
                                                                                                           featureIdx,
                                                                                                           neural,
                                                                                                           statsCache);
    addUnivariatePlots(univariate,
                       data,
                       runCfg,
                       canPlot,
                       plotterUnivariate,
                       neuralApprovedNumericFeatures);
    advance("Generated univariate plots");

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
                                                runCfg.tuning,
                                                fastModeEnabled ? runCfg.fastMaxBivariatePairs : 0,
                                                std::max<size_t>(8, std::min<size_t>(120, neuralApprovedNumericFeatures.size() * 6)));
    advance("Analyzed bivariate pairs");

    ReportEngine bivariate;
    bivariate.addTitle("Bivariate Analysis");
    if (totalPossiblePairs > bivariatePairs.size()) {
        bivariate.addParagraph("Fast mode active: pair evaluation was capped for runtime safety. Results below cover the highest-priority numeric columns only.");
    } else {
        bivariate.addParagraph("All numeric pair combinations are included below (nC2). Significant table is dynamically filtered using statistical significance, effect size, and fold-stability-weighted neural relevance.");
    }
    bivariate.addParagraph("Neural-lattice relevance score prioritizes practical effect size over raw p-value magnitude; only selected pairs are considered significant findings.");

    std::vector<std::vector<std::string>> allRows;
    std::vector<std::vector<std::string>> sigRows;
    size_t statSigCount = 0;
    for (const auto& p : bivariatePairs) {
        if (p.statSignificant) statSigCount++;
        allRows.push_back({
            p.featureA,
            p.featureB,
            toFixed(p.r),
            toFixed(p.spearman),
            toFixed(p.kendallTau),
            toFixed(p.r2),
            toFixed(p.effectSize, 6),
            toFixed(p.foldStability, 6),
            toFixed(p.slope),
            toFixed(p.intercept),
            toFixed(p.tStat, 6),
            toFixed(p.pValue, 6),
            p.statSignificant ? "yes" : "no",
            toFixed(p.neuralScore, 6),
            p.selected ? "yes" : "no",
            p.filteredAsRedundant ? "yes" : "no",
            p.filteredAsStructural ? "yes" : "no",
            p.leakageRisk ? "yes" : "no",
            p.redundancyGroup.empty() ? "-" : p.redundancyGroup,
            p.relationLabel.empty() ? "-" : p.relationLabel,
            p.fitLineAdded ? "yes" : "no",
            p.confidenceBandAdded ? "yes" : "no",
            p.stackedPlotPath.empty() ? "-" : p.stackedPlotPath,
            p.residualPlotPath.empty() ? "-" : p.residualPlotPath,
            p.facetedPlotPath.empty() ? "-" : p.facetedPlotPath,
            p.plotPath.empty() ? "-" : p.plotPath
        });

        if (p.selected) {
            sigRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.spearman),
                toFixed(p.kendallTau),
                toFixed(p.r2),
                toFixed(p.effectSize, 6),
                toFixed(p.foldStability, 6),
                toFixed(p.slope),
                toFixed(p.intercept),
                toFixed(p.tStat, 6),
                toFixed(p.pValue, 6),
                toFixed(p.neuralScore, 6),
                p.relationLabel.empty() ? "-" : p.relationLabel,
                p.fitLineAdded ? "yes" : "no",
                p.confidenceBandAdded ? "yes" : "no",
                p.stackedPlotPath.empty() ? "-" : p.stackedPlotPath,
                p.residualPlotPath.empty() ? "-" : p.residualPlotPath,
                p.facetedPlotPath.empty() ? "-" : p.facetedPlotPath,
                p.plotPath.empty() ? "-" : p.plotPath
            });
            if (!p.plotPath.empty()) {
                bivariate.addImage("Significant Pair: " + p.featureA + " vs " + p.featureB, p.plotPath);
            }
            if (!p.stackedPlotPath.empty()) {
                bivariate.addImage("Stacked Profile: " + p.featureA + " vs " + p.featureB, p.stackedPlotPath);
            }
            if (!p.residualPlotPath.empty()) {
                bivariate.addImage("Residual Plot: " + p.featureA + " vs " + p.featureB, p.residualPlotPath);
            }
            if (!p.facetedPlotPath.empty()) {
                bivariate.addImage("Faceted Scatter: " + p.featureA + " vs " + p.featureB, p.facetedPlotPath);
            }
        }
    }

    std::vector<std::string> topTakeaways;
    std::unordered_set<size_t> usedFeatures;
    std::vector<PairInsight> rankedTakeaways;
    for (const auto& p : bivariatePairs) {
        if (!p.selected) continue;
        if (p.filteredAsRedundant || p.filteredAsStructural || p.leakageRisk) continue;
        rankedTakeaways.push_back(p);
    }
    std::sort(rankedTakeaways.begin(), rankedTakeaways.end(), [](const PairInsight& a, const PairInsight& b) {
        if (a.effectSize == b.effectSize) return a.neuralScore > b.neuralScore;
        return a.effectSize > b.effectSize;
    });
    for (const auto& p : rankedTakeaways) {
        if (topTakeaways.size() >= 3) break;
        if (usedFeatures.count(p.idxA) || usedFeatures.count(p.idxB)) continue;
        std::string label;
        if (std::abs(p.r) >= 0.90) {
            label = p.featureA + " is a strong proxy for " + p.featureB;
        } else if (std::abs(p.r) >= 0.70) {
            label = p.featureA + " and " + p.featureB + " encode overlapping signal";
        } else {
            label = p.featureA + " and " + p.featureB + " show moderate but stable association";
        }
        label += " (effect=" + toFixed(p.effectSize, 3) + ", stability=" + toFixed(p.foldStability, 3) + ")";
        topTakeaways.push_back(label);
        usedFeatures.insert(p.idxA);
        usedFeatures.insert(p.idxB);
    }

    std::vector<std::string> redundancyDrops;
    std::unordered_set<std::string> seenDrop;
    for (const auto& p : bivariatePairs) {
        if (!p.filteredAsRedundant || p.redundancyGroup.empty()) continue;
        const double impA = importanceByIndex.count(p.idxA) ? importanceByIndex[p.idxA] : 0.0;
        const double impB = importanceByIndex.count(p.idxB) ? importanceByIndex[p.idxB] : 0.0;
        const std::string drop = (impA < impB) ? p.featureA : p.featureB;
        const std::string keep = (impA < impB) ? p.featureB : p.featureA;
        const std::string recommendation = "drop " + drop + " (redundant with " + keep + ")";
        if (seenDrop.insert(recommendation).second) {
            redundancyDrops.push_back(recommendation);
            if (redundancyDrops.size() >= 8) break;
        }
    }

    bivariate.addParagraph("Total pairs evaluated: " + std::to_string(allRows.size()));
    bivariate.addParagraph("Statistically significant pairs (p<" + toFixed(MathUtils::getSignificanceAlpha(), 4) + "): " + std::to_string(statSigCount));
    bivariate.addParagraph("Final neural-selected significant pairs: " + std::to_string(sigRows.size()));
    const size_t redundantPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsRedundant; }));
    const size_t structuralPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsStructural; }));
    const size_t leakagePairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.leakageRisk; }));
    bivariate.addParagraph("Information-theoretic filtering: redundant=" + std::to_string(redundantPairs) + ", structural=" + std::to_string(structuralPairs) + ", leakage-risk=" + std::to_string(leakagePairs) + ".");
    bivariate.addTable("All Pairwise Results", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "stat_sig", "neural_score", "selected", "redundant", "structural", "leakage_risk", "cluster_rep", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, allRows);
    bivariate.addTable("Final Significant Results", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, sigRows);

    const auto contingency = analyzeContingencyPairs(data);
    if (!contingency.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto& c : contingency) {
            rows.push_back({c.catA, c.catB, toFixed(c.chi2, 4), toFixed(c.pValue, 6), toFixed(c.cramerV, 4), toFixed(c.oddsRatio, 4), toFixed(c.oddsCiLow, 4), toFixed(c.oddsCiHigh, 4)});
        }
        bivariate.addTable("Categorical Contingency Analysis", {"Cat A", "Cat B", "chi2", "p_value", "cramers_v", "odds_ratio", "or_ci_low", "or_ci_high"}, rows);
    }

    const auto anovaRows = analyzeAnovaPairs(data);
    if (!anovaRows.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto& a : anovaRows) {
            rows.push_back({a.categorical, a.numeric, toFixed(a.fStat, 4), toFixed(a.pValue, 6), toFixed(a.eta2, 4), a.tukeySummary});
        }
        bivariate.addTable("One-Way ANOVA (Categorical→Numeric)", {"Categorical", "Numeric", "F", "p_value", "eta_squared", "posthoc_tukey"}, rows);
    }

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
    neuralReport.addParagraph(std::string("Task type inferred from target: ") + targetContext.semantics.inferredTask);
    neuralReport.addTable("Auto Decision Log", {"Decision", "Value"}, {
        {"Target Selection", targetContext.userProvidedTarget ? "user-specified" : "auto"},
        {"Target Strategy", targetContext.choice.strategyUsed},
        {"Target Column", runCfg.targetColumn},
        {"Target Task", targetContext.semantics.inferredTask},
        {"Target Cardinality", std::to_string(targetContext.semantics.cardinality)},
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
        {"Topology", neural.topology},
        {"Input Nodes", std::to_string(neural.inputNodes)},
        {"Numeric Input Nodes", std::to_string(neural.numericInputNodes)},
        {"Categorical Encoded Nodes", std::to_string(neural.categoricalEncodedNodes)},
        {"Hidden Nodes", std::to_string(neural.hiddenNodes)},
        {"Output Nodes", std::to_string(neural.outputNodes)},
        {"Auxiliary Output Targets", std::to_string(neural.outputAuxTargets)},
        {"Epochs", std::to_string(neural.epochs)},
        {"Batch Size", std::to_string(neural.batchSize)},
        {"Validation Split", toFixed(neural.valSplit, 2)},
        {"L2 Lambda", toFixed(neural.l2Lambda, 4)},
        {"Dropout", toFixed(neural.dropoutRate, 4)},
        {"Explainability", neural.explainabilityMethod},
        {"Categorical Importance Share", toFixed(neural.categoricalImportanceShare, 4)},
        {"Early Stop Patience", std::to_string(neural.earlyStoppingPatience)},
        {"Loss", neural.binaryTarget ? "cross_entropy" : "mse"},
        {"Hidden Activation", neural.hiddenActivation},
        {"Output Activation", neural.outputActivation}
    });

    if (!neural.uncertaintyStd.empty()) {
        std::vector<std::vector<std::string>> uncertaintyRows;
        for (size_t i = 0; i < neural.uncertaintyStd.size(); ++i) {
            const double ciw = (i < neural.uncertaintyCiWidth.size()) ? neural.uncertaintyCiWidth[i] : 0.0;
            uncertaintyRows.push_back({
                "output_" + std::to_string(i + 1),
                toFixed(neural.uncertaintyStd[i], 6),
                toFixed(ciw, 6)
            });
        }
        neuralReport.addTable("Predictive Uncertainty (MC Dropout)", {"Output", "Avg StdDev", "Avg 95% CI Width"}, uncertaintyRows);
    }

    std::vector<std::vector<std::string>> fiRows;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        fiRows.push_back({name, toFixed(imp, 6)});
    }
    neuralReport.addTable("Feature Importance (Neural Explainability)", {"Feature", "Importance"}, fiRows);

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

    if (!neural.gradientNorm.empty() || !neural.weightStd.empty() || !neural.weightMeanAbs.empty()) {
        std::vector<std::vector<std::string>> dynRows;
        const size_t n = std::max({neural.gradientNorm.size(), neural.weightStd.size(), neural.weightMeanAbs.size()});
        for (size_t i = 0; i < n; ++i) {
            const double g = (i < neural.gradientNorm.size()) ? neural.gradientNorm[i] : 0.0;
            const double wstd = (i < neural.weightStd.size()) ? neural.weightStd[i] : 0.0;
            const double wabs = (i < neural.weightMeanAbs.size()) ? neural.weightMeanAbs[i] : 0.0;
            dynRows.push_back({std::to_string(i + 1), toFixed(g, 6), toFixed(wstd, 6), toFixed(wabs, 6)});
        }
        neuralReport.addTable("Neural Training Dynamics", {"Step", "Gradient Norm", "Weight RMS", "Weight Mean Abs"}, dynRows);
    }

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
    advance("Built overall sections");

    ReportEngine finalAnalysis;
    finalAnalysis.addTitle("Final Analysis - Significant Findings Only");
    finalAnalysis.addParagraph("This report contains only neural-net-approved significant findings (dynamic decision engine). Non-selected findings are excluded by design.");
    finalAnalysis.addParagraph("Data Health Score: " + toFixed(dataHealth.score, 1) + "/100 (" + dataHealth.band + "). This score estimates discovered signal strength using completeness, retained feature coverage, significant-pair yield, selected-pair yield, and neural training stability.");

    if (!topTakeaways.empty()) {
        std::vector<std::vector<std::string>> takeawayRows;
        for (size_t i = 0; i < topTakeaways.size(); ++i) {
            takeawayRows.push_back({std::to_string(i + 1), topTakeaways[i]});
        }
        finalAnalysis.addTable("Top 3 Takeaways", {"Rank", "Takeaway"}, takeawayRows);
    }

    if (!redundancyDrops.empty()) {
        std::vector<std::vector<std::string>> dropRows;
        for (const auto& msg : redundancyDrops) {
            dropRows.push_back({msg});
        }
        finalAnalysis.addTable("Redundancy Drop Recommendations", {"Action"}, dropRows);
    }

    finalAnalysis.addTable("Neural-Selected Significant Bivariate Findings", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, sigRows);

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
    advance("Saved reports");
    progress.done("Pipeline complete");
    printPipelineCompletion(runCfg);

    return 0;
}
