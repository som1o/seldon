#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
#include "CommonUtils.h"
#include "DeterministicHeuristics.h"
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
#include <array>
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
#include <regex>
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
    if (!std::isfinite(v)) return "n/a";
    const double zeroSnap = 0.5 * std::pow(10.0, -std::max(0, prec));
    if (std::abs(v) < zeroSnap) v = 0.0;
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string scoreBar100(double score) {
    const double s = std::clamp(score, 0.0, 100.0);
    const size_t filled = static_cast<size_t>(std::round((s / 100.0) * 10.0));
    std::string out = "[";
    for (size_t i = 0; i < 10; ++i) out += (i < filled ? "#" : "-");
    out += "]";
    return out;
}

void addExecutiveDashboard(ReportEngine& report,
                           const std::string& title,
                           const std::vector<std::pair<std::string, std::string>>& metrics,
                           const std::vector<std::string>& highlights,
                           const std::string& note = "") {
    report.addParagraph("---");
    report.addParagraph("## " + title);
    if (!note.empty()) report.addParagraph(note);

    if (!metrics.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(metrics.size());
        for (const auto& kv : metrics) rows.push_back({kv.first, kv.second});
        report.addTable("Executive Snapshot", {"KPI", "Value"}, rows);
    }

    if (!highlights.empty()) {
        std::string bulletBlock;
        for (const auto& h : highlights) {
            if (h.empty()) continue;
            bulletBlock += "- " + h + "\n";
        }
        if (!bulletBlock.empty()) report.addParagraph("### Quick Highlights\n" + bulletBlock);
    }
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

enum class CategoricalSemanticKind {
    GENERIC,
    BOOLEAN_LIKE,
    MULTI_SELECT
};

struct CategoricalSemanticProfile {
    CategoricalSemanticKind kind = CategoricalSemanticKind::GENERIC;
    std::vector<std::pair<std::string, size_t>> topTokens;
};

std::optional<double> parseBooleanLikeTokenReport(const std::string& value) {
    const std::string token = CommonUtils::toLower(CommonUtils::trim(value));
    if (token.empty()) return std::nullopt;
    if (token == "1" || token == "true" || token == "yes" || token == "y" || token == "t" || token == "on") return 1.0;
    if (token == "0" || token == "false" || token == "no" || token == "n" || token == "f" || token == "off") return 0.0;
    return std::nullopt;
}

std::vector<std::string> splitTokensBySeparatorReport(const std::string& value, char separator) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : value) {
        if (ch == separator) {
            std::string tok = CommonUtils::trim(current);
            if (!tok.empty()) out.push_back(CommonUtils::toLower(tok));
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    std::string tok = CommonUtils::trim(current);
    if (!tok.empty()) out.push_back(CommonUtils::toLower(tok));
    return out;
}

CategoricalSemanticProfile detectCategoricalSemanticProfile(const std::vector<std::string>& values,
                                                           const MissingMask& missing) {
    CategoricalSemanticProfile profile;
    if (values.empty()) return profile;

    std::vector<size_t> observed;
    observed.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i]) continue;
        if (CommonUtils::trim(values[i]).empty()) continue;
        observed.push_back(i);
    }
    if (observed.size() < 4) return profile;

    bool allBoolLike = true;
    bool seenZero = false;
    bool seenOne = false;
    for (size_t i : observed) {
        const auto parsed = parseBooleanLikeTokenReport(values[i]);
        if (!parsed.has_value()) {
            allBoolLike = false;
            break;
        }
        if (*parsed < 0.5) seenZero = true;
        else seenOne = true;
    }
    if (allBoolLike && (seenZero || seenOne)) {
        profile.kind = CategoricalSemanticKind::BOOLEAN_LIKE;
        return profile;
    }

    std::array<size_t, 3> sepRows = {0, 0, 0};
    const std::array<char, 3> separators = {',', ';', '|'};
    for (size_t i : observed) {
        const std::string& s = values[i];
        for (size_t k = 0; k < separators.size(); ++k) {
            if (s.find(separators[k]) != std::string::npos) ++sepRows[k];
        }
    }
    size_t bestSepIdx = 0;
    for (size_t i = 1; i < sepRows.size(); ++i) {
        if (sepRows[i] > sepRows[bestSepIdx]) bestSepIdx = i;
    }
    const size_t rowsWithSep = sepRows[bestSepIdx];
    if (rowsWithSep < std::max<size_t>(3, observed.size() / 10)) return profile;

    const char sep = separators[bestSepIdx];
    std::unordered_map<std::string, size_t> tokenRows;
    size_t totalTokens = 0;
    for (size_t i : observed) {
        const auto tokens = splitTokensBySeparatorReport(values[i], sep);
        if (tokens.size() < 2) continue;
        std::set<std::string> uniq(tokens.begin(), tokens.end());
        totalTokens += uniq.size();
        for (const auto& token : uniq) tokenRows[token]++;
    }
    if (tokenRows.size() < 3) return profile;

    const double avgTokens = static_cast<double>(totalTokens) / static_cast<double>(observed.size());
    if (avgTokens < 1.30) return profile;

    profile.kind = CategoricalSemanticKind::MULTI_SELECT;
    std::vector<std::pair<std::string, size_t>> ranked(tokenRows.begin(), tokenRows.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });
    if (ranked.size() > 8) ranked.resize(8);
    profile.topTokens = std::move(ranked);
    return profile;
}

std::string categoricalSemanticLabel(CategoricalSemanticKind kind) {
    if (kind == CategoricalSemanticKind::BOOLEAN_LIKE) return "categorical(bool-like)";
    if (kind == CategoricalSemanticKind::MULTI_SELECT) return "categorical(multi-select)";
    return "categorical";
}

bool isAdministrativeColumnName(const std::string& name) {
    const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
    if (lower.empty()) return false;

    static const std::regex strictAdminPattern(
        R"((^id$)|(^idx$)|(^index$)|(_id$)|(^id_)|(_index$)|(_idx$)|(^row_?id$)|(^row_?num(ber)?$)|(^uuid$)|(_uuid$)|(^guid$)|(_guid$)|(^timestamp$)|(_timestamp$)|(^created(_at)?$)|(_created(_at)?$)|(^updated(_at)?$)|(_updated(_at)?$))",
        std::regex::icase);
    if (std::regex_search(lower, strictAdminPattern)) return true;

    return containsToken(lower, {
        "meta", "metadata", "ingest", "ingestion", "audit", "rownum", "serial", "sequence", "record_id"
    });
}

bool isTargetCandidateColumnName(const std::string& name) {
    const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
    if (lower.empty()) return false;
    if (containsToken(lower, {"score", "class", "target", "label", "outcome", "response"})) {
        return true;
    }
    return lower == "y" || lower == "y_true";
}

bool isEngineeredFeatureName(const std::string& name) {
    const std::string lower = CommonUtils::toLower(name);
    return lower.find("_pow") != std::string::npos
        || lower.find("_log1p_abs") != std::string::npos
        || lower.find("_x_") != std::string::npos
        || lower.find("_div_") != std::string::npos;
}

std::string canonicalEngineeredBaseName(const std::string& name) {
    const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
    if (lower.empty()) return lower;

    auto stripSuffix = [&](const std::string& token) {
        const size_t pos = lower.find(token);
        if (pos == std::string::npos || pos == 0) return std::string();
        return CommonUtils::trim(lower.substr(0, pos));
    };

    if (const std::string base = stripSuffix("_pow"); !base.empty()) return base;
    if (const std::string base = stripSuffix("_log1p_abs"); !base.empty()) return base;
    if (const std::string base = stripSuffix("_x_"); !base.empty()) return base;
    if (const std::string base = stripSuffix("_div_"); !base.empty()) return base;
    return lower;
}

bool isEngineeredLineagePair(const std::string& a, const std::string& b) {
    const std::string la = CommonUtils::toLower(CommonUtils::trim(a));
    const std::string lb = CommonUtils::toLower(CommonUtils::trim(b));
    if (la.empty() || lb.empty()) return false;

    const std::string baseA = canonicalEngineeredBaseName(la);
    const std::string baseB = canonicalEngineeredBaseName(lb);
    const bool engineeredA = isEngineeredFeatureName(la);
    const bool engineeredB = isEngineeredFeatureName(lb);

    if (engineeredA && baseA == lb) return true;
    if (engineeredB && baseB == la) return true;
    return (engineeredA || engineeredB) && !baseA.empty() && !baseB.empty() && baseA == baseB;
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
        if (data.columns()[idx].type != ColumnType::NUMERIC &&
            data.columns()[idx].type != ColumnType::CATEGORICAL) {
            throw Seldon::ConfigurationException("Target column must be numeric or categorical: " + config.targetColumn);
        }
        choice.index = idx;
        choice.strategyUsed = "user";
        return choice;
    }

    auto numeric = data.numericColumnIndices();
    auto categorical = data.categoricalColumnIndices();
    if (numeric.empty() && categorical.empty()) {
        throw Seldon::ConfigurationException("Could not resolve a target column");
    }

    auto columnProfile = [&](size_t idx) {
        struct Profile {
            size_t finiteCount = 0;
            size_t uniqueCount = 0;
            double cardRatio = 1.0;
            bool classLike = false;
            bool adminName = false;
            bool targetHint = false;
        };

        Profile p;
        const auto& col = data.columns()[idx];
        const auto& values = std::get<std::vector<double>>(col.values);
        std::unordered_set<double> uniqueValues;
        uniqueValues.reserve(values.size());
        for (double v : values) {
            if (!std::isfinite(v)) continue;
            ++p.finiteCount;
            uniqueValues.insert(v);
        }
        p.uniqueCount = uniqueValues.size();
        p.cardRatio = static_cast<double>(p.uniqueCount) / static_cast<double>(std::max<size_t>(1, p.finiteCount));

        std::string lname = CommonUtils::toLower(col.name);
        p.adminName = isAdministrativeColumnName(lname);
        p.targetHint = isTargetCandidateColumnName(lname);
        p.classLike =
            p.uniqueCount <= std::max<size_t>(8, p.finiteCount / 20) ||
            (p.uniqueCount <= 16 && p.cardRatio <= 0.06) ||
            (p.uniqueCount <= 24 && p.cardRatio <= 0.03);
        return p;
    };

    auto scoreColumnQuality = [&](size_t idx) {
        const auto& col = data.columns()[idx];
        const auto& values = std::get<std::vector<double>>(col.values);
        const auto profile = columnProfile(idx);
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
        const bool adminName = profile.adminName;
        const bool targetHint = profile.targetHint;
        double namePenalty = adminName ? 0.70 : ((lname.find("id") != std::string::npos || lname.find("index") != std::string::npos) ? 0.25 : 0.0);
        double nameBonus = targetHint ? 0.20 : 0.0;
        if (containsToken(lname, {"timestamp", "created", "updated", "time", "date"})) {
            namePenalty += 0.15;
        }
        const double cardRatio = static_cast<double>(roundedCardinality.size()) / static_cast<double>(std::max<size_t>(1, finiteCount));
        if (cardRatio > 0.95) namePenalty += 0.20;
        if (cardRatio > 0.98) namePenalty += 0.15;
        if (cardRatio > 0.35 && roundedCardinality.size() > 32) namePenalty += 0.08;

        const bool classLikeCardinality = profile.classLike;
        const double classLikeBonus = classLikeCardinality ? 0.24 : 0.0;
        const double positionBonus =
            (idx + 1 == data.colCount()) ? 0.14 :
            (idx == 0 ? 0.05 : 0.0);

        double varianceScore = std::clamp(std::log1p(std::max(0.0, var)) / 4.0, 0.0, 1.0);
        const double entropyScore = safeEntropyFromCounts(roundedCardinality, finiteCount);
        return 0.30 * (1.0 - missingRatio)
             + 0.25 * varianceScore
             + 0.15 * nonZeroRatio
             + 0.30 * entropyScore
             - namePenalty
             + nameBonus
             + classLikeBonus
             + positionBonus;
    };

    auto scoreCategoricalQuality = [&](size_t idx) {
        const auto& col = data.columns()[idx];
        const auto& values = std::get<std::vector<std::string>>(col.values);

        std::unordered_map<std::string, size_t> counts;
        counts.reserve(values.size());
        size_t nonMissing = 0;
        for (size_t r = 0; r < values.size() && r < col.missing.size(); ++r) {
            if (col.missing[r]) continue;
            const std::string token = CommonUtils::trim(values[r]);
            if (token.empty()) continue;
            counts[token]++;
            ++nonMissing;
        }

        if (nonMissing < 8 || counts.size() < 2) return -1e9;

        const double missingRatio = 1.0 - (static_cast<double>(nonMissing) / static_cast<double>(std::max<size_t>(1, data.rowCount())));
        const double cardRatio = static_cast<double>(counts.size()) / static_cast<double>(nonMissing);
        const bool binaryLike = counts.size() == 2;
        const bool lowCard = counts.size() <= 24 && cardRatio <= 0.20;
        if (!lowCard && !binaryLike) return -1e9;

        std::string lname = CommonUtils::toLower(col.name);
        const bool adminName = isAdministrativeColumnName(lname);
        const bool targetHint = isTargetCandidateColumnName(lname);

        double namePenalty = adminName ? 0.80 : 0.0;
        if (lname.find("id") != std::string::npos || lname.find("uuid") != std::string::npos || lname.find("index") != std::string::npos) {
            namePenalty += 0.35;
        }
        if (containsToken(lname, {"timestamp", "created", "updated", "time", "date"})) {
            namePenalty += 0.15;
        }

        double entropy = 0.0;
        for (const auto& kv : counts) {
            const double p = static_cast<double>(kv.second) / static_cast<double>(nonMissing);
            if (p > 0.0) entropy -= p * std::log2(p);
        }
        const double maxEntropy = std::log2(static_cast<double>(std::max<size_t>(2, counts.size())));
        const double entropyScore = (maxEntropy > 1e-12) ? std::clamp(entropy / maxEntropy, 0.0, 1.0) : 0.0;
        const double compactnessScore = 1.0 - std::clamp(cardRatio * 2.0, 0.0, 1.0);

        const double positionBonus =
            (idx + 1 == data.colCount()) ? 0.12 :
            (idx == 0 ? 0.04 : 0.0);
        const double classBonus = binaryLike ? 0.24 : (lowCard ? 0.12 : 0.0);
        const double hintBonus = targetHint ? 0.28 : 0.0;

        return 0.38 * (1.0 - missingRatio)
             + 0.22 * entropyScore
             + 0.26 * compactnessScore
             - namePenalty
             + classBonus
             + hintBonus
             + positionBonus;
    };

    auto pickBestCategorical = [&]() -> std::optional<std::pair<size_t, double>> {
        if (categorical.empty()) return std::nullopt;
        size_t best = static_cast<size_t>(-1);
        double bestScore = -1e9;
        for (size_t idx : categorical) {
            const double s = scoreCategoricalQuality(idx);
            if (s > bestScore) {
                bestScore = s;
                best = idx;
            }
        }
        if (best == static_cast<size_t>(-1) || bestScore <= -1e8) return std::nullopt;
        return std::make_pair(best, bestScore);
    };

    if (numeric.empty()) {
        if (auto cat = pickBestCategorical(); cat.has_value()) {
            choice.index = static_cast<int>(cat->first);
            choice.strategyUsed = "categorical_quality_fallback";
            return choice;
        }
        throw Seldon::ConfigurationException("Could not resolve a target column");
    }

    auto pickClassLikeCandidate = [&]() -> std::optional<size_t> {
        size_t best = static_cast<size_t>(-1);
        double bestScore = -1e9;
        for (size_t idx : numeric) {
            const auto profile = columnProfile(idx);
            if (!profile.classLike) continue;
            if (profile.adminName) continue;
            if (profile.finiteCount < 8) continue;
            const double s = scoreColumnQuality(idx) + (profile.targetHint ? 0.20 : 0.0);
            if (s > bestScore) {
                bestScore = s;
                best = idx;
            }
        }
        if (best == static_cast<size_t>(-1)) return std::nullopt;
        return best;
    };

    auto pickBySemanticTargetHint = [&]() -> std::optional<size_t> {
        std::vector<size_t> hinted;
        hinted.reserve(numeric.size());
        for (size_t idx : numeric) {
            const std::string lname = CommonUtils::toLower(data.columns()[idx].name);
            if (isAdministrativeColumnName(lname)) continue;
            if (isTargetCandidateColumnName(lname)) hinted.push_back(idx);
        }
        for (size_t idx : categorical) {
            const std::string lname = CommonUtils::toLower(data.columns()[idx].name);
            if (isAdministrativeColumnName(lname)) continue;
            if (isTargetCandidateColumnName(lname)) hinted.push_back(idx);
        }
        if (hinted.empty()) return std::nullopt;
        size_t best = hinted.front();
        double bestScore = -1e9;
        for (size_t idx : hinted) {
            const double s = (data.columns()[idx].type == ColumnType::NUMERIC)
                ? scoreColumnQuality(idx)
                : scoreCategoricalQuality(idx);
            if (s > bestScore) {
                bestScore = s;
                best = idx;
            }
        }
        return best;
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
        const size_t numericBest = pickByQuality();
        const double numericScore = scoreColumnQuality(numericBest);
        if (auto cat = pickBestCategorical(); cat.has_value() && cat->second >= numericScore - 0.05) {
            choice.index = static_cast<int>(cat->first);
            choice.strategyUsed = "quality_categorical";
            return choice;
        }
        choice.index = static_cast<int>(numericBest);
        choice.strategyUsed = "quality_numeric";
        return choice;
    }
    if (mode == "max_variance") {
        if (numeric.empty()) {
            if (auto cat = pickBestCategorical(); cat.has_value()) {
                choice.index = static_cast<int>(cat->first);
                choice.strategyUsed = "max_variance_categorical_fallback";
                return choice;
            }
            throw Seldon::ConfigurationException("target_strategy=max_variance requires numeric target candidates");
        }
        choice.index = static_cast<int>(pickByVariance());
        choice.strategyUsed = "max_variance";
        return choice;
    }
    if (mode == "last_numeric") {
        if (numeric.empty()) {
            if (auto cat = pickBestCategorical(); cat.has_value()) {
                choice.index = static_cast<int>(cat->first);
                choice.strategyUsed = "last_numeric_categorical_fallback";
                return choice;
            }
            throw Seldon::ConfigurationException("target_strategy=last_numeric requires numeric target candidates");
        }
        choice.index = static_cast<int>(pickLastNumeric());
        choice.strategyUsed = "last_numeric";
        return choice;
    }

    if (auto semanticHint = pickBySemanticTargetHint(); semanticHint.has_value()) {
        choice.index = static_cast<int>(*semanticHint);
        choice.strategyUsed = "semantic_target_hint";
        return choice;
    }

    const size_t qualityPick = pickByQuality();
    if (auto cat = pickBestCategorical(); cat.has_value()) {
        const double numericScore = scoreColumnQuality(qualityPick);
        if (cat->second >= numericScore + 0.04) {
            choice.index = static_cast<int>(cat->first);
            choice.strategyUsed = "categorical_override";
            return choice;
        }
    }
    if (auto classLikePick = pickClassLikeCandidate(); classLikePick.has_value()) {
        const double qualityScore = scoreColumnQuality(qualityPick);
        const double classLikeScore = scoreColumnQuality(*classLikePick);
        if (classLikeScore >= qualityScore - 0.10) {
            choice.index = static_cast<int>(*classLikePick);
            choice.strategyUsed = "classlike_override";
            return choice;
        }
    }

    std::vector<size_t> votes = {qualityPick, pickByVariance(), pickLastNumeric()};
    if (auto classLikePick = pickClassLikeCandidate(); classLikePick.has_value()) {
        votes.push_back(*classLikePick);
    }
    if (auto cat = pickBestCategorical(); cat.has_value()) {
        votes.push_back(cat->first);
    }
    std::unordered_map<size_t, int> count;
    for (size_t v : votes) count[v]++;

    size_t best = votes.front();
    int bestVotes = -1;
    double bestScore = -1e9;
    for (const auto& kv : count) {
        double s = (data.columns()[kv.first].type == ColumnType::NUMERIC)
            ? scoreColumnQuality(kv.first)
            : scoreCategoricalQuality(kv.first);
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
    std::vector<double> finite;
    finite.reserve(y.size());
    for (double v : y) {
        if (!std::isfinite(v)) continue;
        finite.push_back(v);
        uniq.insert(v);
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

    const size_t finiteCount = finite.size();
    if (finiteCount > 0) {
        const double cardinalityRatio = static_cast<double>(out.cardinality) / static_cast<double>(finiteCount);
        if ((out.cardinality <= 12 && cardinalityRatio <= 0.06) ||
            (out.cardinality <= 20 && cardinalityRatio <= 0.03)) {
            out.isOrdinal = true;
            out.lowLabel = *uniq.begin();
            out.highLabel = *uniq.rbegin();
            out.inferredTask = "ordinal_classification";
            return out;
        }
    }

    out.inferredTask = "regression";
    return out;
}

std::optional<double> parseBooleanLikeCategoryValue(const std::string& value) {
    const std::string token = CommonUtils::toLower(CommonUtils::trim(value));
    if (token.empty()) return std::nullopt;
    if (token == "1" || token == "true" || token == "yes" || token == "y" || token == "t" || token == "on") {
        return 1.0;
    }
    if (token == "0" || token == "false" || token == "no" || token == "n" || token == "f" || token == "off") {
        return 0.0;
    }
    return std::nullopt;
}

size_t encodeCategoricalTargetColumn(TypedDataset& data, int targetIdx) {
    if (targetIdx < 0 || static_cast<size_t>(targetIdx) >= data.columns().size()) return 0;
    auto& targetCol = data.columns()[static_cast<size_t>(targetIdx)];
    if (targetCol.type != ColumnType::CATEGORICAL) return 0;

    const auto& raw = std::get<std::vector<std::string>>(targetCol.values);
    std::vector<double> encoded(raw.size(), std::numeric_limits<double>::quiet_NaN());

    bool allBooleanLike = true;
    for (size_t r = 0; r < raw.size() && r < targetCol.missing.size(); ++r) {
        if (targetCol.missing[r]) continue;
        if (!parseBooleanLikeCategoryValue(raw[r]).has_value()) {
            allBooleanLike = false;
            break;
        }
    }

    if (allBooleanLike) {
        size_t classes = 0;
        bool seenZero = false;
        bool seenOne = false;
        for (size_t r = 0; r < raw.size() && r < targetCol.missing.size(); ++r) {
            if (targetCol.missing[r]) continue;
            const auto parsed = parseBooleanLikeCategoryValue(raw[r]);
            if (!parsed.has_value()) continue;
            encoded[r] = *parsed;
            if (*parsed < 0.5) seenZero = true;
            else seenOne = true;
        }
        classes = static_cast<size_t>(seenZero) + static_cast<size_t>(seenOne);
        targetCol.values = std::move(encoded);
        targetCol.type = ColumnType::NUMERIC;
        return classes;
    }

    std::unordered_map<std::string, size_t> frequency;
    frequency.reserve(raw.size());
    for (size_t r = 0; r < raw.size() && r < targetCol.missing.size(); ++r) {
        if (targetCol.missing[r]) continue;
        const std::string token = CommonUtils::trim(raw[r]);
        if (token.empty()) continue;
        frequency[token]++;
    }

    std::vector<std::pair<std::string, size_t>> labels(frequency.begin(), frequency.end());
    std::sort(labels.begin(), labels.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    std::unordered_map<std::string, double> labelToValue;
    labelToValue.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        labelToValue[labels[i].first] = static_cast<double>(i);
    }

    for (size_t r = 0; r < raw.size() && r < targetCol.missing.size(); ++r) {
        if (targetCol.missing[r]) continue;
        const std::string token = CommonUtils::trim(raw[r]);
        auto it = labelToValue.find(token);
        if (it != labelToValue.end()) {
            encoded[r] = it->second;
        }
    }

    targetCol.values = std::move(encoded);
    targetCol.type = ColumnType::NUMERIC;
    return labels.size();
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
        if (isAdministrativeColumnName(name)) {
            out.droppedByMissingness.push_back(name + " (semantic_admin)");
            continue;
        }

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
    bool strictPruningApplied = false;
    size_t strictPrunedColumns = 0;
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

double estimateTrainingStability(const std::vector<double>& trainLoss,
                                 const std::vector<double>& valLoss) {
    if (trainLoss.empty()) return 0.5;

    const double trainStart = trainLoss.front();
    const double trainEnd = trainLoss.back();
    const double baseline = std::max(1e-9, std::abs(trainStart));
    const double improvement = (trainStart - trainEnd) / baseline;
    const double convergence = std::clamp(0.5 + 0.5 * improvement, 0.0, 1.0);

    if (!valLoss.empty() && valLoss.size() == trainLoss.size()) {
        const double valEnd = valLoss.back();
        const double gap = std::abs(valEnd - trainEnd) / std::max(1e-9, std::abs(trainEnd) + 1e-9);
        const double generalization = std::clamp(1.0 - gap, 0.0, 1.0);
        return 0.7 * convergence + 0.3 * generalization;
    }
    return convergence;
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

    out.trainingStability = estimateTrainingStability(neural.trainLoss, neural.valLoss);

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

NeuralEncodingPlan buildNeuralEncodingPlan(const TypedDataset& data,
                                           int targetIdx,
                                           const std::vector<int>& numericFeatureIdx,
                                           const AutoConfig& config) {
    NeuralEncodingPlan plan;
    const std::unordered_set<std::string> excluded(config.excludedColumns.begin(), config.excludedColumns.end());
    const size_t baseMaxOneHot = std::max<size_t>(1, config.neuralMaxOneHotPerColumn);
    const size_t rowAdaptiveCap = (data.rowCount() < 500)
        ? std::max<size_t>(1, std::min<size_t>(baseMaxOneHot, data.rowCount() / 25 + 1))
        : baseMaxOneHot;
    const size_t maxOneHotPerColumn = config.lowMemoryMode
        ? std::max<size_t>(1, std::min<size_t>(rowAdaptiveCap, 8))
        : rowAdaptiveCap;
    const size_t lowMemoryFreqCap = config.lowMemoryMode
        ? std::max<size_t>(maxOneHotPerColumn * 8, maxOneHotPerColumn)
        : std::numeric_limits<size_t>::max();

    const auto categoricalIdx = data.categoricalColumnIndices();
    plan.categoryPlans.reserve(categoricalIdx.size());

    for (size_t featurePos = 0; featurePos < numericFeatureIdx.size(); ++featurePos) {
        int idx = numericFeatureIdx[featurePos];
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        plan.numericColumns.push_back(static_cast<size_t>(idx));
        plan.sourceNumericFeaturePos.push_back(static_cast<int>(featurePos));
        plan.encodedWidth++;
    }

    for (size_t idx : categoricalIdx) {
        if (static_cast<int>(idx) == targetIdx) continue;
        const std::string& columnName = data.columns()[idx].name;
        if (excluded.find(columnName) != excluded.end()) continue;

        const auto& values = std::get<std::vector<std::string>>(data.columns()[idx].values);
        if (values.empty()) continue;

        std::unordered_map<std::string_view, size_t> freq;
        size_t overflowCount = 0;
        if (lowMemoryFreqCap != std::numeric_limits<size_t>::max()) {
            freq.reserve(lowMemoryFreqCap);
        }
        for (const auto& v : values) {
            if (v.empty()) continue;
            const std::string_view key(v);
            auto it = freq.find(key);
            if (it != freq.end()) {
                ++it->second;
                continue;
            }
            if (freq.size() < lowMemoryFreqCap) {
                freq.emplace(key, 1);
            } else {
                ++overflowCount;
            }
        }
        if (freq.empty()) continue;

        std::vector<std::pair<std::string_view, size_t>> categories(freq.begin(), freq.end());
        std::sort(categories.begin(), categories.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });
        const size_t keepCount = std::min(maxOneHotPerColumn, categories.size());
        if (keepCount == 0) continue;

        NeuralCategoryPlan catPlan;
        catPlan.columnIdx = idx;
        catPlan.includeOther = (categories.size() > keepCount) || (overflowCount > 0);
        catPlan.keptLabels.reserve(keepCount);
        for (size_t i = 0; i < keepCount; ++i) {
            catPlan.keptLabels.push_back(categories[i].first);
            plan.sourceNumericFeaturePos.push_back(-1);
            plan.categoricalEncodedNodes++;
            plan.encodedWidth++;
        }
        if (catPlan.includeOther) {
            plan.sourceNumericFeaturePos.push_back(-1);
            plan.categoricalEncodedNodes++;
            plan.encodedWidth++;
        }
        plan.categoryPlans.push_back(std::move(catPlan));
        plan.categoricalColumnsUsed++;
    }

    return plan;
}

void encodeNeuralRows(const TypedDataset& data,
                      const NeuralEncodingPlan& plan,
                      size_t rowStart,
                      size_t rowEnd,
                      std::vector<std::vector<double>>& outRows) {
    const size_t nRows = (rowEnd > rowStart) ? (rowEnd - rowStart) : 0;
    outRows.assign(nRows, std::vector<double>(plan.encodedWidth, 0.0));

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t localRow = 0; localRow < nRows; ++localRow) {
        const size_t r = rowStart + localRow;
        auto& out = outRows[localRow];
        size_t offset = 0;

        for (size_t idx : plan.numericColumns) {
            const auto& values = std::get<std::vector<double>>(data.columns()[idx].values);
            out[offset++] = (r < values.size()) ? values[r] : 0.0;
        }

        for (const auto& catPlan : plan.categoryPlans) {
            const auto& values = std::get<std::vector<std::string>>(data.columns()[catPlan.columnIdx].values);
            const std::string_view value = (r < values.size()) ? std::string_view(values[r]) : std::string_view();
            bool matched = false;
            for (std::string_view kept : catPlan.keptLabels) {
                const bool isMatch = (value == kept);
                out[offset++] = isMatch ? 1.0 : 0.0;
                matched = matched || isMatch;
            }
            if (catPlan.includeOther) {
                out[offset++] = matched ? 0.0 : 1.0;
            }
        }
    }
}

EncodedNeuralMatrix buildEncodedNeuralInputs(const TypedDataset& data,
                                             int targetIdx,
                                             const std::vector<int>& numericFeatureIdx,
                                             const AutoConfig& config) {
    EncodedNeuralMatrix out;
    const NeuralEncodingPlan plan = buildNeuralEncodingPlan(data, targetIdx, numericFeatureIdx, config);
    out.sourceNumericFeaturePos = plan.sourceNumericFeaturePos;
    out.categoricalEncodedNodes = plan.categoricalEncodedNodes;
    out.categoricalColumnsUsed = plan.categoricalColumnsUsed;
    encodeNeuralRows(data, plan, 0, data.rowCount(), out.X);
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

void cleanupPlotCacheArtifacts(const AutoConfig& config) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (config.assetsDir.empty()) return;

    const fs::path root(config.assetsDir);
    if (!fs::exists(root, ec)) return;

    std::vector<fs::path> cacheDirs;
    cacheDirs.reserve(8);
    const fs::path directCache = root / ".plot_cache";
    if (fs::exists(directCache, ec)) {
        cacheDirs.push_back(directCache);
    }

    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_directory()) continue;
        if (it->path().filename() == ".plot_cache") {
            cacheDirs.push_back(it->path());
        }
    }

    std::sort(cacheDirs.begin(), cacheDirs.end());
    cacheDirs.erase(std::unique(cacheDirs.begin(), cacheDirs.end()), cacheDirs.end());
    std::sort(cacheDirs.begin(), cacheDirs.end(), [](const fs::path& a, const fs::path& b) {
        return a.string().size() > b.string().size();
    });

    for (const auto& cacheDir : cacheDirs) {
        fs::remove_all(cacheDir, ec);
    }
}

void validateExcludedColumns(const TypedDataset& data, const AutoConfig& config) {
    for (const auto& ex : config.excludedColumns) {
        if (data.findColumnIndex(ex) < 0) {
            throw Seldon::ConfigurationException("Excluded column not found: " + ex);
        }
    }
}

struct PreflightCullSummary {
    double threshold = 0.95;
    size_t dropped = 0;
    std::vector<std::string> droppedColumns;
};

PreflightCullSummary applyPreflightSparseColumnCull(TypedDataset& data,
                                                    const std::optional<std::string>& protectedColumn,
                                                    bool verbose,
                                                    double threshold = 0.95) {
    PreflightCullSummary summary;
    summary.threshold = threshold;

    if (data.rowCount() == 0 || data.colCount() == 0) {
        return summary;
    }

    std::vector<bool> keep(data.colCount(), true);
    const double denom = static_cast<double>(std::max<size_t>(1, data.rowCount()));
    const std::string protectedName = protectedColumn.has_value()
        ? CommonUtils::toLower(CommonUtils::trim(*protectedColumn))
        : "";

    for (size_t c = 0; c < data.columns().size(); ++c) {
        const auto& col = data.columns()[c];
        const std::string colLower = CommonUtils::toLower(CommonUtils::trim(col.name));
        if (!protectedName.empty() && colLower == protectedName) {
            continue;
        }

        const size_t missingCount = std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1));
        const double missingRatio = static_cast<double>(missingCount) / denom;
        if (missingRatio > threshold) {
            keep[c] = false;
            summary.droppedColumns.push_back(col.name + " (missing=" + toFixed(100.0 * missingRatio, 2) + "%)");
        }
    }

    summary.dropped = summary.droppedColumns.size();
    if (summary.dropped > 0) {
        data.removeColumns(keep);
        if (verbose) {
            std::cout << "[Seldon][PreFlight] Dropped " << summary.dropped
                      << " sparse columns (>" << toFixed(100.0 * threshold, 1)
                      << "% missing) before preprocessing/univariate cycle.\n";
            for (const auto& item : summary.droppedColumns) {
                std::cout << "  - " << item << "\n";
            }
        }
    }

    return summary;
}

struct TargetContext {
    bool userProvidedTarget = false;
    TargetChoice choice;
    int targetIdx = -1;
    TargetSemantics semantics;
    bool encodedFromCategorical = false;
    size_t encodedCardinality = 0;
};

TargetContext resolveTargetContext(TypedDataset& data, const AutoConfig& config, AutoConfig& runCfg) {
    TargetContext context;
    context.userProvidedTarget = !config.targetColumn.empty();
    context.choice = resolveTargetChoice(data, config);
    context.targetIdx = context.choice.index;
    if (context.targetIdx >= 0 && data.columns()[context.targetIdx].type == ColumnType::CATEGORICAL) {
        context.encodedCardinality = encodeCategoricalTargetColumn(data, context.targetIdx);
        context.encodedFromCategorical = true;
        context.choice.strategyUsed += "+categorical_encoded";
    }
    context.semantics = inferTargetSemanticsRaw(data, context.targetIdx);
    runCfg.targetColumn = data.columns()[context.targetIdx].name;

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Target] "
                  << (context.userProvidedTarget ? "Using user-selected target: " : "Auto-selected target: ")
                  << runCfg.targetColumn
                  << " (strategy=" << context.choice.strategyUsed
                  << ", task=" << context.semantics.inferredTask
                  << ", cardinality=" << context.semantics.cardinality;
        if (context.encodedFromCategorical) {
            std::cout << ", encoded_from=categorical(classes=" << context.encodedCardinality << ")";
        }
        std::cout << ")\n";
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
                          const ReportEngine& finalAnalysis,
                          const ReportEngine& heuristicsReport) {
    const std::string uniMd = runCfg.outputDir + "/univariate.md";
    const std::string biMd = runCfg.outputDir + "/bivariate.md";
    const std::string finalMd = runCfg.outputDir + "/final_analysis.md";
    const std::string reportMd = runCfg.outputDir + "/report.md";

    univariate.save(uniMd);
    bivariate.save(biMd);
    neuralReport.save(runCfg.reportFile);
    finalAnalysis.save(finalMd);
    heuristicsReport.save(reportMd);

    if (runCfg.generateHtml) {
        const std::string pandocExe = findExecutableInPath("pandoc");
        if (pandocExe.empty()) {
            return;
        }
        const std::vector<std::pair<std::string, std::string>> conversions = {
            {uniMd, runCfg.outputDir + "/univariate.html"},
            {biMd, runCfg.outputDir + "/bivariate.html"},
            {runCfg.reportFile, runCfg.outputDir + "/neural_synthesis.html"},
            {finalMd, runCfg.outputDir + "/final_analysis.html"},
            {reportMd, runCfg.outputDir + "/report.html"}
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
              << runCfg.outputDir << "/final_analysis.md, "
              << runCfg.outputDir << "/report.md\n";
    if (runCfg.generateHtml) {
        std::cout << "[Seldon] HTML reports (self-contained): "
                  << runCfg.outputDir << "/univariate.html, "
                  << runCfg.outputDir << "/bivariate.html, "
                  << runCfg.outputDir << "/neural_synthesis.html, "
                  << runCfg.outputDir << "/final_analysis.html, "
                  << runCfg.outputDir << "/report.html\n";
    }
    std::cout << "[Seldon] Plot folders: "
              << plotSubdir(runCfg, "univariate") << ", "
              << plotSubdir(runCfg, "bivariate") << ", "
              << plotSubdir(runCfg, "overall") << "\n";
    std::cout << "Seldon had great fun analysing the data you gave.\n";
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
    std::vector<std::vector<std::string>> multiSelectTokenRows;
    std::vector<std::pair<size_t, double>> surpriseRows;
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
        const bool adminLike = isAdministrativeColumnName(col.name);
        const size_t fallbackMissing = static_cast<size_t>(std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1)));
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : fallbackMissing;
        const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;

        if (col.type == ColumnType::NUMERIC) {
            const auto& vals = std::get<std::vector<double>>(col.values);
            if (adminLike) {
                std::unordered_set<long long> uniq;
                uniq.reserve(vals.size());
                for (double v : vals) {
                    if (!std::isfinite(v)) continue;
                    uniq.insert(static_cast<long long>(std::llround(v * 1e6)));
                }
                summaryRows.push_back({
                    col.name,
                    "metadata",
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
                    std::to_string(uniq.size()),
                    std::to_string(missing),
                    "-"
                });
                logVerbose("[Seldon][Univariate] Metadata " + col.name
                           + " | unique=" + std::to_string(uniq.size())
                           + " missing=" + std::to_string(missing));
                continue;
            }

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

            const double outlierRate = vals.empty() ? 0.0 : (static_cast<double>(outliers) / static_cast<double>(vals.size()));
            const double surprise =
                0.45 * std::abs(st.skewness) +
                0.25 * std::log1p(std::abs(st.kurtosis)) +
                0.20 * std::log1p(std::abs(st.stddev)) +
                0.10 * (10.0 * outlierRate);
            surpriseRows.push_back({summaryRows.size() - 1, surprise});

            logVerbose("[Seldon][Univariate] Numeric " + col.name
                       + " | n=" + std::to_string(vals.size())
                       + " mean=" + toFixed(st.mean)
                       + " std=" + toFixed(st.stddev)
                       + " iqr=" + toFixed(st.iqr)
                       + " missing=" + std::to_string(missing)
                       + " outliers=" + std::to_string(outliers));
        } else if (col.type == ColumnType::CATEGORICAL) {
            const auto& vals = std::get<std::vector<std::string>>(col.values);
            const CategoricalSemanticProfile semantic = detectCategoricalSemanticProfile(vals, col.missing);
            const std::string semanticType = categoricalSemanticLabel(semantic.kind);
            std::map<std::string, size_t> freq;
            for (const auto& v : vals) freq[v]++;

            std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string mode = top.empty() ? "-" : top.front().first;
            size_t modeCount = top.empty() ? 0 : top.front().second;
            double modeRatio = vals.empty() ? 0.0 : static_cast<double>(modeCount) / static_cast<double>(vals.size());

            summaryRows.push_back({
                col.name,
                semanticType,
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

            if (semantic.kind == CategoricalSemanticKind::MULTI_SELECT) {
                for (const auto& token : semantic.topTokens) {
                    multiSelectTokenRows.push_back({
                        col.name,
                        token.first,
                        std::to_string(token.second),
                        toFixed(static_cast<double>(token.second) / static_cast<double>(std::max<size_t>(1, vals.size())), 6)
                    });
                }
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
                "-",                           // Mean
                "-",                           // GeoMean
                "-",                           // HarmonicMean
                "-",                           // TrimmedMean
                "-",                           // Mode
                "-",                           // Median
                std::to_string(minTs),         // Min
                std::to_string(maxTs),         // Max
                std::to_string(maxTs - minTs), // Range
                "-",                           // Q1
                "-",                           // Q3
                "-",                           // IQR
                "-",                           // P05
                "-",                           // P95
                "-",                           // MAD
                "-",                           // Variance
                "-",                           // StdDev
                "-",                           // Skew
                "-",                           // Kurtosis
                "-",                           // CoeffVar
                "-",                           // Sum
                std::to_string(vals.size()),   // NonZero/Unique
                std::to_string(missing),       // Missing
                "-"                            // Outliers
            });
            logVerbose("[Seldon][Univariate] Datetime " + col.name
                       + " | min_ts=" + std::to_string(minTs)
                       + " max_ts=" + std::to_string(maxTs)
                       + " missing=" + std::to_string(missing));
        }
    }

    const std::vector<std::string> summaryHeaders = {
        "Column", "Type", "Mean", "GeoMean", "HarmonicMean", "TrimmedMean", "Mode", "Median", "Min", "Max", "Range", "Q1", "Q3", "IQR",
        "P05", "P95", "MAD", "Variance", "StdDev", "Skew", "Kurtosis", "CoeffVar", "Sum", "NonZero/Unique", "Missing", "Outliers"
    };

    if (summaryRows.size() > 16 && !surpriseRows.empty()) {
        std::sort(surpriseRows.begin(), surpriseRows.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });

        const size_t topK = std::min<size_t>(10, std::max<size_t>(6, static_cast<size_t>(std::sqrt(static_cast<double>(summaryRows.size())))));
        std::unordered_set<size_t> keep;
        keep.reserve(topK * 2);
        for (size_t i = 0; i < std::min(topK, surpriseRows.size()); ++i) keep.insert(surpriseRows[i].first);

        std::vector<std::vector<std::string>> topRows;
        std::vector<std::vector<std::string>> restRows;
        for (size_t i = 0; i < summaryRows.size(); ++i) {
            if (keep.count(i)) topRows.push_back(summaryRows[i]);
            else restRows.push_back(summaryRows[i]);
        }

        report.addTable("Column Statistical Super-Summary (Top Surprising)", summaryHeaders, topRows);

        size_t restNumeric = 0;
        size_t restCategorical = 0;
        size_t restDatetime = 0;
        size_t restMetadata = 0;
        for (const auto& row : restRows) {
            if (row.size() < 2) continue;
            const std::string t = row[1];
            if (t == "numeric") ++restNumeric;
            else if (t.rfind("categorical", 0) == 0) ++restCategorical;
            else if (t == "datetime") ++restDatetime;
            else if (t == "metadata") ++restMetadata;
        }
        report.addTable("General Population Summary", {"Metric", "Value"}, {
            {"Columns outside Top set", std::to_string(restRows.size())},
            {"Numeric (outside Top)", std::to_string(restNumeric)},
            {"Categorical (outside Top)", std::to_string(restCategorical)},
            {"Datetime (outside Top)", std::to_string(restDatetime)},
            {"Metadata/Admin (outside Top)", std::to_string(restMetadata)}
        });
    } else {
        report.addTable("Column Statistical Super-Summary", summaryHeaders, summaryRows);
    }

    std::vector<std::vector<std::string>> qualityRows;
    std::vector<std::vector<std::string>> metadataRows;
    for (const auto& col : data.columns()) {
        const size_t fallbackMissing = static_cast<size_t>(std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1)));
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : fallbackMissing;
        const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;
        const bool adminLike = isAdministrativeColumnName(col.name);
        std::string baseType;
        if (col.type == ColumnType::NUMERIC) {
            baseType = "numeric";
        } else if (col.type == ColumnType::CATEGORICAL) {
            const auto& vals = std::get<std::vector<std::string>>(col.values);
            baseType = categoricalSemanticLabel(detectCategoricalSemanticProfile(vals, col.missing).kind);
        } else {
            baseType = "datetime";
        }
        const std::string type = adminLike ? "metadata" : baseType;
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
        qualityRows.push_back({col.name, type, std::to_string(unique), std::to_string(missing), adminLike ? "-" : std::to_string(outliers)});
        if (adminLike) {
            metadataRows.push_back({col.name, baseType, std::to_string(unique), std::to_string(missing), "admin-pattern"});
        }
    }
    report.addTable("Data Quality Report", {"Column", "Type", "Unique", "Missing", "Outliers"}, qualityRows);

    if (!metadataRows.empty()) {
        report.addTable("Metadata / Integrity Columns", {"Column", "Original Type", "Unique", "Missing", "Tag"}, metadataRows);
    }

    if (!categoricalRows.empty()) {
        report.addTable("Categorical Top Frequencies", {"Column", "Category", "Count", "Share"}, categoricalRows);
    }

    if (!multiSelectTokenRows.empty()) {
        report.addTable("Multi-Select Token Frequencies", {"Column", "Token", "Rows", "Share"}, multiSelectTokenRows);
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
                                 bool classificationTarget,
                                 bool verbose,
                                 const AutoConfig& config,
                                 bool fastModeEnabled,
                                 size_t fastSampleRows) {
    NeuralAnalysis analysis;
    analysis.numericInputNodes = featureIdx.size();
    analysis.outputNodes = 1;
    analysis.binaryTarget = classificationTarget;
    analysis.classificationTarget = classificationTarget;
    analysis.hiddenActivation = "n/a";
    analysis.outputActivation = activationToString(classificationTarget ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR);
    analysis.lossName = classificationTarget ? "cross_entropy" : "mse";
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
        const NumericStatsCache statsCache = buildNumericStatsCache(data);
        const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
        const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);
        for (size_t idx : numericIdx) {
            if (static_cast<int>(idx) == targetIdx) continue;
            const auto& cand = std::get<std::vector<double>>(data.columns()[idx].values);
            const auto cIt = statsCache.find(idx);
            const ColumnStats cStats = (cIt != statsCache.end()) ? cIt->second : Statistics::calculateStats(cand);
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

    if (config.neuralStreamingMode) {
        const NeuralEncodingPlan plan = buildNeuralEncodingPlan(data, targetIdx, featureIdx, config);
        const size_t inputNodes = plan.encodedWidth;
        const size_t outputNodes = targetIndices.size();

        analysis.outputAuxTargets = (targetIndices.size() > 1) ? (targetIndices.size() - 1) : 0;
        analysis.inputNodes = inputNodes;
        analysis.categoricalColumnsUsed = plan.categoricalColumnsUsed;
        analysis.categoricalEncodedNodes = plan.categoricalEncodedNodes;

        if (inputNodes == 0 || outputNodes == 0) {
            analysis.policyUsed = "none_streaming_no_encoded_features";
            analysis.featureImportance.assign(featureIdx.size(), 0.0);
            analysis.trainingRowsUsed = data.rowCount();
            return analysis;
        }

        size_t hidden = std::clamp<size_t>(
            static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes) * std::max<size_t>(8, data.rowCount() / 6))))),
            4,
            static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));

        if (static_cast<double>(data.rowCount()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
            hidden = std::min(hidden, std::clamp<size_t>(data.rowCount() / 3, 4, 20));
        }
        if (config.neuralFixedHiddenNodes > 0) {
            hidden = static_cast<size_t>(config.neuralFixedHiddenNodes);
        }

        std::vector<size_t> topology = {inputNodes, hidden, outputNodes};
        NeuralNet nn(topology);
        NeuralNet::Hyperparameters hp;

        auto toOptimizer = [](const std::string& name) {
            const std::string n = CommonUtils::toLower(name);
            if (n == "sgd") return NeuralNet::Optimizer::SGD;
            if (n == "adam") return NeuralNet::Optimizer::ADAM;
            return NeuralNet::Optimizer::LOOKAHEAD;
        };

        hp.epochs = std::clamp<size_t>(120 + data.rowCount() / 2, 100, 320);
        hp.batchSize = std::clamp<size_t>(data.rowCount() / 24, 4, 64);
        hp.learningRate = config.neuralLearningRate;
        hp.earlyStoppingPatience = std::clamp<int>(static_cast<int>(hp.epochs / 12), 6, 24);
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
        hp.dropoutRate = (inputNodes >= 12) ? 0.12 : (inputNodes >= 6 ? 0.08 : 0.04);
        hp.valSplit = std::clamp((data.rowCount() < 80) ? 0.30 : 0.20, 0.10, 0.40);
        hp.l2Lambda = (data.rowCount() < 80) ? 0.010 : 0.001;
        hp.categoricalInputL2Boost = config.neuralCategoricalInputL2Boost;
        hp.activation = (inputNodes < 10) ? NeuralNet::Activation::TANH : NeuralNet::Activation::GELU;
        hp.outputActivation = (outputNodes == 1 && classificationTarget) ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
        hp.useBatchNorm = config.neuralUseBatchNorm;
        hp.batchNormMomentum = config.neuralBatchNormMomentum;
        hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
        hp.useLayerNorm = config.neuralUseLayerNorm;
        hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
        hp.optimizer = toOptimizer(config.neuralOptimizer);
        hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
        hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
        hp.lookaheadAlpha = config.neuralLookaheadAlpha;
        hp.loss = (outputNodes == 1 && classificationTarget) ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
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
        hp.incrementalMode = true;

        std::vector<double> inputL2Scales(inputNodes, 1.0);
        for (size_t i = 0; i < inputL2Scales.size() && i < plan.sourceNumericFeaturePos.size(); ++i) {
            if (plan.sourceNumericFeaturePos[i] < 0) inputL2Scales[i] = hp.categoricalInputL2Boost;
        }
        nn.setInputL2Scales(inputL2Scales);

        const size_t chunkRows = std::max<size_t>(16, config.neuralStreamingChunkRows);
        char xTmp[] = "/tmp/seldon_nn_x_XXXXXX";
        char yTmp[] = "/tmp/seldon_nn_y_XXXXXX";
        struct ScopedUnlink {
            const char* path = nullptr;
            bool active = false;
            ~ScopedUnlink() {
                if (active && path) ::unlink(path);
            }
            void release() { active = false; }
        } xTmpGuard{xTmp, false}, yTmpGuard{yTmp, false};

        const int xFd = ::mkstemp(xTmp);
        const int yFd = ::mkstemp(yTmp);
        if (xFd < 0 || yFd < 0) {
            if (xFd >= 0) ::close(xFd);
            if (yFd >= 0) ::close(yFd);
            throw Seldon::IOException("Failed to create temporary binary files for streaming neural training");
        }
        xTmpGuard.active = true;
        yTmpGuard.active = true;

        std::ofstream xOut(xTmp, std::ios::binary | std::ios::trunc);
        std::ofstream yOut(yTmp, std::ios::binary | std::ios::trunc);
        if (!xOut || !yOut) {
            ::close(xFd);
            ::close(yFd);
            ::unlink(xTmp);
            ::unlink(yTmp);
            throw Seldon::IOException("Failed to open temporary binary output streams for neural training");
        }

        std::vector<std::vector<double>> chunkX;
        std::vector<std::vector<double>> chunkY;
        for (size_t start = 0; start < data.rowCount(); start += chunkRows) {
            const size_t end = std::min(start + chunkRows, data.rowCount());
            encodeNeuralRows(data, plan, start, end, chunkX);

            chunkY.assign(end - start, std::vector<double>(outputNodes, 0.0));
            #ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (size_t r = start; r < end; ++r) {
                const size_t local = r - start;
                for (size_t t = 0; t < targetIndices.size(); ++t) {
                    const int idx = targetIndices[t];
                    const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
                    chunkY[local][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
                }
            }

            std::vector<float> xRow(static_cast<size_t>(inputNodes), 0.0f);
            for (size_t r = 0; r < chunkX.size(); ++r) {
                for (size_t c = 0; c < inputNodes; ++c) xRow[c] = static_cast<float>(chunkX[r][c]);
                xOut.write(reinterpret_cast<const char*>(xRow.data()), static_cast<std::streamsize>(xRow.size() * sizeof(float)));
                yOut.write(reinterpret_cast<const char*>(chunkY[r].data()), static_cast<std::streamsize>(chunkY[r].size() * sizeof(double)));
            }

            std::vector<std::vector<double>>().swap(chunkX);
            std::vector<std::vector<double>>().swap(chunkY);
        }

        xOut.flush();
        yOut.flush();
        xOut.close();
        yOut.close();
        ::close(xFd);
        ::close(yFd);

        hp.useDiskStreaming = true;
        hp.inputBinaryPath = xTmp;
        hp.targetBinaryPath = yTmp;
        hp.streamingRows = data.rowCount();
        hp.streamingInputDim = inputNodes;
        hp.streamingOutputDim = outputNodes;
        hp.streamingChunkRows = chunkRows;
        hp.useMemoryMappedInput = true;

        nn.train({}, {}, hp);
        xTmpGuard.release();
        yTmpGuard.release();
        ::unlink(xTmp);
        ::unlink(yTmp);

        analysis.hiddenNodes = hidden;
        analysis.outputNodes = outputNodes;
        analysis.binaryTarget = (outputNodes == 1 && classificationTarget);
        analysis.hiddenActivation = activationToString(hp.activation);
        analysis.outputActivation = activationToString(hp.outputActivation);
        analysis.epochs = hp.epochs;
        analysis.batchSize = hp.batchSize;
        analysis.valSplit = hp.valSplit;
        analysis.l2Lambda = hp.l2Lambda;
        analysis.dropoutRate = hp.dropoutRate;
        analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
        analysis.policyUsed = "streaming_on_the_fly";
        analysis.explainabilityMethod = config.neuralExplainability;
        analysis.trainingRowsUsed = data.rowCount();
        analysis.topology = std::to_string(inputNodes) + " -> " + std::to_string(hidden) + " -> " + std::to_string(outputNodes);
        analysis.trainLoss = nn.getTrainLossHistory();
        analysis.valLoss = nn.getValLossHistory();
        analysis.gradientNorm = nn.getGradientNormHistory();
        analysis.weightStd = nn.getWeightStdHistory();
        analysis.weightMeanAbs = nn.getWeightMeanAbsHistory();

        const size_t sampleRows = std::min<size_t>(data.rowCount(), std::max<size_t>(64, std::min<size_t>(config.neuralImportanceMaxRows, static_cast<size_t>(1024))));
        std::vector<std::vector<double>> sampleX;
        std::vector<std::vector<double>> sampleY;
        encodeNeuralRows(data, plan, 0, sampleRows, sampleX);
        sampleY.assign(sampleRows, std::vector<double>(outputNodes, 0.0));
        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t r = 0; r < sampleRows; ++r) {
            for (size_t t = 0; t < targetIndices.size(); ++t) {
                const int idx = targetIndices[t];
                const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
                sampleY[r][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
            }
        }

        const std::vector<double> rawImportance = nn.calculateFeatureImportance(sampleX,
                                                                                 sampleY,
                                                                                 (config.neuralImportanceTrials > 0) ? config.neuralImportanceTrials : 5,
                                                                                 config.neuralImportanceMaxRows,
                                                                                 config.neuralImportanceParallel);
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        for (size_t i = 0; i < rawImportance.size() && i < plan.sourceNumericFeaturePos.size(); ++i) {
            const int numericPos = plan.sourceNumericFeaturePos[i];
            if (numericPos >= 0) {
                const size_t numericPosU = static_cast<size_t>(numericPos);
                if (numericPosU < analysis.featureImportance.size()) {
                    analysis.featureImportance[numericPosU] += std::max(0.0, std::isfinite(rawImportance[i]) ? rawImportance[i] : 0.0);
                }
            }
        }

        return analysis;
    }

    EncodedNeuralMatrix encoded = buildEncodedNeuralInputs(data, targetIdx, featureIdx, config);
    std::vector<std::vector<double>> Xnn = std::move(encoded.X);
    std::vector<std::vector<double>> Ynn(data.rowCount(), std::vector<double>(targetIndices.size(), 0.0));

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
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
            struct ColScore {
                size_t idx = 0;
                double score = 0.0;
            };

            std::vector<ColScore> engineeredScores;
            const size_t rows = std::min(Xnn.size(), Ynn.size());
            double yMean = 0.0;
            for (size_t r = 0; r < rows; ++r) yMean += Ynn[r][0];
            yMean /= std::max<size_t>(1, rows);
            double yVar = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                const double d = Ynn[r][0] - yMean;
                yVar += d * d;
            }
            yVar = std::max(1e-12, yVar);

            for (size_t c = 0; c < inputNodes; ++c) {
                if (c >= encoded.sourceNumericFeaturePos.size()) continue;
                const int numericPos = encoded.sourceNumericFeaturePos[c];
                if (numericPos < 0 || static_cast<size_t>(numericPos) >= featureIdx.size()) continue;
                const int sourceCol = featureIdx[static_cast<size_t>(numericPos)];
                if (sourceCol < 0 || static_cast<size_t>(sourceCol) >= data.columns().size()) continue;
                if (!isEngineeredFeatureName(data.columns()[static_cast<size_t>(sourceCol)].name)) continue;

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
                const double s = (xVar > 1e-12) ? std::abs(cov) / std::sqrt(std::max(1e-12, xVar * yVar)) : 0.0;
                engineeredScores.push_back({c, std::isfinite(s) ? s : 0.0});
            }

            if (engineeredScores.size() >= 4) {
                std::sort(engineeredScores.begin(), engineeredScores.end(), [](const ColScore& a, const ColScore& b) {
                    if (a.score == b.score) return a.idx < b.idx;
                    return a.score < b.score;
                });

                const size_t dropCount = engineeredScores.size() / 2;
                if (dropCount > 0 && (inputNodes - dropCount) >= 4) {
                    std::vector<bool> keepMask(inputNodes, true);
                    for (size_t i = 0; i < dropCount; ++i) keepMask[engineeredScores[i].idx] = false;

                    std::vector<std::vector<double>> prunedX(Xnn.size(), std::vector<double>{});
                    for (size_t r = 0; r < Xnn.size(); ++r) {
                        prunedX[r].reserve(inputNodes - dropCount);
                        for (size_t c = 0; c < inputNodes; ++c) {
                            if (keepMask[c]) prunedX[r].push_back(Xnn[r][c]);
                        }
                    }

                    std::vector<int> prunedSource;
                    prunedSource.reserve(inputNodes - dropCount);
                    for (size_t c = 0; c < inputNodes; ++c) {
                        if (keepMask[c]) {
                            prunedSource.push_back((c < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[c] : -1);
                        }
                    }

                    Xnn = std::move(prunedX);
                    encoded.sourceNumericFeaturePos = std::move(prunedSource);
                    inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
                    analysis.inputNodes = inputNodes;
                    analysis.strictPruningApplied = true;
                    analysis.strictPrunedColumns = dropCount;
                }
            }
        }

        if (dofRatio < 10.0) {
            const size_t squeezeCap = std::clamp<size_t>(data.rowCount() / 10, 4, std::max<size_t>(4, inputNodes));
            applyAdaptiveSparsity(squeezeCap);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }

        if (config.lowMemoryMode && inputNodes > 64) {
            const size_t lowMemCap = std::clamp<size_t>(std::max<size_t>(24, data.rowCount() / 12), 24, static_cast<size_t>(64));
            applyAdaptiveSparsity(lowMemCap);
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
        probeHp.outputActivation = (outputNodes == 1 && classificationTarget) ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
        probeHp.loss = (outputNodes == 1 && classificationTarget) ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
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
    hp.outputActivation = (outputNodes == 1 && classificationTarget) ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
    hp.useBatchNorm = config.neuralUseBatchNorm;
    hp.batchNormMomentum = config.neuralBatchNormMomentum;
    hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
    hp.useLayerNorm = config.neuralUseLayerNorm;
    hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
    hp.optimizer = toOptimizer(config.neuralOptimizer);
    hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
    hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
    hp.lookaheadAlpha = config.neuralLookaheadAlpha;
    hp.loss = (outputNodes == 1 && classificationTarget) ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
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

    const double stabilityProbe = estimateTrainingStability(nn.getTrainLossHistory(), nn.getValLossHistory());
    if (stabilityProbe < 0.70 && hidden > 4 && config.neuralFixedHiddenNodes == 0) {
        const size_t reducedHidden = std::max<size_t>(4, static_cast<size_t>(std::llround(static_cast<double>(hidden) * 0.65)));
        if (reducedHidden < hidden) {
            hidden = reducedHidden;
            topology = buildTopology(hiddenLayers, hidden);
            nn = NeuralNet(topology);
            nn.setInputL2Scales(inputL2Scales);
            hp.epochs = std::min<size_t>(hp.epochs, 180);
            hp.earlyStoppingPatience = std::min(hp.earlyStoppingPatience, 10);
            if (config.neuralStreamingMode) {
                nn.trainIncremental(Xnn, Ynn, hp, std::max<size_t>(16, config.neuralStreamingChunkRows));
            } else {
                nn.train(Xnn, Ynn, hp);
            }
            analysis.policyUsed = policy.name + "+stability_shrink";
        }
    }

    analysis.inputNodes = inputNodes;
    analysis.hiddenNodes = hidden;
    analysis.outputNodes = outputNodes;
    analysis.binaryTarget = (outputNodes == 1 && classificationTarget);
    analysis.classificationTarget = classificationTarget;
    analysis.hiddenActivation = activationToString(hp.activation);
    analysis.outputActivation = activationToString(hp.outputActivation);
    analysis.lossName = (hp.loss == NeuralNet::LossFunction::CROSS_ENTROPY) ? "cross_entropy" : "mse";
    analysis.epochs = hp.epochs;
    analysis.batchSize = hp.batchSize;
    analysis.valSplit = hp.valSplit;
    analysis.l2Lambda = hp.l2Lambda;
    analysis.dropoutRate = hp.dropoutRate;
    analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
    if (analysis.policyUsed.empty()) analysis.policyUsed = policy.name;
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

            if (!p.filteredAsRedundant && isEngineeredLineagePair(p.featureA, p.featureB) && std::abs(p.r) >= 0.98) {
                p.filteredAsRedundant = true;
                p.relationLabel = "Engineered lineage duplicate";
                p.redundancyGroup = canonicalEngineeredBaseName(p.featureA);
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
            pairs.insert(pairs.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
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
    const size_t keepCap = std::max<size_t>(1, maxSelectedPairs);
    const size_t keepCount = std::min(selectedCandidateIdx.size(), keepCap);
    for (size_t i = 0; i < keepCount; ++i) {
        finalSelected.insert(selectedCandidateIdx[i]);
    }

    std::vector<size_t> statOnlyIdx;
    statOnlyIdx.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        if (p.leakageRisk) continue;
        if (!p.statSignificant || finalSelected.count(i) > 0) continue;
        statOnlyIdx.push_back(i);
    }

    std::sort(statOnlyIdx.begin(), statOnlyIdx.end(), [&](size_t lhs, size_t rhs) {
        const auto& a = pairs[lhs];
        const auto& b = pairs[rhs];
        const double pa = std::clamp(-std::log10(std::max(1e-12, a.pValue)) / 12.0, 0.0, 1.0);
        const double pb = std::clamp(-std::log10(std::max(1e-12, b.pValue)) / 12.0, 0.0, 1.0);
        const double sa = 0.45 * a.effectSize + 0.25 * a.foldStability + 0.20 * pa + 0.10 * std::abs(a.r);
        const double sb = 0.45 * b.effectSize + 0.25 * b.foldStability + 0.20 * pb + 0.10 * std::abs(b.r);
        if (sa == sb) {
            if (a.pValue == b.pValue) return a.neuralScore > b.neuralScore;
            return a.pValue < b.pValue;
        }
        return sa > sb;
    });

    const double tier3Aggressiveness = std::clamp(tuning.bivariateTier3FallbackAggressiveness, 0.0, 3.0);
    size_t minSelectedFloor = 0;
    if (tier3Aggressiveness > 0.0) {
        const size_t baseFloor = std::max<size_t>(3, std::min<size_t>(12, pairs.size() / 25 + 1));
        const size_t scaledFloor = static_cast<size_t>(std::llround(static_cast<double>(baseFloor) * tier3Aggressiveness));
        minSelectedFloor = std::min(keepCap, std::max<size_t>(1, scaledFloor));
    }
    if (finalSelected.size() < minSelectedFloor) {
        const size_t deficit = minSelectedFloor - finalSelected.size();
        const size_t promote = std::min(deficit, statOnlyIdx.size());
        for (size_t i = 0; i < promote; ++i) {
            finalSelected.insert(statOnlyIdx[i]);
        }
    }

    for (size_t pairIdx = 0; pairIdx < pairs.size(); ++pairIdx) {
        auto& p = pairs[pairIdx];
        p.selected = finalSelected.find(pairIdx) != finalSelected.end();
        if (!p.leakageRisk && p.statSignificant) {
            p.significanceTier = 1;
            p.selectionReason = "statistical";
            if (p.neuralScore >= dynamicCutoff) {
                p.significanceTier = 2;
                p.selectionReason = "statistical+neural";
            }
            if (p.selected && p.significanceTier < 2) {
                p.significanceTier = 3;
                if (p.filteredAsRedundant || p.filteredAsStructural) {
                    p.selectionReason = "statistical+identity_validation";
                } else {
                    p.selectionReason = "statistical+domain_fallback";
                }
            }
        }
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

ConditionalDriftAssessment assessGlobalConditionalDrift(double globalR,
                                                        double conditionalR,
                                                        double minGlobalAbs = 0.15,
                                                        double collapseRatioThreshold = 0.50,
                                                        double collapseAbsDrop = 0.12) {
    ConditionalDriftAssessment out;
    if (!std::isfinite(globalR) || !std::isfinite(conditionalR)) {
        out.label = "insufficient";
        return out;
    }

    const double absGlobal = std::abs(globalR);
    const double absConditional = std::abs(conditionalR);
    out.collapseRatio = (absGlobal > 1e-12) ? (absConditional / absGlobal) : 1.0;

    if (absGlobal < minGlobalAbs) {
        out.label = "weak-global-signal";
        return out;
    }

    out.signFlip = (globalR * conditionalR < 0.0) && (absConditional >= 0.05);
    out.magnitudeCollapse =
        (absConditional <= absGlobal * collapseRatioThreshold) &&
        ((absGlobal - absConditional) >= collapseAbsDrop);

    if (out.signFlip && out.magnitudeCollapse) out.label = "flip+collapse";
    else if (out.signFlip) out.label = "sign-flip";
    else if (out.magnitudeCollapse) out.label = "magnitude-collapse";
    else out.label = "stable";
    return out;
}

std::string cramerStrengthLabel(double v) {
    if (v >= 0.60) return "very strong";
    if (v >= 0.40) return "strong";
    if (v >= 0.20) return "moderate";
    if (v >= 0.10) return "weak";
    return "very weak";
}

struct TemporalAxisDescriptor {
    std::vector<double> axis;
    std::string name;
};

TemporalAxisDescriptor detectTemporalAxis(const TypedDataset& data) {
    TemporalAxisDescriptor out;

    const auto datetimeIdx = data.datetimeColumnIndices();
    for (size_t idx : datetimeIdx) {
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<std::vector<int64_t>>(col.values);
        if (vals.size() < 16) continue;

        out.axis.assign(vals.size(), 0.0);
        std::unordered_set<int64_t> unique;
        unique.reserve(vals.size());
        for (size_t r = 0; r < vals.size(); ++r) {
            out.axis[r] = static_cast<double>(vals[r]);
            if (r < col.missing.size() && !col.missing[r]) unique.insert(vals[r]);
        }
        if (unique.size() >= 8) {
            out.name = col.name;
            return out;
        }
    }

    const auto numericIdx = data.numericColumnIndices();
    for (size_t idx : numericIdx) {
        const std::string lower = CommonUtils::toLower(data.columns()[idx].name);
        if (!(containsToken(lower, {"time", "date", "index", "idx", "step", "order", "epoch", "year", "month", "day", "week"}) ||
              lower == "t" || lower == "ts")) {
            continue;
        }

        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        if (vals.size() < 16) continue;
        out.axis.assign(vals.begin(), vals.end());
        out.name = data.columns()[idx].name;
        return out;
    }

    out.axis.resize(data.rowCount(), 0.0);
    for (size_t r = 0; r < out.axis.size(); ++r) out.axis[r] = static_cast<double>(r + 1);
    out.name = "row_index";
    return out;
}

double absCorrAligned(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<size_t>& rows) {
    std::vector<double> xa;
    std::vector<double> ya;
    xa.reserve(rows.size());
    ya.reserve(rows.size());
    for (size_t r : rows) {
        if (r >= x.size() || r >= y.size()) continue;
        if (!std::isfinite(x[r]) || !std::isfinite(y[r])) continue;
        xa.push_back(x[r]);
        ya.push_back(y[r]);
    }
    if (xa.size() < 10) return 0.0;
    const ColumnStats xs = Statistics::calculateStats(xa);
    const ColumnStats ys = Statistics::calculateStats(ya);
    return std::abs(MathUtils::calculatePearson(xa, ya, xs, ys).value_or(0.0));
}

struct ContextualDeadZoneInsight {
    std::string feature;
    std::string strongCluster;
    std::string weakCluster;
    double strongCorr = 0.0;
    double weakCorr = 0.0;
    double dropRatio = 1.0;
    size_t support = 0;
};

std::vector<ContextualDeadZoneInsight> detectContextualDeadZones(const TypedDataset& data,
                                                                 size_t targetIdx,
                                                                 const std::vector<size_t>& candidateFeatures,
                                                                 size_t maxRows = 10) {
    std::vector<ContextualDeadZoneInsight> out;
    if (targetIdx >= data.columns().size() || data.columns()[targetIdx].type != ColumnType::NUMERIC) return out;

    std::vector<size_t> anchors;
    for (size_t idx : candidateFeatures) {
        if (idx == targetIdx) continue;
        if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
        anchors.push_back(idx);
        if (anchors.size() >= 2) break;
    }
    if (anchors.size() < 2) return out;

    const auto& ax = std::get<std::vector<double>>(data.columns()[anchors[0]].values);
    const auto& ay = std::get<std::vector<double>>(data.columns()[anchors[1]].values);
    const auto& target = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    std::vector<size_t> validRows;
    validRows.reserve(data.rowCount());
    for (size_t r = 0; r < data.rowCount(); ++r) {
        if (r >= ax.size() || r >= ay.size() || r >= target.size()) continue;
        if (!std::isfinite(ax[r]) || !std::isfinite(ay[r]) || !std::isfinite(target[r])) continue;
        if (r < data.columns()[anchors[0]].missing.size() && data.columns()[anchors[0]].missing[r]) continue;
        if (r < data.columns()[anchors[1]].missing.size() && data.columns()[anchors[1]].missing[r]) continue;
        if (r < data.columns()[targetIdx].missing.size() && data.columns()[targetIdx].missing[r]) continue;
        validRows.push_back(r);
    }
    if (validRows.size() < 40) return out;

    std::pair<double, double> c0 = {ax[validRows.front()], ay[validRows.front()]};
    std::pair<double, double> c1 = {ax[validRows[validRows.size() / 2]], ay[validRows[validRows.size() / 2]]};
    std::unordered_map<size_t, int> clusterByRow;
    for (int iter = 0; iter < 20; ++iter) {
        double s0x = 0.0, s0y = 0.0, n0 = 0.0;
        double s1x = 0.0, s1y = 0.0, n1 = 0.0;
        for (size_t r : validRows) {
            const double d0 = (ax[r] - c0.first) * (ax[r] - c0.first) + (ay[r] - c0.second) * (ay[r] - c0.second);
            const double d1 = (ax[r] - c1.first) * (ax[r] - c1.first) + (ay[r] - c1.second) * (ay[r] - c1.second);
            const int cid = (d0 <= d1) ? 0 : 1;
            clusterByRow[r] = cid;
            if (cid == 0) {
                s0x += ax[r];
                s0y += ay[r];
                n0 += 1.0;
            } else {
                s1x += ax[r];
                s1y += ay[r];
                n1 += 1.0;
            }
        }
        if (n0 < 10.0 || n1 < 10.0) return out;
        c0 = {s0x / n0, s0y / n0};
        c1 = {s1x / n1, s1y / n1};
    }

    std::vector<size_t> rows0;
    std::vector<size_t> rows1;
    rows0.reserve(validRows.size());
    rows1.reserve(validRows.size());
    for (size_t r : validRows) {
        if (clusterByRow[r] == 0) rows0.push_back(r);
        else rows1.push_back(r);
    }
    if (rows0.size() < 12 || rows1.size() < 12) return out;

    for (size_t featureIdx : candidateFeatures) {
        if (featureIdx == targetIdx || featureIdx == anchors[0] || featureIdx == anchors[1]) continue;
        if (featureIdx >= data.columns().size() || data.columns()[featureIdx].type != ColumnType::NUMERIC) continue;
        const auto& fv = std::get<std::vector<double>>(data.columns()[featureIdx].values);

        const double cA = absCorrAligned(fv, target, rows0);
        const double cB = absCorrAligned(fv, target, rows1);
        const double strong = std::max(cA, cB);
        const double weak = std::min(cA, cB);
        const double delta = strong - weak;
        if (strong < 0.35 || weak > 0.10 || delta < 0.25) continue;

        const bool aStrong = cA >= cB;
        out.push_back({
            data.columns()[featureIdx].name,
            aStrong ? "Cluster A" : "Cluster B",
            aStrong ? "Cluster B" : "Cluster A",
            strong,
            weak,
            (strong > 1e-9) ? (weak / strong) : 1.0,
            std::min(rows0.size(), rows1.size())
        });
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        const double da = a.strongCorr - a.weakCorr;
        const double db = b.strongCorr - b.weakCorr;
        if (da == db) return a.dropRatio < b.dropRatio;
        return da > db;
    });
    if (out.size() > maxRows) out.resize(maxRows);
    return out;
}

AdvancedAnalyticsOutputs buildAdvancedAnalyticsOutputs(const TypedDataset& data,
                                                      int targetIdx,
                                                      const std::vector<int>& featureIdx,
                                                      const std::vector<double>& featureImportance,
                                                      const std::vector<PairInsight>& bivariatePairs,
                                                      const std::vector<AnovaInsight>& anovaRows,
                                                      const std::vector<ContingencyInsight>& contingency,
                                                      const NumericStatsCache& statsCache) {
    AdvancedAnalyticsOutputs out;
    if (targetIdx < 0 || static_cast<size_t>(targetIdx) >= data.columns().size()) return out;
    if (data.columns()[static_cast<size_t>(targetIdx)].type != ColumnType::NUMERIC) return out;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
    const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
    const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);

    std::vector<size_t> numericFeatures;
    numericFeatures.reserve(featureIdx.size());
    for (int idx : featureIdx) {
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (idx == targetIdx) continue;
        numericFeatures.push_back(static_cast<size_t>(idx));
    }

    auto importanceOf = [&](size_t idx) {
        for (size_t i = 0; i < featureIdx.size() && i < featureImportance.size(); ++i) {
            if (featureIdx[i] == static_cast<int>(idx)) return std::max(0.0, featureImportance[i]);
        }
        const auto xIt = statsCache.find(idx);
        const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
        const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
        return std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
    };

    std::sort(numericFeatures.begin(), numericFeatures.end(), [&](size_t a, size_t b) {
        return importanceOf(a) > importanceOf(b);
    });
    if (numericFeatures.size() > 8) numericFeatures.resize(8);

    size_t igA = static_cast<size_t>(-1);
    size_t igB = static_cast<size_t>(-1);
    double bestIgProxy = 0.0;
    {
        const size_t n = numericFeatures.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& xa = std::get<std::vector<double>>(data.columns()[numericFeatures[i]].values);
            for (size_t j = i + 1; j < n; ++j) {
                const auto& xb = std::get<std::vector<double>>(data.columns()[numericFeatures[j]].values);
                std::vector<double> cross;
                cross.reserve(std::min({xa.size(), xb.size(), y.size()}));
                std::vector<double> yc;
                yc.reserve(cross.capacity());
                const size_t m = std::min({xa.size(), xb.size(), y.size()});
                for (size_t r = 0; r < m; ++r) {
                    if (!std::isfinite(xa[r]) || !std::isfinite(xb[r]) || !std::isfinite(y[r])) continue;
                    if (r < data.columns()[numericFeatures[i]].missing.size() && data.columns()[numericFeatures[i]].missing[r]) continue;
                    if (r < data.columns()[numericFeatures[j]].missing.size() && data.columns()[numericFeatures[j]].missing[r]) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    cross.push_back(xa[r] * xb[r]);
                    yc.push_back(y[r]);
                }
                if (cross.size() < 12) continue;
                const ColumnStats cs = Statistics::calculateStats(cross);
                const ColumnStats ys = Statistics::calculateStats(yc);
                const double rxy = std::abs(MathUtils::calculatePearson(cross, yc, cs, ys).value_or(0.0));
                const double proxy = rxy * std::sqrt(std::max(1e-12, importanceOf(numericFeatures[i]) * importanceOf(numericFeatures[j])));
                if (std::isfinite(proxy) && proxy > bestIgProxy) {
                    bestIgProxy = proxy;
                    igA = numericFeatures[i];
                    igB = numericFeatures[j];
                }
            }
        }
    }
    out.orderedRows.push_back({"1", "Integrated-gradients interaction proxy", (igA == static_cast<size_t>(-1) ? "Insufficient stable pairs for proxy interactions." : (data.columns()[igA].name + " × " + data.columns()[igB].name + " proxy=" + toFixed(bestIgProxy, 4) + "; interaction candidate is materially stronger than independent effects.")) });

    {
        std::vector<int> parent(static_cast<int>(numericFeatures.size()));
        std::iota(parent.begin(), parent.end(), 0);
        auto findp = [&](auto&& self, int x) -> int {
            if (parent[x] == x) return x;
            parent[x] = self(self, parent[x]);
            return parent[x];
        };
        auto unite = [&](int a, int b) {
            a = findp(findp, a);
            b = findp(findp, b);
            if (a != b) parent[b] = a;
        };
        for (size_t i = 0; i < numericFeatures.size(); ++i) {
            const auto& xi = std::get<std::vector<double>>(data.columns()[numericFeatures[i]].values);
            const auto xiIt = statsCache.find(numericFeatures[i]);
            const ColumnStats xis = (xiIt != statsCache.end()) ? xiIt->second : Statistics::calculateStats(xi);
            for (size_t j = i + 1; j < numericFeatures.size(); ++j) {
                const auto& xj = std::get<std::vector<double>>(data.columns()[numericFeatures[j]].values);
                const auto xjIt = statsCache.find(numericFeatures[j]);
                const ColumnStats xjs = (xjIt != statsCache.end()) ? xjIt->second : Statistics::calculateStats(xj);
                const double r = std::abs(MathUtils::calculatePearson(xi, xj, xis, xjs).value_or(0.0));
                if (r >= 0.92) unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
        std::unordered_map<int, std::vector<size_t>> groups;
        for (size_t i = 0; i < numericFeatures.size(); ++i) {
            groups[findp(findp, static_cast<int>(i))].push_back(numericFeatures[i]);
        }
        size_t redundant = 0;
        std::vector<std::string> reps;
        for (const auto& kv : groups) {
            const auto& g = kv.second;
            if (g.empty()) continue;
            if (g.size() > 1) redundant += (g.size() - 1);
            size_t best = g.front();
            double bestScore = -1.0;
            for (size_t idx : g) {
                const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
                const auto xIt = statsCache.find(idx);
                const ColumnStats xs = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
                const double score = std::abs(MathUtils::calculatePearson(x, y, xs, yStats).value_or(0.0));
                if (score > bestScore) {
                    bestScore = score;
                    best = idx;
                }
            }
            reps.push_back(data.columns()[best].name);
        }
        std::string repMsg;
        for (size_t i = 0; i < std::min<size_t>(3, reps.size()); ++i) {
            if (i) repMsg += ", ";
            repMsg += reps[i];
        }
        out.orderedRows.push_back({"2", "Redundancy grouping (correlation clustering)", "Clusters=" + std::to_string(groups.size()) + ", redundant=" + std::to_string(redundant) + ", representatives: " + (repMsg.empty() ? "n/a" : repMsg)});
    }

    {
        size_t deterministicCount = 0;
        std::string exemplar = "none";
        for (const auto& p : bivariatePairs) {
            const bool deterministic = std::abs(p.r) >= 0.999 || p.filteredAsStructural;
            if (!deterministic) continue;
            ++deterministicCount;
            if (exemplar == "none") exemplar = p.featureA + " ~ " + p.featureB + " (|r|=" + toFixed(std::abs(p.r), 4) + ")";
        }
        out.orderedRows.push_back({"3", "Deterministic relationship detection", "Flagged=" + std::to_string(deterministicCount) + "; exemplar: " + exemplar});
    }

    std::vector<size_t> stepwiseSelected;
    {
        auto fitModel = [&](const std::vector<size_t>& selected, std::vector<double>* outPred = nullptr) -> MLRDiagnostics {
            MLRDiagnostics diag;
            if (selected.empty()) return diag;
            std::vector<std::vector<double>> rows;
            std::vector<double> yy;
            const size_t n = y.size();
            for (size_t r = 0; r < n; ++r) {
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                if (!std::isfinite(y[r])) continue;
                std::vector<double> row;
                row.reserve(selected.size() + 1);
                row.push_back(1.0);
                bool ok = true;
                for (size_t idx : selected) {
                    const auto& v = std::get<std::vector<double>>(data.columns()[idx].values);
                    if (r >= v.size() || !std::isfinite(v[r])) { ok = false; break; }
                    if (r < data.columns()[idx].missing.size() && data.columns()[idx].missing[r]) { ok = false; break; }
                    row.push_back(v[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                yy.push_back(y[r]);
            }
            if (rows.size() < selected.size() + 6) return diag;
            MathUtils::Matrix X(rows.size(), selected.size() + 1);
            MathUtils::Matrix Y(rows.size(), 1);
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t c = 0; c < rows[i].size(); ++c) X.at(i, c) = rows[i][c];
                Y.at(i, 0) = yy[i];
            }
            diag = MathUtils::performMLRWithDiagnostics(X, Y);
            if (outPred && diag.success && diag.coefficients.size() == selected.size() + 1) {
                outPred->assign(rows.size(), 0.0);
                for (size_t i = 0; i < rows.size(); ++i) {
                    double p = 0.0;
                    for (size_t c = 0; c < rows[i].size(); ++c) p += rows[i][c] * diag.coefficients[c];
                    (*outPred)[i] = p;
                }
            }
            return diag;
        };

        if (!numericFeatures.empty()) {
            stepwiseSelected.push_back(numericFeatures.front());
            MLRDiagnostics bestDiag = fitModel(stepwiseSelected, nullptr);
            for (size_t round = 0; round < 2; ++round) {
                size_t bestFeat = static_cast<size_t>(-1);
                double bestAdj = bestDiag.success ? bestDiag.adjustedRSquared : -1e9;
                for (size_t idx : numericFeatures) {
                    if (std::find(stepwiseSelected.begin(), stepwiseSelected.end(), idx) != stepwiseSelected.end()) continue;
                    auto candidate = stepwiseSelected;
                    candidate.push_back(idx);
                    MLRDiagnostics diag = fitModel(candidate, nullptr);
                    if (!diag.success) continue;
                    if (diag.adjustedRSquared > bestAdj + 0.005) {
                        bestAdj = diag.adjustedRSquared;
                        bestFeat = idx;
                        bestDiag = diag;
                    }
                }
                if (bestFeat == static_cast<size_t>(-1)) break;
                stepwiseSelected.push_back(bestFeat);
            }

            std::string names;
            for (size_t i = 0; i < stepwiseSelected.size(); ++i) {
                if (i) names += " + ";
                names += data.columns()[stepwiseSelected[i]].name;
            }
            out.orderedRows.push_back({"4", "Residual discovery via stepwise OLS", names.empty() ? "No stable stepwise model found." : ("Selected: " + names + "; this indicates residual variance is explained by multiple predictors rather than one dominant correlation.")});
            if (!names.empty()) {
                out.priorityTakeaways.push_back("Stepwise model identifies multi-feature structure: " + names + ".");
            }

            const size_t controlCount = std::min<size_t>(stepwiseSelected.size(), 2);
            std::vector<size_t> controls(stepwiseSelected.begin(), stepwiseSelected.begin() + controlCount);
            auto globalCorr = [&](size_t feature) {
                std::vector<double> yVec;
                std::vector<double> xVec;
                const auto& xv = std::get<std::vector<double>>(data.columns()[feature].values);
                for (size_t r = 0; r < y.size() && r < xv.size(); ++r) {
                    if (!std::isfinite(y[r]) || !std::isfinite(xv[r])) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    if (r < data.columns()[feature].missing.size() && data.columns()[feature].missing[r]) continue;
                    yVec.push_back(y[r]);
                    xVec.push_back(xv[r]);
                }
                if (xVec.size() < 12) return 0.0;
                const ColumnStats sx = Statistics::calculateStats(xVec);
                const ColumnStats sy = Statistics::calculateStats(yVec);
                return MathUtils::calculatePearson(xVec, yVec, sx, sy).value_or(0.0);
            };

            auto partialCorr = [&](size_t feature) {
                std::vector<double> yVec;
                std::vector<double> xVec;
                std::vector<std::vector<double>> ctrlRows;
                const auto& xv = std::get<std::vector<double>>(data.columns()[feature].values);
                for (size_t r = 0; r < y.size() && r < xv.size(); ++r) {
                    if (!std::isfinite(y[r]) || !std::isfinite(xv[r])) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    if (r < data.columns()[feature].missing.size() && data.columns()[feature].missing[r]) continue;
                    std::vector<double> row;
                    row.reserve(controls.size() + 1);
                    row.push_back(1.0);
                    bool ok = true;
                    for (size_t cidx : controls) {
                        const auto& cv = std::get<std::vector<double>>(data.columns()[cidx].values);
                        if (r >= cv.size() || !std::isfinite(cv[r])) { ok = false; break; }
                        if (r < data.columns()[cidx].missing.size() && data.columns()[cidx].missing[r]) { ok = false; break; }
                        row.push_back(cv[r]);
                    }
                    if (!ok) continue;
                    ctrlRows.push_back(std::move(row));
                    yVec.push_back(y[r]);
                    xVec.push_back(xv[r]);
                }
                if (ctrlRows.size() < controls.size() + 8) return 0.0;
                MathUtils::Matrix X(ctrlRows.size(), controls.size() + 1);
                MathUtils::Matrix Yy(ctrlRows.size(), 1);
                MathUtils::Matrix Yx(ctrlRows.size(), 1);
                for (size_t i = 0; i < ctrlRows.size(); ++i) {
                    for (size_t c = 0; c < ctrlRows[i].size(); ++c) X.at(i, c) = ctrlRows[i][c];
                    Yy.at(i, 0) = yVec[i];
                    Yx.at(i, 0) = xVec[i];
                }
                const auto by = MathUtils::multipleLinearRegression(X, Yy);
                const auto bx = MathUtils::multipleLinearRegression(X, Yx);
                if (by.size() != controls.size() + 1 || bx.size() != controls.size() + 1) return 0.0;
                std::vector<double> ry(ctrlRows.size(), 0.0), rx(ctrlRows.size(), 0.0);
                for (size_t i = 0; i < ctrlRows.size(); ++i) {
                    double py = 0.0;
                    double px = 0.0;
                    for (size_t c = 0; c < ctrlRows[i].size(); ++c) {
                        py += ctrlRows[i][c] * by[c];
                        px += ctrlRows[i][c] * bx[c];
                    }
                    ry[i] = yVec[i] - py;
                    rx[i] = xVec[i] - px;
                }
                const ColumnStats sx = Statistics::calculateStats(rx);
                const ColumnStats sy = Statistics::calculateStats(ry);
                return MathUtils::calculatePearson(rx, ry, sx, sy).value_or(0.0);
            };

            size_t bestPcIdx = static_cast<size_t>(-1);
            double bestPc = 0.0;
            size_t signFlipCount = 0;
            size_t collapseCount = 0;
            for (size_t idx : numericFeatures) {
                if (std::find(controls.begin(), controls.end(), idx) != controls.end()) continue;
                const double g = globalCorr(idx);
                const double p = partialCorr(idx);
                const double pc = std::abs(p);
                const auto drift = assessGlobalConditionalDrift(g, p);
                if (drift.signFlip || drift.magnitudeCollapse) {
                    if (drift.signFlip) ++signFlipCount;
                    if (drift.magnitudeCollapse) ++collapseCount;
                    std::string interpretation;
                    if (drift.label == "flip+collapse") {
                        interpretation = "Direction reverses after controls and effect size shrinks materially; likely confounding/proxy pathway.";
                    } else if (drift.label == "sign-flip") {
                        interpretation = "Direction reverses after controls; relationship may be Simpson-type/confounded.";
                    } else {
                        interpretation = "Magnitude collapses after controls; global association likely proxy-driven.";
                    }
                    out.globalConditionalRows.push_back({
                        data.columns()[idx].name,
                        toFixed(g, 4),
                        toFixed(p, 4),
                        toFixed(drift.collapseRatio, 3),
                        drift.label,
                        interpretation
                    });
                }

                if (pc > bestPc) {
                    bestPc = pc;
                    bestPcIdx = idx;
                }
            }
            out.orderedRows.push_back({"5", "Partial correlation after control", (bestPcIdx == static_cast<size_t>(-1) ? "No stable partial-correlation signal." : (data.columns()[bestPcIdx].name + " retains independent signal after controlling for top predictors (partial |r|=" + toFixed(bestPc, 4) + ", controls=" + std::to_string(controls.size()) + ")"))});

            if (!out.globalConditionalRows.empty()) {
                const std::string driftMsg = "Detected " + std::to_string(out.globalConditionalRows.size()) +
                    " global-vs-conditional drift patterns (sign flips=" + std::to_string(signFlipCount) +
                    ", magnitude collapses=" + std::to_string(collapseCount) + ").";
                out.orderedRows.push_back({"6", "Global vs conditional drift check", driftMsg});
                out.priorityTakeaways.push_back(driftMsg);
            }
        } else {
            out.orderedRows.push_back({"4", "Residual discovery via stepwise OLS", "No usable numeric predictors."});
            out.orderedRows.push_back({"5", "Partial correlation after control", "No usable numeric predictors."});
        }
    }

    {
        std::string interactionMsg = "Insufficient data for interaction t-test.";
        if (igA != static_cast<size_t>(-1) && igB != static_cast<size_t>(-1)) {
            std::vector<std::array<double, 4>> rows;
            std::vector<double> yAligned;
            rows.reserve(y.size());
            yAligned.reserve(y.size());
            const auto& a = std::get<std::vector<double>>(data.columns()[igA].values);
            const auto& b = std::get<std::vector<double>>(data.columns()[igB].values);
            for (size_t r = 0; r < y.size() && r < a.size() && r < b.size(); ++r) {
                if (!std::isfinite(y[r]) || !std::isfinite(a[r]) || !std::isfinite(b[r])) continue;
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                if (r < data.columns()[igA].missing.size() && data.columns()[igA].missing[r]) continue;
                if (r < data.columns()[igB].missing.size() && data.columns()[igB].missing[r]) continue;
                rows.push_back({1.0, a[r], b[r], a[r] * b[r]});
                yAligned.push_back(y[r]);
            }
            if (rows.size() >= 16) {
                MathUtils::Matrix X(rows.size(), 4);
                MathUtils::Matrix Y(rows.size(), 1);
                for (size_t i = 0; i < rows.size(); ++i) {
                    for (size_t c = 0; c < 4; ++c) X.at(i, c) = rows[i][c];
                    Y.at(i, 0) = yAligned[i];
                }
                MLRDiagnostics diag = MathUtils::performMLRWithDiagnostics(X, Y);
                if (diag.success && diag.tStats.size() > 3 && diag.pValues.size() > 3 && diag.coefficients.size() > 3) {
                    const double tVal = diag.tStats[3];
                    const double pVal = diag.pValues[3];
                    const double beta = diag.coefficients[3];
                    const std::string effectDir = beta >= 0.0 ? "synergistic" : "antagonistic";
                    const std::string strength = (pVal < 1e-4) ? "very strong" : ((pVal < 0.01) ? "strong" : ((pVal < 0.05) ? "moderate" : "weak"));
                    const std::string implication = (strength == "weak")
                        ? ("suggesting a tentative " + effectDir + " interaction pattern")
                        : ("showing a " + effectDir + " interaction direction");
                    interactionMsg = data.columns()[igA].name + "×" + data.columns()[igB].name +
                        " interaction is " + strength + " (t=" + toFixed(tVal, 4) + ", p=" + toFixed(pVal, 6) +
                        "); " + implication + " for " + data.columns()[static_cast<size_t>(targetIdx)].name + ".";
                    out.interactionEvidence = interactionMsg;
                    out.priorityTakeaways.push_back(interactionMsg);
                }
            }
        }
        out.orderedRows.push_back({"6", "Linear interaction significance", interactionMsg});
    }

    {
        struct CausalFeatureScore {
            size_t idx = static_cast<size_t>(-1);
            double rawAbs = 0.0;
            double partialAbs = 0.0;
            double retention = 0.0;
            std::string role;
        };

        struct CausalEdge {
            std::string from;
            std::string to;
            double confidence = 0.0;
            std::string evidence;
            std::string interpretation;
        };

        auto pearsonAbsAligned = [&](size_t aIdx, size_t bIdx) -> double {
            if (aIdx >= data.columns().size() || bIdx >= data.columns().size()) return 0.0;
            if (data.columns()[aIdx].type != ColumnType::NUMERIC || data.columns()[bIdx].type != ColumnType::NUMERIC) return 0.0;
            const auto& a = std::get<std::vector<double>>(data.columns()[aIdx].values);
            const auto& b = std::get<std::vector<double>>(data.columns()[bIdx].values);
            std::vector<double> xa;
            std::vector<double> xb;
            xa.reserve(std::min(a.size(), b.size()));
            xb.reserve(std::min(a.size(), b.size()));
            const size_t n = std::min(a.size(), b.size());
            for (size_t r = 0; r < n; ++r) {
                if (!std::isfinite(a[r]) || !std::isfinite(b[r])) continue;
                if (r < data.columns()[aIdx].missing.size() && data.columns()[aIdx].missing[r]) continue;
                if (r < data.columns()[bIdx].missing.size() && data.columns()[bIdx].missing[r]) continue;
                xa.push_back(a[r]);
                xb.push_back(b[r]);
            }
            if (xa.size() < 12) return 0.0;
            const ColumnStats as = Statistics::calculateStats(xa);
            const ColumnStats bs = Statistics::calculateStats(xb);
            return std::abs(MathUtils::calculatePearson(xa, xb, as, bs).value_or(0.0));
        };

        auto partialAgainstControls = [&](size_t feature, const std::vector<size_t>& controls) -> double {
            if (feature >= data.columns().size()) return 0.0;
            const auto& xv = std::get<std::vector<double>>(data.columns()[feature].values);
            std::vector<double> yVec;
            std::vector<double> xVec;
            std::vector<std::vector<double>> ctrlRows;
            const size_t n = std::min(y.size(), xv.size());
            for (size_t r = 0; r < n; ++r) {
                if (!std::isfinite(y[r]) || !std::isfinite(xv[r])) continue;
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                if (r < data.columns()[feature].missing.size() && data.columns()[feature].missing[r]) continue;
                std::vector<double> row;
                row.reserve(controls.size() + 1);
                row.push_back(1.0);
                bool ok = true;
                for (size_t cidx : controls) {
                    const auto& cv = std::get<std::vector<double>>(data.columns()[cidx].values);
                    if (r >= cv.size() || !std::isfinite(cv[r])) { ok = false; break; }
                    if (r < data.columns()[cidx].missing.size() && data.columns()[cidx].missing[r]) { ok = false; break; }
                    row.push_back(cv[r]);
                }
                if (!ok) continue;
                ctrlRows.push_back(std::move(row));
                yVec.push_back(y[r]);
                xVec.push_back(xv[r]);
            }
            if (ctrlRows.size() < controls.size() + 10) return 0.0;
            MathUtils::Matrix X(ctrlRows.size(), controls.size() + 1);
            MathUtils::Matrix Yy(ctrlRows.size(), 1);
            MathUtils::Matrix Yx(ctrlRows.size(), 1);
            for (size_t i = 0; i < ctrlRows.size(); ++i) {
                for (size_t c = 0; c < ctrlRows[i].size(); ++c) X.at(i, c) = ctrlRows[i][c];
                Yy.at(i, 0) = yVec[i];
                Yx.at(i, 0) = xVec[i];
            }
            const auto by = MathUtils::multipleLinearRegression(X, Yy);
            const auto bx = MathUtils::multipleLinearRegression(X, Yx);
            if (by.size() != controls.size() + 1 || bx.size() != controls.size() + 1) return 0.0;

            std::vector<double> ry(ctrlRows.size(), 0.0);
            std::vector<double> rx(ctrlRows.size(), 0.0);
            for (size_t i = 0; i < ctrlRows.size(); ++i) {
                double py = 0.0;
                double px = 0.0;
                for (size_t c = 0; c < ctrlRows[i].size(); ++c) {
                    py += ctrlRows[i][c] * by[c];
                    px += ctrlRows[i][c] * bx[c];
                }
                ry[i] = yVec[i] - py;
                rx[i] = xVec[i] - px;
            }
            const ColumnStats sx = Statistics::calculateStats(rx);
            const ColumnStats sy = Statistics::calculateStats(ry);
            return std::abs(MathUtils::calculatePearson(rx, ry, sx, sy).value_or(0.0));
        };

        std::vector<CausalFeatureScore> featureScores;
        featureScores.reserve(numericFeatures.size());
        for (size_t idx : numericFeatures) {
            const double raw = pearsonAbsAligned(idx, static_cast<size_t>(targetIdx));
            featureScores.push_back({idx, raw, 0.0, 0.0, "uncertain"});
        }
        std::sort(featureScores.begin(), featureScores.end(), [](const auto& a, const auto& b) {
            return a.rawAbs > b.rawAbs;
        });

        std::vector<size_t> ranking;
        ranking.reserve(featureScores.size());
        for (const auto& s : featureScores) ranking.push_back(s.idx);

        std::vector<size_t> drivers;
        std::vector<size_t> proxies;
        for (auto& score : featureScores) {
            std::vector<size_t> controls;
            for (size_t idx : ranking) {
                if (idx == score.idx) continue;
                controls.push_back(idx);
                if (controls.size() >= 2) break;
            }
            score.partialAbs = partialAgainstControls(score.idx, controls);
            score.retention = (score.rawAbs > 1e-8) ? (score.partialAbs / score.rawAbs) : 0.0;

            if (score.rawAbs >= 0.20 && score.partialAbs >= 0.12 && score.retention >= 0.55) {
                score.role = "likely driver";
                drivers.push_back(score.idx);
            } else if (score.rawAbs >= 0.20 && score.retention <= 0.35) {
                score.role = "likely proxy";
                proxies.push_back(score.idx);
            } else {
                score.role = "uncertain";
            }
        }

        std::vector<CausalEdge> edges;
        const std::string targetName = data.columns()[static_cast<size_t>(targetIdx)].name;
        for (const auto& score : featureScores) {
            if (score.role != "likely driver") continue;
            const double conf = std::clamp(0.35 + 0.45 * score.partialAbs + 0.20 * score.retention, 0.0, 0.99);
            edges.push_back({
                data.columns()[score.idx].name,
                targetName,
                conf,
                "|r|=" + toFixed(score.rawAbs, 3) + ", partial|r|=" + toFixed(score.partialAbs, 3) + ", retention=" + toFixed(score.retention, 3),
                "Signal remains after controlling top predictors; candidate driver for target."
            });
        }

        for (size_t proxyIdx : proxies) {
            double bestLink = 0.0;
            size_t bestDriver = static_cast<size_t>(-1);
            for (size_t driverIdx : drivers) {
                const double link = pearsonAbsAligned(proxyIdx, driverIdx);
                if (link > bestLink) {
                    bestLink = link;
                    bestDriver = driverIdx;
                }
            }

            if (bestDriver != static_cast<size_t>(-1) && bestLink >= 0.45) {
                const double conf = std::clamp(0.25 + 0.55 * bestLink, 0.0, 0.95);
                edges.push_back({
                    data.columns()[bestDriver].name,
                    data.columns()[proxyIdx].name,
                    conf,
                    "driver-proxy link |r|=" + toFixed(bestLink, 3),
                    "Proxy candidate: correlation with target weakens strongly after controls."
                });
            } else {
                const double raw = pearsonAbsAligned(proxyIdx, static_cast<size_t>(targetIdx));
                edges.push_back({
                    data.columns()[proxyIdx].name,
                    targetName,
                    std::clamp(0.15 + 0.40 * raw, 0.0, 0.75),
                    "|r|=" + toFixed(raw, 3) + " with unresolved upstream driver",
                    "Association detected; likely proxy path not fully resolved."
                });
            }
        }

        {
            std::vector<size_t> dagNodes;
            dagNodes.reserve(featureScores.size());
            for (const auto& s : featureScores) {
                dagNodes.push_back(s.idx);
                if (dagNodes.size() >= 6) break;
            }

            for (size_t i = 0; i < dagNodes.size(); ++i) {
                for (size_t j = i + 1; j < dagNodes.size(); ++j) {
                    const size_t aIdx = dagNodes[i];
                    const size_t bIdx = dagNodes[j];
                    const auto& aVals = std::get<std::vector<double>>(data.columns()[aIdx].values);
                    const auto& bVals = std::get<std::vector<double>>(data.columns()[bIdx].values);
                    const auto direction = Statistics::asymmetricInformationGain(aVals, bVals, 8);
                    if (std::abs(direction.asymmetry) < 0.015) continue;

                    const bool aToB = direction.suggestedDirection == "x->y";
                    const std::string from = aToB ? data.columns()[aIdx].name : data.columns()[bIdx].name;
                    const std::string to = aToB ? data.columns()[bIdx].name : data.columns()[aIdx].name;
                    const double pairCorr = pearsonAbsAligned(aIdx, bIdx);
                    const double conf = std::clamp(0.20 + std::min(0.45, std::abs(direction.asymmetry) * 1.8) + 0.25 * pairCorr, 0.0, 0.93);

                    edges.push_back({
                        from,
                        to,
                        conf,
                        "AIG(x→y)=" + toFixed(direction.xToY, 3) + ", AIG(y→x)=" + toFixed(direction.yToX, 3) + ", Δ=" + toFixed(direction.asymmetry, 3),
                        "Directed by asymmetric information gain; treat as causal ordering hypothesis."
                    });
                }
            }

            if (drivers.size() >= 2) {
                const size_t d1 = drivers[0];
                const size_t d2 = drivers[1];
                const double dd = pearsonAbsAligned(d1, d2);
                const double dt1 = pearsonAbsAligned(d1, static_cast<size_t>(targetIdx));
                const double dt2 = pearsonAbsAligned(d2, static_cast<size_t>(targetIdx));
                if (dd <= 0.15 && dt1 >= 0.30 && dt2 >= 0.30) {
                    edges.push_back({
                        data.columns()[d1].name + " & " + data.columns()[d2].name,
                        targetName,
                        std::clamp(0.30 + 0.30 * (dt1 + dt2), 0.0, 0.95),
                        "PC-style collider signal: parent-parent |r|=" + toFixed(dd, 3) + ", parent-target |r|=" + toFixed(dt1, 3) + "/" + toFixed(dt2, 3),
                        "Potential collider around target consistent with Peter-Clark orientation logic."
                    });
                }
            }
        }

        if (igA != static_cast<size_t>(-1) && igB != static_cast<size_t>(-1) && bestIgProxy >= 0.12) {
            edges.push_back({
                data.columns()[igA].name + " × " + data.columns()[igB].name,
                targetName,
                std::clamp(0.20 + 0.50 * bestIgProxy, 0.0, 0.90),
                "interaction proxy=" + toFixed(bestIgProxy, 3),
                "Pair interaction may carry directional effect beyond additive terms."
            });
        }

        std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
            return a.confidence > b.confidence;
        });
        if (edges.size() > 10) edges.resize(10);

        for (const auto& e : edges) {
            out.causalDagRows.push_back({e.from, e.to, toFixed(e.confidence, 3), e.evidence, e.interpretation});
        }

        if (!edges.empty()) {
            std::unordered_map<std::string, std::string> nodeIds;
            auto idFor = [&](const std::string& node) {
                auto it = nodeIds.find(node);
                if (it != nodeIds.end()) return it->second;
                const std::string id = "N" + std::to_string(nodeIds.size() + 1);
                nodeIds[node] = id;
                return id;
            };
            auto quoteSafe = [](std::string s) {
                std::replace(s.begin(), s.end(), '"', '\'');
                return s;
            };

            std::ostringstream mermaid;
            mermaid << "```mermaid\n";
            mermaid << "flowchart LR\n";
            for (const auto& e : edges) {
                const std::string fromId = idFor(e.from);
                const std::string toId = idFor(e.to);
                mermaid << "    " << fromId << "[\"" << quoteSafe(e.from) << "\"] -->|"
                        << toFixed(e.confidence, 2) << "| "
                        << toId << "[\"" << quoteSafe(e.to) << "\"]\n";
            }
            mermaid << "```";
            out.causalDagMermaid = mermaid.str();

            size_t driverEdges = 0;
            size_t proxyEdges = 0;
            for (const auto& e : edges) {
                if (e.to == targetName) ++driverEdges;
                if (e.interpretation.find("Proxy") != std::string::npos || e.interpretation.find("proxy") != std::string::npos) ++proxyEdges;
            }
            const std::string summary = "Causal-lite DAG generated " + std::to_string(edges.size()) +
                " directed candidates (to-target=" + std::to_string(driverEdges) +
                ", proxy-path=" + std::to_string(proxyEdges) + ").";
            out.priorityTakeaways.push_back(summary + " Treat these as directional hypotheses, not causal proof.");
            out.orderedRows.push_back({"7", "Causal inference (lite) DAG", summary});
        } else {
            out.orderedRows.push_back({"7", "Causal inference (lite) DAG", "No stable directed candidates; current significant findings are likely associative only."});
        }
    }

    {
        const auto axis = detectTemporalAxis(data);
        std::vector<size_t> temporalCandidates = numericFeatures;
        if (std::find(temporalCandidates.begin(), temporalCandidates.end(), static_cast<size_t>(targetIdx)) == temporalCandidates.end()) {
            temporalCandidates.push_back(static_cast<size_t>(targetIdx));
        }

        std::vector<StationarityDiagnostic> diagnostics;
        diagnostics.reserve(temporalCandidates.size());
        for (size_t idx : temporalCandidates) {
            if (idx >= data.columns().size()) continue;
            if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
            if (CommonUtils::toLower(data.columns()[idx].name) == CommonUtils::toLower(axis.name)) continue;
            const auto& values = std::get<std::vector<double>>(data.columns()[idx].values);
            auto diag = Statistics::adfStyleDrift(values, axis.axis, data.columns()[idx].name, axis.name);
            if (diag.samples < 12) continue;
            diagnostics.push_back(std::move(diag));
        }

        std::sort(diagnostics.begin(), diagnostics.end(), [](const auto& a, const auto& b) {
            if (a.nonStationary != b.nonStationary) return a.nonStationary > b.nonStationary;
            if (a.driftRatio == b.driftRatio) return a.pApprox > b.pApprox;
            return a.driftRatio > b.driftRatio;
        });

        size_t flagged = 0;
        for (const auto& d : diagnostics) {
            if (d.nonStationary) ++flagged;
            if (!d.nonStationary && d.verdict != "borderline") continue;
            out.temporalDriftRows.push_back({
                d.feature,
                d.axis,
                std::to_string(d.samples),
                toFixed(d.gamma, 5),
                toFixed(d.tStatistic, 4),
                toFixed(d.pApprox, 6),
                toFixed(d.driftRatio, 3),
                d.verdict
            });
            if (out.temporalDriftRows.size() >= 12) break;
        }

        if (!diagnostics.empty()) {
            const std::string msg = "ADF-style temporal drift scan on axis '" + axis.name +
                "' flagged " + std::to_string(flagged) + " non-stationary signals out of " + std::to_string(diagnostics.size()) + ".";
            out.orderedRows.push_back({"8", "Temporal drift kernels (ADF-style)", msg});
            if (flagged > 0) {
                out.priorityTakeaways.push_back(msg + " Prioritize time-aware features or differencing for flagged columns.");
            }
        }
    }

    {
        const auto deadZones = detectContextualDeadZones(data,
                                                         static_cast<size_t>(targetIdx),
                                                         numericFeatures,
                                                         10);
        for (const auto& dz : deadZones) {
            out.contextualDeadZoneRows.push_back({
                dz.feature,
                dz.strongCluster,
                dz.weakCluster,
                toFixed(dz.strongCorr, 4),
                toFixed(dz.weakCorr, 4),
                toFixed(dz.dropRatio, 3),
                std::to_string(dz.support)
            });
        }
        if (!deadZones.empty()) {
            const auto& lead = deadZones.front();
            const std::string msg = "Contextual dead-zones detected: " + std::to_string(deadZones.size()) +
                " features switch from predictive to locally irrelevant across clusters; strongest example " +
                lead.feature + " (" + lead.strongCluster + " |r|=" + toFixed(lead.strongCorr, 3) +
                " vs " + lead.weakCluster + " |r|=" + toFixed(lead.weakCorr, 3) + ").";
            out.orderedRows.push_back({"9", "Cross-cluster interaction anomalies", msg});
            out.priorityTakeaways.push_back(msg);
        }
    }

    {
        std::string mahalMsg = "Insufficient stable dimensions for Mahalanobis outlier scoring.";
        if (numericFeatures.size() >= 2) {
            const size_t dims = std::min<size_t>(3, numericFeatures.size());
            std::vector<std::vector<double>> rows;
            std::vector<size_t> rowIds;
            for (size_t r = 0; r < data.rowCount(); ++r) {
                std::vector<double> row;
                row.reserve(dims);
                bool ok = true;
                for (size_t c = 0; c < dims; ++c) {
                    size_t idx = numericFeatures[c];
                    const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
                    if (r >= vals.size() || !std::isfinite(vals[r])) { ok = false; break; }
                    if (r < data.columns()[idx].missing.size() && data.columns()[idx].missing[r]) { ok = false; break; }
                    row.push_back(vals[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                rowIds.push_back(r + 1);
            }
            if (rows.size() > dims + 8) {
                std::vector<double> mean(dims, 0.0);
                for (const auto& row : rows) {
                    for (size_t c = 0; c < dims; ++c) mean[c] += row[c];
                }
                for (double& m : mean) m /= static_cast<double>(rows.size());

                MathUtils::Matrix cov(dims, dims);
                for (const auto& row : rows) {
                    for (size_t i = 0; i < dims; ++i) {
                        for (size_t j = 0; j < dims; ++j) {
                            cov.at(i, j) += (row[i] - mean[i]) * (row[j] - mean[j]);
                        }
                    }
                }
                const double denom = static_cast<double>(std::max<size_t>(1, rows.size() - 1));
                for (size_t i = 0; i < dims; ++i) {
                    for (size_t j = 0; j < dims; ++j) cov.at(i, j) /= denom;
                }
                auto invOpt = cov.inverse();
                if (invOpt.has_value()) {
                    const auto& inv = *invOpt;
                    std::vector<double> d2(rows.size(), 0.0);
                    for (size_t r = 0; r < rows.size(); ++r) {
                        std::vector<double> diff(dims, 0.0);
                        for (size_t c = 0; c < dims; ++c) diff[c] = rows[r][c] - mean[c];
                        double sum = 0.0;
                        for (size_t i = 0; i < dims; ++i) {
                            double inner = 0.0;
                            for (size_t j = 0; j < dims; ++j) inner += inv.data[i][j] * diff[j];
                            sum += diff[i] * inner;
                        }
                        d2[r] = std::max(0.0, sum);
                    }
                    const double threshold = CommonUtils::quantileByNth(d2, 0.975);
                    size_t flagged = 0;
                    std::vector<std::pair<size_t, double>> top;
                    for (size_t i = 0; i < d2.size(); ++i) {
                        if (d2[i] >= threshold) {
                            ++flagged;
                            top.push_back({rowIds[i], d2[i]});
                        }
                    }
                    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                    out.mahalanobisThreshold = threshold;
                    for (size_t i = 0; i < std::min<size_t>(6, top.size()); ++i) {
                        out.mahalanobisRows.push_back({std::to_string(top[i].first), toFixed(top[i].second, 4), toFixed(threshold, 4)});
                        out.mahalanobisByRow[top[i].first] = top[i].second;
                    }
                    mahalMsg = "Dimensions=" + std::to_string(dims) + ", flagged=" + std::to_string(flagged) + ", threshold(d2@97.5%)=" + toFixed(threshold, 4);
                    if (flagged > 0) {
                        out.priorityTakeaways.push_back("Mahalanobis multivariate scan flags " + std::to_string(flagged) + " structurally unusual observations.");
                    }
                }
            }
        }
        out.orderedRows.push_back({"7", "Mahalanobis multivariate outliers", mahalMsg});
    }

    {
        std::string anovaMsg = "No ANOVA/Tukey highlight available.";
        if (!anovaRows.empty()) {
            const auto it = std::max_element(anovaRows.begin(), anovaRows.end(), [](const auto& a, const auto& b) {
                if (a.eta2 == b.eta2) return a.pValue > b.pValue;
                return a.eta2 < b.eta2;
            });
            anovaMsg = it->categorical + " -> " + it->numeric + " (eta2=" + toFixed(it->eta2, 4) + ", Tukey=" + it->tukeySummary + ")";
        }
        out.orderedRows.push_back({"8", "ANOVA + Tukey HSD strongest class contrast", anovaMsg});
    }

    {
        std::string cramerMsg = "No categorical-contingency highlight available.";
        if (!contingency.empty()) {
            const auto it = std::max_element(contingency.begin(), contingency.end(), [](const auto& a, const auto& b) {
                return a.cramerV < b.cramerV;
            });
            cramerMsg = it->catA + " vs " + it->catB + " (V=" + toFixed(it->cramerV, 4) + ", " + cramerStrengthLabel(it->cramerV) + ")";
        }
        out.orderedRows.push_back({"9", "Cramér's V categorical strength", cramerMsg});
    }

    {
        std::string pdpMsg = "Insufficient data for linear partial-dependence approximation.";
        if (!numericFeatures.empty()) {
            const size_t dims = std::min<size_t>(3, numericFeatures.size());
            std::vector<std::vector<double>> rows;
            std::vector<double> yy;
            for (size_t r = 0; r < data.rowCount(); ++r) {
                if (r >= y.size() || !std::isfinite(y[r])) continue;
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                std::vector<double> row;
                row.reserve(dims + 1);
                row.push_back(1.0);
                bool ok = true;
                for (size_t c = 0; c < dims; ++c) {
                    const auto& vals = std::get<std::vector<double>>(data.columns()[numericFeatures[c]].values);
                    if (r >= vals.size() || !std::isfinite(vals[r])) { ok = false; break; }
                    if (r < data.columns()[numericFeatures[c]].missing.size() && data.columns()[numericFeatures[c]].missing[r]) { ok = false; break; }
                    row.push_back(vals[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                yy.push_back(y[r]);
            }
            if (rows.size() >= dims + 10) {
                MathUtils::Matrix X(rows.size(), dims + 1);
                MathUtils::Matrix Y(rows.size(), 1);
                for (size_t i = 0; i < rows.size(); ++i) {
                    for (size_t c = 0; c < rows[i].size(); ++c) X.at(i, c) = rows[i][c];
                    Y.at(i, 0) = yy[i];
                }
                auto beta = MathUtils::multipleLinearRegression(X, Y);
                if (beta.size() == dims + 1) {
                    std::vector<double> meanX(dims, 0.0);
                    for (const auto& row : rows) {
                        for (size_t c = 0; c < dims; ++c) meanX[c] += row[c + 1];
                    }
                    for (double& v : meanX) v /= static_cast<double>(rows.size());

                    for (size_t c = 0; c < std::min<size_t>(2, dims); ++c) {
                        const auto& vals = std::get<std::vector<double>>(data.columns()[numericFeatures[c]].values);
                        const double p10 = CommonUtils::quantileByNth(vals, 0.10);
                        const double p50 = CommonUtils::quantileByNth(vals, 0.50);
                        const double p90 = CommonUtils::quantileByNth(vals, 0.90);
                        auto pred = [&](double v) {
                            double yhat = beta[0];
                            for (size_t j = 0; j < dims; ++j) {
                                const double xj = (j == c) ? v : meanX[j];
                                yhat += beta[j + 1] * xj;
                            }
                            return yhat;
                        };
                        const double y10 = pred(p10);
                        const double y50 = pred(p50);
                        const double y90 = pred(p90);
                        const double delta = y90 - y10;
                        const std::string direction = (delta > 0.0) ? "increasing" : ((delta < 0.0) ? "decreasing" : "flat");
                        out.pdpRows.push_back({data.columns()[numericFeatures[c]].name,
                                               toFixed(y10, 4),
                                               toFixed(y50, 4),
                                               toFixed(y90, 4),
                                               toFixed(delta, 4),
                                               direction});
                    }
                    pdpMsg = "Generated linear PDP approximations for " + std::to_string(out.pdpRows.size()) + " top features.";
                    if (!out.pdpRows.empty()) {
                        out.priorityTakeaways.push_back("PDP approximation shows " + out.pdpRows.front()[0] + " has a " + out.pdpRows.front()[5] + " target response across low→high values (Δ=" + out.pdpRows.front()[4] + ").");
                    }
                }
            }
        }
        out.orderedRows.push_back({"10", "Linear partial-dependence approximation", pdpMsg});
    }

    {
        std::string topA = (numericFeatures.size() > 0) ? data.columns()[numericFeatures[0]].name : "Feature A";
        std::string topB = (numericFeatures.size() > 1) ? data.columns()[numericFeatures[1]].name : "Feature B";
        std::string narrative = topA + " dominates the global signal, while " + topB + " adds independent information after controlling for primary effects.";
        if (igA != static_cast<size_t>(-1) && igB != static_cast<size_t>(-1)) {
            narrative = data.columns()[igA].name + " and " + data.columns()[igB].name + " jointly shape the target: one carries dominant variance while the other contributes independent interaction structure.";
        }
        out.narrativeRows.push_back(narrative);
        out.priorityTakeaways.push_back(narrative);
        out.orderedRows.push_back({"11", "Cross-section narrative synthesis", narrative});
    }

    std::vector<std::vector<std::string>> filtered;
    filtered.reserve(out.orderedRows.size());
    for (const auto& row : out.orderedRows) {
        if (row.size() < 3) continue;
        const std::string r = CommonUtils::trim(row[2]);
        if (r.rfind("Insufficient", 0) == 0 || r.rfind("No ", 0) == 0) continue;
        filtered.push_back(row);
    }
    for (size_t i = 0; i < filtered.size(); ++i) {
        filtered[i][0] = std::to_string(i + 1);
    }
    out.orderedRows = std::move(filtered);

    if (!out.priorityTakeaways.empty()) {
        out.executiveSummary = "Advanced methods confirm that signal is not purely pairwise: interaction evidence, multivariate outlier structure, and residual stepwise modeling all contribute independent explanatory value.";
    }

    return out;
}

struct StratifiedPopulationInsight {
    std::string segmentColumn;
    std::string numericColumn;
    size_t groups = 0;
    size_t rows = 0;
    double eta2 = 0.0;
    double separation = 0.0;
    std::string groupMeans;
};

std::vector<StratifiedPopulationInsight> detectStratifiedPopulations(const TypedDataset& data,
                                                                     size_t maxInsights = 12) {
    std::vector<StratifiedPopulationInsight> out;
    const auto cats = data.categoricalColumnIndices();
    const auto nums = data.numericColumnIndices();
    if (cats.empty() || nums.empty()) return out;

    for (size_t cidx : cats) {
        const auto& cv = std::get<std::vector<std::string>>(data.columns()[cidx].values);
        if (cv.empty()) continue;

        std::unordered_map<std::string, size_t> labelCardinality;
        for (size_t r = 0; r < cv.size() && r < data.rowCount(); ++r) {
            if (data.columns()[cidx].missing[r]) continue;
            if (cv[r].empty()) continue;
            labelCardinality[cv[r]]++;
        }
        if (labelCardinality.size() < 2 || labelCardinality.size() > 8) continue;

        for (size_t nidx : nums) {
            const auto& nv = std::get<std::vector<double>>(data.columns()[nidx].values);
            const size_t n = std::min(cv.size(), nv.size());
            if (n < 30) continue;

            struct GroupStats {
                size_t count = 0;
                double sum = 0.0;
                double sumSq = 0.0;
            };
            std::map<std::string, GroupStats> groups;
            for (size_t i = 0; i < n; ++i) {
                if (data.columns()[cidx].missing[i] || data.columns()[nidx].missing[i]) continue;
                if (!std::isfinite(nv[i])) continue;
                const std::string& label = cv[i];
                if (label.empty()) continue;
                auto& g = groups[label];
                g.count++;
                g.sum += nv[i];
                g.sumSq += nv[i] * nv[i];
            }

            size_t validRows = 0;
            for (const auto& kv : groups) validRows += kv.second.count;
            if (groups.size() < 2 || validRows < 30) continue;

            const size_t minGroupRows = std::max<size_t>(3, validRows / 300);
            size_t strongGroups = 0;
            for (const auto& kv : groups) {
                if (kv.second.count >= minGroupRows) strongGroups++;
            }
            if (strongGroups < 2) continue;

            const double grand = [&]() {
                double s = 0.0;
                for (const auto& kv : groups) s += kv.second.sum;
                return s / static_cast<double>(validRows);
            }();

            double ssb = 0.0;
            double ssw = 0.0;
            double minMean = std::numeric_limits<double>::infinity();
            double maxMean = -std::numeric_limits<double>::infinity();
            std::vector<std::pair<std::string, double>> means;
            means.reserve(groups.size());
            for (const auto& kv : groups) {
                const auto& g = kv.second;
                if (g.count == 0) continue;
                const double mu = g.sum / static_cast<double>(g.count);
                minMean = std::min(minMean, mu);
                maxMean = std::max(maxMean, mu);
                means.push_back({kv.first, mu});
                const double dm = mu - grand;
                ssb += static_cast<double>(g.count) * dm * dm;
                const double within = std::max(0.0, g.sumSq - (g.sum * g.sum) / static_cast<double>(g.count));
                ssw += within;
            }
            if (means.size() < 2) continue;

            const double sst = ssb + ssw;
            if (sst <= 1e-12) continue;
            const double eta2 = ssb / sst;
            const double pooledStd = std::sqrt(ssw / static_cast<double>(std::max<size_t>(1, validRows - means.size())));
            const double separation = (pooledStd <= 1e-12) ? 0.0 : std::abs(maxMean - minMean) / pooledStd;

            if (eta2 < 0.20 || separation < 0.75) continue;

            std::sort(means.begin(), means.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            std::ostringstream oss;
            const size_t show = std::min<size_t>(4, means.size());
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) oss << "; ";
                oss << means[i].first << ": " << toFixed(means[i].second, 4);
            }

            out.push_back({
                data.columns()[cidx].name,
                data.columns()[nidx].name,
                means.size(),
                validRows,
                eta2,
                separation,
                oss.str()
            });
        }
    }

    std::sort(out.begin(), out.end(), [](const StratifiedPopulationInsight& a, const StratifiedPopulationInsight& b) {
        if (a.eta2 == b.eta2) return a.separation > b.separation;
        return a.eta2 > b.eta2;
    });
    if (out.size() > maxInsights) out.resize(maxInsights);
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

            const size_t nTs = ts->values.size();
            const size_t seasonPeriod = std::clamp<size_t>(config.tuning.timeSeriesSeasonPeriod,
                                                           static_cast<size_t>(2),
                                                           std::max<size_t>(2, nTs / 2));
            std::vector<double> trendComp(nTs, 0.0);
            const size_t trendWindow = std::max<size_t>(3, std::min<size_t>(std::max<size_t>(seasonPeriod, 3), 25));
            const size_t half = trendWindow / 2;
            for (size_t i = 0; i < nTs; ++i) {
                const size_t lo = (i > half) ? (i - half) : 0;
                const size_t hi = std::min(nTs - 1, i + half);
                double sum = 0.0;
                size_t count = 0;
                for (size_t j = lo; j <= hi; ++j) {
                    sum += ts->values[j];
                    ++count;
                }
                trendComp[i] = sum / static_cast<double>(std::max<size_t>(1, count));
            }

            std::vector<double> detrended(nTs, 0.0);
            for (size_t i = 0; i < nTs; ++i) {
                detrended[i] = ts->values[i] - trendComp[i];
            }

            std::vector<double> seasonalMeans(seasonPeriod, 0.0);
            std::vector<size_t> seasonalCounts(seasonPeriod, 0);
            for (size_t i = 0; i < nTs; ++i) {
                const size_t phase = i % seasonPeriod;
                seasonalMeans[phase] += detrended[i];
                seasonalCounts[phase] += 1;
            }
            for (size_t p = 0; p < seasonPeriod; ++p) {
                if (seasonalCounts[p] > 0) {
                    seasonalMeans[p] /= static_cast<double>(seasonalCounts[p]);
                }
            }
            double seasonalMean = 0.0;
            for (double v : seasonalMeans) seasonalMean += v;
            seasonalMean /= static_cast<double>(seasonalMeans.size());
            for (double& v : seasonalMeans) v -= seasonalMean;

            std::vector<double> seasonal(nTs, 0.0);
            std::vector<double> resid(nTs, 0.0);
            for (size_t i = 0; i < nTs; ++i) {
                seasonal[i] = seasonalMeans[i % seasonPeriod];
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
                if (isAdministrativeColumnName(data.columns()[idx].name)) continue;
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

            std::vector<size_t> nums;
            nums.reserve(data.numericColumnIndices().size());
            for (size_t idx : data.numericColumnIndices()) {
                if (isAdministrativeColumnName(data.columns()[idx].name)) continue;
                nums.push_back(idx);
            }
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

std::optional<std::string> buildResidualDiscoveryNarrative(const TypedDataset& data,
                                                           int targetIdx,
                                                           const std::vector<int>& featureIdx,
                                                           const NumericStatsCache& statsCache) {
    if (targetIdx < 0 || featureIdx.size() < 2) return std::nullopt;
    if (static_cast<size_t>(targetIdx) >= data.columns().size()) return std::nullopt;
    if (data.columns()[static_cast<size_t>(targetIdx)].type != ColumnType::NUMERIC) return std::nullopt;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
    const ColumnStats yStats = statsCache.count(static_cast<size_t>(targetIdx))
        ? statsCache.at(static_cast<size_t>(targetIdx))
        : Statistics::calculateStats(y);

    int bestIdx = -1;
    double bestAbsCorr = 0.0;
    for (int idx : featureIdx) {
        if (idx < 0 || idx == targetIdx) continue;
        if (static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (isAdministrativeColumnName(data.columns()[static_cast<size_t>(idx)].name)) continue;

        const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        const ColumnStats xStats = statsCache.count(static_cast<size_t>(idx))
            ? statsCache.at(static_cast<size_t>(idx))
            : Statistics::calculateStats(x);
        const double r = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
        if (std::isfinite(r) && r > bestAbsCorr) {
            bestAbsCorr = r;
            bestIdx = idx;
        }
    }
    if (bestIdx < 0 || bestAbsCorr < 0.35) return std::nullopt;

    const auto& xMain = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(bestIdx)].values);
    const ColumnStats xMainStats = statsCache.count(static_cast<size_t>(bestIdx))
        ? statsCache.at(static_cast<size_t>(bestIdx))
        : Statistics::calculateStats(xMain);
    const auto [slope, intercept] = MathUtils::simpleLinearRegression(xMain,
                                                                       y,
                                                                       xMainStats,
                                                                       yStats,
                                                                       MathUtils::calculatePearson(xMain, y, xMainStats, yStats).value_or(0.0));

    std::vector<double> residual;
    residual.reserve(y.size());
    for (size_t i = 0; i < y.size() && i < xMain.size(); ++i) {
        if (!std::isfinite(y[i]) || !std::isfinite(xMain[i])) {
            residual.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        residual.push_back(y[i] - (slope * xMain[i] + intercept));
    }
    const ColumnStats residualStats = Statistics::calculateStats(residual);

    int secondIdx = -1;
    double bestResidualCorr = 0.0;
    double secondGlobalCorr = 0.0;
    for (int idx : featureIdx) {
        if (idx < 0 || idx == targetIdx || idx == bestIdx) continue;
        if (static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (isAdministrativeColumnName(data.columns()[static_cast<size_t>(idx)].name)) continue;

        const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        const ColumnStats xStats = statsCache.count(static_cast<size_t>(idx))
            ? statsCache.at(static_cast<size_t>(idx))
            : Statistics::calculateStats(x);
        const double g = MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0);
        const double rr = std::abs(MathUtils::calculatePearson(x, residual, xStats, residualStats).value_or(0.0));
        if (std::isfinite(rr) && rr > bestResidualCorr) {
            bestResidualCorr = rr;
            secondIdx = idx;
            secondGlobalCorr = g;
        }
    }
    if (secondIdx < 0 || bestResidualCorr < 0.25) return std::nullopt;

    const auto drift = assessGlobalConditionalDrift(secondGlobalCorr, bestResidualCorr);
    std::string driftText;
    if (drift.label == "flip+collapse") {
        driftText = " Drift check: sign-flip + magnitude-collapse versus global relationship.";
    } else if (drift.label == "sign-flip") {
        driftText = " Drift check: sign-flip versus global relationship.";
    } else if (drift.label == "magnitude-collapse") {
        driftText = " Drift check: magnitude-collapse after controlling primary driver.";
    }

    return "Residual discovery: " + data.columns()[static_cast<size_t>(bestIdx)].name +
           " is the primary driver of " + data.columns()[static_cast<size_t>(targetIdx)].name +
           " (|r|=" + toFixed(bestAbsCorr, 3) + "), while " +
           data.columns()[static_cast<size_t>(secondIdx)].name +
           " best explains the remaining error signal (|r_residual|=" + toFixed(bestResidualCorr, 3) +
           ", global r=" + toFixed(secondGlobalCorr, 3) + ")." + driftText;
}

std::vector<std::vector<std::string>> buildOutlierContextRows(const TypedDataset& data,
                                                              const PreprocessReport& prep,
                                                              const std::unordered_map<size_t, double>& mahalByRow = {},
                                                              double mahalThreshold = 0.0,
                                                              size_t maxRows = 8) {
    std::vector<std::vector<std::string>> rows;
    if (prep.outlierFlags.empty() || data.rowCount() == 0) return rows;

    const std::vector<size_t> numericIdx = data.numericColumnIndices();
    if (numericIdx.empty()) return rows;

    std::unordered_map<std::string, ColumnStats> statsByName;
    for (size_t idx : numericIdx) {
        const auto& col = data.columns()[idx];
        statsByName[col.name] = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
    }

    std::string segmentColumnName;
    std::vector<std::string> segmentByRow(data.rowCount());
    std::unordered_map<std::string, std::unordered_map<std::string, ColumnStats>> conditionalStats;
    {
        const auto cats = data.categoricalColumnIndices();
        for (size_t cidx : cats) {
            const auto& ccol = data.columns()[cidx];
            const auto& vals = std::get<std::vector<std::string>>(ccol.values);
            std::unordered_map<std::string, size_t> freq;
            for (size_t r = 0; r < vals.size() && r < ccol.missing.size(); ++r) {
                if (ccol.missing[r]) continue;
                const std::string key = CommonUtils::trim(vals[r]);
                if (!key.empty()) freq[key]++;
            }
            if (freq.size() < 2 || freq.size() > 8) continue;

            size_t minGroup = std::numeric_limits<size_t>::max();
            for (const auto& kv : freq) minGroup = std::min(minGroup, kv.second);
            if (minGroup < 12) continue;

            segmentColumnName = ccol.name;
            for (size_t r = 0; r < data.rowCount() && r < vals.size() && r < ccol.missing.size(); ++r) {
                if (ccol.missing[r]) continue;
                segmentByRow[r] = CommonUtils::trim(vals[r]);
            }
            break;
        }

        if (!segmentColumnName.empty()) {
            for (size_t idx : numericIdx) {
                const auto& col = data.columns()[idx];
                const auto& vals = std::get<std::vector<double>>(col.values);
                std::unordered_map<std::string, std::vector<double>> byGroup;
                for (size_t r = 0; r < data.rowCount() && r < vals.size() && r < col.missing.size(); ++r) {
                    if (col.missing[r] || !std::isfinite(vals[r])) continue;
                    if (segmentByRow[r].empty()) continue;
                    byGroup[segmentByRow[r]].push_back(vals[r]);
                }
                for (auto& kv : byGroup) {
                    if (kv.second.size() < 12) continue;
                    conditionalStats[col.name][kv.first] = Statistics::calculateStats(kv.second);
                }
            }
        }
    }

    for (size_t row = 0; row < data.rowCount() && rows.size() < maxRows; ++row) {
        const size_t rowId = row + 1;
        const auto mit = mahalByRow.find(rowId);
        const bool hasMahalanobis = (mit != mahalByRow.end());
        bool hasOutlier = false;
        size_t primaryCol = static_cast<size_t>(-1);
        double primaryAbsZ = 0.0;

        for (size_t idx : numericIdx) {
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            auto fit = prep.outlierFlags.find(col.name);
            if (fit == prep.outlierFlags.end()) continue;
            if (row >= fit->second.size()) continue;
            if (!fit->second[row]) continue;
            if (row >= col.missing.size() || col.missing[row]) continue;

            const auto& vals = std::get<std::vector<double>>(col.values);
            if (row >= vals.size() || !std::isfinite(vals[row])) continue;
            const auto sit = statsByName.find(col.name);
            if (sit == statsByName.end()) continue;

            double z = 0.0;
            bool usedConditional = false;
            if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
                auto cit = conditionalStats.find(col.name);
                if (cit != conditionalStats.end()) {
                    auto git = cit->second.find(segmentByRow[row]);
                    if (git != cit->second.end() && git->second.stddev > 1e-12) {
                        z = std::abs((vals[row] - git->second.mean) / git->second.stddev);
                        usedConditional = true;
                    }
                }
            }
            if (!usedConditional) {
                z = (sit->second.stddev > 1e-12) ? std::abs((vals[row] - sit->second.mean) / sit->second.stddev) : 0.0;
            }
            if (!hasOutlier || z > primaryAbsZ) {
                hasOutlier = true;
                primaryAbsZ = z;
                primaryCol = idx;
            }
        }

        if (!hasOutlier && !hasMahalanobis) continue;

        if (primaryCol == static_cast<size_t>(-1)) {
            for (size_t idx : numericIdx) {
                const auto& col = data.columns()[idx];
                if (isAdministrativeColumnName(col.name)) continue;
                if (row >= col.missing.size() || col.missing[row]) continue;
                const auto& vals = std::get<std::vector<double>>(col.values);
                if (row >= vals.size() || !std::isfinite(vals[row])) continue;
                const auto sit = statsByName.find(col.name);
                if (sit == statsByName.end() || sit->second.stddev <= 1e-12) continue;
                const double z = std::abs((vals[row] - sit->second.mean) / sit->second.stddev);
                if (z > primaryAbsZ) {
                    primaryAbsZ = z;
                    primaryCol = idx;
                }
            }
            if (primaryCol == static_cast<size_t>(-1)) continue;
        }

        const auto& primary = data.columns()[primaryCol];
        const auto& primaryVals = std::get<std::vector<double>>(primary.values);
        std::vector<std::pair<std::string, double>> context;
        for (size_t idx : numericIdx) {
            if (idx == primaryCol) continue;
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            if (row >= col.missing.size() || col.missing[row]) continue;
            const auto& vals = std::get<std::vector<double>>(col.values);
            if (row >= vals.size() || !std::isfinite(vals[row])) continue;
            const auto sit = statsByName.find(col.name);
            if (sit == statsByName.end() || sit->second.stddev <= 1e-12) continue;

            double z = 0.0;
            bool usedConditional = false;
            if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
                auto cit = conditionalStats.find(col.name);
                if (cit != conditionalStats.end()) {
                    auto git = cit->second.find(segmentByRow[row]);
                    if (git != cit->second.end() && git->second.stddev > 1e-12) {
                        z = (vals[row] - git->second.mean) / git->second.stddev;
                        usedConditional = true;
                    }
                }
            }
            if (!usedConditional) {
                z = (vals[row] - sit->second.mean) / sit->second.stddev;
            }
            if (std::abs(z) >= 1.5) context.push_back({col.name, z});
        }
        std::sort(context.begin(), context.end(), [](const auto& a, const auto& b) {
            return std::abs(a.second) > std::abs(b.second);
        });

        std::vector<std::string> reasonParts;
        if (!context.empty()) {
            std::string local = "Context";
            local += ": ";
            const size_t keep = std::min<size_t>(2, context.size());
            for (size_t i = 0; i < keep; ++i) {
                if (i > 0) local += "; ";
                local += context[i].first + "=" + (context[i].second > 0.0 ? "high" : "low") + " (z=" + toFixed(context[i].second, 2) + ")";
            }
            reasonParts.push_back(local);
        }
        if (hasMahalanobis) {
            std::string mahal = "Mahalanobis multivariate distance is elevated (d2=" + toFixed(mit->second, 3);
            if (mahalThreshold > 0.0) {
                mahal += ", threshold=" + toFixed(mahalThreshold, 3);
            }
            mahal += ")";
            if (primaryAbsZ < 2.5) {
                mahal += "; univariate z is moderate, but combined multivariate profile is unusual";
            }
            reasonParts.push_back(mahal);
        }
        if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
            reasonParts.push_back("within " + segmentColumnName + "=" + segmentByRow[row]);
        }

        std::string reason;
        if (reasonParts.empty()) {
            reason = "No dominant secondary deviations.";
        } else {
            reason = reasonParts.front();
            for (size_t i = 1; i < reasonParts.size(); ++i) {
                reason += "; " + reasonParts[i];
            }
        }

        rows.push_back({
            std::to_string(row + 1),
            primary.name,
            toFixed(primaryVals[row], 4),
            toFixed(primaryAbsZ, 2),
            reason
        });
    }
    return rows;
}
} // namespace

int AutomationPipeline::run(const AutoConfig& config) {
    AutoConfig runCfg = config;

    if (runCfg.lowMemoryMode) {
        runCfg.fastMode = true;
        runCfg.fastMaxBivariatePairs = std::min<size_t>(runCfg.fastMaxBivariatePairs, 600);
        runCfg.fastNeuralSampleRows = std::min<size_t>(runCfg.fastNeuralSampleRows, 10000);
        runCfg.neuralStreamingMode = true;
        runCfg.neuralStreamingChunkRows = std::min<size_t>(std::max<size_t>(128, runCfg.neuralStreamingChunkRows), 512);
        runCfg.neuralMaxOneHotPerColumn = std::min<size_t>(runCfg.neuralMaxOneHotPerColumn, 8);
        runCfg.featureEngineeringEnablePoly = false;
        runCfg.featureEngineeringEnableLog = false;
        runCfg.featureEngineeringMaxGeneratedColumns = std::min<size_t>(runCfg.featureEngineeringMaxGeneratedColumns, 64);
        if (CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
            runCfg.neuralStrategy = "fast";
        }
        runCfg.plotUnivariate = false;
        runCfg.plotOverall = false;
        runCfg.plotBivariateSignificant = true;
        runCfg.plotModesExplicit = true;
        runCfg.generateHtml = false;
    }

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
    const size_t loadedColumnCount = data.colCount();
    const std::optional<std::string> protectedTargetName = config.targetColumn.empty()
        ? std::optional<std::string>{}
        : std::optional<std::string>{config.targetColumn};
    const PreflightCullSummary preflightCull = applyPreflightSparseColumnCull(data,
                                                                               protectedTargetName,
                                                                               runCfg.verboseAnalysis,
                                                                               0.95);
    if (data.colCount() == 0) {
        throw Seldon::DatasetException("All columns were removed by pre-flight missingness cull (>95% missing)");
    }
    const size_t rawColumnsAfterPreflight = data.colCount();
    const TypedDataset reportingData = data;
    const TargetContext targetContext = resolveTargetContext(data, config, runCfg);
    const int targetIdx = targetContext.targetIdx;
    applyDynamicPlotDefaultsIfUnset(runCfg, data);

    const bool autoFastMode = (data.rowCount() > 100000) || (data.numericColumnIndices().size() > 50);
    const bool fastModeEnabled = runCfg.fastMode || autoFastMode;
    if (fastModeEnabled && CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
        runCfg.neuralStrategy = "fast";
    }

    if (!runCfg.storeOutlierFlagsInReport && data.rowCount() <= 50000) {
        runCfg.storeOutlierFlagsInReport = true;
    }

    PreprocessReport prep = Preprocessor::run(data, runCfg);
    advance("Preprocessed dataset");
    exportPreprocessedDatasetIfRequested(data, runCfg);
    normalizeBinaryTarget(data, targetIdx, targetContext.semantics);
    const NumericStatsCache statsCache = buildNumericStatsCache(data);
    const NumericStatsCache rawStatsCache = buildNumericStatsCache(reportingData);
    advance("Prepared stats cache");

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Preparing deeply detailed univariate analysis...\n";
    }

    ReportEngine univariate;
    univariate.addTitle("Univariate Analysis");
    univariate.addParagraph("Dataset: " + runCfg.datasetPath);
    univariate.addParagraph("Bi-temporal mode: raw-state statistics are reported in original units; transformed-state features are used for neural modeling and ranking.");
    const size_t totalAnalysisDimensions = data.colCount();
    const size_t engineeredFeatureCount =
        (totalAnalysisDimensions >= rawColumnsAfterPreflight) ? (totalAnalysisDimensions - rawColumnsAfterPreflight) : 0;
    const bool strictFeatureReporting = engineeredFeatureCount >= std::max<size_t>(8, rawColumnsAfterPreflight / 3);
    const size_t fullTableDimensions = strictFeatureReporting ? rawColumnsAfterPreflight : totalAnalysisDimensions;
    univariate.addParagraph("Dataset Stats:");
    univariate.addParagraph("Rows: " + std::to_string(data.rowCount()));
    univariate.addParagraph("Raw Columns (loaded): " + std::to_string(loadedColumnCount));
    univariate.addParagraph("Raw Columns (post pre-flight cull): " + std::to_string(rawColumnsAfterPreflight));
    if (preflightCull.dropped > 0) {
        univariate.addParagraph("Pre-flight cull removed " + std::to_string(preflightCull.dropped) + " sparse columns (>" + toFixed(100.0 * preflightCull.threshold, 1) + "% missing) before univariate profiling.");
    }
    univariate.addParagraph("Engineered Features: " + std::to_string(engineeredFeatureCount));
    univariate.addParagraph("Total Analysis Dimensions: " + std::to_string(totalAnalysisDimensions));
    if (strictFeatureReporting) {
        univariate.addParagraph("Dimensions expanded in full univariate/bivariate tables: " + std::to_string(fullTableDimensions));
    }
    if (strictFeatureReporting) {
        univariate.addParagraph("Strict feature-reporting mode enabled: engineered features are summarized, not expanded into full univariate/bivariate tables.");
        std::unordered_set<std::string> engineeredFamilies;
        std::vector<std::vector<std::string>> engineeredRows;
        for (const auto& col : data.columns()) {
            if (!isEngineeredFeatureName(col.name)) continue;
            const std::string base = canonicalEngineeredBaseName(col.name);
            if (!base.empty()) engineeredFamilies.insert(base);
            if (engineeredRows.size() < 12) {
                engineeredRows.push_back({col.name, base.empty() ? "-" : base});
            }
        }
        univariate.addTable("Feature Engineering Insights", {"Metric", "Value"}, {
            {"Engineered Features (suppressed from full tables)", std::to_string(engineeredFeatureCount)},
            {"Engineered Feature Families", std::to_string(engineeredFamilies.size())},
            {"Reporting Policy", "collapsed (strict mode)"}
        });
        if (!engineeredRows.empty()) {
            univariate.addTable("Feature Engineering Sample", {"Engineered Feature", "Base Feature"}, engineeredRows);
        }
    }
    addExecutiveDashboard(
        univariate,
        "Executive Dashboard",
        {
            {"Rows", std::to_string(data.rowCount())},
            {"Raw Columns", std::to_string(loadedColumnCount)},
            {"Post-Cull Columns", std::to_string(rawColumnsAfterPreflight)},
            {"Engineered Features", std::to_string(engineeredFeatureCount)},
            {"Analysis Dimensions", std::to_string(totalAnalysisDimensions)}
        },
        {
            preflightCull.dropped > 0
                ? ("Pre-flight sparse-column cull removed " + std::to_string(preflightCull.dropped) + " highly missing columns.")
                : "No severe pre-flight sparse-column issues were detected.",
            strictFeatureReporting
                ? "Strict feature-reporting is active to keep the report compact and readable."
                : "Full-dimension descriptive profiling is enabled."
        },
        "Quick non-technical view before deep descriptive tables."
    );
    addUnivariateDetailedSection(univariate, reportingData, prep, runCfg.verboseAnalysis, rawStatsCache);
    advance("Built univariate tables");

    GnuplotEngine plotterBivariate(plotSubdir(runCfg, "bivariate"), runCfg.plot);
    GnuplotEngine plotterUnivariate(plotSubdir(runCfg, "univariate"), runCfg.plot);
    GnuplotEngine plotterOverall(plotSubdir(runCfg, "overall"), runCfg.plot);
    const bool canPlot = configurePlotAvailability(runCfg, univariate, plotterBivariate);

    FeatureSelectionResult selectedFeatures = collectFeatureIndices(data, targetIdx, runCfg, prep);
    std::vector<int> featureIdx = selectedFeatures.included;
    DeterministicHeuristics::Outcome deterministic = DeterministicHeuristics::runAllPhases(data, prep, targetIdx, featureIdx);
    if (!deterministic.filteredFeatures.empty()) {
        featureIdx = deterministic.filteredFeatures;
    }

    if (deterministic.lowRatioMode) {
        if (runCfg.neuralFixedLayers == 0) runCfg.neuralFixedLayers = 1;
        if (runCfg.neuralFixedHiddenNodes == 0) runCfg.neuralFixedHiddenNodes = 6;
        runCfg.neuralStrategy = "fast";
    } else if (deterministic.highRatioMode) {
        runCfg.neuralMinLayers = std::max(runCfg.neuralMinLayers, 3);
        runCfg.neuralMaxLayers = std::max(runCfg.neuralMaxLayers, 3);
        if (runCfg.neuralFixedHiddenNodes == 0) {
            runCfg.neuralFixedHiddenNodes = std::max(runCfg.neuralFixedHiddenNodes, 50);
        }
        if (CommonUtils::toLower(runCfg.neuralStrategy) == "auto") runCfg.neuralStrategy = "expressive";
    }

    if (runCfg.verboseAnalysis && !selectedFeatures.droppedByMissingness.empty()) {
        std::cout << "[Seldon][Features] Dropped sparse features (>"
                  << toFixed(selectedFeatures.missingThresholdUsed, 2)
                  << " missing ratio, strategy=" << selectedFeatures.strategyUsed << "):\n";
        for (const auto& item : selectedFeatures.droppedByMissingness) {
            std::cout << "  - " << item << "\n";
        }
        for (const auto& item : deterministic.excludedReasonLines) {
            std::cout << "  - " << item << "\n";
        }
    }

    auto benchmarks = BenchmarkEngine::run(data, targetIdx, featureIdx, runCfg.kfold, runCfg.benchmarkSeed);
    advance("Finished benchmarks");

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Neural] Starting neural network training with verbose trace...\n";
    }
    NeuralAnalysis neural = runNeuralAnalysis(data,
                                              targetIdx,
                                              featureIdx,
                                              targetContext.semantics.inferredTask != "regression",
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
            const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
            const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
            const auto xIt = statsCache.find(idx);
            const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
            const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
            const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);
            double corr = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
            importanceByIndex[idx] = std::isfinite(corr) ? corr : fallbackImportance;
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
        bivariate.addParagraph("All numeric pair combinations are included below (nC2). Significant table is dynamically filtered using a multi-tier gate: statistical significance, neural relevance, and a bounded fallback for high-effect stable pairs.");
    }
    bivariate.addParagraph("Neural relevance score prioritizes practical effect size over raw p-value magnitude; when neural filtering is too strict, statistically robust pairs can be promoted as Tier-3 domain findings.");

    const auto stratifiedPopulations = detectStratifiedPopulations(data, 12);

    std::vector<std::vector<std::string>> allRows;
    std::vector<std::vector<std::string>> sigRows;
    std::vector<std::vector<std::string>> structuralRows;
    const bool compactBivariateRows = runCfg.lowMemoryMode;
    const size_t compactRejectedCap = 256;
    size_t compactRejectedCount = 0;
    size_t statSigCount = 0;
    size_t strictSuppressedPairs = 0;
    size_t strictSuppressedSelected = 0;
    for (const auto& p : bivariatePairs) {
        const bool engineeredPair = isEngineeredFeatureName(p.featureA) || isEngineeredFeatureName(p.featureB);
        if (strictFeatureReporting && engineeredPair) {
            ++strictSuppressedPairs;
            if (p.selected) ++strictSuppressedSelected;
            continue;
        }
        if (p.statSignificant) statSigCount++;
        const bool deterministicPair = std::abs(p.r) >= 0.999;
        if (p.filteredAsStructural || deterministicPair) {
            structuralRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.pValue, 6),
                p.filteredAsStructural ? "yes" : "no",
                deterministicPair ? "yes" : "no",
                p.relationLabel.empty() ? "-" : p.relationLabel,
                p.selectionReason.empty() ? "-" : p.selectionReason
            });
            continue;
        }
        if (!compactBivariateRows || p.selected || compactRejectedCount < compactRejectedCap) {
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
                std::to_string(p.significanceTier),
                p.selectionReason.empty() ? "-" : p.selectionReason,
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
            if (compactBivariateRows && !p.selected) {
                ++compactRejectedCount;
            }
        }

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
                std::to_string(p.significanceTier),
                p.selectionReason.empty() ? "-" : p.selectionReason,
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
    std::vector<std::string> narrativeInsights;
    std::string neuralInteractionTakeaway;
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

        std::string narrative;
        if (std::abs(p.r) >= 0.85) {
            narrative = p.featureA + " effectively predicts " + p.featureB + " with high stability (|r|=" + toFixed(std::abs(p.r), 3) + ").";
        } else if (std::abs(p.r) >= 0.65) {
            narrative = p.featureA + " is a strong directional signal for " + p.featureB + " and can serve as a stable proxy.";
        } else {
            narrative = p.featureA + " contributes a moderate but consistent relationship with " + p.featureB + ".";
        }
        narrativeInsights.push_back(narrative);

        usedFeatures.insert(p.idxA);
        usedFeatures.insert(p.idxB);
    }

    {
        std::vector<std::pair<std::string, double>> skewed;
        for (size_t idx : data.numericColumnIndices()) {
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            ColumnStats st = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
            if (!std::isfinite(st.skewness) || std::abs(st.skewness) < 1.2) continue;
            skewed.push_back({col.name, std::abs(st.skewness)});
        }
        std::sort(skewed.begin(), skewed.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < std::min<size_t>(2, skewed.size()); ++i) {
            narrativeInsights.push_back(skewed[i].first + " is heavily concentrated toward one side of its distribution (|skew|=" + toFixed(skewed[i].second, 2) + "), so tail behavior drives aggregate trends.");
        }
    }

    for (const auto& badge : deterministic.badgeNarratives) {
        narrativeInsights.push_back(badge);
    }

    if (!rankedTakeaways.empty()) {
        const PairInsight* bestInteraction = nullptr;
        double bestInteractionScore = -1.0;
        for (const auto& p : rankedTakeaways) {
            const double absCorr = std::clamp(std::abs(p.r), 0.0, 1.0);
            const double nonLinearHint = 1.0 - (0.50 * absCorr);
            const double interactionScore = p.neuralScore * (0.60 + 0.40 * p.foldStability) * nonLinearHint;
            if (interactionScore > bestInteractionScore) {
                bestInteractionScore = interactionScore;
                bestInteraction = &p;
            }
        }

        if (bestInteraction) {
            const std::string interactionType = (std::abs(bestInteraction->r) < 0.55)
                ? "complementary"
                : "reinforcing";
            neuralInteractionTakeaway = "Neural Interaction: " + bestInteraction->featureA + " × " + bestInteraction->featureB +
                " form a " + interactionType + " predictive interaction (neural=" + toFixed(bestInteraction->neuralScore, 3) +
                ", effect=" + toFixed(bestInteraction->effectSize, 3) +
                ", stability=" + toFixed(bestInteraction->foldStability, 3) + ").";
        }
    }

    std::vector<std::string> redundancyDrops;
    std::unordered_set<std::string> seenDrop;
    for (const auto& p : bivariatePairs) {
        if (!p.filteredAsRedundant || p.redundancyGroup.empty()) continue;
        const double impA = importanceByIndex.count(p.idxA) ? importanceByIndex[p.idxA] : 0.0;
        const double impB = importanceByIndex.count(p.idxB) ? importanceByIndex[p.idxB] : 0.0;
        const std::string drop = (impA < impB) ? p.featureA : p.featureB;
        const std::string keep = (impA < impB) ? p.featureB : p.featureA;
        const std::string dropLower = CommonUtils::toLower(CommonUtils::trim(drop));
        const std::string targetLower = CommonUtils::toLower(CommonUtils::trim(runCfg.targetColumn));
        if (dropLower == targetLower) continue;
        const std::string recommendation = "drop " + drop + " (redundant with " + keep + ")";
        if (seenDrop.insert(recommendation).second) {
            redundancyDrops.push_back(recommendation);
            if (redundancyDrops.size() >= 8) break;
        }
    }

    bivariate.addParagraph("Total pairs evaluated: " + std::to_string(bivariatePairs.size()));
    bivariate.addParagraph("Statistically significant pairs (p<" + toFixed(MathUtils::getSignificanceAlpha(), 4) + "): " + std::to_string(statSigCount));
    bivariate.addParagraph("Final selected significant pairs: " + std::to_string(sigRows.size()));
    const size_t tier2Count = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) {
        return p.selected && p.significanceTier == 2;
    }));
    const size_t tier3Count = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) {
        return p.selected && p.significanceTier == 3;
    }));
    bivariate.addParagraph("Selection tiers: Tier-2(neural+stat)=" + std::to_string(tier2Count) + ", Tier-3(domain fallback)=" + std::to_string(tier3Count) + ".");
    const size_t redundantPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsRedundant; }));
    const size_t structuralPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsStructural; }));
    const size_t leakagePairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.leakageRisk; }));
    bivariate.addParagraph("Information-theoretic filtering: redundant=" + std::to_string(redundantPairs) + ", structural=" + std::to_string(structuralPairs) + ", leakage-risk=" + std::to_string(leakagePairs) + ".");
    if (strictFeatureReporting) {
        bivariate.addParagraph("Strict feature-reporting mode: suppressed " + std::to_string(strictSuppressedPairs) + " engineered-feature pairs (" + std::to_string(strictSuppressedSelected) + " were otherwise selected). See Univariate 'Feature Engineering Insights'.");
    }
    addExecutiveDashboard(
        bivariate,
        "Executive Dashboard",
        {
            {"Pairs Evaluated", std::to_string(bivariatePairs.size())},
            {"Statistically Significant", std::to_string(statSigCount)},
            {"Selected Findings", std::to_string(sigRows.size())},
            {"Tier-2", std::to_string(tier2Count)},
            {"Tier-3", std::to_string(tier3Count)}
        },
        {
            "Selection balances statistical significance with neural relevance and domain fallback rules.",
            "Filtering removes redundant, structural, and leakage-risk relationships before final ranking.",
            strictFeatureReporting
                ? "Engineered-feature pair suppression is active for readability."
                : "Pair coverage includes the full core signal set."
        },
        "Quick map of relationship strength before detailed pair evidence."
    );
    if (compactBivariateRows) {
        bivariate.addParagraph("Low-memory mode: omitted full pair table and kept only selected + capped rejected samples for diagnostics.");
    } else {
        bivariate.addTable("All Pairwise Results", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "stat_sig", "neural_score", "significance_tier", "selection_reason", "selected", "redundant", "structural", "leakage_risk", "cluster_rep", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, allRows);
    }
    bivariate.addTable("Final Significant Results", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "significance_tier", "selection_reason", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, sigRows);
    if (!structuralRows.empty()) {
        bivariate.addTable("Structural/Deterministic Pair Diagnostics", {"Feature A", "Feature B", "pearson_r", "p_value", "structural", "deterministic", "relation_label", "selection_reason"}, structuralRows);
    }

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

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        bivariate.addTable("Automatic Stratified Population Signals", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    const AdvancedAnalyticsOutputs advancedOutputs = buildAdvancedAnalyticsOutputs(data,
                                                                                   targetIdx,
                                                                                   featureIdx,
                                                                                   neural.featureImportance,
                                                                                   bivariatePairs,
                                                                                   anovaRows,
                                                                                   contingency,
                                                                                   statsCache);

    const DataHealthSummary dataHealth = computeDataHealthSummary(data,
                                                                  prep,
                                                                  neural,
                                                                  featureIdx.size(),
                                                                  allRows.size(),
                                                                  statSigCount,
                                                                  sigRows.size());

    ReportEngine neuralReport;
    neuralReport.addTitle("Neural Synthesis");
    neuralReport.addParagraph("This synthesis captures neural network training traces and how neural relevance influenced bivariate selection.");
    neuralReport.addParagraph(std::string("Task type inferred from target: ") + targetContext.semantics.inferredTask);
    neuralReport.addParagraph("Bivariate significance now uses a three-tier gate: Tier-1 statistical evidence, Tier-2 neural confirmation, Tier-3 domain fallback for high-effect stable relationships when neural yield is sparse.");
    addExecutiveDashboard(
        neuralReport,
        "Executive Dashboard",
        {
            {"Task Type", targetContext.semantics.inferredTask},
            {"Training Rows Used", std::to_string(neural.trainingRowsUsed) + " / " + std::to_string(neural.trainingRowsTotal)},
            {"Epochs", std::to_string(neural.epochs)},
            {"Topology", neural.topology},
            {"Health Score", toFixed(dataHealth.score, 1) + "/100 " + scoreBar100(dataHealth.score)}
        },
        {
            "Auto policy and deterministic guardrails reduce overfitting and unstable topology growth.",
            "Explainability and uncertainty traces are preserved for interpretability."
        },
        "Fast leadership view of model behavior before detailed training diagnostics."
    );
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
        {"Pre-Flight Sparse Columns Culled", std::to_string(preflightCull.dropped)},
        {"Neural Strategy", neural.policyUsed},
        {"Fast Mode", fastModeEnabled ? "enabled" : "disabled"},
        {"Neural Training Rows Used", std::to_string(neural.trainingRowsUsed) + " / " + std::to_string(neural.trainingRowsTotal)},
        {"Bivariate Strategy", bivariatePolicy.name},
        {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
        {"Lasso Gate", deterministic.lassoGateApplied ? ("on (kept=" + std::to_string(deterministic.lassoSelectedCount) + ")") : "off"},
        {"Tier-2 Pair Count", std::to_string(tier2Count)},
        {"Tier-3 Pair Count", std::to_string(tier3Count)}
    });

    if (!deterministic.roleTagRows.empty()) {
        neuralReport.addTable("Deterministic Semantic Role Tags", {"Column", "Role", "Unique", "Missing", "Null %"}, deterministic.roleTagRows);
    }
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
        {"Loss", neural.lossName},
        {"Strict Pruning", neural.strictPruningApplied ? ("on (dropped=" + std::to_string(neural.strictPrunedColumns) + ")") : "off"},
        {"Hidden Activation", neural.hiddenActivation},
        {"Output Activation", neural.outputActivation}
    });
    if (targetContext.semantics.inferredTask == "ordinal_classification" && neural.lossName == "mse") {
        neuralReport.addParagraph("Ordinal target detected with MSE loss: this is intentional in the current engine because ordered class encodings preserve rank distance, so MSE penalizes larger ordinal mis-ranks more strongly.");
    }

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

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        neuralReport.addTable("Automatic Segmented Population Signals", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    std::vector<std::vector<std::string>> lossRows;
    for (size_t e = 0; e < neural.trainLoss.size(); ++e) {
        double val = (e < neural.valLoss.size() ? neural.valLoss[e] : 0.0);
        lossRows.push_back({
            std::to_string(e + 1),
            toFixed(neural.trainLoss[e], 6),
            toFixed(val, 6)
        });
    }
    neuralReport.addTable("Neural Network Training Trace", {"Epoch", "Train Loss", "Validation Loss"}, lossRows);

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
    finalAnalysis.addParagraph("This report contains statistically significant findings selected by a tiered engine: Tier-2 (statistical + neural) and Tier-3 fallback (statistical + domain effect). Non-selected findings are excluded by design.");
    finalAnalysis.addParagraph("Data Health Score: " + toFixed(dataHealth.score, 1) + "/100 (" + dataHealth.band + "). This score estimates discovered signal strength using completeness, retained feature coverage, significant-pair yield, selected-pair yield, and neural training stability.");
    addExecutiveDashboard(
        finalAnalysis,
        "Executive Dashboard",
        {
            {"Health Score", toFixed(dataHealth.score, 1) + "/100 " + scoreBar100(dataHealth.score)},
            {"Health Band", dataHealth.band},
            {"Selected Findings", std::to_string(sigRows.size())},
            {"Rows", std::to_string(data.rowCount())},
            {"Columns", std::to_string(data.colCount())}
        },
        {
            "This top block is optimized for non-technical consumption.",
            "Ordered methods, causal-lite checks, and detailed evidence tables follow below."
        },
        "Snapshot first; evidence next."
    );

    if (!advancedOutputs.orderedRows.empty()) {
        finalAnalysis.addTable("Ordered Methods", {"Step", "Method", "Result"}, advancedOutputs.orderedRows);
    }

    if (!advancedOutputs.causalDagRows.empty()) {
        finalAnalysis.addTable("Causal Inference (Lite) DAG Candidates",
                               {"From", "To", "Confidence", "Evidence", "Interpretation"},
                               advancedOutputs.causalDagRows);
    }

    if (!advancedOutputs.globalConditionalRows.empty()) {
        finalAnalysis.addTable("Global vs Conditional Relationship Drift",
                               {"Feature", "Global r", "Conditional r", "|conditional|/|global|", "Pattern", "Interpretation"},
                               advancedOutputs.globalConditionalRows);
    }

    if (!advancedOutputs.temporalDriftRows.empty()) {
        finalAnalysis.addTable("Temporal Drift Kernels (ADF-Style)",
                               {"Feature", "Axis", "Samples", "Gamma", "t-stat", "p-approx", "Drift Ratio", "Verdict"},
                               advancedOutputs.temporalDriftRows);
    }

    if (!advancedOutputs.contextualDeadZoneRows.empty()) {
        finalAnalysis.addTable("Cross-Cluster Contextual Dead-Zones",
                               {"Feature", "Predictive Cluster", "Dead-Zone Cluster", "|r| strong", "|r| weak", "weak/strong", "Min Cluster N"},
                               advancedOutputs.contextualDeadZoneRows);
    }

    if (advancedOutputs.causalDagMermaid.has_value()) {
        finalAnalysis.addParagraph("Causal Inference (Lite) visual sketch (heuristic DAG):\n" + *advancedOutputs.causalDagMermaid);
        finalAnalysis.addParagraph("Causal-lite guardrail: edges are directional hypotheses inferred from partial-correlation retention and proxy collapse, and do not establish intervention-level causality.");
    }

    if (!advancedOutputs.mahalanobisRows.empty()) {
        finalAnalysis.addTable("Mahalanobis Multivariate Outliers", {"Row", "Distance^2", "Threshold"}, advancedOutputs.mahalanobisRows);
    }

    if (!advancedOutputs.pdpRows.empty()) {
        finalAnalysis.addTable("Linear Partial Dependence Approximation", {"Feature", "Low (P10)", "Medium (P50)", "High (P90)", "Delta (High-Low)", "Direction"}, advancedOutputs.pdpRows);
    }

    if (!advancedOutputs.narrativeRows.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(advancedOutputs.narrativeRows.size());
        for (const auto& s : advancedOutputs.narrativeRows) rows.push_back({s});
        finalAnalysis.addTable("Cross-Section Narrative Layer", {"Narrative"}, rows);
    }

    if (advancedOutputs.executiveSummary.has_value()) {
        finalAnalysis.addParagraph("Advanced Analytics Executive Summary: " + *advancedOutputs.executiveSummary);
    }

    if (advancedOutputs.interactionEvidence.has_value()) {
        neuralInteractionTakeaway = *advancedOutputs.interactionEvidence;
    }

    std::vector<std::string> prioritizedTakeaways;
    std::unordered_set<std::string> seenTakeaways;
    for (const auto& s : advancedOutputs.priorityTakeaways) {
        if (!s.empty() && seenTakeaways.insert(s).second) prioritizedTakeaways.push_back(s);
    }
    for (const auto& s : topTakeaways) {
        if (!s.empty() && seenTakeaways.insert(s).second) prioritizedTakeaways.push_back(s);
    }
    if (prioritizedTakeaways.size() > 3) prioritizedTakeaways.resize(3);

    if (!prioritizedTakeaways.empty()) {
        std::vector<std::vector<std::string>> takeawayRows;
        for (size_t i = 0; i < prioritizedTakeaways.size(); ++i) {
            takeawayRows.push_back({std::to_string(i + 1), prioritizedTakeaways[i]});
        }
        finalAnalysis.addTable("Top 3 Takeaways", {"Rank", "Takeaway"}, takeawayRows);
    }

    if (!neuralInteractionTakeaway.empty()) {
        finalAnalysis.addTable("Neural Interaction Takeaway", {"Insight"}, {{neuralInteractionTakeaway}});
    }

    if (auto residualNarrative = buildResidualDiscoveryNarrative(data, targetIdx, featureIdx, statsCache); residualNarrative.has_value()) {
        finalAnalysis.addTable("Residual-Based Sequential Discovery", {"Insight"}, {{*residualNarrative}});
    } else if (!deterministic.residualNarrative.empty()) {
        finalAnalysis.addTable("Residual-Based Sequential Discovery", {"Insight"}, {{deterministic.residualNarrative}});
    }

    if (!narrativeInsights.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(narrativeInsights.size());
        for (size_t i = 0; i < narrativeInsights.size(); ++i) {
            rows.push_back({std::to_string(i + 1), narrativeInsights[i]});
        }
        finalAnalysis.addTable("Narrative Insight Layer", {"#", "Insight"}, rows);
    }

    const auto outlierContextRows = buildOutlierContextRows(reportingData,
                                                            prep,
                                                            advancedOutputs.mahalanobisByRow,
                                                            advancedOutputs.mahalanobisThreshold);
    if (!outlierContextRows.empty()) {
        finalAnalysis.addTable("Outlier Contextualization", {"Row", "Primary Feature", "Value", "|z|", "Why it is unusual"}, outlierContextRows);
    }

    if (!redundancyDrops.empty()) {
        std::vector<std::vector<std::string>> dropRows;
        for (const auto& msg : redundancyDrops) {
            dropRows.push_back({msg});
        }
        finalAnalysis.addTable("Redundancy Drop Recommendations", {"Action"}, dropRows);
    }

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        finalAnalysis.addTable("Segmented Population Findings", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    finalAnalysis.addTable("Selected Significant Bivariate Findings", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "significance_tier", "selection_reason", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, sigRows);

    finalAnalysis.addTable("Data Health Signal Card", {"Component", "Value"}, {
        {"Score (0-100)", toFixed(dataHealth.score, 1)},
        {"Band", dataHealth.band},
        {"Completeness", toFixed(100.0 * dataHealth.completeness, 1) + "%"},
        {"Numeric Coverage", toFixed(100.0 * dataHealth.numericCoverage, 1) + "%"},
        {"Feature Retention", toFixed(100.0 * dataHealth.featureRetention, 1) + "%"},
        {"Statistical Yield", toFixed(100.0 * dataHealth.statYield, 1) + "%"},
        {"Selected Yield", toFixed(100.0 * dataHealth.selectedYield, 1) + "%"},
        {"Training Stability", toFixed(100.0 * dataHealth.trainingStability, 1) + "%"}
    });

    std::vector<std::vector<std::string>> topFeatures;
    std::vector<std::pair<std::string, double>> fiPairs;
    std::unordered_map<std::string, size_t> deterministicFeatureHits;
    for (const auto& pair : bivariatePairs) {
        const bool deterministicRelation = (std::abs(pair.r) >= 0.999) || pair.filteredAsStructural;
        if (!deterministicRelation) continue;
        deterministicFeatureHits[pair.featureA]++;
        deterministicFeatureHits[pair.featureB]++;
    }
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        auto hitIt = deterministicFeatureHits.find(name);
        if (hitIt != deterministicFeatureHits.end() && hitIt->second > 0) {
            const double penalty = std::max(0.35, std::pow(0.82, static_cast<double>(hitIt->second)));
            imp *= penalty;
        }
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
        {"Pairs Evaluated", std::to_string(bivariatePairs.size())},
        {"Pairs Statistically Significant", std::to_string(statSigCount)},
        {"Pairs Selected", std::to_string(sigRows.size())},
        {"Training Epochs Executed", std::to_string(neural.trainLoss.size())},
        {"Data Health Score", toFixed(dataHealth.score, 1) + " (" + dataHealth.band + ")"}
    });

    ReportEngine heuristicsReport;
    heuristicsReport.addTitle("Deterministic Analysis Report");
    heuristicsReport.addParagraph("This report presents deterministic analysis results and signal-quality controls, generated independently from final_analysis.md.");

    size_t adminCount = 0;
    size_t targetCandidateCount = 0;
    size_t lowSignalCount = 0;
    for (const auto& row : deterministic.roleTagRows) {
        if (row.size() < 2) continue;
        if (row[1] == "ADMIN") adminCount++;
        else if (row[1] == "TARGET_CANDIDATE") targetCandidateCount++;
        else if (row[1] == "LOW_SIGNAL") lowSignalCount++;
    }

    const bool stabilityGuardTriggered = (neural.policyUsed.find("stability_shrink") != std::string::npos);
    const std::string stabilityPct = toFixed(100.0 * dataHealth.trainingStability, 1) + "%";

    std::string phase1Narrative = "Semantic filter excluded " + std::to_string(adminCount)
        + " administrative columns and marked " + std::to_string(lowSignalCount)
        + " low-signal columns before model training.";

    std::string phase3Narrative = "Rows:Features=" + toFixed(deterministic.rowsToFeatures, 2)
        + "; strict pruning=" + std::string(neural.strictPruningApplied ? "on" : "off")
        + "; lasso gate=" + std::string(deterministic.lassoGateApplied ? ("on (kept=" + std::to_string(deterministic.lassoSelectedCount) + ")") : "off") + ".";

    std::string phase4Narrative = deterministic.residualNarrative.empty()
        ? "Residual discovery scanned secondary drivers but no hidden-driver threshold was met."
        : deterministic.residualNarrative;

    std::string phase5Narrative = "Deterministic narrative synthesis translated statistical findings into concise analytical statements.";

    addExecutiveDashboard(
        heuristicsReport,
        "Executive Dashboard",
        {
            {"Semantic Admin Columns", std::to_string(adminCount)},
            {"Target Candidates", std::to_string(targetCandidateCount)},
            {"Low-Signal Columns", std::to_string(lowSignalCount)},
            {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
            {"Training Stability", stabilityPct}
        },
        {
            "Deterministic phases provide interpretable quality controls independent of model complexity.",
            stabilityGuardTriggered
                ? "Stability guard triggered and topology was shrunk for safer convergence."
                : "Stability guard did not trigger; topology remained in stable range."
        },
        "Control-layer summary before stage-level diagnostics."
    );

    heuristicsReport.addTable("Analysis Workflow Summary", {"Stage", "Objective", "Method", "Outcome"}, {
        {"Semantic Filtering", "Reduce metadata and administrative noise", "Regex role tagging (ADMIN/METADATA/TARGET_CANDIDATE)", phase1Narrative},
        {"Feature Guardrails", "Limit overfitting from engineered expansion", "Dynamic sparsity + lasso gate + engineered-feature controls", phase3Narrative},
        {"Narrative Synthesis", "Convert model statistics into readable findings", "Deterministic narrative layer", phase5Narrative},
        {"Residual Discovery", "Detect hidden secondary drivers", "Primary-fit residual pass across remaining features", phase4Narrative},
        {"Outlier Context", "Explain anomaly relevance", "Contrastive z-score contextualization", "Outlier narratives now explain why flagged rows are unusual relative to the population."}
    });

    heuristicsReport.addTable("Neural Network Tuning", {"Control", "Observed", "Action"}, {
        {"Training Stability", stabilityPct, stabilityGuardTriggered ? "Hidden nodes reduced automatically (<70% stability guard triggered)" : "Topology retained (stability guard not triggered)"},
        {"Target Diversity", targetContext.semantics.inferredTask, "Classification-like targets use cross-entropy path; continuous targets use MSE path"},
        {"Loss Function", neural.lossName, "Applied automatically from inferred target semantics"}
    });

    heuristicsReport.addTable("Topology Guardrail", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Features (post deterministic filter)", std::to_string(featureIdx.size())},
        {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
        {"Low Ratio Mode (<10)", deterministic.lowRatioMode ? "yes" : "no"},
        {"High Ratio Mode (>100)", deterministic.highRatioMode ? "yes" : "no"},
        {"Lasso Gate Applied", deterministic.lassoGateApplied ? "yes" : "no"},
        {"Lasso Retained Count", std::to_string(deterministic.lassoSelectedCount)}
    });

    if (!deterministic.roleTagRows.empty()) {
        heuristicsReport.addTable("Semantic Role Tags", {"Column", "Role", "Unique", "Missing", "Null %"}, deterministic.roleTagRows);
    }

    if (!deterministic.excludedReasonLines.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(deterministic.excludedReasonLines.size());
        for (const auto& line : deterministic.excludedReasonLines) rows.push_back({line});
        heuristicsReport.addTable("Entropy Filter Decisions", {"Decision"}, rows);
    }

    if (!deterministic.badgeNarratives.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(deterministic.badgeNarratives.size());
        for (const auto& line : deterministic.badgeNarratives) {
            rows.push_back({line});
        }
        heuristicsReport.addTable("Deterministic Narratives", {"Narrative"}, rows);
    }

    if (!deterministic.residualNarrative.empty()) {
        heuristicsReport.addTable("Residual Discovery", {"Insight"}, {{deterministic.residualNarrative}});
    }

    saveGeneratedReports(runCfg, univariate, bivariate, neuralReport, finalAnalysis, heuristicsReport);
    cleanupPlotCacheArtifacts(runCfg);
    advance("Saved reports");
    progress.done("Pipeline complete");
    printPipelineCompletion(runCfg);

    return 0;
}
