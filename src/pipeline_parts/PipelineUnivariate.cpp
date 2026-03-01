#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
#include "CausalDiscovery.h"
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
#include <spawn.h>
#include <set>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <cstdlib>
#ifdef SELDON_USE_NATIVE_PARQUET
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif

extern char** environ;

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
    const int devNull = ::open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (devNull < 0) return -1;

    const int devNullFlags = ::fcntl(devNull, F_GETFD);
    if (devNullFlags < 0 || ::fcntl(devNull, F_SETFD, devNullFlags | FD_CLOEXEC) < 0) {
        ::close(devNull);
        return -1;
    }

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        ::close(devNull);
        return -1;
    }

    if (::posix_spawn_file_actions_adddup2(&actions, devNull, STDOUT_FILENO) != 0 ||
        ::posix_spawn_file_actions_adddup2(&actions, devNull, STDERR_FILENO) != 0 ||
        ::posix_spawn_file_actions_addclose(&actions, devNull) != 0) {
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(devNull);
        return -1;
    }

    posix_spawnattr_t attr;
    if (::posix_spawnattr_init(&attr) != 0) {
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(devNull);
        return -1;
    }

    short spawnFlags = 0;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    spawnFlags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    if (spawnFlags != 0 && ::posix_spawnattr_setflags(&attr, spawnFlags) != 0) {
        ::posix_spawnattr_destroy(&attr);
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(devNull);
        return -1;
    }

    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int spawnRc = ::posix_spawn(&pid, executable.c_str(), &actions, &attr, argv.data(), environ);

    ::posix_spawnattr_destroy(&attr);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(devNull);

    if (spawnRc != 0 || pid <= 0) return -1;

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

#ifdef SELDON_USE_NATIVE_PARQUET
bool exportParquetNative(const TypedDataset& data,
                        const std::string& parquetPath,
                        std::string& errorOut) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(data.columns().size());
    arrays.reserve(data.columns().size());

    for (const auto& col : data.columns()) {
        if (col.type == ColumnType::NUMERIC) {
            arrow::DoubleBuilder builder;
            const auto& vals = std::get<std::vector<double>>(col.values);
            for (size_t r = 0; r < data.rowCount(); ++r) {
                if (r < col.missing.size() && col.missing[r]) {
                    if (!builder.AppendNull().ok()) {
                        errorOut = "Failed to append null for numeric column '" + col.name + "'";
                        return false;
                    }
                    continue;
                }
                const double v = (r < vals.size()) ? vals[r] : std::numeric_limits<double>::quiet_NaN();
                if (!std::isfinite(v)) {
                    if (!builder.AppendNull().ok()) {
                        errorOut = "Failed to append NaN-null for numeric column '" + col.name + "'";
                        return false;
                    }
                } else if (!builder.Append(v).ok()) {
                    errorOut = "Failed to append numeric value for column '" + col.name + "'";
                    return false;
                }
            }
            std::shared_ptr<arrow::Array> arr;
            auto status = builder.Finish(&arr);
            if (!status.ok()) {
                errorOut = "Failed to finalize numeric Arrow array for column '" + col.name + "': " + status.ToString();
                return false;
            }
            fields.push_back(arrow::field(col.name, arrow::float64(), true));
            arrays.push_back(arr);
        } else if (col.type == ColumnType::DATETIME) {
            arrow::Int64Builder builder;
            const auto& vals = std::get<std::vector<int64_t>>(col.values);
            for (size_t r = 0; r < data.rowCount(); ++r) {
                if (r < col.missing.size() && col.missing[r]) {
                    if (!builder.AppendNull().ok()) {
                        errorOut = "Failed to append null for datetime column '" + col.name + "'";
                        return false;
                    }
                    continue;
                }
                const int64_t v = (r < vals.size()) ? vals[r] : 0;
                if (!builder.Append(v).ok()) {
                    errorOut = "Failed to append datetime value for column '" + col.name + "'";
                    return false;
                }
            }
            std::shared_ptr<arrow::Array> arr;
            auto status = builder.Finish(&arr);
            if (!status.ok()) {
                errorOut = "Failed to finalize datetime Arrow array for column '" + col.name + "': " + status.ToString();
                return false;
            }
            fields.push_back(arrow::field(col.name, arrow::int64(), true));
            arrays.push_back(arr);
        } else {
            arrow::StringBuilder builder;
            const auto& vals = std::get<std::vector<std::string>>(col.values);
            for (size_t r = 0; r < data.rowCount(); ++r) {
                if (r < col.missing.size() && col.missing[r]) {
                    if (!builder.AppendNull().ok()) {
                        errorOut = "Failed to append null for categorical column '" + col.name + "'";
                        return false;
                    }
                    continue;
                }
                const std::string v = (r < vals.size()) ? vals[r] : "";
                if (!builder.Append(v).ok()) {
                    errorOut = "Failed to append categorical value for column '" + col.name + "'";
                    return false;
                }
            }
            std::shared_ptr<arrow::Array> arr;
            auto status = builder.Finish(&arr);
            if (!status.ok()) {
                errorOut = "Failed to finalize categorical Arrow array for column '" + col.name + "': " + status.ToString();
                return false;
            }
            fields.push_back(arrow::field(col.name, arrow::utf8(), true));
            arrays.push_back(arr);
        }
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, arrays, static_cast<int64_t>(data.rowCount()));

    auto outRes = arrow::io::FileOutputStream::Open(parquetPath);
    if (!outRes.ok()) {
        errorOut = "Failed to open parquet output path: " + outRes.status().ToString();
        return false;
    }
    std::shared_ptr<arrow::io::FileOutputStream> sink = outRes.ValueOrDie();

    const int64_t chunkRows = std::max<int64_t>(1024, std::min<int64_t>(65536, static_cast<int64_t>(data.rowCount())));
    auto writeStatus = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, chunkRows);
    if (!writeStatus.ok()) {
        errorOut = "Parquet write failed: " + writeStatus.ToString();
        return false;
    }
    auto closeStatus = sink->Close();
    if (!closeStatus.ok()) {
        errorOut = "Failed to close parquet output stream: " + closeStatus.ToString();
        return false;
    }
    return true;
}
#endif

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
        const auto parsed = CommonUtils::parseBooleanLikeToken(values[i]);
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

struct DomainRuleBundle {
    std::unordered_set<std::string> suppressCausalColumns;
    std::unordered_set<std::string> downweightImportanceColumns;
    std::vector<std::string> loadedFrom;
};

std::string normalizeRuleName(const std::string& value) {
    return CommonUtils::toLower(CommonUtils::trim(value));
}

void appendRuleValues(const std::string& rhs, std::unordered_set<std::string>& out) {
    std::stringstream ss(rhs);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string norm = normalizeRuleName(token);
        if (!norm.empty()) out.insert(norm);
    }
}

DomainRuleBundle loadDomainRules(const AutoConfig& config) {
    DomainRuleBundle out;
    std::vector<std::filesystem::path> candidates;
    if (!config.datasetPath.empty()) {
        const std::filesystem::path datasetPath(config.datasetPath);
        candidates.push_back(datasetPath.parent_path() / "domain_rules.txt");
        candidates.push_back(datasetPath.parent_path() / "seldon_domain_rules.txt");
        candidates.push_back(datasetPath.string() + ".rules.txt");
    }

    for (const auto& path : candidates) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) continue;
        std::ifstream in(path);
        if (!in.good()) continue;

        out.loadedFrom.push_back(path.string());
        std::string line;
        while (std::getline(in, line)) {
            const std::string trimmed = CommonUtils::trim(line);
            if (trimmed.empty() || trimmed[0] == '#') continue;
            const size_t pos = trimmed.find(':');
            if (pos == std::string::npos) continue;
            const std::string key = normalizeRuleName(trimmed.substr(0, pos));
            const std::string rhs = trimmed.substr(pos + 1);
            if (key == "suppress_causal" || key == "non_causal" || key == "constructed_index" || key == "block_causal") {
                appendRuleValues(rhs, out.suppressCausalColumns);
            } else if (key == "downweight_importance" || key == "deprioritize") {
                appendRuleValues(rhs, out.downweightImportanceColumns);
            }
        }
    }
    return out;
}

bool isCausalEdgeSuppressedByRule(const std::string& from,
                                  const std::string& to,
                                  const DomainRuleBundle& rules) {
    if (rules.suppressCausalColumns.empty()) return false;
    const std::string fromNorm = normalizeRuleName(from);
    const std::string toNorm = normalizeRuleName(to);
    return rules.suppressCausalColumns.count(fromNorm) > 0 ||
           rules.suppressCausalColumns.count(toNorm) > 0;
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

std::vector<std::string> engineeredRootTokens(const std::string& name) {
    std::vector<std::string> roots;
    const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
    if (lower.empty()) return roots;

    auto pushUnique = [&](const std::string& token) {
        const std::string t = CommonUtils::trim(token);
        if (t.empty()) return;
        if (std::find(roots.begin(), roots.end(), t) == roots.end()) {
            roots.push_back(t);
        }
    };

    const size_t mulPos = lower.find("_x_");
    if (mulPos != std::string::npos) {
        pushUnique(lower.substr(0, mulPos));
        pushUnique(lower.substr(mulPos + 3));
        return roots;
    }

    const size_t divPos = lower.find("_div_");
    if (divPos != std::string::npos) {
        pushUnique(lower.substr(0, divPos));
        pushUnique(lower.substr(divPos + 5));
        return roots;
    }

    if (const std::string base = canonicalEngineeredBaseName(lower); !base.empty()) {
        pushUnique(base);
    } else {
        pushUnique(lower);
    }
    return roots;
}

bool sharesEngineeredRootIdentity(const std::string& a, const std::string& b) {
    const bool engineeredA = isEngineeredFeatureName(a);
    const bool engineeredB = isEngineeredFeatureName(b);
    if (!engineeredA && !engineeredB) return false;

    const std::vector<std::string> rootsA = engineeredRootTokens(a);
    const std::vector<std::string> rootsB = engineeredRootTokens(b);
    if (rootsA.empty() || rootsB.empty()) return false;

    for (const auto& ra : rootsA) {
        if (std::find(rootsB.begin(), rootsB.end(), ra) != rootsB.end()) {
            return true;
        }
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

std::vector<size_t> buildNormalizedTopology(size_t inputNodes,
                                            size_t outputNodes,
                                            size_t hiddenLayers,
                                            size_t firstHidden,
                                            size_t maxHiddenNodes) {
    std::vector<size_t> topology;
    const size_t safeInput = std::max<size_t>(1, inputNodes);
    const size_t safeOutput = std::max<size_t>(1, outputNodes);
    const size_t safeLayers = std::max<size_t>(1, hiddenLayers);
    const size_t safeMaxHidden = std::max<size_t>(4, maxHiddenNodes);
    const size_t safeFirstHidden = std::clamp<size_t>(firstHidden, 4, safeMaxHidden);
    const size_t tailTarget = std::clamp<size_t>(std::max<size_t>(8, safeOutput * 2), 4, safeFirstHidden);

    topology.reserve(safeLayers + 2);
    topology.push_back(safeInput);

    size_t prevWidth = safeFirstHidden;
    for (size_t l = 0; l < safeLayers; ++l) {
        double widthD = static_cast<double>(safeFirstHidden);
        if (safeLayers > 1) {
            const double t = static_cast<double>(l) / static_cast<double>(safeLayers - 1);
            const double ratio = static_cast<double>(tailTarget) / static_cast<double>(safeFirstHidden);
            widthD = static_cast<double>(safeFirstHidden) * std::pow(ratio, t);
        }

        size_t width = std::clamp<size_t>(static_cast<size_t>(std::llround(widthD)), 4, safeMaxHidden);
        if (l > 0) {
            width = std::min(width, prevWidth > 4 ? prevWidth - 1 : static_cast<size_t>(4));
            const size_t maxDropStep = std::max<size_t>(4, static_cast<size_t>(std::ceil(static_cast<double>(prevWidth) / 2.8)));
            width = std::max(width, maxDropStep);
        }

        prevWidth = width;
        topology.push_back(width);
    }

    topology.push_back(safeOutput);
    return topology;
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

double quantileFromSorted(const std::vector<double>& sortedValues, double q) {
    if (sortedValues.empty()) return 0.0;
    const double qq = std::clamp(q, 0.0, 1.0);
    const double pos = qq * static_cast<double>(sortedValues.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sortedValues[lo];
    const double w = pos - static_cast<double>(lo);
    return sortedValues[lo] * (1.0 - w) + sortedValues[hi] * w;
}

double computePsiBetweenSlices(const std::vector<double>& reference,
                               const std::vector<double>& monitor,
                               size_t bins = 5) {
    if (reference.size() < 16 || monitor.size() < 16 || bins < 2) return 0.0;

    std::vector<double> sortedRef = reference;
    std::sort(sortedRef.begin(), sortedRef.end());

    std::vector<double> edges;
    edges.reserve(bins + 1);
    for (size_t b = 0; b <= bins; ++b) {
        const double q = static_cast<double>(b) / static_cast<double>(bins);
        edges.push_back(quantileFromSorted(sortedRef, q));
    }

    size_t uniqueEdges = 1;
    for (size_t i = 1; i < edges.size(); ++i) {
        if (std::abs(edges[i] - edges[i - 1]) > 1e-12) uniqueEdges++;
    }
    if (uniqueEdges < 3) return 0.0;

    std::vector<double> refCounts(bins, 0.0);
    std::vector<double> monCounts(bins, 0.0);

    auto assignBin = [&](double v) {
        if (v <= edges.front()) return static_cast<size_t>(0);
        if (v >= edges.back()) return bins - 1;
        auto it = std::upper_bound(edges.begin(), edges.end(), v);
        size_t idx = static_cast<size_t>(std::distance(edges.begin(), it));
        if (idx == 0) return static_cast<size_t>(0);
        return std::min(bins - 1, idx - 1);
    };

    for (double v : reference) {
        if (!std::isfinite(v)) continue;
        refCounts[assignBin(v)] += 1.0;
    }
    for (double v : monitor) {
        if (!std::isfinite(v)) continue;
        monCounts[assignBin(v)] += 1.0;
    }

    const double refTotal = std::accumulate(refCounts.begin(), refCounts.end(), 0.0);
    const double monTotal = std::accumulate(monCounts.begin(), monCounts.end(), 0.0);
    if (refTotal <= 0.0 || monTotal <= 0.0) return 0.0;

    constexpr double eps = 1e-6;
    double psi = 0.0;
    for (size_t b = 0; b < bins; ++b) {
        const double r = std::max(eps, refCounts[b] / refTotal);
        const double m = std::max(eps, monCounts[b] / monTotal);
        psi += (m - r) * std::log(m / r);
    }
    return std::max(0.0, psi);
}

OodDriftDiagnostics computeOodDriftDiagnostics(const std::vector<std::vector<double>>& X,
                                               const AutoConfig& config) {
    OodDriftDiagnostics out;
    if (!config.neuralOodEnabled || X.size() < 40 || X.front().empty()) return out;

    const size_t rows = X.size();
    const size_t cols = X.front().size();
    const size_t referenceRows = std::clamp<size_t>(static_cast<size_t>(std::llround(0.70 * static_cast<double>(rows))), 24, rows - 12);
    const size_t monitorRows = rows - referenceRows;
    if (referenceRows < 24 || monitorRows < 12) return out;

    std::vector<std::vector<double>> refCols(cols);
    std::vector<std::vector<double>> monCols(cols);
    for (size_t c = 0; c < cols; ++c) {
        refCols[c].reserve(referenceRows);
        monCols[c].reserve(monitorRows);
    }

    for (size_t r = 0; r < referenceRows; ++r) {
        const auto& row = X[r];
        if (row.size() != cols) continue;
        for (size_t c = 0; c < cols; ++c) {
            const double v = row[c];
            if (std::isfinite(v)) refCols[c].push_back(v);
        }
    }

    std::vector<double> muRaw(cols, 0.0);
    std::vector<double> sigmaRaw(cols, 1.0);
    for (size_t c = 0; c < cols; ++c) {
        if (refCols[c].empty()) continue;
        for (double v : refCols[c]) muRaw[c] += v;
        muRaw[c] /= static_cast<double>(refCols[c].size());
        double var = 0.0;
        for (double v : refCols[c]) {
            const double d = v - muRaw[c];
            var += d * d;
        }
        const size_t denom = std::max<size_t>(1, refCols[c].size() > 1 ? refCols[c].size() - 1 : 1);
        sigmaRaw[c] = std::sqrt(var / static_cast<double>(denom));
        if (!std::isfinite(sigmaRaw[c]) || sigmaRaw[c] < 1e-9) sigmaRaw[c] = 1.0;
    }

    auto absCorr = [&](const std::vector<double>& a, const std::vector<double>& b) {
        const size_t n = std::min(a.size(), b.size());
        if (n < 16) return 0.0;
        double ma = 0.0;
        double mb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ma += a[i];
            mb += b[i];
        }
        ma /= static_cast<double>(n);
        mb /= static_cast<double>(n);
        double va = 0.0;
        double vb = 0.0;
        double cov = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double da = a[i] - ma;
            const double db = b[i] - mb;
            va += da * da;
            vb += db * db;
            cov += da * db;
        }
        if (va <= 1e-12 || vb <= 1e-12) return 0.0;
        return std::abs(cov / std::sqrt(va * vb));
    };

    std::vector<size_t> candidates;
    candidates.reserve(cols);
    for (size_t c = 0; c < cols; ++c) {
        if (refCols[c].size() >= 24 && sigmaRaw[c] > 1e-7) {
            candidates.push_back(c);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        if (sigmaRaw[a] == sigmaRaw[b]) return a < b;
        return sigmaRaw[a] > sigmaRaw[b];
    });
    const size_t capCandidates = std::min<size_t>(candidates.size(), 48);
    candidates.resize(capCandidates);

    std::vector<size_t> selected;
    selected.reserve(std::min<size_t>(24, candidates.size()));
    constexpr double kOrthogonalityCorrCap = 0.90;
    for (size_t c : candidates) {
        bool orthogonalEnough = true;
        for (size_t s : selected) {
            if (absCorr(refCols[c], refCols[s]) >= kOrthogonalityCorrCap) {
                orthogonalEnough = false;
                break;
            }
        }
        if (orthogonalEnough) selected.push_back(c);
        if (selected.size() >= 24) break;
    }
    if (selected.size() < 2 && !candidates.empty()) {
        selected.assign(candidates.begin(), candidates.begin() + std::min<size_t>(2, candidates.size()));
    }
    if (selected.empty()) return out;

    const size_t dim = selected.size();
    std::vector<double> mu(dim, 0.0);
    std::vector<double> sigma(dim, 1.0);
    for (size_t k = 0; k < dim; ++k) {
        mu[k] = muRaw[selected[k]];
        sigma[k] = sigmaRaw[selected[k]];
    }

    MathUtils::Matrix cov(dim, dim);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i; j < dim; ++j) {
            double s = 0.0;
            size_t n = std::min(refCols[selected[i]].size(), refCols[selected[j]].size());
            for (size_t r = 0; r < n; ++r) {
                s += (refCols[selected[i]][r] - mu[i]) * (refCols[selected[j]][r] - mu[j]);
            }
            const double denom = static_cast<double>(std::max<size_t>(1, n > 1 ? n - 1 : 1));
            const double v = s / denom;
            cov.at(i, j) = v;
            cov.at(j, i) = v;
        }
    }

    double trace = 0.0;
    for (size_t i = 0; i < dim; ++i) trace += std::max(0.0, cov.at(i, i));
    const double baseScale = (trace > 0.0) ? (trace / static_cast<double>(dim)) : 1.0;
    double lambda = std::max(1e-8, 1e-6 * baseScale);

    std::optional<MathUtils::Matrix> invCov;
    MathUtils::Matrix covReg = cov;
    for (int attempt = 0; attempt < 8; ++attempt) {
        covReg = cov;
        for (size_t i = 0; i < dim; ++i) covReg.at(i, i) += lambda;
        covReg.setInversionTolerance(std::max(1e-12, lambda * 1e-3));
        invCov = covReg.inverse();
        if (invCov.has_value()) break;
        lambda *= 10.0;
    }

    std::vector<double> invDiag(dim, 1.0);
    bool useFullMahalanobis = invCov.has_value();
    if (!useFullMahalanobis) {
        for (size_t i = 0; i < dim; ++i) {
            invDiag[i] = 1.0 / std::max(1e-8, covReg.at(i, i));
        }
    }

    auto computeDistanceForRow = [&](const std::vector<double>& row,
                                     bool* extremeFeatureOut,
                                     std::vector<std::vector<double>>* pushToMonitorCols) -> std::optional<double> {
        if (row.size() != cols) return std::nullopt;
        std::vector<double> dvec(dim, 0.0);
        double z2 = 0.0;
        size_t valid = 0;
        bool extremeFeature = false;
        for (size_t k = 0; k < dim; ++k) {
            const size_t c = selected[k];
            const double v = row[c];
            if (!std::isfinite(v)) continue;
            const double centered = v - mu[k];
            dvec[k] = centered;
            const double z = centered / sigma[k];
            z2 += z * z;
            ++valid;
            if (std::abs(z) >= config.neuralOodZThreshold) extremeFeature = true;
            if (pushToMonitorCols != nullptr) (*pushToMonitorCols)[c].push_back(v);
        }
        if (valid == 0) return std::nullopt;

        double md2 = 0.0;
        if (useFullMahalanobis) {
            for (size_t i = 0; i < dim; ++i) {
                double rowDot = 0.0;
                for (size_t j = 0; j < dim; ++j) {
                    rowDot += invCov->at(i, j) * dvec[j];
                }
                md2 += dvec[i] * rowDot;
            }
        } else {
            for (size_t i = 0; i < dim; ++i) {
                md2 += dvec[i] * dvec[i] * invDiag[i];
            }
        }
        md2 = std::max(0.0, md2);
        const double mahal = std::sqrt(md2 / static_cast<double>(std::max<size_t>(1, dim)));
        const double rmsZ = std::sqrt(z2 / static_cast<double>(valid));
        if (extremeFeatureOut != nullptr) *extremeFeatureOut = extremeFeature;
        return std::optional<double>(std::max(mahal, rmsZ));
    };

    std::vector<double> refDistances;
    refDistances.reserve(referenceRows);
    for (size_t r = 0; r < referenceRows; ++r) {
        const auto maybeDist = computeDistanceForRow(X[r], nullptr, nullptr);
        if (maybeDist.has_value() && std::isfinite(*maybeDist)) {
            refDistances.push_back(*maybeDist);
        }
    }
    double refMean = 0.0;
    double refStd = 1.0;
    if (!refDistances.empty()) {
        refMean = std::accumulate(refDistances.begin(), refDistances.end(), 0.0) / static_cast<double>(refDistances.size());
        double var = 0.0;
        for (double d : refDistances) {
            const double dd = d - refMean;
            var += dd * dd;
        }
        const size_t denom = std::max<size_t>(1, refDistances.size() > 1 ? refDistances.size() - 1 : 1);
        refStd = std::sqrt(var / static_cast<double>(denom));
        if (!std::isfinite(refStd) || refStd < 1e-6) refStd = 1.0;
    }

    size_t oodHits = 0;
    double distSum = 0.0;
    double distMax = 0.0;
    for (size_t r = referenceRows; r < rows; ++r) {
        const auto& row = X[r];
        bool extremeFeature = false;
        const auto maybeDist = computeDistanceForRow(row, &extremeFeature, &monCols);
        if (!maybeDist.has_value() || !std::isfinite(*maybeDist)) continue;
        const double distance = *maybeDist;
        const double distanceZ = (distance - refMean) / refStd;

        distSum += distance;
        distMax = std::max(distMax, distance);
        if (distanceZ >= config.neuralOodZThreshold || (extremeFeature && distance >= config.neuralOodDistanceThreshold)) {
            ++oodHits;
        }
    }

    const double monitorRowsD = static_cast<double>(std::max<size_t>(1, monitorRows));
    out.oodRate = static_cast<double>(oodHits) / monitorRowsD;
    out.meanDistance = distSum / monitorRowsD;
    out.maxDistance = distMax;
    out.referenceRows = referenceRows;
    out.monitorRows = monitorRows;

    double psiSum = 0.0;
    size_t psiCount = 0;
    double psiMax = 0.0;
    for (size_t c : selected) {
        const double psi = computePsiBetweenSlices(refCols[c], monCols[c], 5);
        if (!std::isfinite(psi)) continue;
        psiSum += psi;
        ++psiCount;
        psiMax = std::max(psiMax, psi);
    }
    out.psiMean = (psiCount > 0) ? (psiSum / static_cast<double>(psiCount)) : 0.0;
    out.psiMax = psiMax;

    if (out.psiMean >= config.neuralDriftPsiCritical || out.psiMax >= (config.neuralDriftPsiCritical * 1.5) || out.oodRate >= 0.25) {
        out.driftBand = "critical";
    } else if (out.psiMean >= config.neuralDriftPsiWarning || out.psiMax >= config.neuralDriftPsiCritical || out.oodRate >= 0.10) {
        out.driftBand = "warning";
    } else {
        out.driftBand = "stable";
    }
    out.warning = out.driftBand != "stable";
    return out;
}

double computeConfidenceScore(const std::vector<double>& uncertaintyStd,
                              const std::vector<double>& ensembleStd,
                              double oodRate,
                              const std::string& driftBand) {
    auto avg = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0.0;
        size_t n = 0;
        for (double x : v) {
            if (!std::isfinite(x)) continue;
            s += std::max(0.0, x);
            ++n;
        }
        return (n > 0) ? (s / static_cast<double>(n)) : 0.0;
    };

    const double unc = avg(uncertaintyStd);
    const double ens = avg(ensembleStd);
    const double uncPenalty = unc / (1.0 + unc);
    const double ensPenalty = ens / (1.0 + ens);
    const double oodPenalty = std::clamp(oodRate, 0.0, 1.0);
    const double driftPenalty = (driftBand == "critical") ? 0.30 : ((driftBand == "warning") ? 0.15 : 0.0);

    const double totalPenalty = std::clamp(0.50 * uncPenalty + 0.20 * ensPenalty + 0.30 * oodPenalty + driftPenalty, 0.0, 1.0);
    return std::clamp(1.0 - totalPenalty, 0.0, 1.0);
}

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
    out.driftPsiMean = std::max(0.0, std::isfinite(neural.driftPsiMean) ? neural.driftPsiMean : 0.0);

    const double score01 =
        0.30 * out.completeness +
        0.10 * out.numericCoverage +
        0.15 * out.featureRetention +
        0.20 * out.statYield +
        0.15 * out.selectedYield +
        0.10 * out.trainingStability;

    double driftPenalty = 0.0;
    if (out.driftPsiMean > 0.10) {
        driftPenalty += std::min(0.20, (out.driftPsiMean - 0.10) * 0.20);
    }
    if (out.driftPsiMean > 0.50) {
        driftPenalty += std::min(0.30, 0.12 + (out.driftPsiMean - 0.50) * 0.35);
    }
    if (neural.driftBand == "critical") {
        driftPenalty += 0.08;
    } else if (neural.driftBand == "warning") {
        driftPenalty += 0.03;
    }
    out.driftPenalty = std::clamp(driftPenalty, 0.0, 0.55);

    const double adjustedScore01 = std::clamp(score01 * (1.0 - out.driftPenalty), 0.0, 1.0);
    out.score = std::clamp(adjustedScore01 * 100.0, 0.0, 100.0);
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

std::vector<int> selectAuxiliaryNumericTargets(const TypedDataset& data,
                                               int primaryTargetIdx,
                                               size_t maxAuxTargets,
                                               const NumericStatsCache& statsCache) {
    std::vector<int> targetIndices;
    if (primaryTargetIdx < 0 || static_cast<size_t>(primaryTargetIdx) >= data.columns().size()) return targetIndices;
    if (data.columns()[static_cast<size_t>(primaryTargetIdx)].type != ColumnType::NUMERIC) return targetIndices;

    targetIndices.push_back(primaryTargetIdx);
    if (maxAuxTargets == 0) return targetIndices;

    const auto numericIdx = data.numericColumnIndices();
    if (numericIdx.empty()) return targetIndices;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(primaryTargetIdx)].values);
    const auto yIt = statsCache.find(static_cast<size_t>(primaryTargetIdx));
    const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);

    std::vector<std::pair<int, double>> candidates;
    candidates.reserve(numericIdx.size());
    for (size_t idx : numericIdx) {
        if (static_cast<int>(idx) == primaryTargetIdx) continue;
        const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
        const auto xIt = statsCache.find(idx);
        const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
        const double corr = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
        if (std::isfinite(corr)) {
            candidates.push_back({static_cast<int>(idx), corr});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    const size_t keep = std::min(maxAuxTargets, candidates.size());
    for (size_t i = 0; i < keep; ++i) {
        targetIndices.push_back(candidates[i].first);
    }
    return targetIndices;
}

std::string escapeJsonString(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (char ch : in) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

std::optional<std::string> writeCausalDagEditor(const AutoConfig& cfg,
                                                const std::vector<std::vector<std::string>>& causalRows) {
    if (cfg.assetsDir.empty()) return std::nullopt;

    namespace fs = std::filesystem;
    const fs::path editorPath = fs::path(cfg.assetsDir) / "dag_editor.html";
    std::error_code ec;
    fs::create_directories(editorPath.parent_path(), ec);

    std::unordered_set<std::string> nodeSet;
    std::vector<std::pair<std::string, std::string>> edges;
    edges.reserve(causalRows.size());
    for (const auto& row : causalRows) {
        if (row.size() < 2) continue;
        const std::string from = CommonUtils::trim(row[0]);
        const std::string to = CommonUtils::trim(row[1]);
        if (from.empty() || to.empty()) continue;
        nodeSet.insert(from);
        nodeSet.insert(to);
        edges.push_back({from, to});
    }

    std::vector<std::string> nodes(nodeSet.begin(), nodeSet.end());
    std::sort(nodes.begin(), nodes.end());

    std::ofstream out(editorPath);
    if (!out.good()) return std::nullopt;

    out << "<!doctype html>\n<html><head><meta charset=\"utf-8\"/>\n";
    out << "<title>Seldon DAG Editor</title>\n";
    out << "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:16px;background:#fafafa;}";
    out << "h1{font-size:18px;margin:0 0 10px;} .row{display:flex;gap:14px;align-items:flex-start;}";
    out << "#canvas{background:#fff;border:1px solid #ddd;border-radius:8px;} .panel{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;min-width:340px;}";
    out << "table{border-collapse:collapse;width:100%;font-size:12px;} th,td{border:1px solid #ddd;padding:4px 6px;}";
    out << "button,select,input{font-size:12px;padding:4px 6px;margin:2px 0;} .muted{color:#666;font-size:12px;}";
    out << "textarea{width:100%;min-height:120px;font-family:ui-monospace,Consolas,monospace;font-size:12px;}";
    out << "</style></head><body>\n";
    out << "<h1>Seldon Interactive DAG Editor</h1>\n";
    out << "<p class=\"muted\">Drag nodes in the canvas. Add, remove, or reverse edges. Export updated Mermaid below.</p>\n";
    out << "<div class=\"row\">\n";
    out << "<svg id=\"canvas\" width=\"920\" height=\"560\"></svg>\n";
    out << "<div class=\"panel\">\n";
    out << "<div><strong>Add Edge</strong></div>\n";
    out << "<select id=\"fromSel\"></select> → <select id=\"toSel\"></select> <button id=\"addBtn\">Add</button><br/>\n";
    out << "<button id=\"reverseBtn\">Reverse Selected</button> <button id=\"deleteBtn\">Delete Selected</button>\n";
    out << "<div style=\"margin-top:8px\"><strong>Edges</strong></div>\n";
    out << "<table><thead><tr><th>#</th><th>From</th><th>To</th></tr></thead><tbody id=\"edgeBody\"></tbody></table>\n";
    out << "<div style=\"margin-top:8px\"><strong>Mermaid Export</strong></div>\n";
    out << "<textarea id=\"mermaidOut\" readonly></textarea>\n";
    out << "<button id=\"copyBtn\">Copy Mermaid</button>\n";
    out << "</div></div>\n";

    out << "<script>\n";
    out << "const nodes = [";
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i) out << ",";
        out << "\"" << escapeJsonString(nodes[i]) << "\"";
    }
    out << "];\n";

    out << "let edges = [";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i) out << ",";
        out << "{from:\"" << escapeJsonString(edges[i].first) << "\",to:\"" << escapeJsonString(edges[i].second) << "\"}";
    }
    out << "];\n";

    out << "const svg=document.getElementById('canvas');const ns='http://www.w3.org/2000/svg';let selected=-1;\n";
    out << "const pos={};const W=900,H=540;nodes.forEach((n,i)=>{const ang=(2*Math.PI*i)/Math.max(1,nodes.length);pos[n]={x:W/2+220*Math.cos(ang),y:H/2+200*Math.sin(ang)};});\n";
    out << "function ensureUniqueEdges(){const seen=new Set();edges=edges.filter(e=>{if(!e.from||!e.to||e.from===e.to) return false; const k=e.from+'\\u2192'+e.to; if(seen.has(k)) return false; seen.add(k); return true;});}\n";
    out << "function arrow(x1,y1,x2,y2){const dx=x2-x1,dy=y2-y1,d=Math.max(1,Math.hypot(dx,dy));const ux=dx/d,uy=dy/d;const tx=x2-26*ux,ty=y2-26*uy;return {x1:x1+26*ux,y1:y1+26*uy,x2:tx,y2:ty,ux,uy};}\n";
    out << "function render(){ensureUniqueEdges();while(svg.firstChild) svg.removeChild(svg.firstChild);\n";
    out << "edges.forEach((e,idx)=>{const a=arrow(pos[e.from].x,pos[e.from].y,pos[e.to].x,pos[e.to].y);const l=document.createElementNS(ns,'line');l.setAttribute('x1',a.x1);l.setAttribute('y1',a.y1);l.setAttribute('x2',a.x2);l.setAttribute('y2',a.y2);l.setAttribute('stroke',idx===selected?'#d12':'#666');l.setAttribute('stroke-width',idx===selected?'3':'2');l.style.cursor='pointer';l.onclick=()=>{selected=idx;render();};svg.appendChild(l);const p=document.createElementNS(ns,'polygon');const px=a.x2,py=a.y2;const s=7;const p1=(px)+','+(py);const p2=(px- s*a.ux + s*a.uy)+','+(py- s*a.uy - s*a.ux);const p3=(px- s*a.ux - s*a.uy)+','+(py- s*a.uy + s*a.ux);p.setAttribute('points',p1+' '+p2+' '+p3);p.setAttribute('fill',idx===selected?'#d12':'#666');svg.appendChild(p);});\n";
    out << "nodes.forEach(n=>{const g=document.createElementNS(ns,'g');g.style.cursor='move';const c=document.createElementNS(ns,'circle');c.setAttribute('cx',pos[n].x);c.setAttribute('cy',pos[n].y);c.setAttribute('r',22);c.setAttribute('fill','#f3f7ff');c.setAttribute('stroke','#3366cc');c.setAttribute('stroke-width','2');g.appendChild(c);const t=document.createElementNS(ns,'text');t.setAttribute('x',pos[n].x);t.setAttribute('y',pos[n].y+4);t.setAttribute('font-size','11');t.setAttribute('text-anchor','middle');t.textContent=n;g.appendChild(t);let drag=false;g.onmousedown=(ev)=>{drag=true;ev.preventDefault();};window.addEventListener('mouseup',()=>drag=false);window.addEventListener('mousemove',(ev)=>{if(!drag) return;const r=svg.getBoundingClientRect();pos[n].x=Math.max(30,Math.min(W-30,ev.clientX-r.left));pos[n].y=Math.max(30,Math.min(H-30,ev.clientY-r.top));render();});svg.appendChild(g);});\n";
    out << "const body=document.getElementById('edgeBody');body.innerHTML='';edges.forEach((e,idx)=>{const tr=document.createElement('tr');if(idx===selected) tr.style.background='#fff2f2';tr.onclick=()=>{selected=idx;render();};tr.innerHTML='<td>'+ (idx+1) +'</td><td>'+e.from+'</td><td>'+e.to+'</td>';body.appendChild(tr);});\n";
    out << "const mer=['```mermaid','flowchart LR'];edges.forEach(e=>{mer.push('    '+safeId(e.from)+'[\"'+escapeMer(e.from)+'\"] --> '+safeId(e.to)+'[\"'+escapeMer(e.to)+'\"]');});mer.push('```');document.getElementById('mermaidOut').value=mer.join('\\n');}\n";
    out << "function safeId(n){return 'N'+(nodes.indexOf(n)+1);} function escapeMer(s){return String(s).replace(/\"/g,\"'\");}\n";
    out << "const fromSel=document.getElementById('fromSel');const toSel=document.getElementById('toSel');nodes.forEach(n=>{const o1=document.createElement('option');o1.textContent=n;o1.value=n;fromSel.appendChild(o1);const o2=document.createElement('option');o2.textContent=n;o2.value=n;toSel.appendChild(o2);});\n";
    out << "document.getElementById('addBtn').onclick=()=>{const from=fromSel.value,to=toSel.value;if(!from||!to||from===to) return;edges.push({from,to});selected=edges.length-1;render();};\n";
    out << "document.getElementById('reverseBtn').onclick=()=>{if(selected<0||selected>=edges.length) return;const e=edges[selected];edges[selected]={from:e.to,to:e.from};render();};\n";
    out << "document.getElementById('deleteBtn').onclick=()=>{if(selected<0||selected>=edges.length) return;edges.splice(selected,1);selected=Math.min(selected,edges.length-1);render();};\n";
    out << "document.getElementById('copyBtn').onclick=async()=>{try{await navigator.clipboard.writeText(document.getElementById('mermaidOut').value);}catch(_){}};\n";
    out << "render();\n";
    out << "</script></body></html>\n";

    if (!out.good()) return std::nullopt;

    const fs::path reportDir = cfg.outputDir.empty() ? editorPath.parent_path() : fs::path(cfg.outputDir);
    const fs::path relativeEditorPath = fs::relative(editorPath, reportDir, ec);
    if (!ec && !relativeEditorPath.empty()) {
        return relativeEditorPath.generic_string();
    }
    return editorPath.filename().generic_string();
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
    univariate.addParagraph(canPlot ? "Gnuplot detected." : "Gnuplot not available: visualizations were omitted for runtime/portability and analysis continues normally.");
    if (!canPlot && (runCfg.plotUnivariate || runCfg.plotOverall || runCfg.plotBivariateSignificant)) {
        runCfg.plotUnivariate = false;
        runCfg.plotOverall = false;
        runCfg.plotBivariateSignificant = false;
        univariate.addParagraph("Requested plotting features were disabled because gnuplot is unavailable in PATH; this is not a statistical failure.");
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

struct ReportSaveSummary {
    bool htmlRequested = false;
    bool pandocAvailable = false;
    size_t htmlAttempted = 0;
    size_t htmlSucceeded = 0;
};

ReportSaveSummary saveGeneratedReports(const AutoConfig& runCfg,
                          const ReportEngine& univariate,
                          const ReportEngine& bivariate,
                          const ReportEngine& neuralReport,
                          const ReportEngine& finalAnalysis,
                          const ReportEngine& heuristicsReport) {
    ReportSaveSummary summary;
    namespace fs = std::filesystem;
    const std::string uniMd = runCfg.outputDir + "/univariate.md";
    const std::string biMd = runCfg.outputDir + "/bivariate.md";
    const std::string finalMd = runCfg.outputDir + "/final_analysis.md";
    const std::string reportMd = runCfg.outputDir + "/report.md";
    const std::string neuralHtml = (fs::path(runCfg.reportFile).parent_path() /
                                    (fs::path(runCfg.reportFile).stem().string() + ".html")).string();

    univariate.save(uniMd);
    bivariate.save(biMd);
    neuralReport.save(runCfg.reportFile);
    finalAnalysis.save(finalMd);
    heuristicsReport.save(reportMd);

    if (runCfg.generateHtml) {
        summary.htmlRequested = true;
        const std::string pandocExe = findExecutableInPath("pandoc");
        if (pandocExe.empty()) {
            std::cout << "[Seldon][Warning] HTML export requested but 'pandoc' was not found in PATH. Markdown reports were generated only.\n";
            return summary;
        }
        summary.pandocAvailable = true;
        const std::vector<std::pair<std::string, std::string>> conversions = {
            {uniMd, runCfg.outputDir + "/univariate.html"},
            {biMd, runCfg.outputDir + "/bivariate.html"},
            {runCfg.reportFile, neuralHtml},
            {finalMd, runCfg.outputDir + "/final_analysis.html"},
            {reportMd, runCfg.outputDir + "/report.html"}
        };
        summary.htmlAttempted = conversions.size();
        for (const auto& [src, dst] : conversions) {
            const int rc = spawnProcessSilenced(pandocExe, {src, "-o", dst, "--standalone", "--self-contained"});
            if (rc == 0) ++summary.htmlSucceeded;
        }
        if (summary.htmlSucceeded < summary.htmlAttempted) {
            std::cout << "[Seldon][Warning] HTML conversion completed for " << summary.htmlSucceeded
                      << "/" << summary.htmlAttempted << " files. Check pandoc runtime dependencies.\n";
        }
    }

    return summary;
}

void printPipelineCompletion(const AutoConfig& runCfg,
                             bool htmlRequested,
                             bool pandocAvailable,
                             size_t htmlAttempted,
                             size_t htmlSucceeded) {
    namespace fs = std::filesystem;
    const std::string neuralHtml = (fs::path(runCfg.reportFile).parent_path() /
                                    (fs::path(runCfg.reportFile).stem().string() + ".html")).string();
    std::cout << "[Seldon] Automated pipeline complete.\n";
    std::cout << "[Seldon] Report directory: " << runCfg.outputDir << "\n";
    std::cout << "[Seldon] Reports: "
              << runCfg.outputDir << "/univariate.md, "
              << runCfg.outputDir << "/bivariate.md, "
              << runCfg.reportFile << ", "
              << runCfg.outputDir << "/final_analysis.md, "
              << runCfg.outputDir << "/report.md\n";
    if (htmlRequested && pandocAvailable && htmlSucceeded > 0) {
        std::cout << "[Seldon] HTML reports (self-contained): "
                  << runCfg.outputDir << "/univariate.html, "
                  << runCfg.outputDir << "/bivariate.html, "
                  << neuralHtml << ", "
                  << runCfg.outputDir << "/final_analysis.html, "
                  << runCfg.outputDir << "/report.html"
                  << " (" << htmlSucceeded << "/" << htmlAttempted << " succeeded)\n";
    } else if (htmlRequested && !pandocAvailable) {
        std::cout << "[Seldon][Warning] HTML export skipped because pandoc is unavailable.\n";
    } else if (htmlRequested && pandocAvailable && htmlSucceeded == 0) {
        std::cout << "[Seldon][Warning] HTML export attempted but no HTML files were produced successfully.\n";
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

    std::ofstream out(csvPath, std::ios::binary);
    if (!out) {
        throw Seldon::IOException("Unable to write preprocessed export: " + csvPath);
    }
    std::vector<char> ioBuffer(1 << 20, '\0');
    out.rdbuf()->pubsetbuf(ioBuffer.data(), static_cast<std::streamsize>(ioBuffer.size()));

    const auto& cols = data.columns();
    const size_t colCount = cols.size();

    for (size_t c = 0; c < colCount; ++c) {
        if (c) out << ',';
        out << '"' << cols[c].name << '"';
    }
    out << '\n';

    std::string chunk;
    chunk.reserve(1 << 20);
    for (size_t r = 0; r < data.rowCount(); ++r) {
        for (size_t c = 0; c < colCount; ++c) {
            if (c) chunk.push_back(',');
            if (cols[c].missing[r]) {
                continue;
            }
            if (cols[c].type == ColumnType::NUMERIC) {
                const auto& vals = std::get<std::vector<double>>(cols[c].values);
                chunk += std::to_string(vals[r]);
            } else if (cols[c].type == ColumnType::DATETIME) {
                const auto& vals = std::get<std::vector<int64_t>>(cols[c].values);
                chunk += std::to_string(vals[r]);
            } else {
                std::string s = std::get<std::vector<std::string>>(cols[c].values)[r];
                std::replace(s.begin(), s.end(), '"', '\'');
                chunk.push_back('"');
                chunk += s;
                chunk.push_back('"');
            }
        }
        chunk.push_back('\n');
        if (chunk.size() >= (1 << 20)) {
            out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            chunk.clear();
        }
    }
    if (!chunk.empty()) {
        out.write(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    }

    if (runCfg.exportPreprocessed == "parquet") {
        const std::string parquetPath = basePath + ".parquet";
#ifdef SELDON_USE_NATIVE_PARQUET
        std::string parquetError;
        if (!exportParquetNative(data, parquetPath, parquetError)) {
            std::cout << "[Seldon][Warning] Native parquet export failed: " << parquetError
                      << ". CSV export is available at " << csvPath << "\n";
        }
#else
        std::cout << "[Seldon][Warning] Parquet export requested, but this build was compiled without native parquet support. "
                  << "Rebuild with Arrow/Parquet libraries enabled. CSV export is available at " << csvPath << "\n";
#endif
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

