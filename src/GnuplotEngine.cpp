#include "GnuplotEngine.h"
#include "PlotHeuristics.h"
#include "StatsUtils.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

namespace {
std::string normalizePlotLabel(const std::string& label, size_t maxLen = 28) {
    std::string out;
    out.reserve(label.size());
    for (unsigned char ch : label) {
        if (ch >= 32 && ch <= 126) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('?');
        }
    }
    if (out.size() > maxLen) {
        out = out.substr(0, maxLen - 3) + "...";
    }
    return out;
}

std::string normalizePlotTitle(const std::string& title,
                               size_t wrapAt = 54,
                               size_t maxTotal = 140) {
    std::string cleaned;
    cleaned.reserve(title.size() + 16);

    auto appendSpaceIfNeeded = [&]() {
        if (!cleaned.empty() && cleaned.back() != ' ' && cleaned.back() != '\n') cleaned.push_back(' ');
    };

    auto isAlphaNum = [](char c) {
        const unsigned char uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) != 0;
    };

    auto appendSeparator = [&](const std::string& sep) {
        if (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
        if (!cleaned.empty() && cleaned.back() != '\n') cleaned.push_back(' ');
        cleaned += sep;
        cleaned.push_back(' ');
    };

    for (size_t i = 0; i < title.size(); ++i) {
        const char ch = title[i];
        const char prevRaw = (i > 0) ? title[i - 1] : '\0';
        const char nextRaw = (i + 1 < title.size()) ? title[i + 1] : '\0';

        if (ch == ':') {
            appendSeparator("|");
            continue;
        }
        if (ch == '|') {
            appendSeparator("|");
            continue;
        }
        if (ch == '-' && nextRaw == '>') {
            appendSeparator("vs");
            ++i;
            continue;
        }
        if (ch == '<' && nextRaw == '-') {
            appendSeparator("vs");
            ++i;
            continue;
        }
        if (ch == '-') {
            const bool hyphenatedWord = isAlphaNum(prevRaw) && isAlphaNum(nextRaw);
            if (hyphenatedWord) {
                appendSpaceIfNeeded();
            } else {
                appendSeparator("|");
            }
            continue;
        }
        if (ch == '_') {
            appendSpaceIfNeeded();
            continue;
        }
        if (ch == '\n' || ch == '\t' || ch == '\r') {
            appendSpaceIfNeeded();
            continue;
        }

        const bool isUpper = std::isupper(static_cast<unsigned char>(ch));
        const bool isDigit = std::isdigit(static_cast<unsigned char>(ch));
        const bool prevLower = !cleaned.empty() && std::islower(static_cast<unsigned char>(cleaned.back()));
        const bool prevDigit = !cleaned.empty() && std::isdigit(static_cast<unsigned char>(cleaned.back()));
        if (!cleaned.empty() && ((isUpper && prevLower) || (isDigit && !prevDigit) || (!isDigit && prevDigit))) {
            appendSpaceIfNeeded();
        }

        if (ch >= 32 && ch <= 126) {
            cleaned.push_back(ch);
        }
    }

    std::string collapsed;
    collapsed.reserve(cleaned.size());
    bool inSpace = false;
    for (char ch : cleaned) {
        if (ch == ' ') {
            if (!inSpace) collapsed.push_back(ch);
            inSpace = true;
        } else {
            collapsed.push_back(ch);
            inSpace = false;
        }
    }
    if (!collapsed.empty() && collapsed.front() == ' ') collapsed.erase(collapsed.begin());
    while (!collapsed.empty() && collapsed.back() == ' ') collapsed.pop_back();

    if (collapsed.size() > maxTotal) {
        collapsed = collapsed.substr(0, maxTotal - 3) + "...";
    }

    std::string lower = collapsed;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const bool hasVs = lower.find(" vs ") != std::string::npos;
    const bool hasPipe = collapsed.find('|') != std::string::npos;
    if (!hasVs && !hasPipe) {
        const size_t overPos = lower.find(" over ");
        if (overPos != std::string::npos) {
            collapsed.replace(overPos, 6, " vs ");
        } else {
            const size_t byPos = lower.find(" by ");
            if (byPos != std::string::npos) {
                collapsed.replace(byPos, 4, " | by ");
            }
        }
    }

    if (collapsed.size() <= wrapAt) return collapsed;

    std::string wrapped;
    wrapped.reserve(collapsed.size() + 8);
    size_t lineStart = 0;
    while (lineStart < collapsed.size()) {
        size_t remaining = collapsed.size() - lineStart;
        if (remaining <= wrapAt) {
            wrapped.append(collapsed.substr(lineStart));
            break;
        }

        size_t split = lineStart + wrapAt;
        size_t bestSpace = collapsed.rfind(' ', split);
        if (bestSpace == std::string::npos || bestSpace <= lineStart + (wrapAt / 2)) {
            bestSpace = split;
        }

        wrapped.append(collapsed.substr(lineStart, bestSpace - lineStart));
        wrapped.append("\\n");
        lineStart = (bestSpace < collapsed.size() && collapsed[bestSpace] == ' ') ? (bestSpace + 1) : bestSpace;
    }

    return wrapped;
}

std::string quoteForDatafileString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string toLowerCopy(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

uint64_t fnv1a64(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string toHex(uint64_t value) {
    std::ostringstream os;
    os << std::hex << value;
    return os.str();
}

std::vector<std::pair<double, double>> finitePairs(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<std::pair<double, double>> out;
    const size_t n = std::min(x.size(), y.size());
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        out.push_back({x[i], y[i]});
    }
    return out;
}

std::vector<double> finiteValues(const std::vector<double>& values) {
    std::vector<double> out;
    out.reserve(values.size());
    for (double v : values) {
        if (std::isfinite(v)) out.push_back(v);
    }
    return out;
}

struct AxisRange {
    double lo = 0.0;
    double hi = 1.0;
    bool valid = false;
};

AxisRange robustAxisRange(const std::vector<double>& values, double padFraction = 0.06) {
    AxisRange out;
    if (values.empty()) return out;

    double minV = std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    for (double v : values) {
        if (!std::isfinite(v)) continue;
        minV = std::min(minV, v);
        maxV = std::max(maxV, v);
    }
    if (!std::isfinite(minV) || !std::isfinite(maxV)) return out;

    if (std::abs(maxV - minV) <= 1e-12) {
        const double pad = std::max(1.0, std::abs(minV) * 0.15);
        out.lo = minV - pad;
        out.hi = maxV + pad;
    } else {
        const double span = maxV - minV;
        const double pad = std::max(1e-9, span * std::max(0.01, padFraction));
        out.lo = minV - pad;
        out.hi = maxV + pad;
    }
    out.valid = std::isfinite(out.lo) && std::isfinite(out.hi) && (out.hi > out.lo);
    return out;
}

void applyAxisFormatting(std::ostringstream& script, const char axis, const AxisRange& range) {
    if (!range.valid) return;
    if (axis == 'x') {
        script << "set xrange [" << range.lo << ":" << range.hi << "]\n";
    } else if (axis == 'y') {
        script << "set yrange [" << range.lo << ":" << range.hi << "]\n";
    }

    const double maxAbs = std::max(std::abs(range.lo), std::abs(range.hi));
    const double span = std::max(1e-12, range.hi - range.lo);
    if (maxAbs >= 1e6 || maxAbs <= 1e-5 || (maxAbs / span) >= 1e4) {
        script << "set format " << axis << " '%.3e'\n";
    } else {
        script << "set format " << axis << " '%.6g'\n";
    }
}

struct CategoricalSeries {
    std::vector<std::string> labels;
    std::vector<double> values;
};

CategoricalSeries aggregateTopCategories(const std::vector<std::string>& labels,
                                         const std::vector<double>& values,
                                         size_t maxCategories,
                                         const std::string& otherLabel = "Other") {
    CategoricalSeries out;
    const size_t n = std::min(labels.size(), values.size());
    if (n == 0) return out;

    std::vector<std::pair<std::string, double>> rows;
    rows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i]) || values[i] <= 0.0) continue;
        rows.push_back({labels[i], values[i]});
    }
    if (rows.empty()) return out;

    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    const size_t keep = (rows.size() > maxCategories) ? (maxCategories - 1) : rows.size();
    double other = 0.0;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i < keep) {
            out.labels.push_back(rows[i].first);
            out.values.push_back(rows[i].second);
        } else {
            other += rows[i].second;
        }
    }
    if (other > 0.0) {
        out.labels.push_back(otherLabel);
        out.values.push_back(other);
    }
    return out;
}

std::vector<std::pair<double, double>> downsampleEvenly(const std::vector<std::pair<double, double>>& values,
                                                        size_t maxPoints) {
    if (values.size() <= maxPoints || maxPoints < 50) return values;

    std::vector<std::pair<double, double>> sorted = values;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first < b.first;
    });

    std::vector<std::pair<double, double>> out;
    out.reserve(maxPoints);
    const double step = static_cast<double>(sorted.size() - 1) / static_cast<double>(maxPoints - 1);
    for (size_t i = 0; i < maxPoints; ++i) {
        const size_t idx = static_cast<size_t>(std::llround(step * static_cast<double>(i)));
        out.push_back(sorted[std::min(idx, sorted.size() - 1)]);
    }
    return out;
}

double meanOf(const std::vector<double>& values) {
    return StatsUtils::runningMean(values);
}

double stddevSample(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    const double mu = meanOf(values);
    double s2 = 0.0;
    for (double v : values) {
        const double d = v - mu;
        s2 += d * d;
    }
    s2 /= static_cast<double>(values.size() - 1);
    return std::sqrt(std::max(0.0, s2));
}

double silvermanBandwidth(const std::vector<double>& sorted) {
    if (sorted.size() < 2) return 1.0;
    const double sigma = stddevSample(sorted);
    const double iqr = StatsUtils::percentileSorted(sorted, 0.75) - StatsUtils::percentileSorted(sorted, 0.25);
    double robustSigma = sigma;
    if (iqr > 0.0) {
        robustSigma = std::min(sigma > 0.0 ? sigma : iqr / 1.34, iqr / 1.34);
    }
    if (!std::isfinite(robustSigma) || robustSigma <= 1e-12) robustSigma = std::max(1e-3, sigma);
    const double n = static_cast<double>(sorted.size());
    double h = 1.06 * robustSigma * std::pow(std::max(1.0, n), -0.2);
    if (!std::isfinite(h) || h <= 1e-12) h = 1e-3;
    return h;
}

std::vector<double> kdeEvaluate(const std::vector<double>& sample,
                                const std::vector<double>& grid,
                                double bandwidth) {
    std::vector<double> density(grid.size(), 0.0);
    if (sample.empty() || grid.empty() || bandwidth <= 0.0) return density;

    const double inv = 1.0 / (std::sqrt(2.0 * 3.14159265358979323846) * bandwidth * static_cast<double>(sample.size()));
    for (size_t i = 0; i < grid.size(); ++i) {
        double s = 0.0;
        const double gx = grid[i];
        for (double v : sample) {
            const double z = (gx - v) / bandwidth;
            s += std::exp(-0.5 * z * z);
        }
        density[i] = inv * s;
    }
    return density;
}

double rmse(const std::vector<double>& y, const std::vector<double>& yhat) {
    if (y.empty() || y.size() != yhat.size()) return std::numeric_limits<double>::infinity();
    double sse = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double e = y[i] - yhat[i];
        sse += e * e;
    }
    return std::sqrt(sse / static_cast<double>(y.size()));
}

std::vector<double> linearFitPredict(const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     double& slope,
                                     double& intercept) {
    std::vector<double> yhat(y.size(), 0.0);
    if (x.size() != y.size() || x.size() < 2) return yhat;

    const double xMean = meanOf(x);
    const double yMean = meanOf(y);
    double sxx = 0.0;
    double sxy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double xd = x[i] - xMean;
        sxx += xd * xd;
        sxy += xd * (y[i] - yMean);
    }
    slope = (std::abs(sxx) <= 1e-12) ? 0.0 : (sxy / sxx);
    intercept = yMean - slope * xMean;
    for (size_t i = 0; i < x.size(); ++i) yhat[i] = slope * x[i] + intercept;
    return yhat;
}

bool solve3x3(double a[3][3], double b[3], double out[3]) {
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int r = i + 1; r < 3; ++r) {
            if (std::abs(a[r][i]) > std::abs(a[pivot][i])) pivot = r;
        }
        if (std::abs(a[pivot][i]) < 1e-12) return false;
        if (pivot != i) {
            for (int c = 0; c < 3; ++c) std::swap(a[i][c], a[pivot][c]);
            std::swap(b[i], b[pivot]);
        }

        const double div = a[i][i];
        for (int c = i; c < 3; ++c) a[i][c] /= div;
        b[i] /= div;

        for (int r = 0; r < 3; ++r) {
            if (r == i) continue;
            const double factor = a[r][i];
            for (int c = i; c < 3; ++c) a[r][c] -= factor * a[i][c];
            b[r] -= factor * b[i];
        }
    }
    out[0] = b[0];
    out[1] = b[1];
    out[2] = b[2];
    return true;
}

std::vector<double> poly2Predict(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 bool& ok) {
    ok = false;
    std::vector<double> yhat(y.size(), 0.0);
    if (x.size() != y.size() || x.size() < 6) return yhat;

    double sx = 0.0;
    double sx2 = 0.0;
    double sx3 = 0.0;
    double sx4 = 0.0;
    double sy = 0.0;
    double sxy = 0.0;
    double sx2y = 0.0;
    const double n = static_cast<double>(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        const double xi = x[i];
        const double x2 = xi * xi;
        sx += xi;
        sx2 += x2;
        sx3 += x2 * xi;
        sx4 += x2 * x2;
        sy += y[i];
        sxy += xi * y[i];
        sx2y += x2 * y[i];
    }

    double A[3][3] = {{n, sx, sx2}, {sx, sx2, sx3}, {sx2, sx3, sx4}};
    double B[3] = {sy, sxy, sx2y};
    double C[3] = {0.0, 0.0, 0.0};
    if (!solve3x3(A, B, C)) return yhat;

    for (size_t i = 0; i < x.size(); ++i) {
        yhat[i] = C[0] + C[1] * x[i] + C[2] * x[i] * x[i];
    }
    ok = true;
    return yhat;
}

std::vector<double> movingAveragePredict(const std::vector<double>& y, size_t window) {
    std::vector<double> out(y.size(), 0.0);
    if (y.empty()) return out;
    if (window < 2) return y;
    double sum = 0.0;
    size_t used = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        sum += y[i];
        ++used;
        if (used > window) {
            sum -= y[i - window];
            --used;
        }
        out[i] = sum / static_cast<double>(used);
    }
    return out;
}

std::string ganttColorForSemantic(const std::string& semantic, const std::string& fallback) {
    const std::string s = toLowerCopy(semantic);
    if (s.find("urgent") != std::string::npos || s.find("high") != std::string::npos || s.find("block") != std::string::npos) return "#dc2626";
    if (s.find("done") != std::string::npos || s.find("closed") != std::string::npos || s.find("complete") != std::string::npos) return "#059669";
    if (s.find("progress") != std::string::npos || s.find("doing") != std::string::npos || s.find("active") != std::string::npos) return "#2563eb";
    if (s.find("low") != std::string::npos || s.find("backlog") != std::string::npos || s.find("todo") != std::string::npos) return "#7c3aed";
    return fallback;
}

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

int spawnGnuplot(const std::string& executable,
                 const std::string& scriptPath,
                 const std::string& stderrPath) {
    const int errFd = ::open(stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (errFd < 0) return -1;

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(errFd);
        return -1;
    }

    if (pid == 0) {
        if (::dup2(errFd, STDERR_FILENO) < 0) {
            _exit(127);
        }
        ::close(errFd);
        const char* argv[] = {executable.c_str(), scriptPath.c_str(), nullptr};
        ::execv(executable.c_str(), const_cast<char* const*>(argv));
        _exit(127);
    }

    ::close(errFd);
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}
} // namespace

std::string GnuplotEngine::sanitizeId(const std::string& id) {
    std::string out = id;
    std::replace_if(out.begin(), out.end(), [](unsigned char c) {
        return !(std::isalnum(c) || c == '_' || c == '-');
    }, '_');
    if (out.empty()) out = "plot";
    return out;
}

std::string GnuplotEngine::quoteForGnuplot(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('\'');
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string GnuplotEngine::terminalForFormat(const std::string& format, int width, int height) {
    if (format == "svg") return "svg size " + std::to_string(width) + "," + std::to_string(height);
    if (format == "pdf") return "pdfcairo size 11in,8in";
    return "pngcairo size " + std::to_string(width) + "," + std::to_string(height);
}

std::string GnuplotEngine::styledHeader(const std::string& id, const std::string& title) const {
    const std::string safeId = sanitizeId(id);
    const bool darkTheme = (cfg_.theme == "dark");
    const std::string titleColor = darkTheme ? "#f9fafb" : "#1f2937";
    const std::string borderColor = darkTheme ? "#6b7280" : "#9ca3af";
    const std::string ticColor = darkTheme ? "#e5e7eb" : "#374151";
    const std::string gridColor = darkTheme ? "#374151" : "#e5e7eb";
    const std::string bgColor = darkTheme ? "#111827" : "#ffffff";

    std::ostringstream script;
    script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << " enhanced\n";
    script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
    script << "set object 999 rect from graph 0,0 to graph 1,1 behind fc rgb " << quoteForGnuplot(bgColor) << " fs solid 1.0 noborder\n";
        script << "set title " << quoteForGnuplot(normalizePlotTitle(title))
            << " tc rgb " << quoteForGnuplot(titleColor)
            << " font ',12'\n";
    script << "set tmargin 2.5\nset bmargin 3.5\nset lmargin 7\nset rmargin 3\n";
    script << "set border linewidth " << cfg_.lineWidth << " lc rgb " << quoteForGnuplot(borderColor) << "\n";
    script << "set tics textcolor rgb " << quoteForGnuplot(ticColor) << "\n";
    script << "set tics out nomirror\nset mxtics 2\nset mytics 2\n";
    if (cfg_.showGrid) {
        script << "set grid back lc rgb " << quoteForGnuplot(gridColor) << " lw 1 dt 2\n";
    } else {
        script << "unset grid\n";
    }
    script << "set key top left opaque box lc rgb " << quoteForGnuplot(borderColor) << "\n";
    script << "set style line 1 lc rgb '#2563eb' lw " << cfg_.lineWidth << " pt 7 ps " << cfg_.pointSize << "\n";
    script << "set style line 2 lc rgb '#dc2626' lw " << cfg_.lineWidth << "\n";
    script << "set style line 3 lc rgb '#059669' lw " << cfg_.lineWidth << "\n";
    script << "set style line 4 lc rgb '#7c3aed' lw " << cfg_.lineWidth << "\n";
    script << "set style line 5 lc rgb '#d97706' lw " << cfg_.lineWidth << "\n";
    script << "set style line 6 lc rgb '#0891b2' lw " << cfg_.lineWidth << "\n";
    return script.str();
}

GnuplotEngine::GnuplotEngine(std::string assetsDir, PlotConfig cfg)
    : assetsDir_(std::move(assetsDir)), cfg_(std::move(cfg)) {
    std::filesystem::create_directories(assetsDir_);
    std::filesystem::create_directories(assetsDir_ + "/.plot_cache");
}

bool GnuplotEngine::isAvailable() const {
    return !findExecutableInPath("gnuplot").empty();
}

std::string GnuplotEngine::runScript(const std::string& id, const std::string& dataContent, const std::string& scriptContent) {
    const std::string safeId = sanitizeId(id);
    const std::string dataFile = assetsDir_ + "/" + safeId + ".dat";
    const std::string scriptFile = assetsDir_ + "/" + safeId + ".plt";
    const std::string outputFile = assetsDir_ + "/" + safeId + "." + cfg_.format;
    const std::string errFile = assetsDir_ + "/" + safeId + ".err.log";
    const std::string cacheHashFile = assetsDir_ + "/.plot_cache/" + safeId + ".hash";

    const std::string cacheKey = toHex(fnv1a64(
        dataContent + "\n@@\n" + scriptContent +
        "\nfmt=" + cfg_.format +
        "\ntheme=" + cfg_.theme +
        "\ngrid=" + std::string(cfg_.showGrid ? "1" : "0") +
        "\nlineWidth=" + std::to_string(cfg_.lineWidth) +
        "\npointSize=" + std::to_string(cfg_.pointSize)));

    {
        std::ifstream in(cacheHashFile);
        std::string existing;
        if (in && std::getline(in, existing)) {
            if (existing == cacheKey && std::filesystem::exists(outputFile)) {
                return outputFile;
            }
        }
    }

    std::ofstream dout(dataFile);
    if (!dout) return "";
    dout << dataContent;
    if (!dout.good()) return "";
    dout.close();

    std::ofstream sout(scriptFile);
    if (!sout) return "";
    sout << scriptContent;
    if (!sout.good()) return "";
    sout.close();

    const std::string gnuplotExe = findExecutableInPath("gnuplot");
    if (gnuplotExe.empty()) {
        return "";
    }
    const int rc = spawnGnuplot(gnuplotExe, scriptFile, errFile);

    std::error_code ec;
    std::filesystem::remove(dataFile, ec);
    std::filesystem::remove(scriptFile, ec);

    if (rc != 0 || !std::filesystem::exists(outputFile)) {
        std::ifstream errIn(errFile);
        std::string firstLine;
        std::getline(errIn, firstLine);
        if (!firstLine.empty()) {
            std::cerr << "[Seldon][Plot] Generation failed for id='" << safeId
                      << "' rc=" << rc
                      << " output='" << outputFile
                      << "' stderr='" << firstLine << "' full_log='" << errFile << "'\n";
        } else {
            std::cerr << "[Seldon][Plot] Generation failed for id='" << safeId
                      << "' rc=" << rc
                      << " output='" << outputFile
                      << "' full_log='" << errFile << "'\n";
        }
        return "";
    }

    std::filesystem::remove(errFile, ec);
    std::ofstream hout(cacheHashFile);
    if (hout) {
        hout << cacheKey;
    }

    return outputFile;
}

std::string GnuplotEngine::histogram(const std::string& id,
                                     const std::vector<double>& values,
                                     const std::string& title) {
    return histogram(id, values, title, HistogramOptions{});
}

std::string GnuplotEngine::histogram(const std::string& id,
                                     const std::vector<double>& values,
                                     const std::string& title,
                                     const HistogramOptions& options) {
    std::vector<double> vals = finiteValues(values);
    if (vals.empty()) return "";
    std::sort(vals.begin(), vals.end());

    const double minV = vals.front();
    const double maxV = vals.back();
    const double range = maxV - minV;

    double binWidth = 1.0;
    bool useSturges = false;
    if (range <= 1e-12) {
        binWidth = 1.0;
    } else if (options.adaptiveBinning) {
        useSturges = PlotHeuristics::shouldUseSturgesGrouping(vals, 1e-12);
        if (useSturges) {
            const size_t sturgesBins = PlotHeuristics::sturgesBinCount(vals.size());
            binWidth = range / static_cast<double>(std::max<size_t>(1, sturgesBins));
        } else {
            const double n = static_cast<double>(vals.size());
            const double iqr = StatsUtils::percentileSorted(vals, 0.75) - StatsUtils::percentileSorted(vals, 0.25);
            const double sigma = stddevSample(vals);
            double fdWidth = (iqr > 1e-12) ? (2.0 * iqr * std::pow(n, -1.0 / 3.0)) : 0.0;
            double scottWidth = (sigma > 1e-12) ? (3.5 * sigma * std::pow(n, -1.0 / 3.0)) : 0.0;
            if (fdWidth > 1e-12) {
                binWidth = fdWidth;
            } else if (scottWidth > 1e-12) {
                binWidth = scottWidth;
            } else {
                const size_t sturgesBins = PlotHeuristics::sturgesBinCount(vals.size());
                binWidth = range / static_cast<double>(std::max<size_t>(1, sturgesBins));
            }

            const double binsRaw = std::max(1.0, range / std::max(binWidth, 1e-9));
            const double binsClamped = std::clamp(binsRaw, 5.0, 96.0);
            binWidth = range / binsClamped;
        }
        if (binWidth < 1e-6) binWidth = 1e-6;
    } else {
        binWidth = std::max(1e-6, range / 20.0);
    }

    std::ostringstream data;
    for (double v : vals) data << v << "\n";

    bool drawKde = options.withDensityOverlay && vals.size() >= 80 && range > 1e-9;
    if (drawKde) {
        data << "\n\n";
        const size_t gridN = 160;
        std::vector<double> grid;
        grid.reserve(gridN);
        for (size_t i = 0; i < gridN; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(gridN - 1);
            grid.push_back(minV + t * range);
        }
        const double h = silvermanBandwidth(vals);
        std::vector<double> d = kdeEvaluate(vals, grid, h);
        for (size_t i = 0; i < gridN; ++i) {
            const double scaled = d[i] * static_cast<double>(vals.size()) * binWidth;
            data << grid[i] << " " << scaled << "\n";
        }
    }

    const std::string safeId = sanitizeId(id);
    const AxisRange xAxis = robustAxisRange(vals, 0.06);
    std::ostringstream script;
    script << styledHeader(id, title + (useSturges ? " (Sturges grouped)" : ""));
    script << "set xlabel 'Value' tc rgb '#374151'\n";
    script << "set ylabel 'Frequency' tc rgb '#374151'\n";
    script << "set yrange [0:*]\n";
    if (xAxis.valid) applyAxisFormatting(script, 'x', xAxis);
    script << "set boxwidth 0.9 relative\nset style fill solid 0.85 border lc rgb '#1d4ed8'\n";
    script << "binwidth=" << binWidth << "\n";
    script << "binstart=" << minV << "\n";
    script << "bin(x,width,start)=start+width*floor((x-start)/width)\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " index 0 using (bin($1,binwidth,binstart)):(1.0) smooth freq with boxes ls 1 title 'Histogram'";
    if (drawKde) {
        script << ", " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " index 1 using 1:2 with lines ls 2 title 'KDE (scaled)'";
    }
    script << "\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::bar(const std::string& id,
                               const std::vector<std::string>& labels,
                               const std::vector<double>& values,
                               const std::string& title) {
    const CategoricalSeries grouped = aggregateTopCategories(labels, values, 12);
    if (grouped.labels.size() < 2 || grouped.values.size() < 2) return "";
    if (PlotHeuristics::shouldAvoidCategoryHeavyCharts(labels.size(), 30)) return "";

    std::ostringstream data;
    double maxValue = 0.0;
    for (size_t i = 0; i < grouped.labels.size() && i < grouped.values.size(); ++i) {
        if (grouped.values[i] > maxValue) maxValue = grouped.values[i];
        data << i << " " << grouped.values[i] << "\n";
    }
    if (maxValue < 1.0) maxValue = 1.0;

    std::ostringstream xtics;
    xtics << "set xtics (";
    for (size_t i = 0; i < grouped.labels.size() && i < grouped.values.size(); ++i) {
        if (i > 0) xtics << ", ";
        xtics << quoteForGnuplot(normalizePlotLabel(grouped.labels[i])) << " " << i;
    }
    xtics << ") rotate by -25 font ',8'\n";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title + ((grouped.labels.size() < labels.size()) ? " (top categories)" : ""));
    script << "set style fill solid 0.85 border lc rgb '#1d4ed8'\n";
    script << "set boxwidth 0.8\nset yrange [0:" << (maxValue * 1.12) << "]\n";
    script << "set xlabel ''\nset ylabel 'Count' tc rgb '#374151'\n";
    script << xtics.str();
    script << "set xrange [-1:" << (grouped.labels.size()) << "]\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with boxes ls 1 notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::stackedBar(const std::string& id,
                                      const std::vector<std::string>& categories,
                                      const std::vector<std::string>& seriesLabels,
                                      const std::vector<std::vector<double>>& seriesValues,
                                      const std::string& title,
                                      const std::string& yLabel) {
    const size_t c = categories.size();
    const size_t s = seriesLabels.size();
    if (c == 0 || s == 0 || seriesValues.size() != s) return "";
    for (const auto& series : seriesValues) {
        if (series.size() != c) return "";
    }

    std::ostringstream data;
    double maxStack = 0.0;
    for (size_t i = 0; i < c; ++i) {
        data << quoteForDatafileString(normalizePlotLabel(categories[i], 20));
        double stackSum = 0.0;
        for (size_t j = 0; j < s; ++j) {
            const double v = std::max(0.0, std::isfinite(seriesValues[j][i]) ? seriesValues[j][i] : 0.0);
            stackSum += v;
            data << " " << v;
        }
        data << "\n";
        if (stackSum > maxStack) maxStack = stackSum;
    }
    if (maxStack <= 0.0) return "";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set style data histograms\n";
    script << "set style histogram rowstacked\n";
    script << "set style fill solid 0.88 border lc rgb '#ffffff'\n";
    script << "set boxwidth 0.82\n";
    script << "set ylabel " << quoteForGnuplot(yLabel) << " tc rgb '#374151'\n";
    script << "set yrange [0:" << (maxStack * 1.15) << "]\n";
    script << "set xtics rotate by -25 font ',8'\n";
    script << "set key top right\n";
    script << "plot ";
    for (size_t j = 0; j < s; ++j) {
        if (j > 0) script << ", ";
        script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " using " << (j + 2) << ":xtic(1) title " << quoteForGnuplot(normalizePlotLabel(seriesLabels[j], 24))
               << " ls " << ((j % 6) + 1);
    }
    script << "\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::scatter(const std::string& id,
                                   const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::string& title,
                                   bool withFitLine,
                                   double fitSlope,
                                   double fitIntercept,
                                   const std::string& fitLabel,
                                   bool withConfidenceBand,
                                   double confidenceZ,
                                   size_t downsampleThreshold) {
    std::vector<std::pair<double, double>> pts = finitePairs(x, y);
    if (pts.size() < 2) return "";

    const size_t maxPts = std::max<size_t>(2000, downsampleThreshold);
    const bool downsampled = pts.size() > maxPts;
    std::vector<std::pair<double, double>> drawPts = downsampleEvenly(pts, maxPts);

    std::ostringstream data;
    for (const auto& p : drawPts) data << p.first << " " << p.second << "\n";

    std::vector<std::pair<double, double>> fitPts = drawPts;
    std::sort(fitPts.begin(), fitPts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    bool canBand = withFitLine && withConfidenceBand && pts.size() >= 10;
    double xMean = 0.0;
    double sxx = 0.0;
    double sse = 0.0;
    if (canBand) {
        std::vector<double> xv;
        xv.reserve(pts.size());
        xMean = 0.0;
        for (const auto& p : pts) {
            xv.push_back(p.first);
            xMean += p.first;
        }
        xMean /= static_cast<double>(pts.size());
        for (const auto& p : pts) {
            const double xd = p.first - xMean;
            sxx += xd * xd;
            const double yhat = fitSlope * p.first + fitIntercept;
            const double e = p.second - yhat;
            sse += e * e;
        }
        if (sxx <= 1e-12) canBand = false;
    }

    data << "\n\n";
    for (const auto& p : fitPts) {
        const double yhat = fitSlope * p.first + fitIntercept;
        double lo = yhat;
        double hi = yhat;
        if (canBand) {
            const double dof = static_cast<double>(std::max<size_t>(1, pts.size() - 2));
            const double s2 = sse / dof;
            const double se = std::sqrt(std::max(0.0, s2) * (1.0 / static_cast<double>(pts.size()) + ((p.first - xMean) * (p.first - xMean)) / sxx));
            const double half = std::max(0.0, confidenceZ) * se;
            lo = yhat - half;
            hi = yhat + half;
        }
        data << p.first << " " << yhat << " " << lo << " " << hi << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title + (downsampled ? " (downsampled)" : ""));
    script << "set xlabel 'X' tc rgb '#374151'\n";
    script << "set ylabel 'Y' tc rgb '#374151'\n";
    script << "set key top left\n";
    std::vector<double> xv;
    std::vector<double> yv;
    xv.reserve(drawPts.size());
    yv.reserve(drawPts.size());
    for (const auto& p : drawPts) {
        xv.push_back(p.first);
        yv.push_back(p.second);
    }
    const AxisRange xAxis = robustAxisRange(xv, 0.08);
    const AxisRange yAxis = robustAxisRange(yv, 0.08);
    if (xAxis.valid) applyAxisFormatting(script, 'x', xAxis);
    if (yAxis.valid) applyAxisFormatting(script, 'y', yAxis);

    script << "plot ";
    if (withFitLine && canBand) {
        script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " index 1 using 1:3:4 with filledcurves lc rgb '#93c5fd' fs solid 0.30 title 'Confidence band', ";
    }
    script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " index 0 using 1:2 with points ls 1 title 'Observations'";
    if (withFitLine) {
        script << ", " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " index 1 using 1:2 with lines ls 2 title " << quoteForGnuplot(fitLabel);
    }
    script << "\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::residual(const std::string& id,
                                    const std::vector<double>& fitted,
                                    const std::vector<double>& residuals,
                                    const std::string& title) {
    const size_t n = std::min(fitted.size(), residuals.size());
    if (n < 3) return "";

    std::ostringstream data;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(fitted[i]) || !std::isfinite(residuals[i])) continue;
        data << fitted[i] << " " << residuals[i] << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Fitted value' tc rgb '#374151'\n";
    script << "set ylabel 'Residual' tc rgb '#374151'\n";
    script << "set key top left\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2 with points ls 1 title 'Residuals', 0 with lines ls 2 title 'Zero line'\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::line(const std::string& id,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::string& title) {
    if (x.empty() || y.empty()) return "";

    std::ostringstream data;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) data << x[i] << " " << y[i] << "\n";

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Epoch' tc rgb '#374151'\n";
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    std::vector<double> xv;
    std::vector<double> yv;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        xv.push_back(x[i]);
        yv.push_back(y[i]);
    }
    const AxisRange xAxis = robustAxisRange(xv, 0.04);
    const AxisRange yAxis = robustAxisRange(yv, 0.08);
    if (xAxis.valid) applyAxisFormatting(script, 'x', xAxis);
    if (yAxis.valid) applyAxisFormatting(script, 'y', yAxis);
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with lines ls 1 title 'Series'\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::multiLine(const std::string& id,
                                     const std::vector<double>& x,
                                     const std::vector<std::vector<double>>& series,
                                     const std::vector<std::string>& labels,
                                     const std::string& title,
                                     const std::string& yLabel) {
    if (x.empty() || series.empty()) return "";
    const size_t m = std::min(series.size(), labels.size());
    if (m == 0) return "";

    size_t n = x.size();
    for (size_t j = 0; j < m; ++j) n = std::min(n, series[j].size());
    if (n == 0) return "";

    std::ostringstream data;
    for (size_t i = 0; i < n; ++i) {
        data << x[i];
        for (size_t j = 0; j < m; ++j) data << " " << series[j][i];
        data << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Epoch' tc rgb '#374151'\n";
    script << "set ylabel " << quoteForGnuplot(yLabel) << " tc rgb '#374151'\n";
    script << "set key top right\n";
    script << "plot ";
    for (size_t j = 0; j < m; ++j) {
        if (j > 0) script << ", ";
        script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " using 1:" << (j + 2)
               << " with lines ls " << ((j % 6) + 1)
               << " title " << quoteForGnuplot(normalizePlotLabel(labels[j], 24));
    }
    script << "\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::ogive(const std::string& id,
                                 const std::vector<double>& values,
                                 const std::string& title) {
    std::vector<double> sorted = finiteValues(values);
    if (sorted.size() < 2) return "";
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream data;
    const size_t n = sorted.size();
    for (size_t i = 0; i < n; ++i) {
        const double cumulativePct = (100.0 * static_cast<double>(i + 1)) / static_cast<double>(n);
        data << sorted[i] << " " << cumulativePct << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Value' tc rgb '#374151'\n";
    script << "set ylabel 'Cumulative %' tc rgb '#374151'\n";
    script << "set yrange [0:100]\n";
    script << "set key top left\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2 with linespoints ls 3 pt 7 ps " << std::max(0.1, cfg_.pointSize * 0.65) << " title 'Ogive'\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::box(const std::string& id,
                               const std::vector<double>& values,
                               const std::string& title) {
    std::vector<double> vals = finiteValues(values);
    if (vals.size() < 4) return "";

    std::ostringstream data;
    for (double v : vals) data << v << "\n";

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel ''\n";
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    script << "set style data boxplot\n";
    script << "set style boxplot outliers pointtype 7\n";
    script << "set style fill solid 0.45 border lc rgb '#1d4ed8'\n";
    script << "set boxwidth 0.35\n";
    script << "set xrange [0.5:1.5]\n";
    script << "set xtics (" << quoteForGnuplot("Distribution") << " 1)\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using (1):1 title 'Box Plot' lc rgb '#2563eb'\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::categoricalDistribution(const std::string& id,
                                                   const std::vector<std::string>& categories,
                                                   const std::vector<double>& values,
                                                   const std::string& title) {
    const size_t n = std::min(categories.size(), values.size());
    if (n < 20) return "";

    std::map<std::string, std::vector<double>> byCat;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) continue;
        if (categories[i].empty()) continue;
        byCat[categories[i]].push_back(values[i]);
    }
    if (byCat.size() < 2) return "";

    std::vector<std::pair<std::string, std::vector<double>>> ordered;
    ordered.reserve(byCat.size());
    for (auto& kv : byCat) ordered.push_back(kv);
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();
    });
    if (ordered.size() > 10) ordered.resize(10);

    std::vector<size_t> sizes;
    sizes.reserve(ordered.size());
    for (const auto& kv : ordered) sizes.push_back(kv.second.size());
    std::sort(sizes.begin(), sizes.end());
    const size_t medianPerCat = sizes[sizes.size() / 2];
    const bool useViolin = ordered.size() <= 6 && medianPerCat >= 35;

    std::ostringstream data;
    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title + (useViolin ? " (violin mode)" : " (boxen mode)"));
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    script << "set xrange [0.5:" << (ordered.size() + 0.5) << "]\n";
    script << "set xtics (";
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(ordered[i].first, 16)) << " " << (i + 1);
    }
    script << ") rotate by -25 font ',8'\n";

    if (useViolin) {
        for (size_t i = 0; i < ordered.size(); ++i) {
            auto vals = ordered[i].second;
            std::sort(vals.begin(), vals.end());
            const double minV = vals.front();
            const double maxV = vals.back();
            const double span = std::max(1e-9, maxV - minV);

            std::vector<double> grid;
            grid.reserve(90);
            for (size_t g = 0; g < 90; ++g) {
                const double t = static_cast<double>(g) / 89.0;
                grid.push_back(minV + span * t);
            }
            std::vector<double> dens = kdeEvaluate(vals, grid, silvermanBandwidth(vals));
            const double maxD = *std::max_element(dens.begin(), dens.end());
            const double scale = (maxD <= 1e-12) ? 0.0 : (0.40 / maxD);

            for (size_t g = 0; g < grid.size(); ++g) {
                const double half = dens[g] * scale;
                data << grid[g] << " " << ((i + 1) - half) << " " << ((i + 1) + half) << "\n";
            }
            data << "\n\n";
        }

        script << "plot ";
        for (size_t i = 0; i < ordered.size(); ++i) {
            if (i > 0) script << ", ";
            script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
                   << " index " << i
                   << " using 2:1:3 with filledcurves lc rgb '#93c5fd' fs solid 0.45 notitle";
        }
        script << "\n";
    } else {
        for (size_t i = 0; i < ordered.size(); ++i) {
            for (double v : ordered[i].second) {
                data << (i + 1) << " " << v << "\n";
            }
        }

        script << "set style data boxplot\n";
        script << "set style boxplot outliers pointtype 7\n";
        script << "set style fill solid 0.35 border lc rgb '#1d4ed8'\n";
        script << "set boxwidth 0.45\n";
        script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " using 1:2 title 'Distribution' lc rgb '#2563eb'\n";
    }

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::facetedScatter(const std::string& id,
                                          const std::vector<double>& x,
                                          const std::vector<double>& y,
                                          const std::vector<std::string>& facets,
                                          const std::string& title,
                                          size_t maxFacets) {
    const size_t n = std::min({x.size(), y.size(), facets.size()});
    if (n < 24) return "";

    std::unordered_map<std::string, std::vector<std::pair<double, double>>> byFacet;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        if (facets[i].empty()) continue;
        byFacet[facets[i]].push_back({x[i], y[i]});
    }

    std::vector<std::pair<std::string, std::vector<std::pair<double, double>>>> ranked;
    ranked.reserve(byFacet.size());
    for (auto& kv : byFacet) ranked.push_back(kv);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();
    });

    std::vector<std::pair<std::string, std::vector<std::pair<double, double>>>> keep;
    for (const auto& kv : ranked) {
        if (kv.second.size() < 8) continue;
        keep.push_back(kv);
        if (keep.size() >= std::max<size_t>(2, maxFacets)) break;
    }
    if (keep.size() < 2) return "";

    std::ostringstream data;
    for (const auto& kv : keep) {
        const auto sampled = downsampleEvenly(kv.second, 3000);
        for (const auto& p : sampled) data << p.first << " " << p.second << "\n";
        data << "\n\n";
    }

    const size_t cols = (keep.size() <= 2) ? keep.size() : 3;
    const size_t rows = (keep.size() + cols - 1) / cols;
    const std::string safeId = sanitizeId(id);

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set multiplot layout " << rows << "," << cols << " rowsfirst title " << quoteForGnuplot(title) << "\n";
    script << "set key off\n";
    for (size_t i = 0; i < keep.size(); ++i) {
        script << "set title " << quoteForGnuplot(normalizePlotLabel(keep[i].first, 24)) << "\n";
        script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " index " << i << " using 1:2 with points ls 1 notitle\n";
    }
    script << "unset multiplot\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::parallelCoordinates(const std::string& id,
                                               const std::vector<std::vector<double>>& matrix,
                                               const std::vector<std::string>& axisLabels,
                                               const std::string& title) {
    if (matrix.empty() || axisLabels.size() < 3) return "";
    const size_t cols = axisLabels.size();

    std::vector<std::vector<double>> rows;
    rows.reserve(matrix.size());
    for (const auto& row : matrix) {
        if (row.size() != cols) continue;
        bool ok = true;
        for (double v : row) {
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
        }
        if (ok) rows.push_back(row);
    }
    if (rows.size() < 8) return "";

    const size_t maxRows = 220;
    if (rows.size() > maxRows) {
        const double step = static_cast<double>(rows.size() - 1) / static_cast<double>(maxRows - 1);
        std::vector<std::vector<double>> sampled;
        sampled.reserve(maxRows);
        for (size_t i = 0; i < maxRows; ++i) {
            const size_t idx = static_cast<size_t>(std::llround(i * step));
            sampled.push_back(rows[std::min(idx, rows.size() - 1)]);
        }
        rows = std::move(sampled);
    }

    std::vector<double> mins(cols, std::numeric_limits<double>::infinity());
    std::vector<double> maxs(cols, -std::numeric_limits<double>::infinity());
    for (const auto& row : rows) {
        for (size_t c = 0; c < cols; ++c) {
            mins[c] = std::min(mins[c], row[c]);
            maxs[c] = std::max(maxs[c], row[c]);
        }
    }

    std::ostringstream data;
    for (const auto& row : rows) {
        for (size_t c = 0; c < cols; ++c) {
            const double span = maxs[c] - mins[c];
            const double norm = (span <= 1e-12) ? 0.5 : ((row[c] - mins[c]) / span);
            data << (c + 1) << " " << std::clamp(norm, 0.0, 1.0) << "\n";
        }
        data << "\n\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set ylabel 'Normalized value' tc rgb '#374151'\n";
    script << "set yrange [0:1]\n";
    script << "set xrange [1:" << cols << "]\n";
    script << "set xtics (";
    for (size_t c = 0; c < cols; ++c) {
        if (c > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(axisLabels[c], 16)) << " " << (c + 1);
    }
    script << ") rotate by -25 font ',8'\n";
    script << "unset key\n";
    script << "plot for [i=0:" << (rows.size() - 1) << "] "
           << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " index i using 1:2 with lines lc rgb '#2563eb' lw 1 notitle\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::timeSeriesTrend(const std::string& id,
                                           const std::vector<double>& x,
                                           const std::vector<double>& y,
                                           const std::string& title,
                                           bool xIsUnixTime) {
    std::vector<std::pair<double, double>> pts = finitePairs(x, y);
    if (pts.size() < 5) return "";

    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<double> xv;
    std::vector<double> yv;
    xv.reserve(pts.size());
    yv.reserve(pts.size());
    for (const auto& p : pts) {
        xv.push_back(p.first);
        yv.push_back(p.second);
    }

    double slope = 0.0;
    double intercept = 0.0;
    std::vector<double> linear = linearFitPredict(xv, yv, slope, intercept);
    const double linearRmse = rmse(yv, linear);

    bool polyOk = false;
    std::vector<double> poly2 = poly2Predict(xv, yv, polyOk);
    const double polyRmse = polyOk ? rmse(yv, poly2) : std::numeric_limits<double>::infinity();

    const size_t maWindow = std::clamp<size_t>(xv.size() / 10, 3, 25);
    std::vector<double> moving = movingAveragePredict(yv, maWindow);
    const double movingRmse = rmse(yv, moving);

    std::string label = "Linear trend";
    std::vector<double> trend = linear;
    double best = linearRmse;

    if (polyRmse * 1.05 < best) {
        best = polyRmse;
        trend = poly2;
        label = "Polynomial trend";
    }
    if (movingRmse * 1.02 < best) {
        trend = moving;
        label = "Moving-average trend";
    }

    std::ostringstream data;
    for (size_t i = 0; i < xv.size(); ++i) {
        data << xv[i] << " " << yv[i] << " " << trend[i] << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title + " (" + label + ")");
    if (xIsUnixTime) {
        script << "set xdata time\n";
        script << "set timefmt '%s'\n";
        script << "set format x '%Y-%m-%d'\n";
        script << "set xtics rotate by -25\n";
        script << "set xlabel 'Time' tc rgb '#374151'\n";
    } else {
        script << "set xlabel 'Time index' tc rgb '#374151'\n";
    }
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    script << "set key top left\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2 with lines ls 1 title 'Series', "
           << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:3 with lines ls 2 title " << quoteForGnuplot(label) << "\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::pie(const std::string& id,
                               const std::vector<std::string>& labels,
                               const std::vector<double>& values,
                               const std::string& title) {
    CategoricalSeries grouped = aggregateTopCategories(labels, values, 9);
    const size_t n = std::min(grouped.labels.size(), grouped.values.size());
    if (n < 2) return "";

    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (grouped.values[i] > 0.0) total += grouped.values[i];
    }
    if (total <= 0.0) return "";

    const std::vector<std::string> palette = {
        "#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4f46e5"
    };

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "unset key\n";
    script << "unset border\n";
    script << "unset xtics\n";
    script << "unset ytics\n";
    script << "unset grid\n";
    script << "set size square\n";
    script << "set xrange [-1.25:1.25]\n";
    script << "set yrange [-1.25:1.25]\n";

    constexpr double kPi = 3.14159265358979323846;
    double angleStart = 0.0;
    int objectId = 1;
    int labelId = 1;

    for (size_t i = 0; i < n; ++i) {
        if (grouped.values[i] <= 0.0) continue;
        const double frac = grouped.values[i] / total;
        const double angleEnd = angleStart + frac * 2.0 * kPi;

        script << "set object " << objectId++ << " polygon from 0,0 ";
        const int seg = 36;
        for (int s = 0; s <= seg; ++s) {
            const double t = angleStart + (angleEnd - angleStart) * static_cast<double>(s) / static_cast<double>(seg);
            const double px = std::cos(t);
            const double py = std::sin(t);
            script << "to " << px << "," << py << " ";
        }
        script << "fs solid 0.93 border lc rgb '#ffffff' fc rgb "
               << quoteForGnuplot(palette[i % palette.size()]) << "\n";

        const double mid = 0.5 * (angleStart + angleEnd);
        const double lx = 1.12 * std::cos(mid);
        const double ly = 1.12 * std::sin(mid);
        const int pct = static_cast<int>(std::round(frac * 100.0));
        script << "set label " << labelId++ << " "
             << quoteForGnuplot(normalizePlotLabel(grouped.labels[i], 20) + " (" + std::to_string(pct) + "%)")
               << " at " << lx << "," << ly << " center tc rgb '#111827' font ',8'\n";

        angleStart = angleEnd;
    }

    script << "plot 1/0 notitle\n";
    return runScript(id, "", script.str());
}

std::string GnuplotEngine::gantt(const std::string& id,
                                 const std::vector<std::string>& taskNames,
                                 const std::vector<int64_t>& startUnixSeconds,
                                 const std::vector<int64_t>& endUnixSeconds,
                                 const std::vector<std::string>& semantics,
                                 const std::string& title) {
    const size_t n = std::min(taskNames.size(), std::min(startUnixSeconds.size(), endUnixSeconds.size()));
    if (n == 0) return "";

    int64_t minStart = startUnixSeconds[0];
    int64_t maxEnd = endUnixSeconds[0];
    for (size_t i = 0; i < n; ++i) {
        minStart = std::min(minStart, startUnixSeconds[i]);
        maxEnd = std::max(maxEnd, endUnixSeconds[i]);
    }
    if (maxEnd <= minStart) return "";

    const std::vector<std::string> palette = {
        "#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4f46e5"
    };

    std::ostringstream ytics;
    ytics << "set ytics (";
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) ytics << ", ";
        ytics << quoteForGnuplot(normalizePlotLabel(taskNames[i], 32)) << " " << (i + 1);
    }
    ytics << ")\n";

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xdata time\n";
    script << "set timefmt '%s'\n";
    script << "set format x '%Y-%m-%d'\n";
    script << "set xtics rotate by -25\n";
    script << "set xlabel 'Timeline' tc rgb '#374151'\n";
    script << "set ylabel 'Task' tc rgb '#374151'\n";
    script << "set yrange [0.5:" << (static_cast<double>(n) + 0.5) << "]\n";
    script << ytics.str();
    script << "set xrange [" << quoteForGnuplot(std::to_string(minStart)) << ":"
           << quoteForGnuplot(std::to_string(maxEnd + 86400)) << "]\n";
    script << "unset key\n";

    int objectId = 1;
    for (size_t i = 0; i < n; ++i) {
        const double yLow = static_cast<double>(i + 1) - 0.35;
        const double yHigh = static_cast<double>(i + 1) + 0.35;
        std::string semantic = taskNames[i];
        if (i < semantics.size() && !semantics[i].empty()) semantic = semantics[i];
        const std::string color = ganttColorForSemantic(semantic, palette[i % palette.size()]);
        script << "set object " << objectId++
               << " rect from " << quoteForGnuplot(std::to_string(startUnixSeconds[i])) << "," << yLow
               << " to " << quoteForGnuplot(std::to_string(endUnixSeconds[i])) << "," << yHigh
               << " fc rgb " << quoteForGnuplot(color)
               << " fs solid 0.85 border lc rgb '#ffffff'\n";
    }

    script << "plot 1/0 notitle\n";
    return runScript(id, "", script.str());
}

std::string GnuplotEngine::heatmap(const std::string& id,
                                   const std::vector<std::vector<double>>& matrix,
                                   const std::string& title,
                                   const std::vector<std::string>& labels) {
    if (matrix.empty() || matrix.front().empty()) return "";

    std::ostringstream data;
    for (size_t r = 0; r < matrix.size(); ++r) {
        for (size_t c = 0; c < matrix[r].size(); ++c) {
            data << c << " " << r << " " << matrix[r][c] << "\n";
        }
        data << "\n";
    }

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set view map\nset pm3d at b\nunset key\n";
    script << "set palette defined (-1 '#1e3a8a', 0 '#f3f4f6', 1 '#991b1b')\n";
    script << "set cbrange [-1:1]\n";
    if (!labels.empty() && labels.size() == matrix.size()) {
        script << "set xtics (";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) script << ", ";
            script << quoteForGnuplot(normalizePlotLabel(labels[i], 12)) << " " << i;
        }
        script << ") rotate by -35 font ',7'\n";
        script << "set ytics (";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) script << ", ";
            script << quoteForGnuplot(normalizePlotLabel(labels[i], 12)) << " " << i;
        }
        script << ") font ',7'\n";
    }
    script << "unset grid\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2:3 with pm3d\n";
    return runScript(id, data.str(), script.str());
}
