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
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

extern char** environ;

namespace {
std::string trimAsciiSpaces(const std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) ++start;
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(start, end - start);
}

std::string humanizeTextToken(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 12);

    auto isAlphaNum = [](char c) {
        const unsigned char uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) != 0;
    };

    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        const char prevRaw = (i > 0) ? text[i - 1] : '\0';
        const char nextRaw = (i + 1 < text.size()) ? text[i + 1] : '\0';

        if (ch < 32 || ch > 126) {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
            continue;
        }

        if (ch == '_' || ch == '\t' || ch == '\r' || ch == '\n') {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
            continue;
        }

        if ((ch == ':' || ch == '|') && !out.empty()) {
            if (out.back() == ' ') out.pop_back();
            out.push_back(' ');
            out.push_back(ch);
            out.push_back(' ');
            continue;
        }

        if ((ch == '-' || ch == '/' || ch == '+') && isAlphaNum(prevRaw) && isAlphaNum(nextRaw)) {
            out.push_back(ch);
            continue;
        }
        if (ch == '-' || ch == '/' || ch == '+') {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
            continue;
        }

        const bool isUpper = std::isupper(static_cast<unsigned char>(ch));
        const bool isLower = std::islower(static_cast<unsigned char>(ch));
        const bool isDigit = std::isdigit(static_cast<unsigned char>(ch));
        const bool prevLower = !out.empty() && std::islower(static_cast<unsigned char>(out.back()));
        const bool prevUpper = !out.empty() && std::isupper(static_cast<unsigned char>(out.back()));
        const bool prevDigit = !out.empty() && std::isdigit(static_cast<unsigned char>(out.back()));
        const bool nextLower = std::islower(static_cast<unsigned char>(nextRaw));

        if (!out.empty()) {
            const bool camelBoundary = isUpper && (prevLower || (prevUpper && nextLower));
            const bool digitBoundary = (isDigit && !prevDigit) || (!isDigit && prevDigit && (isUpper || isLower));
            if ((camelBoundary || digitBoundary) && out.back() != ' ') out.push_back(' ');
        }

        out.push_back(ch);
    }

    std::string collapsed;
    collapsed.reserve(out.size());
    bool inSpace = false;
    for (char ch : out) {
        if (ch == ' ') {
            if (!inSpace) collapsed.push_back(' ');
            inSpace = true;
        } else {
            collapsed.push_back(ch);
            inSpace = false;
        }
    }
    return trimAsciiSpaces(collapsed);
}

std::string normalizePlotLabel(const std::string& label, size_t maxLen = 28) {
    std::string out = humanizeTextToken(label);
    if (out.empty()) out = "Unnamed";
    if (out.size() > maxLen) {
        out = out.substr(0, maxLen - 3) + "...";
    }
    return out;
}

std::string normalizePlotTitle(const std::string& title,
                               size_t wrapAt = 60,
                               size_t maxTotal = 140) {
    std::string collapsed = humanizeTextToken(title);
    if (collapsed.empty()) collapsed = "Seldon Plot";

    if (collapsed.size() > maxTotal) {
        collapsed = collapsed.substr(0, maxTotal - 3) + "...";
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

bool splitByKeyword(const std::string& text,
                    const std::string& keyword,
                    std::string& left,
                    std::string& right) {
    if (text.empty() || keyword.empty()) return false;
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::string keyLower = keyword;
    std::transform(keyLower.begin(), keyLower.end(), keyLower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const size_t pos = lower.find(keyLower);
    if (pos == std::string::npos) return false;

    left = trimAsciiSpaces(text.substr(0, pos));
    right = trimAsciiSpaces(text.substr(pos + keyword.size()));
    return !left.empty() && !right.empty();
}

struct AxisLabelGuess {
    std::string xLabel;
    std::string yLabel;
    bool valid = false;
};

AxisLabelGuess inferAxisLabelsFromTitle(const std::string& rawTitle,
                                        const std::string& defaultX,
                                        const std::string& defaultY) {
    AxisLabelGuess out;
    out.xLabel = defaultX;
    out.yLabel = defaultY;

    const std::string title = humanizeTextToken(rawTitle);
    if (title.empty()) return out;

    std::string left;
    std::string right;
    if (splitByKeyword(title, " vs ", left, right) ||
        splitByKeyword(title, " -> ", left, right)) {
        out.xLabel = normalizePlotLabel(left, 32);
        out.yLabel = normalizePlotLabel(right, 32);
        out.valid = true;
        return out;
    }
    if (splitByKeyword(title, " over ", left, right) ||
        splitByKeyword(title, " by ", left, right)) {
        out.yLabel = normalizePlotLabel(left, 32);
        out.xLabel = normalizePlotLabel(right, 32);
        out.valid = true;
        return out;
    }
    return out;
}

bool appearsUnixTimeAxis(const std::vector<double>& x) {
    if (x.size() < 4) return false;
    double minV = std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    size_t monotonic = 0;
    size_t adjacent = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) continue;
        minV = std::min(minV, x[i]);
        maxV = std::max(maxV, x[i]);
        if (i > 0 && std::isfinite(x[i - 1])) {
            ++adjacent;
            if (x[i] >= x[i - 1]) ++monotonic;
        }
    }
    if (!std::isfinite(minV) || !std::isfinite(maxV) || maxV <= minV) return false;
    const bool plausibleEpoch = minV > 1e8 && maxV < 5e10;
    const bool meaningfulSpan = (maxV - minV) >= 3600.0;
    const bool mostlyMonotonic = adjacent == 0 || (static_cast<double>(monotonic) / static_cast<double>(adjacent)) >= 0.7;
    return plausibleEpoch && meaningfulSpan && mostlyMonotonic;
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

std::vector<size_t> greedyClusteredOrder(const std::vector<std::vector<double>>& matrix) {
    const size_t n = matrix.size();
    if (n == 0) return {};
    std::vector<size_t> order;
    order.reserve(n);
    std::vector<uint8_t> used(n, static_cast<uint8_t>(0));

    // Choose a seed with the highest aggregate absolute similarity so the walk
    // starts from a dense region of the matrix.
    size_t seed = 0;
    double bestRowDispersion = -1.0;
    for (size_t i = 0; i < n; ++i) {
        if (i >= matrix[i].size()) continue;
        double s = 0.0;
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            if (i == j) continue;
            s += std::abs(matrix[i][j]);
        }
        if (s > bestRowDispersion) {
            bestRowDispersion = s;
            seed = i;
        }
    }

    order.push_back(seed);
    used[seed] = static_cast<uint8_t>(1);
    // Greedily append the most similar unused neighbor to keep related rows/cols adjacent.
    while (order.size() < n) {
        const size_t last = order.back();
        size_t next = static_cast<size_t>(-1);
        double best = -1.0;
        for (size_t j = 0; j < n; ++j) {
            if (used[j]) continue;
            if (last >= matrix.size() || j >= matrix[last].size()) continue;
            const double sim = std::abs(matrix[last][j]);
            if (sim > best) {
                best = sim;
                next = j;
            }
        }
        if (next == static_cast<size_t>(-1)) {
            for (size_t j = 0; j < n; ++j) {
                if (!used[j]) {
                    next = j;
                    break;
                }
            }
        }
        order.push_back(next);
        used[next] = static_cast<uint8_t>(1);
    }
    return order;
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
        script << "set format " << axis << " '%.5g'\n";
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

    if (values.size() > 1000000) {
        std::vector<std::pair<double, double>> sample;
        sample.reserve(maxPoints);
        uint64_t state = 0x9e3779b97f4a7c15ULL;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i < maxPoints) {
                sample.push_back(values[i]);
            } else {
                state ^= (state << 7);
                state ^= (state >> 9);
                state += 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(i);
                const size_t j = static_cast<size_t>(state % static_cast<uint64_t>(i + 1));
                if (j < maxPoints) {
                    sample[j] = values[i];
                }
            }
        }
        std::sort(sample.begin(), sample.end(), [](const auto& a, const auto& b) {
            if (a.first == b.first) return a.second < b.second;
            return a.first < b.first;
        });
        return sample;
    }

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
    const int errFd = ::open(stderrPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (errFd < 0) return -1;

    int fdFlags = ::fcntl(errFd, F_GETFD);
    if (fdFlags < 0 || ::fcntl(errFd, F_SETFD, fdFlags | FD_CLOEXEC) < 0) {
        ::close(errFd);
        return -1;
    }

    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        ::close(errFd);
        return -1;
    }

    if (::posix_spawn_file_actions_adddup2(&actions, errFd, STDERR_FILENO) != 0 ||
        ::posix_spawn_file_actions_addclose(&actions, errFd) != 0) {
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(errFd);
        return -1;
    }

    posix_spawnattr_t attr;
    if (::posix_spawnattr_init(&attr) != 0) {
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(errFd);
        return -1;
    }

    short spawnFlags = 0;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    spawnFlags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    if (spawnFlags != 0 && ::posix_spawnattr_setflags(&attr, spawnFlags) != 0) {
        ::posix_spawnattr_destroy(&attr);
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(errFd);
        return -1;
    }

    const char* argvRaw[] = {executable.c_str(), scriptPath.c_str(), nullptr};
    char* const* argv = const_cast<char* const*>(argvRaw);
    pid_t pid = -1;
    const int spawnRc = ::posix_spawn(&pid, executable.c_str(), &actions, &attr, argv, environ);

    ::posix_spawnattr_destroy(&attr);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(errFd);

    if (spawnRc != 0 || pid <= 0) return -1;

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
            << " font ',14'\n";
    script << "set tmargin 3.4\nset bmargin 4.6\nset lmargin 8.6\nset rmargin 3.2\n";
    script << "set border linewidth " << cfg_.lineWidth << " lc rgb " << quoteForGnuplot(borderColor) << "\n";
    script << "set tics textcolor rgb " << quoteForGnuplot(ticColor) << " font ',10'\n";
    script << "set tics out nomirror\nset mxtics 2\nset mytics 2\n";
    if (cfg_.showGrid) {
        script << "set grid back lc rgb " << quoteForGnuplot(gridColor) << " lw 1 dt 2\n";
    } else {
        script << "unset grid\n";
    }
    script << "set key top left opaque box lc rgb " << quoteForGnuplot(borderColor) << " font ',10'\n";
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
    static const std::string gnuplotExeCached = findExecutableInPath("gnuplot");
    if (gnuplotExeCached.empty()) {
        return "";
    }

    const std::string safeId = sanitizeId(id);
    const std::string dataFile = assetsDir_ + "/" + safeId + ".dat";
    const std::string scriptFile = assetsDir_ + "/" + safeId + ".plt";
    const std::string outputFile = assetsDir_ + "/" + safeId + "." + cfg_.format;
    const std::string errFile = assetsDir_ + "/" + safeId + ".err.log";
    const std::string cacheHashFile = assetsDir_ + "/.plot_cache/" + safeId + ".hash";

    const std::string combined = dataContent + "\n@@\n" + scriptContent +
        "\nfmt=" + cfg_.format +
        "\ntheme=" + cfg_.theme +
        "\ngrid=" + std::string(cfg_.showGrid ? "1" : "0") +
        "\nlineWidth=" + std::to_string(cfg_.lineWidth) +
        "\npointSize=" + std::to_string(cfg_.pointSize);
    const uint64_t h1 = fnv1a64(combined);
    const uint64_t h2 = fnv1a64(std::string(combined.rbegin(), combined.rend()));
    const std::string cacheKey = toHex(h1) + ":" + toHex(h2) + ":" + std::to_string(combined.size());

    {
        std::ifstream in(cacheHashFile);
        std::string existing;
        if (in && std::getline(in, existing)) {
            if (existing == cacheKey && std::filesystem::exists(outputFile)) {
                return outputFile;
            }
        }
    }

    std::ofstream dout(dataFile, std::ios::binary);
    if (!dout) return "";
    std::vector<char> dBuf(256 * 1024, '\0');
    dout.rdbuf()->pubsetbuf(dBuf.data(), static_cast<std::streamsize>(dBuf.size()));
    dout << dataContent;
    if (!dout.good()) return "";
    dout.close();

    std::ofstream sout(scriptFile, std::ios::binary);
    if (!sout) return "";
    std::vector<char> sBuf(64 * 1024, '\0');
    sout.rdbuf()->pubsetbuf(sBuf.data(), static_cast<std::streamsize>(sBuf.size()));
    sout << scriptContent;
    if (!sout.good()) return "";
    sout.close();

    const int rc = spawnGnuplot(gnuplotExeCached, scriptFile, errFile);

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
        hout.flush();
        if (!hout.good()) {
            std::cerr << "[Seldon][Plot] Failed to flush cache hash file: '" << cacheHashFile << "'\n";
        }
        hout.close();
        if (!hout) {
            std::cerr << "[Seldon][Plot] Failed to close cache hash file cleanly: '" << cacheHashFile << "'\n";
        }
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    const AxisRange xAxis = robustAxisRange(vals, 0.06);
    std::ostringstream script;
    script << styledHeader(id, title + (useSturges ? " (Sturges grouped)" : ""));
    script << "set xlabel 'Value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set ylabel 'Frequency' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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
    xtics << ") rotate by -25 font ',10'\n";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    script << styledHeader(id, title + ((grouped.labels.size() < labels.size()) ? " (top categories)" : ""));
    script << "set style fill solid 0.85 border lc rgb '#1d4ed8'\n";
    script << "set boxwidth 0.8\nset yrange [0:" << (maxValue * 1.12) << "]\n";
    script << "set xlabel ''\nset ylabel 'Count' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    script << styledHeader(id, title);
    script << "set style data histograms\n";
    script << "set style histogram rowstacked\n";
    script << "set style fill solid 0.88 border lc rgb '#ffffff'\n";
    script << "set boxwidth 0.82\n";
    script << "set ylabel " << quoteForGnuplot(yLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set yrange [0:" << (maxStack * 1.15) << "]\n";
    script << "set xtics rotate by -25 font ',10'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    const AxisLabelGuess labels = inferAxisLabelsFromTitle(title, "X", "Y");
    std::ostringstream script;
    script << styledHeader(id, title + (downsampled ? " (downsampled)" : ""));
    script << "set xlabel " << quoteForGnuplot(labels.xLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set ylabel " << quoteForGnuplot(labels.yLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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

std::string GnuplotEngine::scatter3D(const std::string& id,
                                     const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     const std::vector<double>& z,
                                     const std::string& title) {
    const size_t n = std::min({x.size(), y.size(), z.size()});
    if (n < 4) return "";

    std::ostringstream data;
    size_t kept = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i])) continue;
        data << x[i] << " " << y[i] << " " << z[i] << "\n";
        ++kept;
    }
    if (kept < 4) return "";

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'X'\n";
    script << "set ylabel 'Y'\n";
    script << "set zlabel 'Z'\n";
    script << "set ticslevel 0\n";
    script << "set key off\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2:3 with points pt 7 ps " << std::max(0.4, cfg_.pointSize)
           << " lc rgb '#2563eb' notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::surface(const std::string& id,
                                   const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::vector<std::vector<double>>& z,
                                   const std::string& title) {
    if (x.empty() || y.empty() || z.empty()) return "";
    if (z.size() != y.size()) return "";

    std::ostringstream data;
    for (size_t r = 0; r < y.size(); ++r) {
        if (r >= z.size()) break;
        const auto& row = z[r];
        for (size_t c = 0; c < x.size() && c < row.size(); ++c) {
            if (!std::isfinite(row[c])) continue;
            data << x[c] << " " << y[r] << " " << row[c] << "\n";
        }
        data << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set hidden3d\n";
    script << "set pm3d depthorder\n";
    script << "set xlabel 'X'\n";
    script << "set ylabel 'Y'\n";
    script << "set zlabel 'Value'\n";
    script << "set palette defined (0 '#1d4ed8', 1 '#60a5fa', 2 '#f59e0b', 3 '#dc2626')\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2:3 with pm3d title 'Surface'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Fitted value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set ylabel 'Residual' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    const AxisLabelGuess labels = inferAxisLabelsFromTitle(title, "Index", "Value");
    std::ostringstream script;
    script << styledHeader(id, title);
    std::vector<double> xv;
    std::vector<double> yv;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        xv.push_back(x[i]);
        yv.push_back(y[i]);
    }
    const bool xAsTime = appearsUnixTimeAxis(xv);
    if (xAsTime) {
        script << "set xdata time\nset timefmt '%s'\nset format x '%Y-%m-%d'\n";
        script << "set xtics rotate by -25\n";
        script << "set xlabel 'Time' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    } else {
        script << "set xlabel " << quoteForGnuplot(labels.xLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    }
    script << "set ylabel " << quoteForGnuplot(labels.yLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    const AxisRange xAxis = robustAxisRange(xv, 0.04);
    const AxisRange yAxis = robustAxisRange(yv, 0.08);
    if (xAxis.valid && !xAsTime) applyAxisFormatting(script, 'x', xAxis);
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    const AxisLabelGuess axisLabels = inferAxisLabelsFromTitle(title, "Index", yLabel);
    std::ostringstream script;
    script << styledHeader(id, title);
    std::vector<double> xv;
    xv.reserve(n);
    std::vector<double> yv;
    yv.reserve(n * m);
    for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(x[i])) xv.push_back(x[i]);
        for (size_t j = 0; j < m; ++j) {
            if (std::isfinite(series[j][i])) yv.push_back(series[j][i]);
        }
    }
    const bool xAsTime = appearsUnixTimeAxis(xv);
    if (xAsTime) {
        script << "set xdata time\nset timefmt '%s'\nset format x '%Y-%m-%d'\n";
        script << "set xtics rotate by -25\n";
        script << "set xlabel 'Time' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    } else {
        script << "set xlabel " << quoteForGnuplot(axisLabels.xLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    }
    script << "set ylabel " << quoteForGnuplot(axisLabels.yLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    const AxisRange xAxis = robustAxisRange(xv, 0.05);
    const AxisRange yAxis = robustAxisRange(yv, 0.08);
    if (xAxis.valid && !xAsTime) applyAxisFormatting(script, 'x', xAxis);
    if (yAxis.valid) applyAxisFormatting(script, 'y', yAxis);
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set ylabel 'Cumulative %' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel ''\n";
    script << "set ylabel 'Value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    script << styledHeader(id, title + (useViolin ? " (violin mode)" : " (boxen mode)"));
    script << "set ylabel 'Value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set xrange [0.5:" << (ordered.size() + 0.5) << "]\n";
    script << "set xtics (";
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(ordered[i].first, 16)) << " " << (i + 1);
    }
    script << ") rotate by -25 font ',10'\n";

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

std::string GnuplotEngine::violin(const std::string& id,
                                  const std::vector<std::string>& categories,
                                  const std::vector<double>& values,
                                  const std::string& title) {
    const size_t n = std::min(categories.size(), values.size());
    if (n < 20) return "";

    std::map<std::string, std::vector<double>> byCat;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i]) || categories[i].empty()) continue;
        byCat[categories[i]].push_back(values[i]);
    }
    if (byCat.size() < 2) return "";

    std::vector<std::pair<std::string, std::vector<double>>> ordered;
    for (auto& kv : byCat) ordered.push_back(kv);
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();
    });
    if (ordered.size() > 10) ordered.resize(10);

    std::ostringstream data;
    for (size_t i = 0; i < ordered.size(); ++i) {
        auto vals = ordered[i].second;
        std::sort(vals.begin(), vals.end());
        if (vals.size() < 6) continue;
        const double minV = vals.front();
        const double maxV = vals.back();
        const double span = std::max(1e-9, maxV - minV);

        std::vector<double> grid;
        grid.reserve(100);
        for (size_t g = 0; g < 100; ++g) {
            const double t = static_cast<double>(g) / 99.0;
            grid.push_back(minV + span * t);
        }
        const std::vector<double> dens = kdeEvaluate(vals, grid, silvermanBandwidth(vals));
        const double maxD = dens.empty() ? 0.0 : *std::max_element(dens.begin(), dens.end());
        const double scale = (maxD <= 1e-12) ? 0.0 : (0.42 / maxD);

        for (size_t g = 0; g < grid.size(); ++g) {
            const double half = dens[g] * scale;
            data << grid[g] << " " << ((i + 1) - half) << " " << ((i + 1) + half) << "\n";
        }
        data << "\n\n";
    }

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set ylabel 'Value'\n";
    script << "set xrange [0.5:" << (ordered.size() + 0.5) << "]\n";
    script << "set xtics (";
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(ordered[i].first, 16)) << " " << (i + 1);
    }
    script << ") rotate by -25 font ',10'\n";
    script << "plot ";
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(assetsDir_ + "/" + sanitizeId(id) + ".dat")
               << " index " << i
               << " using 2:1:3 with filledcurves lc rgb '#93c5fd' fs solid 0.45 notitle";
    }
    script << "\n";

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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set ylabel 'Normalized value' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set yrange [0:1]\n";
    script << "set xrange [1:" << cols << "]\n";
    script << "set xtics (";
    for (size_t c = 0; c < cols; ++c) {
        if (c > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(axisLabels[c], 16)) << " " << (c + 1);
    }
    script << ") rotate by -25 font ',10'\n";
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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    std::ostringstream script;
    script << styledHeader(id, title + " (" + label + ")");
    if (xIsUnixTime) {
        script << "set xdata time\n";
        script << "set timefmt '%s'\n";
        script << "set format x '%Y-%m-%d'\n";
        script << "set xtics rotate by -25\n";
        script << "set xlabel 'Time' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    } else {
        script << "set xlabel 'Time index' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    }
    AxisLabelGuess labels = inferAxisLabelsFromTitle(title, "Time", "Value");
    script << "set ylabel " << quoteForGnuplot(labels.yLabel) << " tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    const AxisRange yAxis = robustAxisRange(yv, 0.08);
    if (yAxis.valid) applyAxisFormatting(script, 'y', yAxis);
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
             << " at " << lx << "," << ly << " center tc rgb '#111827' font ',10'\n";

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
    const std::string axisColor = (cfg_.theme == "dark") ? "#e5e7eb" : "#374151";
    script << styledHeader(id, title);
    script << "set xdata time\n";
    script << "set timefmt '%s'\n";
    script << "set format x '%Y-%m-%d'\n";
    script << "set xtics rotate by -25\n";
    script << "set xlabel 'Timeline' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set ylabel 'Task' tc rgb " << quoteForGnuplot(axisColor) << " font ',11'\n";
    script << "set yrange [0.5:" << (static_cast<double>(n) + 0.5) << "]\n";
    script << ytics.str();
    script << "set xrange [" << minStart << ":" << (maxEnd + 86400) << "]\n";
    script << "unset key\n";

    int objectId = 1;
    for (size_t i = 0; i < n; ++i) {
        const double yLow = static_cast<double>(i + 1) - 0.35;
        const double yHigh = static_cast<double>(i + 1) + 0.35;
        std::string semantic;
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
        script << ") rotate by -35 font ',9'\n";
        script << "set ytics (";
        for (size_t i = 0; i < labels.size(); ++i) {
            if (i > 0) script << ", ";
            script << quoteForGnuplot(normalizePlotLabel(labels[i], 12)) << " " << i;
        }
        script << ") font ',9'\n";
    }
    script << "unset grid\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2:3 with pm3d\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::clusteredHeatmap(const std::string& id,
                                            const std::vector<std::vector<double>>& matrix,
                                            const std::string& title,
                                            const std::vector<std::string>& labels,
                                            bool withDendrogram) {
    if (matrix.empty() || matrix.front().empty()) return "";
    const std::vector<size_t> order = greedyClusteredOrder(matrix);
    if (order.empty()) return "";

    std::vector<std::vector<double>> clustered(order.size(), std::vector<double>(order.size(), 0.0));
    std::vector<std::string> clusteredLabels;
    clusteredLabels.reserve(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        const size_t ri = order[i];
        clusteredLabels.push_back((ri < labels.size()) ? labels[ri] : ("f" + std::to_string(ri + 1)));
        for (size_t j = 0; j < order.size(); ++j) {
            const size_t cj = order[j];
            if (ri < matrix.size() && cj < matrix[ri].size()) {
                clustered[i][j] = matrix[ri][cj];
            }
        }
    }

    if (!withDendrogram) {
        return heatmap(id, clustered, title, clusteredLabels);
    }

    std::ostringstream data;
    for (size_t r = 0; r < clustered.size(); ++r) {
        for (size_t c = 0; c < clustered[r].size(); ++c) {
            data << c << " " << r << " " << clustered[r][c] << "\n";
        }
        data << "\n";
    }
    data << "\n\n";
    for (size_t i = 1; i < clustered.size(); ++i) {
        const double d = 1.0 - std::abs(clustered[i - 1][i]);
        data << d << " " << i << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title + " (clustered with dendrogram)");
    script << "set multiplot layout 1,2 columnsfirst title ''\n";

    script << "unset key\n";
    script << "set size ratio -1\n";
    script << "set lmargin 6\nset rmargin 2\nset tmargin 3\nset bmargin 4\n";
    script << "set xlabel 'Distance'\nset ylabel ''\n";
    script << "set ytics (";
    for (size_t i = 0; i < clusteredLabels.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(clusteredLabels[i], 12)) << " " << i;
    }
    script << ") font ',8'\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " index 1 using 1:2 with impulses lc rgb '#374151' lw 2 notitle\n";

    script << "set view map\nset pm3d at b\nunset key\n";
    script << "set palette defined (-1 '#1e3a8a', 0 '#f3f4f6', 1 '#991b1b')\n";
    script << "set cbrange [-1:1]\n";
    script << "set xtics (";
    for (size_t i = 0; i < clusteredLabels.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(clusteredLabels[i], 12)) << " " << i;
    }
    script << ") rotate by -35 font ',8'\n";
    script << "set ytics (";
    for (size_t i = 0; i < clusteredLabels.size(); ++i) {
        if (i > 0) script << ", ";
        script << quoteForGnuplot(normalizePlotLabel(clusteredLabels[i], 12)) << " " << i;
    }
    script << ") font ',8'\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " index 0 using 1:2:3 with pm3d\n";
    script << "unset multiplot\n";

    return runScript(id, data.str(), script.str());
}
