#include "CausalDiscovery.h"

#include "CommonUtils.h"
#include "MathUtils.h"
#include "Statistics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
constexpr double kBicSigma2Floor = 1e-12;
constexpr double kBicTieThreshold = 1e-6;


struct CausalDataView {
    std::vector<size_t> nodeColumnIdx;
    std::vector<std::vector<double>> rows;
};

struct TemporalSeries {
    std::vector<double> axis;
    std::string axisName;
    bool valid = false;
};

struct CiTestResult {
    bool independent = false;
    double pValue = 1.0;
    double effect = 0.0;
    bool usedKernel = false;
};

struct DiscoveryCoreResult {
    std::vector<std::vector<bool>> directed;
    std::vector<std::vector<bool>> undirected;
    std::vector<std::vector<std::unordered_set<size_t>>> sepsets;
    bool usedLiNGAM = false;
};

bool hasToken(const std::string& text, const std::vector<std::string>& tokens) {
    const std::string lower = CommonUtils::toLower(text);
    for (const auto& token : tokens) {
        if (lower.find(token) != std::string::npos) return true;
    }
    return false;
}

std::vector<std::string> engineeredRoots(const std::string& name) {
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
    const size_t powPos = lower.find("_pow");
    if (powPos != std::string::npos && powPos > 0) {
        pushUnique(lower.substr(0, powPos));
        return roots;
    }
    const size_t logPos = lower.find("_log1p_abs");
    if (logPos != std::string::npos && logPos > 0) {
        pushUnique(lower.substr(0, logPos));
        return roots;
    }

    pushUnique(lower);
    return roots;
}

bool sharesEngineeredFamilyRoot(const std::string& a, const std::string& b) {
    const std::string la = CommonUtils::toLower(CommonUtils::trim(a));
    const std::string lb = CommonUtils::toLower(CommonUtils::trim(b));
    if (la.empty() || lb.empty()) return false;

    const bool engineeredA = la.find("_pow") != std::string::npos || la.find("_log1p_abs") != std::string::npos ||
                             la.find("_x_") != std::string::npos || la.find("_div_") != std::string::npos;
    const bool engineeredB = lb.find("_pow") != std::string::npos || lb.find("_log1p_abs") != std::string::npos ||
                             lb.find("_x_") != std::string::npos || lb.find("_div_") != std::string::npos;
    if (!engineeredA && !engineeredB) return false;

    const std::vector<std::string> rootsA = engineeredRoots(la);
    const std::vector<std::string> rootsB = engineeredRoots(lb);
    for (const auto& ra : rootsA) {
        if (std::find(rootsB.begin(), rootsB.end(), ra) != rootsB.end()) {
            return true;
        }
    }
    return false;
}

std::optional<double> pearsonAligned(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 4) return std::nullopt;
    const ColumnStats as = Statistics::calculateStats(a);
    const ColumnStats bs = Statistics::calculateStats(b);
    return MathUtils::calculatePearson(a, b, as, bs);
}

double safeAbsCorr(const std::vector<double>& a, const std::vector<double>& b) {
    return std::abs(pearsonAligned(a, b).value_or(0.0));
}

std::vector<double> regressResidual(const std::vector<double>& y,
                                    const std::vector<std::vector<double>>& controls) {
    if (y.empty()) return {};
    if (controls.empty()) return y;

    MathUtils::Matrix X(y.size(), controls.size() + 1);
    MathUtils::Matrix Y(y.size(), 1);
    for (size_t r = 0; r < y.size(); ++r) {
        X.at(r, 0) = 1.0;
        for (size_t c = 0; c < controls.size(); ++c) X.at(r, c + 1) = controls[c][r];
        Y.at(r, 0) = y[r];
    }

    const auto beta = MathUtils::multipleLinearRegression(X, Y);
    if (beta.size() != controls.size() + 1) return y;

    std::vector<double> residual(y.size(), 0.0);
    for (size_t r = 0; r < y.size(); ++r) {
        double pred = beta[0];
        for (size_t c = 0; c < controls.size(); ++c) pred += beta[c + 1] * controls[c][r];
        residual[r] = y[r] - pred;
    }
    return residual;
}

double medianDistance(const std::vector<double>& v) {
    if (v.size() < 4) return 1.0;
    std::vector<double> dist;
    dist.reserve((v.size() * (v.size() - 1)) / 2);
    for (size_t i = 0; i < v.size(); ++i) {
        for (size_t j = i + 1; j < v.size(); ++j) {
            dist.push_back(std::abs(v[i] - v[j]));
        }
    }
    if (dist.empty()) return 1.0;
    const size_t mid = dist.size() / 2;
    std::nth_element(dist.begin(), dist.begin() + mid, dist.end());
    return std::max(1e-6, dist[mid]);
}

double hsicRbf(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    if (n != y.size() || n < 8) return 0.0;

    const double sx = medianDistance(x);
    const double sy = medianDistance(y);
    const double gx = 1.0 / (2.0 * sx * sx);
    const double gy = 1.0 / (2.0 * sy * sy);

    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::vector<double> rowK(n, 0.0), rowL(n, 0.0);
    double meanK = 0.0;
    double meanL = 0.0;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double dx = x[i] - x[j];
            const double dy = y[i] - y[j];
            K[i][j] = std::exp(-gx * dx * dx);
            L[i][j] = std::exp(-gy * dy * dy);
            rowK[i] += K[i][j];
            rowL[i] += L[i][j];
            meanK += K[i][j];
            meanL += L[i][j];
        }
    }

    meanK /= static_cast<double>(n * n);
    meanL /= static_cast<double>(n * n);
    for (size_t i = 0; i < n; ++i) {
        rowK[i] /= static_cast<double>(n);
        rowL[i] /= static_cast<double>(n);
    }

    double stat = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double Kc = K[i][j] - rowK[i] - rowK[j] + meanK;
            const double Lc = L[i][j] - rowL[i] - rowL[j] + meanL;
            stat += Kc * Lc;
        }
    }
    return stat / static_cast<double>(n * n);
}

CiTestResult conditionalIndependenceTest(const std::vector<std::vector<double>>& rows,
                                         size_t xIdx,
                                         size_t yIdx,
                                         const std::vector<size_t>& condSet,
                                         double alpha,
                                         bool enableKernelFallback,
                                         std::mt19937& rng) {
    CiTestResult out;
    if (rows.size() < 12) {
        out.independent = true;
        return out;
    }

    const size_t n = rows.size();
    std::vector<double> x(n), y(n);
    std::vector<std::vector<double>> controls;
    controls.reserve(condSet.size());
    for (size_t c : condSet) controls.push_back(std::vector<double>(n, 0.0));

    for (size_t r = 0; r < n; ++r) {
        x[r] = rows[r][xIdx];
        y[r] = rows[r][yIdx];
        for (size_t ci = 0; ci < condSet.size(); ++ci) controls[ci][r] = rows[r][condSet[ci]];
    }

    // Step 1: regress out conditioning variables and test residual association.
    const std::vector<double> rx = regressResidual(x, controls);
    const std::vector<double> ry = regressResidual(y, controls);
    const double r = std::clamp(pearsonAligned(rx, ry).value_or(0.0), -0.999999, 0.999999);
    out.effect = std::abs(r);

    // Step 2: Fisher z-transform approximates p-value for partial correlation.
    if (n > condSet.size() + 3) {
        const double fisher = 0.5 * std::log((1.0 + r) / (1.0 - r));
        const double z = std::abs(fisher) * std::sqrt(static_cast<double>(n) - static_cast<double>(condSet.size()) - 3.0);
        out.pValue = std::erfc(z / std::sqrt(2.0));
        out.independent = out.pValue > alpha;
    }

    // Step 3: if close to decision boundary, use lightweight kernel fallback.
    if (!enableKernelFallback) return out;
    if (out.pValue < alpha * 0.5 || out.pValue > alpha * 2.0) return out;

    const size_t kernelMaxN = 220;
    std::vector<double> kx;
    std::vector<double> ky;
    kx.reserve(std::min(n, kernelMaxN));
    ky.reserve(std::min(n, kernelMaxN));

    if (n <= kernelMaxN) {
        kx = rx;
        ky = ry;
    } else {
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (size_t i = 0; i < kernelMaxN; ++i) {
            const size_t idx = dist(rng);
            kx.push_back(rx[idx]);
            ky.push_back(ry[idx]);
        }
    }

    const double observed = hsicRbf(kx, ky);
    if (!std::isfinite(observed)) return out;

    size_t ge = 1;
    const size_t perms = 30;
    std::vector<double> perm = ky;
    for (size_t p = 0; p < perms; ++p) {
        std::shuffle(perm.begin(), perm.end(), rng);
        const double s = hsicRbf(kx, perm);
        if (s >= observed) ++ge;
    }

    out.usedKernel = true;
    out.pValue = static_cast<double>(ge) / static_cast<double>(perms + 1);
    out.independent = out.pValue > alpha;
    return out;
}

bool hasAnyEdge(const std::vector<std::vector<bool>>& directed,
                const std::vector<std::vector<bool>>& undirected,
                size_t a,
                size_t b) {
    return undirected[a][b] || directed[a][b] || directed[b][a];
}

bool hasDirectedPath(const std::vector<std::vector<bool>>& directed,
                     size_t src,
                     size_t dst) {
    if (src == dst) return true;
    std::vector<bool> seen(directed.size(), false);
    std::vector<size_t> stack{src};
    seen[src] = true;
    while (!stack.empty()) {
        const size_t u = stack.back();
        stack.pop_back();
        for (size_t v = 0; v < directed.size(); ++v) {
            if (!directed[u][v] || seen[v]) continue;
            if (v == dst) return true;
            seen[v] = true;
            stack.push_back(v);
        }
    }
    return false;
}

bool orientEdge(std::vector<std::vector<bool>>& directed,
                std::vector<std::vector<bool>>& undirected,
                const std::vector<std::vector<bool>>& forbidden,
                size_t from,
                size_t to) {
    if (from == to) return false;
    if (forbidden[from][to]) return false;
    if (directed[to][from]) return false;
    if (hasDirectedPath(directed, to, from)) return false;

    const bool changed = !directed[from][to] || undirected[from][to] || undirected[to][from];
    directed[from][to] = true;
    directed[to][from] = false;
    undirected[from][to] = false;
    undirected[to][from] = false;
    return changed;
}

void combinationsRec(const std::vector<size_t>& items,
                     size_t k,
                     size_t start,
                     std::vector<size_t>& current,
                     std::vector<std::vector<size_t>>& out) {
    if (current.size() == k) {
        out.push_back(current);
        return;
    }
    for (size_t i = start; i < items.size(); ++i) {
        current.push_back(items[i]);
        combinationsRec(items, k, i + 1, current, out);
        current.pop_back();
    }
}

std::vector<std::vector<size_t>> combinations(const std::vector<size_t>& items, size_t k) {
    std::vector<std::vector<size_t>> out;
    if (k == 0) {
        out.push_back({});
        return out;
    }
    if (items.size() < k) return out;
    std::vector<size_t> cur;
    cur.reserve(k);
    combinationsRec(items, k, 0, cur, out);
    return out;
}

double nonGaussianityScore(const std::vector<double>& v) {
    if (v.size() < 12) return 0.0;
    const ColumnStats s = Statistics::calculateStats(v);
    const double skew = std::abs(s.skewness);
    const double kurt = std::abs(s.kurtosis - 3.0);
    return skew + 0.5 * kurt;
}

std::vector<size_t> directLingamOrder(const std::vector<std::vector<double>>& rows) {
    const size_t p = rows.empty() ? 0 : rows.front().size();
    std::vector<size_t> remaining(p);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<size_t> order;
    order.reserve(p);

    while (!remaining.empty()) {
        size_t bestNode = remaining.front();
        double bestScore = std::numeric_limits<double>::infinity();
        for (size_t cand : remaining) {
            double dep = 0.0;
            size_t cnt = 0;
            std::vector<double> x(rows.size(), 0.0);
            for (size_t r = 0; r < rows.size(); ++r) x[r] = rows[r][cand];

            for (size_t oth : remaining) {
                if (oth == cand) continue;
                std::vector<double> y(rows.size(), 0.0);
                for (size_t r = 0; r < rows.size(); ++r) y[r] = rows[r][oth];
                const std::vector<double> resid = regressResidual(y, {x});
                dep += safeAbsCorr(x, resid);
                ++cnt;
            }
            const double score = (cnt == 0) ? 0.0 : (dep / static_cast<double>(cnt));
            if (score < bestScore) {
                bestScore = score;
                bestNode = cand;
            }
        }

        order.push_back(bestNode);
        remaining.erase(std::remove(remaining.begin(), remaining.end(), bestNode), remaining.end());
    }
    return order;
}

TemporalSeries detectTemporalSeries(const TypedDataset& data) {
    TemporalSeries out;

    for (size_t idx : data.datetimeColumnIndices()) {
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<std::vector<int64_t>>(col.values);
        if (vals.size() < 16) continue;
        std::unordered_set<int64_t> uniq;
        uniq.reserve(vals.size());
        out.axis.assign(vals.size(), 0.0);
        for (size_t i = 0; i < vals.size(); ++i) {
            out.axis[i] = static_cast<double>(vals[i]);
            if (i < col.missing.size() && !col.missing[i]) uniq.insert(vals[i]);
        }
        if (uniq.size() >= 8) {
            out.axisName = col.name;
            out.valid = true;
            return out;
        }
    }

    for (size_t idx : data.numericColumnIndices()) {
        const auto& col = data.columns()[idx];
        if (!hasToken(col.name, {"time", "date", "index", "idx", "step", "epoch", "year", "month", "day", "week"}) &&
            CommonUtils::toLower(col.name) != "t") {
            continue;
        }
        const auto& vals = std::get<std::vector<double>>(col.values);
        if (vals.size() < 16) continue;
        out.axis = vals;
        out.axisName = col.name;
        out.valid = true;
        return out;
    }

    return out;
}

CausalDataView buildCompleteCaseView(const TypedDataset& data,
                                     const std::vector<size_t>& candidateFeatures,
                                     size_t targetIdx,
                                     size_t maxFeatures) {
    CausalDataView out;

    std::vector<size_t> nodes;
    nodes.reserve(maxFeatures + 1);
    nodes.push_back(targetIdx);
    for (size_t idx : candidateFeatures) {
        if (idx == targetIdx) continue;
        if (idx >= data.columns().size()) continue;
        if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
        nodes.push_back(idx);
        if (nodes.size() >= maxFeatures + 1) break;
    }

    if (nodes.size() < 3) return out;

    const size_t nRows = data.rowCount();
    out.nodeColumnIdx = nodes;
    out.rows.reserve(nRows);
    for (size_t r = 0; r < nRows; ++r) {
        std::vector<double> row;
        row.reserve(nodes.size());
        bool ok = true;
        for (size_t idx : nodes) {
            const auto& col = data.columns()[idx];
            const auto& vals = std::get<std::vector<double>>(col.values);
            if (r >= vals.size()) { ok = false; break; }
            if (r < col.missing.size() && col.missing[r]) { ok = false; break; }
            if (!std::isfinite(vals[r])) { ok = false; break; }
            row.push_back(vals[r]);
        }
        if (ok) out.rows.push_back(std::move(row));
    }

    if (out.rows.size() < 30) {
        out.nodeColumnIdx.clear();
        out.rows.clear();
        return out;
    }

    constexpr size_t kMaxCausalRows = 2500;
    if (out.rows.size() > kMaxCausalRows) {
        std::vector<std::vector<double>> reduced;
        reduced.reserve(kMaxCausalRows);
        const double step = static_cast<double>(out.rows.size() - 1) / static_cast<double>(kMaxCausalRows - 1);
        size_t prev = static_cast<size_t>(-1);
        for (size_t i = 0; i < kMaxCausalRows; ++i) {
            size_t pick = static_cast<size_t>(std::llround(step * static_cast<double>(i)));
            pick = std::min(pick, out.rows.size() - 1);
            if (pick == prev && pick + 1 < out.rows.size()) ++pick;
            reduced.push_back(std::move(out.rows[pick]));
            prev = pick;
        }
        out.rows = std::move(reduced);
    }
    return out;
}

std::vector<std::vector<bool>> inferForbiddenMatrix(const TypedDataset& data,
                                                    const std::vector<size_t>& nodeCols) {
    const size_t p = nodeCols.size();
    std::vector<std::vector<bool>> forbidden(p, std::vector<bool>(p, false));

    for (size_t i = 0; i < p; ++i) {
        const std::string nameI = data.columns()[nodeCols[i]].name;
        const bool lagI = hasToken(nameI, {"lag", "prev", "past", "t-1", "l1"});
        const bool idI = hasToken(nameI, {"_id", " id", "uuid", "identifier", "index", "idx", "code"});
        const bool timeI = hasToken(nameI, {"time", "date", "epoch", "year", "month", "day", "week"});

        for (size_t j = 0; j < p; ++j) {
            if (i == j) continue;
            const std::string nameJ = data.columns()[nodeCols[j]].name;
            const bool lagJ = hasToken(nameJ, {"lag", "prev", "past", "t-1", "l1"});
            const bool timeJ = hasToken(nameJ, {"time", "date", "epoch", "year", "month", "day", "week"});

            if (idI) forbidden[j][i] = true;
            if (idI) forbidden[i][j] = true;
            if (lagI && !lagJ) forbidden[j][i] = true;
            if (timeI && !timeJ) forbidden[j][i] = true;
        }
    }

    return forbidden;
}

DiscoveryCoreResult discoverCore(const std::vector<std::vector<double>>& rows,
                                 const std::vector<std::vector<bool>>& forbidden,
                                 const CausalDiscoveryOptions& options,
                                 std::mt19937& rng) {
    const size_t p = rows.empty() ? 0 : rows.front().size();
    DiscoveryCoreResult out;
    out.directed.assign(p, std::vector<bool>(p, false));
    out.undirected.assign(p, std::vector<bool>(p, false));
    out.sepsets.assign(p, std::vector<std::unordered_set<size_t>>(p));
    std::unordered_map<std::string, CiTestResult> ciCache;
    ciCache.reserve(p * p * 4);

    if (p < 3 || rows.size() < 30) return out;

    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i + 1; j < p; ++j) {
            if (forbidden[i][j] && forbidden[j][i]) continue;
            out.undirected[i][j] = true;
            out.undirected[j][i] = true;
        }
    }

    for (size_t level = 0; level <= options.maxConditionSet; ++level) {
        bool removedAny = false;
        for (size_t i = 0; i < p; ++i) {
            for (size_t j = i + 1; j < p; ++j) {
                if (!out.undirected[i][j]) continue;

                std::vector<size_t> adjI;
                for (size_t k = 0; k < p; ++k) {
                    if (k == i || k == j) continue;
                    if (out.undirected[i][k] || out.directed[i][k] || out.directed[k][i]) adjI.push_back(k);
                }
                if (adjI.size() < level) continue;

                const auto cands = combinations(adjI, level);
                bool separated = false;
                for (const auto& cond : cands) {
                    std::string cacheKey = std::to_string(std::min(i, j)) + "|" + std::to_string(std::max(i, j)) + "|";
                    for (size_t c : cond) {
                        cacheKey += std::to_string(c);
                        cacheKey.push_back(',');
                    }

                    CiTestResult ci;
                    const auto cacheIt = ciCache.find(cacheKey);
                    if (cacheIt != ciCache.end()) {
                        ci = cacheIt->second;
                    } else {
                        ci = conditionalIndependenceTest(rows, i, j, cond, options.alpha,
                                                         options.enableKernelCiFallback,
                                                         rng);
                        ciCache.emplace(std::move(cacheKey), ci);
                    }
                    if (!ci.independent) continue;
                    out.undirected[i][j] = false;
                    out.undirected[j][i] = false;
                    out.sepsets[i][j].insert(cond.begin(), cond.end());
                    out.sepsets[j][i].insert(cond.begin(), cond.end());
                    separated = true;
                    removedAny = true;
                    break;
                }
                if (separated) continue;
            }
        }
        if (!removedAny) break;
    }

    for (size_t k = 0; k < p; ++k) {
        std::vector<size_t> nbr;
        for (size_t v = 0; v < p; ++v) {
            if (v == k) continue;
            if (out.undirected[k][v]) nbr.push_back(v);
        }

        for (size_t a = 0; a < nbr.size(); ++a) {
            for (size_t b = a + 1; b < nbr.size(); ++b) {
                const size_t i = nbr[a];
                const size_t j = nbr[b];
                if (hasAnyEdge(out.directed, out.undirected, i, j)) continue;
                if (out.sepsets[i][j].find(k) != out.sepsets[i][j].end()) continue;
                orientEdge(out.directed, out.undirected, forbidden, i, k);
                orientEdge(out.directed, out.undirected, forbidden, j, k);
            }
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < p; ++j) {
                if (!out.directed[i][j]) continue;
                for (size_t k = 0; k < p; ++k) {
                    if (!out.undirected[j][k]) continue;
                    if (hasAnyEdge(out.directed, out.undirected, i, k)) continue;
                    changed |= orientEdge(out.directed, out.undirected, forbidden, j, k);
                }
            }
        }

        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < p; ++j) {
                if (!out.undirected[i][j]) continue;
                for (size_t k = 0; k < p; ++k) {
                    if (out.directed[i][k] && out.directed[k][j]) {
                        changed |= orientEdge(out.directed, out.undirected, forbidden, i, j);
                    }
                }
            }
        }

        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < p; ++j) {
                if (!out.undirected[i][j]) continue;
                for (size_t k = 0; k < p; ++k) {
                    if (!out.undirected[i][k] || !out.directed[k][j]) continue;
                    for (size_t l = k + 1; l < p; ++l) {
                        if (!out.undirected[i][l] || !out.directed[l][j]) continue;
                        if (hasAnyEdge(out.directed, out.undirected, k, l)) continue;
                        changed |= orientEdge(out.directed, out.undirected, forbidden, i, j);
                    }
                }
            }
        }
    }

    if (options.enableLiNGAM) {
        double ng = 0.0;
        for (size_t c = 0; c < p; ++c) {
            std::vector<double> col(rows.size(), 0.0);
            for (size_t r = 0; r < rows.size(); ++r) col[r] = rows[r][c];
            ng += nonGaussianityScore(col);
        }
        ng /= static_cast<double>(p);
        if (ng >= 0.80) {
            const std::vector<size_t> order = directLingamOrder(rows);
            std::vector<size_t> pos(p, 0);
            for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = i;
            for (size_t i = 0; i < p; ++i) {
                for (size_t j = i + 1; j < p; ++j) {
                    if (!out.undirected[i][j]) continue;
                    if (pos[i] < pos[j]) orientEdge(out.directed, out.undirected, forbidden, i, j);
                    else orientEdge(out.directed, out.undirected, forbidden, j, i);
                }
            }
            out.usedLiNGAM = true;
        }
    }

    return out;
}

std::vector<size_t> parentSetOf(const std::vector<std::vector<bool>>& directed, size_t node) {
    std::vector<size_t> parents;
    parents.reserve(directed.size());
    for (size_t p = 0; p < directed.size(); ++p) {
        if (directed[p][node]) parents.push_back(p);
    }
    return parents;
}

double bicForNodeGivenParents(const std::vector<std::vector<double>>& rows,
                              size_t nodeIdx,
                              const std::vector<size_t>& parentIdx) {
    if (rows.size() < 8) return std::numeric_limits<double>::infinity();

    MathUtils::Matrix X(rows.size(), parentIdx.size() + 1);
    MathUtils::Matrix Y(rows.size(), 1);
    for (size_t r = 0; r < rows.size(); ++r) {
        X.at(r, 0) = 1.0;
        for (size_t c = 0; c < parentIdx.size(); ++c) {
            X.at(r, c + 1) = rows[r][parentIdx[c]];
        }
        Y.at(r, 0) = rows[r][nodeIdx];
    }

    const auto beta = MathUtils::multipleLinearRegression(X, Y);
    if (beta.size() != parentIdx.size() + 1) return std::numeric_limits<double>::infinity();

    double rss = 0.0;
    for (size_t r = 0; r < rows.size(); ++r) {
        double pred = beta[0];
        for (size_t c = 0; c < parentIdx.size(); ++c) {
            pred += beta[c + 1] * rows[r][parentIdx[c]];
        }
        const double e = rows[r][nodeIdx] - pred;
        rss += e * e;
    }

    const double n = static_cast<double>(rows.size());
    const double k = static_cast<double>(parentIdx.size() + 1);
    const double sigma2 = std::max(kBicSigma2Floor, rss / n);
    return n * std::log(sigma2) + k * std::log(n);
}

double totalDagBic(const std::vector<std::vector<double>>& rows,
                   const std::vector<std::vector<bool>>& directed) {
    const size_t p = directed.size();
    double total = 0.0;
    for (size_t node = 0; node < p; ++node) {
        const std::vector<size_t> parents = parentSetOf(directed, node);
        const double bic = bicForNodeGivenParents(rows, node, parents);
        if (!std::isfinite(bic)) return std::numeric_limits<double>::infinity();
        total += bic;
    }
    return total;
}

void applyGesOrientation(DiscoveryCoreResult& core,
                         const std::vector<std::vector<double>>& rows,
                         const std::vector<std::vector<bool>>& forbidden) {
    const size_t p = core.undirected.size();
    if (p < 2 || rows.size() < 16) return;

    auto currentScore = totalDagBic(rows, core.directed);
    if (!std::isfinite(currentScore)) {
        currentScore = 0.0;
    }

    // Forward phase: greedily orient undirected edges by best BIC improvement.
    bool improved = true;
    while (improved) {
        improved = false;
        double bestDelta = 0.0;
        std::vector<std::vector<bool>> bestDirected = core.directed;
        std::vector<std::vector<bool>> bestUndirected = core.undirected;

        for (size_t i = 0; i < p; ++i) {
            for (size_t j = i + 1; j < p; ++j) {
                if (!core.undirected[i][j]) continue;
                if (forbidden[i][j] && forbidden[j][i]) continue;

                for (int dir = 0; dir < 2; ++dir) {
                    const size_t from = (dir == 0) ? i : j;
                    const size_t to = (dir == 0) ? j : i;

                    auto directedCand = core.directed;
                    auto undirectedCand = core.undirected;
                    if (!orientEdge(directedCand, undirectedCand, forbidden, from, to)) continue;

                    const double candScore = totalDagBic(rows, directedCand);
                    if (!std::isfinite(candScore)) continue;
                    const double delta = currentScore - candScore;
                    if (delta > bestDelta + kBicTieThreshold) {
                        bestDelta = delta;
                        bestDirected = std::move(directedCand);
                        bestUndirected = std::move(undirectedCand);
                    }
                }
            }
        }

        if (bestDelta > kBicTieThreshold) {
            core.directed = std::move(bestDirected);
            core.undirected = std::move(bestUndirected);
            currentScore -= bestDelta;
            improved = true;
        }
    }

    // Backward phase: attempt beneficial edge reversals on already-directed edges.
    improved = true;
    while (improved) {
        improved = false;
        double bestDelta = 0.0;
        std::vector<std::vector<bool>> bestDirected = core.directed;

        for (size_t from = 0; from < p; ++from) {
            for (size_t to = 0; to < p; ++to) {
                if (from == to || !core.directed[from][to]) continue;
                if (forbidden[to][from]) continue;

                auto directedCand = core.directed;
                auto undirectedCand = core.undirected;
                directedCand[from][to] = false;
                directedCand[to][from] = false;
                if (!orientEdge(directedCand, undirectedCand, forbidden, to, from)) continue;

                const double candScore = totalDagBic(rows, directedCand);
                if (!std::isfinite(candScore)) continue;
                const double delta = currentScore - candScore;
                if (delta > bestDelta + kBicTieThreshold) {
                    bestDelta = delta;
                    bestDirected = std::move(directedCand);
                }
            }
        }

        if (bestDelta > kBicTieThreshold) {
            core.directed = std::move(bestDirected);
            currentScore -= bestDelta;
            improved = true;
        }
    }
}

bool hasLikelyLatentConfounding(const std::vector<std::vector<double>>& rows,
                                size_t fromIdx,
                                size_t toIdx) {
    if (rows.size() < 20) return false;
    std::vector<double> x(rows.size(), 0.0);
    std::vector<double> y(rows.size(), 0.0);
    for (size_t r = 0; r < rows.size(); ++r) {
        x[r] = rows[r][fromIdx];
        y[r] = rows[r][toIdx];
    }

    const auto beta = MathUtils::simpleLinearRegression(x,
                                                         y,
                                                         Statistics::calculateStats(x),
                                                         Statistics::calculateStats(y),
                                                         pearsonAligned(x, y).value_or(0.0));
    std::vector<double> resid(y.size(), 0.0);
    for (size_t i = 0; i < y.size(); ++i) {
        resid[i] = y[i] - (beta.first * x[i] + beta.second);
    }
    const double confoundSignal = std::abs(MathUtils::calculatePearson(x,
                                                                       resid,
                                                                       Statistics::calculateStats(x),
                                                                       Statistics::calculateStats(resid)).value_or(0.0));
    return confoundSignal >= 0.22;
}

std::optional<double> grangerPValue(const TypedDataset& data,
                                    size_t fromCol,
                                    size_t toCol,
                                    const TemporalSeries& ts) {
    if (!ts.valid) return std::nullopt;
    if (fromCol >= data.columns().size() || toCol >= data.columns().size()) return std::nullopt;
    if (data.columns()[fromCol].type != ColumnType::NUMERIC || data.columns()[toCol].type != ColumnType::NUMERIC) return std::nullopt;

    const auto& x = std::get<std::vector<double>>(data.columns()[fromCol].values);
    const auto& y = std::get<std::vector<double>>(data.columns()[toCol].values);
    const size_t n = std::min({x.size(), y.size(), ts.axis.size()});

    std::vector<std::array<double, 3>> obs;
    obs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(ts.axis[i])) continue;
        if (i < data.columns()[fromCol].missing.size() && data.columns()[fromCol].missing[i]) continue;
        if (i < data.columns()[toCol].missing.size() && data.columns()[toCol].missing[i]) continue;
        obs.push_back({ts.axis[i], x[i], y[i]});
    }
    if (obs.size() < 24) return std::nullopt;

    std::sort(obs.begin(), obs.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });

    std::vector<std::array<double, 2>> rows;
    std::vector<double> yt;
    rows.reserve(obs.size() - 1);
    yt.reserve(obs.size() - 1);
    for (size_t t = 1; t < obs.size(); ++t) {
        rows.push_back({obs[t - 1][1], obs[t - 1][2]});
        yt.push_back(obs[t][2]);
    }
    if (rows.size() < 20) return std::nullopt;

    MathUtils::Matrix Xr(rows.size(), 2);
    MathUtils::Matrix Xf(rows.size(), 3);
    MathUtils::Matrix Y(rows.size(), 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        Xr.at(i, 0) = 1.0;
        Xr.at(i, 1) = rows[i][1];
        Xf.at(i, 0) = 1.0;
        Xf.at(i, 1) = rows[i][1];
        Xf.at(i, 2) = rows[i][0];
        Y.at(i, 0) = yt[i];
    }

    const auto br = MathUtils::multipleLinearRegression(Xr, Y);
    const auto bf = MathUtils::multipleLinearRegression(Xf, Y);
    if (br.size() != 2 || bf.size() != 3) return std::nullopt;

    double rssR = 0.0;
    double rssF = 0.0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const double pr = br[0] + br[1] * rows[i][1];
        const double pf = bf[0] + bf[1] * rows[i][1] + bf[2] * rows[i][0];
        const double er = yt[i] - pr;
        const double ef = yt[i] - pf;
        rssR += er * er;
        rssF += ef * ef;
    }

    const double df2 = static_cast<double>(rows.size()) - 3.0;
    if (df2 <= 1.0 || rssF <= 1e-12 || rssR <= rssF) return std::nullopt;
    const double f = ((rssR - rssF) / 1.0) / (rssF / df2);
    if (!std::isfinite(f) || f <= 0.0) return std::nullopt;
    return std::exp(-0.5 * f);
}

std::optional<double> icpStabilityScore(const TypedDataset& data,
                                        size_t fromCol,
                                        size_t toCol,
                                        const TemporalSeries& ts) {
    if (!ts.valid) return std::nullopt;
    const auto& x = std::get<std::vector<double>>(data.columns()[fromCol].values);
    const auto& y = std::get<std::vector<double>>(data.columns()[toCol].values);
    const size_t n = std::min({x.size(), y.size(), ts.axis.size()});

    std::vector<std::array<double, 3>> obs;
    obs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(ts.axis[i])) continue;
        if (i < data.columns()[fromCol].missing.size() && data.columns()[fromCol].missing[i]) continue;
        if (i < data.columns()[toCol].missing.size() && data.columns()[toCol].missing[i]) continue;
        obs.push_back({ts.axis[i], x[i], y[i]});
    }
    if (obs.size() < 30) return std::nullopt;

    std::sort(obs.begin(), obs.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });

    std::vector<double> globalX;
    std::vector<double> globalY;
    globalX.reserve(obs.size());
    globalY.reserve(obs.size());
    for (const auto& o : obs) {
        globalX.push_back(o[1]);
        globalY.push_back(o[2]);
    }
    const double globalSlope = MathUtils::simpleLinearRegression(globalX, globalY,
                                                                 Statistics::calculateStats(globalX),
                                                                 Statistics::calculateStats(globalY),
                                                                 pearsonAligned(globalX, globalY).value_or(0.0)).first;

    const size_t q1 = obs.size() / 3;
    const size_t q2 = (2 * obs.size()) / 3;
    const std::array<std::pair<size_t, size_t>, 3> spans{{{0, q1}, {q1, q2}, {q2, obs.size()}}};
    std::vector<double> slopes;
    for (const auto& span : spans) {
        if (span.second <= span.first + 6) continue;
        std::vector<double> ex;
        std::vector<double> ey;
        ex.reserve(span.second - span.first);
        ey.reserve(span.second - span.first);
        for (size_t i = span.first; i < span.second; ++i) {
            ex.push_back(obs[i][1]);
            ey.push_back(obs[i][2]);
        }
        const double r = pearsonAligned(ex, ey).value_or(0.0);
        const double slope = MathUtils::simpleLinearRegression(ex, ey,
                                                               Statistics::calculateStats(ex),
                                                               Statistics::calculateStats(ey),
                                                               r).first;
        if (std::isfinite(slope)) slopes.push_back(slope);
    }

    if (slopes.size() < 2 || !std::isfinite(globalSlope)) return std::nullopt;
    const auto [minIt, maxIt] = std::minmax_element(slopes.begin(), slopes.end());
    const double span = std::abs(*maxIt - *minIt);
    const double denom = std::max(0.1, std::abs(globalSlope));
    return span / denom;
}

} // namespace

CausalDiscoveryResult CausalDiscovery::discover(const TypedDataset& data,
                                                const std::vector<size_t>& candidateFeatures,
                                                size_t targetIdx,
                                                const CausalDiscoveryOptions& options) {
    CausalDiscoveryResult result;
    if (targetIdx >= data.columns().size()) return result;
    if (data.columns()[targetIdx].type != ColumnType::NUMERIC) return result;

    const CausalDataView view = buildCompleteCaseView(data, candidateFeatures, targetIdx, options.maxFeatures);
    if (view.nodeColumnIdx.empty() || view.rows.empty()) {
        result.notes.push_back("Causal discovery skipped: insufficient complete-case rows after alignment.");
        return result;
    }

    std::mt19937 rng(options.randomSeed);
    const auto forbidden = inferForbiddenMatrix(data, view.nodeColumnIdx);

    DiscoveryCoreResult core = discoverCore(view.rows, forbidden, options, rng);
    if (options.enableGES) {
        applyGesOrientation(core, view.rows, forbidden);
    }
    result.usedLiNGAM = core.usedLiNGAM;

    const size_t effectiveBootstrapSamples = [&]() {
        if (options.bootstrapSamples == 0) return static_cast<size_t>(0);
        if (view.rows.size() >= 8000) return std::min<size_t>(options.bootstrapSamples, 24);
        if (view.rows.size() >= 4000) return std::min<size_t>(options.bootstrapSamples, 40);
        if (view.rows.size() >= 2000) return std::min<size_t>(options.bootstrapSamples, 64);
        return options.bootstrapSamples;
    }();

    std::map<std::pair<size_t, size_t>, size_t> bootstrapCounts;
    if (effectiveBootstrapSamples > 0) {
        const size_t p = core.directed.size();
        std::vector<size_t> edgeCounts(p * p, 0);

        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t b = 0; b < effectiveBootstrapSamples; ++b) {
            std::mt19937 sampleRng(static_cast<uint32_t>(options.randomSeed + b * 104729u + 31u));
            std::uniform_int_distribution<size_t> pick(0, view.rows.size() - 1);
            std::vector<std::vector<double>> sample;
            sample.reserve(view.rows.size());
            for (size_t i = 0; i < view.rows.size(); ++i) sample.push_back(view.rows[pick(sampleRng)]);
            std::mt19937 localRng(static_cast<uint32_t>(options.randomSeed + b * 7919u + 17u));
            const DiscoveryCoreResult boot = discoverCore(sample, forbidden, options, localRng);

            std::vector<size_t> localEdgeCounts(p * p, 0);
            for (size_t i = 0; i < boot.directed.size(); ++i) {
                for (size_t j = 0; j < boot.directed.size(); ++j) {
                    if (i == j || !boot.directed[i][j]) continue;
                    localEdgeCounts[i * p + j] += 1;
                }
            }

            #ifdef USE_OPENMP
            #pragma omp critical
            #endif
            {
                for (size_t idx = 0; idx < edgeCounts.size(); ++idx) {
                    edgeCounts[idx] += localEdgeCounts[idx];
                }
            }
        }

        for (size_t i = 0; i < p; ++i) {
            for (size_t j = 0; j < p; ++j) {
                const size_t count = edgeCounts[i * p + j];
                if (count == 0) continue;
                bootstrapCounts[{i, j}] = count;
            }
        }
        result.bootstrapRuns = effectiveBootstrapSamples;
    }

    const TemporalSeries temporal = detectTemporalSeries(data);
    size_t suppressedByEngineeredFamilyGuard = 0;

    for (size_t i = 0; i < core.directed.size(); ++i) {
        for (size_t j = 0; j < core.directed.size(); ++j) {
            if (i == j || !core.directed[i][j]) continue;

            const std::string& fromName = data.columns()[view.nodeColumnIdx[i]].name;
            const std::string& toName = data.columns()[view.nodeColumnIdx[j]].name;
            if (sharesEngineeredFamilyRoot(fromName, toName)) {
                ++suppressedByEngineeredFamilyGuard;
                continue;
            }

            std::vector<double> xi(view.rows.size(), 0.0);
            std::vector<double> yj(view.rows.size(), 0.0);
            for (size_t r = 0; r < view.rows.size(); ++r) {
                xi[r] = view.rows[r][i];
                yj[r] = view.rows[r][j];
            }

            const double absR = safeAbsCorr(xi, yj);
            const double support = (result.bootstrapRuns == 0)
                ? 0.0
                : static_cast<double>(bootstrapCounts[{i, j}]) / static_cast<double>(result.bootstrapRuns);

            std::string validation = "no proxy validation";
            if (options.enableGrangerValidation || options.enableIcpValidation) {
                std::vector<std::string> checks;
                if (options.enableGrangerValidation) {
                    const auto p = grangerPValue(data, view.nodeColumnIdx[i], view.nodeColumnIdx[j], temporal);
                    if (p.has_value()) {
                        checks.push_back("Granger p=" + std::to_string(*p).substr(0, 6));
                    }
                }
                if (options.enableIcpValidation) {
                    const auto inv = icpStabilityScore(data, view.nodeColumnIdx[i], view.nodeColumnIdx[j], temporal);
                    if (inv.has_value()) {
                        checks.push_back("ICP drift=" + std::to_string(*inv).substr(0, 5));
                    }
                }
                if (!checks.empty()) {
                    validation.clear();
                    for (size_t ci = 0; ci < checks.size(); ++ci) {
                        if (ci) validation += ", ";
                        validation += checks[ci];
                    }
                }
            }

            const double confidence = std::clamp(0.20 + 0.35 * absR + 0.45 * support, 0.0, 0.99);
            const bool latentHint = options.enableFCI && hasLikelyLatentConfounding(view.rows, i, j);
            result.edges.push_back({
                view.nodeColumnIdx[i],
                view.nodeColumnIdx[j],
                confidence,
                support,
                "|r|=" + std::to_string(absR).substr(0, 5) + ", support=" + std::to_string(100.0 * support).substr(0, 5) + "%" +
                    std::string(latentHint ? ", latent_hint=possible" : ""),
                "PC/Meek" + std::string(options.enableGES ? " + GES" : "") +
                    std::string(core.usedLiNGAM ? " + LiNGAM" : "") +
                    std::string(options.enableFCI ? " + FCI checks" : "") +
                    std::string((options.markExperimentalHeuristics && (options.enableGES || options.enableFCI))
                                    ? " [experimental-lite]"
                                    : "") +
                    "; " + validation
            });
        }
    }

    std::sort(result.edges.begin(), result.edges.end(), [](const auto& a, const auto& b) {
        return a.confidence > b.confidence;
    });
    if (result.edges.size() > 12) result.edges.resize(12);

    if (temporal.valid) {
        result.notes.push_back("Temporal axis detected for proxy intervention checks: " + temporal.axisName + ".");
    }
    result.notes.push_back("Constraint-based PC discovery executed with max conditioning set " + std::to_string(options.maxConditionSet) + ".");
    if (effectiveBootstrapSamples < options.bootstrapSamples) {
        result.notes.push_back("Bootstrap samples auto-capped for runtime at " + std::to_string(effectiveBootstrapSamples) +
                               " (requested " + std::to_string(options.bootstrapSamples) + ").");
    }
    if (options.enableGES) {
        result.notes.push_back("Score-based GES orientation pass applied on unresolved undirected edges.");
    }
    if (options.enableFCI) {
        result.notes.push_back("FCI-style latent-confounding hints computed from residual dependence checks.");
    }
    if (options.markExperimentalHeuristics && (options.enableGES || options.enableFCI)) {
        result.notes.push_back("Experimental notice: additional approximations are enabled and require external validation for high-stakes causal decisions.");
    }
    if (core.usedLiNGAM) {
        result.notes.push_back("Non-Gaussian signal detected; applied DirectLiNGAM-style orientation for unresolved edges.");
    }
    if (suppressedByEngineeredFamilyGuard > 0) {
        result.notes.push_back("Causal semantic guard suppressed " + std::to_string(suppressedByEngineeredFamilyGuard) +
                               " base/engineered-family edge(s) to avoid identity-lineage causality artifacts.");
    }

    return result;
}
