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

namespace {

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

    const std::vector<double> rx = regressResidual(x, controls);
    const std::vector<double> ry = regressResidual(y, controls);
    const double r = std::clamp(pearsonAligned(rx, ry).value_or(0.0), -0.999999, 0.999999);
    out.effect = std::abs(r);

    if (n > condSet.size() + 3) {
        const double fisher = 0.5 * std::log((1.0 + r) / (1.0 - r));
        const double z = std::abs(fisher) * std::sqrt(static_cast<double>(n) - static_cast<double>(condSet.size()) - 3.0);
        out.pValue = std::erfc(z / std::sqrt(2.0));
        out.independent = out.pValue > alpha;
    }

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
                    const auto ci = conditionalIndependenceTest(rows, i, j, cond, options.alpha,
                                                                options.enableKernelCiFallback,
                                                                rng);
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
    result.usedLiNGAM = core.usedLiNGAM;

    std::map<std::pair<size_t, size_t>, size_t> bootstrapCounts;
    if (options.bootstrapSamples > 0) {
        std::uniform_int_distribution<size_t> pick(0, view.rows.size() - 1);
        for (size_t b = 0; b < options.bootstrapSamples; ++b) {
            std::vector<std::vector<double>> sample;
            sample.reserve(view.rows.size());
            for (size_t i = 0; i < view.rows.size(); ++i) sample.push_back(view.rows[pick(rng)]);
            std::mt19937 localRng(static_cast<uint32_t>(options.randomSeed + b * 7919u + 17u));
            const DiscoveryCoreResult boot = discoverCore(sample, forbidden, options, localRng);
            for (size_t i = 0; i < boot.directed.size(); ++i) {
                for (size_t j = 0; j < boot.directed.size(); ++j) {
                    if (i == j || !boot.directed[i][j]) continue;
                    bootstrapCounts[{i, j}] += 1;
                }
            }
            ++result.bootstrapRuns;
        }
    }

    const TemporalSeries temporal = detectTemporalSeries(data);

    for (size_t i = 0; i < core.directed.size(); ++i) {
        for (size_t j = 0; j < core.directed.size(); ++j) {
            if (i == j || !core.directed[i][j]) continue;

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
            result.edges.push_back({
                view.nodeColumnIdx[i],
                view.nodeColumnIdx[j],
                confidence,
                support,
                "|r|=" + std::to_string(absR).substr(0, 5) + ", support=" + std::to_string(100.0 * support).substr(0, 5) + "%",
                "PC/Meek orientation" + std::string(core.usedLiNGAM ? " + LiNGAM order" : "") + "; " + validation
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
    if (core.usedLiNGAM) {
        result.notes.push_back("Non-Gaussian signal detected; applied DirectLiNGAM-style orientation for unresolved edges.");
    }

    return result;
}
