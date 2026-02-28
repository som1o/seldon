#include "MathUtils.h"
#include "CommonUtils.h"
#include "Statistics.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <limits>
#include <unordered_map>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
constexpr double kNearOneCorrelationThreshold = 0.9999999;
constexpr double kVeryLargeTStatisticCutoff = 1e10;

struct RuntimeConfig {
    double significanceAlpha = 0.05;
    double numericEpsilon = 1e-12;
    size_t betaFallbackIntervalsStart = 4096;
    size_t betaFallbackIntervalsMax = 65536;
    double betaFallbackTolerance = 1e-8;
};

RuntimeConfig& runtimeConfig() {
    thread_local RuntimeConfig cfg;
    return cfg;
}

double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

double midpointIntegrateBetaRegularized(double a, double b, double x, size_t intervals) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    const double logBeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double h = x / static_cast<double>(intervals);
    double sum = 0.0;

    for (size_t i = 0; i < intervals; ++i) {
        double t = (static_cast<double>(i) + 0.5) * h;
        t = std::clamp(t, 1e-15, 1.0 - 1e-15);
        double logPdf = (a - 1.0) * std::log(t) + (b - 1.0) * std::log(1.0 - t) - logBeta;
        sum += std::exp(logPdf);
    }

    return clamp01(sum * h);
}

double midpointIntegrateBetaAdaptive(double a, double b, double x) {
    const RuntimeConfig& cfg = runtimeConfig();
    size_t startIntervals = std::max<size_t>(256, cfg.betaFallbackIntervalsStart);
    size_t maxIntervals = std::max(startIntervals, cfg.betaFallbackIntervalsMax);

    double prev = midpointIntegrateBetaRegularized(a, b, x, startIntervals);
    for (size_t intervals = startIntervals * 2; intervals <= maxIntervals; intervals *= 2) {
        double cur = midpointIntegrateBetaRegularized(a, b, x, intervals);
        // Midpoint has O(h^2) error; this scaled delta approximates residual truncation error.
        const double richardsonErr = std::abs(cur - prev) / 3.0;
        if (richardsonErr < cfg.betaFallbackTolerance) return cur;
        prev = cur;
    }
    return prev;
}

std::pair<double, bool> betaContinuedFraction(double a, double b, double x) {
    const int maxIter = std::clamp<int>(400 + static_cast<int>(std::ceil((a + b) * 0.75)), 400, 2000);
    constexpr double eps = 3e-14;
    constexpr double fpmin = 1e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;

    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= maxIter; ++m) {
        const int m2 = 2 * m;

        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (std::abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;

        if (std::abs(del - 1.0) <= eps) {
            return {h, true};
        }
    }
    return {h, false};
}

bool solveUpperTriangular(const MathUtils::Matrix& R,
                          size_t n,
                          const std::vector<double>& b,
                          std::vector<double>& x) {
    if (b.size() != n) return false;
    x.assign(n, 0.0);
    for (size_t i = n; i-- > 0;) {
        double rhs = b[i];
        for (size_t j = i + 1; j < n; ++j) {
            rhs -= R.data[i][j] * x[j];
        }
        const double diag = R.data[i][i];
        if (std::abs(diag) <= runtimeConfig().numericEpsilon) return false;
        x[i] = rhs / diag;
    }
    return true;
}

bool solveLowerFromUpperTranspose(const MathUtils::Matrix& R,
                                  size_t n,
                                  const std::vector<double>& b,
                                  std::vector<double>& x) {
    if (b.size() != n) return false;
    x.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double rhs = b[i];
        for (size_t j = 0; j < i; ++j) {
            rhs -= R.data[j][i] * x[j];
        }
        const double diag = R.data[i][i];
        if (std::abs(diag) <= runtimeConfig().numericEpsilon) return false;
        x[i] = rhs / diag;
    }
    return true;
}

std::vector<double> averageRanks(const std::vector<double>& values) {
    std::vector<size_t> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (values[a] == values[b]) return a < b;
        return values[a] < values[b];
    });

    std::vector<double> ranks(values.size(), 0.0);
    size_t i = 0;
    while (i < idx.size()) {
        size_t j = i + 1;
        while (j < idx.size() && values[idx[j]] == values[idx[i]]) ++j;
        const double rank = (static_cast<double>(i + 1) + static_cast<double>(j)) * 0.5;
        for (size_t k = i; k < j; ++k) ranks[idx[k]] = rank;
        i = j;
    }
    return ranks;
}
}

// Regularized incomplete beta function I_x(a, b) using continued fractions (Lentz's method)
static double betainc(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return NAN;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    if (a <= 0.0 || b <= 0.0 || !std::isfinite(a) || !std::isfinite(b)) return NAN;

    const double lnBeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double bt = std::exp(a * std::log(x) + b * std::log(1.0 - x) - lnBeta);
    const bool useDirect = (x < (a + 1.0) / (a + b + 2.0));

    if (useDirect) {
        auto [cf, ok] = betaContinuedFraction(a, b, x);
        if (ok && std::isfinite(cf)) {
            return clamp01(bt * cf / a);
        }
    } else {
        auto [cf, ok] = betaContinuedFraction(b, a, 1.0 - x);
        if (ok && std::isfinite(cf)) {
            return clamp01(1.0 - bt * cf / b);
        }
    }

    return midpointIntegrateBetaAdaptive(a, b, x);
}

// Two-tailed p-value from t-statistic using analytical beta distribution
static double pvalue_from_t(double t, size_t df) {
    if (df == 0) return 1.0;
    double nu = static_cast<double>(df);
    double t_abs = std::abs(t);
    
    // Handle very large t (numerically safe fallback)
    if (!std::isfinite(t_abs) || t_abs > kVeryLargeTStatisticCutoff) return 0.0;
    
    double x = nu / (nu + t_abs * t_abs);
    return betainc(nu / 2.0, 0.5, x); // two-tailed
}

double MathUtils::getPValueFromT(double t, size_t df) {
    return pvalue_from_t(t, df);
}

double MathUtils::getCriticalT(double alpha, size_t df) {
    if (df == 0 || !std::isfinite(alpha) || alpha <= 0.0 || alpha >= 1.0) return 0.0;

    const double target = alpha;
    double lo = 0.0;
    double hi = 2.0;
    while (getPValueFromT(hi, df) > target && hi < 1e6) {
        hi *= 2.0;
    }

    for (int it = 0; it < 70; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double p = getPValueFromT(mid, df);
        if (p > target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

void MathUtils::setSignificanceAlpha(double alpha) {
    if (alpha > 0.0 && alpha < 1.0) {
        runtimeConfig().significanceAlpha = alpha;
    }
}

double MathUtils::getSignificanceAlpha() noexcept {
    return runtimeConfig().significanceAlpha;
}

void MathUtils::setNumericTuning(double numericEpsilon,
                                 size_t betaIntervalsStart,
                                 size_t betaIntervalsMax,
                                 double betaTolerance) {
    RuntimeConfig& cfg = runtimeConfig();
    if (numericEpsilon > 0.0) cfg.numericEpsilon = numericEpsilon;
    if (betaIntervalsStart >= 256) cfg.betaFallbackIntervalsStart = betaIntervalsStart;
    if (betaIntervalsMax >= cfg.betaFallbackIntervalsStart) cfg.betaFallbackIntervalsMax = betaIntervalsMax;
    if (betaTolerance > 0.0) cfg.betaFallbackTolerance = betaTolerance;
}

Significance MathUtils::calculateSignificance(double r, size_t n) {
    Significance sig{0.0, 0.0, false};
    if (n <= 2) return sig;
    
    // Avoid division by zero if correlation is effectively perfect.
    if (std::abs(r) >= kNearOneCorrelationThreshold) {
        sig.p_value = 0.0;
        sig.t_stat = (r > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
        sig.is_significant = true;
        return sig;
    }

    size_t df = n - 2;
    sig.t_stat = r * std::sqrt(static_cast<double>(df) / (1.0 - r * r));
    sig.p_value = getPValueFromT(sig.t_stat, df);
    sig.is_significant = (sig.p_value < runtimeConfig().significanceAlpha);

    return sig;
}

std::optional<double> MathUtils::calculatePearson(const std::vector<double>& x, const std::vector<double>& y, 
                                           const ColumnStats& statsX, const ColumnStats& statsY) {
    if (x.size() != y.size() || x.size() < 2) return std::nullopt;
    if (statsX.stddev == 0 || statsY.stddev == 0) return std::nullopt;

    size_t n = x.size();
    
    // Modern C++ algorithm replacing raw loops
    double covarianceSum = std::inner_product(x.begin(), x.end(), y.begin(), 0.0,
        std::plus<>(),
        [&statsX, &statsY](double valX, double valY) {
            return (valX - statsX.mean) * (valY - statsY.mean);
        });
    
    // Pearson r = Covariance(X,Y) / (StdDevX * StdDevY)
    double covariance = covarianceSum / (n - 1);
    return covariance / (statsX.stddev * statsY.stddev);
}

std::optional<double> MathUtils::calculateSpearman(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 3) return std::nullopt;
    const std::vector<double> rx = averageRanks(x);
    const std::vector<double> ry = averageRanks(y);
    const ColumnStats sx = Statistics::calculateStats(rx);
    const ColumnStats sy = Statistics::calculateStats(ry);
    return calculatePearson(rx, ry, sx, sy);
}

std::optional<double> MathUtils::calculateKendallTau(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 3) return std::nullopt;
    long long concordant = 0;
    long long discordant = 0;
    long long tieX = 0;
    long long tieY = 0;

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = i + 1; j < x.size(); ++j) {
            const double dx = x[i] - x[j];
            const double dy = y[i] - y[j];
            const int sx = (dx > 0.0) - (dx < 0.0);
            const int sy = (dy > 0.0) - (dy < 0.0);
            if (sx == 0 && sy == 0) continue;
            if (sx == 0) {
                ++tieX;
            } else if (sy == 0) {
                ++tieY;
            } else if (sx == sy) {
                ++concordant;
            } else {
                ++discordant;
            }
        }
    }

    const double a = static_cast<double>(concordant + discordant + tieX);
    const double b = static_cast<double>(concordant + discordant + tieY);
    const double denom = std::sqrt(std::max(0.0, a * b));
    if (denom <= 1e-12) return std::nullopt;
    return static_cast<double>(concordant - discordant) / denom;
}

std::pair<double, double> MathUtils::simpleLinearRegression(const std::vector<double>& /*x*/, const std::vector<double>& /*y*/,
                                                            const ColumnStats& statsX, const ColumnStats& statsY, double pearsonR) {
    if (!std::isfinite(statsX.stddev) || !std::isfinite(statsY.stddev) || !std::isfinite(statsX.mean) || !std::isfinite(statsY.mean)) {
        return {0.0, 0.0};
    }
    if (!std::isfinite(pearsonR)) return {0.0, 0.0};
    if (std::abs(statsX.stddev) <= runtimeConfig().numericEpsilon) return {0.0, statsY.mean};
    // m = r * (Sy / Sx)
    double m = pearsonR * (statsY.stddev / statsX.stddev);
    // c = My - m*Mx
    double c = statsY.mean - (m * statsX.mean);
    if (!std::isfinite(m) || !std::isfinite(c)) return {0.0, statsY.mean};
    return {m, c};
}

MathUtils::Matrix MathUtils::Matrix::identity(size_t n) {
    Matrix res(n, n);
    for (size_t i = 0; i < n; ++i) res.at(i, i) = 1.0;
    return res;
}

MathUtils::Matrix MathUtils::Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            result.at(c, r) = at(r, c);
        }
    }
    return result;
}

MathUtils::Matrix MathUtils::Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    Matrix result(rows, other.cols);
    Matrix otherT = other.transpose();

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t r = 0; r < rows; ++r) {
        const auto& leftRow = data.at(r);
        auto& outRow = result.data[r];
        for (size_t c = 0; c < other.cols; ++c) {
            const auto& rightRow = otherT.data.at(c);
            double sum = 0.0;
            #ifdef USE_OPENMP
            #pragma omp simd reduction(+:sum)
            #endif
            for (size_t k = 0; k < cols; ++k) {
                sum += leftRow[k] * rightRow[k];
            }
            outRow[c] = sum;
        }
    }
    return result;
}

std::optional<MathUtils::Matrix> MathUtils::Matrix::inverse() const {
    if (rows != cols) throw std::invalid_argument("Only square matrices can be inverted.");
    const size_t n = rows;
    Matrix work = *this;
    Matrix inv = Matrix::identity(n);
    Matrix original = *this;
    std::vector<size_t> colPerm(n, 0);
    std::vector<size_t> rowPerm(n, 0);
    std::iota(colPerm.begin(), colPerm.end(), 0);
    std::iota(rowPerm.begin(), rowPerm.end(), 0);

    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::abs(work.at(i, j)));
        }
    }
    const double eps = runtimeConfig().numericEpsilon;
    if (scale <= eps) return std::nullopt;
    const double heuristicTol = std::max(eps, std::numeric_limits<double>::epsilon() * scale * static_cast<double>(n));
    const double pivotTolerance = (getInversionTolerance() > 0.0)
        ? std::max(getInversionTolerance(), heuristicTol)
        : heuristicTol;

    for (size_t i = 0; i < n; ++i) {
        size_t pivotRow = i;
        size_t pivotCol = i;
        double pivotAbs = 0.0;
        for (size_t r = i; r < n; ++r) {
            for (size_t c = i; c < n; ++c) {
                const double v = std::abs(work.data[r][c]);
                if (v > pivotAbs) {
                    pivotAbs = v;
                    pivotRow = r;
                    pivotCol = c;
                }
            }
        }
        if (pivotAbs <= pivotTolerance) return std::nullopt;

        if (pivotRow != i) {
            std::swap(work.data[i], work.data[pivotRow]);
            std::swap(inv.data[i], inv.data[pivotRow]);
            std::swap(rowPerm[i], rowPerm[pivotRow]);
        }

        if (pivotCol != i) {
            for (size_t r = 0; r < n; ++r) {
                std::swap(work.data[r][i], work.data[r][pivotCol]);
            }
            std::swap(colPerm[i], colPerm[pivotCol]);
        }

        const double pivot = work.at(i, i);
        for (size_t j = 0; j < n; ++j) {
            work.at(i, j) /= pivot;
            inv.at(i, j) /= pivot;
        }

        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            const double factor = work.at(r, i);
            if (std::abs(factor) <= pivotTolerance) {
                work.at(r, i) = 0.0;
                continue;
            }
            for (size_t c = 0; c < n; ++c) {
                work.at(r, c) -= factor * work.at(i, c);
                inv.at(r, c) -= factor * inv.at(i, c);
            }
            work.at(r, i) = 0.0;
        }
    }

    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result.data[colPerm[i]] = inv.data[i];
    }

    auto residualError = [&](const Matrix& candidate) {
        Matrix recon = original.multiply(candidate);
        double maxAbsErr = 0.0;
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < n; ++c) {
                const double expected = (r == c) ? 1.0 : 0.0;
                maxAbsErr = std::max(maxAbsErr, std::abs(recon.at(r, c) - expected));
            }
        }
        return maxAbsErr;
    };

    Matrix alt(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            alt.at(colPerm[i], rowPerm[j]) = inv.at(i, j);
        }
    }

    const double errPrimary = residualError(result);
    const double errAlt = residualError(alt);
    return (errAlt + pivotTolerance < errPrimary) ? alt : result;
}

MathUtils::NumericSummary MathUtils::summarizeNumeric(const std::vector<double>& values,
                                                      const ColumnStats* precomputedStats) {
    NumericSummary out;
    if (values.empty()) return out;

    std::vector<double> finiteValues;
    finiteValues.reserve(values.size());
    for (double value : values) {
        if (std::isfinite(value)) {
            finiteValues.push_back(value);
        }
    }
    if (finiteValues.empty()) return out;

    const ColumnStats stats = precomputedStats ? *precomputedStats : Statistics::calculateStats(finiteValues);
    out.mean = stats.mean;
    const double trimQ = 0.10;
    out.median = stats.median;
    out.variance = stats.variance;
    out.stddev = stats.stddev;
    out.skewness = stats.skewness;
    out.kurtosis = stats.kurtosis;

    out.min = *std::min_element(finiteValues.begin(), finiteValues.end());
    out.max = *std::max_element(finiteValues.begin(), finiteValues.end());
    out.range = out.max - out.min;

    out.q1 = CommonUtils::quantileByNth(finiteValues, 0.25);
    out.q3 = CommonUtils::quantileByNth(finiteValues, 0.75);
    out.iqr = out.q3 - out.q1;
    out.p05 = CommonUtils::quantileByNth(finiteValues, 0.05);
    out.p95 = CommonUtils::quantileByNth(finiteValues, 0.95);

    std::vector<double> absDev(finiteValues.size(), 0.0);
    std::vector<double> sorted = finiteValues;
    std::sort(sorted.begin(), sorted.end());
    const size_t trimN = static_cast<size_t>(std::floor(trimQ * static_cast<double>(sorted.size())));
    const size_t keepStart = std::min(trimN, sorted.size());
    const size_t keepEnd = std::max(keepStart, sorted.size() - trimN);
    if (keepEnd > keepStart) {
        double trimSum = 0.0;
        for (size_t i = keepStart; i < keepEnd; ++i) trimSum += sorted[i];
        out.trimmedMean = trimSum / static_cast<double>(keepEnd - keepStart);
    } else {
        out.trimmedMean = out.mean;
    }

    {
        std::unordered_map<long long, size_t> freq;
        size_t best = 0;
        long long bestKey = 0;
        for (double v : finiteValues) {
            const long long q = static_cast<long long>(std::llround(v * 1e6));
            size_t c = ++freq[q];
            if (c > best) {
                best = c;
                bestKey = q;
            }
        }
        out.mode = static_cast<double>(bestKey) / 1e6;
    }

    bool hasNegative = false;
    size_t zeroCount = 0;
    size_t positiveCount = 0;
    double logSum = 0.0;
    double invSum = 0.0;
    for (size_t i = 0; i < finiteValues.size(); ++i) {
        absDev[i] = std::abs(finiteValues[i] - out.median);
        out.sum += finiteValues[i];
        if (std::abs(finiteValues[i]) > 1e-12) out.nonZero++;
        if (finiteValues[i] < 0.0) {
            hasNegative = true;
        } else if (std::abs(finiteValues[i]) <= 1e-12) {
            ++zeroCount;
        } else {
            ++positiveCount;
            logSum += std::log(finiteValues[i]);
            invSum += 1.0 / finiteValues[i];
        }
    }
    out.mad = CommonUtils::quantileByNth(absDev, 0.5);
    if (hasNegative) {
        out.geometricMean = 0.0;
        out.harmonicMean = 0.0;
    } else if (zeroCount > 0) {
        out.geometricMean = 0.0;
        out.harmonicMean = 0.0;
    } else if (positiveCount > 0) {
        out.geometricMean = std::exp(logSum / static_cast<double>(positiveCount));
        out.harmonicMean = static_cast<double>(positiveCount) / std::max(invSum, 1e-12);
    } else {
        out.geometricMean = 0.0;
        out.harmonicMean = 0.0;
    }

    if (std::abs(out.mean) > 1e-12) out.coeffVar = out.stddev / std::abs(out.mean);
    return out;
}

/**
 * Computes the Householder QR decomposition of the matrix.
 * Transforms current matrix into Q (orthogonal) and R (upper triangular) such that A = QR.
 * Uses Householder reflections for superior numerical stability.
 */
void MathUtils::Matrix::qrDecomposition(Matrix& Q, Matrix& R) const {
    size_t m = rows;
    size_t n = cols;
    const double eps = runtimeConfig().numericEpsilon;
    Q = Matrix::identity(m);
    R = *this;

    for (size_t k = 0; k < n && k < m - 1; ++k) {
        // Construct the vector x from the k-th column, starting at row k
        std::vector<double> x(m - k);
        double normX = 0;
        for (size_t i = k; i < m; ++i) {
            x[i - k] = R.data[i][k];
            normX += x[i - k] * x[i - k];
        }
        normX = std::sqrt(normX);

        if (normX <= eps) continue;

        // Choose sign to avoid cancellation in Householder vector u
        double alpha = (R.data[k][k] > 0 ? -1.0 : 1.0) * normX;
        std::vector<double> u = x;
        u[0] -= alpha;
        
        // Normalize u to create the reflector v = u / ||u||
        double normU = 0;
        for (double val : u) normU += val * val;
        normU = std::sqrt(normU);
        if (normU > eps) {
            for (double& val : u) val /= normU;
        } else {
            continue;
        }

        // Apply reflection to R: R = (I - 2vv^T)R = R - 2v(v^T R)
        // Only impacts rows k..m and columns k..n
        for (size_t j = k; j < n; ++j) {
            double dot = 0;
            for (size_t i = k; i < m; ++i) dot += u[i - k] * R.data[i][j];
            for (size_t i = k; i < m; ++i) R.data[i][j] -= 2.0 * u[i - k] * dot;
        }

        // Accumulate reflectors into Q: Q = Q(I - 2vv^T) = Q - 2(Qv)v^T
        for (size_t i = 0; i < m; ++i) {
            double dot = 0;
            for (size_t j = k; j < m; ++j) dot += Q.data[i][j] * u[j - k];
            for (size_t j = k; j < m; ++j) Q.data[i][j] -= 2.0 * dot * u[j - k];
        }
    }
}

std::vector<double> MathUtils::multipleLinearRegression(const Matrix& X, const Matrix& Y) {
    if (X.rows != Y.rows) throw std::invalid_argument("X and Y row dimensions must match for MLR.");
    if (X.rows < X.cols) return std::vector<double>(); // Underdetermined
    
    // Solve X * beta = Y using QR Decomposition
    Matrix Q(0, 0), R(0, 0);
    X.qrDecomposition(Q, R);

    // Check condition number / rank deficiency
    double max_diag = 0.0;
    double min_diag = 1e30;
    for (size_t i = 0; i < X.cols; ++i) {
        double val = std::abs(R.data[i][i]);
        if (val > max_diag) max_diag = val;
        if (val < min_diag) min_diag = val;
    }

    const double rankTol = std::max(runtimeConfig().numericEpsilon,
                                    std::numeric_limits<double>::epsilon() * std::max(1.0, max_diag) * static_cast<double>(X.cols));
    if (min_diag <= rankTol) {
        return std::vector<double>(); // Singular or ill-conditioned matrix
    }

    // Q is orthogonal (m x m), R is upper triangular (m x n)
    // Minimizer solves: R * beta = Q^T * Y
    Matrix QT = Q.transpose();
    Matrix QTY = QT.multiply(Y);

    const size_t n = X.cols;
    std::vector<double> rhs(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        rhs[i] = QTY.data[i][0];
    }

    std::vector<double> beta;
    if (!solveUpperTriangular(R, n, rhs, beta)) {
        return std::vector<double>();
    }
    return beta;
}

MLRDiagnostics MathUtils::performMLRWithDiagnostics(const Matrix& X, const Matrix& Y) {
    MLRDiagnostics diag;
    diag.success = false;
    if (X.rows != Y.rows || X.rows <= X.cols) return diag;

    // 1. Solve for coefficients using existing QR logic (or just reuse logic)
    std::vector<double> beta = multipleLinearRegression(X, Y);
    if (beta.empty()) return diag;
    diag.coefficients = beta;

    size_t n = X.rows;
    size_t k = X.cols - 1; // Number of predictors (excluding intercept)

    // 2. Calculate RSS and TSS
    double rss = 0, ySum = 0;
    std::vector<double> y_pred(n);
    for (size_t i = 0; i < n; ++i) {
        y_pred[i] = 0;
        for (size_t j = 0; j < X.cols; ++j) {
            y_pred[i] += X.data[i][j] * beta[j];
        }
        double diff = Y.data[i][0] - y_pred[i];
        rss += diff * diff;
        ySum += Y.data[i][0];
    }
    double yMean = ySum / n;
    double tss = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = Y.data[i][0] - yMean;
        tss += d * d;
    }

    diag.rSquared = (tss > 0) ? (1.0 - rss / tss) : 0.0;
    const size_t dfResidual = n - k - 1;
    if (dfResidual == 0) return diag;
    diag.adjustedRSquared = 1.0 - (1.0 - diag.rSquared) * (double(n - 1) / dfResidual);
    
    // F-statistic
    if (rss > 0 && k > 0) {
        diag.fStatistic = ((tss - rss) / k) / (rss / dfResidual);
        // Model p-value from F(k, n-k-1) - using beta distribution relation
        // F = (d2 * I_x(d1/2, d2/2)) / (d1 * (1 - I_x(d1/2, d2/2))) where x is from F
        // Easier: p-value = betainc(d2/2, d1/2, d2 / (d2 + d1*F))
        diag.modelPValue = betainc(dfResidual / 2.0, k / 2.0, dfResidual / (dfResidual + k * diag.fStatistic));
    } else {
        diag.fStatistic = 0; diag.modelPValue = 1.0;
    }

    // 3. Standard Errors for coefficients
    // SE = sqrt(s^2 * diag((X^TX)^-1)), where (X^TX)^-1 = R^-1 * (R^-1)^T from QR.
    Matrix Q(0, 0), R(0, 0);
    X.qrDecomposition(Q, R);

    const size_t p = X.cols;
    double max_diag = 0.0;
    double min_diag = 1e30;
    for (size_t i = 0; i < p; ++i) {
        const double v = std::abs(R.data[i][i]);
        if (v > max_diag) max_diag = v;
        if (v < min_diag) min_diag = v;
    }
    const double rankTol = std::max(runtimeConfig().numericEpsilon,
                                    std::numeric_limits<double>::epsilon() * std::max(1.0, max_diag) * static_cast<double>(p));
    if (min_diag <= rankTol) return diag;

    const double s2 = rss / static_cast<double>(dfResidual);
    diag.stdErrors.resize(p);
    diag.tStats.resize(p);
    diag.pValues.resize(p);

    std::vector<double> e(p, 0.0);
    std::vector<double> z;
    std::vector<double> col;
    for (size_t j = 0; j < p; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;

        if (!solveLowerFromUpperTranspose(R, p, e, z)) return diag;
        if (!solveUpperTriangular(R, p, z, col)) return diag;

        double variance = s2 * col[j];
        if (variance < 0.0 && std::abs(variance) < 1e-15) variance = 0.0;
        if (variance < 0.0) return diag;

        diag.stdErrors[j] = std::sqrt(variance);
        if (diag.stdErrors[j] > 0.0) {
            diag.tStats[j] = diag.coefficients[j] / diag.stdErrors[j];
            diag.pValues[j] = getPValueFromT(diag.tStats[j], dfResidual);
        } else {
            diag.tStats[j] = 0.0;
            diag.pValues[j] = 1.0;
        }
    }

    diag.success = true;
    return diag;
}
