#include "MathUtils.h"
#include "CommonUtils.h"
#include "Statistics.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
double gSignificanceAlpha = 0.05;
double gNumericEpsilon = 1e-12;
size_t gBetaFallbackIntervalsStart = 4096;
size_t gBetaFallbackIntervalsMax = 65536;
double gBetaFallbackTolerance = 1e-8;

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
    size_t startIntervals = std::max<size_t>(256, gBetaFallbackIntervalsStart);
    size_t maxIntervals = std::max(startIntervals, gBetaFallbackIntervalsMax);

    double prev = midpointIntegrateBetaRegularized(a, b, x, startIntervals);
    for (size_t intervals = startIntervals * 2; intervals <= maxIntervals; intervals *= 2) {
        double cur = midpointIntegrateBetaRegularized(a, b, x, intervals);
        // Midpoint has O(h^2) error; this scaled delta approximates residual truncation error.
        const double richardsonErr = std::abs(cur - prev) / 3.0;
        if (richardsonErr < gBetaFallbackTolerance) return cur;
        prev = cur;
    }
    return prev;
}

std::pair<double, bool> betaContinuedFraction(double a, double b, double x) {
    constexpr int maxIter = 400;
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
        if (std::abs(diag) <= gNumericEpsilon) return false;
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
        if (std::abs(diag) <= gNumericEpsilon) return false;
        x[i] = rhs / diag;
    }
    return true;
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
    if (t_abs > 1e10) return 0.0;
    
    double x = nu / (nu + t_abs * t_abs);
    return betainc(nu / 2.0, 0.5, x); // two-tailed
}

double MathUtils::getPValueFromT(double t, size_t df) {
    return pvalue_from_t(t, df);
}

double MathUtils::getCriticalT(double alpha, size_t df) {
    if (df == 0) return 0.0;
    
    // For alpha = 0.05, two-tailed z = 1.96
    double p = 1.0 - alpha / 2.0;
    
    // Simple z-approximation for large df
    double z = 0.0;
    if (p < 0.5) {
        // Not expected for common alpha like 0.05
        z = -std::sqrt(-2.0 * std::log(p));
    } else {
        z = std::sqrt(-2.0 * std::log(1.0 - p));
    }
    
    // Refine z (Abramowitz and Stegun)
    const double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
    const double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
    double t = std::sqrt(-2.0 * std::log(1.0 - p));
    z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0);

    if (df > 100) return z;

    // Student's t adjustment for smaller df
    return z * (1.0 + (z * z + 1.0) / (4.0 * df) + (5.0 * z * z * z * z + 16.0 * z * z + 3.0) / (96.0 * df * df));
}

void MathUtils::setSignificanceAlpha(double alpha) {
    if (alpha > 0.0 && alpha < 1.0) {
        gSignificanceAlpha = alpha;
    }
}

double MathUtils::getSignificanceAlpha() {
    return gSignificanceAlpha;
}

void MathUtils::setNumericTuning(double numericEpsilon,
                                 size_t betaIntervalsStart,
                                 size_t betaIntervalsMax,
                                 double betaTolerance) {
    if (numericEpsilon > 0.0) gNumericEpsilon = numericEpsilon;
    if (betaIntervalsStart >= 256) gBetaFallbackIntervalsStart = betaIntervalsStart;
    if (betaIntervalsMax >= gBetaFallbackIntervalsStart) gBetaFallbackIntervalsMax = betaIntervalsMax;
    if (betaTolerance > 0.0) gBetaFallbackTolerance = betaTolerance;
}

Significance MathUtils::calculateSignificance(double r, size_t n) {
    Significance sig{0.0, 0.0, false};
    if (n <= 2) return sig;
    
    // Avoid division by zero if r is perfectly 1 or -1
    if (std::abs(r) >= 0.9999999) {
        sig.p_value = gNumericEpsilon;
        sig.t_stat = (r > 0 ? 1e6 : -1e6);
        sig.is_significant = true;
        return sig;
    }

    size_t df = n - 2;
    sig.t_stat = r * std::sqrt(static_cast<double>(df) / (1.0 - r * r));
    sig.p_value = getPValueFromT(sig.t_stat, df);
    sig.is_significant = (sig.p_value < gSignificanceAlpha);

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
        [statsX, statsY](double valX, double valY) {
            return (valX - statsX.mean) * (valY - statsY.mean);
        });
    
    // Pearson r = Covariance(X,Y) / (StdDevX * StdDevY)
    double covariance = covarianceSum / (n - 1);
    return covariance / (statsX.stddev * statsY.stddev);
}

std::pair<double, double> MathUtils::simpleLinearRegression(const std::vector<double>& x, const std::vector<double>& y,
                                                            const ColumnStats& statsX, const ColumnStats& statsY, double pearsonR) {
    if (statsX.stddev == 0) return {0.0, 0.0};
    // m = r * (Sy / Sx)
    double m = pearsonR * (statsY.stddev / statsX.stddev);
    // c = My - m*Mx
    double c = statsY.mean - (m * statsX.mean);
    return {m, c};
}

MathUtils::Matrix MathUtils::Matrix::identity(size_t n) {
    Matrix res(n, n);
    for (size_t i = 0; i < n; ++i) res.data[i][i] = 1.0;
    return res;
}

MathUtils::Matrix MathUtils::Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            result.data[c][r] = data[r][c];
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
        const auto& leftRow = data[r];
        auto& outRow = result.data[r];
        for (size_t c = 0; c < other.cols; ++c) {
            const auto& rightRow = otherT.data[c];
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
    std::vector<size_t> colPerm(n, 0);
    std::iota(colPerm.begin(), colPerm.end(), 0);

    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            scale = std::max(scale, std::abs(work.data[i][j]));
        }
    }
    if (scale <= gNumericEpsilon) return std::nullopt;
    const double pivotTolerance = std::max(gNumericEpsilon, gNumericEpsilon * scale * static_cast<double>(n));

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
        }

        if (pivotCol != i) {
            for (size_t r = 0; r < n; ++r) {
                std::swap(work.data[r][i], work.data[r][pivotCol]);
            }
            std::swap(colPerm[i], colPerm[pivotCol]);
        }

        const double pivot = work.data[i][i];
        for (size_t j = 0; j < n; ++j) {
            work.data[i][j] /= pivot;
            inv.data[i][j] /= pivot;
        }

        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            const double factor = work.data[r][i];
            if (std::abs(factor) <= pivotTolerance) {
                work.data[r][i] = 0.0;
                continue;
            }
            for (size_t c = 0; c < n; ++c) {
                work.data[r][c] -= factor * work.data[i][c];
                inv.data[r][c] -= factor * inv.data[i][c];
            }
            work.data[r][i] = 0.0;
        }
    }

    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result.data[colPerm[i]] = inv.data[i];
    }
    return result;
}

MathUtils::NumericSummary MathUtils::summarizeNumeric(const std::vector<double>& values,
                                                      const ColumnStats* precomputedStats) {
    NumericSummary out;
    if (values.empty()) return out;

    const ColumnStats stats = precomputedStats ? *precomputedStats : Statistics::calculateStats(values);
    out.mean = stats.mean;
    out.median = stats.median;
    out.variance = stats.variance;
    out.stddev = stats.stddev;
    out.skewness = stats.skewness;
    out.kurtosis = stats.kurtosis;

    out.min = *std::min_element(values.begin(), values.end());
    out.max = *std::max_element(values.begin(), values.end());
    out.range = out.max - out.min;

    out.q1 = CommonUtils::quantileByNth(values, 0.25);
    out.q3 = CommonUtils::quantileByNth(values, 0.75);
    out.iqr = out.q3 - out.q1;
    out.p05 = CommonUtils::quantileByNth(values, 0.05);
    out.p95 = CommonUtils::quantileByNth(values, 0.95);

    std::vector<double> absDev(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        absDev[i] = std::abs(values[i] - out.median);
        out.sum += values[i];
        if (std::abs(values[i]) > 1e-12) out.nonZero++;
    }
    out.mad = CommonUtils::quantileByNth(absDev, 0.5);

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

        if (normX <= 1e-12) continue;

        // Choose sign to avoid cancellation in Householder vector u
        double alpha = (R.data[k][k] > 0 ? -1.0 : 1.0) * normX;
        std::vector<double> u = x;
        u[0] -= alpha;
        
        // Normalize u to create the reflector v = u / ||u||
        double normU = 0;
        for (double val : u) normU += val * val;
        normU = std::sqrt(normU);
        if (normU > 1e-12) {
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

    if (min_diag < 1e-12 || (max_diag / min_diag) > 1e10) {
        return std::vector<double>(); // Singular or ill-conditioned matrix
    }

    // Q is orthogonal (m x m), R is upper triangular (m x n)
    // Minimizer solves: R * beta = Q^T * Y
    Matrix QT = Q.transpose();
    Matrix QTY = QT.multiply(Y);

    size_t n = X.cols;
    std::vector<double> beta(n, 0.0);

    // Back substitution
    for (size_t i = n; i-- > 0;) {
        double sum = QTY.data[i][0];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= R.data[i][j] * beta[j];
        }
        beta[i] = sum / R.data[i][i];
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
    if (min_diag <= gNumericEpsilon || (max_diag / min_diag) > 1e10) return diag;

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
