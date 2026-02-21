#include "MathUtils.h"
#include "StatsEngine.h" // Needed for ColumnStats
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <algorithm>

// Regularized incomplete beta function I_x(a, b) using continued fractions (Lentz's method)
static double betainc(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return NAN;
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    // Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - betainc(b, a, 1.0 - x);
    }

    double logBeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    double front = std::exp(a * std::log(x) + b * std::log(1.0 - x) - logBeta) / a;

    // Continued fraction using Lentz's method
    double f = 1.0, c = 1.0, d = 0.0;
    const double tiny = 1e-30;
    const double eps = 1e-15;

    for (int i = 1; i <= 200; ++i) {
        double m = i / 2.0;
        double numerator;
        if (i % 2 == 0) {
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        } else {
            double m_prev = (i - 1) / 2.0;
            numerator = -((a + m_prev) * (a + b + m_prev) * x) / ((a + 2.0 * m_prev) * (a + 2.0 * m_prev + 1.0));
        }

        d = 1.0 + numerator * d;
        if (std::abs(d) < tiny) d = tiny;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (std::abs(c) < tiny) c = tiny;

        double delta = c * d;
        f *= delta;
        if (std::abs(delta - 1.0) < eps) return front * f;
    }

    return front * f; // Fallback if max iterations reached
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

Significance MathUtils::calculateSignificance(double r, size_t n) {
    Significance sig{0.0, 0.0, false};
    if (n <= 2) return sig;
    
    // Avoid division by zero if r is perfectly 1 or -1
    if (std::abs(r) >= 0.9999999) {
        sig.p_value = 1e-12;
        sig.t_stat = (r > 0 ? 1e6 : -1e6);
        sig.is_significant = true;
        return sig;
    }

    size_t df = n - 2;
    sig.t_stat = r * std::sqrt(static_cast<double>(df) / (1.0 - r * r));
    sig.p_value = getPValueFromT(sig.t_stat, df);
    sig.is_significant = (sig.p_value < 0.05);

    return sig;
}

std::optional<double> MathUtils::calculatePearson(const std::vector<double>& x, const std::vector<double>& y, 
                                           const ColumnStats& statsX, const ColumnStats& statsY) {
    if (x.size() != y.size() || x.empty()) return std::nullopt;
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
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < other.cols; ++c) {
            for (size_t k = 0; k < cols; ++k) {
                result.data[r][c] += data[r][k] * other.data[k][c];
            }
        }
    }
    return result;
}

// Basic implementation of Full Pivoting Gaussian Elimination for Matrix Inversion
std::optional<MathUtils::Matrix> MathUtils::Matrix::inverse() const {
    if (rows != cols) throw std::invalid_argument("Only square matrices can be inverted.");
    size_t n = rows;
    Matrix augmented(n, 2 * n);

    // Create [Matrix | Identity]
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented.data[i][j] = data[i][j];
            augmented.data[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gaussian elimination with partial pivoting
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(augmented.data[k][i]) > std::abs(augmented.data[maxRow][i])) {
                maxRow = k;
            }
        }

        // Swap rows
        std::swap(augmented.data[i], augmented.data[maxRow]);

        // Check for singularity
        if (std::abs(augmented.data[i][i]) < 1e-12) {
            return std::nullopt; 
        }

        // Divide pivot row by pivot value
        double pivot = augmented.data[i][i];
        for (size_t j = i; j < 2 * n; ++j) {
            augmented.data[i][j] /= pivot;
        }

        // Eliminate column entries in other rows
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = augmented.data[k][i];
                for (size_t j = i; j < 2 * n; ++j) {
                    augmented.data[k][j] -= factor * augmented.data[i][j];
                }
            }
        }
    }

    // Extract inverted matrix
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.data[i][j] = augmented.data[i][j + n];
        }
    }
    return result;
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
    for (int i = n - 1; i >= 0; --i) {
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
    diag.adjustedRSquared = 1.0 - (1.0 - diag.rSquared) * (double(n - 1) / (n - k - 1));
    
    // F-statistic
    if (rss > 0 && k > 0) {
        diag.fStatistic = ((tss - rss) / k) / (rss / (n - k - 1));
        // Model p-value from F(k, n-k-1) - using beta distribution relation
        // F = (d2 * I_x(d1/2, d2/2)) / (d1 * (1 - I_x(d1/2, d2/2))) where x is from F
        // Easier: p-value = betainc(d2/2, d1/2, d2 / (d2 + d1*F))
        diag.modelPValue = betainc((n - k - 1) / 2.0, k / 2.0, (n - k - 1) / (n - k - 1 + k * diag.fStatistic));
    } else {
        diag.fStatistic = 0; diag.modelPValue = 1.0;
    }

    // 3. Standard Errors for coefficients
    // SE = sqrt( s^2 * diag((X^TX)^-1) )
    Matrix XT = X.transpose();
    Matrix XTX = XT.multiply(X);
    auto XTX_inv_opt = XTX.inverse();
    
    if (XTX_inv_opt) {
        Matrix XTX_inv = *XTX_inv_opt;
        double s2 = rss / (n - k - 1);
        diag.stdErrors.resize(X.cols);
        diag.tStats.resize(X.cols);
        diag.pValues.resize(X.cols);
        
        for (size_t j = 0; j < X.cols; ++j) {
            diag.stdErrors[j] = std::sqrt(s2 * XTX_inv.data[j][j]);
            diag.tStats[j] = diag.coefficients[j] / diag.stdErrors[j];
            diag.pValues[j] = getPValueFromT(diag.tStats[j], n - k - 1);
        }
    } else {
        // Fallback or mark as failed
        return diag;
    }

    diag.success = true;
    return diag;
}
