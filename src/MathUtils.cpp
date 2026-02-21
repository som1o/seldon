#include "MathUtils.h"
#include "StatsEngine.h" // Needed for ColumnStats
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

static double estimatePValueFromT(double t, size_t df) {
    if (df <= 0) return 1.0;
    t = std::abs(t);
    
    // Large degree of freedom approximation (Normal distribution)
    if (df >= 100) {
        return std::erfc(t / std::sqrt(2.0));
    }
    
    double nu = static_cast<double>(df);
    double c = std::tgamma((nu + 1.0) / 2.0) / (std::sqrt(nu * M_PI) * std::tgamma(nu / 2.0));
    
    // Transform integral from t to infinity into bounded arctan(t/sqrt(nu)) to pi/2
    double start = std::atan(t / std::sqrt(nu));
    double end = M_PI / 2.0;
    int steps = 50; // Reduced from 1000 for efficiency
    double step = (end - start) / steps;
    double p = 0.0;
    
    for (int i = 0; i < steps; ++i) {
        double y1 = start + i * step;
        double y2 = start + (i + 1) * step;
        double ym = (y1 + y2) / 2.0;
        
        auto f = [c, nu](double y) {
            double cos_y = std::cos(y);
            return c * std::sqrt(nu) * std::pow(cos_y, nu - 1.0);
        };
        
        p += (f(y1) + 4.0 * f(ym) + f(y2)) * (step / 6.0);
    }
    
    return p * 2.0; // Two-tailed p-value
}

Significance MathUtils::calculateSignificance(double r, size_t n) {
    Significance sig{0.0, 1.0, false};
    if (n <= 2) return sig;
    
    // Avoid division by zero if r is perfectly 1 or -1
    if (std::abs(r) == 1.0) {
        sig.p_value = 0.0;
        sig.is_significant = true;
        return sig;
    }

    // Calculate t-statistic: t = r * sqrt((n-2) / (1-r^2))
    size_t df = n - 2;
    sig.t_stat = r * std::sqrt(static_cast<double>(df) / (1.0 - r * r));
    
    // Calculate 2-tailed p-value
    sig.p_value = estimatePValueFromT(sig.t_stat, df);
    sig.is_significant = (sig.p_value < 0.05);

    return sig;
}

double MathUtils::calculatePearson(const std::vector<double>& x, const std::vector<double>& y, 
                                   const ColumnStats& statsX, const ColumnStats& statsY) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    if (statsX.stddev == 0 || statsY.stddev == 0) return 0.0;

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
MathUtils::Matrix MathUtils::Matrix::inverse() const {
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
            return Matrix(0, 0); // Return empty on failure silently
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

// Computes the Householder QR decomposition
void MathUtils::Matrix::qrDecomposition(Matrix& Q, Matrix& R) const {
    size_t m = rows;
    size_t n = cols;
    Q = Matrix::identity(m);
    R = *this;

    for (size_t k = 0; k < n && k < m - 1; ++k) {
        std::vector<double> x(m - k);
        double normX = 0;
        for (size_t i = k; i < m; ++i) {
            x[i - k] = R.data[i][k];
            normX += x[i - k] * x[i - k];
        }
        normX = std::sqrt(normX);

        if (normX <= 1e-12) continue;

        double alpha = (R.data[k][k] > 0 ? -1.0 : 1.0) * normX;
        std::vector<double> u = x;
        u[0] -= alpha;
        
        double normU = 0;
        for (double val : u) normU += val * val;
        normU = std::sqrt(normU);
        if (normU > 1e-12) {
            for (double& val : u) val /= normU;
        } else {
            continue;
        }

        // R = R - 2 u (u^T R)
        for (size_t j = k; j < n; ++j) {
            double dot = 0;
            for (size_t i = k; i < m; ++i) dot += u[i - k] * R.data[i][j];
            for (size_t i = k; i < m; ++i) R.data[i][j] -= 2.0 * u[i - k] * dot;
        }

        // Q = Q - 2 (Q u) u^T
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
