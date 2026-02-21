#include "MathUtils.h"
#include "StatsEngine.h" // Needed for ColumnStats
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

// Simple approximation of the Student's t-distribution cumulative distribution function (CDF) 
// for evaluating the p-value without pulling in boost::math. Valid for large n.
static double estimatePValueFromT(double t, size_t df) {
    if (df <= 0) return 1.0;
    // For n > 30, t-distribution approaches standard normal. 
    // We use a simple approximation here for the agentic scope.
    double x = t / std::sqrt(2.0);
    // std::erfc(x) = 1 - erf(x), mapping directly to two-tailed test
    double p = std::erfc(std::abs(x)); 
    return p;
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
    double covarianceSum = 0;
    for (size_t i = 0; i < n; ++i) {
        covarianceSum += (x[i] - statsX.mean) * (y[i] - statsY.mean);
    }
    
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
        if (std::abs(augmented.data[i][i]) < 1e-9) {
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

std::vector<double> MathUtils::multipleLinearRegression(const Matrix& X, const Matrix& Y) {
    if (X.rows != Y.rows) throw std::invalid_argument("X and Y row dimensions must match for MLR.");
    
    // Normal Equation: beta = (X^T * X)^-1 * X^T * Y
    Matrix XT = X.transpose();
    Matrix XTX = XT.multiply(X);
    
    // Check if invertible
    Matrix XTX_inv = XTX.inverse();
    if (XTX_inv.rows == 0) {
        return std::vector<double>(); 
    }

    Matrix XT_Y = XT.multiply(Y);
    Matrix betaMatrix = XTX_inv.multiply(XT_Y);

    std::vector<double> beta(betaMatrix.rows);
    for (size_t i = 0; i < betaMatrix.rows; ++i) {
        beta[i] = betaMatrix.data[i][0];
    }

    return beta;
}
