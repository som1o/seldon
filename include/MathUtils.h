#pragma once
#include <vector>
#include <string>
#include <optional>

// Forward declarations to avoid complex circular includes where possible
struct ColumnStats;

struct Significance {
    double t_stat;
    double p_value;
    bool is_significant; // Default checking against p < 0.05
};

struct MLRDiagnostics {
    std::vector<double> coefficients;
    std::vector<double> stdErrors;
    std::vector<double> tStats;
    std::vector<double> pValues;
    double rSquared;
    double adjustedRSquared;
    double fStatistic;
    double modelPValue;
    bool success;
};

class MathUtils {
public:
    // Pearson Correlation Coefficient calculation
    static std::optional<double> calculatePearson(const std::vector<double>& x, const std::vector<double>& y, 
                                          const ColumnStats& statsX, const ColumnStats& statsY);

    // Calculate statistical significance for a given Pearson r and sample size n
    static Significance calculateSignificance(double r, size_t n);

    // Primitive statistical functions
    static double getPValueFromT(double t, size_t df);
    static double getCriticalT(double alpha, size_t df); // Two-tailed

    // Simple Linear Regression: y = mx + c. Returns {slope(m), intercept(c)}
    static std::pair<double, double> simpleLinearRegression(const std::vector<double>& x, const std::vector<double>& y,
                                                            const ColumnStats& statsX, const ColumnStats& statsY, double pearsonR);

    // Basic Matrix operations for Multiple Linear Regression
    struct Matrix {
        std::vector<std::vector<double>> data;
        size_t rows;
        size_t cols;

        Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}

        static Matrix identity(size_t n);
        Matrix transpose() const;
        Matrix multiply(const Matrix& other) const;
        std::optional<Matrix> inverse() const; // Using Gaussian Elimination
        void qrDecomposition(Matrix& Q, Matrix& R) const;
    };

    // Multiple Linear Regression using Normal Equation: beta = (X^T * X)^-1 * X^T * Y
    // Returns the vector of beta coefficients.
    static std::vector<double> multipleLinearRegression(const Matrix& X, const Matrix& Y);

    // Advanced version with full statistical diagnostics
    static MLRDiagnostics performMLRWithDiagnostics(const Matrix& X, const Matrix& Y);
};
