#pragma once
#include "Statistics.h"
#include <vector>
#include <string>
#include <optional>

struct Significance {
    double t_stat;
    double p_value;
    bool is_significant; // Checked against MathUtils::getSignificanceAlpha()
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
    struct NumericSummary {
        double mean = 0.0;
        double median = 0.0;
        double variance = 0.0;
        double stddev = 0.0;
        double skewness = 0.0;
        double kurtosis = 0.0;
        double min = 0.0;
        double max = 0.0;
        double range = 0.0;
        double q1 = 0.0;
        double q3 = 0.0;
        double iqr = 0.0;
        double p05 = 0.0;
        double p95 = 0.0;
        double mad = 0.0;
        double coeffVar = 0.0;
        double sum = 0.0;
        size_t nonZero = 0;
    };

    static void setSignificanceAlpha(double alpha);
    static double getSignificanceAlpha();
    static void setNumericTuning(double numericEpsilon,
                                 size_t betaIntervalsStart,
                                 size_t betaIntervalsMax,
                                 double betaTolerance);

    /**
     * @brief Computes Pearson's correlation coefficient between two numeric vectors.
     * @pre x.size() == y.size() and size >= 2.
     * @pre statsX/stddev and statsY/stddev correspond to x and y and are non-zero.
     * @post Returns std::nullopt when correlation is undefined.
     */
    static std::optional<double> calculatePearson(const std::vector<double>& x, const std::vector<double>& y, 
                                          const ColumnStats& statsX, const ColumnStats& statsY);

    /**
     * @brief Computes t-statistic and p-value significance metadata for Pearson r.
     * @pre n is sample size used to compute r.
     * @post Returns non-significant result for n <= 2.
     */
    static Significance calculateSignificance(double r, size_t n);

    /**
     * @brief Two-tailed p-value from t-statistic and degrees of freedom.
     * @pre df >= 0.
     */
    static double getPValueFromT(double t, size_t df);

    /**
     * @brief Approximates critical t value for two-tailed alpha.
     * @pre 0 < alpha < 1 and df >= 0.
     */
    static double getCriticalT(double alpha, size_t df); // Two-tailed

    /**
     * @brief Computes simple linear regression parameters y = m*x + c.
     * @pre statsX/stddev and statsY/stddev correspond to x and y.
     * @post Returns {0,0} when slope is undefined.
     */
    static std::pair<double, double> simpleLinearRegression(const std::vector<double>& x, const std::vector<double>& y,
                                                            const ColumnStats& statsX, const ColumnStats& statsY, double pearsonR);

    /**
     * @brief Computes consistent descriptive numeric stats used across reports.
     * @pre values may be empty.
     * @post Returns zero-initialized summary for empty input.
     */
    static NumericSummary summarizeNumeric(const std::vector<double>& values,
                                           const ColumnStats* precomputedStats = nullptr);

    // Basic Matrix operations for Multiple Linear Regression
    struct Matrix {
        std::vector<std::vector<double>> data;
        size_t rows;
        size_t cols;

        Matrix(size_t r, size_t c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}

        /**
         * @brief Builds identity matrix I(n).
         * @pre n >= 0.
         */
        static Matrix identity(size_t n);

        /**
         * @brief Returns transpose of current matrix.
         */
        Matrix transpose() const;

        /**
         * @brief Matrix multiplication this * other.
         * @pre this->cols == other.rows.
         * @throws std::invalid_argument on shape mismatch.
         */
        Matrix multiply(const Matrix& other) const;

        /**
         * @brief Inverts a square matrix using full-pivot Gauss-Jordan elimination.
         * @pre rows == cols.
         * @post Returns std::nullopt for singular/near-singular matrices.
         * @throws std::invalid_argument when matrix is not square.
         */
        std::optional<Matrix> inverse() const;

        /**
         * @brief Computes Householder QR decomposition: A = Q*R.
         * @pre Q and R are output matrices and will be overwritten.
         */
        void qrDecomposition(Matrix& Q, Matrix& R) const;
    };

    /**
     * @brief Solves multiple linear regression coefficients from design matrix X and target Y.
     * @pre X.rows == Y.rows.
     * @post Returns empty vector for underdetermined or ill-conditioned problems.
     * @throws std::invalid_argument when row dimensions mismatch.
     */
    static std::vector<double> multipleLinearRegression(const Matrix& X, const Matrix& Y);

    /**
     * @brief Performs MLR and returns coefficients with diagnostics.
     * @pre X.rows == Y.rows and residual degrees of freedom > 0.
     * @post success=false when regression cannot be solved robustly.
     */
    static MLRDiagnostics performMLRWithDiagnostics(const Matrix& X, const Matrix& Y);
};
