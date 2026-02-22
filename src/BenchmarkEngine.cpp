#include "BenchmarkEngine.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
using Matrix = std::vector<std::vector<double>>;

std::vector<double> fitLinear(const Matrix& X, const std::vector<double>& y, double lambda = 0.0) {
    if (X.empty()) return {};
    size_t n = X.size();
    size_t p = X[0].size();

    Matrix a(p + 1, std::vector<double>(p + 2, 0.0));
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> row(p + 1, 1.0);
        for (size_t j = 0; j < p; ++j) row[j + 1] = X[i][j];
        for (size_t r = 0; r < p + 1; ++r) {
            for (size_t c = 0; c < p + 1; ++c) a[r][c] += row[r] * row[c];
            a[r][p + 1] += row[r] * y[i];
        }
    }
    for (size_t d = 1; d < p + 1; ++d) a[d][d] += lambda;

    for (size_t i = 0; i < p + 1; ++i) {
        size_t piv = i;
        for (size_t r = i + 1; r < p + 1; ++r) if (std::abs(a[r][i]) > std::abs(a[piv][i])) piv = r;
        std::swap(a[i], a[piv]);
        if (std::abs(a[i][i]) < 1e-12) continue;
        double div = a[i][i];
        for (size_t c = i; c < p + 2; ++c) a[i][c] /= div;
        for (size_t r = 0; r < p + 1; ++r) {
            if (r == i) continue;
            double fac = a[r][i];
            for (size_t c = i; c < p + 2; ++c) a[r][c] -= fac * a[i][c];
        }
    }

    std::vector<double> beta(p + 1, 0.0);
    for (size_t i = 0; i < p + 1; ++i) beta[i] = a[i][p + 1];
    return beta;
}

double predictLinear(const std::vector<double>& beta, const std::vector<double>& x) {
    double y = beta.empty() ? 0.0 : beta[0];
    for (size_t i = 0; i < x.size() && (i + 1) < beta.size(); ++i) y += beta[i + 1] * x[i];
    return y;
}

BenchmarkResult evalLinear(const std::string& name, const Matrix& X, const std::vector<double>& y, int kfold, double lambda = 0.0) {
    BenchmarkResult res;
    res.model = name;
    if (X.empty()) return res;

    size_t n = X.size();
    int folds = std::max(2, std::min<int>(kfold, static_cast<int>(n)));
    size_t foldSize = std::max<size_t>(1, n / folds);

    std::vector<double> preds(n, 0.0);
    for (int f = 0; f < folds; ++f) {
        size_t s = static_cast<size_t>(f) * foldSize;
        size_t e = (f == folds - 1) ? n : std::min(n, s + foldSize);

        Matrix Xtr;
        std::vector<double> ytr;
        for (size_t i = 0; i < n; ++i) {
            if (i >= s && i < e) continue;
            Xtr.push_back(X[i]);
            ytr.push_back(y[i]);
        }

        auto beta = fitLinear(Xtr, ytr, lambda);
        for (size_t i = s; i < e; ++i) preds[i] = predictLinear(beta, X[i]);
    }

    double mse = 0.0, mean = 0.0;
    for (double v : y) mean += v;
    mean /= n;
    double tss = 0.0, rss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = y[i] - preds[i];
        mse += d * d;
        rss += d * d;
        double t = y[i] - mean;
        tss += t * t;
    }
    res.rmse = std::sqrt(mse / n);
    res.r2 = (tss > 0 ? 1.0 - rss / tss : 0.0);
    res.actual = y;
    res.predicted = preds;
    return res;
}

BenchmarkResult evalTreeStump(const Matrix& X, const std::vector<double>& y, int kfold) {
    BenchmarkResult res;
    res.model = "DecisionTreeStump";
    if (X.empty() || X[0].empty()) return res;

    size_t n = X.size();
    size_t p = X[0].size();
    std::vector<double> preds(n, 0.0);
    std::vector<double> gain(p, 0.0);

    int folds = std::max(2, std::min<int>(kfold, static_cast<int>(n)));
    size_t foldSize = std::max<size_t>(1, n / folds);

    for (int f = 0; f < folds; ++f) {
        size_t s = static_cast<size_t>(f) * foldSize;
        size_t e = (f == folds - 1) ? n : std::min(n, s + foldSize);

        size_t bestFeat = 0;
        double bestThresh = 0.0;
        double bestErr = 1e300;

        for (size_t feat = 0; feat < p; ++feat) {
            std::vector<double> cand;
            for (size_t i = 0; i < n; ++i) if (!(i >= s && i < e)) cand.push_back(X[i][feat]);
            if (cand.empty()) continue;
            std::nth_element(cand.begin(), cand.begin() + cand.size() / 2, cand.end());
            double thr = cand[cand.size() / 2];

            double lsum = 0, rsum = 0; size_t lc = 0, rc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (i >= s && i < e) continue;
                if (X[i][feat] <= thr) { lsum += y[i]; lc++; }
                else { rsum += y[i]; rc++; }
            }
            double lmean = lc ? lsum / lc : 0.0;
            double rmean = rc ? rsum / rc : 0.0;

            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (i >= s && i < e) continue;
                double pred = (X[i][feat] <= thr) ? lmean : rmean;
                double d = y[i] - pred;
                err += d * d;
            }
            if (err < bestErr) {
                bestErr = err;
                bestFeat = feat;
                bestThresh = thr;
            }
        }

        gain[bestFeat] += 1.0;

        double lsum = 0, rsum = 0; size_t lc = 0, rc = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i >= s && i < e) continue;
            if (X[i][bestFeat] <= bestThresh) { lsum += y[i]; lc++; }
            else { rsum += y[i]; rc++; }
        }
        double lmean = lc ? lsum / lc : 0.0;
        double rmean = rc ? rsum / rc : 0.0;

        for (size_t i = s; i < e; ++i) preds[i] = (X[i][bestFeat] <= bestThresh) ? lmean : rmean;
    }

    double mse = 0.0, mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double tss = 0.0, rss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = y[i] - preds[i];
        mse += d * d;
        rss += d * d;
        double t = y[i] - mean;
        tss += t * t;
    }

    res.rmse = std::sqrt(mse / n);
    res.r2 = (tss > 0 ? 1.0 - rss / tss : 0.0);
    res.actual = y;
    res.predicted = preds;
    res.featureImportance = gain;
    return res;
}
}

std::vector<BenchmarkResult> BenchmarkEngine::run(const TypedDataset& data, int targetIndex, const std::vector<int>& featureIndices, int kfold) {
    std::vector<BenchmarkResult> out;
    if (targetIndex < 0 || data.columns()[targetIndex].type != ColumnType::NUMERIC) return out;

    const auto& target = std::get<std::vector<double>>(data.columns()[targetIndex].values);
    if (target.empty()) return out;

    Matrix X(data.rowCount(), std::vector<double>(featureIndices.size(), 0.0));
    for (size_t j = 0; j < featureIndices.size(); ++j) {
        int idx = featureIndices[j];
        if (idx < 0 || data.columns()[idx].type != ColumnType::NUMERIC) continue;
        const auto& col = std::get<std::vector<double>>(data.columns()[idx].values);
        for (size_t i = 0; i < data.rowCount(); ++i) X[i][j] = col[i];
    }

    out.push_back(evalLinear("LinearRegression", X, target, kfold, 0.0));
    out.push_back(evalLinear("RidgeRegression", X, target, kfold, 0.5));
    out.push_back(evalTreeStump(X, target, kfold));

    // crude binary accuracy if target looks binary
    bool isBinary = true;
    for (double v : target) {
        if (std::abs(v) > 1e-9 && std::abs(v - 1.0) > 1e-9) { isBinary = false; break; }
    }
    if (isBinary) {
        for (auto& r : out) {
            size_t ok = 0;
            for (size_t i = 0; i < r.actual.size() && i < r.predicted.size(); ++i) {
                int pred = r.predicted[i] >= 0.5 ? 1 : 0;
                int act = r.actual[i] >= 0.5 ? 1 : 0;
                if (pred == act) ok++;
            }
            r.accuracy = r.actual.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(r.actual.size());
        }
    }

    return out;
}
