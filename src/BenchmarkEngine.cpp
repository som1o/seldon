#include "BenchmarkEngine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace {
using Matrix = std::vector<std::vector<double>>;

struct FeatureScaler {
    std::vector<double> mean;
    std::vector<double> stddev;
};

FeatureScaler fitFeatureScaler(const Matrix& X) {
    FeatureScaler scaler;
    if (X.empty()) return scaler;

    size_t p = X[0].size();
    scaler.mean.assign(p, 0.0);
    scaler.stddev.assign(p, 0.0);

    for (const auto& row : X) {
        for (size_t j = 0; j < p; ++j) scaler.mean[j] += row[j];
    }
    for (size_t j = 0; j < p; ++j) scaler.mean[j] /= static_cast<double>(X.size());

    for (const auto& row : X) {
        for (size_t j = 0; j < p; ++j) {
            double d = row[j] - scaler.mean[j];
            scaler.stddev[j] += d * d;
        }
    }
    for (size_t j = 0; j < p; ++j) {
        scaler.stddev[j] = std::sqrt(scaler.stddev[j] / static_cast<double>(std::max<size_t>(1, X.size() - 1)));
        if (scaler.stddev[j] < 1e-12) scaler.stddev[j] = 1.0;
    }
    return scaler;
}

Matrix applyFeatureScaler(const Matrix& X, const FeatureScaler& scaler) {
    Matrix out = X;
    for (auto& row : out) {
        for (size_t j = 0; j < row.size() && j < scaler.mean.size(); ++j) {
            row[j] = (row[j] - scaler.mean[j]) / scaler.stddev[j];
        }
    }
    return out;
}

std::pair<double, double> fitTargetScaler(const std::vector<double>& y) {
    if (y.empty()) return {0.0, 1.0};
    double mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    double var = 0.0;
    for (double v : y) {
        double d = v - mean;
        var += d * d;
    }
    double stddev = std::sqrt(var / static_cast<double>(std::max<size_t>(1, y.size() - 1)));
    if (stddev < 1e-12) stddev = 1.0;
    return {mean, stddev};
}

std::vector<double> fitLinear(const Matrix& X, const std::vector<double>& y, double lambda = 0.0) {
    if (X.empty()) return {};
    size_t n = X.size();
    size_t p = X[0].size();

    Matrix a(p + 1, std::vector<double>(p + 2, 0.0));
    std::vector<double> row(p + 1, 1.0);
    for (size_t i = 0; i < n; ++i) {
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

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(1337);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<double> preds(n, 0.0);
    for (int f = 0; f < folds; ++f) {
        size_t s = static_cast<size_t>(f) * foldSize;
        size_t e = (f == folds - 1) ? n : std::min(n, s + foldSize);
        size_t testCount = e - s;
        size_t trainCount = n - testCount;

        Matrix Xtr;
        Matrix Xte;
        std::vector<double> ytr;
        std::vector<size_t> testRows;
        Xtr.reserve(trainCount);
        Xte.reserve(testCount);
        ytr.reserve(trainCount);
        testRows.reserve(testCount);
        for (size_t i = 0; i < n; ++i) {
            size_t idx = order[i];
            if (i >= s && i < e) {
                Xte.push_back(X[idx]);
                testRows.push_back(idx);
                continue;
            }
            Xtr.push_back(X[idx]);
            ytr.push_back(y[idx]);
        }

        if (Xtr.empty() || Xte.empty()) continue;

        FeatureScaler xScaler = fitFeatureScaler(Xtr);
        Matrix XtrScaled = applyFeatureScaler(Xtr, xScaler);
        Matrix XteScaled = applyFeatureScaler(Xte, xScaler);

        auto [yMean, yStd] = fitTargetScaler(ytr);
        std::vector<double> ytrScaled = ytr;
        for (double& v : ytrScaled) v = (v - yMean) / yStd;

        auto beta = fitLinear(XtrScaled, ytrScaled, lambda);
        for (size_t i = 0; i < XteScaled.size(); ++i) {
            double predScaled = predictLinear(beta, XteScaled[i]);
            preds[testRows[i]] = predScaled * yStd + yMean;
        }
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

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(1337);
    std::shuffle(order.begin(), order.end(), rng);

    for (int f = 0; f < folds; ++f) {
        size_t s = static_cast<size_t>(f) * foldSize;
        size_t e = (f == folds - 1) ? n : std::min(n, s + foldSize);

        size_t bestFeat = 0;
        double bestThresh = 0.0;
        double bestErr = 1e300;

        for (size_t feat = 0; feat < p; ++feat) {
            std::vector<double> cand;
            cand.reserve(n - (e - s));
            for (size_t i = 0; i < n; ++i) {
                if (i >= s && i < e) continue;
                size_t idx = order[i];
                cand.push_back(X[idx][feat]);
            }
            if (cand.empty()) continue;
            std::nth_element(cand.begin(), cand.begin() + cand.size() / 2, cand.end());
            double thr = cand[cand.size() / 2];

            double lsum = 0, rsum = 0; size_t lc = 0, rc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (i >= s && i < e) continue;
                size_t idx = order[i];
                if (X[idx][feat] <= thr) { lsum += y[idx]; lc++; }
                else { rsum += y[idx]; rc++; }
            }
            double lmean = lc ? lsum / lc : 0.0;
            double rmean = rc ? rsum / rc : 0.0;

            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                if (i >= s && i < e) continue;
                size_t idx = order[i];
                double pred = (X[idx][feat] <= thr) ? lmean : rmean;
                double d = y[idx] - pred;
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
            size_t idx = order[i];
            if (X[idx][bestFeat] <= bestThresh) { lsum += y[idx]; lc++; }
            else { rsum += y[idx]; rc++; }
        }
        double lmean = lc ? lsum / lc : 0.0;
        double rmean = rc ? rsum / rc : 0.0;

        for (size_t i = s; i < e; ++i) {
            size_t idx = order[i];
            preds[idx] = (X[idx][bestFeat] <= bestThresh) ? lmean : rmean;
        }
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

    out.push_back(evalLinear("LinearRegression", X, target, kfold, 1e-6));
    out.push_back(evalLinear("RidgeRegression", X, target, kfold, 1.0));
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
            r.hasAccuracy = true;
        }
    }

    return out;
}
