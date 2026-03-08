#include "BenchmarkEngine.h"
#include "MathUtils.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <numeric>
#include <random>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
using Matrix = std::vector<std::vector<double>>;

struct FeatureScaler {
    std::vector<double> mean;
    std::vector<double> stddev;
};

bool isIntegerLike(double v) {
    return std::isfinite(v) && std::abs(v - std::round(v)) <= 1e-9;
}

bool shouldUseStratifiedFolds(const std::vector<double>& y) {
    if (y.size() < 20) return false;
    std::unordered_map<long long, size_t> counts;
    counts.reserve(y.size());
    for (double v : y) {
        if (!isIntegerLike(v)) return false;
        counts[static_cast<long long>(std::llround(v))] += 1;
    }
    if (counts.size() < 2) return false;
    const size_t maxClasses = std::min<size_t>(20, std::max<size_t>(2, y.size() / 8));
    return counts.size() <= maxClasses;
}

std::vector<int> buildFoldAssignment(const std::vector<double>& y, int folds, uint32_t seed) {
    const size_t n = y.size();
    std::vector<int> foldOf(n, 0);
    if (n == 0 || folds <= 1) return foldOf;

    std::mt19937 rng(seed);
    if (!shouldUseStratifiedFolds(y)) {
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t rank = 0; rank < n; ++rank) {
            foldOf[order[rank]] = static_cast<int>(rank % static_cast<size_t>(folds));
        }
        return foldOf;
    }

    std::unordered_map<long long, std::vector<size_t>> classRows;
    classRows.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const long long label = static_cast<long long>(std::llround(y[i]));
        classRows[label].push_back(i);
    }

    for (auto& kv : classRows) {
        auto& rows = kv.second;
        std::shuffle(rows.begin(), rows.end(), rng);
        for (size_t rank = 0; rank < rows.size(); ++rank) {
            foldOf[rows[rank]] = static_cast<int>(rank % static_cast<size_t>(folds));
        }
    }
    return foldOf;
}

FeatureScaler fitFeatureScaler(const Matrix& X, const std::vector<size_t>& trainRows) {
    FeatureScaler scaler;
    if (X.empty() || trainRows.empty()) return scaler;

    size_t p = X[0].size();
    scaler.mean.assign(p, 0.0);
    scaler.stddev.assign(p, 0.0);

    for (size_t idx : trainRows) {
        const auto& row = X[idx];
        for (size_t j = 0; j < p; ++j) scaler.mean[j] += row[j];
    }
    for (size_t j = 0; j < p; ++j) scaler.mean[j] /= static_cast<double>(trainRows.size());

    for (size_t idx : trainRows) {
        const auto& row = X[idx];
        for (size_t j = 0; j < p; ++j) {
            double d = row[j] - scaler.mean[j];
            scaler.stddev[j] += d * d;
        }
    }
    for (size_t j = 0; j < p; ++j) {
        scaler.stddev[j] = std::sqrt(scaler.stddev[j] / static_cast<double>(std::max<size_t>(1, trainRows.size() - 1)));
        if (scaler.stddev[j] < 1e-12) scaler.stddev[j] = 1.0;
    }
    return scaler;
}

void scaleRowInPlace(std::vector<double>& row, const FeatureScaler& scaler) {
    for (size_t j = 0; j < row.size() && j < scaler.mean.size(); ++j) {
        row[j] = (row[j] - scaler.mean[j]) / scaler.stddev[j];
    }
}

void gatherAndScaleRows(const Matrix& X,
                        const std::vector<size_t>& rows,
                        const FeatureScaler& scaler,
                        Matrix& out) {
    out.clear();
    out.reserve(rows.size());
    for (size_t idx : rows) {
        std::vector<double> row = X[idx];
        scaleRowInPlace(row, scaler);
        out.push_back(std::move(row));
    }
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

std::vector<double> fitLinear(Matrix X, std::vector<double> y, double lambda = 0.0) {
    if (X.empty()) return {};

    const size_t n = X.size();
    const size_t p = X[0].size();
    const bool ridge = (lambda > 0.0);
    const size_t extraRows = ridge ? p : 0;

    MathUtils::Matrix design(n + extraRows, p + 1);
    MathUtils::Matrix target(n + extraRows, 1);

    for (size_t i = 0; i < n; ++i) {
        design.data[i][0] = 1.0;
        for (size_t j = 0; j < p; ++j) {
            design.data[i][j + 1] = X[i][j];
        }
        target.data[i][0] = y[i];
    }

    if (ridge) {
        const double penalty = std::sqrt(lambda);
        for (size_t j = 0; j < p; ++j) {
            const size_t row = n + j;
            design.data[row][j + 1] = penalty;
            target.data[row][0] = 0.0;
        }
    }

    return MathUtils::multipleLinearRegression(std::move(design), std::move(target));
}

double predictLinear(const std::vector<double>& beta, const std::vector<double>& x) {
    double y = beta.empty() ? 0.0 : beta[0];
    for (size_t i = 0; i < x.size() && (i + 1) < beta.size(); ++i) y += beta[i + 1] * x[i];
    return y;
}

BenchmarkResult evalLinear(const std::string& name, const Matrix& X, const std::vector<double>& y, int kfold, uint32_t seed, double lambda = 0.0) {
    BenchmarkResult res;
    res.model = name;
    if (X.empty()) return res;

    size_t n = X.size();
    int folds = std::max(2, std::min<int>(kfold, static_cast<int>(n)));
    const std::vector<int> foldOf = buildFoldAssignment(y, folds, seed);

    std::vector<double> preds(n, 0.0);
    Matrix XtrScaled;
    Matrix XteScaled;
    std::vector<double> ytr;
    std::vector<double> ytrScaled;
    std::vector<size_t> testRows;
    std::vector<size_t> trainRows;
    for (int f = 0; f < folds; ++f) {
        ytr.clear();
        testRows.clear();
        trainRows.clear();
        ytr.reserve(n);
        testRows.reserve(n / static_cast<size_t>(folds) + 1);
        trainRows.reserve(n);
        for (size_t idx = 0; idx < n; ++idx) {
            if (foldOf[idx] == f) {
                testRows.push_back(idx);
                continue;
            }
            trainRows.push_back(idx);
            ytr.push_back(y[idx]);
        }

        if (trainRows.empty() || testRows.empty()) continue;

        FeatureScaler xScaler = fitFeatureScaler(X, trainRows);
        gatherAndScaleRows(X, trainRows, xScaler, XtrScaled);
        gatherAndScaleRows(X, testRows, xScaler, XteScaled);

        auto [yMean, yStd] = fitTargetScaler(ytr);
        ytrScaled = ytr;
        for (double& v : ytrScaled) v = (v - yMean) / yStd;

        auto beta = fitLinear(std::move(XtrScaled), std::move(ytrScaled), lambda);
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

BenchmarkResult evalTreeStump(const Matrix& X, const std::vector<double>& y, int kfold, uint32_t seed) {
    BenchmarkResult res;
    res.model = "DecisionTreeStump";
    if (X.empty() || X[0].empty()) return res;

    size_t n = X.size();
    size_t p = X[0].size();
    std::vector<double> preds(n, 0.0);

    int folds = std::max(2, std::min<int>(kfold, static_cast<int>(n)));
    const std::vector<int> foldOf = buildFoldAssignment(y, folds, seed);

    for (int f = 0; f < folds; ++f) {
        size_t bestFeat = 0;
        double bestThresh = 0.0;
        double bestErr = 1e300;

        for (size_t feat = 0; feat < p; ++feat) {
            std::vector<double> cand;
            cand.reserve(n);
            for (size_t idx = 0; idx < n; ++idx) {
                if (foldOf[idx] == f) continue;
                cand.push_back(X[idx][feat]);
            }
            if (cand.empty()) continue;
            std::nth_element(cand.begin(), cand.begin() + cand.size() / 2, cand.end());
            double thr = cand[cand.size() / 2];

            double lsum = 0, rsum = 0; size_t lc = 0, rc = 0;
            for (size_t idx = 0; idx < n; ++idx) {
                if (foldOf[idx] == f) continue;
                if (X[idx][feat] <= thr) { lsum += y[idx]; lc++; }
                else { rsum += y[idx]; rc++; }
            }
            double lmean = lc ? lsum / lc : 0.0;
            double rmean = rc ? rsum / rc : 0.0;

            double err = 0.0;
            for (size_t idx = 0; idx < n; ++idx) {
                if (foldOf[idx] == f) continue;
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

        double lsum = 0, rsum = 0; size_t lc = 0, rc = 0;
        for (size_t idx = 0; idx < n; ++idx) {
            if (foldOf[idx] == f) continue;
            if (X[idx][bestFeat] <= bestThresh) { lsum += y[idx]; lc++; }
            else { rsum += y[idx]; rc++; }
        }
        double lmean = lc ? lsum / lc : 0.0;
        double rmean = rc ? rsum / rc : 0.0;

        for (size_t idx = 0; idx < n; ++idx) {
            if (foldOf[idx] != f) continue;
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
    res.featureImportance.clear();
    return res;
}

BenchmarkResult evalBaggedTreeStumps(const Matrix& X, const std::vector<double>& y, int kfold, uint32_t seed, size_t bags = 25) {
    BenchmarkResult res;
    res.model = "BaggedTreeStumps";
    if (X.empty() || X[0].empty()) return res;

    const size_t n = X.size();
    const size_t p = X[0].size();
    std::vector<double> preds(n, 0.0);

    int folds = std::max(2, std::min<int>(kfold, static_cast<int>(n)));
    const std::vector<int> foldOf = buildFoldAssignment(y, folds, seed);

    for (int f = 0; f < folds; ++f) {
        std::vector<size_t> trainRows;
        std::vector<size_t> testRows;
        trainRows.reserve(n);
        testRows.reserve(n / static_cast<size_t>(folds) + 1);
        for (size_t i = 0; i < n; ++i) {
            if (foldOf[i] == f) testRows.push_back(i);
            else trainRows.push_back(i);
        }
        if (trainRows.empty() || testRows.empty()) continue;

        std::vector<double> foldPred(testRows.size(), 0.0);
        std::mt19937 rng(seed ^ static_cast<uint32_t>(0x9E3779B9u + static_cast<uint32_t>(f * 31)));
        std::uniform_int_distribution<size_t> rowDist(0, trainRows.size() - 1);

        for (size_t b = 0; b < std::max<size_t>(1, bags); ++b) {
            std::vector<size_t> sampled;
            sampled.reserve(trainRows.size());
            for (size_t k = 0; k < trainRows.size(); ++k) sampled.push_back(trainRows[rowDist(rng)]);

            size_t bestFeat = 0;
            double bestThresh = 0.0;
            double bestErr = std::numeric_limits<double>::infinity();

            for (size_t feat = 0; feat < p; ++feat) {
                std::vector<double> cand;
                cand.reserve(sampled.size());
                for (size_t idx : sampled) cand.push_back(X[idx][feat]);
                if (cand.empty()) continue;
                std::nth_element(cand.begin(), cand.begin() + cand.size() / 2, cand.end());
                const double thr = cand[cand.size() / 2];

                double lsum = 0.0, rsum = 0.0;
                size_t lc = 0, rc = 0;
                for (size_t idx : sampled) {
                    if (X[idx][feat] <= thr) { lsum += y[idx]; ++lc; }
                    else { rsum += y[idx]; ++rc; }
                }
                const double lmean = lc ? lsum / static_cast<double>(lc) : 0.0;
                const double rmean = rc ? rsum / static_cast<double>(rc) : 0.0;

                double err = 0.0;
                for (size_t idx : sampled) {
                    const double pred = (X[idx][feat] <= thr) ? lmean : rmean;
                    const double d = y[idx] - pred;
                    err += d * d;
                }
                if (err < bestErr) {
                    bestErr = err;
                    bestFeat = feat;
                    bestThresh = thr;
                }
            }

            double lsum = 0.0, rsum = 0.0;
            size_t lc = 0, rc = 0;
            for (size_t idx : sampled) {
                if (X[idx][bestFeat] <= bestThresh) { lsum += y[idx]; ++lc; }
                else { rsum += y[idx]; ++rc; }
            }
            const double lmean = lc ? lsum / static_cast<double>(lc) : 0.0;
            const double rmean = rc ? rsum / static_cast<double>(rc) : 0.0;

            for (size_t i = 0; i < testRows.size(); ++i) {
                foldPred[i] += (X[testRows[i]][bestFeat] <= bestThresh) ? lmean : rmean;
            }
        }

        const double invBags = 1.0 / static_cast<double>(std::max<size_t>(1, bags));
        for (size_t i = 0; i < testRows.size(); ++i) preds[testRows[i]] = foldPred[i] * invBags;
    }

    double mse = 0.0;
    const double mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    double tss = 0.0, rss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = y[i] - preds[i];
        mse += d * d;
        rss += d * d;
        const double t = y[i] - mean;
        tss += t * t;
    }

    res.rmse = std::sqrt(mse / static_cast<double>(n));
    res.r2 = (tss > 0.0) ? 1.0 - (rss / tss) : 0.0;
    res.actual = y;
    res.predicted = preds;
    return res;
}
}

std::vector<BenchmarkResult> BenchmarkEngine::run(const TypedDataset& data, int targetIndex, const std::vector<int>& featureIndices, int kfold, uint32_t seed) {
    std::vector<BenchmarkResult> out;
    if (targetIndex < 0 || data.columns()[targetIndex].type != ColumnType::NUMERIC) return out;

    const auto& target = std::get<std::vector<double>>(data.columns()[targetIndex].values);
    if (target.empty()) return out;
    if (target.size() < 2) {
        BenchmarkResult tiny;
        tiny.model = "InsufficientRows";
        tiny.rmse = 0.0;
        tiny.r2 = 0.0;
        tiny.actual = target;
        tiny.predicted = target;
        out.push_back(std::move(tiny));
        return out;
    }

    Matrix X(data.rowCount(), std::vector<double>(featureIndices.size(), 0.0));
    for (size_t j = 0; j < featureIndices.size(); ++j) {
        int idx = featureIndices[j];
        if (idx < 0 || data.columns()[idx].type != ColumnType::NUMERIC) continue;
        const auto& col = std::get<std::vector<double>>(data.columns()[idx].values);
        for (size_t i = 0; i < data.rowCount(); ++i) X[i][j] = col[i];
    }

    BenchmarkResult linear;
    BenchmarkResult ridge;
    BenchmarkResult stump;
    BenchmarkResult baggedStump;

    #ifdef USE_OPENMP
    #pragma omp parallel sections default(none) shared(linear, ridge, stump, baggedStump, X, target, kfold, seed)
    {
        #pragma omp section
        { linear = evalLinear("LinearRegression", X, target, kfold, seed, 1e-6); }
        #pragma omp section
        { ridge = evalLinear("RidgeRegression", X, target, kfold, seed ^ 0x9e3779b9U, 1.0); }
        #pragma omp section
        { stump = evalTreeStump(X, target, kfold, seed ^ 0x85ebca6bU); }
        #pragma omp section
        { baggedStump = evalBaggedTreeStumps(X, target, kfold, seed ^ 0xc2b2ae35U, 25); }
    }
    #else
    linear = evalLinear("LinearRegression", X, target, kfold, seed, 1e-6);
    ridge = evalLinear("RidgeRegression", X, target, kfold, seed ^ 0x9e3779b9U, 1.0);
    stump = evalTreeStump(X, target, kfold, seed ^ 0x85ebca6bU);
    baggedStump = evalBaggedTreeStumps(X, target, kfold, seed ^ 0xc2b2ae35U, 25);
    #endif

    out.push_back(std::move(linear));
    out.push_back(std::move(ridge));
    out.push_back(std::move(stump));
    out.push_back(std::move(baggedStump));

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

MultiTargetBenchmarkSummary BenchmarkEngine::runMultiTarget(const TypedDataset& data,
                                                            const std::vector<int>& targetIndices,
                                                            const std::vector<int>& featureIndices,
                                                            int kfold,
                                                            uint32_t seed) {
    MultiTargetBenchmarkSummary summary;
    if (targetIndices.empty()) return summary;

    std::vector<int> uniqueTargets;
    uniqueTargets.reserve(targetIndices.size());
    for (int idx : targetIndices) {
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
        if (std::find(uniqueTargets.begin(), uniqueTargets.end(), idx) != uniqueTargets.end()) continue;
        uniqueTargets.push_back(idx);
    }
    if (uniqueTargets.empty()) return summary;

    summary.targetIndices = uniqueTargets;
    summary.targetNames.reserve(uniqueTargets.size());
    summary.perTargetResults.reserve(uniqueTargets.size());

    struct AggregateState {
        std::string model;
        double rmseSum = 0.0;
        double r2Sum = 0.0;
        size_t metricCount = 0;
        double accuracySum = 0.0;
        size_t accuracyCount = 0;
    };
    std::unordered_map<std::string, AggregateState> aggregate;

    for (size_t i = 0; i < uniqueTargets.size(); ++i) {
        const int targetIdx = uniqueTargets[i];
        summary.targetNames.push_back(data.columns()[static_cast<size_t>(targetIdx)].name);
        const uint32_t targetSeed = seed ^ static_cast<uint32_t>(0x9e3779b9U + static_cast<uint32_t>(targetIdx * 31) + static_cast<uint32_t>(i * 17));
        std::vector<BenchmarkResult> perTarget = run(data, targetIdx, featureIndices, kfold, targetSeed);

        for (const auto& result : perTarget) {
            auto it = aggregate.find(result.model);
            if (it == aggregate.end()) {
                AggregateState init;
                init.model = result.model;
                it = aggregate.emplace(result.model, std::move(init)).first;
            }

            if (std::isfinite(result.rmse)) {
                it->second.rmseSum += result.rmse;
                it->second.metricCount += 1;
            }
            if (std::isfinite(result.r2)) {
                it->second.r2Sum += result.r2;
            }
            if (result.hasAccuracy && std::isfinite(result.accuracy)) {
                it->second.accuracySum += result.accuracy;
                it->second.accuracyCount += 1;
            }
        }

        summary.perTargetResults.push_back(std::move(perTarget));
    }

    summary.aggregateByModel.reserve(aggregate.size());
    for (const auto& kv : aggregate) {
        const auto& state = kv.second;
        if (state.metricCount == 0) continue;
        BenchmarkResult agg;
        agg.model = state.model;
        agg.rmse = state.rmseSum / static_cast<double>(state.metricCount);
        agg.r2 = state.r2Sum / static_cast<double>(state.metricCount);
        agg.hasAccuracy = state.accuracyCount > 0;
        agg.accuracy = agg.hasAccuracy
            ? (state.accuracySum / static_cast<double>(state.accuracyCount))
            : 0.0;
        summary.aggregateByModel.push_back(std::move(agg));
    }

    std::sort(summary.aggregateByModel.begin(), summary.aggregateByModel.end(), [](const BenchmarkResult& a, const BenchmarkResult& b) {
        if (a.rmse == b.rmse) return a.r2 > b.r2;
        return a.rmse < b.rmse;
    });

    return summary;
}
