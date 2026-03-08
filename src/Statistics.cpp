#include "Statistics.h"

#include "AutoConfig.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace {
constexpr size_t kMinAdfLagSamples = 12;
constexpr double kStationaryGammaThreshold = -0.08;
constexpr double kStationaryPValueThreshold = 0.05;
constexpr double kStddevFloor = 1e-9;

std::vector<uint8_t> buildFiniteMask(const std::vector<double>& values) {
    std::vector<uint8_t> mask(values.size(), static_cast<uint8_t>(0));
    for (size_t i = 0; i < values.size(); ++i) {
        mask[i] = std::isfinite(values[i]) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }
    return mask;
}

double safeLog2(double p) {
    if (p <= 1e-12) return 0.0;
    return std::log(p) / std::log(2.0);
}

std::vector<size_t> quantileBinIndex(const std::vector<double>& values, size_t bins) {
    std::vector<size_t> out(values.size(), 0);
    if (values.empty() || bins < 2) return out;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    std::vector<double> cuts;
    cuts.reserve(bins - 1);
    for (size_t q = 1; q < bins; ++q) {
        const double pos = (static_cast<double>(q) / static_cast<double>(bins)) * static_cast<double>(sorted.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        const double w = pos - static_cast<double>(lo);
        const double cut = sorted[lo] * (1.0 - w) + sorted[hi] * w;
        cuts.push_back(cut);
    }

    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<size_t>(std::upper_bound(cuts.begin(), cuts.end(), values[i]) - cuts.begin());
    }
    return out;
}

double normalTail2SidedApprox(double zAbs) {
    if (!std::isfinite(zAbs)) return 1.0;
    return std::erfc(zAbs / std::sqrt(2.0));
}
} // namespace

ColumnStats Statistics::calculateStats(const std::vector<double>& col) {
    ColumnStats stats{0, 0, 0, 0, 0, 0};
    if (col.empty()) return stats;

    const std::vector<uint8_t> finiteMask = buildFiniteMask(col);

    double mean = 0.0;
    double m2 = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < col.size(); ++i) {
        if (!finiteMask[i]) continue;
        const double value = col[i];
        ++count;
        double delta = value - mean;
        mean += delta / static_cast<double>(count);
        double delta2 = value - mean;
        m2 += delta * delta2;
    }
    if (count == 0) return stats;

    const size_t n = count;
    stats.mean = mean;
    stats.variance = (count > 1) ? (m2 / static_cast<double>(count - 1)) : 0.0;
    stats.stddev = std::sqrt(stats.variance);

    std::vector<double> medianWork;
    medianWork.reserve(n);
    for (size_t i = 0; i < col.size(); ++i) {
        if (finiteMask[i]) medianWork.push_back(col[i]);
    }
    size_t mid = n / 2;
    std::nth_element(medianWork.begin(), medianWork.begin() + mid, medianWork.end());
    double upper = medianWork[mid];
    if (n % 2 == 0) {
        std::nth_element(medianWork.begin(), medianWork.begin() + (mid - 1), medianWork.begin() + mid);
        stats.median = (medianWork[mid - 1] + upper) / 2.0;
    } else {
        stats.median = upper;
    }

    if (n > 2 && stats.stddev > 0) {
        double m3 = 0, m4 = 0;
        for (size_t i = 0; i < col.size(); ++i) {
            if (!finiteMask[i]) continue;
            const double val = col[i];
            double diff = val - stats.mean;
            double diff2 = diff * diff;
            m3 += diff2 * diff;
            m4 += diff2 * diff2;
        }

        double term1 = static_cast<double>(n) / ((n - 1) * (n - 2));
        double stddev2 = stats.stddev * stats.stddev;
        double stddev3 = stddev2 * stats.stddev;
        stats.skewness = term1 * (m3 / stddev3);

        if (n > 3) {
            double termK1 = (static_cast<double>(n) * (n + 1)) / ((n - 1.0) * (n - 2.0) * (n - 3.0));
            double nMinus1 = (n - 1.0);
            double termK2 = (3.0 * nMinus1 * nMinus1) / ((n - 2.0) * (n - 3.0));
            double stddev4 = stddev2 * stddev2;
            stats.kurtosis = termK1 * (m4 / stddev4) - termK2;
        }
    }

    return stats;
}

StationarityDiagnostic Statistics::adfStyleDrift(const std::vector<double>& series,
                                                 const std::vector<double>& axis,
                                                 const std::string& featureName,
                                                 const std::string& axisName) {
    StationarityDiagnostic out;
    out.feature = featureName;
    out.axis = axisName;

    const size_t n = std::min(series.size(), axis.size());
    if (n < 16) {
        out.verdict = "insufficient";
        return out;
    }

    std::vector<size_t> order;
    order.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(series[i]) || !std::isfinite(axis[i])) continue;
        order.push_back(i);
    }
    if (order.size() < 16) {
        out.verdict = "insufficient";
        return out;
    }

    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (axis[lhs] == axis[rhs]) return series[lhs] < series[rhs];
        return axis[lhs] < axis[rhs];
    });

    std::vector<double> lag;
    std::vector<double> delta;
    std::vector<double> yVals;
    lag.reserve(order.size() - 1);
    delta.reserve(order.size() - 1);
    yVals.reserve(order.size());
    if (!order.empty()) yVals.push_back(series[order.front()]);
    for (size_t i = 1; i < order.size(); ++i) {
        const double y0 = series[order[i - 1]];
        const double y1 = series[order[i]];
        lag.push_back(y0);
        delta.push_back(y1 - y0);
        yVals.push_back(y1);
    }

    out.samples = lag.size();
    if (lag.size() < kMinAdfLagSamples) {
        out.verdict = "insufficient";
        return out;
    }

    const double xMean = std::accumulate(lag.begin(), lag.end(), 0.0) / static_cast<double>(lag.size());
    const double yMean = std::accumulate(delta.begin(), delta.end(), 0.0) / static_cast<double>(delta.size());

    double sxx = 0.0;
    double sxy = 0.0;
    for (size_t i = 0; i < lag.size(); ++i) {
        const double dx = lag[i] - xMean;
        const double dy = delta[i] - yMean;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    if (sxx <= 1e-12) {
        out.verdict = "insufficient";
        return out;
    }

    const double gamma = sxy / sxx;
    const double alpha = yMean - gamma * xMean;
    out.gamma = gamma;

    double rss = 0.0;
    for (size_t i = 0; i < lag.size(); ++i) {
        const double pred = alpha + gamma * lag[i];
        const double e = delta[i] - pred;
        rss += e * e;
    }
    const double dof = static_cast<double>(lag.size() > 2 ? lag.size() - 2 : 1);
    const double sigma2 = rss / dof;
    const double seGamma = std::sqrt(std::max(0.0, sigma2 / sxx));

    if (seGamma > 1e-12) {
        out.tStatistic = gamma / seGamma;
        out.pApprox = normalTail2SidedApprox(std::abs(out.tStatistic));
    } else {
        out.tStatistic = 0.0;
        out.pApprox = 1.0;
    }

    const size_t q = std::max<size_t>(3, lag.size() / 4);
    double earlyMean = 0.0;
    double lateMean = 0.0;
    for (size_t i = 0; i < q; ++i) earlyMean += lag[i];
    for (size_t i = lag.size() - q; i < lag.size(); ++i) lateMean += lag[i];
    earlyMean /= static_cast<double>(q);
    lateMean /= static_cast<double>(q);

    const ColumnStats ys = Statistics::calculateStats(yVals);
    const double scale = std::max(kStddevFloor, ys.stddev);
    out.driftRatio = std::abs(lateMean - earlyMean) / scale;

    const HeuristicTuningConfig tuning{};
    const bool weakMeanReversion = gamma > tuning.stationarityWeakMeanReversionGamma;
    const bool notSignificant = out.pApprox > tuning.stationarityNotSignificantPValue;
    const bool sizableDrift = out.driftRatio >= tuning.stationaritySizableDriftRatio;
    out.nonStationary = weakMeanReversion && (notSignificant || sizableDrift);

    if (out.nonStationary) {
        out.verdict = "non-stationary";
    } else if (gamma <= kStationaryGammaThreshold && out.pApprox <= kStationaryPValueThreshold && out.driftRatio < 0.25) {
        out.verdict = "stationary";
    } else {
        out.verdict = "borderline";
    }
    return out;
}

AsymmetricDirectionScore Statistics::asymmetricInformationGain(const std::vector<double>& x,
                                                               const std::vector<double>& y,
                                                               size_t bins) {
    AsymmetricDirectionScore out;
    const size_t n = std::min(x.size(), y.size());
    if (n < 20 || bins < 2) return out;

    std::vector<double> xa;
    std::vector<double> ya;
    xa.reserve(n);
    ya.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        xa.push_back(x[i]);
        ya.push_back(y[i]);
    }
    if (xa.size() < 20) return out;

    bins = std::max<size_t>(2, std::min<size_t>(bins, 16));
    const auto xBin = quantileBinIndex(xa, bins);
    const auto yBin = quantileBinIndex(ya, bins);

    std::vector<double> px(bins, 0.0);
    std::vector<double> py(bins, 0.0);
    std::vector<std::vector<double>> pxy(bins, std::vector<double>(bins, 0.0));
    const double invN = 1.0 / static_cast<double>(xa.size());
    for (size_t i = 0; i < xa.size(); ++i) {
        px[xBin[i]] += invN;
        py[yBin[i]] += invN;
        pxy[xBin[i]][yBin[i]] += invN;
    }

    auto entropy = [&](const std::vector<double>& p) {
        double h = 0.0;
        for (double v : p) {
            if (v <= 1e-12) continue;
            h -= v * safeLog2(v);
        }
        return h;
    };

    const double hx = entropy(px);
    const double hy = entropy(py);

    double hyGivenX = 0.0;
    for (size_t xb = 0; xb < bins; ++xb) {
        if (px[xb] <= 1e-12) continue;
        double h = 0.0;
        for (size_t yb = 0; yb < bins; ++yb) {
            const double p = pxy[xb][yb] / px[xb];
            if (p <= 1e-12) continue;
            h -= p * safeLog2(p);
        }
        hyGivenX += px[xb] * h;
    }

    double hxGivenY = 0.0;
    for (size_t yb = 0; yb < bins; ++yb) {
        if (py[yb] <= 1e-12) continue;
        double h = 0.0;
        for (size_t xb = 0; xb < bins; ++xb) {
            const double p = pxy[xb][yb] / py[yb];
            if (p <= 1e-12) continue;
            h -= p * safeLog2(p);
        }
        hxGivenY += py[yb] * h;
    }

    const double igXtoY = std::max(0.0, hy - hyGivenX);
    const double igYtoX = std::max(0.0, hx - hxGivenY);
    out.igXtoY = igXtoY;
    out.igYtoX = igYtoX;
    const double scale = std::max(1e-12, std::max(igXtoY, igYtoX));
    out.asymmetry = (igXtoY - igYtoX) / scale;

    const double eps = 1e-4;
    if (out.asymmetry > eps) out.suggestedDirection = "x->y";
    else if (out.asymmetry < -eps) out.suggestedDirection = "y->x";
    else out.suggestedDirection = "undirected";

    return out;
}
