#include "Preprocessor.h"
#include "CommonUtils.h"
#include "MathUtils.h"
#include "SeldonExceptions.h"
#include "Statistics.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <iterator>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <unordered_map>
#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
using NumVec = std::vector<double>;
using StrVec = std::vector<std::string>;
using TimeVec = std::vector<int64_t>;
using MissingMask = ::MissingMask;

constexpr double kNumericEpsilon = 1e-12;
constexpr double kNeighborDistanceFloor = 1e-9;
constexpr double kModifiedZScaleFactor = 0.6745;
constexpr double kMatrixCompletionBlend = 0.5;
std::recursive_mutex gImputationCrossColumnMutex;

enum class UnitSemanticKind {
    UNKNOWN,
    CURRENCY,
    PERCENT,
    RATIO,
    RATE,
    COUNT,
    DURATION,
    DISTANCE,
    MASS,
    VOLUME,
    TEMPERATURE,
    SCORE
};

UnitSemanticKind inferUnitSemanticKind(const std::string& name) {
    const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
    if (lower.empty()) return UnitSemanticKind::UNKNOWN;

    auto hasAny = [&](std::initializer_list<const char*> tokens) {
        for (const char* token : tokens) {
            if (lower.find(token) != std::string::npos) return true;
        }
        return false;
    };

    if (hasAny({"price", "cost", "revenue", "salary", "income", "expense", "budget", "usd", "eur", "gbp", "jpy", "amount", "payment"})) {
        return UnitSemanticKind::CURRENCY;
    }
    if (hasAny({"percent", "percentage", "pct", "%"})) {
        return UnitSemanticKind::PERCENT;
    }
    if (hasAny({"ratio"})) {
        return UnitSemanticKind::RATIO;
    }
    if (hasAny({"rate", "per_", "per ", "/"})) {
        return UnitSemanticKind::RATE;
    }
    if (hasAny({"count", "qty", "quantity", "num", "number", "users", "cases", "population", "clicks", "visits"})) {
        return UnitSemanticKind::COUNT;
    }
    if (hasAny({"duration", "latency", "time", "seconds", "second", "minutes", "minute", "hours", "hour", "days", "day", "ms"})) {
        return UnitSemanticKind::DURATION;
    }
    if (hasAny({"distance", "km", "kilometer", "mile", "meter", "meters", "miles"})) {
        return UnitSemanticKind::DISTANCE;
    }
    if (hasAny({"weight", "mass", "kg", "kilogram", "lb", "pound", "grams", "gram"})) {
        return UnitSemanticKind::MASS;
    }
    if (hasAny({"volume", "litre", "liter", "ml", "gallon"})) {
        return UnitSemanticKind::VOLUME;
    }
    if (hasAny({"temp", "temperature", "celsius", "fahrenheit", "kelvin"})) {
        return UnitSemanticKind::TEMPERATURE;
    }
    if (hasAny({"score", "index", "rating"})) {
        return UnitSemanticKind::SCORE;
    }
    return UnitSemanticKind::UNKNOWN;
}

bool isDimensionlessUnit(UnitSemanticKind kind) {
    return kind == UnitSemanticKind::PERCENT ||
           kind == UnitSemanticKind::RATIO ||
           kind == UnitSemanticKind::SCORE;
}

bool areUnitsRatioCompatible(UnitSemanticKind a, UnitSemanticKind b) {
    if (a == UnitSemanticKind::UNKNOWN || b == UnitSemanticKind::UNKNOWN) return true;
    if (a == b) return true;
    if (isDimensionlessUnit(a) || isDimensionlessUnit(b)) return true;

    const bool currencyPerCount =
        (a == UnitSemanticKind::CURRENCY && b == UnitSemanticKind::COUNT) ||
        (b == UnitSemanticKind::CURRENCY && a == UnitSemanticKind::COUNT);
    if (currencyPerCount) return true;

    const bool distancePerDuration =
        (a == UnitSemanticKind::DISTANCE && b == UnitSemanticKind::DURATION) ||
        (b == UnitSemanticKind::DISTANCE && a == UnitSemanticKind::DURATION);
    if (distancePerDuration) return true;

    const bool countPerDuration =
        (a == UnitSemanticKind::COUNT && b == UnitSemanticKind::DURATION) ||
        (b == UnitSemanticKind::COUNT && a == UnitSemanticKind::DURATION);
    if (countPerDuration) return true;

    return false;
}

void imputeNumeric(NumVec& values, MissingMask& missing, const std::string& method);

struct NumericSnapshotStore {
    std::vector<size_t> numericIndices;
    std::unordered_map<size_t, size_t> posByColumn;
    std::vector<NumVec> values;
    std::vector<MissingMask> missing;
};

NumericSnapshotStore buildNumericSnapshotStore(const TypedDataset& data) {
    NumericSnapshotStore snapshot;
    snapshot.numericIndices = data.numericColumnIndices();
    snapshot.posByColumn.reserve(snapshot.numericIndices.size());
    snapshot.values.reserve(snapshot.numericIndices.size());
    snapshot.missing.reserve(snapshot.numericIndices.size());

    for (size_t pos = 0; pos < snapshot.numericIndices.size(); ++pos) {
        const size_t colIdx = snapshot.numericIndices[pos];
        snapshot.posByColumn[colIdx] = pos;
        const auto& col = data.columns()[colIdx];
        snapshot.values.push_back(std::get<NumVec>(col.values));
        snapshot.missing.push_back(col.missing);
    }
    return snapshot;
}

const NumVec* snapshotValuesForColumn(const NumericSnapshotStore& snapshot, size_t colIdx) {
    const auto it = snapshot.posByColumn.find(colIdx);
    if (it == snapshot.posByColumn.end()) return nullptr;
    return &snapshot.values[it->second];
}

const MissingMask* snapshotMissingForColumn(const NumericSnapshotStore& snapshot, size_t colIdx) {
    const auto it = snapshot.posByColumn.find(colIdx);
    if (it == snapshot.posByColumn.end()) return nullptr;
    return &snapshot.missing[it->second];
}

void imputeNumericKnnWithSnapshot(NumVec& targetVals,
                                  MissingMask& targetMissing,
                                  const NumericSnapshotStore& snapshot,
                                  size_t targetCol,
                                  size_t k = 5) {
    std::vector<size_t> predictors;
    predictors.reserve(snapshot.numericIndices.size());
    for (size_t idx : snapshot.numericIndices) {
        if (idx == targetCol) continue;
        predictors.push_back(idx);
    }
    if (predictors.empty()) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    for (size_t row = 0; row < targetVals.size() && row < targetMissing.size(); ++row) {
        if (!targetMissing[row] && std::isfinite(targetVals[row])) continue;

        std::vector<std::pair<double, double>> neighbors;
        neighbors.reserve(128);
        for (size_t cand = 0; cand < targetVals.size() && cand < targetMissing.size(); ++cand) {
            if (targetMissing[cand] || !std::isfinite(targetVals[cand])) continue;

            double dist = 0.0;
            size_t overlap = 0;
            for (size_t pIdx : predictors) {
                const NumVec* pVals = snapshotValuesForColumn(snapshot, pIdx);
                const MissingMask* pMissing = snapshotMissingForColumn(snapshot, pIdx);
                if (pVals == nullptr || pMissing == nullptr) continue;
                if (row >= pVals->size() || cand >= pVals->size()) continue;
                if (row < pMissing->size() && (*pMissing)[row]) continue;
                if (cand < pMissing->size() && (*pMissing)[cand]) continue;
                if (!std::isfinite((*pVals)[row]) || !std::isfinite((*pVals)[cand])) continue;
                const double d = (*pVals)[row] - (*pVals)[cand];
                dist += d * d;
                ++overlap;
            }
            if (overlap == 0) continue;
            neighbors.push_back({std::sqrt(dist / static_cast<double>(overlap)), targetVals[cand]});
        }

        if (neighbors.empty()) continue;
        const size_t keep = std::min(k, neighbors.size());
        std::nth_element(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(keep - 1), neighbors.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        double wsum = 0.0;
        double ysum = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            const double w = 1.0 / std::max(kNeighborDistanceFloor, neighbors[i].first);
            wsum += w;
            ysum += w * neighbors[i].second;
        }
        if (wsum > 0.0) {
            targetVals[row] = ysum / wsum;
            targetMissing[row] = static_cast<uint8_t>(0);
        }
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

void imputeNumericMiceWithSnapshot(NumVec& targetVals,
                                   MissingMask& targetMissing,
                                   const NumericSnapshotStore& snapshot,
                                   size_t targetCol) {
    std::vector<size_t> predictors;
    predictors.reserve(snapshot.numericIndices.size());
    for (size_t idx : snapshot.numericIndices) {
        if (idx == targetCol) continue;
        predictors.push_back(idx);
    }
    if (predictors.empty()) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    std::vector<std::pair<size_t, double>> ranked;
    ranked.reserve(predictors.size());
    for (size_t pIdx : predictors) {
        const NumVec* pVals = snapshotValuesForColumn(snapshot, pIdx);
        const MissingMask* pMissing = snapshotMissingForColumn(snapshot, pIdx);
        if (pVals == nullptr || pMissing == nullptr) continue;

        std::vector<double> x;
        std::vector<double> y;
        const size_t n = std::min(targetVals.size(), pVals->size());
        x.reserve(n);
        y.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (i < targetMissing.size() && targetMissing[i]) continue;
            if (i < pMissing->size() && (*pMissing)[i]) continue;
            if (!std::isfinite(targetVals[i]) || !std::isfinite((*pVals)[i])) continue;
            x.push_back((*pVals)[i]);
            y.push_back(targetVals[i]);
        }
        if (x.size() < 12) continue;
        const double r = std::abs(MathUtils::calculatePearson(x, y, Statistics::calculateStats(x), Statistics::calculateStats(y)).value_or(0.0));
        if (std::isfinite(r)) ranked.push_back({pIdx, r});
    }
    if (ranked.empty()) {
        imputeNumericKnnWithSnapshot(targetVals, targetMissing, snapshot, targetCol, 5);
        return;
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    predictors.clear();
    for (size_t i = 0; i < std::min<size_t>(4, ranked.size()); ++i) predictors.push_back(ranked[i].first);

    std::vector<std::vector<double>> rows;
    std::vector<double> ytrain;
    for (size_t i = 0; i < targetVals.size(); ++i) {
        if (i < targetMissing.size() && targetMissing[i]) continue;
        if (!std::isfinite(targetVals[i])) continue;

        std::vector<double> row;
        row.reserve(predictors.size() + 1);
        row.push_back(1.0);
        bool ok = true;
        for (size_t pIdx : predictors) {
            const NumVec* pVals = snapshotValuesForColumn(snapshot, pIdx);
            const MissingMask* pMissing = snapshotMissingForColumn(snapshot, pIdx);
            if (pVals == nullptr || pMissing == nullptr) { ok = false; break; }
            if (i >= pVals->size()) { ok = false; break; }
            if (i < pMissing->size() && (*pMissing)[i]) { ok = false; break; }
            if (!std::isfinite((*pVals)[i])) { ok = false; break; }
            row.push_back((*pVals)[i]);
        }
        if (!ok) continue;
        rows.push_back(std::move(row));
        ytrain.push_back(targetVals[i]);
    }

    if (rows.size() < predictors.size() + 8) {
        imputeNumericKnnWithSnapshot(targetVals, targetMissing, snapshot, targetCol, 5);
        return;
    }

    MathUtils::Matrix X(rows.size(), predictors.size() + 1);
    MathUtils::Matrix Y(rows.size(), 1);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < rows[r].size(); ++c) X.at(r, c) = rows[r][c];
        Y.at(r, 0) = ytrain[r];
    }
    const auto beta = MathUtils::multipleLinearRegression(X, Y);
    if (beta.size() != predictors.size() + 1) {
        imputeNumericKnnWithSnapshot(targetVals, targetMissing, snapshot, targetCol, 5);
        return;
    }

    for (size_t i = 0; i < targetVals.size(); ++i) {
        if (i >= targetMissing.size() || !targetMissing[i]) continue;
        double pred = beta[0];
        bool ok = true;
        for (size_t p = 0; p < predictors.size(); ++p) {
            const size_t pIdx = predictors[p];
            const NumVec* pVals = snapshotValuesForColumn(snapshot, pIdx);
            const MissingMask* pMissing = snapshotMissingForColumn(snapshot, pIdx);
            if (pVals == nullptr || pMissing == nullptr) { ok = false; break; }
            if (i >= pVals->size()) { ok = false; break; }
            if (i < pMissing->size() && (*pMissing)[i]) { ok = false; break; }
            if (!std::isfinite((*pVals)[i])) { ok = false; break; }
            pred += beta[p + 1] * (*pVals)[i];
        }
        if (!ok || !std::isfinite(pred)) continue;
        targetVals[i] = pred;
        targetMissing[i] = static_cast<uint8_t>(0);
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

void imputeNumericMatrixCompletionWithSnapshot(NumVec& targetVals,
                                               MissingMask& targetMissing,
                                               const NumericSnapshotStore& snapshot,
                                               size_t targetCol,
                                               size_t iterations = 3) {
    if (snapshot.numericIndices.size() < 2) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    for (size_t it = 0; it < iterations; ++it) {
        double colMean = 0.0;
        size_t colCount = 0;
        for (size_t r = 0; r < targetVals.size(); ++r) {
            if (r < targetMissing.size() && targetMissing[r]) continue;
            if (!std::isfinite(targetVals[r])) continue;
            colMean += targetVals[r];
            ++colCount;
        }
        colMean = (colCount > 0) ? (colMean / static_cast<double>(colCount)) : 0.0;

        for (size_t r = 0; r < targetVals.size(); ++r) {
            if (r >= targetMissing.size() || !targetMissing[r]) continue;

            double rowMean = 0.0;
            size_t rowCount = 0;
            for (size_t cIdx : snapshot.numericIndices) {
                if (cIdx == targetCol) {
                    if (r < targetMissing.size() && targetMissing[r]) continue;
                    if (!std::isfinite(targetVals[r])) continue;
                    rowMean += targetVals[r];
                    ++rowCount;
                    continue;
                }

                const NumVec* vals = snapshotValuesForColumn(snapshot, cIdx);
                const MissingMask* missing = snapshotMissingForColumn(snapshot, cIdx);
                if (vals == nullptr || missing == nullptr) continue;
                if (r >= vals->size()) continue;
                if (r < missing->size() && (*missing)[r]) continue;
                if (!std::isfinite((*vals)[r])) continue;
                rowMean += (*vals)[r];
                ++rowCount;
            }

            const double blend = (rowCount > 0)
                ? (kMatrixCompletionBlend * colMean + (1.0 - kMatrixCompletionBlend) * (rowMean / static_cast<double>(rowCount)))
                : colMean;
            targetVals[r] = std::isfinite(blend) ? blend : colMean;
            targetMissing[r] = static_cast<uint8_t>(0);
        }
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

int64_t floorDiv(int64_t value, int64_t divisor) {
    int64_t q = value / divisor;
    int64_t r = value % divisor;
    if (r != 0 && ((r > 0) != (divisor > 0))) {
        --q;
    }
    return q;
}

std::tuple<int, unsigned, unsigned> civilFromDays(int64_t z) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int y = static_cast<int>(yoe) + static_cast<int>(era) * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
    return {y, m, d};
}

int dayOfWeekFromUnixSeconds(int64_t unixSeconds) {
    const int64_t days = floorDiv(unixSeconds, 86400);
    int weekday = static_cast<int>((days + 4) % 7);
    if (weekday < 0) weekday += 7;
    return weekday; // 0=Sunday, 6=Saturday
}

std::string uniqueColumnName(const TypedDataset& data, const std::string& base) {
    if (data.findColumnIndex(base) < 0) return base;
    int suffix = 2;
    while (true) {
        std::string candidate = base + "_" + std::to_string(suffix);
        if (data.findColumnIndex(candidate) < 0) return candidate;
        ++suffix;
    }
}

void validateImputationStrategy(const std::string& strategy, ColumnType type, const std::string& columnName) {
    const std::string m = CommonUtils::toLower(strategy);
    bool valid = false;
    if (type == ColumnType::NUMERIC) {
        valid = (m == "auto" || m == "mean" || m == "median" || m == "zero" || m == "interpolate" ||
                 m == "knn" || m == "mice" || m == "matrix_completion");
    } else if (type == ColumnType::CATEGORICAL) {
        valid = (m == "auto" || m == "mode");
    } else {
        valid = (m == "auto" || m == "interpolate");
    }
    if (!valid) {
        throw Seldon::ConfigurationException("Invalid imputation strategy for column '" + columnName + "': " + strategy);
    }
}

std::string sanitizeTokenForColumn(std::string_view token) {
    std::string out;
    out.reserve(token.size());
    bool prevUnderscore = false;
    for (char ch : token) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch)) {
            out.push_back(static_cast<char>(std::tolower(uch)));
            prevUnderscore = false;
        } else if (!prevUnderscore) {
            out.push_back('_');
            prevUnderscore = true;
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.empty()) out = "token";
    if (out.size() > 24) out.resize(24);
    return out;
}

std::vector<std::string> splitTokensBySeparator(const std::string& value, char separator) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : value) {
        if (ch == separator) {
            std::string tok = CommonUtils::trim(current);
            if (!tok.empty()) out.push_back(CommonUtils::toLower(tok));
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    std::string tok = CommonUtils::trim(current);
    if (!tok.empty()) out.push_back(CommonUtils::toLower(tok));
    return out;
}

std::string reserveUniqueName(std::unordered_set<std::string>& reserved, const std::string& base) {
    if (reserved.find(base) == reserved.end()) {
        reserved.insert(base);
        return base;
    }
    size_t suffix = 2;
    while (true) {
        std::string candidate = base + "_" + std::to_string(suffix);
        if (reserved.find(candidate) == reserved.end()) {
            reserved.insert(candidate);
            return candidate;
        }
        ++suffix;
    }
}

void addCategoricalSemanticIndicators(TypedDataset& data, const AutoConfig& config) {
    const auto categoricalIndices = data.categoricalColumnIndices();
    if (categoricalIndices.empty()) return;

    std::unordered_set<std::string> reservedNames;
    reservedNames.reserve(data.columns().size() * 2);
    for (const auto& col : data.columns()) reservedNames.insert(col.name);

    std::vector<TypedColumn> engineered;
    engineered.reserve(categoricalIndices.size() * 3);

    constexpr size_t kMaxDerivedGlobal = 64;
    constexpr size_t kMaxTokenIndicatorsPerColumn = 6;

    for (size_t idx : categoricalIndices) {
        if (engineered.size() >= kMaxDerivedGlobal) break;
        const auto& col = data.columns()[idx];
        if (!config.targetColumn.empty() && col.name == config.targetColumn) continue;

        const auto& values = std::get<StrVec>(col.values);
        if (values.empty()) continue;

        std::vector<size_t> observedRows;
        observedRows.reserve(values.size());
        for (size_t r = 0; r < values.size() && r < col.missing.size(); ++r) {
            if (col.missing[r]) continue;
            if (CommonUtils::trim(values[r]).empty()) continue;
            observedRows.push_back(r);
        }
        if (observedRows.size() < 4) continue;

        bool allBooleanLike = true;
        bool seenZero = false;
        bool seenOne = false;
        for (size_t r : observedRows) {
            const auto parsed = CommonUtils::parseBooleanLikeToken(values[r]);
            if (!parsed.has_value()) {
                allBooleanLike = false;
                break;
            }
            if (*parsed < 0.5) seenZero = true;
            else seenOne = true;
        }

        if (allBooleanLike && (seenZero || seenOne)) {
            TypedColumn boolCol;
            boolCol.name = reserveUniqueName(reservedNames, col.name + "_bool");
            boolCol.type = ColumnType::NUMERIC;
            boolCol.values = std::vector<double>(values.size(), 0.0);
            boolCol.missing = MissingMask(values.size(), static_cast<uint8_t>(0));
            auto& out = std::get<NumVec>(boolCol.values);
            for (size_t r = 0; r < values.size() && r < col.missing.size(); ++r) {
                if (col.missing[r]) {
                    out[r] = 0.0;
                    continue;
                }
                const auto parsed = CommonUtils::parseBooleanLikeToken(values[r]);
                out[r] = parsed.has_value() ? *parsed : 0.0;
            }
            engineered.push_back(std::move(boolCol));
            if (engineered.size() >= kMaxDerivedGlobal) break;
        }

        std::array<size_t, 3> sepRows = {0, 0, 0};
        const std::array<char, 3> separators = {',', ';', '|'};
        for (size_t r : observedRows) {
            const std::string s = values[r];
            for (size_t i = 0; i < separators.size(); ++i) {
                if (s.find(separators[i]) != std::string::npos) {
                    ++sepRows[i];
                }
            }
        }

        size_t bestSepIndex = 0;
        for (size_t i = 1; i < sepRows.size(); ++i) {
            if (sepRows[i] > sepRows[bestSepIndex]) bestSepIndex = i;
        }
        const size_t rowsWithSep = sepRows[bestSepIndex];
        if (rowsWithSep < std::max<size_t>(3, observedRows.size() / 10)) continue;
        const char separator = separators[bestSepIndex];

        std::unordered_map<std::string, size_t> tokenRows;
        tokenRows.reserve(observedRows.size() * 2);
        std::vector<size_t> perRowTokenCount(values.size(), 0);
        size_t totalTokens = 0;
        for (size_t r : observedRows) {
            const auto tokens = splitTokensBySeparator(values[r], separator);
            if (tokens.size() < 2) continue;
            std::set<std::string> uniqueRowTokens(tokens.begin(), tokens.end());
            perRowTokenCount[r] = uniqueRowTokens.size();
            totalTokens += uniqueRowTokens.size();
            for (const auto& token : uniqueRowTokens) {
                tokenRows[token]++;
            }
        }

        if (tokenRows.size() < 3) continue;
        const double avgTokens = static_cast<double>(totalTokens) / static_cast<double>(observedRows.size());
        if (avgTokens < 1.30) continue;

        size_t minCount = std::max<size_t>(2, observedRows.size() / 25);

        std::vector<std::pair<std::string, size_t>> rankedTokens(tokenRows.begin(), tokenRows.end());
        std::sort(rankedTokens.begin(), rankedTokens.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });

        size_t minTokenCount = values.empty() ? 0 : values.size();
        size_t maxTokenCount = 0;
        for (size_t r = 0; r < values.size(); ++r) {
            minTokenCount = std::min(minTokenCount, perRowTokenCount[r]);
            maxTokenCount = std::max(maxTokenCount, perRowTokenCount[r]);
        }
        if (maxTokenCount > minTokenCount && engineered.size() < kMaxDerivedGlobal) {
            TypedColumn countCol;
            countCol.name = reserveUniqueName(reservedNames, col.name + "_selection_count");
            countCol.type = ColumnType::NUMERIC;
            countCol.values = std::vector<double>(values.size(), 0.0);
            countCol.missing = MissingMask(values.size(), static_cast<uint8_t>(0));
            auto& out = std::get<NumVec>(countCol.values);
            for (size_t r = 0; r < values.size(); ++r) {
                out[r] = static_cast<double>(perRowTokenCount[r]);
            }
            engineered.push_back(std::move(countCol));
        }

        for (const auto& kv : rankedTokens) {
            if (engineered.size() >= kMaxDerivedGlobal) break;
            if (kv.second < minCount) continue;

            const std::string tokenLabel = sanitizeTokenForColumn(kv.first);
            if (tokenLabel.empty()) continue;

            TypedColumn tokenCol;
            tokenCol.name = reserveUniqueName(reservedNames, col.name + "_has_" + tokenLabel);
            tokenCol.type = ColumnType::NUMERIC;
            tokenCol.values = std::vector<double>(values.size(), 0.0);
            tokenCol.missing = MissingMask(values.size(), static_cast<uint8_t>(0));
            auto& out = std::get<NumVec>(tokenCol.values);

            for (size_t r = 0; r < values.size() && r < col.missing.size(); ++r) {
                if (col.missing[r]) {
                    out[r] = 0.0;
                    continue;
                }
                const auto tokens = splitTokensBySeparator(values[r], separator);
                bool hit = false;
                for (const auto& tok : tokens) {
                    if (tok == kv.first) {
                        hit = true;
                        break;
                    }
                }
                out[r] = hit ? 1.0 : 0.0;
            }

            engineered.push_back(std::move(tokenCol));
            const size_t perColumnAdded = std::count_if(
                engineered.begin(),
                engineered.end(),
                [&](const TypedColumn& c) {
                    return c.name.rfind(col.name + "_has_", 0) == 0;
                });
            if (perColumnAdded >= kMaxTokenIndicatorsPerColumn) break;
        }
    }

    if (!engineered.empty()) {
        auto& cols = data.columns();
        cols.insert(cols.end(), std::make_move_iterator(engineered.begin()), std::make_move_iterator(engineered.end()));
    }
}

bool isBinarySeries(const NumVec& values) {
    bool seenZero = false;
    bool seenOne = false;
    for (double v : values) {
        if (std::abs(v) <= 1e-9) {
            seenZero = true;
        } else if (std::abs(v - 1.0) <= 1e-9) {
            seenOne = true;
        } else {
            return false;
        }
    }
    return seenZero || seenOne;
}

void imputeNumeric(NumVec& values, MissingMask& missing, const std::string& method) {
    NumVec valid;
    valid.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && std::isfinite(values[i])) valid.push_back(values[i]);

    double fill = 0.0;
    std::string m = CommonUtils::toLower(method);
    if (m == "median") fill = CommonUtils::medianByNth(valid);
    else if (m == "zero") fill = 0.0;
    else {
        double sum = 0.0;
        for (double v : valid) sum += v;
        fill = valid.empty() ? 0.0 : (sum / valid.size());
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) {
            values[i] = fill;
            missing[i] = static_cast<uint8_t>(0);
        }
    }
}

std::vector<size_t> numericPredictorColumns(const TypedDataset& data, size_t targetCol) {
    std::vector<size_t> predictors;
    for (size_t idx : data.numericColumnIndices()) {
        if (idx == targetCol) continue;
        predictors.push_back(idx);
    }
    return predictors;
}

void imputeNumericKnnColumn(TypedDataset& data, size_t targetCol, size_t k = 5) {
    std::lock_guard<std::recursive_mutex> lock(gImputationCrossColumnMutex);
    if (targetCol >= data.columns().size()) return;
    auto& targetTyped = data.columns()[targetCol];
    if (targetTyped.type != ColumnType::NUMERIC) return;

    auto& targetVals = std::get<NumVec>(targetTyped.values);
    auto& targetMissing = targetTyped.missing;
    const std::vector<size_t> predictors = numericPredictorColumns(data, targetCol);
    if (predictors.empty()) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    for (size_t row = 0; row < targetVals.size() && row < targetMissing.size(); ++row) {
        if (!targetMissing[row] && std::isfinite(targetVals[row])) continue;

        std::vector<std::pair<double, double>> neighbors;
        neighbors.reserve(128);
        for (size_t cand = 0; cand < targetVals.size() && cand < targetMissing.size(); ++cand) {
            if (targetMissing[cand] || !std::isfinite(targetVals[cand])) continue;

            double dist = 0.0;
            size_t overlap = 0;
            for (size_t pIdx : predictors) {
                const auto& pCol = data.columns()[pIdx];
                const auto& pVals = std::get<NumVec>(pCol.values);
                if (row >= pVals.size() || cand >= pVals.size()) continue;
                if (row < pCol.missing.size() && pCol.missing[row]) continue;
                if (cand < pCol.missing.size() && pCol.missing[cand]) continue;
                if (!std::isfinite(pVals[row]) || !std::isfinite(pVals[cand])) continue;
                const double d = pVals[row] - pVals[cand];
                dist += d * d;
                ++overlap;
            }
            if (overlap == 0) continue;
            neighbors.push_back({std::sqrt(dist / static_cast<double>(overlap)), targetVals[cand]});
        }

        if (neighbors.empty()) continue;
        const size_t keep = std::min(k, neighbors.size());
        std::nth_element(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(keep - 1), neighbors.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        double wsum = 0.0;
        double ysum = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            const double w = 1.0 / std::max(kNeighborDistanceFloor, neighbors[i].first);
            wsum += w;
            ysum += w * neighbors[i].second;
        }
        if (wsum > 0.0) {
            targetVals[row] = ysum / wsum;
            targetMissing[row] = static_cast<uint8_t>(0);
        }
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

void imputeNumericMiceColumn(TypedDataset& data, size_t targetCol) {
    std::lock_guard<std::recursive_mutex> lock(gImputationCrossColumnMutex);
    if (targetCol >= data.columns().size()) return;
    auto& targetTyped = data.columns()[targetCol];
    if (targetTyped.type != ColumnType::NUMERIC) return;

    auto& targetVals = std::get<NumVec>(targetTyped.values);
    auto& targetMissing = targetTyped.missing;
    std::vector<size_t> predictors = numericPredictorColumns(data, targetCol);
    if (predictors.empty()) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    std::vector<std::pair<size_t, double>> ranked;
    ranked.reserve(predictors.size());
    for (size_t pIdx : predictors) {
        const auto& pCol = data.columns()[pIdx];
        const auto& pVals = std::get<NumVec>(pCol.values);
        std::vector<double> x;
        std::vector<double> y;
        const size_t n = std::min(targetVals.size(), pVals.size());
        x.reserve(n);
        y.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (i < targetMissing.size() && targetMissing[i]) continue;
            if (i < pCol.missing.size() && pCol.missing[i]) continue;
            if (!std::isfinite(targetVals[i]) || !std::isfinite(pVals[i])) continue;
            x.push_back(pVals[i]);
            y.push_back(targetVals[i]);
        }
        if (x.size() < 12) continue;
        const double r = std::abs(MathUtils::calculatePearson(x, y, Statistics::calculateStats(x), Statistics::calculateStats(y)).value_or(0.0));
        if (std::isfinite(r)) ranked.push_back({pIdx, r});
    }
    if (ranked.empty()) {
        imputeNumericKnnColumn(data, targetCol, 5);
        return;
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    predictors.clear();
    for (size_t i = 0; i < std::min<size_t>(4, ranked.size()); ++i) predictors.push_back(ranked[i].first);

    std::vector<std::vector<double>> rows;
    std::vector<double> ytrain;
    for (size_t i = 0; i < targetVals.size(); ++i) {
        if (i < targetMissing.size() && targetMissing[i]) continue;
        if (!std::isfinite(targetVals[i])) continue;

        std::vector<double> row;
        row.reserve(predictors.size() + 1);
        row.push_back(1.0);
        bool ok = true;
        for (size_t pIdx : predictors) {
            const auto& pCol = data.columns()[pIdx];
            const auto& pVals = std::get<NumVec>(pCol.values);
            if (i >= pVals.size()) { ok = false; break; }
            if (i < pCol.missing.size() && pCol.missing[i]) { ok = false; break; }
            if (!std::isfinite(pVals[i])) { ok = false; break; }
            row.push_back(pVals[i]);
        }
        if (!ok) continue;
        rows.push_back(std::move(row));
        ytrain.push_back(targetVals[i]);
    }

    if (rows.size() < predictors.size() + 8) {
        imputeNumericKnnColumn(data, targetCol, 5);
        return;
    }

    MathUtils::Matrix X(rows.size(), predictors.size() + 1);
    MathUtils::Matrix Y(rows.size(), 1);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < rows[r].size(); ++c) X.at(r, c) = rows[r][c];
        Y.at(r, 0) = ytrain[r];
    }
    const auto beta = MathUtils::multipleLinearRegression(X, Y);
    if (beta.size() != predictors.size() + 1) {
        imputeNumericKnnColumn(data, targetCol, 5);
        return;
    }

    for (size_t i = 0; i < targetVals.size(); ++i) {
        if (i >= targetMissing.size() || !targetMissing[i]) continue;
        double pred = beta[0];
        bool ok = true;
        for (size_t p = 0; p < predictors.size(); ++p) {
            const auto pIdx = predictors[p];
            const auto& pCol = data.columns()[pIdx];
            const auto& pVals = std::get<NumVec>(pCol.values);
            if (i >= pVals.size()) { ok = false; break; }
            if (i < pCol.missing.size() && pCol.missing[i]) { ok = false; break; }
            if (!std::isfinite(pVals[i])) { ok = false; break; }
            pred += beta[p + 1] * pVals[i];
        }
        if (!ok || !std::isfinite(pred)) continue;
        targetVals[i] = pred;
        targetMissing[i] = static_cast<uint8_t>(0);
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

void imputeNumericMatrixCompletionColumn(TypedDataset& data, size_t targetCol, size_t iterations = 3) {
    std::lock_guard<std::recursive_mutex> lock(gImputationCrossColumnMutex);
    if (targetCol >= data.columns().size()) return;
    auto& targetTyped = data.columns()[targetCol];
    if (targetTyped.type != ColumnType::NUMERIC) return;

    auto& targetVals = std::get<NumVec>(targetTyped.values);
    auto& targetMissing = targetTyped.missing;
    std::vector<size_t> numCols = data.numericColumnIndices();
    if (numCols.size() < 2) {
        imputeNumeric(targetVals, targetMissing, "mean");
        return;
    }

    for (size_t it = 0; it < iterations; ++it) {
        double colMean = 0.0;
        size_t colCount = 0;
        for (size_t r = 0; r < targetVals.size(); ++r) {
            if (r < targetMissing.size() && targetMissing[r]) continue;
            if (!std::isfinite(targetVals[r])) continue;
            colMean += targetVals[r];
            ++colCount;
        }
        colMean = (colCount > 0) ? (colMean / static_cast<double>(colCount)) : 0.0;

        for (size_t r = 0; r < targetVals.size(); ++r) {
            if (r >= targetMissing.size() || !targetMissing[r]) continue;

            double rowMean = 0.0;
            size_t rowCount = 0;
            for (size_t cIdx : numCols) {
                const auto& c = data.columns()[cIdx];
                const auto& vals = std::get<NumVec>(c.values);
                if (r >= vals.size()) continue;
                if (r < c.missing.size() && c.missing[r]) continue;
                if (!std::isfinite(vals[r])) continue;
                rowMean += vals[r];
                ++rowCount;
            }

            const double blend = (rowCount > 0)
                ? (kMatrixCompletionBlend * colMean + (1.0 - kMatrixCompletionBlend) * (rowMean / static_cast<double>(rowCount)))
                : colMean;
            targetVals[r] = std::isfinite(blend) ? blend : colMean;
            targetMissing[r] = static_cast<uint8_t>(0);
        }
    }

    imputeNumeric(targetVals, targetMissing, "mean");
}

void interpolateSeries(NumVec& values, MissingMask& missing) {
    int lastKnown = -1;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!missing[i]) {
            if (lastKnown >= 0 && static_cast<size_t>(lastKnown + 1) < i) {
                double left = values[lastKnown];
                double right = values[i];
                size_t span = i - static_cast<size_t>(lastKnown);
                for (size_t k = 1; k < span; ++k) {
                    double t = static_cast<double>(k) / static_cast<double>(span);
                    values[lastKnown + static_cast<int>(k)] = left + t * (right - left);
                    missing[lastKnown + static_cast<int>(k)] = static_cast<uint8_t>(0);
                }
            }
            lastKnown = static_cast<int>(i);
        }
    }

    // forward/back fill edges
    for (size_t i = 1; i < values.size(); ++i) if (missing[i] && !missing[i - 1]) { values[i] = values[i - 1]; missing[i] = static_cast<uint8_t>(0); }
    for (size_t i = values.size(); i-- > 1;) {
        if (missing[i - 1] && !missing[i]) {
            values[i - 1] = values[i];
            missing[i - 1] = static_cast<uint8_t>(0);
        }
    }
}

void imputeCategorical(StrVec& values, MissingMask& missing, const std::string& method) {
    std::unordered_map<std::string, size_t> freq;
    for (size_t i = 0; i < values.size(); ++i) if (!missing[i] && !values[i].empty()) freq[values[i]]++;

    std::string mode = "Unknown";
    size_t best = 0;
    for (const auto& kv : freq) {
        if (kv.second > best) { best = kv.second; mode = kv.first; }
    }

    std::string m = CommonUtils::toLower(method);
    std::string fill = (m == "mode" || m == "auto") ? mode : mode;
    for (size_t i = 0; i < values.size(); ++i) {
        if (missing[i] || values[i].empty()) {
            values[i] = fill;
            missing[i] = static_cast<uint8_t>(0);
        }
    }
}

void imputeDatetime(TimeVec& values, MissingMask& missing, const std::string& method) {
    std::string m = CommonUtils::toLower(method);
    if (m == "interpolate" || m == "auto") {
        NumVec temp(values.begin(), values.end());
        interpolateSeries(temp, missing);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<int64_t>(temp[i]);
    } else {
        int64_t fill = 0;
        for (size_t i = 0; i < values.size(); ++i) if (!missing[i]) { fill = values[i]; break; }
        for (size_t i = 0; i < values.size(); ++i) if (missing[i]) { values[i] = fill; missing[i] = static_cast<uint8_t>(0); }
    }
}

void addTemporalDateFeatures(TypedDataset& data, const AutoConfig& config) {
    const auto datetimeIndices = data.datetimeColumnIndices();
    if (datetimeIndices.empty()) return;

    std::vector<TypedColumn> engineered;
    engineered.reserve(datetimeIndices.size() * 5);

    for (size_t idx : datetimeIndices) {
        const auto& srcCol = data.columns()[idx];
        const auto& rawValues = std::get<TimeVec>(srcCol.values);
        const MissingMask& rawMissing = srcCol.missing;

        std::string strategy = "auto";
        auto it = config.columnImputation.find(srcCol.name);
        if (it != config.columnImputation.end()) strategy = it->second;
        validateImputationStrategy(strategy, ColumnType::DATETIME, srcCol.name);

        TimeVec imputedValues = rawValues;
        MissingMask imputedMissing = rawMissing;
        imputeDatetime(imputedValues, imputedMissing, strategy);

        std::vector<double> dayOfWeek(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> month(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> dayOfYear(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> quarter(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> isWeekend(imputedValues.size(), std::numeric_limits<double>::quiet_NaN());
        MissingMask derivedMissing(imputedValues.size(), static_cast<uint8_t>(0));

        for (size_t r = 0; r < imputedValues.size(); ++r) {
            if (r < imputedMissing.size() && imputedMissing[r]) {
                derivedMissing[r] = static_cast<uint8_t>(1);
                continue;
            }

            const int64_t ts = imputedValues[r];
            const int weekday = dayOfWeekFromUnixSeconds(ts);
            const int64_t days = floorDiv(ts, 86400);
            const auto [year, monthValue, day] = civilFromDays(days);
            const bool leap = ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0));
            static const int cumDaysNorm[12] = {0,31,59,90,120,151,181,212,243,273,304,334};
            static const int cumDaysLeap[12] = {0,31,60,91,121,152,182,213,244,274,305,335};
            const int* cum = leap ? cumDaysLeap : cumDaysNorm;

            dayOfWeek[r] = static_cast<double>(weekday);
            month[r] = static_cast<double>(monthValue);
            dayOfYear[r] = static_cast<double>(cum[std::max(1u, monthValue) - 1] + static_cast<int>(day));
            quarter[r] = static_cast<double>(((std::max(1u, monthValue) - 1) / 3) + 1);
            isWeekend[r] = (weekday == 0 || weekday == 6) ? 1.0 : 0.0;
        }

        TypedColumn dowCol;
        dowCol.name = uniqueColumnName(data, srcCol.name + "_DayOfWeek");
        dowCol.type = ColumnType::NUMERIC;
        dowCol.values = std::move(dayOfWeek);
        dowCol.missing = derivedMissing;
        engineered.push_back(std::move(dowCol));

        TypedColumn monthCol;
        monthCol.name = uniqueColumnName(data, srcCol.name + "_Month");
        monthCol.type = ColumnType::NUMERIC;
        monthCol.values = std::move(month);
        monthCol.missing = derivedMissing;
        engineered.push_back(std::move(monthCol));

        TypedColumn weekendCol;
        weekendCol.name = uniqueColumnName(data, srcCol.name + "_IsWeekend");
        weekendCol.type = ColumnType::NUMERIC;
        weekendCol.values = std::move(isWeekend);
        weekendCol.missing = derivedMissing;
        engineered.push_back(std::move(weekendCol));

        TypedColumn doyCol;
        doyCol.name = uniqueColumnName(data, srcCol.name + "_DayOfYear");
        doyCol.type = ColumnType::NUMERIC;
        doyCol.values = std::move(dayOfYear);
        doyCol.missing = derivedMissing;
        engineered.push_back(std::move(doyCol));

        TypedColumn quarterCol;
        quarterCol.name = uniqueColumnName(data, srcCol.name + "_Quarter");
        quarterCol.type = ColumnType::NUMERIC;
        quarterCol.values = std::move(quarter);
        quarterCol.missing = std::move(derivedMissing);
        engineered.push_back(std::move(quarterCol));
    }

    auto& columns = data.columns();
    columns.insert(columns.end(), std::make_move_iterator(engineered.begin()), std::make_move_iterator(engineered.end()));
}

void addAutoNumericFeatureEngineering(TypedDataset& data, const AutoConfig& config) {
    auto numericIdx = data.numericColumnIndices();
    if (numericIdx.empty()) return;

    const int targetIdx = config.targetColumn.empty() ? -1 : data.findColumnIndex(config.targetColumn);
    const bool hasNumericTarget =
        targetIdx >= 0 &&
        static_cast<size_t>(targetIdx) < data.columns().size() &&
        data.columns()[static_cast<size_t>(targetIdx)].type == ColumnType::NUMERIC;

    std::vector<size_t> candidates;
    candidates.reserve(numericIdx.size());
    for (size_t idx : numericIdx) {
        if (hasNumericTarget && static_cast<int>(idx) == targetIdx) continue;
        candidates.push_back(idx);
    }
    if (candidates.empty()) return;

    std::vector<double> targetVals;
    MissingMask targetMissing;
    if (hasNumericTarget) {
        targetVals = std::get<NumVec>(data.columns()[static_cast<size_t>(targetIdx)].values);
        targetMissing = data.columns()[static_cast<size_t>(targetIdx)].missing;
    }

    auto absCorrWithTarget = [&](const std::vector<double>& x, const MissingMask& xMissing) {
        if (!hasNumericTarget) return 0.0;
        const size_t n = std::min({x.size(), xMissing.size(), targetVals.size(), targetMissing.size()});
        std::vector<double> cleanX;
        std::vector<double> cleanY;
        cleanX.reserve(n);
        cleanY.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (xMissing[i] || targetMissing[i]) continue;
            if (!std::isfinite(x[i]) || !std::isfinite(targetVals[i])) continue;
            cleanX.push_back(x[i]);
            cleanY.push_back(targetVals[i]);
        }
        if (cleanX.size() < 16) return 0.0;
        const ColumnStats sx = Statistics::calculateStats(cleanX);
        const ColumnStats sy = Statistics::calculateStats(cleanY);
        return std::abs(MathUtils::calculatePearson(cleanX, cleanY, sx, sy).value_or(0.0));
    };

    auto varianceObserved = [&](const std::vector<double>& x, const MissingMask& xMissing) {
        std::vector<double> clean;
        clean.reserve(x.size());
        for (size_t i = 0; i < x.size() && i < xMissing.size(); ++i) {
            if (xMissing[i]) continue;
            if (!std::isfinite(x[i])) continue;
            clean.push_back(x[i]);
        }
        if (clean.size() < 3) return 0.0;
        const ColumnStats s = Statistics::calculateStats(clean);
        return std::max(0.0, s.variance);
    };

    std::vector<std::pair<size_t, double>> rankedBase;
    rankedBase.reserve(candidates.size());
    for (size_t idx : candidates) {
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<NumVec>(col.values);
        const double score = hasNumericTarget
            ? absCorrWithTarget(vals, col.missing)
            : std::sqrt(std::max(0.0, varianceObserved(vals, col.missing)));
        rankedBase.push_back({idx, std::isfinite(score) ? score : 0.0});
    }
    std::sort(rankedBase.begin(), rankedBase.end(), [](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return a.second > b.second;
    });

    const size_t maxBase = std::min<size_t>(rankedBase.size(), std::max<size_t>(2, config.featureEngineeringMaxBase));
    std::vector<size_t> base;
    base.reserve(maxBase);
    for (size_t i = 0; i < maxBase; ++i) base.push_back(rankedBase[i].first);

    const size_t maxPairwise = std::min<size_t>(base.size(), std::max<size_t>(2, config.featureEngineeringMaxPairwiseDiscovery));
    std::vector<size_t> pairwiseBase(base.begin(), base.begin() + maxPairwise);

    std::vector<TypedColumn> engineered;
    engineered.reserve(maxBase * (2 + static_cast<size_t>(std::max(1, config.featureEngineeringDegree))));
    const size_t maxGenerated = std::max<size_t>(16, config.featureEngineeringMaxGeneratedColumns);

    std::unordered_map<size_t, double> baseCorr;
    if (hasNumericTarget) {
        for (size_t idx : base) {
            const auto& col = data.columns()[idx];
            const auto& vals = std::get<NumVec>(col.values);
            baseCorr[idx] = absCorrWithTarget(vals, col.missing);
        }
    }

    auto canAddMore = [&]() {
        return engineered.size() < maxGenerated;
    };

    auto shouldKeepDerived = [&](const std::vector<double>& x,
                                 const MissingMask& xMissing,
                                 size_t parentA,
                                 size_t parentB) {
        const double var = varianceObserved(x, xMissing);
        if (var <= config.tuning.featureMinVariance) return false;

        auto absCorrObservedWithParent = [&](size_t parentIdx) {
            if (parentIdx >= data.columns().size()) return 0.0;
            if (data.columns()[parentIdx].type != ColumnType::NUMERIC) return 0.0;
            const auto& parentCol = data.columns()[parentIdx];
            const auto& pv = std::get<NumVec>(parentCol.values);
            const size_t n = std::min({x.size(), xMissing.size(), pv.size(), parentCol.missing.size()});
            std::vector<double> cleanX;
            std::vector<double> cleanP;
            cleanX.reserve(n);
            cleanP.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                if (xMissing[i] || parentCol.missing[i]) continue;
                if (!std::isfinite(x[i]) || !std::isfinite(pv[i])) continue;
                cleanX.push_back(x[i]);
                cleanP.push_back(pv[i]);
            }
            if (cleanX.size() < 16) return 0.0;
            const ColumnStats sx = Statistics::calculateStats(cleanX);
            const ColumnStats sp = Statistics::calculateStats(cleanP);
            return std::abs(MathUtils::calculatePearson(cleanX, cleanP, sx, sp).value_or(0.0));
        };

        constexpr double kStrictParentCollinearityThreshold = 0.985;
        const double corrParentA = absCorrObservedWithParent(parentA);
        const double corrParentB = (parentB == parentA) ? corrParentA : absCorrObservedWithParent(parentB);
        if (std::max(corrParentA, corrParentB) >= kStrictParentCollinearityThreshold) {
            return false;
        }

        if (!hasNumericTarget) return true;

        const double derivedCorr = absCorrWithTarget(x, xMissing);
        const double baseA = baseCorr.count(parentA) ? baseCorr[parentA] : 0.0;
        const double baseB = baseCorr.count(parentB) ? baseCorr[parentB] : 0.0;
        const double baseBest = std::max(baseA, baseB);
        const double minCorr = std::max(0.05, baseBest * 1.01);
        const double minLift = 0.015;
        return derivedCorr >= minCorr && (derivedCorr + minLift) >= baseBest;
    };

    const int maxDegree = std::clamp(config.featureEngineeringDegree, 1, 4);

    for (size_t idx : base) {
        if (!canAddMore()) {
            break;
        }
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<NumVec>(col.values);

        if (config.featureEngineeringEnablePoly) {
            for (int p = 2; p <= maxDegree; ++p) {
                if (!canAddMore()) {
                    break;
                }
                std::vector<double> poly(vals.size(), 0.0);
                for (size_t i = 0; i < vals.size(); ++i) {
                    poly[i] = std::pow(vals[i], static_cast<double>(p));
                }
                if (shouldKeepDerived(poly, col.missing, idx, idx)) {
                    TypedColumn c;
                    c.name = uniqueColumnName(data, col.name + "_pow" + std::to_string(p));
                    c.type = ColumnType::NUMERIC;
                    c.values = std::move(poly);
                    c.missing = col.missing;
                    engineered.push_back(std::move(c));
                }
            }
        }

        if (config.featureEngineeringEnableLog) {
            if (!canAddMore()) {
                break;
            }
            std::vector<double> logSigned(vals.size(), 0.0);
            for (size_t i = 0; i < vals.size(); ++i) {
                const double v = vals[i];
                const double mag = std::log1p(std::abs(v));
                logSigned[i] = (v < 0.0) ? -mag : mag;
            }
            if (shouldKeepDerived(logSigned, col.missing, idx, idx)) {
                TypedColumn c;
                c.name = uniqueColumnName(data, col.name + "_log1p_abs");
                c.type = ColumnType::NUMERIC;
                c.values = std::move(logSigned);
                c.missing = col.missing;
                engineered.push_back(std::move(c));
            }
        }
    }

    if ((config.featureEngineeringEnablePoly || config.featureEngineeringEnableRatioProductDiscovery) && canAddMore()) {
        for (size_t i = 0; i < pairwiseBase.size(); ++i) {
            if (!canAddMore()) {
                break;
            }
            for (size_t j = i + 1; j < pairwiseBase.size(); ++j) {
                if (!canAddMore()) {
                    break;
                }
                const size_t idxA = pairwiseBase[i];
                const size_t idxB = pairwiseBase[j];
                const auto& a = data.columns()[idxA];
                const auto& b = data.columns()[idxB];
                const auto& av = std::get<NumVec>(a.values);
                const auto& bv = std::get<NumVec>(b.values);
                const size_t n = std::min(av.size(), bv.size());

                if (hasNumericTarget) {
                    const double corrA = baseCorr.count(idxA) ? baseCorr[idxA] : 0.0;
                    const double corrB = baseCorr.count(idxB) ? baseCorr[idxB] : 0.0;
                    constexpr double kModerateCorrThreshold = 0.20;
                    constexpr double kNearPerfectCorrThreshold = 0.98;
                    if (corrA < kModerateCorrThreshold || corrB < kModerateCorrThreshold) {
                        continue;
                    }
                    if (std::max(corrA, corrB) >= kNearPerfectCorrThreshold) {
                        continue;
                    }
                }

                std::vector<double> inter(n, 0.0);
                MissingMask miss(n, static_cast<uint8_t>(0));
                for (size_t r = 0; r < n; ++r) {
                    if (a.missing[r] || b.missing[r]) {
                        miss[r] = static_cast<uint8_t>(1);
                        continue;
                    }
                    inter[r] = av[r] * bv[r];
                }

                if (shouldKeepDerived(inter, miss, idxA, idxB)) {
                    TypedColumn c;
                    c.name = uniqueColumnName(data, a.name + "_x_" + b.name);
                    c.type = ColumnType::NUMERIC;
                    c.values = std::move(inter);
                    c.missing = std::move(miss);
                    engineered.push_back(std::move(c));
                }

                if (!config.featureEngineeringEnableRatioProductDiscovery) continue;
                if (!canAddMore()) break;

                const UnitSemanticKind unitA = inferUnitSemanticKind(a.name);
                const UnitSemanticKind unitB = inferUnitSemanticKind(b.name);
                const bool ratioCompatible = areUnitsRatioCompatible(unitA, unitB);
                if (!ratioCompatible) {
                    continue;
                }

                const double denomEps = std::max(1e-9, config.tuning.numericEpsilon * 1000.0);

                std::vector<double> ratioAB(n, 0.0);
                MissingMask missAB(n, static_cast<uint8_t>(0));
                for (size_t r = 0; r < n; ++r) {
                    if (a.missing[r] || b.missing[r] || std::abs(bv[r]) <= denomEps) {
                        missAB[r] = static_cast<uint8_t>(1);
                        continue;
                    }
                    ratioAB[r] = av[r] / bv[r];
                }
                if (shouldKeepDerived(ratioAB, missAB, idxA, idxB)) {
                    TypedColumn c;
                    c.name = uniqueColumnName(data, a.name + "_div_" + b.name);
                    c.type = ColumnType::NUMERIC;
                    c.values = std::move(ratioAB);
                    c.missing = std::move(missAB);
                    engineered.push_back(std::move(c));
                }

                if (!canAddMore()) break;

                std::vector<double> ratioBA(n, 0.0);
                MissingMask missBA(n, static_cast<uint8_t>(0));
                for (size_t r = 0; r < n; ++r) {
                    if (a.missing[r] || b.missing[r] || std::abs(av[r]) <= denomEps) {
                        missBA[r] = static_cast<uint8_t>(1);
                        continue;
                    }
                    ratioBA[r] = bv[r] / av[r];
                }
                if (shouldKeepDerived(ratioBA, missBA, idxA, idxB)) {
                    TypedColumn c;
                    c.name = uniqueColumnName(data, b.name + "_div_" + a.name);
                    c.type = ColumnType::NUMERIC;
                    c.values = std::move(ratioBA);
                    c.missing = std::move(missBA);
                    engineered.push_back(std::move(c));
                }
            }
        }
    }

    auto& cols = data.columns();
    cols.insert(cols.end(), std::make_move_iterator(engineered.begin()), std::make_move_iterator(engineered.end()));
}

std::vector<bool> detectOutliersIQR(const NumVec& values, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    if (values.size() < 4) return flags;

    const double q1 = CommonUtils::quantileByNth(values, 0.25);
    const double q3 = CommonUtils::quantileByNth(values, 0.75);
    double iqr = q3 - q1;
    double lo = q1 - iqrMultiplier * iqr;
    double hi = q3 + iqrMultiplier * iqr;

    for (size_t i = 0; i < values.size(); ++i) flags[i] = (values[i] < lo || values[i] > hi);
    return flags;
}

std::vector<bool> detectOutliersIQRObserved(const NumVec& values, const MissingMask& missing, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    observed.reserve(values.size());
    std::vector<size_t> observedIdx;
    observedIdx.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i]) continue;
        if (!std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        observedIdx.push_back(i);
    }
    if (observed.size() < 4) return flags;

    std::vector<bool> obsFlags = detectOutliersIQR(observed, iqrMultiplier);
    for (size_t i = 0; i < obsFlags.size() && i < observedIdx.size(); ++i) {
        flags[observedIdx[i]] = obsFlags[i];
    }
    return flags;
}

std::vector<bool> detectOutliersZ(const NumVec& values, double zThreshold) {
    std::vector<bool> flags(values.size(), false);
    if (values.size() < 3) return flags;

    double mean = 0.0;
    for (double v : values) mean += v;
    mean /= values.size();

    double var = 0.0;
    for (double v : values) { double d = v - mean; var += d * d; }
    var /= std::max<size_t>(1, values.size() - 1);
    double sd = std::sqrt(var);
    if (sd <= kNumericEpsilon) return flags;

    for (size_t i = 0; i < values.size(); ++i) {
        double z = std::abs((values[i] - mean) / sd);
        flags[i] = z > zThreshold;
    }
    return flags;
}

std::vector<bool> detectOutliersZObserved(const NumVec& values, const MissingMask& missing, double zThreshold) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    observed.reserve(values.size());
    std::vector<size_t> observedIdx;
    observedIdx.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i]) continue;
        if (!std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        observedIdx.push_back(i);
    }
    if (observed.size() < 3) return flags;

    std::vector<bool> obsFlags = detectOutliersZ(observed, zThreshold);
    for (size_t i = 0; i < obsFlags.size() && i < observedIdx.size(); ++i) {
        flags[observedIdx[i]] = obsFlags[i];
    }
    return flags;
}

std::vector<bool> detectOutliersModifiedZObserved(const NumVec& values, const MissingMask& missing, double zThreshold) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    std::vector<size_t> obsIdx;
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }
    if (observed.size() < 4) return flags;

    const double med = CommonUtils::medianByNth(observed);
    NumVec absDev(observed.size(), 0.0);
    for (size_t i = 0; i < observed.size(); ++i) absDev[i] = std::abs(observed[i] - med);
    const double mad = CommonUtils::medianByNth(absDev);
    if (mad <= kNumericEpsilon) return flags;

    for (size_t i = 0; i < observed.size(); ++i) {
        const double mz = kModifiedZScaleFactor * (observed[i] - med) / mad;
        if (std::abs(mz) > zThreshold) flags[obsIdx[i]] = true;
    }
    return flags;
}

std::vector<bool> detectOutliersAdjustedBoxplotObserved(const NumVec& values, const MissingMask& missing, double iqrMultiplier) {
    std::vector<bool> flags(values.size(), false);
    NumVec observed;
    std::vector<size_t> obsIdx;
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }
    if (observed.size() < 8) return flags;

    const double q1 = CommonUtils::quantileByNth(observed, 0.25);
    const double med = CommonUtils::quantileByNth(observed, 0.50);
    const double q3 = CommonUtils::quantileByNth(observed, 0.75);
    const double iqr = q3 - q1;
    if (iqr <= kNumericEpsilon) return flags;

    const double bowleySkew = (q3 + q1 - 2.0 * med) / iqr;
    const double loFactor = iqrMultiplier * std::exp(-4.0 * bowleySkew);
    const double hiFactor = iqrMultiplier * std::exp(3.0 * bowleySkew);
    const double lo = q1 - loFactor * iqr;
    const double hi = q3 + hiFactor * iqr;

    for (size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] < lo || observed[i] > hi) flags[obsIdx[i]] = true;
    }
    return flags;
}

std::vector<bool> detectOutliersLofObserved(const NumVec& values,
                                            const MissingMask& missing,
                                            const HeuristicTuningConfig& tuning) {
    std::vector<bool> flags(values.size(), false);

    std::vector<double> observed;
    std::vector<size_t> obsIdx;
    observed.reserve(values.size());
    obsIdx.reserve(values.size());
    for (size_t i = 0; i < values.size() && i < missing.size(); ++i) {
        if (missing[i] || !std::isfinite(values[i])) continue;
        observed.push_back(values[i]);
        obsIdx.push_back(i);
    }

    const size_t n = observed.size();
    if (n < 6) return flags;

    const size_t k = std::min<size_t>(20, n - 1);
    const double lofThreshold = std::max(1.5, tuning.lofThresholdFloor);

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (observed[a] == observed[b]) return a < b;
        return observed[a] < observed[b];
    });

    std::vector<size_t> pos(n);
    for (size_t rank = 0; rank < n; ++rank) {
        pos[order[rank]] = rank;
    }

    std::vector<std::vector<size_t>> knn(n);
    std::vector<double> kDistance(n, 0.0);
    for (size_t obs = 0; obs < n; ++obs) {
        const size_t centerPos = pos[obs];
        size_t left = centerPos;
        size_t right = centerPos + 1;
        knn[obs].reserve(k);

        while (knn[obs].size() < k) {
            const bool leftAvailable = left > 0;
            const bool rightAvailable = right < n;

            if (!leftAvailable && !rightAvailable) break;

            if (!rightAvailable) {
                --left;
                knn[obs].push_back(order[left]);
            } else if (!leftAvailable) {
                knn[obs].push_back(order[right]);
                ++right;
            } else {
                const double dl = std::abs(observed[obs] - observed[order[left - 1]]);
                const double dr = std::abs(observed[obs] - observed[order[right]]);
                if (dl <= dr) {
                    --left;
                    knn[obs].push_back(order[left]);
                } else {
                    knn[obs].push_back(order[right]);
                    ++right;
                }
            }
        }

        double kd = 0.0;
        for (size_t neighbor : knn[obs]) {
            kd = std::max(kd, std::abs(observed[obs] - observed[neighbor]));
        }
        kDistance[obs] = kd;
    }

    std::vector<double> lrd(n, 0.0);
    for (size_t obs = 0; obs < n; ++obs) {
        double reachSum = 0.0;
        for (size_t neighbor : knn[obs]) {
            const double dist = std::abs(observed[obs] - observed[neighbor]);
            reachSum += std::max(kDistance[neighbor], dist);
        }
        if (reachSum <= kNumericEpsilon) {
            lrd[obs] = 0.0;
        } else {
            lrd[obs] = static_cast<double>(knn[obs].size()) / reachSum;
        }
    }

    for (size_t obs = 0; obs < n; ++obs) {
        if (lrd[obs] <= kNumericEpsilon) continue;
        double ratioSum = 0.0;
        for (size_t neighbor : knn[obs]) {
            ratioSum += lrd[neighbor] / lrd[obs];
        }
        const double lof = ratioSum / static_cast<double>(knn[obs].size());
        if (lof > lofThreshold) {
            flags[obsIdx[obs]] = true;
        }
    }

    return flags;
}

void capOutliers(NumVec& values, const std::vector<bool>& flags) {
    NumVec inliers;
    for (size_t i = 0; i < values.size(); ++i) if (!flags[i]) inliers.push_back(values[i]);
    if (inliers.empty()) return;
    auto mm = std::minmax_element(inliers.begin(), inliers.end());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!flags[i]) continue;
        if (values[i] < *mm.first) values[i] = *mm.first;
        if (values[i] > *mm.second) values[i] = *mm.second;
    }
}

}

PreprocessReport Preprocessor::run(TypedDataset& data, const AutoConfig& config) {
    PreprocessReport report;
    report.originalRowCount = data.rowCount();

    addTemporalDateFeatures(data, config);
    addAutoNumericFeatureEngineering(data, config);

    // Outlier flags are computed from observed raw numeric values before imputation.
    std::unordered_map<std::string, std::vector<bool>> detectedFlags;
    const std::string selectedOutlierMethod = CommonUtils::toLower(config.outlierMethod);
    for (const auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        const auto& values = std::get<NumVec>(col.values);

        const std::string method = selectedOutlierMethod;
        const std::unordered_map<std::string, std::function<std::vector<bool>()>> outlierFactory = {
            {"iqr", [&]() { return detectOutliersIQRObserved(values, col.missing, config.tuning.outlierIqrMultiplier); }},
            {"zscore", [&]() { return detectOutliersZObserved(values, col.missing, config.tuning.outlierZThreshold); }},
            {"modified_zscore", [&]() { return detectOutliersModifiedZObserved(values, col.missing, config.tuning.outlierZThreshold); }},
            {"adjusted_boxplot", [&]() { return detectOutliersAdjustedBoxplotObserved(values, col.missing, config.tuning.outlierIqrMultiplier); }},
            {"lof", [&]() { return detectOutliersLofObserved(values, col.missing, config.tuning); }},
            {"lof_fallback_modified_zscore", [&]() { return detectOutliersLofObserved(values, col.missing, config.tuning); }}
        };

        const auto itFactory = outlierFactory.find(method);
        std::vector<bool> flags = (itFactory != outlierFactory.end()) ? itFactory->second()
                                                                       : outlierFactory.at("iqr")();

        detectedFlags[col.name] = flags;
        if (config.storeOutlierFlagsInReport) {
            report.outlierFlags[col.name] = flags;
        }
        report.outlierCounts[col.name] = std::count(flags.begin(), flags.end(), true);
    }

    addCategoricalSemanticIndicators(data, config);

    // missing counts + imputation
    std::vector<std::string> strategies(data.columns().size(), "auto");
    std::vector<std::string> normalizedStrategies(data.columns().size(), "auto");
    for (size_t colIdx = 0; colIdx < data.columns().size(); ++colIdx) {
        const auto& col = data.columns()[colIdx];
        size_t missingCount = std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1));
        report.missingCounts[col.name] = missingCount;

        auto it = config.columnImputation.find(col.name);
        if (it != config.columnImputation.end()) strategies[colIdx] = it->second;
        validateImputationStrategy(strategies[colIdx], col.type, col.name);
        normalizedStrategies[colIdx] = CommonUtils::toLower(strategies[colIdx]);
    }

    const std::vector<size_t> numericCols = data.numericColumnIndices();
    if (!numericCols.empty()) {
        const NumericSnapshotStore numericSnapshot = buildNumericSnapshotStore(data);
        std::vector<NumVec> numericValuesResult(numericCols.size());
        std::vector<MissingMask> numericMissingResult(numericCols.size());

        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t pos = 0; pos < numericCols.size(); ++pos) {
            const size_t colIdx = numericCols[pos];
            const auto& col = data.columns()[colIdx];

            NumVec values = std::get<NumVec>(col.values);
            MissingMask missing = col.missing;
            const std::string& strategy = strategies[colIdx];
            const std::string& normalizedStrategy = normalizedStrategies[colIdx];

            if (normalizedStrategy == "interpolate") {
                interpolateSeries(values, missing);
            } else if (normalizedStrategy == "knn") {
                imputeNumericKnnWithSnapshot(values, missing, numericSnapshot, colIdx, 5);
            } else if (normalizedStrategy == "mice") {
                imputeNumericMiceWithSnapshot(values, missing, numericSnapshot, colIdx);
            } else if (normalizedStrategy == "matrix_completion") {
                imputeNumericMatrixCompletionWithSnapshot(values, missing, numericSnapshot, colIdx, 3);
            } else {
                imputeNumeric(values, missing, strategy);
            }

            numericValuesResult[pos] = std::move(values);
            numericMissingResult[pos] = std::move(missing);
        }

        for (size_t pos = 0; pos < numericCols.size(); ++pos) {
            const size_t colIdx = numericCols[pos];
            auto& col = data.columns()[colIdx];
            col.values = std::move(numericValuesResult[pos]);
            col.missing = std::move(numericMissingResult[pos]);
        }
    }

    for (size_t colIdx = 0; colIdx < data.columns().size(); ++colIdx) {
        auto& col = data.columns()[colIdx];
        if (col.type == ColumnType::NUMERIC) continue;

        const std::string& strategy = strategies[colIdx];
        if (col.type == ColumnType::CATEGORICAL) {
            auto& values = std::get<StrVec>(col.values);
            imputeCategorical(values, col.missing, strategy.empty() ? "mode" : strategy);
        } else {
            auto& values = std::get<TimeVec>(col.values);
            imputeDatetime(values, col.missing, strategy.empty() ? "interpolate" : strategy);
        }
    }

    // Outlier flow remains two-phase: detect first, then apply one action.
    MissingMask keep(data.rowCount(), static_cast<uint8_t>(1));
    std::string action = CommonUtils::toLower(config.outlierAction);
    if (action == "remove") {
        for (const auto& col : data.columns()) {
            if (col.type != ColumnType::NUMERIC) continue;
            const auto fit = detectedFlags.find(col.name);
            if (fit == detectedFlags.end()) continue;
            const auto& flags = fit->second;
            for (size_t r = 0; r < flags.size(); ++r) {
                if (flags[r]) keep[r] = static_cast<uint8_t>(0);
            }
        }
        data.removeRows(keep);
    } else if (action == "cap") {
        for (auto& col : data.columns()) {
            if (col.type != ColumnType::NUMERIC) continue;
            auto fit = detectedFlags.find(col.name);
            if (fit == detectedFlags.end()) continue;
            auto& values = std::get<NumVec>(col.values);
            capOutliers(values, fit->second);
        }
    }

    // scaling
    for (auto& col : data.columns()) {
        if (col.type != ColumnType::NUMERIC) continue;
        auto& values = std::get<NumVec>(col.values);
        if (values.empty()) continue;

        if (!config.targetColumn.empty() && col.name == config.targetColumn && isBinarySeries(values)) {
            ScalingParams params;
            params.method = ScalingMethod::NONE;
            auto mm = std::minmax_element(values.begin(), values.end());
            params.min = *mm.first;
            params.max = *mm.second;
            params.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
            report.scaling[col.name] = params;
            continue;
        }

        auto mm = std::minmax_element(values.begin(), values.end());
        double minv = *mm.first;
        double maxv = *mm.second;
        double mean = 0.0;
        for (double v : values) mean += v;
        mean /= values.size();
        double var = 0.0;
        for (double v : values) { double d = v - mean; var += d * d; }
        double stdv = std::sqrt(var / std::max<size_t>(1, values.size() - 1));

        ScalingMethod method = ScalingMethod::NONE;
        std::string pref = CommonUtils::toLower(config.scalingMethod);
        if (pref == "zscore") method = ScalingMethod::ZSCORE;
        else if (pref == "minmax") method = ScalingMethod::MINMAX;
        else if (pref == "none") method = ScalingMethod::NONE;
        else {
            double range = maxv - minv;
            method = (range > 100.0 || std::abs(mean) > 100.0) ? ScalingMethod::MINMAX : ScalingMethod::ZSCORE;
        }

        ScalingParams params;
        params.method = method;
        params.mean = mean;
        params.stddev = (stdv <= 1e-12 ? 1.0 : stdv);
        params.min = minv;
        params.max = maxv;

        if (method == ScalingMethod::ZSCORE) {
            for (double& v : values) v = (v - params.mean) / params.stddev;
        } else if (method == ScalingMethod::MINMAX) {
            double range = (params.max - params.min);
            if (std::abs(range) < 1e-12) range = 1.0;
            for (double& v : values) v = (v - params.min) / range;
        }

        report.scaling[col.name] = params;
    }

    return report;
}
