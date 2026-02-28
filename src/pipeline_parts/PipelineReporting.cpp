std::optional<std::string> buildResidualDiscoveryNarrative(const TypedDataset& data,
                                                           int targetIdx,
                                                           const std::vector<int>& featureIdx,
                                                           const NumericStatsCache& statsCache) {
    if (targetIdx < 0 || featureIdx.size() < 2) return std::nullopt;
    if (static_cast<size_t>(targetIdx) >= data.columns().size()) return std::nullopt;
    if (data.columns()[static_cast<size_t>(targetIdx)].type != ColumnType::NUMERIC) return std::nullopt;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
    const ColumnStats yStats = statsCache.count(static_cast<size_t>(targetIdx))
        ? statsCache.at(static_cast<size_t>(targetIdx))
        : Statistics::calculateStats(y);

    int bestIdx = -1;
    double bestAbsCorr = 0.0;
    for (int idx : featureIdx) {
        if (idx < 0 || idx == targetIdx) continue;
        if (static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (isAdministrativeColumnName(data.columns()[static_cast<size_t>(idx)].name)) continue;

        const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        const ColumnStats xStats = statsCache.count(static_cast<size_t>(idx))
            ? statsCache.at(static_cast<size_t>(idx))
            : Statistics::calculateStats(x);
        const double r = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
        if (std::isfinite(r) && r > bestAbsCorr) {
            bestAbsCorr = r;
            bestIdx = idx;
        }
    }
    if (bestIdx < 0 || bestAbsCorr < 0.35) return std::nullopt;

    const auto& xMain = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(bestIdx)].values);
    const ColumnStats xMainStats = statsCache.count(static_cast<size_t>(bestIdx))
        ? statsCache.at(static_cast<size_t>(bestIdx))
        : Statistics::calculateStats(xMain);
    const auto [slope, intercept] = MathUtils::simpleLinearRegression(xMain,
                                                                       y,
                                                                       xMainStats,
                                                                       yStats,
                                                                       MathUtils::calculatePearson(xMain, y, xMainStats, yStats).value_or(0.0));

    std::vector<double> residual;
    residual.reserve(y.size());
    for (size_t i = 0; i < y.size() && i < xMain.size(); ++i) {
        if (!std::isfinite(y[i]) || !std::isfinite(xMain[i])) {
            residual.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        residual.push_back(y[i] - (slope * xMain[i] + intercept));
    }
    const ColumnStats residualStats = Statistics::calculateStats(residual);

    int secondIdx = -1;
    double bestResidualCorr = 0.0;
    double secondGlobalCorr = 0.0;
    for (int idx : featureIdx) {
        if (idx < 0 || idx == targetIdx || idx == bestIdx) continue;
        if (static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (isAdministrativeColumnName(data.columns()[static_cast<size_t>(idx)].name)) continue;

        const auto& x = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
        const ColumnStats xStats = statsCache.count(static_cast<size_t>(idx))
            ? statsCache.at(static_cast<size_t>(idx))
            : Statistics::calculateStats(x);
        const double g = MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0);
        const double rr = std::abs(MathUtils::calculatePearson(x, residual, xStats, residualStats).value_or(0.0));
        if (std::isfinite(rr) && rr > bestResidualCorr) {
            bestResidualCorr = rr;
            secondIdx = idx;
            secondGlobalCorr = g;
        }
    }
    if (secondIdx < 0 || bestResidualCorr < 0.25) return std::nullopt;

    const auto drift = assessGlobalConditionalDrift(secondGlobalCorr, bestResidualCorr);
    std::string driftText;
    if (drift.label == "flip+collapse") {
        driftText = " Drift check: sign reversal + magnitude collapse versus global relationship.";
    } else if (drift.label == "sign-flip") {
        driftText = " Drift check: sign reversal versus global relationship.";
    } else if (drift.label == "magnitude-collapse") {
        driftText = " Drift check: magnitude collapse after controlling primary driver.";
    }

    return "Residual discovery: " + data.columns()[static_cast<size_t>(bestIdx)].name +
           " is the primary driver of " + data.columns()[static_cast<size_t>(targetIdx)].name +
           " (|r|=" + toFixed(bestAbsCorr, 3) + "), while " +
           data.columns()[static_cast<size_t>(secondIdx)].name +
           " best explains the remaining error signal (|r_residual|=" + toFixed(bestResidualCorr, 3) +
           ", global r=" + toFixed(secondGlobalCorr, 3) + ")." + driftText;
}

std::vector<std::vector<std::string>> buildOutlierContextRows(const TypedDataset& data,
                                                              const PreprocessReport& prep,
                                                              const std::unordered_map<size_t, double>& mahalByRow = {},
                                                              double mahalThreshold = 0.0,
                                                              size_t maxRows = 8) {
    std::vector<std::vector<std::string>> rows;
    if (prep.outlierFlags.empty() || data.rowCount() == 0) return rows;

    const std::vector<size_t> numericIdx = data.numericColumnIndices();
    if (numericIdx.empty()) return rows;

    std::unordered_map<std::string, ColumnStats> statsByName;
    for (size_t idx : numericIdx) {
        const auto& col = data.columns()[idx];
        statsByName[col.name] = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
    }

    std::string segmentColumnName;
    std::vector<std::string> segmentByRow(data.rowCount());
    std::unordered_map<std::string, std::unordered_map<std::string, ColumnStats>> conditionalStats;
    {
        const auto cats = data.categoricalColumnIndices();
        for (size_t cidx : cats) {
            const auto& ccol = data.columns()[cidx];
            const auto& vals = std::get<std::vector<std::string>>(ccol.values);
            std::unordered_map<std::string, size_t> freq;
            for (size_t r = 0; r < vals.size() && r < ccol.missing.size(); ++r) {
                if (ccol.missing[r]) continue;
                const std::string key = CommonUtils::trim(vals[r]);
                if (!key.empty()) freq[key]++;
            }
            if (freq.size() < 2 || freq.size() > 8) continue;

            size_t minGroup = std::numeric_limits<size_t>::max();
            for (const auto& kv : freq) minGroup = std::min(minGroup, kv.second);
            if (minGroup < 12) continue;

            segmentColumnName = ccol.name;
            for (size_t r = 0; r < data.rowCount() && r < vals.size() && r < ccol.missing.size(); ++r) {
                if (ccol.missing[r]) continue;
                segmentByRow[r] = CommonUtils::trim(vals[r]);
            }
            break;
        }

        if (!segmentColumnName.empty()) {
            for (size_t idx : numericIdx) {
                const auto& col = data.columns()[idx];
                const auto& vals = std::get<std::vector<double>>(col.values);
                std::unordered_map<std::string, std::vector<double>> byGroup;
                for (size_t r = 0; r < data.rowCount() && r < vals.size() && r < col.missing.size(); ++r) {
                    if (col.missing[r] || !std::isfinite(vals[r])) continue;
                    if (segmentByRow[r].empty()) continue;
                    byGroup[segmentByRow[r]].push_back(vals[r]);
                }
                for (auto& kv : byGroup) {
                    if (kv.second.size() < 12) continue;
                    conditionalStats[col.name][kv.first] = Statistics::calculateStats(kv.second);
                }
            }
        }
    }

    for (size_t row = 0; row < data.rowCount() && rows.size() < maxRows; ++row) {
        const size_t rowId = row + 1;
        const auto mit = mahalByRow.find(rowId);
        const bool hasMahalanobis = (mit != mahalByRow.end());
        bool hasOutlier = false;
        size_t primaryCol = static_cast<size_t>(-1);
        double primaryAbsZ = 0.0;

        for (size_t idx : numericIdx) {
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            auto fit = prep.outlierFlags.find(col.name);
            if (fit == prep.outlierFlags.end()) continue;
            if (row >= fit->second.size()) continue;
            if (!fit->second[row]) continue;
            if (row >= col.missing.size() || col.missing[row]) continue;

            const auto& vals = std::get<std::vector<double>>(col.values);
            if (row >= vals.size() || !std::isfinite(vals[row])) continue;
            const auto sit = statsByName.find(col.name);
            if (sit == statsByName.end()) continue;

            double z = 0.0;
            bool usedConditional = false;
            if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
                auto cit = conditionalStats.find(col.name);
                if (cit != conditionalStats.end()) {
                    auto git = cit->second.find(segmentByRow[row]);
                    if (git != cit->second.end() && git->second.stddev > 1e-12) {
                        z = std::abs((vals[row] - git->second.mean) / git->second.stddev);
                        usedConditional = true;
                    }
                }
            }
            if (!usedConditional) {
                z = (sit->second.stddev > 1e-12) ? std::abs((vals[row] - sit->second.mean) / sit->second.stddev) : 0.0;
            }
            if (!hasOutlier || z > primaryAbsZ) {
                hasOutlier = true;
                primaryAbsZ = z;
                primaryCol = idx;
            }
        }

        if (!hasOutlier && !hasMahalanobis) continue;

        if (primaryCol == static_cast<size_t>(-1)) {
            for (size_t idx : numericIdx) {
                const auto& col = data.columns()[idx];
                if (isAdministrativeColumnName(col.name)) continue;
                if (row >= col.missing.size() || col.missing[row]) continue;
                const auto& vals = std::get<std::vector<double>>(col.values);
                if (row >= vals.size() || !std::isfinite(vals[row])) continue;
                const auto sit = statsByName.find(col.name);
                if (sit == statsByName.end() || sit->second.stddev <= 1e-12) continue;
                const double z = std::abs((vals[row] - sit->second.mean) / sit->second.stddev);
                if (z > primaryAbsZ) {
                    primaryAbsZ = z;
                    primaryCol = idx;
                }
            }
            if (primaryCol == static_cast<size_t>(-1)) continue;
        }

        const auto& primary = data.columns()[primaryCol];
        const auto& primaryVals = std::get<std::vector<double>>(primary.values);
        std::vector<std::pair<std::string, double>> context;
        for (size_t idx : numericIdx) {
            if (idx == primaryCol) continue;
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            if (row >= col.missing.size() || col.missing[row]) continue;
            const auto& vals = std::get<std::vector<double>>(col.values);
            if (row >= vals.size() || !std::isfinite(vals[row])) continue;
            const auto sit = statsByName.find(col.name);
            if (sit == statsByName.end() || sit->second.stddev <= 1e-12) continue;

            double z = 0.0;
            bool usedConditional = false;
            if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
                auto cit = conditionalStats.find(col.name);
                if (cit != conditionalStats.end()) {
                    auto git = cit->second.find(segmentByRow[row]);
                    if (git != cit->second.end() && git->second.stddev > 1e-12) {
                        z = (vals[row] - git->second.mean) / git->second.stddev;
                        usedConditional = true;
                    }
                }
            }
            if (!usedConditional) {
                z = (vals[row] - sit->second.mean) / sit->second.stddev;
            }
            if (std::abs(z) >= 1.5) context.push_back({col.name, z});
        }
        std::sort(context.begin(), context.end(), [](const auto& a, const auto& b) {
            return std::abs(a.second) > std::abs(b.second);
        });

        std::vector<std::string> reasonParts;
        if (!context.empty()) {
            std::string local = "Context";
            local += ": ";
            const size_t keep = std::min<size_t>(2, context.size());
            for (size_t i = 0; i < keep; ++i) {
                if (i > 0) local += "; ";
                local += context[i].first + "=" + (context[i].second > 0.0 ? "high" : "low") + " (z=" + toFixed(context[i].second, 2) + ")";
            }
            reasonParts.push_back(local);
        }
        if (hasMahalanobis) {
            std::string mahal = "Mahalanobis multivariate distance is elevated (d2=" + toFixed(mit->second, 3);
            if (mahalThreshold > 0.0) {
                mahal += ", threshold=" + toFixed(mahalThreshold, 3);
            }
            mahal += ")";
            if (primaryAbsZ < 2.5) {
                mahal += "; univariate z is moderate, but combined multivariate profile is unusual";
            }
            reasonParts.push_back(mahal);
        }
        if (!segmentColumnName.empty() && row < segmentByRow.size() && !segmentByRow[row].empty()) {
            reasonParts.push_back("within " + segmentColumnName + "=" + segmentByRow[row]);
        }

        std::string reason;
        if (reasonParts.empty()) {
            reason = "No dominant secondary deviations.";
        } else {
            reason = reasonParts.front();
            for (size_t i = 1; i < reasonParts.size(); ++i) {
                reason += "; " + reasonParts[i];
            }
        }

        rows.push_back({
            std::to_string(row + 1),
            primary.name,
            toFixed(primaryVals[row], 4),
            toFixed(primaryAbsZ, 2),
            reason
        });
    }
    return rows;
}
} // namespace

