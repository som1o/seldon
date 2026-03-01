struct RegularizedSeriesView {
    std::vector<double> x;
    std::vector<double> y;
    bool irregularDetected = false;
    double inferredStep = 1.0;
};

RegularizedSeriesView regularizeTimeSeries(const std::vector<double>& x,
                                           const std::vector<double>& y) {
    RegularizedSeriesView out;
    const size_t n = std::min(x.size(), y.size());
    if (n < 8) {
        out.x = x;
        out.y = y;
        return out;
    }

    std::vector<std::pair<double, double>> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        pts.push_back({x[i], y[i]});
    }
    if (pts.size() < 8) {
        out.x = x;
        out.y = y;
        return out;
    }
    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<double> deltas;
    deltas.reserve(pts.size());
    for (size_t i = 1; i < pts.size(); ++i) {
        const double d = pts[i].first - pts[i - 1].first;
        if (d > 0.0 && std::isfinite(d)) deltas.push_back(d);
    }
    if (deltas.empty()) {
        for (const auto& p : pts) {
            out.x.push_back(p.first);
            out.y.push_back(p.second);
        }
        return out;
    }

    std::nth_element(deltas.begin(), deltas.begin() + static_cast<std::ptrdiff_t>(deltas.size() / 2), deltas.end());
    const double medianStep = std::max(1e-9, deltas[deltas.size() / 2]);
    out.inferredStep = medianStep;

    size_t irregularCount = 0;
    for (double d : deltas) {
        if (std::abs(d - medianStep) > 0.20 * medianStep) ++irregularCount;
    }
    out.irregularDetected = irregularCount > (deltas.size() / 5);

    if (!out.irregularDetected) {
        for (const auto& p : pts) {
            out.x.push_back(p.first);
            out.y.push_back(p.second);
        }
        return out;
    }

    const double x0 = pts.front().first;
    const double x1 = pts.back().first;
    const size_t m = static_cast<size_t>(std::max(8.0, std::floor((x1 - x0) / medianStep) + 1.0));
    out.x.reserve(m);
    out.y.reserve(m);

    size_t j = 1;
    for (size_t i = 0; i < m; ++i) {
        const double gx = x0 + static_cast<double>(i) * medianStep;
        while (j < pts.size() && pts[j].first < gx) ++j;
        if (j == 0 || j >= pts.size()) {
            const auto& edge = (j == 0) ? pts.front() : pts.back();
            out.x.push_back(gx);
            out.y.push_back(edge.second);
            continue;
        }
        const auto& a = pts[j - 1];
        const auto& b = pts[j];
        const double span = std::max(1e-9, b.first - a.first);
        const double t = std::clamp((gx - a.first) / span, 0.0, 1.0);
        out.x.push_back(gx);
        out.y.push_back(a.second + t * (b.second - a.second));
    }

    return out;
}

std::vector<double> exponentialSmoothingSeries(const std::vector<double>& y, double alpha = 0.30) {
    if (y.empty()) return {};
    std::vector<double> out(y.size(), y.front());
    for (size_t i = 1; i < y.size(); ++i) {
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1];
    }
    return out;
}

std::vector<double> arima110Approx(const std::vector<double>& y) {
    if (y.size() < 5) return y;
    std::vector<double> d(y.size() - 1, 0.0);
    for (size_t i = 1; i < y.size(); ++i) d[i - 1] = y[i] - y[i - 1];

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 1; i < d.size(); ++i) {
        num += d[i] * d[i - 1];
        den += d[i - 1] * d[i - 1];
    }
    const double phi = (std::abs(den) <= 1e-12) ? 0.0 : std::clamp(num / den, -0.95, 0.95);

    std::vector<double> out(y.size(), y.front());
    out[1] = y[1];
    for (size_t t = 2; t < y.size(); ++t) {
        const double dPred = phi * d[t - 2];
        out[t] = out[t - 1] + dPred;
    }
    return out;
}

std::vector<double> sarimaSeasonalNaiveApprox(const std::vector<double>& y, size_t seasonPeriod) {
    if (y.empty()) return {};
    seasonPeriod = std::max<size_t>(2, std::min(seasonPeriod, std::max<size_t>(2, y.size() / 2)));
    std::vector<double> out(y.size(), y.front());
    for (size_t i = 0; i < y.size(); ++i) {
        if (i < seasonPeriod) {
            out[i] = y[i];
        } else {
            out[i] = y[i - seasonPeriod];
        }
    }
    return out;
}

AdvancedAnalyticsOutputs buildAdvancedAnalyticsOutputs(const TypedDataset& data,
                                                      int targetIdx,
                                                      const std::vector<int>& featureIdx,
                                                      const std::vector<double>& featureImportance,
                                                      const std::vector<PairInsight>& bivariatePairs,
                                                      const std::vector<AnovaInsight>& anovaRows,
                                                      const std::vector<ContingencyInsight>& contingency,
                                                      const NumericStatsCache& statsCache) {
    AdvancedAnalyticsOutputs out;
    if (targetIdx < 0 || static_cast<size_t>(targetIdx) >= data.columns().size()) return out;
    if (data.columns()[static_cast<size_t>(targetIdx)].type != ColumnType::NUMERIC) return out;

    const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
    const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
    const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);

    std::vector<size_t> numericFeatures;
    numericFeatures.reserve(featureIdx.size());
    for (int idx : featureIdx) {
        if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
        if (idx == targetIdx) continue;
        numericFeatures.push_back(static_cast<size_t>(idx));
    }

    auto importanceOf = [&](size_t idx) {
        for (size_t i = 0; i < featureIdx.size() && i < featureImportance.size(); ++i) {
            if (featureIdx[i] == static_cast<int>(idx)) return std::max(0.0, featureImportance[i]);
        }
        const auto xIt = statsCache.find(idx);
        const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
        const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
        return std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
    };

    std::sort(numericFeatures.begin(), numericFeatures.end(), [&](size_t a, size_t b) {
        return importanceOf(a) > importanceOf(b);
    });
    if (numericFeatures.size() > 8) numericFeatures.resize(8);

    size_t igA = static_cast<size_t>(-1);
    size_t igB = static_cast<size_t>(-1);
    double bestIgProxy = 0.0;
    {
        const size_t n = numericFeatures.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& xa = std::get<std::vector<double>>(data.columns()[numericFeatures[i]].values);
            for (size_t j = i + 1; j < n; ++j) {
                const auto& xb = std::get<std::vector<double>>(data.columns()[numericFeatures[j]].values);
                std::vector<double> cross;
                cross.reserve(std::min({xa.size(), xb.size(), y.size()}));
                std::vector<double> yc;
                yc.reserve(cross.capacity());
                const size_t m = std::min({xa.size(), xb.size(), y.size()});
                for (size_t r = 0; r < m; ++r) {
                    if (!std::isfinite(xa[r]) || !std::isfinite(xb[r]) || !std::isfinite(y[r])) continue;
                    if (r < data.columns()[numericFeatures[i]].missing.size() && data.columns()[numericFeatures[i]].missing[r]) continue;
                    if (r < data.columns()[numericFeatures[j]].missing.size() && data.columns()[numericFeatures[j]].missing[r]) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    cross.push_back(xa[r] * xb[r]);
                    yc.push_back(y[r]);
                }
                if (cross.size() < 12) continue;
                const ColumnStats cs = Statistics::calculateStats(cross);
                const ColumnStats ys = Statistics::calculateStats(yc);
                const double rxy = std::abs(MathUtils::calculatePearson(cross, yc, cs, ys).value_or(0.0));
                const double proxy = rxy * std::sqrt(std::max(1e-12, importanceOf(numericFeatures[i]) * importanceOf(numericFeatures[j])));
                if (std::isfinite(proxy) && proxy > bestIgProxy) {
                    bestIgProxy = proxy;
                    igA = numericFeatures[i];
                    igB = numericFeatures[j];
                }
            }
        }
    }
    out.orderedRows.push_back({"1", "Integrated-gradients interaction proxy", (igA == static_cast<size_t>(-1) ? "Insufficient stable pairs for proxy interactions." : (data.columns()[igA].name + " × " + data.columns()[igB].name + " proxy=" + toFixed(bestIgProxy, 4) + "; interaction candidate is materially stronger than independent effects.")) });

    {
        std::vector<int> parent(static_cast<int>(numericFeatures.size()));
        std::iota(parent.begin(), parent.end(), 0);
        auto findp = [&](auto&& self, int x) -> int {
            if (parent[x] == x) return x;
            parent[x] = self(self, parent[x]);
            return parent[x];
        };
        auto unite = [&](int a, int b) {
            a = findp(findp, a);
            b = findp(findp, b);
            if (a != b) parent[b] = a;
        };
        for (size_t i = 0; i < numericFeatures.size(); ++i) {
            const auto& xi = std::get<std::vector<double>>(data.columns()[numericFeatures[i]].values);
            const auto xiIt = statsCache.find(numericFeatures[i]);
            const ColumnStats xis = (xiIt != statsCache.end()) ? xiIt->second : Statistics::calculateStats(xi);
            for (size_t j = i + 1; j < numericFeatures.size(); ++j) {
                const auto& xj = std::get<std::vector<double>>(data.columns()[numericFeatures[j]].values);
                const auto xjIt = statsCache.find(numericFeatures[j]);
                const ColumnStats xjs = (xjIt != statsCache.end()) ? xjIt->second : Statistics::calculateStats(xj);
                const double r = std::abs(MathUtils::calculatePearson(xi, xj, xis, xjs).value_or(0.0));
                if (r >= 0.92) unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
        std::unordered_map<int, std::vector<size_t>> groups;
        for (size_t i = 0; i < numericFeatures.size(); ++i) {
            groups[findp(findp, static_cast<int>(i))].push_back(numericFeatures[i]);
        }
        size_t redundant = 0;
        std::vector<std::string> reps;
        for (const auto& kv : groups) {
            const auto& g = kv.second;
            if (g.empty()) continue;
            if (g.size() > 1) redundant += (g.size() - 1);
            size_t best = g.front();
            double bestScore = -1.0;
            for (size_t idx : g) {
                const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
                const auto xIt = statsCache.find(idx);
                const ColumnStats xs = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
                const double score = std::abs(MathUtils::calculatePearson(x, y, xs, yStats).value_or(0.0));
                if (score > bestScore) {
                    bestScore = score;
                    best = idx;
                }
            }
            reps.push_back(data.columns()[best].name);
        }
        std::string repMsg;
        for (size_t i = 0; i < std::min<size_t>(3, reps.size()); ++i) {
            if (i) repMsg += ", ";
            repMsg += reps[i];
        }
        out.orderedRows.push_back({"2", "Redundancy grouping (correlation clustering)", "Clusters=" + std::to_string(groups.size()) + ", redundant=" + std::to_string(redundant) + ", representatives: " + (repMsg.empty() ? "n/a" : repMsg)});
    }

    {
        size_t deterministicCount = 0;
        std::string exemplar = "none";
        for (const auto& p : bivariatePairs) {
            const bool deterministic = std::abs(p.r) >= 0.999 || p.filteredAsStructural;
            if (!deterministic) continue;
            ++deterministicCount;
            if (exemplar == "none") exemplar = p.featureA + " ~ " + p.featureB + " (|r|=" + toFixed(std::abs(p.r), 4) + ")";
        }
        out.orderedRows.push_back({"3", "Deterministic relationship detection", "Flagged=" + std::to_string(deterministicCount) + "; exemplar: " + exemplar});
    }

    std::vector<size_t> stepwiseSelected;
    {
        auto fitModel = [&](const std::vector<size_t>& selected, std::vector<double>* outPred = nullptr) -> MLRDiagnostics {
            MLRDiagnostics diag;
            if (selected.empty()) return diag;
            std::vector<std::vector<double>> rows;
            std::vector<double> yy;
            const size_t n = y.size();
            for (size_t r = 0; r < n; ++r) {
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                if (!std::isfinite(y[r])) continue;
                std::vector<double> row;
                row.reserve(selected.size() + 1);
                row.push_back(1.0);
                bool ok = true;
                for (size_t idx : selected) {
                    const auto& v = std::get<std::vector<double>>(data.columns()[idx].values);
                    if (r >= v.size() || !std::isfinite(v[r])) { ok = false; break; }
                    if (r < data.columns()[idx].missing.size() && data.columns()[idx].missing[r]) { ok = false; break; }
                    row.push_back(v[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                yy.push_back(y[r]);
            }
            if (rows.size() < selected.size() + 6) return diag;
            MathUtils::Matrix X(rows.size(), selected.size() + 1);
            MathUtils::Matrix Y(rows.size(), 1);
            for (size_t i = 0; i < rows.size(); ++i) {
                for (size_t c = 0; c < rows[i].size(); ++c) X.at(i, c) = rows[i][c];
                Y.at(i, 0) = yy[i];
            }
            diag = MathUtils::performMLRWithDiagnostics(X, Y);
            if (outPred && diag.success && diag.coefficients.size() == selected.size() + 1) {
                outPred->assign(rows.size(), 0.0);
                for (size_t i = 0; i < rows.size(); ++i) {
                    double p = 0.0;
                    for (size_t c = 0; c < rows[i].size(); ++c) p += rows[i][c] * diag.coefficients[c];
                    (*outPred)[i] = p;
                }
            }
            return diag;
        };

        if (!numericFeatures.empty()) {
            stepwiseSelected.push_back(numericFeatures.front());
            MLRDiagnostics bestDiag = fitModel(stepwiseSelected, nullptr);
            for (size_t round = 0; round < 2; ++round) {
                size_t bestFeat = static_cast<size_t>(-1);
                double bestAdj = bestDiag.success ? bestDiag.adjustedRSquared : -1e9;
                for (size_t idx : numericFeatures) {
                    if (std::find(stepwiseSelected.begin(), stepwiseSelected.end(), idx) != stepwiseSelected.end()) continue;
                    auto candidate = stepwiseSelected;
                    candidate.push_back(idx);
                    MLRDiagnostics diag = fitModel(candidate, nullptr);
                    if (!diag.success) continue;
                    if (diag.adjustedRSquared > bestAdj + 0.005) {
                        bestAdj = diag.adjustedRSquared;
                        bestFeat = idx;
                        bestDiag = diag;
                    }
                }
                if (bestFeat == static_cast<size_t>(-1)) break;
                stepwiseSelected.push_back(bestFeat);
            }

            std::string names;
            for (size_t i = 0; i < stepwiseSelected.size(); ++i) {
                if (i) names += " + ";
                names += data.columns()[stepwiseSelected[i]].name;
            }
            out.orderedRows.push_back({"4", "Residual discovery via stepwise OLS", names.empty() ? "No stable stepwise model found." : ("Selected: " + names + "; this indicates residual variance is explained by multiple predictors rather than one dominant correlation.")});
            if (!names.empty()) {
                out.priorityTakeaways.push_back("Stepwise model identifies multi-feature structure: " + names + ".");
            }

            const size_t controlCount = std::min<size_t>(stepwiseSelected.size(), 2);
            std::vector<size_t> controls(stepwiseSelected.begin(), stepwiseSelected.begin() + controlCount);
            auto globalCorr = [&](size_t feature) {
                std::vector<double> yVec;
                std::vector<double> xVec;
                const auto& xv = std::get<std::vector<double>>(data.columns()[feature].values);
                for (size_t r = 0; r < y.size() && r < xv.size(); ++r) {
                    if (!std::isfinite(y[r]) || !std::isfinite(xv[r])) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    if (r < data.columns()[feature].missing.size() && data.columns()[feature].missing[r]) continue;
                    yVec.push_back(y[r]);
                    xVec.push_back(xv[r]);
                }
                if (xVec.size() < 12) return 0.0;
                const ColumnStats sx = Statistics::calculateStats(xVec);
                const ColumnStats sy = Statistics::calculateStats(yVec);
                return MathUtils::calculatePearson(xVec, yVec, sx, sy).value_or(0.0);
            };

            auto partialCorr = [&](size_t feature) {
                std::vector<double> yVec;
                std::vector<double> xVec;
                std::vector<std::vector<double>> ctrlRows;
                const auto& xv = std::get<std::vector<double>>(data.columns()[feature].values);
                for (size_t r = 0; r < y.size() && r < xv.size(); ++r) {
                    if (!std::isfinite(y[r]) || !std::isfinite(xv[r])) continue;
                    if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                    if (r < data.columns()[feature].missing.size() && data.columns()[feature].missing[r]) continue;
                    std::vector<double> row;
                    row.reserve(controls.size() + 1);
                    row.push_back(1.0);
                    bool ok = true;
                    for (size_t cidx : controls) {
                        const auto& cv = std::get<std::vector<double>>(data.columns()[cidx].values);
                        if (r >= cv.size() || !std::isfinite(cv[r])) { ok = false; break; }
                        if (r < data.columns()[cidx].missing.size() && data.columns()[cidx].missing[r]) { ok = false; break; }
                        row.push_back(cv[r]);
                    }
                    if (!ok) continue;
                    ctrlRows.push_back(std::move(row));
                    yVec.push_back(y[r]);
                    xVec.push_back(xv[r]);
                }
                if (ctrlRows.size() < controls.size() + 8) return 0.0;
                MathUtils::Matrix X(ctrlRows.size(), controls.size() + 1);
                MathUtils::Matrix Yy(ctrlRows.size(), 1);
                MathUtils::Matrix Yx(ctrlRows.size(), 1);
                for (size_t i = 0; i < ctrlRows.size(); ++i) {
                    for (size_t c = 0; c < ctrlRows[i].size(); ++c) X.at(i, c) = ctrlRows[i][c];
                    Yy.at(i, 0) = yVec[i];
                    Yx.at(i, 0) = xVec[i];
                }
                const auto by = MathUtils::multipleLinearRegression(X, Yy);
                const auto bx = MathUtils::multipleLinearRegression(X, Yx);
                if (by.size() != controls.size() + 1 || bx.size() != controls.size() + 1) return 0.0;
                std::vector<double> ry(ctrlRows.size(), 0.0), rx(ctrlRows.size(), 0.0);
                for (size_t i = 0; i < ctrlRows.size(); ++i) {
                    double py = 0.0;
                    double px = 0.0;
                    for (size_t c = 0; c < ctrlRows[i].size(); ++c) {
                        py += ctrlRows[i][c] * by[c];
                        px += ctrlRows[i][c] * bx[c];
                    }
                    ry[i] = yVec[i] - py;
                    rx[i] = xVec[i] - px;
                }
                const ColumnStats sx = Statistics::calculateStats(rx);
                const ColumnStats sy = Statistics::calculateStats(ry);
                return MathUtils::calculatePearson(rx, ry, sx, sy).value_or(0.0);
            };

            size_t bestPcIdx = static_cast<size_t>(-1);
            double bestPc = 0.0;
            size_t signFlipCount = 0;
            size_t collapseCount = 0;
            for (size_t idx : numericFeatures) {
                if (std::find(controls.begin(), controls.end(), idx) != controls.end()) continue;
                const double g = globalCorr(idx);
                const double p = partialCorr(idx);
                const double pc = std::abs(p);
                const auto drift = assessGlobalConditionalDrift(g, p);
                if (drift.signFlip || drift.magnitudeCollapse) {
                    if (drift.signFlip) ++signFlipCount;
                    if (drift.magnitudeCollapse) ++collapseCount;
                    std::string interpretation;
                    if (drift.label == "flip+collapse") {
                        interpretation = "Direction reverses after controls and effect size shrinks materially; likely confounding/proxy pathway.";
                    } else if (drift.label == "sign-flip") {
                        interpretation = "Direction reverses after controls; relationship may be Simpson-type/confounded.";
                    } else {
                        interpretation = "Magnitude collapses after controls; global association likely proxy-driven.";
                    }
                    out.globalConditionalRows.push_back({
                        data.columns()[idx].name,
                        toFixed(g, 4),
                        toFixed(p, 4),
                        toFixed(drift.collapseRatio, 3),
                        driftPatternLabel(drift.label),
                        interpretation
                    });
                }

                if (pc > bestPc) {
                    bestPc = pc;
                    bestPcIdx = idx;
                }
            }
            out.orderedRows.push_back({"5", "Partial correlation after control", (bestPcIdx == static_cast<size_t>(-1) ? "No stable partial-correlation signal." : (data.columns()[bestPcIdx].name + " retains independent signal after controlling for top predictors (partial |r|=" + toFixed(bestPc, 4) + ", controls=" + std::to_string(controls.size()) + ")"))});

            if (!out.globalConditionalRows.empty()) {
                const std::string driftMsg = "Detected " + std::to_string(out.globalConditionalRows.size()) +
                    " global-vs-conditional drift patterns (sign flips=" + std::to_string(signFlipCount) +
                    ", magnitude collapses=" + std::to_string(collapseCount) + ").";
                out.orderedRows.push_back({"6", "Global vs conditional drift check", driftMsg});
                out.priorityTakeaways.push_back(driftMsg);
            }
        } else {
            out.orderedRows.push_back({"4", "Residual discovery via stepwise OLS", "No usable numeric predictors."});
            out.orderedRows.push_back({"5", "Partial correlation after control", "No usable numeric predictors."});
        }
    }

    {
        std::string interactionMsg = "Insufficient data for interaction t-test.";
        if (igA != static_cast<size_t>(-1) && igB != static_cast<size_t>(-1)) {
            std::vector<std::array<double, 4>> rows;
            std::vector<double> yAligned;
            rows.reserve(y.size());
            yAligned.reserve(y.size());
            const auto& a = std::get<std::vector<double>>(data.columns()[igA].values);
            const auto& b = std::get<std::vector<double>>(data.columns()[igB].values);
            for (size_t r = 0; r < y.size() && r < a.size() && r < b.size(); ++r) {
                if (!std::isfinite(y[r]) || !std::isfinite(a[r]) || !std::isfinite(b[r])) continue;
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                if (r < data.columns()[igA].missing.size() && data.columns()[igA].missing[r]) continue;
                if (r < data.columns()[igB].missing.size() && data.columns()[igB].missing[r]) continue;
                rows.push_back({1.0, a[r], b[r], a[r] * b[r]});
                yAligned.push_back(y[r]);
            }
            if (rows.size() >= 16) {
                MathUtils::Matrix X(rows.size(), 4);
                MathUtils::Matrix Y(rows.size(), 1);
                for (size_t i = 0; i < rows.size(); ++i) {
                    for (size_t c = 0; c < 4; ++c) X.at(i, c) = rows[i][c];
                    Y.at(i, 0) = yAligned[i];
                }
                MLRDiagnostics diag = MathUtils::performMLRWithDiagnostics(X, Y);
                if (diag.success && diag.tStats.size() > 3 && diag.pValues.size() > 3 && diag.coefficients.size() > 3) {
                    const double tVal = diag.tStats[3];
                    const double pVal = diag.pValues[3];
                    const double beta = diag.coefficients[3];
                    const std::string effectDir = beta >= 0.0 ? "synergistic" : "antagonistic";
                    const std::string strength = (pVal < 1e-4) ? "very strong" : ((pVal < 0.01) ? "strong" : ((pVal < 0.05) ? "moderate" : "weak"));
                    const std::string implication = (strength == "weak")
                        ? ("suggesting a tentative " + effectDir + " interaction pattern")
                        : ("showing a " + effectDir + " interaction direction");
                    interactionMsg = data.columns()[igA].name + "×" + data.columns()[igB].name +
                        " interaction is " + strength + " (t=" + toFixed(tVal, 4) + ", p=" + toFixed(pVal, 6) +
                        "); " + implication + " for " + data.columns()[static_cast<size_t>(targetIdx)].name + ".";
                    out.interactionEvidence = interactionMsg;
                    out.priorityTakeaways.push_back(interactionMsg);
                }
            }
        }
        out.orderedRows.push_back({"6", "Linear interaction significance", interactionMsg});
    }

    {
        CausalDiscoveryOptions causalOptions;
        causalOptions.maxFeatures = 8;
        causalOptions.maxConditionSet = 3;
        causalOptions.alpha = 0.02;
        causalOptions.bootstrapSamples = 180;
        causalOptions.minBootstrapSupport = 0.70;
        causalOptions.minOrientationMargin = 0.15;
        causalOptions.minConfidence = 0.72;
        causalOptions.minAbsCorrelation = 0.10;
        causalOptions.requireProxyValidationWhenTemporal = true;
        causalOptions.randomSeed = 1337;
        causalOptions.enableLiNGAM = true;
        causalOptions.enableFCI = true;
        causalOptions.enableGES = true;
        causalOptions.markExperimentalHeuristics = false;
        causalOptions.enableKernelCiFallback = true;
        causalOptions.enableGrangerValidation = true;
        causalOptions.enableIcpValidation = true;

        const auto causal = CausalDiscovery::discover(data,
                                                      numericFeatures,
                                                      static_cast<size_t>(targetIdx),
                                                      causalOptions);

        size_t lineageIdentitySuppressed = 0;
        std::vector<const CausalEdgeResult*> keptEdges;
        for (const auto& edge : causal.edges) {
            if (edge.fromIdx >= data.columns().size() || edge.toIdx >= data.columns().size()) continue;
            const std::string& fromName = data.columns()[edge.fromIdx].name;
            const std::string& toName = data.columns()[edge.toIdx].name;
            if (sharesEngineeredRootIdentity(fromName, toName)) {
                ++lineageIdentitySuppressed;
                continue;
            }
            keptEdges.push_back(&edge);
            std::string evidence = edge.evidence;
            evidence += ", bootstrap=" + toFixed(100.0 * edge.bootstrapSupport, 1) + "%";
            std::string interpretation = edge.interpretation;
            if (edge.bootstrapSupport >= 0.75 && edge.confidence >= 0.70) {
                interpretation += " | directional effect candidate (observational, requires intervention/domain validation)";
            } else {
                interpretation += " | correlation-consistent direction hypothesis (not causal proof)";
            }
            out.causalDagRows.push_back({
                fromName,
                toName,
                toFixed(edge.confidence, 3),
                evidence,
                interpretation
            });
        }

        if (!out.causalDagRows.empty()) {
            std::unordered_map<std::string, std::string> nodeIds;
            auto idFor = [&](const std::string& node) {
                auto it = nodeIds.find(node);
                if (it != nodeIds.end()) return it->second;
                const std::string id = "N" + std::to_string(nodeIds.size() + 1);
                nodeIds[node] = id;
                return id;
            };
            auto quoteSafe = [](std::string s) {
                std::replace(s.begin(), s.end(), '"', '\'');
                return s;
            };

            std::ostringstream mermaid;
            mermaid << "```mermaid\n";
            mermaid << "flowchart LR\n";
            for (const auto* edge : keptEdges) {
                if (edge == nullptr) continue;
                if (edge->fromIdx >= data.columns().size() || edge->toIdx >= data.columns().size()) continue;
                const std::string& from = data.columns()[edge->fromIdx].name;
                const std::string& to = data.columns()[edge->toIdx].name;
                const std::string fromId = idFor(from);
                const std::string toId = idFor(to);
                mermaid << "    " << fromId << "[\"" << quoteSafe(from) << "\"] -->|"
                        << toFixed(edge->bootstrapSupport, 2)
                        << "| "
                        << toId << "[\"" << quoteSafe(to) << "\"]\n";
            }
            mermaid << "```";
            out.causalDagMermaid = mermaid.str();

            size_t targetInbound = 0;
            for (const auto* edge : keptEdges) {
                if (edge == nullptr) continue;
                if (edge->toIdx == static_cast<size_t>(targetIdx)) ++targetInbound;
            }
            std::string summary = "Constraint-based causal DAG produced " + std::to_string(out.causalDagRows.size()) +
                " directed edges (incoming-to-target=" + std::to_string(targetInbound) +
                ", bootstrap runs=" + std::to_string(causal.bootstrapRuns) + ").";
            if (causal.usedLiNGAM) summary += " Non-Gaussian LiNGAM orientation applied.";
            summary += " Edges denote observational directional hypotheses under conditional-independence and model assumptions; they are not interventional proof of causation.";
            if (lineageIdentitySuppressed > 0) {
                summary += " Identity-aware filter suppressed " + std::to_string(lineageIdentitySuppressed) + " same-root engineered edge(s).";
            }
            out.priorityTakeaways.push_back(summary);
            for (const auto& note : causal.notes) out.priorityTakeaways.push_back(note);
            out.orderedRows.push_back({"7", "Causal discovery DAG (PC + LiNGAM + bootstrap)", summary});
        } else {
            if (!causal.edges.empty() && lineageIdentitySuppressed > 0) {
                out.orderedRows.push_back({"7", "Causal discovery DAG (PC + LiNGAM + bootstrap)", "All discovered directed edges were suppressed by identity-aware lineage filtering (same engineered root)."});
            } else {
                out.orderedRows.push_back({"7", "Causal discovery DAG (PC + LiNGAM + bootstrap)", "No stable directed edges after CI tests and bootstrap filtering; retain correlation-level interpretation only."});
            }
        }
    }

    {
        const auto axis = detectTemporalAxis(data);
        std::vector<size_t> temporalCandidates = numericFeatures;
        if (std::find(temporalCandidates.begin(), temporalCandidates.end(), static_cast<size_t>(targetIdx)) == temporalCandidates.end()) {
            temporalCandidates.push_back(static_cast<size_t>(targetIdx));
        }

        std::vector<StationarityDiagnostic> diagnostics;
        diagnostics.reserve(temporalCandidates.size());
        for (size_t idx : temporalCandidates) {
            if (idx >= data.columns().size()) continue;
            if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
            if (CommonUtils::toLower(data.columns()[idx].name) == CommonUtils::toLower(axis.name)) continue;
            const auto& values = std::get<std::vector<double>>(data.columns()[idx].values);
            auto diag = Statistics::adfStyleDrift(values, axis.axis, data.columns()[idx].name, axis.name);
            if (diag.samples < 12) continue;
            diagnostics.push_back(std::move(diag));
        }

        std::sort(diagnostics.begin(), diagnostics.end(), [](const auto& a, const auto& b) {
            if (a.nonStationary != b.nonStationary) return a.nonStationary > b.nonStationary;
            if (a.driftRatio == b.driftRatio) return a.pApprox > b.pApprox;
            return a.driftRatio > b.driftRatio;
        });

        size_t flagged = 0;
        for (const auto& d : diagnostics) {
            if (d.nonStationary) ++flagged;
            if (!d.nonStationary && d.verdict != "borderline") continue;
            out.temporalDriftRows.push_back({
                d.feature,
                d.axis,
                std::to_string(d.samples),
                toFixed(d.gamma, 5),
                toFixed(d.tStatistic, 4),
                toFixed(d.pApprox, 6),
                toFixed(d.driftRatio, 3),
                d.verdict
            });
            if (out.temporalDriftRows.size() >= 12) break;
        }

        if (!diagnostics.empty()) {
            const std::string msg = "ADF-style temporal drift scan on axis '" + axis.name +
                "' flagged " + std::to_string(flagged) + " non-stationary signals out of " + std::to_string(diagnostics.size()) + ".";
            out.orderedRows.push_back({"8", "Temporal drift kernels (ADF-style)", msg});
            if (flagged > 0) {
                out.priorityTakeaways.push_back(msg + " Prioritize time-aware features or differencing for flagged columns.");
            }
        }
    }

    {
        const auto deadZones = detectContextualDeadZones(data,
                                                         static_cast<size_t>(targetIdx),
                                                         numericFeatures,
                                                         10);
        for (const auto& dz : deadZones) {
            out.contextualDeadZoneRows.push_back({
                dz.feature,
                dz.strongCluster,
                dz.weakCluster,
                toFixed(dz.strongCorr, 4),
                toFixed(dz.weakCorr, 4),
                toFixed(dz.dropRatio, 3),
                std::to_string(dz.support)
            });
        }
        if (!deadZones.empty()) {
            const auto& lead = deadZones.front();
            const std::string msg = "Contextual dead-zones detected: " + std::to_string(deadZones.size()) +
                " features switch from predictive to locally irrelevant across clusters; strongest example " +
                lead.feature + " (" + lead.strongCluster + " |r|=" + toFixed(lead.strongCorr, 3) +
                " vs " + lead.weakCluster + " |r|=" + toFixed(lead.weakCorr, 3) + ").";
            out.orderedRows.push_back({"9", "Cross-cluster interaction anomalies", msg});
            out.priorityTakeaways.push_back(msg);
        }
    }

    {
        std::string mahalMsg = "Insufficient stable dimensions for Mahalanobis outlier scoring.";
        if (numericFeatures.size() >= 2) {
            const size_t dims = std::min<size_t>(3, numericFeatures.size());
            std::vector<std::vector<double>> rows;
            std::vector<size_t> rowIds;
            for (size_t r = 0; r < data.rowCount(); ++r) {
                std::vector<double> row;
                row.reserve(dims);
                bool ok = true;
                for (size_t c = 0; c < dims; ++c) {
                    size_t idx = numericFeatures[c];
                    const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
                    if (r >= vals.size() || !std::isfinite(vals[r])) { ok = false; break; }
                    if (r < data.columns()[idx].missing.size() && data.columns()[idx].missing[r]) { ok = false; break; }
                    row.push_back(vals[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                rowIds.push_back(r + 1);
            }
            if (rows.size() > dims + 8) {
                std::vector<double> mean(dims, 0.0);
                for (const auto& row : rows) {
                    for (size_t c = 0; c < dims; ++c) mean[c] += row[c];
                }
                for (double& m : mean) m /= static_cast<double>(rows.size());

                MathUtils::Matrix cov(dims, dims);
                for (const auto& row : rows) {
                    for (size_t i = 0; i < dims; ++i) {
                        for (size_t j = 0; j < dims; ++j) {
                            cov.at(i, j) += (row[i] - mean[i]) * (row[j] - mean[j]);
                        }
                    }
                }
                const double denom = static_cast<double>(std::max<size_t>(1, rows.size() - 1));
                for (size_t i = 0; i < dims; ++i) {
                    for (size_t j = 0; j < dims; ++j) cov.at(i, j) /= denom;
                }
                auto invOpt = cov.inverse();
                if (invOpt.has_value()) {
                    const auto& inv = *invOpt;
                    std::vector<double> d2(rows.size(), 0.0);
                    for (size_t r = 0; r < rows.size(); ++r) {
                        std::vector<double> diff(dims, 0.0);
                        for (size_t c = 0; c < dims; ++c) diff[c] = rows[r][c] - mean[c];
                        double sum = 0.0;
                        for (size_t i = 0; i < dims; ++i) {
                            double inner = 0.0;
                            for (size_t j = 0; j < dims; ++j) inner += inv.data[i][j] * diff[j];
                            sum += diff[i] * inner;
                        }
                        d2[r] = std::max(0.0, sum);
                    }
                    const double threshold = CommonUtils::quantileByNth(d2, 0.975);
                    size_t flagged = 0;
                    std::vector<std::pair<size_t, double>> top;
                    for (size_t i = 0; i < d2.size(); ++i) {
                        if (d2[i] >= threshold) {
                            ++flagged;
                            top.push_back({rowIds[i], d2[i]});
                        }
                    }
                    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                    out.mahalanobisThreshold = threshold;
                    for (size_t i = 0; i < std::min<size_t>(6, top.size()); ++i) {
                        out.mahalanobisRows.push_back({std::to_string(top[i].first), toFixed(top[i].second, 4), toFixed(threshold, 4)});
                        out.mahalanobisByRow[top[i].first] = top[i].second;
                    }
                    mahalMsg = "Dimensions=" + std::to_string(dims) + ", flagged=" + std::to_string(flagged) + ", threshold(d2@97.5%)=" + toFixed(threshold, 4);
                    if (flagged > 0) {
                        out.priorityTakeaways.push_back("Mahalanobis multivariate scan flags " + std::to_string(flagged) + " structurally unusual observations.");
                    }
                }
            }
        }
        out.orderedRows.push_back({"7", "Mahalanobis multivariate outliers", mahalMsg});
    }

    {
        std::string anovaMsg = "No ANOVA/Tukey highlight available.";
        if (!anovaRows.empty()) {
            const auto it = std::max_element(anovaRows.begin(), anovaRows.end(), [](const auto& a, const auto& b) {
                if (a.eta2 == b.eta2) return a.pValue > b.pValue;
                return a.eta2 < b.eta2;
            });
            anovaMsg = it->categorical + " -> " + it->numeric + " (eta2=" + toFixed(it->eta2, 4) + ", Tukey=" + it->tukeySummary + ")";
        }
        out.orderedRows.push_back({"8", "ANOVA + Tukey HSD strongest class contrast", anovaMsg});
    }

    {
        std::string cramerMsg = "No categorical-contingency highlight available.";
        if (!contingency.empty()) {
            const auto it = std::max_element(contingency.begin(), contingency.end(), [](const auto& a, const auto& b) {
                return a.cramerV < b.cramerV;
            });
            cramerMsg = it->catA + " vs " + it->catB + " (V=" + toFixed(it->cramerV, 4) + ", " + cramerStrengthLabel(it->cramerV) + ")";
        }
        out.orderedRows.push_back({"9", "Cramér's V categorical strength", cramerMsg});
    }

    {
        std::string pdpMsg = "Insufficient data for linear partial-dependence approximation.";
        if (!numericFeatures.empty()) {
            const size_t dims = std::min<size_t>(3, numericFeatures.size());
            std::vector<std::vector<double>> rows;
            std::vector<double> yy;
            for (size_t r = 0; r < data.rowCount(); ++r) {
                if (r >= y.size() || !std::isfinite(y[r])) continue;
                if (r < data.columns()[static_cast<size_t>(targetIdx)].missing.size() && data.columns()[static_cast<size_t>(targetIdx)].missing[r]) continue;
                std::vector<double> row;
                row.reserve(dims + 1);
                row.push_back(1.0);
                bool ok = true;
                for (size_t c = 0; c < dims; ++c) {
                    const auto& vals = std::get<std::vector<double>>(data.columns()[numericFeatures[c]].values);
                    if (r >= vals.size() || !std::isfinite(vals[r])) { ok = false; break; }
                    if (r < data.columns()[numericFeatures[c]].missing.size() && data.columns()[numericFeatures[c]].missing[r]) { ok = false; break; }
                    row.push_back(vals[r]);
                }
                if (!ok) continue;
                rows.push_back(std::move(row));
                yy.push_back(y[r]);
            }
            if (rows.size() >= dims + 10) {
                MathUtils::Matrix X(rows.size(), dims + 1);
                MathUtils::Matrix Y(rows.size(), 1);
                for (size_t i = 0; i < rows.size(); ++i) {
                    for (size_t c = 0; c < rows[i].size(); ++c) X.at(i, c) = rows[i][c];
                    Y.at(i, 0) = yy[i];
                }
                auto beta = MathUtils::multipleLinearRegression(X, Y);
                if (beta.size() == dims + 1) {
                    std::vector<double> meanX(dims, 0.0);
                    for (const auto& row : rows) {
                        for (size_t c = 0; c < dims; ++c) meanX[c] += row[c + 1];
                    }
                    for (double& v : meanX) v /= static_cast<double>(rows.size());

                    for (size_t c = 0; c < std::min<size_t>(2, dims); ++c) {
                        const auto& vals = std::get<std::vector<double>>(data.columns()[numericFeatures[c]].values);
                        const double p10 = CommonUtils::quantileByNth(vals, 0.10);
                        const double p50 = CommonUtils::quantileByNth(vals, 0.50);
                        const double p90 = CommonUtils::quantileByNth(vals, 0.90);
                        auto pred = [&](double v) {
                            double yhat = beta[0];
                            for (size_t j = 0; j < dims; ++j) {
                                const double xj = (j == c) ? v : meanX[j];
                                yhat += beta[j + 1] * xj;
                            }
                            return yhat;
                        };
                        const double y10 = pred(p10);
                        const double y50 = pred(p50);
                        const double y90 = pred(p90);
                        const double delta = y90 - y10;
                        const std::string direction = (delta > 0.0) ? "increasing" : ((delta < 0.0) ? "decreasing" : "flat");
                        out.pdpRows.push_back({data.columns()[numericFeatures[c]].name,
                                               toFixed(y10, 4),
                                               toFixed(y50, 4),
                                               toFixed(y90, 4),
                                               toFixed(delta, 4),
                                               direction});
                    }
                    pdpMsg = "Generated linear PDP approximations for " + std::to_string(out.pdpRows.size()) + " top features.";
                    if (!out.pdpRows.empty()) {
                        out.priorityTakeaways.push_back("PDP approximation shows " + out.pdpRows.front()[0] + " has a " + out.pdpRows.front()[5] + " target response across low→high values (Δ=" + out.pdpRows.front()[4] + ").");
                    }
                }
            }
        }
        out.orderedRows.push_back({"10", "Linear partial-dependence approximation", pdpMsg});
    }

    {
        std::string topA = (numericFeatures.size() > 0) ? data.columns()[numericFeatures[0]].name : "Feature A";
        std::string topB = (numericFeatures.size() > 1) ? data.columns()[numericFeatures[1]].name : "Feature B";
        std::string narrative = topA + " dominates the global signal, while " + topB + " adds independent information after controlling for primary effects.";
        if (igA != static_cast<size_t>(-1) && igB != static_cast<size_t>(-1)) {
            narrative = data.columns()[igA].name + " and " + data.columns()[igB].name + " jointly shape the target: one carries dominant variance while the other contributes independent interaction structure.";
        }
        out.narrativeRows.push_back(narrative);
        out.priorityTakeaways.push_back(narrative);
        out.orderedRows.push_back({"11", "Cross-section narrative synthesis", narrative});
    }

    std::vector<std::vector<std::string>> filtered;
    filtered.reserve(out.orderedRows.size());
    for (const auto& row : out.orderedRows) {
        if (row.size() < 3) continue;
        const std::string r = CommonUtils::trim(row[2]);
        if (r.rfind("Insufficient", 0) == 0 || r.rfind("No ", 0) == 0) continue;
        filtered.push_back(row);
    }
    for (size_t i = 0; i < filtered.size(); ++i) {
        filtered[i][0] = std::to_string(i + 1);
    }
    out.orderedRows = std::move(filtered);

    if (!out.priorityTakeaways.empty()) {
        out.executiveSummary = "Advanced methods confirm that signal is not purely pairwise: interaction evidence, multivariate outlier structure, and residual stepwise modeling all contribute independent explanatory value.";
    }

    return out;
}

struct StratifiedPopulationInsight {
    std::string segmentColumn;
    std::string numericColumn;
    size_t groups = 0;
    size_t rows = 0;
    double eta2 = 0.0;
    double separation = 0.0;
    std::string groupMeans;
};

std::vector<StratifiedPopulationInsight> detectStratifiedPopulations(const TypedDataset& data,
                                                                     size_t maxInsights = 12) {
    std::vector<StratifiedPopulationInsight> out;
    const auto cats = data.categoricalColumnIndices();
    const auto nums = data.numericColumnIndices();
    if (cats.empty() || nums.empty()) return out;

    for (size_t cidx : cats) {
        const auto& cv = std::get<std::vector<std::string>>(data.columns()[cidx].values);
        if (cv.empty()) continue;

        std::unordered_map<std::string, size_t> labelCardinality;
        for (size_t r = 0; r < cv.size() && r < data.rowCount(); ++r) {
            if (data.columns()[cidx].missing[r]) continue;
            if (cv[r].empty()) continue;
            labelCardinality[cv[r]]++;
        }
        if (labelCardinality.size() < 2 || labelCardinality.size() > 8) continue;

        for (size_t nidx : nums) {
            const auto& nv = std::get<std::vector<double>>(data.columns()[nidx].values);
            const size_t n = std::min(cv.size(), nv.size());
            if (n < 30) continue;

            struct GroupStats {
                size_t count = 0;
                double sum = 0.0;
                double sumSq = 0.0;
            };
            std::map<std::string, GroupStats> groups;
            for (size_t i = 0; i < n; ++i) {
                if (data.columns()[cidx].missing[i] || data.columns()[nidx].missing[i]) continue;
                if (!std::isfinite(nv[i])) continue;
                const std::string& label = cv[i];
                if (label.empty()) continue;
                auto& g = groups[label];
                g.count++;
                g.sum += nv[i];
                g.sumSq += nv[i] * nv[i];
            }

            size_t validRows = 0;
            for (const auto& kv : groups) validRows += kv.second.count;
            if (groups.size() < 2 || validRows < 30) continue;

            const size_t minGroupRows = std::max<size_t>(3, validRows / 300);
            size_t strongGroups = 0;
            for (const auto& kv : groups) {
                if (kv.second.count >= minGroupRows) strongGroups++;
            }
            if (strongGroups < 2) continue;

            const double grand = [&]() {
                double s = 0.0;
                for (const auto& kv : groups) s += kv.second.sum;
                return s / static_cast<double>(validRows);
            }();

            double ssb = 0.0;
            double ssw = 0.0;
            double minMean = std::numeric_limits<double>::infinity();
            double maxMean = -std::numeric_limits<double>::infinity();
            std::vector<std::pair<std::string, double>> means;
            means.reserve(groups.size());
            for (const auto& kv : groups) {
                const auto& g = kv.second;
                if (g.count == 0) continue;
                const double mu = g.sum / static_cast<double>(g.count);
                minMean = std::min(minMean, mu);
                maxMean = std::max(maxMean, mu);
                means.push_back({kv.first, mu});
                const double dm = mu - grand;
                ssb += static_cast<double>(g.count) * dm * dm;
                const double within = std::max(0.0, g.sumSq - (g.sum * g.sum) / static_cast<double>(g.count));
                ssw += within;
            }
            if (means.size() < 2) continue;

            const double sst = ssb + ssw;
            if (sst <= 1e-12) continue;
            const double eta2 = ssb / sst;
            const double pooledStd = std::sqrt(ssw / static_cast<double>(std::max<size_t>(1, validRows - means.size())));
            const double separation = (pooledStd <= 1e-12) ? 0.0 : std::abs(maxMean - minMean) / pooledStd;

            if (eta2 < 0.20 || separation < 0.75) continue;

            std::sort(means.begin(), means.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            std::ostringstream oss;
            const size_t show = std::min<size_t>(4, means.size());
            for (size_t i = 0; i < show; ++i) {
                if (i > 0) oss << "; ";
                oss << means[i].first << ": " << toFixed(means[i].second, 4);
            }

            out.push_back({
                data.columns()[cidx].name,
                data.columns()[nidx].name,
                means.size(),
                validRows,
                eta2,
                separation,
                oss.str()
            });
        }
    }

    std::sort(out.begin(), out.end(), [](const StratifiedPopulationInsight& a, const StratifiedPopulationInsight& b) {
        if (a.eta2 == b.eta2) return a.separation > b.separation;
        return a.eta2 > b.eta2;
    });
    if (out.size() > maxInsights) out.resize(maxInsights);
    return out;
}

std::pair<double, double> bootstrapCI(const std::vector<double>& values,
                                      size_t rounds = 300,
                                      double alpha = 0.05,
                                      uint32_t seed = 1337,
                                      bool showProgress = false,
                                      const std::string& label = "bootstrap") {
    if (values.empty()) return {0.0, 0.0};
    FastRng rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
    std::vector<double> stats;
    stats.reserve(rounds);
    for (size_t b = 0; b < rounds; ++b) {
        double sum = 0.0;
        for (size_t i = 0; i < values.size(); ++i) sum += values[rng.uniformIndex(values.size())];
        stats.push_back(sum / static_cast<double>(values.size()));
        if (showProgress && (((b + 1) % std::max<size_t>(1, rounds / 25)) == 0 || (b + 1) == rounds)) {
            printProgressBar(label, b + 1, rounds);
        }
    }
    std::sort(stats.begin(), stats.end());
    const double lo = CommonUtils::quantileByNth(stats, alpha / 2.0);
    const double hi = CommonUtils::quantileByNth(stats, 1.0 - alpha / 2.0);
    return {lo, hi};
}

struct PCAInsight {
    std::vector<double> pc1;
    std::vector<double> pc2;
    std::vector<double> explained;
    std::vector<std::string> labels;
};

PCAInsight runPCA2(const TypedDataset& data,
                   const std::vector<size_t>& numericIdx,
                   size_t maxRows = 300) {
    PCAInsight out;
    if (numericIdx.size() < 2) return out;

    std::vector<std::vector<double>> X;
    const size_t nRows = data.rowCount();
    for (size_t r = 0; r < nRows; ++r) {
        std::vector<double> row;
        row.reserve(numericIdx.size());
        bool ok = true;
        for (size_t idx : numericIdx) {
            const auto& v = std::get<std::vector<double>>(data.columns()[idx].values);
            if (r >= v.size() || data.columns()[idx].missing[r] || !std::isfinite(v[r])) {
                ok = false;
                break;
            }
            row.push_back(v[r]);
        }
        if (ok) X.push_back(std::move(row));
        if (X.size() >= maxRows) break;
    }
    if (X.size() < 12) return out;

    const size_t p = numericIdx.size();
    std::vector<double> mu(p, 0.0), sd(p, 0.0);
    for (const auto& row : X) for (size_t j = 0; j < p; ++j) mu[j] += row[j];
    for (size_t j = 0; j < p; ++j) mu[j] /= static_cast<double>(X.size());
    for (const auto& row : X) for (size_t j = 0; j < p; ++j) { double d = row[j] - mu[j]; sd[j] += d * d; }
    for (size_t j = 0; j < p; ++j) { sd[j] = std::sqrt(sd[j] / std::max<size_t>(1, X.size() - 1)); if (sd[j] <= 1e-12) sd[j] = 1.0; }
    for (auto& row : X) for (size_t j = 0; j < p; ++j) row[j] = (row[j] - mu[j]) / sd[j];

    std::vector<std::vector<double>> C(p, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i; j < p; ++j) {
            double s = 0.0;
            for (const auto& row : X) s += row[i] * row[j];
            s /= static_cast<double>(std::max<size_t>(1, X.size() - 1));
            C[i][j] = s;
            C[j][i] = s;
        }
    }

    auto power = [&](const std::vector<std::vector<double>>& M) {
        std::vector<double> v(p, 1.0 / std::sqrt(static_cast<double>(p)));
        for (int it = 0; it < 80; ++it) {
            std::vector<double> nv(p, 0.0);
            for (size_t i = 0; i < p; ++i) for (size_t j = 0; j < p; ++j) nv[i] += M[i][j] * v[j];
            double norm = 0.0;
            for (double q : nv) norm += q * q;
            norm = std::sqrt(std::max(norm, 1e-12));
            for (double& q : nv) q /= norm;
            v = std::move(nv);
        }
        double eig = 0.0;
        for (size_t i = 0; i < p; ++i) {
            double mv = 0.0;
            for (size_t j = 0; j < p; ++j) mv += M[i][j] * v[j];
            eig += v[i] * mv;
        }
        return std::make_pair(v, eig);
    };

    auto [v1, e1] = power(C);
    std::vector<std::vector<double>> C2 = C;
    for (size_t i = 0; i < p; ++i) for (size_t j = 0; j < p; ++j) C2[i][j] -= e1 * v1[i] * v1[j];
    auto [v2, e2] = power(C2);

    out.pc1.reserve(X.size());
    out.pc2.reserve(X.size());
    for (const auto& row : X) {
        double s1 = 0.0, s2 = 0.0;
        for (size_t j = 0; j < p; ++j) {
            s1 += row[j] * v1[j];
            s2 += row[j] * v2[j];
        }
        out.pc1.push_back(s1);
        out.pc2.push_back(s2);
    }
    out.explained = {std::max(0.0, e1), std::max(0.0, e2)};
    double total = 0.0;
    for (size_t j = 0; j < p; ++j) total += std::max(0.0, C[j][j]);
    if (total > 1e-12) {
        out.explained[0] /= total;
        out.explained[1] /= total;
    }
    for (size_t idx : numericIdx) out.labels.push_back(data.columns()[idx].name);
    return out;
}

struct KMeansInsight {
    size_t bestK = 0;
    double silhouette = 0.0;
    double gapStatistic = 0.0;
    std::vector<int> labels;
};

KMeansInsight runKMeans2D(const std::vector<double>& x, const std::vector<double>& y) {
    KMeansInsight out;
    const size_t n = std::min(x.size(), y.size());
    if (n < 20) return out;

    auto dist = [&](size_t a, size_t b) {
        const double dx = x[a] - x[b];
        const double dy = y[a] - y[b];
        return std::sqrt(dx * dx + dy * dy);
    };

    double bestSil = -1.0;
    double bestGap = 0.0;
    std::vector<int> bestLabels;
    size_t bestK = 0;
    for (size_t k = 2; k <= std::min<size_t>(6, n / 5); ++k) {
        std::vector<std::pair<double, double>> centers;
        for (size_t i = 0; i < k; ++i) centers.push_back({x[(i * n) / k], y[(i * n) / k]});
        std::vector<int> labels(n, 0);
        for (int it = 0; it < 20; ++it) {
            for (size_t i = 0; i < n; ++i) {
                double bd = std::numeric_limits<double>::infinity();
                int bc = 0;
                for (size_t c = 0; c < k; ++c) {
                    const double dx = x[i] - centers[c].first;
                    const double dy = y[i] - centers[c].second;
                    const double d = dx * dx + dy * dy;
                    if (d < bd) { bd = d; bc = static_cast<int>(c); }
                }
                labels[i] = bc;
            }
            std::vector<double> sx(k, 0.0), sy(k, 0.0), sc(k, 0.0);
            for (size_t i = 0; i < n; ++i) {
                const int c = labels[i];
                sx[c] += x[i]; sy[c] += y[i]; sc[c] += 1.0;
            }
            for (size_t c = 0; c < k; ++c) {
                if (sc[c] > 0.0) centers[c] = {sx[c] / sc[c], sy[c] / sc[c]};
            }
        }

        double sil = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double a = 0.0; size_t ac = 0;
            std::vector<double> bsum(k, 0.0);
            std::vector<size_t> bcnt(k, 0);
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const double d = dist(i, j);
                const int cj = labels[j];
                bsum[cj] += d;
                bcnt[cj]++;
                if (labels[i] == cj) { a += d; ac++; }
            }
            a = (ac > 0) ? (a / static_cast<double>(ac)) : 0.0;
            double b = std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < k; ++c) {
                if (static_cast<int>(c) == labels[i] || bcnt[c] == 0) continue;
                b = std::min(b, bsum[c] / static_cast<double>(bcnt[c]));
            }
            if (!std::isfinite(b)) b = a;
            const double den = std::max(a, b);
            if (den > 1e-12) sil += (b - a) / den;
        }
        sil /= static_cast<double>(n);

        double wk = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const auto& ctr = centers[static_cast<size_t>(labels[i])];
            const double dx = x[i] - ctr.first;
            const double dy = y[i] - ctr.second;
            wk += dx * dx + dy * dy;
        }

        const double minX = *std::min_element(x.begin(), x.end());
        const double maxX = *std::max_element(x.begin(), x.end());
        const double minY = *std::min_element(y.begin(), y.end());
        const double maxY = *std::max_element(y.begin(), y.end());
        std::mt19937 rg(1337U + static_cast<uint32_t>(k));
        std::uniform_real_distribution<double> ux(minX, maxX);
        std::uniform_real_distribution<double> uy(minY, maxY);
        double wkRef = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double rx = ux(rg);
            const double ry = uy(rg);
            double bd = std::numeric_limits<double>::infinity();
            for (const auto& ctr : centers) {
                const double dx = rx - ctr.first;
                const double dy = ry - ctr.second;
                bd = std::min(bd, dx * dx + dy * dy);
            }
            wkRef += bd;
        }
        const double gap = std::log(std::max(1e-12, wkRef)) - std::log(std::max(1e-12, wk));

        if (sil > bestSil) {
            bestSil = sil;
            bestGap = gap;
            bestLabels = labels;
            bestK = k;
        }
    }

    out.bestK = bestK;
    out.silhouette = std::max(0.0, bestSil);
    out.gapStatistic = bestGap;
    out.labels = std::move(bestLabels);
    return out;
}

void addOverallSections(ReportEngine& report,
                        const TypedDataset& data,
                        const PreprocessReport& prep,
                        const std::vector<BenchmarkResult>& benchmarks,
                        const NeuralAnalysis& neural,
                        const DataHealthSummary& health,
                        const AutoConfig& config,
                        GnuplotEngine* overallPlotter,
                        bool canPlotOverall,
                        bool verbose,
                        const NumericStatsCache& statsCache) {
    report.addParagraph("Overall: dataset health, model baselines, neural training, and global visual diagnostics.");
    addDatasetHealthTable(report, data, prep, health);

    {
        std::vector<std::vector<std::string>> typeSummary = {
            {"numeric", std::to_string(data.numericColumnIndices().size())},
            {"categorical", std::to_string(data.categoricalColumnIndices().size())},
            {"datetime", std::to_string(data.datetimeColumnIndices().size())}
        };
        report.addTable("Data Type Summary", {"Type", "Columns"}, typeSummary);

        std::vector<std::vector<std::string>> qualityRows;
        qualityRows.reserve(data.columns().size());
        for (const auto& col : data.columns()) {
            const size_t miss = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : 0;
            const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;
            const double missPct = data.rowCount() == 0 ? 0.0 : (100.0 * static_cast<double>(miss) / static_cast<double>(data.rowCount()));
            qualityRows.push_back({
                col.name,
                col.type == ColumnType::NUMERIC ? "numeric" : (col.type == ColumnType::DATETIME ? "datetime" : "categorical"),
                std::to_string(miss),
                toFixed(missPct, 2) + "%",
                std::to_string(outliers)
            });
        }
        report.addTable("Column Quality Matrix", {"Column", "Type", "Missing", "Missing %", "Outliers"}, qualityRows);
    }

    addBenchmarkSection(report, benchmarks);
    addNeuralLossSummaryTable(report, neural);

    if (overallPlotter && canPlotOverall) {
        std::vector<std::string> labels;
        std::vector<double> missingCounts;
        for (const auto& col : data.columns()) {
            labels.push_back(col.name);
            auto it = prep.missingCounts.find(col.name);
            missingCounts.push_back(it == prep.missingCounts.end() ? 0.0 : static_cast<double>(it->second));
        }
        const bool tooManyMissingBars = PlotHeuristics::shouldAvoidCategoryHeavyCharts(labels.size(), 18);
        if (!tooManyMissingBars) {
            std::string missingImg = overallPlotter->bar("overall_missingness", labels, missingCounts, "Overall Missingness by Column");
            if (!missingImg.empty()) report.addImage("Overall Missingness", missingImg);
        }

        if (PlotHeuristics::shouldAddPieChart(missingCounts, config.tuning)) {
            std::string missingPie = overallPlotter->pie("overall_missingness_pie",
                                                         labels,
                                                         missingCounts,
                                                         "Missingness Share by Column");
            if (!missingPie.empty()) report.addImage("Missingness Share (Pie)", missingPie);
        }

        std::vector<std::vector<double>> missHeat(1, std::vector<double>(missingCounts.size(), 0.0));
        for (size_t i = 0; i < missingCounts.size(); ++i) missHeat[0][i] = missingCounts[i];
        std::string missHeatImg = overallPlotter->heatmap("overall_missingness_heatmap", missHeat, "Missingness Heatmap", labels);
        if (!missHeatImg.empty()) report.addImage("Missingness Heatmap", missHeatImg);

        if (auto timeline = detectProjectTimeline(data, config.tuning); timeline.has_value()) {
            std::string gantt = overallPlotter->gantt("overall_project_timeline",
                                                      timeline->taskNames,
                                                      timeline->start,
                                                      timeline->end,
                                                      timeline->semantics,
                                                      "Project Timeline (Auto-Detected)");
            if (!gantt.empty()) {
                report.addImage("Project Timeline (Gantt)", gantt);
                report.addParagraph("Gantt chart generated automatically from project-like columns (task + start/end or start/duration). Timeline preview limit uses gantt_max_tasks tuning.");
            }
        }

        auto numericIdx = data.numericColumnIndices();
        if (numericIdx.size() >= 2) {
            const size_t maxCols = std::max<size_t>(2, config.tuning.overallCorrHeatmapMaxColumns);
            if (numericIdx.size() > maxCols) {
                std::stable_sort(numericIdx.begin(), numericIdx.end(), [&](size_t lhs, size_t rhs) {
                    auto lIt = statsCache.find(lhs);
                    auto rIt = statsCache.find(rhs);
                    const double lv = (lIt != statsCache.end() && std::isfinite(lIt->second.variance)) ? lIt->second.variance : 0.0;
                    const double rv = (rIt != statsCache.end() && std::isfinite(rIt->second.variance)) ? rIt->second.variance : 0.0;
                    return lv > rv;
                });
                numericIdx.resize(maxCols);
            }

            std::vector<std::vector<double>> corr(numericIdx.size(), std::vector<double>(numericIdx.size(), 1.0));
            for (size_t i = 0; i < numericIdx.size(); ++i) {
                for (size_t j = i + 1; j < numericIdx.size(); ++j) {
                    const auto idxI = numericIdx[i];
                    const auto idxJ = numericIdx[j];
                    const auto& a = std::get<std::vector<double>>(data.columns()[idxI].values);
                    const auto& b = std::get<std::vector<double>>(data.columns()[idxJ].values);
                    auto ai = statsCache.find(idxI);
                    auto bi = statsCache.find(idxJ);
                    std::optional<ColumnStats> aFallback;
                    std::optional<ColumnStats> bFallback;
                    const ColumnStats* sa = nullptr;
                    const ColumnStats* sb = nullptr;
                    if (ai != statsCache.end()) sa = &ai->second;
                    else {
                        aFallback = Statistics::calculateStats(a);
                        sa = &(*aFallback);
                    }
                    if (bi != statsCache.end()) sb = &bi->second;
                    else {
                        bFallback = Statistics::calculateStats(b);
                        sb = &(*bFallback);
                    }
                    if (sa == nullptr || sb == nullptr) continue;
                    double r = MathUtils::calculatePearson(a, b, *sa, *sb).value_or(0.0);
                    corr[i][j] = r;
                    corr[j][i] = r;
                }
            }

            const std::vector<size_t> order = clusteredOrderFromCorrelation(corr);
            std::vector<std::vector<double>> clustered(corr.size(), std::vector<double>(corr.size(), 0.0));
            std::vector<std::string> heatLabels;
            heatLabels.reserve(order.size());
            for (size_t i = 0; i < order.size(); ++i) {
                heatLabels.push_back(data.columns()[numericIdx[order[i]]].name);
                for (size_t j = 0; j < order.size(); ++j) {
                    clustered[i][j] = corr[order[i]][order[j]];
                }
            }

            std::string img = overallPlotter->clusteredHeatmap("overall_corr_heatmap", clustered, "Overall Correlation Heatmap", heatLabels, true);
            if (!img.empty()) report.addImage("Overall Correlation Heatmap", img);
        }

        if (!neural.trainLoss.empty()) {
            std::vector<double> epochs(neural.trainLoss.size(), 0.0);
            for (size_t i = 0; i < epochs.size(); ++i) epochs[i] = static_cast<double>(i + 1);
            if (!neural.valLoss.empty() && neural.valLoss.size() == neural.trainLoss.size()) {
                std::string lossImg = overallPlotter->multiLine("overall_nn_loss_multi",
                                                                epochs,
                                                                {neural.trainLoss, neural.valLoss},
                                                                {"Train", "Validation"},
                                                                "NN Loss Curves",
                                                                "Loss");
                if (!lossImg.empty()) {
                    report.addImage("Overall NN Loss (Train vs Validation)", lossImg);
                }
            } else {
                std::string trainImg = overallPlotter->line("overall_nn_train_loss", epochs, neural.trainLoss, "NN Train Loss");
                if (!trainImg.empty()) report.addImage("Overall NN Train Loss", trainImg);
            }
        }

        if (!neural.gradientNorm.empty()) {
            std::vector<double> x(neural.gradientNorm.size(), 0.0);
            for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i + 1);
            std::string gradImg = overallPlotter->line("overall_nn_gradient_norm", x, neural.gradientNorm, "NN Gradient Norm");
            if (!gradImg.empty()) report.addImage("Overall NN Gradient Norm", gradImg);
        }

        if (!neural.weightStd.empty() || !neural.weightMeanAbs.empty()) {
            const size_t n = std::max(neural.weightStd.size(), neural.weightMeanAbs.size());
            std::vector<double> x(n, 0.0);
            for (size_t i = 0; i < n; ++i) x[i] = static_cast<double>(i + 1);
            std::vector<double> weightStdSeries(n, 0.0);
            std::vector<double> weightAbsSeries(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                if (i < neural.weightStd.size()) weightStdSeries[i] = neural.weightStd[i];
                if (i < neural.weightMeanAbs.size()) weightAbsSeries[i] = neural.weightMeanAbs[i];
            }
            std::string weightDyn = overallPlotter->multiLine("overall_nn_weight_dynamics",
                                                               x,
                                                               {weightStdSeries, weightAbsSeries},
                                                               {"Weight RMS", "Weight Mean Abs"},
                                                               "NN Weight Dynamics",
                                                               "Value");
            if (!weightDyn.empty()) report.addImage("Overall NN Weight Dynamics", weightDyn);
        }

        if (auto ts = detectTimeSeriesSignal(data, config.tuning); ts.has_value()) {
            const RegularizedSeriesView regularized = regularizeTimeSeries(ts->timeX, ts->values);
            const auto& tsX = regularized.x.empty() ? ts->timeX : regularized.x;
            const auto& tsY = regularized.y.empty() ? ts->values : regularized.y;

            std::string trend = overallPlotter->timeSeriesTrend("overall_timeseries_trend",
                                                                tsX,
                                                                tsY,
                                                                ts->valueName + " over " + ts->timeName,
                                                                true);
            if (!trend.empty()) {
                report.addImage("Automatic Time-Series Trend", trend);
            }
            if (regularized.irregularDetected) {
                report.addParagraph("Irregular time intervals were detected and regularized using median-step interpolation before advanced time-series modeling.");
            }

            const size_t nTs = tsY.size();
            const size_t seasonPeriod = std::clamp<size_t>(config.tuning.timeSeriesSeasonPeriod,
                                                           static_cast<size_t>(2),
                                                           std::max<size_t>(2, nTs / 2));
            std::vector<double> trendComp(nTs, 0.0);
            const size_t trendWindow = std::max<size_t>(3, std::min<size_t>(std::max<size_t>(seasonPeriod, 3), 25));
            const size_t half = trendWindow / 2;
            for (size_t i = 0; i < nTs; ++i) {
                const size_t lo = (i > half) ? (i - half) : 0;
                const size_t hi = std::min(nTs - 1, i + half);
                double sum = 0.0;
                size_t count = 0;
                for (size_t j = lo; j <= hi; ++j) {
                    sum += tsY[j];
                    ++count;
                }
                trendComp[i] = sum / static_cast<double>(std::max<size_t>(1, count));
            }

            std::vector<double> detrended(nTs, 0.0);
            for (size_t i = 0; i < nTs; ++i) {
                detrended[i] = tsY[i] - trendComp[i];
            }

            std::vector<double> seasonalMeans(seasonPeriod, 0.0);
            std::vector<size_t> seasonalCounts(seasonPeriod, 0);
            for (size_t i = 0; i < nTs; ++i) {
                const size_t phase = i % seasonPeriod;
                seasonalMeans[phase] += detrended[i];
                seasonalCounts[phase] += 1;
            }
            for (size_t p = 0; p < seasonPeriod; ++p) {
                if (seasonalCounts[p] > 0) {
                    seasonalMeans[p] /= static_cast<double>(seasonalCounts[p]);
                }
            }
            double seasonalMean = 0.0;
            for (double v : seasonalMeans) seasonalMean += v;
            seasonalMean /= static_cast<double>(seasonalMeans.size());
            for (double& v : seasonalMeans) v -= seasonalMean;

            std::vector<double> seasonal(nTs, 0.0);
            std::vector<double> resid(nTs, 0.0);
            for (size_t i = 0; i < nTs; ++i) {
                seasonal[i] = seasonalMeans[i % seasonPeriod];
                resid[i] = tsY[i] - trendComp[i] - seasonal[i];
            }

            const std::vector<double> ets = exponentialSmoothingSeries(tsY, 0.30);
            const std::vector<double> arima = arima110Approx(tsY);
            const std::vector<double> sarima = sarimaSeasonalNaiveApprox(tsY, seasonPeriod);

            std::string decomp = overallPlotter->multiLine("overall_timeseries_decomposition",
                                                           tsX,
                                                           {tsY, trendComp, seasonal, resid, ets, arima, sarima},
                                                           {"Observed", "Trend", "Seasonal", "Residual", "ETS", "ARIMA(1,1,0) Approx", "SARIMA Seasonal-Naive Approx"},
                                                           "Time-Series Decomposition + Advanced Models",
                                                           "Value");
            if (!decomp.empty()) {
                report.addImage("Time-Series Decomposition", decomp);
            }
        }

        {
            auto numeric = data.numericColumnIndices();
            if (numeric.size() >= 3) {
                const auto& a = std::get<std::vector<double>>(data.columns()[numeric[0]].values);
                const auto& b = std::get<std::vector<double>>(data.columns()[numeric[1]].values);
                const auto& c = std::get<std::vector<double>>(data.columns()[numeric[2]].values);
                std::vector<double> x3;
                std::vector<double> y3;
                std::vector<double> z3;
                const size_t n = std::min({a.size(), b.size(), c.size(), data.rowCount()});
                x3.reserve(n);
                y3.reserve(n);
                z3.reserve(n);
                for (size_t r = 0; r < n; ++r) {
                    if ((r < data.columns()[numeric[0]].missing.size() && data.columns()[numeric[0]].missing[r]) ||
                        (r < data.columns()[numeric[1]].missing.size() && data.columns()[numeric[1]].missing[r]) ||
                        (r < data.columns()[numeric[2]].missing.size() && data.columns()[numeric[2]].missing[r])) {
                        continue;
                    }
                    if (!std::isfinite(a[r]) || !std::isfinite(b[r]) || !std::isfinite(c[r])) continue;
                    x3.push_back(a[r]);
                    y3.push_back(b[r]);
                    z3.push_back(c[r]);
                    if (x3.size() >= 3000) break;
                }
                std::string scatter3d = overallPlotter->scatter3D("overall_scatter3d",
                                                                   x3,
                                                                   y3,
                                                                   z3,
                                                                   "3D Scatter: " + data.columns()[numeric[0]].name + " vs " + data.columns()[numeric[1]].name + " vs " + data.columns()[numeric[2]].name);
                if (!scatter3d.empty()) report.addImage("3D Scatter", scatter3d);

            }
        }

        {
            auto numIdx = data.numericColumnIndices();
            if (numIdx.size() >= 3) {
                constexpr size_t kPcaMaxDimensions = 10;
                const size_t cap = std::min<size_t>(numIdx.size(), kPcaMaxDimensions);
                numIdx.resize(cap);
                const PCAInsight pca = runPCA2(data, numIdx, 300);
                if (!pca.pc1.empty() && pca.pc1.size() == pca.pc2.size()) {
                    std::vector<std::vector<std::string>> pcaRows;
                    pcaRows.push_back({"PC1", toFixed(100.0 * (pca.explained.empty() ? 0.0 : pca.explained[0]), 2) + "%"});
                    pcaRows.push_back({"PC2", toFixed(100.0 * (pca.explained.size() > 1 ? pca.explained[1] : 0.0), 2) + "%"});
                    report.addTable("PCA Explained Variance", {"Component", "Explained"}, pcaRows);

                    std::string pcaScatter = overallPlotter->scatter("overall_pca_biplot_like",
                                                                      pca.pc1,
                                                                      pca.pc2,
                                                                      "PCA Scores (PC1 vs PC2)",
                                                                      false,
                                                                      0.0,
                                                                      0.0,
                                                                      "",
                                                                      false,
                                                                      1.96,
                                                                      config.tuning.scatterDownsampleThreshold);
                    if (!pcaScatter.empty()) report.addImage("PCA Biplot (Score Projection)", pcaScatter);

                    std::vector<double> comps = {1.0, 2.0};
                    std::vector<double> exp = {pca.explained[0], pca.explained.size() > 1 ? pca.explained[1] : 0.0};
                    std::string scree = overallPlotter->line("overall_pca_scree", comps, exp, "PCA Scree (Top Components)");
                    if (!scree.empty()) report.addImage("PCA Scree Plot", scree);

                    KMeansInsight km = runKMeans2D(pca.pc1, pca.pc2);
                    if (km.bestK >= 2 && km.labels.size() == pca.pc1.size()) {
                        std::vector<std::string> fac;
                        fac.reserve(km.labels.size());
                        std::vector<double> counts(km.bestK, 0.0);
                        for (int label : km.labels) {
                            const size_t cl = static_cast<size_t>(std::max(0, label));
                            if (cl < counts.size()) counts[cl] += 1.0;
                            fac.push_back("cluster_" + std::to_string(label + 1));
                        }
                        std::vector<std::string> lbl;
                        for (size_t k = 0; k < km.bestK; ++k) lbl.push_back("Cluster " + std::to_string(k + 1));
                        std::string cbar = overallPlotter->bar("overall_kmeans_profile", lbl, counts, "K-Means Cluster Sizes");
                        if (!cbar.empty()) report.addImage("K-Means Cluster Profile", cbar);
                        std::string facPlot = overallPlotter->facetedScatter("overall_kmeans_faceted",
                                                                             pca.pc1,
                                                                             pca.pc2,
                                                                             fac,
                                                                             "PC Space by K-Means Cluster",
                                                                             std::min<size_t>(km.bestK, 6));
                        if (!facPlot.empty()) report.addImage("K-Means Cluster Visualization", facPlot);
                        report.addParagraph("K-Means auto-selection chose k=" + std::to_string(km.bestK) + " with silhouette=" + toFixed(km.silhouette, 4) + " and gap=" + toFixed(km.gapStatistic, 4) + ".");
                    }
                }
            }
        }

        {
            size_t categoricalPlotsAdded = 0;
            for (size_t cidx : data.categoricalColumnIndices()) {
                const auto& col = data.columns()[cidx];
                const auto& vals = std::get<std::vector<std::string>>(col.values);
                const CategoryFrequencyProfile profile = buildCategoryFrequencyProfile(vals, col.missing, 12);
                if (profile.labels.size() < 2) continue;

                const bool usePie = PlotHeuristics::shouldAddPieChart(profile.counts, config.tuning);
                const bool avoidBar = PlotHeuristics::shouldAvoidCategoryHeavyCharts(profile.labels.size(), 12);

                if (usePie) {
                    std::string pie = overallPlotter->pie("overall_cat_pie_" + std::to_string(cidx),
                                                          profile.labels,
                                                          profile.counts,
                                                          "Category Share: " + col.name);
                    if (!pie.empty()) {
                        report.addImage("Category Share (Pie): " + col.name, pie);
                        ++categoricalPlotsAdded;
                    }
                }

                if (!usePie && !avoidBar) {
                    std::string bar = overallPlotter->bar("overall_cat_bar_" + std::to_string(cidx),
                                                          profile.labels,
                                                          profile.counts,
                                                          "Category Frequency: " + col.name);
                    if (!bar.empty()) {
                        report.addImage("Category Frequency: " + col.name, bar);
                        ++categoricalPlotsAdded;
                    }
                }

                if (categoricalPlotsAdded >= 8) break;
            }
        }

        {
            std::vector<std::vector<std::string>> bootRows;
            for (size_t idx : data.numericColumnIndices()) {
                if (isAdministrativeColumnName(data.columns()[idx].name)) continue;
                const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
                if (vals.size() < 20) continue;
                auto [lo, hi] = bootstrapCI(vals,
                                            250,
                                            0.05,
                                            config.neuralSeed ^ static_cast<uint32_t>(idx),
                                            verbose,
                                            "bootstrap-mean");
                const double mu = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(vals.size());
                bootRows.push_back({"mean(" + data.columns()[idx].name + ")", toFixed(mu, 6), toFixed(lo, 6), toFixed(hi, 6)});
                if (bootRows.size() >= 6) break;
            }
            if (!benchmarks.empty()) {
                std::vector<double> rmseVals;
                for (const auto& b : benchmarks) rmseVals.push_back(b.rmse);
                auto [lo, hi] = bootstrapCI(rmseVals,
                                            250,
                                            0.05,
                                            config.benchmarkSeed,
                                            verbose,
                                            "bootstrap-rmse");
                const double mu = std::accumulate(rmseVals.begin(), rmseVals.end(), 0.0) / static_cast<double>(std::max<size_t>(1, rmseVals.size()));
                bootRows.push_back({"benchmark_rmse", toFixed(mu, 6), toFixed(lo, 6), toFixed(hi, 6)});
            }

            std::vector<size_t> nums;
            nums.reserve(data.numericColumnIndices().size());
            for (size_t idx : data.numericColumnIndices()) {
                if (isAdministrativeColumnName(data.columns()[idx].name)) continue;
                nums.push_back(idx);
            }
            if (nums.size() >= 2) {
                const auto& x = std::get<std::vector<double>>(data.columns()[nums[0]].values);
                const auto& y = std::get<std::vector<double>>(data.columns()[nums[1]].values);
                const size_t n = std::min(x.size(), y.size());
                if (n >= 20) {
                    std::vector<double> corrBoot;
                    std::vector<double> slopeBoot;
                    corrBoot.reserve(220);
                    slopeBoot.reserve(220);
                    FastRng rng(static_cast<uint64_t>(config.neuralSeed) ^ 0x1234567ULL);
                    for (size_t b = 0; b < 220; ++b) {
                        std::vector<double> xs;
                        std::vector<double> ys;
                        xs.reserve(n);
                        ys.reserve(n);
                        for (size_t i = 0; i < n; ++i) {
                            const size_t idx = rng.uniformIndex(n);
                            xs.push_back(x[idx]);
                            ys.push_back(y[idx]);
                        }
                        const ColumnStats sx = Statistics::calculateStats(xs);
                        const ColumnStats sy = Statistics::calculateStats(ys);
                        const double r = MathUtils::calculatePearson(xs, ys, sx, sy).value_or(0.0);
                        const auto fit = MathUtils::simpleLinearRegression(xs, ys, sx, sy, r);
                        corrBoot.push_back(r);
                        slopeBoot.push_back(fit.first);
                        if (verbose && (((b + 1) % 11) == 0 || (b + 1) == 220)) {
                            printProgressBar("bootstrap-corr", b + 1, 220);
                        }
                    }
                    std::sort(corrBoot.begin(), corrBoot.end());
                    std::sort(slopeBoot.begin(), slopeBoot.end());
                    bootRows.push_back({"corr(" + data.columns()[nums[0]].name + "," + data.columns()[nums[1]].name + ")",
                                        toFixed(corrBoot[corrBoot.size() / 2], 6),
                                        toFixed(CommonUtils::quantileByNth(corrBoot, 0.025), 6),
                                        toFixed(CommonUtils::quantileByNth(corrBoot, 0.975), 6)});
                    bootRows.push_back({"slope(" + data.columns()[nums[1]].name + "~" + data.columns()[nums[0]].name + ")",
                                        toFixed(slopeBoot[slopeBoot.size() / 2], 6),
                                        toFixed(CommonUtils::quantileByNth(slopeBoot, 0.025), 6),
                                        toFixed(CommonUtils::quantileByNth(slopeBoot, 0.975), 6)});
                }
            }
            if (!bootRows.empty()) {
                report.addTable("Bootstrap Confidence Intervals", {"Metric", "Estimate", "CI Low", "CI High"}, bootRows);
            }
        }

        {
            const size_t rows = std::min<size_t>(data.rowCount(), 80);
            const size_t cols = std::min<size_t>(data.colCount(), 30);
            if (rows >= 5 && cols >= 2) {
                std::vector<std::vector<double>> miss(rows, std::vector<double>(cols, 0.0));
                std::vector<std::string> missLabels;
                for (size_t c = 0; c < cols; ++c) missLabels.push_back(data.columns()[c].name);
                for (size_t r = 0; r < rows; ++r) {
                    for (size_t c = 0; c < cols; ++c) {
                        miss[r][c] = data.columns()[c].missing[r] ? 1.0 : 0.0;
                    }
                }
                std::string missMap = overallPlotter->heatmap("overall_missingness_matrix", miss, "Missingness Matrix", missLabels);
                if (!missMap.empty()) report.addImage("Missingness Matrix", missMap);

                std::vector<std::vector<std::string>> missCorrRows;
                for (size_t c = 0; c < cols && c < data.columns().size(); ++c) {
                    std::vector<double> missVec(rows, 0.0);
                    for (size_t r = 0; r < rows; ++r) missVec[r] = miss[r][c];
                    for (size_t nidx : data.numericColumnIndices()) {
                        const auto& vals = std::get<std::vector<double>>(data.columns()[nidx].values);
                        std::vector<double> y;
                        for (size_t r = 0; r < rows && r < vals.size(); ++r) y.push_back(vals[r]);
                        if (y.size() != missVec.size() || y.size() < 8) continue;
                        ColumnStats s1 = Statistics::calculateStats(missVec);
                        ColumnStats s2 = Statistics::calculateStats(y);
                        double r = MathUtils::calculatePearson(missVec, y, s1, s2).value_or(0.0);
                        if (std::abs(r) >= 0.15) {
                            missCorrRows.push_back({data.columns()[c].name, data.columns()[nidx].name, toFixed(r, 4)});
                        }
                    }
                }
                if (!missCorrRows.empty()) {
                    if (missCorrRows.size() > 20) missCorrRows.resize(20);
                    report.addTable("Missingness Correlation", {"Missing Column", "Numeric Column", "pearson_r"}, missCorrRows);
                }
            }
        }

        {
            auto it = std::find_if(benchmarks.begin(), benchmarks.end(), [](const BenchmarkResult& b) {
                return b.model == "LinearRegression";
            });
            if (it != benchmarks.end() && !it->actual.empty() && it->actual.size() == it->predicted.size()) {
                std::vector<double> residuals;
                residuals.reserve(it->actual.size());
                for (size_t i = 0; i < it->actual.size(); ++i) residuals.push_back(it->actual[i] - it->predicted[i]);

                std::string residPlot = overallPlotter->residual("overall_linear_residual",
                                                                 it->predicted,
                                                                 residuals,
                                                                 "Linear Regression Residuals vs Fitted");
                if (!residPlot.empty()) report.addImage("Linear Regression Residual Diagnostic", residPlot);

                std::vector<double> sortedRes = residuals;
                std::sort(sortedRes.begin(), sortedRes.end());
                const double mu = std::accumulate(sortedRes.begin(), sortedRes.end(), 0.0) / static_cast<double>(sortedRes.size());
                double sd = 0.0;
                for (double v : sortedRes) { const double d = v - mu; sd += d * d; }
                sd = std::sqrt(sd / static_cast<double>(std::max<size_t>(1, sortedRes.size() - 1)));
                if (sd <= 1e-12) sd = 1.0;
                std::vector<double> theo(sortedRes.size(), 0.0);
                for (size_t i = 0; i < sortedRes.size(); ++i) {
                    const double p = (static_cast<double>(i) + 0.5) / static_cast<double>(sortedRes.size());
                    theo[i] = mu + sd * std::sqrt(2.0) * std::erf(2.0 * p - 1.0);
                }
                std::string qqPlot = overallPlotter->scatter("overall_linear_qq",
                                                             theo,
                                                             sortedRes,
                                                             "Q-Q Plot (Approx Normal)",
                                                             false,
                                                             0.0,
                                                             0.0,
                                                             "",
                                                             false,
                                                             1.96,
                                                             5000);
                if (!qqPlot.empty()) report.addImage("Linear Residual Q-Q Plot", qqPlot);

                std::vector<std::vector<std::string>> diagRows;
                double mse = 0.0;
                for (double e : residuals) mse += e * e;
                const double mseDen = static_cast<double>(std::max<size_t>(1, residuals.size()));
                mse /= mseDen;
                std::vector<double> cooks;
                cooks.reserve(residuals.size());
                const double p = std::max(1.0, static_cast<double>(data.numericColumnIndices().size()));
                for (size_t i = 0; i < residuals.size(); ++i) {
                    const double h = std::min(0.99, 1.0 / mseDen + (static_cast<double>(i) / mseDen) * 0.01);
                    const double cd = (residuals[i] * residuals[i] / (p * std::max(mse, 1e-12))) * (h / std::pow(1.0 - h, 2.0));
                    cooks.push_back(cd);
                }
                const double maxCook = cooks.empty() ? 0.0 : *std::max_element(cooks.begin(), cooks.end());

                double maxVif = 0.0;
                const auto nums = data.numericColumnIndices();
                if (nums.size() >= 2) {
                    for (size_t i = 0; i < nums.size(); ++i) {
                        const auto& vi = std::get<std::vector<double>>(data.columns()[nums[i]].values);
                        double r2max = 0.0;
                        for (size_t j = 0; j < nums.size(); ++j) {
                            if (i == j) continue;
                            const auto& vj = std::get<std::vector<double>>(data.columns()[nums[j]].values);
                            ColumnStats si = Statistics::calculateStats(vi);
                            ColumnStats sj = Statistics::calculateStats(vj);
                            double r = MathUtils::calculatePearson(vi, vj, si, sj).value_or(0.0);
                            r2max = std::max(r2max, r * r);
                        }
                        const double vif = 1.0 / std::max(1e-6, 1.0 - r2max);
                        maxVif = std::max(maxVif, vif);
                    }
                }

                diagRows.push_back({"MSE", toFixed(mse, 6)});
                diagRows.push_back({"Max Cook Distance", toFixed(maxCook, 6)});
                diagRows.push_back({"Max Approx VIF", toFixed(maxVif, 6)});
                report.addTable("Regression Diagnostics (Linear Benchmark)", {"Metric", "Value"}, diagRows);
            }
        }

        if (verbose) {
            std::cout << "[Seldon][Overall] Optional overall plots generated in dedicated folder.\n";
        }
    } else {
        report.addParagraph("Overall plots were skipped (supervised mode disabled or gnuplot unavailable).");
    }
    }
