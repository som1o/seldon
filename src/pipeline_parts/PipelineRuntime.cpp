int AutomationPipeline::run(const AutoConfig& config) {
    AutoConfig runCfg = config;

    if (runCfg.lowMemoryMode) {
        runCfg.fastMode = true;
        runCfg.fastMaxBivariatePairs = std::min<size_t>(runCfg.fastMaxBivariatePairs, 600);
        runCfg.fastNeuralSampleRows = std::min<size_t>(runCfg.fastNeuralSampleRows, 10000);
        runCfg.neuralStreamingMode = true;
        runCfg.neuralStreamingChunkRows = std::min<size_t>(std::max<size_t>(128, runCfg.neuralStreamingChunkRows), 512);
        runCfg.neuralMaxOneHotPerColumn = std::min<size_t>(runCfg.neuralMaxOneHotPerColumn, 8);
        runCfg.featureEngineeringEnablePoly = false;
        runCfg.featureEngineeringEnableLog = false;
        runCfg.featureEngineeringMaxGeneratedColumns = std::min<size_t>(runCfg.featureEngineeringMaxGeneratedColumns, 64);
        if (CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
            runCfg.neuralStrategy = "fast";
        }
        runCfg.plotUnivariate = false;
        runCfg.plotOverall = false;
        runCfg.plotBivariateSignificant = true;
        runCfg.plotModesExplicit = true;
        runCfg.generateHtml = false;
    }

    runCfg.plot.format = "png";
    {
        namespace fs = std::filesystem;
        const fs::path datasetPath(runCfg.datasetPath);
        fs::path baseDir = datasetPath.parent_path();
        if (baseDir.empty()) baseDir = fs::current_path();
        const std::string stem = datasetPath.stem().string().empty() ? "dataset" : datasetPath.stem().string();
        if (runCfg.outputDir.empty()) {
            runCfg.outputDir = (baseDir / (stem + "_seldon_outputs")).string();
        }
        const fs::path outputPath(runCfg.outputDir);

        fs::path assetsPath(runCfg.assetsDir);
        if (runCfg.assetsDir.empty() || runCfg.assetsDir == "seldon_report_assets") {
            assetsPath = outputPath / "seldon_report_assets";
        } else if (assetsPath.is_relative()) {
            assetsPath = outputPath / assetsPath;
        }
        runCfg.assetsDir = assetsPath.string();

        fs::path reportPath(runCfg.reportFile);
        if (runCfg.reportFile.empty() || runCfg.reportFile == "neural_synthesis.md") {
            reportPath = outputPath / "neural_synthesis.md";
        } else if (reportPath.is_relative()) {
            reportPath = outputPath / reportPath;
        }
        runCfg.reportFile = reportPath.string();
    }
    CliProgressSpinner progress(!runCfg.verboseAnalysis && ::isatty(STDOUT_FILENO));
    constexpr size_t totalSteps = 10;
    size_t currentStep = 0;

    auto advance = [&](const std::string& label) {
        ++currentStep;
        progress.update(label, currentStep, totalSteps);
        if (AutomationPipeline::onProgress)
            AutomationPipeline::onProgress(label, static_cast<int>(currentStep),
                                           static_cast<int>(totalSteps));
    };

    cleanupOutputs(runCfg);
    MathUtils::setSignificanceAlpha(runCfg.tuning.significanceAlpha);
    MathUtils::setNumericTuning(runCfg.tuning.numericEpsilon,
                                runCfg.tuning.betaFallbackIntervalsStart,
                                runCfg.tuning.betaFallbackIntervalsMax,
                                runCfg.tuning.betaFallbackTolerance);

    TypedDataset data(config.datasetPath, config.delimiter);
    if (runCfg.numericLocaleHint == "us") {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::US_THOUSANDS);
    } else if (runCfg.numericLocaleHint == "eu") {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::EUROPEAN);
    } else {
        data.setNumericSeparatorPolicy(TypedDataset::NumericSeparatorPolicy::AUTO);
    }

    if (runCfg.datetimeLocaleHint == "dmy") {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::DMY);
    } else if (runCfg.datetimeLocaleHint == "mdy") {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::MDY);
    } else {
        data.setDateLocaleHint(TypedDataset::DateLocaleHint::AUTO);
    }

    if (!runCfg.columnTypeOverrides.empty()) {
        std::unordered_map<std::string, ColumnType> typeOverrides;
        typeOverrides.reserve(runCfg.columnTypeOverrides.size());
        for (const auto& kv : runCfg.columnTypeOverrides) {
            const std::string normalized = CommonUtils::toLower(CommonUtils::trim(kv.second));
            ColumnType type = ColumnType::CATEGORICAL;
            if (normalized == "numeric") {
                type = ColumnType::NUMERIC;
            } else if (normalized == "datetime") {
                type = ColumnType::DATETIME;
            }
            typeOverrides[CommonUtils::toLower(CommonUtils::trim(kv.first))] = type;
        }
        data.setColumnTypeOverrides(std::move(typeOverrides));
    }

    data.load();
    advance("Loaded dataset");
    if (data.rowCount() == 0 || data.colCount() == 0) {
        throw Seldon::DatasetException("Dataset has no usable rows/columns");
    }

    validateExcludedColumns(data, config);
    const size_t loadedColumnCount = data.colCount();
    const std::optional<std::string> protectedTargetName = config.targetColumn.empty()
        ? std::optional<std::string>{}
        : std::optional<std::string>{config.targetColumn};
    const PreflightCullSummary preflightCull = applyPreflightSparseColumnCull(data,
                                                                               protectedTargetName,
                                                                               runCfg.verboseAnalysis,
                                                                               0.95);
    if (data.colCount() == 0) {
        throw Seldon::DatasetException("All columns were removed by pre-flight missingness cull (>95% missing)");
    }
    const size_t rawColumnsAfterPreflight = data.colCount();
    const TypedDataset reportingData = data;
    const TargetContext targetContext = resolveTargetContext(data, config, runCfg);
    const int targetIdx = targetContext.targetIdx;
    applyDynamicPlotDefaultsIfUnset(runCfg, data);

    const bool autoFastMode = (data.rowCount() > 100000) || (data.numericColumnIndices().size() > 50);
    const bool fastModeEnabled = runCfg.fastMode || autoFastMode;
    if (fastModeEnabled && CommonUtils::toLower(runCfg.neuralStrategy) == "auto") {
        runCfg.neuralStrategy = "fast";
    }

    if (!runCfg.storeOutlierFlagsInReport && data.rowCount() <= 50000) {
        runCfg.storeOutlierFlagsInReport = true;
    }

    PreprocessReport prep = Preprocessor::run(data, runCfg);
    advance("Preprocessed dataset");
    exportPreprocessedDatasetIfRequested(data, runCfg);
    normalizeBinaryTarget(data, targetIdx, targetContext.semantics);
    const NumericStatsCache statsCache = buildNumericStatsCache(data);
    const NumericStatsCache rawStatsCache = buildNumericStatsCache(reportingData);
    advance("Prepared stats cache");

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Preparing deeply detailed univariate analysis...\n";
    }

    ReportEngine univariate;
    univariate.addTitle("Univariate Analysis");
    univariate.addParagraph("Dataset: " + runCfg.datasetPath);
    univariate.addParagraph("Bi-temporal mode: raw-state statistics are reported in original units; transformed-state features are used for neural modeling and ranking.");
    const size_t totalAnalysisDimensions = data.colCount();
    const size_t engineeredFeatureCount =
        (totalAnalysisDimensions >= rawColumnsAfterPreflight) ? (totalAnalysisDimensions - rawColumnsAfterPreflight) : 0;
    const bool strictFeatureReporting = engineeredFeatureCount >= std::max<size_t>(8, rawColumnsAfterPreflight / 3);
    const size_t fullTableDimensions = strictFeatureReporting ? rawColumnsAfterPreflight : totalAnalysisDimensions;
    univariate.addParagraph("Dataset Stats:");
    univariate.addParagraph("Rows: " + std::to_string(data.rowCount()));
    univariate.addParagraph("Raw Columns (loaded): " + std::to_string(loadedColumnCount));
    univariate.addParagraph("Raw Columns (post pre-flight cull): " + std::to_string(rawColumnsAfterPreflight));
    if (preflightCull.dropped > 0) {
        univariate.addParagraph("Pre-flight cull removed " + std::to_string(preflightCull.dropped) + " sparse columns (>" + toFixed(100.0 * preflightCull.threshold, 1) + "% missing) before univariate profiling.");
    }
    univariate.addParagraph("Engineered Features: " + std::to_string(engineeredFeatureCount));
    univariate.addParagraph("Total Analysis Dimensions: " + std::to_string(totalAnalysisDimensions));
    if (strictFeatureReporting) {
        univariate.addParagraph("Dimensions expanded in full univariate/bivariate tables: " + std::to_string(fullTableDimensions));
    }
    if (strictFeatureReporting) {
        univariate.addParagraph("Strict feature-reporting mode enabled: engineered features are summarized, not expanded into full univariate/bivariate tables.");
        std::unordered_set<std::string> engineeredFamilies;
        std::unordered_set<std::string> unitAwareFamilies;
        std::unordered_map<std::string, size_t> familyCounts;
        auto inferUnitSemantic = [](const std::string& name) {
            const std::string lower = CommonUtils::toLower(CommonUtils::trim(name));
            if (lower.empty()) return std::string("unspecified");
            auto hasAny = [&](std::initializer_list<const char*> tokens) {
                for (const char* token : tokens) {
                    if (lower.find(token) != std::string::npos) return true;
                }
                return false;
            };
            if (hasAny({"price", "cost", "revenue", "salary", "income", "expense", "budget", "amount", "usd", "eur", "gbp", "jpy"})) return std::string("currency");
            if (hasAny({"percent", "percentage", "pct", "%"})) return std::string("percentage");
            if (hasAny({"ratio"})) return std::string("ratio");
            if (hasAny({"rate", "per_", "per ", "/"})) return std::string("rate");
            if (hasAny({"count", "qty", "quantity", "num", "number", "cases", "users", "visits", "clicks", "population"})) return std::string("count");
            if (hasAny({"duration", "latency", "time", "seconds", "second", "minutes", "minute", "hours", "hour", "days", "day", "ms"})) return std::string("duration");
            if (hasAny({"distance", "km", "kilometer", "mile", "meter"})) return std::string("distance");
            if (hasAny({"temp", "temperature", "celsius", "fahrenheit", "kelvin"})) return std::string("temperature");
            if (hasAny({"score", "index", "rating"})) return std::string("score/index");
            return std::string("unspecified");
        };
        auto engineeredFamilyLabel = [&](const std::string& featureName) {
            const std::string lower = CommonUtils::toLower(CommonUtils::trim(featureName));
            if (lower.find("_div_") != std::string::npos) {
                const size_t pos = lower.find("_div_");
                const std::string numer = CommonUtils::trim(lower.substr(0, pos));
                const std::string denom = CommonUtils::trim(lower.substr(pos + 5));
                const std::string nu = inferUnitSemantic(numer);
                const std::string du = inferUnitSemantic(denom);
                if (nu == du && nu != "unspecified") return std::string("ratio-normalization(") + nu + ")";
                if ((nu == "currency" && du == "count") || (nu == "count" && du == "currency")) return std::string("rate(currency/count)");
                return std::string("cross-unit ratio(") + nu + "/" + du + ")";
            }
            if (lower.find("_x_") != std::string::npos) return std::string("interaction");
            if (lower.find("_pow") != std::string::npos) return std::string("polynomial");
            if (lower.find("_log1p_abs") != std::string::npos) return std::string("log-scale");
            return std::string("engineered");
        };
        std::vector<std::vector<std::string>> engineeredRows;
        for (const auto& col : data.columns()) {
            if (!isEngineeredFeatureName(col.name)) continue;
            const std::string base = canonicalEngineeredBaseName(col.name);
            if (!base.empty()) engineeredFamilies.insert(base);
            const std::string family = engineeredFamilyLabel(col.name);
            unitAwareFamilies.insert(family);
            familyCounts[family] += 1;
            if (engineeredRows.size() < 12) {
                engineeredRows.push_back({col.name, base.empty() ? "-" : base, family});
            }
        }
        std::string topFamily = "n/a";
        size_t topFamilyCount = 0;
        for (const auto& kv : familyCounts) {
            if (kv.second <= topFamilyCount) continue;
            topFamily = kv.first;
            topFamilyCount = kv.second;
        }
        univariate.addTable("Feature Engineering Insights", {"Metric", "Value"}, {
            {"Engineered Features (suppressed from full tables)", std::to_string(engineeredFeatureCount)},
            {"Engineered Feature Families", std::to_string(engineeredFamilies.size())},
            {"Unit-Aware Family Types", std::to_string(unitAwareFamilies.size())},
            {"Dominant Family", topFamily + " (" + std::to_string(topFamilyCount) + ")"},
            {"Reporting Policy", "collapsed (strict mode)"}
        });
        if (!engineeredRows.empty()) {
            univariate.addTable("Feature Engineering Sample", {"Engineered Feature", "Base Feature", "Semantic Family"}, engineeredRows);
        }
    }
    addExecutiveDashboard(
        univariate,
        "Executive Dashboard",
        {
            {"Rows", std::to_string(data.rowCount())},
            {"Raw Columns", std::to_string(loadedColumnCount)},
            {"Post-Cull Columns", std::to_string(rawColumnsAfterPreflight)},
            {"Engineered Features", std::to_string(engineeredFeatureCount)},
            {"Analysis Dimensions", std::to_string(totalAnalysisDimensions)}
        },
        {
            preflightCull.dropped > 0
                ? ("Pre-flight sparse-column cull removed " + std::to_string(preflightCull.dropped) + " highly missing columns.")
                : "No severe pre-flight sparse-column issues were detected.",
            strictFeatureReporting
                ? "Strict feature-reporting is active to keep the report compact and readable."
                : "Full-dimension descriptive profiling is enabled."
        },
        "Quick non-technical view before deep descriptive tables."
    );
    addUnivariateDetailedSection(univariate, reportingData, prep, runCfg.verboseAnalysis, rawStatsCache);
    advance("Built univariate tables");

    GnuplotEngine plotterBivariate(plotSubdir(runCfg, "bivariate"), runCfg.plot);
    GnuplotEngine plotterUnivariate(plotSubdir(runCfg, "univariate"), runCfg.plot);
    GnuplotEngine plotterOverall(plotSubdir(runCfg, "overall"), runCfg.plot);
    const bool canPlot = configurePlotAvailability(runCfg, univariate, plotterBivariate);

    FeatureSelectionResult selectedFeatures = collectFeatureIndices(data, targetIdx, runCfg, prep);
    std::vector<int> featureIdx = selectedFeatures.included;
    DeterministicHeuristics::Outcome deterministic = DeterministicHeuristics::runAllPhases(data, prep, targetIdx, featureIdx);
    if (!deterministic.filteredFeatures.empty()) {
        featureIdx = deterministic.filteredFeatures;
    }

    std::unordered_set<size_t> targetContaminationIndices;
    std::vector<std::vector<std::string>> targetContaminationRows;
    {
        if (targetIdx >= 0 && static_cast<size_t>(targetIdx) < data.columns().size() &&
            data.columns()[static_cast<size_t>(targetIdx)].type == ColumnType::NUMERIC) {
            const size_t targetIdxU = static_cast<size_t>(targetIdx);
            const auto& targetCol = data.columns()[targetIdxU];
            const auto& targetVals = std::get<std::vector<double>>(targetCol.values);

            auto absCorrWithTargetObserved = [&](size_t featureIdxU) {
                if (featureIdxU >= data.columns().size()) return 0.0;
                if (data.columns()[featureIdxU].type != ColumnType::NUMERIC) return 0.0;
                const auto& featureCol = data.columns()[featureIdxU];
                const auto& featureVals = std::get<std::vector<double>>(featureCol.values);
                const size_t n = std::min({targetVals.size(), featureVals.size(), targetCol.missing.size(), featureCol.missing.size()});
                std::vector<double> fx;
                std::vector<double> ty;
                fx.reserve(n);
                ty.reserve(n);
                for (size_t r = 0; r < n; ++r) {
                    if (targetCol.missing[r] || featureCol.missing[r]) continue;
                    if (!std::isfinite(targetVals[r]) || !std::isfinite(featureVals[r])) continue;
                    fx.push_back(featureVals[r]);
                    ty.push_back(targetVals[r]);
                }
                if (fx.size() < 16) return 0.0;
                const auto fsIt = statsCache.find(featureIdxU);
                const auto tsIt = statsCache.find(targetIdxU);
                const ColumnStats fs = (fsIt != statsCache.end()) ? fsIt->second : Statistics::calculateStats(fx);
                const ColumnStats ts = (tsIt != statsCache.end()) ? tsIt->second : Statistics::calculateStats(ty);
                return std::abs(MathUtils::calculatePearson(fx, ty, fs, ts).value_or(0.0));
            };

            constexpr double kTargetContaminationThreshold = 0.99;
            const auto numericCols = data.numericColumnIndices();
            for (size_t idx : numericCols) {
                if (idx == targetIdxU) continue;
                const double absCorr = absCorrWithTargetObserved(idx);
                if (!std::isfinite(absCorr) || absCorr <= kTargetContaminationThreshold) continue;
                targetContaminationIndices.insert(idx);
                targetContaminationRows.push_back({
                    data.columns()[idx].name,
                    toFixed(absCorr, 6),
                    "Potential Target Contamination (auto-flagged)"
                });
            }

            if (!targetContaminationIndices.empty()) {
                std::vector<int> cleaned;
                cleaned.reserve(featureIdx.size());
                for (int idx : featureIdx) {
                    if (idx < 0) continue;
                    if (targetContaminationIndices.count(static_cast<size_t>(idx)) > 0) continue;
                    cleaned.push_back(idx);
                }
                if (!cleaned.empty()) featureIdx = std::move(cleaned);

                for (size_t idx : targetContaminationIndices) {
                    if (idx >= data.columns().size()) continue;
                    deterministic.excludedReasonLines.push_back(data.columns()[idx].name + ": excluded as potential target contamination (|corr(target)|>0.99)");
                }
            }
        }
    }

    std::unordered_set<size_t> identityBlockedFeatureIndices;
    {
        constexpr double kIdentityCorrThreshold = 0.98;
        auto absCorrAligned = [&](size_t idxA, size_t idxB) {
            if (idxA >= data.columns().size() || idxB >= data.columns().size()) return 0.0;
            if (data.columns()[idxA].type != ColumnType::NUMERIC || data.columns()[idxB].type != ColumnType::NUMERIC) return 0.0;
            const auto& colA = data.columns()[idxA];
            const auto& colB = data.columns()[idxB];
            const auto& a = std::get<std::vector<double>>(colA.values);
            const auto& b = std::get<std::vector<double>>(colB.values);
            const size_t n = std::min({a.size(), b.size(), colA.missing.size(), colB.missing.size()});
            std::vector<double> xa;
            std::vector<double> xb;
            xa.reserve(n);
            xb.reserve(n);
            for (size_t r = 0; r < n; ++r) {
                if (colA.missing[r] || colB.missing[r]) continue;
                if (!std::isfinite(a[r]) || !std::isfinite(b[r])) continue;
                xa.push_back(a[r]);
                xb.push_back(b[r]);
            }
            if (xa.size() < 16) return 0.0;
            const auto saIt = statsCache.find(idxA);
            const auto sbIt = statsCache.find(idxB);
            const ColumnStats sa = (saIt != statsCache.end()) ? saIt->second : Statistics::calculateStats(xa);
            const ColumnStats sb = (sbIt != statsCache.end()) ? sbIt->second : Statistics::calculateStats(xb);
            return std::abs(MathUtils::calculatePearson(xa, xb, sa, sb).value_or(0.0));
        };

        std::vector<size_t> numericFeatureIdx;
        numericFeatureIdx.reserve(featureIdx.size());
        for (int idx : featureIdx) {
            if (idx < 0 || static_cast<size_t>(idx) >= data.columns().size()) continue;
            if (data.columns()[static_cast<size_t>(idx)].type != ColumnType::NUMERIC) continue;
            numericFeatureIdx.push_back(static_cast<size_t>(idx));
        }

        for (size_t i = 0; i < numericFeatureIdx.size(); ++i) {
            const size_t idxA = numericFeatureIdx[i];
            if (identityBlockedFeatureIndices.count(idxA) > 0) continue;
            for (size_t j = i + 1; j < numericFeatureIdx.size(); ++j) {
                const size_t idxB = numericFeatureIdx[j];
                if (identityBlockedFeatureIndices.count(idxB) > 0) continue;
                const double corr = absCorrAligned(idxA, idxB);
                if (corr < kIdentityCorrThreshold) continue;

                const auto saIt = statsCache.find(idxA);
                const auto sbIt = statsCache.find(idxB);
                const double varA = (saIt != statsCache.end() && std::isfinite(saIt->second.variance)) ? std::max(0.0, saIt->second.variance) : 0.0;
                const double varB = (sbIt != statsCache.end() && std::isfinite(sbIt->second.variance)) ? std::max(0.0, sbIt->second.variance) : 0.0;
                const size_t dropIdx = (varA < varB) ? idxA : (varB < varA ? idxB : std::max(idxA, idxB));
                identityBlockedFeatureIndices.insert(dropIdx);
            }
        }

        if (!identityBlockedFeatureIndices.empty()) {
            std::vector<int> kept;
            kept.reserve(featureIdx.size());
            for (int idx : featureIdx) {
                if (idx < 0) continue;
                if (identityBlockedFeatureIndices.count(static_cast<size_t>(idx)) > 0) continue;
                kept.push_back(idx);
            }
            if (!kept.empty()) {
                featureIdx = std::move(kept);
            }
            for (size_t idx : identityBlockedFeatureIndices) {
                if (idx >= data.columns().size()) continue;
                deterministic.excludedReasonLines.push_back(data.columns()[idx].name + ": identity block excluded (|r|>0.98 with higher-variance peer)");
            }
        }
    }
    std::unordered_set<std::string> identityBlockedColumnNames;
    identityBlockedColumnNames.reserve(identityBlockedFeatureIndices.size());
    for (size_t idx : identityBlockedFeatureIndices) {
        if (idx >= data.columns().size()) continue;
        identityBlockedColumnNames.insert(data.columns()[idx].name);
    }

    if (deterministic.lowRatioMode) {
        if (runCfg.neuralFixedLayers == 0) runCfg.neuralFixedLayers = 1;
        if (runCfg.neuralFixedHiddenNodes == 0) runCfg.neuralFixedHiddenNodes = 6;
        runCfg.neuralStrategy = "fast";
    } else if (deterministic.highRatioMode) {
        runCfg.neuralMinLayers = std::max(runCfg.neuralMinLayers, 3);
        runCfg.neuralMaxLayers = std::max(runCfg.neuralMaxLayers, 3);
        if (runCfg.neuralFixedHiddenNodes == 0) {
            runCfg.neuralFixedHiddenNodes = std::max(runCfg.neuralFixedHiddenNodes, 50);
        }
        if (CommonUtils::toLower(runCfg.neuralStrategy) == "auto") runCfg.neuralStrategy = "expressive";
    }

    if (runCfg.verboseAnalysis && !selectedFeatures.droppedByMissingness.empty()) {
        std::cout << "[Seldon][Features] Dropped sparse features (>"
                  << toFixed(selectedFeatures.missingThresholdUsed, 2)
                  << " missing ratio, strategy=" << selectedFeatures.strategyUsed << "):\n";
        for (const auto& item : selectedFeatures.droppedByMissingness) {
            std::cout << "  - " << item << "\n";
        }
        for (const auto& item : deterministic.excludedReasonLines) {
            std::cout << "  - " << item << "\n";
        }
    }

    const std::vector<int> benchmarkTargetIndices = (runCfg.neuralMultiOutput && runCfg.neuralMaxAuxTargets > 0)
        ? selectAuxiliaryNumericTargets(data, targetIdx, runCfg.neuralMaxAuxTargets, statsCache)
        : std::vector<int>{targetIdx};

    auto benchmarks = BenchmarkEngine::run(data, targetIdx, featureIdx, runCfg.kfold, runCfg.benchmarkSeed);
    const MultiTargetBenchmarkSummary multiTargetBenchmarks = (benchmarkTargetIndices.size() > 1)
        ? BenchmarkEngine::runMultiTarget(data,
                                          benchmarkTargetIndices,
                                          featureIdx,
                                          runCfg.kfold,
                                          runCfg.benchmarkSeed ^ 0x6a09e667U)
        : MultiTargetBenchmarkSummary{};
    advance("Finished benchmarks");

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Neural] Starting neural network training with verbose trace...\n";
    }
    NeuralAnalysis neural = runNeuralAnalysis(data,
                                              targetIdx,
                                              featureIdx,
                                              targetContext.semantics.inferredTask != "regression",
                                              targetContext.semantics.isOrdinal,
                                              targetContext.semantics.cardinality,
                                              runCfg.verboseAnalysis,
                                              runCfg,
                                              fastModeEnabled,
                                              runCfg.fastNeuralSampleRows);
    advance("Completed neural analysis");
    neural.featureImportance = buildCoherentImportance(data, targetIdx, featureIdx, neural, benchmarks, runCfg, statsCache);
    const std::unordered_set<size_t> neuralApprovedNumericFeatures = computeNeuralApprovedNumericFeatures(data,
                                                                                                           targetIdx,
                                                                                                           featureIdx,
                                                                                                           neural,
                                                                                                           statsCache);
    addUnivariatePlots(univariate,
                       data,
                       runCfg,
                       canPlot,
                       plotterUnivariate,
                       neuralApprovedNumericFeatures);
    advance("Generated univariate plots");

    BivariateScoringPolicy bivariatePolicy = chooseBivariatePolicy(runCfg, neural);

    std::unordered_map<size_t, double> importanceByIndex;
    double fallbackImportance = 0.0;
    for (double v : neural.featureImportance) fallbackImportance += v;
    if (!neural.featureImportance.empty()) fallbackImportance /= neural.featureImportance.size();
    for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size(); ++i) {
        importanceByIndex[static_cast<size_t>(featureIdx[i])] = neural.featureImportance[i];
    }
    importanceByIndex[static_cast<size_t>(targetIdx)] = fallbackImportance;

    const DomainRuleBundle domainRules = loadDomainRules(runCfg);

    const auto numeric = data.numericColumnIndices();
    for (size_t idx : numeric) {
        if (importanceByIndex.find(idx) == importanceByIndex.end()) {
            const auto& x = std::get<std::vector<double>>(data.columns()[idx].values);
            const auto& y = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(targetIdx)].values);
            const auto xIt = statsCache.find(idx);
            const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
            const ColumnStats xStats = (xIt != statsCache.end()) ? xIt->second : Statistics::calculateStats(x);
            const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);
            double corr = std::abs(MathUtils::calculatePearson(x, y, xStats, yStats).value_or(0.0));
            importanceByIndex[idx] = std::isfinite(corr) ? corr : fallbackImportance;
        }
    }

    if (!domainRules.downweightImportanceColumns.empty()) {
        for (size_t idx : numeric) {
            if (idx >= data.columns().size()) continue;
            const std::string norm = normalizeRuleName(data.columns()[idx].name);
            if (domainRules.downweightImportanceColumns.count(norm) == 0) continue;
            auto it = importanceByIndex.find(idx);
            if (it == importanceByIndex.end()) continue;
            it->second *= 0.25;
        }
    }

        std::unordered_set<size_t> modeledIndices;
    modeledIndices.insert(static_cast<size_t>(targetIdx));
    for (int idx : featureIdx) modeledIndices.insert(static_cast<size_t>(idx));

    GnuplotEngine* bivariatePlotter = (canPlot && runCfg.plotBivariateSignificant) ? &plotterBivariate : nullptr;
    const size_t totalPossiblePairs = numeric.size() < 2
        ? 0
        : (numeric.size() * (numeric.size() - 1)) / 2;
    auto bivariatePairs = analyzeBivariatePairs(data,
                                                importanceByIndex,
                                                modeledIndices,
                                                bivariatePolicy,
                                                bivariatePlotter,
                                                runCfg.verboseAnalysis,
                                                statsCache,
                                                runCfg.tuning.numericEpsilon,
                                                runCfg.tuning,
                                                fastModeEnabled ? runCfg.fastMaxBivariatePairs : 0,
                                                std::max<size_t>(8, std::min<size_t>(120, neuralApprovedNumericFeatures.size() * 6)));

    std::unordered_set<std::string> targetContaminatedNames;
    targetContaminatedNames.reserve(targetContaminationIndices.size());
    for (size_t idx : targetContaminationIndices) {
        if (idx >= data.columns().size()) continue;
        targetContaminatedNames.insert(data.columns()[idx].name);
    }

    const auto stratifiedPopulationsRaw = detectStratifiedPopulations(data, 12);

    const auto textMentionsContamination = [&](const std::string& text) {
        if (targetContaminatedNames.empty() || text.empty()) return false;
        for (const auto& name : targetContaminatedNames) {
            if (name.empty()) continue;
            if (text == name || text.find(name) != std::string::npos) return true;
        }
        return false;
    };

    const auto rowMentionsContamination = [&](const std::vector<std::string>& row) {
        for (const auto& cell : row) {
            if (textMentionsContamination(cell)) return true;
        }
        return false;
    };

    auto stratifiedPopulations = stratifiedPopulationsRaw;
    stratifiedPopulations.erase(
        std::remove_if(stratifiedPopulations.begin(), stratifiedPopulations.end(), [&](const auto& s) {
            return textMentionsContamination(s.segmentColumn) || textMentionsContamination(s.numericColumn) || textMentionsContamination(s.groupMeans);
        }),
        stratifiedPopulations.end());

    size_t contaminationSuppressedPairs = 0;
    if (!targetContaminatedNames.empty()) {
        const auto before = bivariatePairs.size();
        bivariatePairs.erase(std::remove_if(bivariatePairs.begin(), bivariatePairs.end(), [&](const PairInsight& p) {
            return targetContaminatedNames.count(p.featureA) > 0 || targetContaminatedNames.count(p.featureB) > 0;
        }), bivariatePairs.end());
        contaminationSuppressedPairs = before - bivariatePairs.size();
    }
    advance("Analyzed bivariate pairs");

    ReportEngine bivariate;
    bivariate.addTitle("Bivariate Analysis");
    if (totalPossiblePairs > bivariatePairs.size()) {
        bivariate.addParagraph("Fast mode active: pair evaluation was capped for runtime safety. Results below cover the highest-priority numeric columns only.");
    } else {
        bivariate.addParagraph("All numeric pair combinations are included below (nC2). Significant table is dynamically filtered using a multi-tier gate: statistical significance, neural relevance, and a bounded fallback for high-effect stable pairs.");
    }
    bivariate.addParagraph("Neural relevance score prioritizes practical effect size over raw p-value magnitude; when neural filtering is too strict, statistically robust pairs can be promoted as Tier-3 domain findings.");

    std::vector<std::vector<std::string>> allRows;
    std::vector<std::vector<std::string>> sigRows;
    std::vector<std::vector<std::string>> structuralRows;
    const bool compactBivariateRows = runCfg.lowMemoryMode;
    const size_t compactRejectedCap = 256;
    size_t compactRejectedCount = 0;
    size_t statSigCount = 0;
    size_t strictSuppressedPairs = 0;
    size_t strictSuppressedSelected = 0;
    size_t identitySuppressedPairs = 0;
    auto plotFlagText = [&](const PairInsight& p, bool enabled) {
        if (!canPlot) return std::string("n/a");
        if (!p.selected) return std::string("n/a");
        if (p.plotPath.empty()) return std::string("n/a");
        return enabled ? std::string("yes") : std::string("no");
    };
    for (const auto& p : bivariatePairs) {
        if (identityBlockedColumnNames.count(p.featureA) > 0 || identityBlockedColumnNames.count(p.featureB) > 0) {
            ++identitySuppressedPairs;
            continue;
        }
        const bool engineeredPair = isEngineeredFeatureName(p.featureA) || isEngineeredFeatureName(p.featureB);
        if (strictFeatureReporting && engineeredPair) {
            ++strictSuppressedPairs;
            if (p.selected) ++strictSuppressedSelected;
            continue;
        }
        if (p.statSignificant) statSigCount++;
        const bool deterministicPair = std::abs(p.r) >= 0.999;
        if (p.filteredAsStructural || deterministicPair) {
            structuralRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.pValue, 6),
                p.filteredAsStructural ? "yes" : "no",
                deterministicPair ? "yes" : "no",
                p.relationLabel.empty() ? "-" : p.relationLabel,
                p.selectionReason.empty() ? "-" : p.selectionReason
            });
            continue;
        }
        if (!compactBivariateRows || p.selected || compactRejectedCount < compactRejectedCap) {
            allRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.spearman),
                toFixed(p.kendallTau),
                toFixed(p.r2),
                toFixed(p.effectSize, 6),
                toFixed(p.foldStability, 6),
                toFixed(p.slope),
                toFixed(p.intercept),
                toFixed(p.tStat, 6),
                toFixed(p.pValue, 6),
                p.statSignificant ? "yes" : "no",
                toFixed(p.neuralScore, 6),
                std::to_string(p.significanceTier),
                p.selectionReason.empty() ? "-" : p.selectionReason,
                p.selected ? "yes" : "no",
                p.filteredAsRedundant ? "yes" : "no",
                p.filteredAsStructural ? "yes" : "no",
                p.leakageRisk ? "yes" : "no",
                p.redundancyGroup.empty() ? "-" : p.redundancyGroup,
                p.relationLabel.empty() ? "-" : p.relationLabel,
                plotFlagText(p, p.fitLineAdded),
                plotFlagText(p, p.confidenceBandAdded),
                p.stackedPlotPath.empty() ? "-" : p.stackedPlotPath,
                p.residualPlotPath.empty() ? "-" : p.residualPlotPath,
                p.facetedPlotPath.empty() ? "-" : p.facetedPlotPath,
                p.plotPath.empty() ? "-" : p.plotPath
            });
            if (compactBivariateRows && !p.selected) {
                ++compactRejectedCount;
            }
        }

        if (p.selected) {
            sigRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.spearman),
                toFixed(p.kendallTau),
                toFixed(p.r2),
                toFixed(p.effectSize, 6),
                toFixed(p.foldStability, 6),
                toFixed(p.slope),
                toFixed(p.intercept),
                toFixed(p.tStat, 6),
                toFixed(p.pValue, 6),
                toFixed(p.neuralScore, 6),
                std::to_string(p.significanceTier),
                p.selectionReason.empty() ? "-" : p.selectionReason,
                p.relationLabel.empty() ? "-" : p.relationLabel,
                plotFlagText(p, p.fitLineAdded),
                plotFlagText(p, p.confidenceBandAdded),
                p.stackedPlotPath.empty() ? "-" : p.stackedPlotPath,
                p.residualPlotPath.empty() ? "-" : p.residualPlotPath,
                p.facetedPlotPath.empty() ? "-" : p.facetedPlotPath,
                p.plotPath.empty() ? "-" : p.plotPath
            });
            if (!p.plotPath.empty()) {
                bivariate.addImage("Significant Pair: " + p.featureA + " vs " + p.featureB, p.plotPath);
            }
            if (!p.stackedPlotPath.empty()) {
                bivariate.addImage("Stacked Profile: " + p.featureA + " vs " + p.featureB, p.stackedPlotPath);
            }
            if (!p.residualPlotPath.empty()) {
                bivariate.addImage("Residual Plot: " + p.featureA + " vs " + p.featureB, p.residualPlotPath);
            }
            if (!p.facetedPlotPath.empty()) {
                bivariate.addImage("Faceted Scatter: " + p.featureA + " vs " + p.featureB, p.facetedPlotPath);
            }
        }
    }

    std::vector<std::string> topTakeaways;
    std::vector<std::string> narrativeInsights;
    std::string neuralInteractionTakeaway;
    std::unordered_set<size_t> usedFeatures;
    std::vector<PairInsight> rankedTakeaways;
    for (const auto& p : bivariatePairs) {
        if (!p.selected) continue;
        if (p.filteredAsRedundant || p.filteredAsStructural || p.leakageRisk) continue;
        rankedTakeaways.push_back(p);
    }
    std::sort(rankedTakeaways.begin(), rankedTakeaways.end(), [](const PairInsight& a, const PairInsight& b) {
        if (a.effectSize == b.effectSize) return a.neuralScore > b.neuralScore;
        return a.effectSize > b.effectSize;
    });
    for (const auto& p : rankedTakeaways) {
        if (topTakeaways.size() >= 3) break;
        if (usedFeatures.count(p.idxA) || usedFeatures.count(p.idxB)) continue;
        std::string label;
        if (std::abs(p.r) >= 0.90) {
            label = p.featureA + " is a strong proxy for " + p.featureB;
        } else if (std::abs(p.r) >= 0.70) {
            label = p.featureA + " and " + p.featureB + " encode overlapping signal";
        } else {
            label = p.featureA + " and " + p.featureB + " show moderate but stable association";
        }
        label += " (effect=" + toFixed(p.effectSize, 3) + ", stability=" + toFixed(p.foldStability, 3) + ")";
        topTakeaways.push_back(label);

        std::string narrative;
        if (std::abs(p.r) >= 0.85) {
            narrative = p.featureA + " effectively predicts " + p.featureB + " with high stability (|r|=" + toFixed(std::abs(p.r), 3) + ").";
        } else if (std::abs(p.r) >= 0.65) {
            narrative = p.featureA + " is a strong directional signal for " + p.featureB + " and can serve as a stable proxy.";
        } else {
            narrative = p.featureA + " contributes a moderate but consistent relationship with " + p.featureB + ".";
        }
        narrativeInsights.push_back(narrative);

        usedFeatures.insert(p.idxA);
        usedFeatures.insert(p.idxB);
    }

    {
        std::vector<std::pair<std::string, double>> skewed;
        for (size_t idx : data.numericColumnIndices()) {
            const auto& col = data.columns()[idx];
            if (isAdministrativeColumnName(col.name)) continue;
            ColumnStats st = Statistics::calculateStats(std::get<std::vector<double>>(col.values));
            if (!std::isfinite(st.skewness) || std::abs(st.skewness) < 1.2) continue;
            skewed.push_back({col.name, std::abs(st.skewness)});
        }
        std::sort(skewed.begin(), skewed.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < std::min<size_t>(2, skewed.size()); ++i) {
            narrativeInsights.push_back(skewed[i].first + " is heavily concentrated toward one side of its distribution (|skew|=" + toFixed(skewed[i].second, 2) + "), so tail behavior drives aggregate trends.");
        }
    }

    for (const auto& badge : deterministic.badgeNarratives) {
        narrativeInsights.push_back(badge);
    }

    if (!rankedTakeaways.empty()) {
        const PairInsight* bestInteraction = nullptr;
        double bestInteractionScore = -1.0;
        for (const auto& p : rankedTakeaways) {
            const double absCorr = std::clamp(std::abs(p.r), 0.0, 1.0);
            const double nonLinearHint = 1.0 - (0.50 * absCorr);
            const double interactionScore = p.neuralScore * (0.60 + 0.40 * p.foldStability) * nonLinearHint;
            if (interactionScore > bestInteractionScore) {
                bestInteractionScore = interactionScore;
                bestInteraction = &p;
            }
        }

        if (bestInteraction) {
            const std::string interactionType = (std::abs(bestInteraction->r) < 0.55)
                ? "complementary"
                : "reinforcing";
            neuralInteractionTakeaway = "Neural Interaction: " + bestInteraction->featureA + " × " + bestInteraction->featureB +
                " form a " + interactionType + " predictive interaction (neural=" + toFixed(bestInteraction->neuralScore, 3) +
                ", effect=" + toFixed(bestInteraction->effectSize, 3) +
                ", stability=" + toFixed(bestInteraction->foldStability, 3) + ").";
        }
    }

    std::vector<std::string> redundancyDrops;
    std::unordered_set<std::string> seenDrop;
    for (const auto& p : bivariatePairs) {
        if (!p.filteredAsRedundant || p.redundancyGroup.empty()) continue;
        const double impA = importanceByIndex.count(p.idxA) ? importanceByIndex[p.idxA] : 0.0;
        const double impB = importanceByIndex.count(p.idxB) ? importanceByIndex[p.idxB] : 0.0;
        const std::string drop = (impA < impB) ? p.featureA : p.featureB;
        const std::string keep = (impA < impB) ? p.featureB : p.featureA;
        const std::string dropLower = CommonUtils::toLower(CommonUtils::trim(drop));
        const std::string targetLower = CommonUtils::toLower(CommonUtils::trim(runCfg.targetColumn));
        if (dropLower == targetLower) continue;
        const std::string recommendation = "drop " + drop + " (redundant with " + keep + ")";
        if (seenDrop.insert(recommendation).second) {
            redundancyDrops.push_back(recommendation);
            if (redundancyDrops.size() >= 8) break;
        }
    }

    bivariate.addParagraph("Total pairs evaluated: " + std::to_string(bivariatePairs.size()));
    bivariate.addParagraph("Statistically significant pairs (p<" + toFixed(MathUtils::getSignificanceAlpha(), 4) + "): " + std::to_string(statSigCount));
    bivariate.addParagraph("Final selected significant pairs: " + std::to_string(sigRows.size()));
    const size_t tier2Count = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) {
        return p.selected && p.significanceTier == 2;
    }));
    const size_t tier3Count = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) {
        return p.selected && p.significanceTier == 3;
    }));
    bivariate.addParagraph("Selection tiers: Tier-2(neural+stat)=" + std::to_string(tier2Count) + ", Tier-3(domain fallback)=" + std::to_string(tier3Count) + ".");
    const size_t redundantPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsRedundant; }));
    const size_t structuralPairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.filteredAsStructural; }));
    const size_t leakagePairs = static_cast<size_t>(std::count_if(bivariatePairs.begin(), bivariatePairs.end(), [](const PairInsight& p) { return p.leakageRisk; }));
    bivariate.addParagraph("Information-theoretic filtering: redundant=" + std::to_string(redundantPairs) + ", structural=" + std::to_string(structuralPairs) + ", leakage-risk=" + std::to_string(leakagePairs) + ".");
    if (!canPlot) {
        bivariate.addParagraph("Plot columns are marked 'n/a' because visual generation is omitted (environment/runtime mode), not because relationships failed quality checks.");
    }
    if (strictFeatureReporting) {
        bivariate.addParagraph("Strict feature-reporting mode: suppressed " + std::to_string(strictSuppressedPairs) + " engineered-feature pairs (" + std::to_string(strictSuppressedSelected) + " were otherwise selected). See Univariate 'Feature Engineering Insights'.");
    }
    if (contaminationSuppressedPairs > 0) {
        bivariate.addParagraph("Target-contamination guard: suppressed " + std::to_string(contaminationSuppressedPairs) + " bivariate pairs involving near-target-clone features (|corr(target)|>0.99). These are excluded from statistical findings.");
    }
    if (identitySuppressedPairs > 0) {
        bivariate.addParagraph("Identity block: suppressed " + std::to_string(identitySuppressedPairs) + " near-identical pairs (|r|>0.98); lower-variance duplicates were excluded to stabilize downstream neural importance.");
    }
    addExecutiveDashboard(
        bivariate,
        "Executive Dashboard",
        {
            {"Pairs Evaluated", std::to_string(bivariatePairs.size())},
            {"Statistically Significant", std::to_string(statSigCount)},
            {"Selected Findings", std::to_string(sigRows.size())},
            {"Tier-2", std::to_string(tier2Count)},
            {"Tier-3", std::to_string(tier3Count)}
        },
        {
            "Selection balances statistical significance with neural relevance and domain fallback rules.",
            "Filtering removes redundant, structural, and leakage-risk relationships before final ranking.",
            strictFeatureReporting
                ? "Engineered-feature pair suppression is active for readability."
                : "Pair coverage includes the full core signal set."
        },
        "Quick map of relationship strength before detailed pair evidence."
    );
    bivariate.addParagraph("Causal interpretation guardrail: strong correlations indicate statistical dependence, while directed DAG edges are observational causal hypotheses that remain sensitive to hidden confounding and should not be treated as intervention-proof causation.");
    bivariate.addTable("Metric Glossary", {"Metric", "Interpretation"}, {
        {"neural_score", "Composite relevance score blending effect size, stability, modeled-feature coverage, and statistical reliability."},
        {"fold_stability", "Cross-fold consistency of pairwise correlation (1.0 is highly stable; near 0 is unstable)."},
        {"significance_tier", "Tier-2: statistical + neural confirmation; Tier-3: statistical + domain fallback promotion."},
        {"relation_label", "Human-readable relationship descriptor derived from correlation direction/strength or deterministic relation checks."},
        {"fit_line", "Whether a fitted linear overlay is rendered for the selected scatter plot (n/a when plotting is unavailable or pair not plotted)."},
        {"confidence_band", "Whether confidence interval band is rendered around the fitted line for the selected scatter plot (n/a when unavailable)."},
        {"scatter_plot", "Path to generated scatter visualization for selected pairs; '-' means no plot artifact for that row."}
    });

    {
        std::vector<std::string> adminDominatedPairs;
        bool targetAdminPredictable = false;
        const size_t targetIdxU = static_cast<size_t>(targetIdx);
        for (const auto& p : bivariatePairs) {
            if (!p.selected) continue;
            const bool adminA = isAdministrativeColumnName(p.featureA);
            const bool adminB = isAdministrativeColumnName(p.featureB);
            if (!adminA && !adminB) continue;
            adminDominatedPairs.push_back(p.featureA + " vs " + p.featureB);
            if ((p.idxA == targetIdxU && adminB) || (p.idxB == targetIdxU && adminA)) {
                if (std::abs(p.r) >= 0.995) targetAdminPredictable = true;
            }
        }
        if (!adminDominatedPairs.empty()) {
            const size_t showN = std::min<size_t>(3, adminDominatedPairs.size());
            std::string preview;
            for (size_t i = 0; i < showN; ++i) {
                if (i) preview += ", ";
                preview += adminDominatedPairs[i];
            }
            bivariate.addParagraph("Footnote: Administrative/index-like columns appear in selected associations (e.g., " + preview + "). These often reflect row ordering or data collection mechanics, not actionable domain causality.");
        }
        if (targetAdminPredictable) {
            bivariate.addParagraph("Warning: target appears near-perfectly predictable from an administrative/index-like column; treat downstream interpretations with caution.");
        }
    }

    auto trimNonInformativeColumns = [](std::vector<std::string>& headers,
                                        std::vector<std::vector<std::string>>& rows,
                                        const std::vector<std::string>& candidates) {
        auto isNonInformative = [](const std::string& v) {
            const std::string t = CommonUtils::toLower(CommonUtils::trim(v));
            return t.empty() || t == "-" || t == "no" || t == "n/a";
        };

        for (const auto& name : candidates) {
            auto it = std::find(headers.begin(), headers.end(), name);
            if (it == headers.end()) continue;
            const size_t idx = static_cast<size_t>(std::distance(headers.begin(), it));

            bool allNonInformative = true;
            for (const auto& row : rows) {
                if (idx < row.size() && !isNonInformative(row[idx])) {
                    allNonInformative = false;
                    break;
                }
            }
            if (!allNonInformative) continue;

            headers.erase(headers.begin() + static_cast<long>(idx));
            for (auto& row : rows) {
                if (idx < row.size()) row.erase(row.begin() + static_cast<long>(idx));
            }
        }
    };

    std::vector<std::string> allHeaders = {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "stat_sig", "neural_score", "significance_tier", "selection_reason", "selected", "redundant", "structural", "leakage_risk", "cluster_rep", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"};
    std::vector<std::string> sigHeaders = {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "significance_tier", "selection_reason", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"};
    const std::vector<std::string> optionalColumns = {"fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"};
    trimNonInformativeColumns(allHeaders, allRows, optionalColumns);
    trimNonInformativeColumns(sigHeaders, sigRows, optionalColumns);

    if (compactBivariateRows) {
        bivariate.addParagraph("Low-memory mode: omitted full pair table and kept only selected + capped rejected samples for diagnostics.");
    } else {
        bivariate.addTable("All Pairwise Results", allHeaders, allRows);
    }
    bivariate.addTable("Final Significant Results", sigHeaders, sigRows);
    if (!structuralRows.empty()) {
        bivariate.addTable("Structural/Deterministic Pair Diagnostics", {"Feature A", "Feature B", "pearson_r", "p_value", "structural", "deterministic", "relation_label", "selection_reason"}, structuralRows);
    }

    const auto contingencyRaw = analyzeContingencyPairs(data);
    std::vector<ContingencyInsight> contingency;
    contingency.reserve(contingencyRaw.size());
    for (const auto& c : contingencyRaw) {
        if (textMentionsContamination(c.catA) || textMentionsContamination(c.catB)) continue;
        contingency.push_back(c);
    }
    if (!contingency.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto& c : contingency) {
            rows.push_back({c.catA, c.catB, toFixed(c.chi2, 4), toFixed(c.pValue, 6), toFixed(c.cramerV, 4), toFixed(c.oddsRatio, 4), toFixed(c.oddsCiLow, 4), toFixed(c.oddsCiHigh, 4)});
        }
        bivariate.addTable("Categorical Contingency Analysis", {"Cat A", "Cat B", "chi2", "p_value", "cramers_v", "odds_ratio", "or_ci_low", "or_ci_high"}, rows);
    }

    const auto anovaRowsRaw = analyzeAnovaPairs(data);
    std::vector<AnovaInsight> anovaRows;
    anovaRows.reserve(anovaRowsRaw.size());
    for (const auto& a : anovaRowsRaw) {
        if (textMentionsContamination(a.categorical) || textMentionsContamination(a.numeric) || textMentionsContamination(a.tukeySummary)) continue;
        anovaRows.push_back(a);
    }
    if (!anovaRows.empty()) {
        std::vector<std::vector<std::string>> rows;
        for (const auto& a : anovaRows) {
            rows.push_back({a.categorical, a.numeric, toFixed(a.fStat, 4), toFixed(a.pValue, 6), toFixed(a.eta2, 4), a.tukeySummary});
        }
        bivariate.addTable("One-Way ANOVA (Categorical→Numeric)", {"Categorical", "Numeric", "F", "p_value", "eta_squared", "posthoc_tukey"}, rows);
    }

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        bivariate.addTable("Automatic Stratified Population Signals", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    AdvancedAnalyticsOutputs advancedOutputs = buildAdvancedAnalyticsOutputs(data,
                                                                             targetIdx,
                                                                             featureIdx,
                                                                             neural.featureImportance,
                                                                             bivariatePairs,
                                                                             anovaRows,
                                                                             contingency,
                                                                             statsCache);

    size_t causalSuppressedByTargetContamination = 0;
    if (!targetContaminationIndices.empty() && !advancedOutputs.causalDagRows.empty()) {
        std::unordered_set<std::string> contaminatedNames;
        contaminatedNames.reserve(targetContaminationIndices.size());
        for (size_t idx : targetContaminationIndices) {
            if (idx >= data.columns().size()) continue;
            contaminatedNames.insert(data.columns()[idx].name);
        }

        std::vector<std::vector<std::string>> kept;
        kept.reserve(advancedOutputs.causalDagRows.size());
        for (const auto& row : advancedOutputs.causalDagRows) {
            if (row.size() < 2) continue;
            if (contaminatedNames.count(row[0]) > 0 || contaminatedNames.count(row[1]) > 0) {
                ++causalSuppressedByTargetContamination;
                continue;
            }
            kept.push_back(row);
        }
        advancedOutputs.causalDagRows = std::move(kept);
        if (causalSuppressedByTargetContamination > 0) {
            advancedOutputs.causalDagMermaid.reset();
            advancedOutputs.priorityTakeaways.push_back(
                "Target-contamination guard suppressed " + std::to_string(causalSuppressedByTargetContamination) +
                " causal edges involving near-target-clone features (|corr(target)|>0.99).");
            advancedOutputs.orderedRows.push_back({"98", "Target-contamination causal suppression",
                "Suppressed " + std::to_string(causalSuppressedByTargetContamination) +
                " causal edges involving potential target contamination features."});
        }
    }

    size_t causalSuppressedByRules = 0;
    if (!domainRules.suppressCausalColumns.empty() && !advancedOutputs.causalDagRows.empty()) {
        std::vector<std::vector<std::string>> kept;
        kept.reserve(advancedOutputs.causalDagRows.size());
        for (const auto& row : advancedOutputs.causalDagRows) {
            if (row.size() < 2) continue;
            if (isCausalEdgeSuppressedByRule(row[0], row[1], domainRules)) {
                ++causalSuppressedByRules;
                continue;
            }
            kept.push_back(row);
        }
        advancedOutputs.causalDagRows = std::move(kept);
        if (causalSuppressedByRules > 0) {
            advancedOutputs.causalDagMermaid.reset();
            advancedOutputs.priorityTakeaways.push_back(
                "Domain rules suppressed " + std::to_string(causalSuppressedByRules) +
                " causal edges tagged as non-causal/constructed indices.");
            advancedOutputs.orderedRows.push_back({"99", "Domain-rule causal suppression",
                "Suppressed " + std::to_string(causalSuppressedByRules) +
                " causal edges using external domain rules."});
        }
    }

    if (!targetContaminatedNames.empty()) {
        auto filterRows = [&](std::vector<std::vector<std::string>>& rows) {
            rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const std::vector<std::string>& row) {
                return rowMentionsContamination(row);
            }), rows.end());
        };

        filterRows(advancedOutputs.orderedRows);
        filterRows(advancedOutputs.globalConditionalRows);
        filterRows(advancedOutputs.temporalDriftRows);
        filterRows(advancedOutputs.contextualDeadZoneRows);
        filterRows(advancedOutputs.mahalanobisRows);
        filterRows(advancedOutputs.pdpRows);

        advancedOutputs.narrativeRows.erase(
            std::remove_if(advancedOutputs.narrativeRows.begin(), advancedOutputs.narrativeRows.end(),
                           [&](const std::string& row) { return textMentionsContamination(row); }),
            advancedOutputs.narrativeRows.end());

        advancedOutputs.priorityTakeaways.erase(
            std::remove_if(advancedOutputs.priorityTakeaways.begin(), advancedOutputs.priorityTakeaways.end(),
                           [&](const std::string& row) { return textMentionsContamination(row); }),
            advancedOutputs.priorityTakeaways.end());

        if (advancedOutputs.executiveSummary.has_value() && textMentionsContamination(*advancedOutputs.executiveSummary)) {
            advancedOutputs.executiveSummary.reset();
        }
        if (advancedOutputs.interactionEvidence.has_value() && textMentionsContamination(*advancedOutputs.interactionEvidence)) {
            advancedOutputs.interactionEvidence.reset();
        }
    }

    if (!domainRules.loadedFrom.empty()) {
        std::string src = domainRules.loadedFrom.front();
        if (domainRules.loadedFrom.size() > 1) {
            src += " (and " + std::to_string(domainRules.loadedFrom.size() - 1) + " more)";
        }
        advancedOutputs.priorityTakeaways.push_back("Applied external domain metadata rules from: " + src + ".");
    }

    const std::optional<std::string> dagEditorRelativePath = writeCausalDagEditor(runCfg, advancedOutputs.causalDagRows);

    const DataHealthSummary dataHealth = computeDataHealthSummary(data,
                                                                  prep,
                                                                  neural,
                                                                  featureIdx.size(),
                                                                  allRows.size(),
                                                                  statSigCount,
                                                                  sigRows.size());

    ReportEngine neuralReport;
    neuralReport.addTitle("Neural Synthesis");
    neuralReport.addParagraph("This synthesis captures neural network training traces and how neural relevance influenced bivariate selection.");
    neuralReport.addParagraph(std::string("Task type inferred from target: ") + targetContext.semantics.inferredTask);
    neuralReport.addParagraph("Bivariate significance now uses a three-tier gate: Tier-1 statistical evidence, Tier-2 neural confirmation, Tier-3 domain fallback for high-effect stable relationships when neural yield is sparse.");
    addExecutiveDashboard(
        neuralReport,
        "Executive Dashboard",
        {
            {"Task Type", targetContext.semantics.inferredTask},
            {"Training Rows Used", std::to_string(neural.trainingRowsUsed) + " / " + std::to_string(neural.trainingRowsTotal)},
            {"Epochs", std::to_string(neural.epochs)},
            {"Topology", neural.topology},
            {"Health Score", toFixed(dataHealth.score, 1) + "/100 " + scoreBar100(dataHealth.score)},
            {"Predictive Confidence", toFixed(100.0 * neural.confidenceScore, 1) + "%"},
            {"OOD/Drift Sentinel", neural.driftBand}
        },
        {
            "Auto policy and deterministic guardrails reduce overfitting and unstable topology growth.",
            "Explainability and uncertainty traces are preserved for interpretability."
        },
        "Fast leadership view of model behavior before detailed training diagnostics."
    );
    neuralReport.addTable("Neural Metric Glossary", {"Metric", "Interpretation"}, {
        {"Predictive Confidence", "Composite reliability score after uncertainty, ensemble disagreement, OOD rate, and drift penalties."},
        {"OOD Rate", "Share of monitor-window rows with strongly atypical manifold distance."},
        {"Drift PSI", "Population Stability Index between reference and monitor windows; larger values imply stronger distribution shift."}
    });
    std::vector<std::vector<std::string>> decisionLogRows = {
        {"Target Selection", targetContext.userProvidedTarget ? "user-specified" : "auto"},
        {"Target Strategy", targetContext.choice.strategyUsed},
        {"Target Column", runCfg.targetColumn},
        {"Target Task", targetContext.semantics.inferredTask},
        {"Target Cardinality", std::to_string(targetContext.semantics.cardinality)},
        {"Feature Strategy", selectedFeatures.strategyUsed},
        {"Feature Missingness Threshold", toFixed(selectedFeatures.missingThresholdUsed, 4)},
        {"Features Retained", std::to_string(featureIdx.size())},
        {"Categorical Columns Encoded", std::to_string(neural.categoricalColumnsUsed)},
        {"Categorical One-Hot Nodes", std::to_string(neural.categoricalEncodedNodes)},
        {"Sparse Features Dropped", std::to_string(selectedFeatures.droppedByMissingness.size())},
        {"Pre-Flight Sparse Columns Culled", std::to_string(preflightCull.dropped)},
        {"Neural Strategy", neural.policyUsed},
        {"Ordinal Mode", runCfg.neuralOrdinalMode},
        {"Fast Mode", fastModeEnabled ? "enabled" : "disabled"},
        {"Neural Training Rows Used", std::to_string(neural.trainingRowsUsed) + " / " + std::to_string(neural.trainingRowsTotal)},
        {"Bivariate Strategy", bivariatePolicy.name},
        {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
        {"Lasso Gate", deterministic.lassoGateApplied ? ("on (kept=" + std::to_string(deterministic.lassoSelectedCount) + ")") : "off"},
        {"Tier-2 Pair Count", std::to_string(tier2Count)},
        {"Tier-3 Pair Count", std::to_string(tier3Count)}
    };
    if (!multiTargetBenchmarks.targetIndices.empty()) {
        std::string targetList;
        for (size_t i = 0; i < multiTargetBenchmarks.targetNames.size(); ++i) {
            if (i) targetList += ", ";
            targetList += multiTargetBenchmarks.targetNames[i];
        }
        decisionLogRows.push_back({"Benchmark Targets", std::to_string(multiTargetBenchmarks.targetIndices.size()) + " targets"});
        decisionLogRows.push_back({"Benchmark Target Set", targetList.empty() ? "n/a" : targetList});
        decisionLogRows.push_back({"Benchmark Aggregation", "first-class multi-target (per-target + model aggregate)"});
    }
    neuralReport.addTable("Auto Decision Log", {"Decision", "Value"}, decisionLogRows);

    if (!multiTargetBenchmarks.targetIndices.empty()) {
        std::vector<std::vector<std::string>> targetCoverageRows;
        targetCoverageRows.reserve(multiTargetBenchmarks.targetNames.size());
        for (size_t i = 0; i < multiTargetBenchmarks.targetNames.size(); ++i) {
            targetCoverageRows.push_back({
                std::to_string(i + 1),
                multiTargetBenchmarks.targetNames[i],
                std::to_string(multiTargetBenchmarks.targetIndices[i])
            });
        }
        neuralReport.addTable("Multi-Target Benchmark Coverage", {"Rank", "Target", "Column Index"}, targetCoverageRows);

        std::vector<std::vector<std::string>> aggregateRows;
        aggregateRows.reserve(multiTargetBenchmarks.aggregateByModel.size());
        for (const auto& model : multiTargetBenchmarks.aggregateByModel) {
            aggregateRows.push_back({
                model.model,
                toFixed(model.rmse, 6),
                toFixed(model.r2, 6),
                model.hasAccuracy ? toFixed(model.accuracy, 6) : "N/A"
            });
        }
        neuralReport.addTable("Multi-Target Benchmark Aggregate", {"Model", "Mean RMSE", "Mean R2", "Mean Accuracy"}, aggregateRows);

        std::vector<std::vector<std::string>> perTargetRows;
        for (size_t t = 0; t < multiTargetBenchmarks.targetNames.size() && t < multiTargetBenchmarks.perTargetResults.size(); ++t) {
            for (const auto& model : multiTargetBenchmarks.perTargetResults[t]) {
                perTargetRows.push_back({
                    multiTargetBenchmarks.targetNames[t],
                    model.model,
                    toFixed(model.rmse, 6),
                    toFixed(model.r2, 6),
                    model.hasAccuracy ? toFixed(model.accuracy, 6) : "N/A"
                });
            }
        }
        neuralReport.addTable("Multi-Target Benchmark Per-Target", {"Target", "Model", "RMSE", "R2", "Accuracy"}, perTargetRows);
    }

    if (!deterministic.roleTagRows.empty()) {
        neuralReport.addTable("Deterministic Semantic Role Tags", {"Column", "Role", "Unit Semantics", "Unique", "Missing", "Null %"}, deterministic.roleTagRows);
    }
    if (!selectedFeatures.droppedByMissingness.empty()) {
        neuralReport.addParagraph("Sparse numeric features dropped before modeling: " + std::to_string(selectedFeatures.droppedByMissingness.size()));
    }
    neuralReport.addTable("Neural Network Auto-Defaults", {"Setting", "Value"}, {
        {"Neural Policy", neural.policyUsed},
        {"Topology", neural.topology},
        {"Input Nodes", std::to_string(neural.inputNodes)},
        {"Numeric Input Nodes", std::to_string(neural.numericInputNodes)},
        {"Categorical Encoded Nodes", std::to_string(neural.categoricalEncodedNodes)},
        {"Hidden Nodes", std::to_string(neural.hiddenNodes)},
        {"Output Nodes", std::to_string(neural.outputNodes)},
        {"Auxiliary Output Targets", std::to_string(neural.outputAuxTargets)},
        {"Epochs", std::to_string(neural.epochs)},
        {"Batch Size", std::to_string(neural.batchSize)},
        {"Validation Split", toFixed(neural.valSplit, 2)},
        {"L2 Lambda", toFixed(neural.l2Lambda, 4)},
        {"Dropout", toFixed(neural.dropoutRate, 4)},
        {"Explainability", neural.explainabilityMethod},
        {"Categorical Importance Share", toFixed(neural.categoricalImportanceShare, 4)},
        {"Early Stop Patience", std::to_string(neural.earlyStoppingPatience)},
        {"Loss", neural.lossName},
        {"Strict Pruning", neural.strictPruningApplied ? ("on (dropped=" + std::to_string(neural.strictPrunedColumns) + ")") : "off"},
        {"Hidden Activation", neural.hiddenActivation},
        {"Output Activation", neural.outputActivation}
    });
    if (targetContext.semantics.inferredTask == "ordinal_classification") {
        if (neural.lossName == "mse") {
            neuralReport.addParagraph("Ordinal modeling mode: rank-regression (MSE). This is intentional for multi-level ordinal targets because it preserves order distance and avoids forcing unordered class boundaries.");
        } else {
            neuralReport.addParagraph("Ordinal modeling mode used binary cross-entropy because target cardinality is binary-compatible under current ordinal settings.");
        }
    }

    if (!neural.uncertaintyStd.empty()) {
        std::vector<std::vector<std::string>> uncertaintyRows;
        for (size_t i = 0; i < neural.uncertaintyStd.size(); ++i) {
            const double ciw = (i < neural.uncertaintyCiWidth.size()) ? neural.uncertaintyCiWidth[i] : 0.0;
            uncertaintyRows.push_back({
                "output_" + std::to_string(i + 1),
                toFixed(neural.uncertaintyStd[i], 6),
                toFixed(ciw, 6)
            });
        }
        neuralReport.addTable("Predictive Uncertainty (MC Dropout)", {"Output", "Avg StdDev", "Avg 95% CI Width"}, uncertaintyRows);
    }

    if (!neural.ensembleStd.empty()) {
        std::vector<std::vector<std::string>> ensembleRows;
        for (size_t i = 0; i < neural.ensembleStd.size(); ++i) {
            ensembleRows.push_back({
                "output_" + std::to_string(i + 1),
                toFixed(neural.ensembleStd[i], 6)
            });
        }
        neuralReport.addTable("Predictive Disagreement (Mini Ensemble)", {"Output", "Across-Model StdDev"}, ensembleRows);
    }

    neuralReport.addTable("OOD & Data Drift Sentinel", {"Metric", "Value"}, {
        {"Confidence Score", toFixed(100.0 * neural.confidenceScore, 2) + "%"},
        {"OOD Rate (monitor window)", toFixed(100.0 * neural.oodRate, 2) + "%"},
        {"OOD Mean RMS-z Distance", toFixed(neural.oodMeanDistance, 4)},
        {"OOD Max RMS-z Distance", toFixed(neural.oodMaxDistance, 4)},
        {"Reference Rows", std::to_string(neural.oodReferenceRows)},
        {"Monitor Rows", std::to_string(neural.oodMonitorRows)},
        {"Drift PSI Mean", toFixed(neural.driftPsiMean, 4)},
        {"Drift PSI Max", toFixed(neural.driftPsiMax, 4)},
        {"Sentinel Band", neural.driftBand}
    });
    if (neural.driftWarning) {
        neuralReport.addParagraph("⚠️ Data Drift Warning: monitor-window distribution diverges from the model reference window. Predictions may be less reliable until data quality or upstream process stability is reviewed.");
        neuralReport.addParagraph("Recommended actions: retrain on recent data windows, validate upstream ingestion/transformation pipelines, and compare feature-level PSI contributors before deployment decisions.");
    }

    std::vector<std::vector<std::string>> fiRows;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        fiRows.push_back({name, toFixed(imp, 6)});
    }
    neuralReport.addTable("Feature Importance (Neural Explainability)", {"Feature", "Importance"}, fiRows);

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        neuralReport.addTable("Automatic Segmented Population Signals", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    std::vector<std::vector<std::string>> lossRows;
    for (size_t e = 0; e < neural.trainLoss.size(); ++e) {
        double val = (e < neural.valLoss.size() ? neural.valLoss[e] : 0.0);
        lossRows.push_back({
            std::to_string(e + 1),
            toFixed(neural.trainLoss[e], 6),
            toFixed(val, 6)
        });
    }
    neuralReport.addTable("Neural Network Training Trace", {"Epoch", "Train Loss", "Validation Loss"}, lossRows);

    if (!neural.gradientNorm.empty() || !neural.weightStd.empty() || !neural.weightMeanAbs.empty()) {
        std::vector<std::vector<std::string>> dynRows;
        const size_t n = std::max({neural.gradientNorm.size(), neural.weightStd.size(), neural.weightMeanAbs.size()});
        for (size_t i = 0; i < n; ++i) {
            const double g = (i < neural.gradientNorm.size()) ? neural.gradientNorm[i] : 0.0;
            const double wstd = (i < neural.weightStd.size()) ? neural.weightStd[i] : 0.0;
            const double wabs = (i < neural.weightMeanAbs.size()) ? neural.weightMeanAbs[i] : 0.0;
            dynRows.push_back({std::to_string(i + 1), toFixed(g, 6), toFixed(wstd, 6), toFixed(wabs, 6)});
        }
        neuralReport.addTable("Neural Training Dynamics", {"Step", "Gradient Norm", "Weight RMS", "Weight Mean Abs"}, dynRows);
    }

    addOverallSections(neuralReport,
                       data,
                       prep,
                       benchmarks,
                       neural,
                       dataHealth,
                       runCfg,
                       &plotterOverall,
                       canPlot && runCfg.plotOverall,
                       runCfg.verboseAnalysis,
                       statsCache);
    advance("Built overall sections");

    ReportEngine finalAnalysis;
    finalAnalysis.addTitle("Final Analysis - Significant Findings Only");
    finalAnalysis.addParagraph("This report contains statistically significant findings selected by a tiered engine: Tier-2 (statistical + neural) and Tier-3 fallback (statistical + domain effect). Non-selected findings are excluded by design.");
    finalAnalysis.addParagraph("Data Health Score: " + toFixed(dataHealth.score, 1) + "/100 (" + dataHealth.band + "). This score estimates discovered signal strength using completeness, retained feature coverage, significant-pair yield, selected-pair yield, and neural training stability.");

    size_t adminSelectedCount = 0;
    bool targetAdminPredictableExec = false;
    const size_t targetIdxUExec = static_cast<size_t>(targetIdx);
    for (const auto& p : bivariatePairs) {
        if (!p.selected) continue;
        const bool adminA = isAdministrativeColumnName(p.featureA);
        const bool adminB = isAdministrativeColumnName(p.featureB);
        if (!adminA && !adminB) continue;
        ++adminSelectedCount;
        if ((p.idxA == targetIdxUExec && adminB) || (p.idxB == targetIdxUExec && adminA)) {
            if (std::abs(p.r) >= 0.995) targetAdminPredictableExec = true;
        }
    }

    std::vector<std::string> execHighlights = {
        "This top block is optimized for non-technical consumption.",
        "Ordered methods, constraint-based causal discovery, and detailed evidence tables follow below."
    };
    if (adminSelectedCount > 0) {
        execHighlights.push_back("Administrative/index-like columns appear in selected findings; interpret those as potential ordering artifacts.");
    }
    if (!domainRules.loadedFrom.empty()) {
        execHighlights.push_back("External domain-rule metadata was applied to suppress non-causal/constructed-edge candidates.");
    }
    if (!targetContaminationRows.empty()) {
        execHighlights.push_back("Potential target-contamination features (|corr(target)|>0.99) were auto-flagged and excluded from causal DAG candidates.");
    }

    addExecutiveDashboard(
        finalAnalysis,
        "Executive Dashboard",
        {
            {"Health Score", toFixed(dataHealth.score, 1) + "/100 " + scoreBar100(dataHealth.score)},
            {"Health Band", dataHealth.band},
            {"Selected Findings", std::to_string(sigRows.size())},
            {"Rows", std::to_string(data.rowCount())},
            {"Columns", std::to_string(data.colCount())}
        },
        execHighlights,
        "Snapshot first; evidence next."
    );
    if (targetAdminPredictableExec) {
        finalAnalysis.addParagraph("Executive caution: target is near-perfectly predictable from an administrative/index-like column, which often reflects dataset ordering rather than a meaningful causal relationship.");
    }

    if (!advancedOutputs.orderedRows.empty()) {
        finalAnalysis.addTable("Ordered Methods", {"Step", "Method", "Result"}, advancedOutputs.orderedRows);
    }

    if (!targetContaminationRows.empty()) {
        finalAnalysis.addTable("Potential Target Contamination", {"Feature", "abs_corr_with_target", "Status"}, targetContaminationRows);
        finalAnalysis.addParagraph("Leak guard: near-target-clone features are treated as potential contamination, excluded from neural predictor inputs, and blocked from causal DAG edge candidates.");
    }

    if (!advancedOutputs.causalDagRows.empty()) {
        finalAnalysis.addParagraph("Causal interpretation note: each DAG edge is a model-derived directional hypothesis from observational dependence tests. Use these as candidates for follow-up validation, not as guaranteed intervention effects.");
        finalAnalysis.addTable("Causal Discovery Graph Candidates",
                               {"From", "To", "Confidence", "Evidence", "Interpretation"},
                               advancedOutputs.causalDagRows);
        if (dagEditorRelativePath.has_value()) {
            finalAnalysis.addParagraph("Interactive DAG editor: [Open DAG Editor](" + *dagEditorRelativePath + ")");
        }
    }

    finalAnalysis.addTable("Bivariate Metric Glossary", {"Metric", "Interpretation"}, {
        {"neural_score", "Composite relevance score blending effect size, stability, modeled-feature coverage, and statistical reliability."},
        {"fold_stability", "Cross-fold consistency of pairwise correlation (1.0 is highly stable; near 0 is unstable)."},
        {"significance_tier", "Tier-2: statistical + neural confirmation; Tier-3: statistical + domain fallback promotion."},
        {"relation_label", "Human-readable relationship descriptor derived from correlation direction/strength or deterministic relation checks."},
        {"fit_line", "Whether a fitted linear overlay is rendered for the selected scatter plot (n/a when plotting is unavailable or pair not plotted)."},
        {"confidence_band", "Whether confidence interval band is rendered around the fitted line for selected scatter plots (n/a when unavailable)."}
    });

    {
        std::vector<std::string> adminDominatedPairs;
        bool targetAdminPredictable = false;
        const size_t targetIdxU = static_cast<size_t>(targetIdx);
        for (const auto& p : bivariatePairs) {
            if (!p.selected) continue;
            const bool adminA = isAdministrativeColumnName(p.featureA);
            const bool adminB = isAdministrativeColumnName(p.featureB);
            if (!adminA && !adminB) continue;
            adminDominatedPairs.push_back(p.featureA + " vs " + p.featureB);
            if ((p.idxA == targetIdxU && adminB) || (p.idxB == targetIdxU && adminA)) {
                if (std::abs(p.r) >= 0.995) targetAdminPredictable = true;
            }
        }

        if (!adminDominatedPairs.empty()) {
            const size_t showN = std::min<size_t>(3, adminDominatedPairs.size());
            std::string preview;
            for (size_t i = 0; i < showN; ++i) {
                if (i) preview += ", ";
                preview += adminDominatedPairs[i];
            }
            finalAnalysis.addParagraph("Context note: selected findings include administrative/index-like columns (e.g., " + preview + "), which commonly capture dataset ordering rather than meaningful domain mechanism.");
        }
        if (targetAdminPredictable) {
            finalAnalysis.addParagraph("One-line caution: target is near-perfectly predictable from an administrative/index-like column; interpret significance as ordering artifact risk before acting on conclusions.");
        }
    }

    if (!advancedOutputs.globalConditionalRows.empty()) {
        finalAnalysis.addTable("Global vs Conditional Relationship Drift",
                               {"Feature", "Global r", "Conditional r", "|conditional|/|global|", "Pattern", "Interpretation"},
                               advancedOutputs.globalConditionalRows);
        finalAnalysis.addParagraph("Recommended action: review confounders for flagged rows, prioritize controlled feature checks, and re-evaluate decisions using conditional rather than global associations.");
    }

    if (!advancedOutputs.temporalDriftRows.empty()) {
        finalAnalysis.addTable("Temporal Drift Kernels (ADF-Style)",
                               {"Feature", "Axis", "Samples", "Gamma", "t-stat", "p-approx", "Drift Ratio", "Verdict"},
                               advancedOutputs.temporalDriftRows);
        finalAnalysis.addParagraph("Recommended action: retrain on recent windows, verify upstream data pipeline changes, and consider differencing/time-aware transforms for non-stationary signals.");
    }

    if (!advancedOutputs.contextualDeadZoneRows.empty()) {
        finalAnalysis.addTable("Cross-Cluster Contextual Dead-Zones",
                               {"Feature", "Predictive Cluster", "Dead-Zone Cluster", "|r| strong", "|r| weak", "weak/strong", "Min Cluster N"},
                               advancedOutputs.contextualDeadZoneRows);
    }

    if (advancedOutputs.causalDagMermaid.has_value()) {
        finalAnalysis.addParagraph("Causal discovery visual sketch (PC/Meek/LiNGAM + bootstrap support):\n" + *advancedOutputs.causalDagMermaid);
        finalAnalysis.addParagraph("Guardrail: graph edges are discovery hypotheses under CI assumptions and should be validated against interventions or external design constraints.");
    }

    if (!advancedOutputs.mahalanobisRows.empty()) {
        finalAnalysis.addTable("Mahalanobis Multivariate Outliers", {"Row", "Distance^2", "Threshold"}, advancedOutputs.mahalanobisRows);
    }

    if (!advancedOutputs.pdpRows.empty()) {
        finalAnalysis.addTable("Linear Partial Dependence Approximation", {"Feature", "Low (P10)", "Medium (P50)", "High (P90)", "Delta (High-Low)", "Direction"}, advancedOutputs.pdpRows);
    }

    if (!advancedOutputs.narrativeRows.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(advancedOutputs.narrativeRows.size());
        for (const auto& s : advancedOutputs.narrativeRows) rows.push_back({s});
        finalAnalysis.addTable("Cross-Section Narrative Layer", {"Narrative"}, rows);
    }

    if (advancedOutputs.executiveSummary.has_value()) {
        finalAnalysis.addParagraph("Advanced Analytics Executive Summary: " + *advancedOutputs.executiveSummary);
    }

    if (advancedOutputs.interactionEvidence.has_value()) {
        neuralInteractionTakeaway = *advancedOutputs.interactionEvidence;
    }

    std::vector<std::string> prioritizedTakeaways;
    std::unordered_set<std::string> seenTakeaways;
    for (const auto& s : advancedOutputs.priorityTakeaways) {
        if (!s.empty() && seenTakeaways.insert(s).second) prioritizedTakeaways.push_back(s);
    }
    for (const auto& s : topTakeaways) {
        if (!s.empty() && seenTakeaways.insert(s).second) prioritizedTakeaways.push_back(s);
    }
    if (prioritizedTakeaways.size() > 3) prioritizedTakeaways.resize(3);

    if (!prioritizedTakeaways.empty()) {
        std::vector<std::vector<std::string>> takeawayRows;
        for (size_t i = 0; i < prioritizedTakeaways.size(); ++i) {
            takeawayRows.push_back({std::to_string(i + 1), prioritizedTakeaways[i]});
        }
        finalAnalysis.addTable("Top 3 Takeaways", {"Rank", "Takeaway"}, takeawayRows);
    }

    if (!neuralInteractionTakeaway.empty()) {
        finalAnalysis.addTable("Neural Interaction Takeaway", {"Insight"}, {{neuralInteractionTakeaway}});
    }

    if (auto residualNarrative = buildResidualDiscoveryNarrative(data, targetIdx, featureIdx, statsCache); residualNarrative.has_value()) {
        finalAnalysis.addTable("Residual-Based Sequential Discovery", {"Insight"}, {{*residualNarrative}});
    } else if (!deterministic.residualNarrative.empty()) {
        finalAnalysis.addTable("Residual-Based Sequential Discovery", {"Insight"}, {{deterministic.residualNarrative}});
    }

    if (!narrativeInsights.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(narrativeInsights.size());
        for (size_t i = 0; i < narrativeInsights.size(); ++i) {
            if (textMentionsContamination(narrativeInsights[i])) continue;
            rows.push_back({std::to_string(rows.size() + 1), narrativeInsights[i]});
        }
        if (!rows.empty()) {
            finalAnalysis.addTable("Narrative Insight Layer", {"#", "Insight"}, rows);
        }
    }

    auto outlierContextRows = buildOutlierContextRows(reportingData,
                                                      prep,
                                                      advancedOutputs.mahalanobisByRow,
                                                      advancedOutputs.mahalanobisThreshold);
    outlierContextRows.erase(std::remove_if(outlierContextRows.begin(), outlierContextRows.end(), [&](const std::vector<std::string>& row) {
        return rowMentionsContamination(row);
    }), outlierContextRows.end());
    if (!outlierContextRows.empty()) {
        finalAnalysis.addTable("Outlier Contextualization", {"Row", "Primary Feature", "Value", "|z|", "Why it is unusual"}, outlierContextRows);
    }

    if (!redundancyDrops.empty()) {
        std::vector<std::vector<std::string>> dropRows;
        for (const auto& msg : redundancyDrops) {
            if (textMentionsContamination(msg)) continue;
            dropRows.push_back({msg});
        }
        if (!dropRows.empty()) {
            finalAnalysis.addTable("Redundancy Drop Recommendations", {"Action"}, dropRows);
        }
    }

    if (!stratifiedPopulations.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(stratifiedPopulations.size());
        for (const auto& s : stratifiedPopulations) {
            rows.push_back({
                s.segmentColumn,
                s.numericColumn,
                std::to_string(s.groups),
                std::to_string(s.rows),
                toFixed(s.eta2, 4),
                toFixed(s.separation, 4),
                s.groupMeans
            });
        }
        finalAnalysis.addTable("Segmented Population Findings", {"Segment Column", "Numeric Feature", "Groups", "Rows", "eta_squared", "separation", "group_means"}, rows);
    }

    finalAnalysis.addTable("Selected Significant Bivariate Findings", {"Feature A", "Feature B", "pearson_r", "spearman_rho", "kendall_tau", "r2", "effect_size", "fold_stability", "slope", "intercept", "t_stat", "p_value", "neural_score", "significance_tier", "selection_reason", "relation_label", "fit_line", "confidence_band", "stacked_plot", "residual_plot", "faceted_plot", "scatter_plot"}, sigRows);

    finalAnalysis.addTable("Data Health Signal Card", {"Component", "Value"}, {
        {"Score (0-100)", toFixed(dataHealth.score, 1)},
        {"Band", dataHealth.band},
        {"Completeness", toFixed(100.0 * dataHealth.completeness, 1) + "%"},
        {"Numeric Coverage", toFixed(100.0 * dataHealth.numericCoverage, 1) + "%"},
        {"Feature Retention", toFixed(100.0 * dataHealth.featureRetention, 1) + "%"},
        {"Statistical Yield", toFixed(100.0 * dataHealth.statYield, 1) + "%"},
        {"Selected Yield", toFixed(100.0 * dataHealth.selectedYield, 1) + "%"},
        {"Training Stability", toFixed(100.0 * dataHealth.trainingStability, 1) + "%"},
        {"Drift PSI Mean", toFixed(dataHealth.driftPsiMean, 4)},
        {"Drift Penalty", toFixed(100.0 * dataHealth.driftPenalty, 1) + "%"}
    });

    {
        std::vector<std::vector<std::string>> actionableHealthRows;
        const double driftPsi = std::max(dataHealth.driftPsiMean, neural.driftPsiMax);
        if (driftPsi > 2.0) {
            std::string driftFeature = "high-PSI contributors";
            if (!advancedOutputs.temporalDriftRows.empty() && !advancedOutputs.temporalDriftRows.front().empty()) {
                driftFeature = advancedOutputs.temporalDriftRows.front()[0];
            }
            actionableHealthRows.push_back({
                "Drift_PSI > 2.0",
                "Action: Check for temporal shifts in " + driftFeature + "."
            });
        }
        if (deterministic.rowsToFeatures < 5.0) {
            actionableHealthRows.push_back({
                "Rows:Features < 5",
                "Action: Reduce feature expansion; dataset is too small for interactions."
            });
        }
        if (!actionableHealthRows.empty()) {
            finalAnalysis.addTable("Actionable Health Actions", {"Trigger", "Action"}, actionableHealthRows);
        }
    }

    finalAnalysis.addTable("Probabilistic Reliability Card", {"Metric", "Value"}, {
        {"Predictive Confidence", toFixed(100.0 * neural.confidenceScore, 2) + "%"},
        {"OOD Rate", toFixed(100.0 * neural.oodRate, 2) + "%"},
        {"Drift Band", neural.driftBand},
        {"Drift PSI Mean", toFixed(neural.driftPsiMean, 4)},
        {"Drift PSI Max", toFixed(neural.driftPsiMax, 4)}
    });
    if (neural.driftWarning) {
        finalAnalysis.addParagraph("⚠️ Reliability Alert: Seldon detected out-of-distribution behavior and/or population drift. Treat downstream forecasts as conditional and prioritize data validation before actioning high-impact decisions.");
    }

    std::vector<std::vector<std::string>> topFeatures;
    std::vector<std::pair<std::string, double>> fiPairs;
    std::unordered_map<std::string, size_t> deterministicFeatureHits;
    for (const auto& pair : bivariatePairs) {
        const bool deterministicRelation = (std::abs(pair.r) >= 0.999) || pair.filteredAsStructural;
        if (!deterministicRelation) continue;
        deterministicFeatureHits[pair.featureA]++;
        deterministicFeatureHits[pair.featureB]++;
    }
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        auto hitIt = deterministicFeatureHits.find(name);
        if (hitIt != deterministicFeatureHits.end() && hitIt->second > 0) {
            const double penalty = std::max(0.35, std::pow(0.82, static_cast<double>(hitIt->second)));
            imp *= penalty;
        }
        fiPairs.push_back({name, imp});
    }
    std::sort(fiPairs.begin(), fiPairs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& kv : fiPairs) {
        topFeatures.push_back({kv.first, toFixed(kv.second, 6)});
    }
    finalAnalysis.addTable("Neural Feature Importance Ranking", {"Feature", "Importance"}, topFeatures);

    finalAnalysis.addTable("Executive Statistics", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Columns", std::to_string(data.colCount())},
        {"Pairs Evaluated", std::to_string(bivariatePairs.size())},
        {"Pairs Statistically Significant", std::to_string(statSigCount)},
        {"Pairs Selected", std::to_string(sigRows.size())},
        {"Training Epochs Executed", std::to_string(neural.trainLoss.size())},
        {"Data Health Score", toFixed(dataHealth.score, 1) + " (" + dataHealth.band + ")"}
    });

    ReportEngine heuristicsReport;
    heuristicsReport.addTitle("Deterministic Analysis Report");
    heuristicsReport.addParagraph("This report presents deterministic analysis results and signal-quality controls, generated independently from final_analysis.md.");

    size_t adminCount = 0;
    size_t targetCandidateCount = 0;
    size_t lowSignalCount = 0;
    for (const auto& row : deterministic.roleTagRows) {
        if (row.size() < 2) continue;
        if (row[1] == "ADMIN") adminCount++;
        else if (row[1] == "TARGET_CANDIDATE") targetCandidateCount++;
        else if (row[1] == "LOW_SIGNAL") lowSignalCount++;
    }

    const bool stabilityGuardTriggered = (neural.policyUsed.find("stability_shrink") != std::string::npos);
    const std::string stabilityPct = toFixed(100.0 * dataHealth.trainingStability, 1) + "%";

    std::string phase1Narrative = "Semantic filter excluded " + std::to_string(adminCount)
        + " administrative columns and marked " + std::to_string(lowSignalCount)
        + " low-signal columns before model training.";

    std::string phase3Narrative = "Rows:Features=" + toFixed(deterministic.rowsToFeatures, 2)
        + "; strict pruning=" + std::string(neural.strictPruningApplied ? "on" : "off")
        + "; lasso gate=" + std::string(deterministic.lassoGateApplied ? ("on (kept=" + std::to_string(deterministic.lassoSelectedCount) + ")") : "off") + ".";

    std::string phase4Narrative = deterministic.residualNarrative.empty()
        ? "Residual discovery scanned secondary drivers but no hidden-driver threshold was met."
        : deterministic.residualNarrative;

    std::string phase5Narrative = "Deterministic narrative synthesis translated statistical findings into concise analytical statements.";

    addExecutiveDashboard(
        heuristicsReport,
        "Executive Dashboard",
        {
            {"Semantic Admin Columns", std::to_string(adminCount)},
            {"Target Candidates", std::to_string(targetCandidateCount)},
            {"Low-Signal Columns", std::to_string(lowSignalCount)},
            {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
            {"Training Stability", stabilityPct}
        },
        {
            "Deterministic phases provide interpretable quality controls independent of model complexity.",
            stabilityGuardTriggered
                ? "Stability guard triggered and topology was shrunk for safer convergence."
                : "Stability guard did not trigger; topology remained in stable range."
        },
        "Control-layer summary before stage-level diagnostics."
    );

    heuristicsReport.addTable("Analysis Workflow Summary", {"Stage", "Objective", "Method", "Outcome"}, {
        {"Semantic Filtering", "Reduce metadata and administrative noise", "Regex role tagging (ADMIN/METADATA/TARGET_CANDIDATE)", phase1Narrative},
        {"Feature Guardrails", "Limit overfitting from engineered expansion", "Dynamic sparsity + lasso gate + engineered-feature controls", phase3Narrative},
        {"Narrative Synthesis", "Convert model statistics into readable findings", "Deterministic narrative layer", phase5Narrative},
        {"Residual Discovery", "Detect hidden secondary drivers", "Primary-fit residual pass across remaining features", phase4Narrative},
        {"Outlier Context", "Explain anomaly relevance", "Contrastive z-score contextualization", "Outlier narratives now explain why flagged rows are unusual relative to the population."}
    });

    heuristicsReport.addTable("Neural Network Tuning", {"Control", "Observed", "Action"}, {
        {"Training Stability", stabilityPct, stabilityGuardTriggered ? "Hidden nodes reduced automatically (<70% stability guard triggered)" : "Topology retained (stability guard not triggered)"},
        {"Target Diversity", targetContext.semantics.inferredTask, "Classification-like targets use cross-entropy path; continuous targets use MSE path"},
        {"Loss Function", neural.lossName, "Applied automatically from inferred target semantics"}
    });

    heuristicsReport.addTable("Topology Guardrail", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Features (post deterministic filter)", std::to_string(featureIdx.size())},
        {"Rows:Features", toFixed(deterministic.rowsToFeatures, 2)},
        {"Low Ratio Mode (<10)", deterministic.lowRatioMode ? "yes" : "no"},
        {"High Ratio Mode (>100)", deterministic.highRatioMode ? "yes" : "no"},
        {"Lasso Gate Applied", deterministic.lassoGateApplied ? "yes" : "no"},
        {"Lasso Retained Count", std::to_string(deterministic.lassoSelectedCount)}
    });

    if (!deterministic.roleTagRows.empty()) {
        heuristicsReport.addTable("Semantic Role Tags", {"Column", "Role", "Unit Semantics", "Unique", "Missing", "Null %"}, deterministic.roleTagRows);
    }

    if (!deterministic.excludedReasonLines.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(deterministic.excludedReasonLines.size());
        for (const auto& line : deterministic.excludedReasonLines) rows.push_back({line});
        heuristicsReport.addTable("Entropy Filter Decisions", {"Decision"}, rows);
    }

    if (!deterministic.badgeNarratives.empty()) {
        std::vector<std::vector<std::string>> rows;
        rows.reserve(deterministic.badgeNarratives.size());
        for (const auto& line : deterministic.badgeNarratives) {
            rows.push_back({line});
        }
        heuristicsReport.addTable("Deterministic Narratives", {"Narrative"}, rows);
    }

    if (!deterministic.residualNarrative.empty()) {
        heuristicsReport.addTable("Residual Discovery", {"Insight"}, {{deterministic.residualNarrative}});
    }

    saveGeneratedReports(runCfg, univariate, bivariate, neuralReport, finalAnalysis, heuristicsReport);
    cleanupPlotCacheArtifacts(runCfg);
    advance("Saved reports");
    progress.done("Pipeline complete");
    printPipelineCompletion(runCfg);

    return 0;
}
