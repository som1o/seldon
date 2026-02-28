NeuralAnalysis runNeuralAnalysis(const TypedDataset& data,
                                 int targetIdx,
                                 const std::vector<int>& featureIdx,
                                 bool classificationTarget,
                                 bool ordinalTarget,
                                 size_t targetCardinality,
                                 bool verbose,
                                 const AutoConfig& config,
                                 bool fastModeEnabled,
                                 size_t fastSampleRows) {
    NeuralAnalysis analysis;
    analysis.numericInputNodes = featureIdx.size();
    analysis.outputNodes = 1;
    analysis.binaryTarget = classificationTarget;
    analysis.classificationTarget = classificationTarget;
    analysis.hiddenActivation = "n/a";
    const bool ordinalNonBinary = ordinalTarget && targetCardinality > 2;
    const bool ordinalBceMode = CommonUtils::toLower(config.neuralOrdinalMode) == "binary_cross_entropy_when_possible";
    const bool useCrossEntropy = classificationTarget && !ordinalNonBinary &&
                                 (ordinalBceMode || !ordinalTarget || targetCardinality <= 2);
    analysis.outputActivation = activationToString(useCrossEntropy ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR);
    analysis.lossName = useCrossEntropy ? "cross_entropy" : "mse";
    analysis.trainingRowsTotal = data.rowCount();

    if (featureIdx.empty()) {
        analysis.policyUsed = "none_no_features";
        return analysis;
    }
    if (data.rowCount() < 2) {
        analysis.policyUsed = "none_too_few_rows";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }

    auto fitStandardScaler = [](const std::vector<std::vector<double>>& matrix,
                                std::vector<double>& means,
                                std::vector<double>& stds) {
        const size_t rows = matrix.size();
        const size_t cols = rows == 0 ? 0 : matrix.front().size();
        means.assign(cols, 0.0);
        stds.assign(cols, 1.0);
        if (rows == 0 || cols == 0) return;

        for (size_t c = 0; c < cols; ++c) {
            double sum = 0.0;
            double sq = 0.0;
            size_t seen = 0;
            for (size_t r = 0; r < rows; ++r) {
                if (c >= matrix[r].size()) continue;
                const double v = matrix[r][c];
                if (!std::isfinite(v)) continue;
                sum += v;
                sq += v * v;
                ++seen;
            }
            if (seen == 0) continue;
            const double mean = sum / static_cast<double>(seen);
            const double var = std::max(0.0, sq / static_cast<double>(seen) - mean * mean);
            means[c] = mean;
            stds[c] = (var > 1e-12) ? std::sqrt(var) : 1.0;
        }
    };

    auto applyStandardScaler = [](std::vector<std::vector<double>>& matrix,
                                  const std::vector<double>& means,
                                  const std::vector<double>& stds) {
        if (matrix.empty()) return;
        const size_t cols = std::min(means.size(), stds.size());
        for (auto& row : matrix) {
            const size_t limit = std::min(row.size(), cols);
            for (size_t c = 0; c < limit; ++c) {
                const double scale = (std::isfinite(stds[c]) && std::abs(stds[c]) > 1e-12) ? stds[c] : 1.0;
                row[c] = (row[c] - means[c]) / scale;
            }
        }
    };

    const auto& y = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    std::vector<int> targetIndices;
    targetIndices.push_back(targetIdx);
    if (config.neuralMultiOutput && config.neuralMaxAuxTargets > 0) {
        const auto numericIdx = data.numericColumnIndices();
        std::vector<std::pair<int, double>> candidates;
        candidates.reserve(numericIdx.size());
        const NumericStatsCache statsCache = buildNumericStatsCache(data);
        const auto yIt = statsCache.find(static_cast<size_t>(targetIdx));
        const ColumnStats yStats = (yIt != statsCache.end()) ? yIt->second : Statistics::calculateStats(y);
        for (size_t idx : numericIdx) {
            if (static_cast<int>(idx) == targetIdx) continue;
            const auto& cand = std::get<std::vector<double>>(data.columns()[idx].values);
            const auto cIt = statsCache.find(idx);
            const ColumnStats cStats = (cIt != statsCache.end()) ? cIt->second : Statistics::calculateStats(cand);
            const double corr = std::abs(MathUtils::calculatePearson(cand, y, cStats, yStats).value_or(0.0));
            if (std::isfinite(corr)) {
                candidates.push_back({static_cast<int>(idx), corr});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });
        const size_t keep = std::min(config.neuralMaxAuxTargets, candidates.size());
        for (size_t i = 0; i < keep; ++i) {
            targetIndices.push_back(candidates[i].first);
        }
    }

    if (config.neuralStreamingMode) {
        const NeuralEncodingPlan plan = buildNeuralEncodingPlan(data, targetIdx, featureIdx, config);
        const size_t inputNodes = plan.encodedWidth;
        const size_t outputNodes = targetIndices.size();

        analysis.outputAuxTargets = (targetIndices.size() > 1) ? (targetIndices.size() - 1) : 0;
        analysis.inputNodes = inputNodes;
        analysis.categoricalColumnsUsed = plan.categoricalColumnsUsed;
        analysis.categoricalEncodedNodes = plan.categoricalEncodedNodes;

        if (inputNodes == 0 || outputNodes == 0) {
            analysis.policyUsed = "none_streaming_no_encoded_features";
            analysis.featureImportance.assign(featureIdx.size(), 0.0);
            analysis.trainingRowsUsed = data.rowCount();
            return analysis;
        }

        size_t hidden = std::clamp<size_t>(
            static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes) * std::max<size_t>(8, data.rowCount() / 6))))),
            4,
            static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));

        if (static_cast<double>(data.rowCount()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
            hidden = std::min(hidden, std::clamp<size_t>(data.rowCount() / 3, 4, 20));
        }
        if (config.neuralFixedHiddenNodes > 0) {
            hidden = static_cast<size_t>(config.neuralFixedHiddenNodes);
        }
        if (data.rowCount() < 500) {
            hidden = std::min(hidden, std::max<size_t>(1, inputNodes));
        }

        std::vector<size_t> topology = {inputNodes, hidden, outputNodes};
        NeuralNet nn(topology);
        NeuralNet::Hyperparameters hp;

        auto toOptimizer = [](const std::string& name) {
            const std::string n = CommonUtils::toLower(name);
            if (n == "sgd") return NeuralNet::Optimizer::SGD;
            if (n == "adam") return NeuralNet::Optimizer::ADAM;
            return NeuralNet::Optimizer::LOOKAHEAD;
        };

        hp.epochs = std::clamp<size_t>(120 + data.rowCount() / 2, 100, 320);
        hp.batchSize = std::clamp<size_t>(data.rowCount() / 24, 4, 64);
        hp.learningRate = config.neuralLearningRate;
        hp.earlyStoppingPatience = std::clamp<int>(static_cast<int>(hp.epochs / 12), 6, 24);
        hp.lrDecay = config.neuralLrDecay;
        hp.lrPlateauPatience = config.neuralLrPlateauPatience;
        hp.lrCooldownEpochs = config.neuralLrCooldownEpochs;
        hp.maxLrReductions = config.neuralMaxLrReductions;
        hp.minLearningRate = config.neuralMinLearningRate;
        hp.lrWarmupEpochs = static_cast<size_t>(std::max(0, config.neuralLrWarmupEpochs));
        hp.useCosineAnnealing = config.neuralUseCosineAnnealing;
        hp.useCyclicalLr = config.neuralUseCyclicalLr;
        hp.lrCycleEpochs = static_cast<size_t>(std::max(2, config.neuralLrCycleEpochs));
        hp.lrScheduleMinFactor = config.neuralLrScheduleMinFactor;
        hp.useValidationLossEma = config.neuralUseValidationLossEma;
        hp.validationLossEmaBeta = config.neuralValidationLossEmaBeta;
        hp.dropoutRate = (inputNodes >= 12) ? 0.12 : (inputNodes >= 6 ? 0.08 : 0.04);
        hp.valSplit = std::clamp((data.rowCount() < 80) ? 0.30 : 0.20, 0.10, 0.40);
        hp.l2Lambda = (data.rowCount() < 80) ? 0.010 : 0.001;
        hp.categoricalInputL2Boost = config.neuralCategoricalInputL2Boost;
        hp.activation = (inputNodes < 10) ? NeuralNet::Activation::TANH : NeuralNet::Activation::GELU;
        hp.outputActivation = useCrossEntropy ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
        hp.useBatchNorm = config.neuralUseBatchNorm;
        hp.batchNormMomentum = config.neuralBatchNormMomentum;
        hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
        hp.useLayerNorm = config.neuralUseLayerNorm;
        hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
        hp.optimizer = toOptimizer(config.neuralOptimizer);
        hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
        hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
        hp.lookaheadAlpha = config.neuralLookaheadAlpha;
        hp.loss = useCrossEntropy ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
        hp.importanceMaxRows = config.neuralImportanceMaxRows;
        hp.importanceParallel = config.neuralImportanceParallel;
        hp.verbose = verbose;
        hp.seed = config.neuralSeed;
        hp.gradientClipNorm = config.gradientClipNorm;
        hp.adaptiveGradientClipping = config.neuralUseAdaptiveGradientClipping;
        hp.adaptiveClipBeta = config.neuralAdaptiveClipBeta;
        hp.adaptiveClipMultiplier = config.neuralAdaptiveClipMultiplier;
        hp.adaptiveClipMin = config.neuralAdaptiveClipMin;
        hp.gradientNoiseStd = config.neuralGradientNoiseStd;
        hp.gradientNoiseDecay = config.neuralGradientNoiseDecay;
        hp.useEmaWeights = config.neuralUseEmaWeights;
        hp.emaDecay = config.neuralEmaDecay;
        hp.labelSmoothing = config.neuralLabelSmoothing;
        hp.gradientAccumulationSteps = static_cast<size_t>(std::max(1, config.neuralGradientAccumulationSteps));
        hp.incrementalMode = true;

        std::vector<double> inputL2Scales(inputNodes, 1.0);
        for (size_t i = 0; i < inputL2Scales.size() && i < plan.sourceNumericFeaturePos.size(); ++i) {
            if (plan.sourceNumericFeaturePos[i] < 0) inputL2Scales[i] = hp.categoricalInputL2Boost;
        }
        nn.setInputL2Scales(inputL2Scales);

        const size_t chunkRows = std::max<size_t>(16, config.neuralStreamingChunkRows);
        std::vector<double> streamingMeans;
        std::vector<double> streamingStds;
        {
            std::vector<std::vector<double>> scaleSample;
            std::vector<std::vector<double>> scaleRows;
            scaleSample.reserve(std::min<size_t>(data.rowCount(), static_cast<size_t>(2048)));
            for (size_t start = 0; start < data.rowCount(); start += chunkRows) {
                const size_t end = std::min(start + chunkRows, data.rowCount());
                encodeNeuralRows(data, plan, start, end, scaleRows);
                for (size_t r = 0; r < scaleRows.size() && scaleSample.size() < 2048; ++r) {
                    scaleSample.push_back(scaleRows[r]);
                }
                if (scaleSample.size() >= 2048) break;
            }
            fitStandardScaler(scaleSample, streamingMeans, streamingStds);
        }

        char xTmp[] = "/tmp/seldon_nn_x_XXXXXX";
        char yTmp[] = "/tmp/seldon_nn_y_XXXXXX";
        struct ScopedUnlink {
            const char* path = nullptr;
            bool active = false;
            ~ScopedUnlink() {
                if (active && path) ::unlink(path);
            }
            void release() { active = false; }
        } xTmpGuard{xTmp, false}, yTmpGuard{yTmp, false};

        const int xFd = ::mkstemp(xTmp);
        const int yFd = ::mkstemp(yTmp);
        if (xFd < 0 || yFd < 0) {
            if (xFd >= 0) ::close(xFd);
            if (yFd >= 0) ::close(yFd);
            throw Seldon::IOException("Failed to create temporary binary files for streaming neural training");
        }
        xTmpGuard.active = true;
        yTmpGuard.active = true;

        std::ofstream xOut(xTmp, std::ios::binary | std::ios::trunc);
        std::ofstream yOut(yTmp, std::ios::binary | std::ios::trunc);
        if (!xOut || !yOut) {
            ::close(xFd);
            ::close(yFd);
            ::unlink(xTmp);
            ::unlink(yTmp);
            throw Seldon::IOException("Failed to open temporary binary output streams for neural training");
        }

        std::vector<std::vector<double>> chunkX;
        std::vector<std::vector<double>> chunkY;
        for (size_t start = 0; start < data.rowCount(); start += chunkRows) {
            const size_t end = std::min(start + chunkRows, data.rowCount());
            encodeNeuralRows(data, plan, start, end, chunkX);
            applyStandardScaler(chunkX, streamingMeans, streamingStds);

            chunkY.assign(end - start, std::vector<double>(outputNodes, 0.0));
            #ifdef USE_OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (size_t r = start; r < end; ++r) {
                const size_t local = r - start;
                for (size_t t = 0; t < targetIndices.size(); ++t) {
                    const int idx = targetIndices[t];
                    const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
                    chunkY[local][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
                }
            }

            std::vector<float> xRow(static_cast<size_t>(inputNodes), 0.0f);
            for (size_t r = 0; r < chunkX.size(); ++r) {
                for (size_t c = 0; c < inputNodes; ++c) xRow[c] = static_cast<float>(chunkX[r][c]);
                xOut.write(reinterpret_cast<const char*>(xRow.data()), static_cast<std::streamsize>(xRow.size() * sizeof(float)));
                yOut.write(reinterpret_cast<const char*>(chunkY[r].data()), static_cast<std::streamsize>(chunkY[r].size() * sizeof(double)));
            }

            std::vector<std::vector<double>>().swap(chunkX);
            std::vector<std::vector<double>>().swap(chunkY);
        }

        xOut.flush();
        yOut.flush();
        xOut.close();
        yOut.close();
        ::close(xFd);
        ::close(yFd);

        hp.useDiskStreaming = true;
        hp.inputBinaryPath = xTmp;
        hp.targetBinaryPath = yTmp;
        hp.streamingRows = data.rowCount();
        hp.streamingInputDim = inputNodes;
        hp.streamingOutputDim = outputNodes;
        hp.streamingChunkRows = chunkRows;
        hp.useMemoryMappedInput = true;

        nn.train({}, {}, hp);
        xTmpGuard.release();
        yTmpGuard.release();
        ::unlink(xTmp);
        ::unlink(yTmp);

        analysis.hiddenNodes = hidden;
        analysis.outputNodes = outputNodes;
        analysis.binaryTarget = (outputNodes == 1 && classificationTarget);
        analysis.hiddenActivation = activationToString(hp.activation);
        analysis.outputActivation = activationToString(hp.outputActivation);
        analysis.epochs = hp.epochs;
        analysis.batchSize = hp.batchSize;
        analysis.valSplit = hp.valSplit;
        analysis.l2Lambda = hp.l2Lambda;
        analysis.dropoutRate = hp.dropoutRate;
        analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
        analysis.policyUsed = "streaming_on_the_fly";
        analysis.explainabilityMethod = config.neuralExplainability;
        analysis.trainingRowsUsed = data.rowCount();
        analysis.topology = std::to_string(inputNodes) + " -> " + std::to_string(hidden) + " -> " + std::to_string(outputNodes);
        analysis.trainLoss = nn.getTrainLossHistory();
        analysis.valLoss = nn.getValLossHistory();
        analysis.gradientNorm = nn.getGradientNormHistory();
        analysis.weightStd = nn.getWeightStdHistory();
        analysis.weightMeanAbs = nn.getWeightMeanAbsHistory();

        const size_t sampleRows = std::min<size_t>(data.rowCount(), std::max<size_t>(64, std::min<size_t>(config.neuralImportanceMaxRows, static_cast<size_t>(1024))));
        std::vector<std::vector<double>> sampleX;
        std::vector<std::vector<double>> sampleY;
        encodeNeuralRows(data, plan, 0, sampleRows, sampleX);
        applyStandardScaler(sampleX, streamingMeans, streamingStds);
        sampleY.assign(sampleRows, std::vector<double>(outputNodes, 0.0));
        #ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t r = 0; r < sampleRows; ++r) {
            for (size_t t = 0; t < targetIndices.size(); ++t) {
                const int idx = targetIndices[t];
                const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
                sampleY[r][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
            }
        }

        const std::vector<double> rawImportance = nn.calculateFeatureImportance(sampleX,
                                                                                 sampleY,
                                                                                 (config.neuralImportanceTrials > 0) ? config.neuralImportanceTrials : 5,
                                                                                 config.neuralImportanceMaxRows,
                                                                                 config.neuralImportanceParallel);
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        for (size_t i = 0; i < rawImportance.size() && i < plan.sourceNumericFeaturePos.size(); ++i) {
            const int numericPos = plan.sourceNumericFeaturePos[i];
            if (numericPos >= 0) {
                const size_t numericPosU = static_cast<size_t>(numericPos);
                if (numericPosU < analysis.featureImportance.size()) {
                    analysis.featureImportance[numericPosU] += std::max(0.0, std::isfinite(rawImportance[i]) ? rawImportance[i] : 0.0);
                }
            }
        }

        if (!sampleX.empty()) {
            const size_t uncertaintyRows = std::min<size_t>(std::min<size_t>(sampleX.size(), 128), static_cast<size_t>(512));
            std::vector<double> outStd(outputNodes, 0.0);
            std::vector<double> outCiWidth(outputNodes, 0.0);
            for (size_t r = 0; r < uncertaintyRows; ++r) {
                const auto unc = nn.predictWithUncertainty(sampleX[r], config.neuralUncertaintySamples, hp.dropoutRate);
                for (size_t j = 0; j < outputNodes && j < unc.stddev.size(); ++j) {
                    outStd[j] += unc.stddev[j];
                    const double width = (j < unc.ciLow.size() && j < unc.ciHigh.size()) ? (unc.ciHigh[j] - unc.ciLow[j]) : 0.0;
                    outCiWidth[j] += width;
                }
            }
            if (uncertaintyRows > 0) {
                for (size_t j = 0; j < outputNodes; ++j) {
                    outStd[j] /= static_cast<double>(uncertaintyRows);
                    outCiWidth[j] /= static_cast<double>(uncertaintyRows);
                }
            }
            analysis.uncertaintyStd = std::move(outStd);
            analysis.uncertaintyCiWidth = std::move(outCiWidth);
        }

        if (config.neuralEnsembleMembers > 1 && sampleX.size() >= 32 && sampleY.size() == sampleX.size()) {
            const size_t members = std::min<size_t>(config.neuralEnsembleMembers, 7);
            const size_t probeRows = std::min<size_t>(sampleX.size(), std::max<size_t>(32, config.neuralEnsembleProbeRows));
            std::vector<std::vector<double>> memberPredSums(members, std::vector<double>(outputNodes, 0.0));

            for (size_t m = 0; m < members; ++m) {
                NeuralNet ens(topology);
                ens.setInputL2Scales(inputL2Scales);
                NeuralNet::Hyperparameters ehp = hp;
                ehp.epochs = std::min<size_t>(hp.epochs, std::max<size_t>(8, config.neuralEnsembleProbeEpochs));
                ehp.batchSize = std::min<size_t>(hp.batchSize, static_cast<size_t>(64));
                ehp.earlyStoppingPatience = std::min(hp.earlyStoppingPatience, 10);
                ehp.verbose = false;
                ehp.seed = config.neuralSeed + static_cast<uint32_t>(101 + 37 * m);
                ens.train(sampleX, sampleY, ehp);

                for (size_t r = 0; r < probeRows; ++r) {
                    const std::vector<double> pred = ens.predict(sampleX[r]);
                    for (size_t j = 0; j < outputNodes && j < pred.size(); ++j) {
                        memberPredSums[m][j] += pred[j];
                    }
                }
            }

            std::vector<double> ensStd(outputNodes, 0.0);
            for (size_t j = 0; j < outputNodes; ++j) {
                double mean = 0.0;
                for (size_t m = 0; m < members; ++m) {
                    mean += memberPredSums[m][j] / static_cast<double>(probeRows);
                }
                mean /= static_cast<double>(members);

                double var = 0.0;
                for (size_t m = 0; m < members; ++m) {
                    const double v = memberPredSums[m][j] / static_cast<double>(probeRows);
                    const double d = v - mean;
                    var += d * d;
                }
                ensStd[j] = std::sqrt(var / static_cast<double>(std::max<size_t>(1, members - 1)));
            }
            analysis.ensembleStd = std::move(ensStd);
        }

        const OodDriftDiagnostics ood = computeOodDriftDiagnostics(sampleX, config);
        analysis.oodRate = ood.oodRate;
        analysis.oodMeanDistance = ood.meanDistance;
        analysis.oodMaxDistance = ood.maxDistance;
        analysis.oodReferenceRows = ood.referenceRows;
        analysis.oodMonitorRows = ood.monitorRows;
        analysis.driftPsiMean = ood.psiMean;
        analysis.driftPsiMax = ood.psiMax;
        analysis.driftBand = ood.driftBand;
        analysis.driftWarning = ood.warning;
        analysis.confidenceScore = computeConfidenceScore(analysis.uncertaintyStd,
                                                          analysis.ensembleStd,
                                                          analysis.oodRate,
                                                          analysis.driftBand);

        return analysis;
    }

    EncodedNeuralMatrix encoded = buildEncodedNeuralInputs(data, targetIdx, featureIdx, config);
    std::vector<std::vector<double>> Xnn = std::move(encoded.X);
    std::vector<double> nnInputMeans;
    std::vector<double> nnInputStds;
    fitStandardScaler(Xnn, nnInputMeans, nnInputStds);
    applyStandardScaler(Xnn, nnInputMeans, nnInputStds);
    std::vector<std::vector<double>> Ynn(data.rowCount(), std::vector<double>(targetIndices.size(), 0.0));

    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t r = 0; r < data.rowCount(); ++r) {
        for (size_t t = 0; t < targetIndices.size(); ++t) {
            const int idx = targetIndices[t];
            const auto& targetVals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(idx)].values);
            Ynn[r][t] = (r < targetVals.size()) ? targetVals[r] : 0.0;
        }
    }

    analysis.outputAuxTargets = (targetIndices.size() > 1) ? (targetIndices.size() - 1) : 0;

    size_t inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
    analysis.inputNodes = inputNodes;
    analysis.categoricalColumnsUsed = encoded.categoricalColumnsUsed;
    analysis.categoricalEncodedNodes = encoded.categoricalEncodedNodes;

    auto applyAdaptiveSparsity = [&](size_t maxColumns) {
        if (Xnn.empty() || Xnn.front().empty()) return;
        const size_t cols = Xnn.front().size();
        if (cols <= maxColumns || maxColumns == 0) return;

        struct ColScore {
            size_t idx = 0;
            double score = 0.0;
        };

        std::vector<ColScore> scores;
        scores.reserve(cols);

        const size_t rows = std::min(Xnn.size(), Ynn.size());
        double yMean = 0.0;
        for (size_t r = 0; r < rows; ++r) yMean += Ynn[r][0];
        yMean /= std::max<size_t>(1, rows);

        double yVar = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            const double d = Ynn[r][0] - yMean;
            yVar += d * d;
        }
        yVar = std::max(yVar, 1e-12);

        for (size_t c = 0; c < cols; ++c) {
            double xMean = 0.0;
            for (size_t r = 0; r < rows; ++r) xMean += Xnn[r][c];
            xMean /= std::max<size_t>(1, rows);

            double xVar = 0.0;
            double cov = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                const double dx = Xnn[r][c] - xMean;
                const double dy = Ynn[r][0] - yMean;
                xVar += dx * dx;
                cov += dx * dy;
            }

            double score = 0.0;
            if (xVar > 1e-12) {
                score = std::abs(cov) / std::sqrt(xVar * yVar);
            }
            if (!std::isfinite(score)) score = 0.0;
            scores.push_back({c, score});
        }

        std::sort(scores.begin(), scores.end(), [](const ColScore& a, const ColScore& b) {
            if (a.score == b.score) return a.idx < b.idx;
            return a.score > b.score;
        });
        scores.resize(maxColumns);
        std::sort(scores.begin(), scores.end(), [](const ColScore& a, const ColScore& b) { return a.idx < b.idx; });

        std::vector<std::vector<double>> squeezedX(Xnn.size(), std::vector<double>{});
        for (size_t r = 0; r < Xnn.size(); ++r) {
            squeezedX[r].reserve(scores.size());
            for (const auto& sc : scores) {
                squeezedX[r].push_back(Xnn[r][sc.idx]);
            }
        }

        std::vector<int> squeezedSource;
        squeezedSource.reserve(scores.size());
        for (const auto& sc : scores) {
            if (sc.idx < encoded.sourceNumericFeaturePos.size()) {
                squeezedSource.push_back(encoded.sourceNumericFeaturePos[sc.idx]);
            } else {
                squeezedSource.push_back(-1);
            }
        }

        Xnn = std::move(squeezedX);
        encoded.sourceNumericFeaturePos = std::move(squeezedSource);
    };

    if (inputNodes > 0) {
        const double dofRatio = static_cast<double>(data.rowCount()) / static_cast<double>(std::max<size_t>(1, inputNodes));
        if (dofRatio < 10.0) {
            struct ColScore {
                size_t idx = 0;
                double score = 0.0;
            };

            std::vector<ColScore> engineeredScores;
            const size_t rows = std::min(Xnn.size(), Ynn.size());
            double yMean = 0.0;
            for (size_t r = 0; r < rows; ++r) yMean += Ynn[r][0];
            yMean /= std::max<size_t>(1, rows);
            double yVar = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                const double d = Ynn[r][0] - yMean;
                yVar += d * d;
            }
            yVar = std::max(1e-12, yVar);

            for (size_t c = 0; c < inputNodes; ++c) {
                if (c >= encoded.sourceNumericFeaturePos.size()) continue;
                const int numericPos = encoded.sourceNumericFeaturePos[c];
                if (numericPos < 0 || static_cast<size_t>(numericPos) >= featureIdx.size()) continue;
                const int sourceCol = featureIdx[static_cast<size_t>(numericPos)];
                if (sourceCol < 0 || static_cast<size_t>(sourceCol) >= data.columns().size()) continue;
                if (!isEngineeredFeatureName(data.columns()[static_cast<size_t>(sourceCol)].name)) continue;

                double xMean = 0.0;
                for (size_t r = 0; r < rows; ++r) xMean += Xnn[r][c];
                xMean /= std::max<size_t>(1, rows);
                double xVar = 0.0;
                double cov = 0.0;
                for (size_t r = 0; r < rows; ++r) {
                    const double dx = Xnn[r][c] - xMean;
                    const double dy = Ynn[r][0] - yMean;
                    xVar += dx * dx;
                    cov += dx * dy;
                }
                const double s = (xVar > 1e-12) ? std::abs(cov) / std::sqrt(std::max(1e-12, xVar * yVar)) : 0.0;
                engineeredScores.push_back({c, std::isfinite(s) ? s : 0.0});
            }

            if (engineeredScores.size() >= 4) {
                std::sort(engineeredScores.begin(), engineeredScores.end(), [](const ColScore& a, const ColScore& b) {
                    if (a.score == b.score) return a.idx < b.idx;
                    return a.score < b.score;
                });

                const size_t dropCount = engineeredScores.size() / 2;
                if (dropCount > 0 && (inputNodes - dropCount) >= 4) {
                    std::vector<bool> keepMask(inputNodes, true);
                    for (size_t i = 0; i < dropCount; ++i) keepMask[engineeredScores[i].idx] = false;

                    std::vector<std::vector<double>> prunedX(Xnn.size(), std::vector<double>{});
                    for (size_t r = 0; r < Xnn.size(); ++r) {
                        prunedX[r].reserve(inputNodes - dropCount);
                        for (size_t c = 0; c < inputNodes; ++c) {
                            if (keepMask[c]) prunedX[r].push_back(Xnn[r][c]);
                        }
                    }

                    std::vector<int> prunedSource;
                    prunedSource.reserve(inputNodes - dropCount);
                    for (size_t c = 0; c < inputNodes; ++c) {
                        if (keepMask[c]) {
                            prunedSource.push_back((c < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[c] : -1);
                        }
                    }

                    Xnn = std::move(prunedX);
                    encoded.sourceNumericFeaturePos = std::move(prunedSource);
                    inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
                    analysis.inputNodes = inputNodes;
                    analysis.strictPruningApplied = true;
                    analysis.strictPrunedColumns = dropCount;
                }
            }
        }

        if (dofRatio < 10.0) {
            const size_t squeezeCap = std::clamp<size_t>(data.rowCount() / 10, 4, std::max<size_t>(4, inputNodes));
            applyAdaptiveSparsity(squeezeCap);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }

        if (config.lowMemoryMode && inputNodes > 64) {
            const size_t lowMemCap = std::clamp<size_t>(std::max<size_t>(24, data.rowCount() / 12), 24, static_cast<size_t>(64));
            applyAdaptiveSparsity(lowMemCap);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }
    }

    analysis.categoricalEncodedNodes = static_cast<size_t>(std::count_if(
        encoded.sourceNumericFeaturePos.begin(),
        encoded.sourceNumericFeaturePos.end(),
        [](int v) { return v < 0; }));

    if (inputNodes == 0) {
        analysis.policyUsed = "none_no_encoded_features";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }

    size_t outputNodes = std::max<size_t>(1, Ynn.empty() ? 1 : Ynn.front().size());

    if (inputNodes >= 48 && Xnn.size() >= 64) {
        if (inputNodes > 96 && !Ynn.empty() && !Ynn.front().empty()) {
            const size_t preProbeCap = std::clamp<size_t>(std::max<size_t>(64, Xnn.size() / 3), 64, 192);
            if (preProbeCap < inputNodes) {
                std::vector<std::pair<size_t, double>> ranked;
                ranked.reserve(inputNodes);

                double yMean = 0.0;
                for (const auto& row : Ynn) yMean += row[0];
                yMean /= static_cast<double>(std::max<size_t>(1, Ynn.size()));
                double yVar = 0.0;
                for (const auto& row : Ynn) {
                    const double d = row[0] - yMean;
                    yVar += d * d;
                }
                yVar = std::max(1e-12, yVar);

                for (size_t c = 0; c < inputNodes; ++c) {
                    double xMean = 0.0;
                    for (const auto& row : Xnn) xMean += row[c];
                    xMean /= static_cast<double>(std::max<size_t>(1, Xnn.size()));
                    double xVar = 0.0;
                    double cov = 0.0;
                    for (size_t r = 0; r < Xnn.size(); ++r) {
                        const double dx = Xnn[r][c] - xMean;
                        const double dy = Ynn[r][0] - yMean;
                        xVar += dx * dx;
                        cov += dx * dy;
                    }
                    const double score = (xVar > 1e-12) ? std::abs(cov) / std::sqrt(std::max(1e-12, xVar * yVar)) : 0.0;
                    ranked.push_back({c, std::isfinite(score) ? score : 0.0});
                }

                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                    if (a.second == b.second) return a.first < b.first;
                    return a.second > b.second;
                });
                ranked.resize(preProbeCap);
                std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

                std::vector<std::vector<double>> reducedX(Xnn.size(), std::vector<double>{});
                for (size_t r = 0; r < Xnn.size(); ++r) {
                    reducedX[r].reserve(ranked.size());
                    for (const auto& kv : ranked) reducedX[r].push_back(Xnn[r][kv.first]);
                }

                std::vector<int> reducedSource;
                reducedSource.reserve(ranked.size());
                for (const auto& kv : ranked) {
                    const size_t idx = kv.first;
                    reducedSource.push_back((idx < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[idx] : -1);
                }

                Xnn = std::move(reducedX);
                encoded.sourceNumericFeaturePos = std::move(reducedSource);
                inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
                analysis.inputNodes = inputNodes;
            }
        }

        const size_t probeRows = std::min<size_t>(Xnn.size(), std::max<size_t>(256, std::min<size_t>(config.neuralImportanceMaxRows, static_cast<size_t>(2000))));
        std::vector<size_t> probeOrder(Xnn.size());
        std::iota(probeOrder.begin(), probeOrder.end(), 0);
        std::mt19937 probeRng(config.neuralSeed ^ 0x85ebca6bU);
        std::shuffle(probeOrder.begin(), probeOrder.end(), probeRng);

        std::vector<std::vector<double>> probeX;
        std::vector<std::vector<double>> probeY;
        probeX.reserve(probeRows);
        probeY.reserve(probeRows);
        for (size_t i = 0; i < probeRows; ++i) {
            probeX.push_back(Xnn[probeOrder[i]]);
            probeY.push_back(Ynn[probeOrder[i]]);
        }

        const size_t probeHidden = std::clamp<size_t>(static_cast<size_t>(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes)))), 8, 48);
        NeuralNet probeNet({inputNodes, probeHidden, outputNodes});
        NeuralNet::Hyperparameters probeHp;
        probeHp.epochs = std::min<size_t>(60, std::max<size_t>(24, 8 + probeRows / 20));
        probeHp.batchSize = std::clamp<size_t>(probeRows / 16, 8, 64);
        probeHp.learningRate = config.neuralLearningRate;
        probeHp.dropoutRate = 0.05;
        probeHp.valSplit = 0.2;
        probeHp.earlyStoppingPatience = 6;
        probeHp.activation = NeuralNet::Activation::RELU;
        probeHp.outputActivation = useCrossEntropy ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
        probeHp.loss = useCrossEntropy ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
        probeHp.verbose = false;
        probeHp.seed = config.neuralSeed + 17;
        probeHp.importanceMaxRows = std::min<size_t>(config.neuralImportanceMaxRows, 2000);
        probeHp.importanceParallel = config.neuralImportanceParallel;
        probeNet.train(probeX, probeY, probeHp);

        const std::vector<double> probeImportance = probeNet.calculateFeatureImportance(
            probeX,
            probeY,
            3,
            std::min<size_t>(config.neuralImportanceMaxRows, 2000),
            config.neuralImportanceParallel);

        const size_t retainCap = std::clamp<size_t>(std::max<size_t>(24, data.rowCount() / 3), 24, inputNodes);
        if (retainCap < inputNodes && probeImportance.size() == inputNodes) {
            std::vector<std::pair<size_t, double>> ranked;
            ranked.reserve(inputNodes);
            for (size_t i = 0; i < inputNodes; ++i) ranked.push_back({i, probeImportance[i]});
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                if (a.second == b.second) return a.first < b.first;
                return a.second > b.second;
            });
            ranked.resize(retainCap);
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

            std::vector<std::vector<double>> prunedX(Xnn.size(), std::vector<double>{});
            for (size_t r = 0; r < Xnn.size(); ++r) {
                prunedX[r].reserve(retainCap);
                for (const auto& kv : ranked) prunedX[r].push_back(Xnn[r][kv.first]);
            }

            std::vector<int> prunedSource;
            prunedSource.reserve(retainCap);
            for (const auto& kv : ranked) {
                const size_t idx = kv.first;
                prunedSource.push_back((idx < encoded.sourceNumericFeaturePos.size()) ? encoded.sourceNumericFeaturePos[idx] : -1);
            }

            Xnn = std::move(prunedX);
            encoded.sourceNumericFeaturePos = std::move(prunedSource);
            inputNodes = Xnn.empty() ? 0 : Xnn.front().size();
            analysis.inputNodes = inputNodes;
        }
    }

    if (inputNodes == 0) {
        analysis.policyUsed = "none_after_pruning";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }
    size_t baseHidden = std::clamp<size_t>(
        static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes) * std::max<size_t>(8, data.rowCount() / 6))))),
        4,
        64);

    size_t baseBatch = 16;
    if (data.rowCount() >= 512) baseBatch = 64;
    else if (data.rowCount() >= 192) baseBatch = 32;
    else if (data.rowCount() < 64) baseBatch = 8;

    size_t baseEpochs = std::clamp<size_t>(120 + data.rowCount() / 2, 120, 320);
    int basePatience = static_cast<int>(std::clamp<size_t>(baseEpochs / 12, 8, 24));
    double baseDropout = (inputNodes >= 12) ? 0.12 : (inputNodes >= 6 ? 0.08 : 0.04);

    struct NeuralPolicy {
        std::string name;
        double hiddenMultiplier;
        double epochMultiplier;
        double dropoutDelta;
        double batchMultiplier;
        int patienceDelta;
    };
    const std::unordered_map<std::string, NeuralPolicy> registry = {
        {"fast", {"fast", 0.80, 0.65, -0.02, 1.40, -3}},
        {"balanced", {"balanced", 1.00, 1.00, 0.00, 1.00, 0}},
        {"expressive", {"expressive", 1.45, 1.45, 0.04, 0.75, 5}},
    };

    std::string requested = CommonUtils::toLower(config.neuralStrategy);
    if (requested == "none") {
        analysis.hiddenNodes = 0;
        analysis.policyUsed = "none";
        analysis.featureImportance.assign(featureIdx.size(), 0.0);
        analysis.trainingRowsUsed = data.rowCount();
        return analysis;
    }
    std::string policyName = requested;
    if (requested == "auto") {
        double complexity = std::log1p(static_cast<double>(data.rowCount())) * std::sqrt(static_cast<double>(std::max<size_t>(1, inputNodes)));
        if (static_cast<double>(data.rowCount()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
            policyName = "fast";
        } else if (complexity < 10.0) policyName = "fast";
        else if (complexity > 22.0) policyName = "expressive";
        else policyName = "balanced";
    }
    if (registry.find(policyName) == registry.end()) policyName = "balanced";
    if (fastModeEnabled) policyName = "fast";
    const NeuralPolicy& policy = registry.at(policyName);

    if (fastModeEnabled && Xnn.size() > fastSampleRows) {
        std::vector<size_t> order(Xnn.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(config.neuralSeed ^ 0x9e3779b9U);
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<std::vector<double>> sampledX;
        std::vector<std::vector<double>> sampledY;
        sampledX.reserve(fastSampleRows);
        sampledY.reserve(fastSampleRows);
        for (size_t i = 0; i < fastSampleRows; ++i) {
            sampledX.push_back(std::move(Xnn[order[i]]));
            sampledY.push_back(std::move(Ynn[order[i]]));
        }
        Xnn = std::move(sampledX);
        Ynn = std::move(sampledY);
    }

    size_t hidden = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseHidden) * policy.hiddenMultiplier)),
                                       4,
                                       static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));
    size_t dynamicEpochs = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseEpochs) * policy.epochMultiplier)), 80, 420);
    size_t dynamicBatch = std::clamp<size_t>(static_cast<size_t>(std::llround(static_cast<double>(baseBatch) * policy.batchMultiplier)), 8, 128);
    int dynamicPatience = std::clamp(basePatience + policy.patienceDelta, 6, 40);
    double dynamicDropout = std::clamp(baseDropout + policy.dropoutDelta, 0.0, 0.35);

    if (Xnn.size() < 80) {
        hidden = std::min<size_t>(hidden, 24);
        dynamicEpochs = std::min<size_t>(dynamicEpochs, 140);
        dynamicPatience = std::min(dynamicPatience, 8);
        dynamicDropout = std::max(dynamicDropout, 0.10);
    }

    if (static_cast<double>(Xnn.size()) < 10.0 * static_cast<double>(std::max<size_t>(1, inputNodes))) {
        const size_t rowDrivenCap = std::clamp<size_t>(Xnn.size() / 3, 4, 20);
        hidden = std::min(hidden, rowDrivenCap);
        dynamicEpochs = std::min<size_t>(dynamicEpochs, 120);
        dynamicPatience = std::min(dynamicPatience, 7);
        dynamicDropout = std::max(dynamicDropout, 0.12);
    }

    size_t hiddenLayers = 1;
    if (config.neuralFixedLayers > 0) {
        hiddenLayers = static_cast<size_t>(config.neuralFixedLayers);
    } else {
        hiddenLayers = static_cast<size_t>(std::clamp<int>(
            1 + ((inputNodes > 24) ? 1 : 0) + ((outputNodes > 1) ? 1 : 0) + ((policyName == "expressive") ? 1 : 0),
            config.neuralMinLayers,
            config.neuralMaxLayers));
        if (outputNodes <= 4 && inputNodes >= 32) {
            hiddenLayers = std::clamp<size_t>(std::max<size_t>(hiddenLayers, 4),
                                              static_cast<size_t>(config.neuralMinLayers),
                                              static_cast<size_t>(config.neuralMaxLayers));
        }
    }

    if (config.neuralFixedHiddenNodes > 0) {
        hidden = static_cast<size_t>(config.neuralFixedHiddenNodes);
    }
    if (data.rowCount() < 500) {
        hidden = std::min(hidden, std::max<size_t>(1, inputNodes));
    }

    std::vector<size_t> topology = buildNormalizedTopology(inputNodes,
                                                           outputNodes,
                                                           hiddenLayers,
                                                           hidden,
                                                           static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));

    if (data.rowCount() < 500 && topology.size() > 2) {
        const size_t smallDataCap = std::max<size_t>(1, inputNodes);
        for (size_t i = 1; i + 1 < topology.size(); ++i) {
            topology[i] = std::min(topology[i], smallDataCap);
        }
    }

    auto estimateTrainableParams = [](const std::vector<size_t>& topo) {
        size_t params = 0;
        if (topo.size() < 2) return params;
        for (size_t i = 1; i < topo.size(); ++i) {
            params += topo[i - 1] * topo[i];
            params += topo[i];
        }
        return params;
    };

    const size_t topologyNodes = std::accumulate(topology.begin(), topology.end(), static_cast<size_t>(0));
    const size_t trainableParams = estimateTrainableParams(topology);
    if (topologyNodes > config.neuralMaxTopologyNodes) {
        throw Seldon::ConfigurationException(
            "Neural topology exceeds neural_max_topology_nodes (" +
            std::to_string(topologyNodes) + " > " + std::to_string(config.neuralMaxTopologyNodes) + ")");
    }
    if (trainableParams > config.neuralMaxTrainableParams) {
        throw Seldon::ConfigurationException(
            "Neural parameter count exceeds neural_max_trainable_params (" +
            std::to_string(trainableParams) + " > " + std::to_string(config.neuralMaxTrainableParams) + ")");
    }

    NeuralNet nn(topology);
    NeuralNet::Hyperparameters hp;

    auto toOptimizer = [](const std::string& name) {
        const std::string n = CommonUtils::toLower(name);
        if (n == "sgd") return NeuralNet::Optimizer::SGD;
        if (n == "adam") return NeuralNet::Optimizer::ADAM;
        return NeuralNet::Optimizer::LOOKAHEAD;
    };

    hp.epochs = dynamicEpochs;
    hp.batchSize = dynamicBatch;
    hp.learningRate = config.neuralLearningRate;
    hp.earlyStoppingPatience = dynamicPatience;
    hp.lrDecay = config.neuralLrDecay;
    hp.lrPlateauPatience = config.neuralLrPlateauPatience;
    hp.lrCooldownEpochs = config.neuralLrCooldownEpochs;
    hp.maxLrReductions = config.neuralMaxLrReductions;
    hp.minLearningRate = config.neuralMinLearningRate;
    hp.lrWarmupEpochs = static_cast<size_t>(std::max(0, config.neuralLrWarmupEpochs));
    hp.useCosineAnnealing = config.neuralUseCosineAnnealing;
    hp.useCyclicalLr = config.neuralUseCyclicalLr;
    hp.lrCycleEpochs = static_cast<size_t>(std::max(2, config.neuralLrCycleEpochs));
    hp.lrScheduleMinFactor = config.neuralLrScheduleMinFactor;
    hp.useValidationLossEma = config.neuralUseValidationLossEma;
    hp.validationLossEmaBeta = config.neuralValidationLossEmaBeta;
    hp.dropoutRate = dynamicDropout;
    double desiredValSplit = (Xnn.size() < 80) ? 0.30 : 0.20;
    if (Xnn.size() >= 16) {
        const size_t minValRows = std::min<size_t>(std::max<size_t>(8, Xnn.size() / 5), Xnn.size() - 8);
        desiredValSplit = std::max(desiredValSplit, static_cast<double>(minValRows) / static_cast<double>(Xnn.size()));
    }
    hp.valSplit = std::clamp(desiredValSplit, 0.10, 0.40);
    hp.l2Lambda = (Xnn.size() < 80) ? 0.010 : 0.001;
    hp.categoricalInputL2Boost = config.neuralCategoricalInputL2Boost;
    if (policyName == "fast") hp.activation = NeuralNet::Activation::RELU;
    else if (policyName == "expressive") hp.activation = NeuralNet::Activation::GELU;
    else hp.activation = (inputNodes < 10) ? NeuralNet::Activation::TANH : NeuralNet::Activation::GELU;
    hp.outputActivation = useCrossEntropy ? NeuralNet::Activation::SIGMOID : NeuralNet::Activation::LINEAR;
    hp.useBatchNorm = config.neuralUseBatchNorm;
    hp.batchNormMomentum = config.neuralBatchNormMomentum;
    hp.batchNormEpsilon = config.neuralBatchNormEpsilon;
    hp.useLayerNorm = config.neuralUseLayerNorm;
    hp.layerNormEpsilon = config.neuralLayerNormEpsilon;
    hp.optimizer = toOptimizer(config.neuralOptimizer);
    hp.lookaheadFastOptimizer = toOptimizer(config.neuralLookaheadFastOptimizer);
    hp.lookaheadSyncPeriod = static_cast<size_t>(std::max(1, config.neuralLookaheadSyncPeriod));
    hp.lookaheadAlpha = config.neuralLookaheadAlpha;
    hp.loss = useCrossEntropy ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
    hp.importanceMaxRows = config.neuralImportanceMaxRows;
    hp.importanceParallel = config.neuralImportanceParallel;
    hp.verbose = verbose;
    hp.seed = config.neuralSeed;
    hp.gradientClipNorm = config.gradientClipNorm;
    hp.adaptiveGradientClipping = config.neuralUseAdaptiveGradientClipping;
    hp.adaptiveClipBeta = config.neuralAdaptiveClipBeta;
    hp.adaptiveClipMultiplier = config.neuralAdaptiveClipMultiplier;
    hp.adaptiveClipMin = config.neuralAdaptiveClipMin;
    hp.gradientNoiseStd = config.neuralGradientNoiseStd;
    hp.gradientNoiseDecay = config.neuralGradientNoiseDecay;
    hp.useEmaWeights = config.neuralUseEmaWeights;
    hp.emaDecay = config.neuralEmaDecay;
    hp.labelSmoothing = config.neuralLabelSmoothing;
    hp.gradientAccumulationSteps = static_cast<size_t>(std::max(1, config.neuralGradientAccumulationSteps));

    std::vector<double> inputL2Scales(inputNodes, 1.0);
    for (size_t i = 0; i < inputL2Scales.size() && i < encoded.sourceNumericFeaturePos.size(); ++i) {
        if (encoded.sourceNumericFeaturePos[i] < 0) {
            inputL2Scales[i] = hp.categoricalInputL2Boost;
        }
    }

    auto buildTopology = [&](size_t layers, size_t firstHidden) {
        std::vector<size_t> built = buildNormalizedTopology(inputNodes,
                                                            outputNodes,
                                                            layers,
                                                            firstHidden,
                                                            static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes)));
        if (data.rowCount() < 500 && built.size() > 2) {
            const size_t smallDataCap = std::max<size_t>(1, inputNodes);
            for (size_t i = 1; i + 1 < built.size(); ++i) {
                built[i] = std::min(built[i], smallDataCap);
            }
        }
        return built;
    };

    if (requested == "auto" && config.neuralFixedLayers == 0 && config.neuralFixedHiddenNodes == 0 && Xnn.size() >= 64) {
        struct ArchCandidate {
            size_t layers;
            size_t hidden;
            NeuralNet::Activation activation;
            double dropout;
        };

        std::vector<ArchCandidate> candidates = {
            {hiddenLayers, hidden, hp.activation, hp.dropoutRate},
            {std::clamp<size_t>(hiddenLayers + 1, static_cast<size_t>(config.neuralMinLayers), static_cast<size_t>(config.neuralMaxLayers)), std::min<size_t>(hidden + hidden / 4, static_cast<size_t>(std::max(4, config.neuralMaxHiddenNodes))), NeuralNet::Activation::GELU, std::clamp(hp.dropoutRate + 0.04, 0.0, 0.40)},
            {std::max<size_t>(1, hiddenLayers), std::max<size_t>(4, hidden - hidden / 5), NeuralNet::Activation::RELU, std::max(0.0, hp.dropoutRate - 0.03)}
        };

        const size_t searchRows = std::min<size_t>(Xnn.size(), 2000);
        std::vector<size_t> order(Xnn.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 searchRng(config.neuralSeed ^ 0xc2b2ae35U);
        std::shuffle(order.begin(), order.end(), searchRng);

        std::vector<std::vector<double>> searchX;
        std::vector<std::vector<double>> searchY;
        searchX.reserve(searchRows);
        searchY.reserve(searchRows);
        for (size_t i = 0; i < searchRows; ++i) {
            searchX.push_back(Xnn[order[i]]);
            searchY.push_back(Ynn[order[i]]);
        }

        double bestScore = std::numeric_limits<double>::infinity();
        ArchCandidate best = candidates.front();

        for (size_t c = 0; c < candidates.size(); ++c) {
            const auto& cand = candidates[c];
            const std::vector<size_t> candTopology = buildTopology(cand.layers, cand.hidden);
            const size_t candNodes = std::accumulate(candTopology.begin(), candTopology.end(), static_cast<size_t>(0));
            const size_t candParams = estimateTrainableParams(candTopology);
            if (candNodes > config.neuralMaxTopologyNodes || candParams > config.neuralMaxTrainableParams) {
                continue;
            }

            NeuralNet candidateNet(candTopology);
            candidateNet.setInputL2Scales(inputL2Scales);

            NeuralNet::Hyperparameters chp = hp;
            chp.epochs = std::min<size_t>(60, std::max<size_t>(24, hp.epochs / 4));
            chp.earlyStoppingPatience = std::min(10, hp.earlyStoppingPatience);
            chp.batchSize = std::min<size_t>(hp.batchSize, 64);
            chp.activation = cand.activation;
            chp.dropoutRate = cand.dropout;
            chp.verbose = false;
            chp.seed = config.neuralSeed + static_cast<uint32_t>(31 + c * 13);
            candidateNet.train(searchX, searchY, chp);

            const auto& vloss = candidateNet.getValLossHistory();
            const auto& tloss = candidateNet.getTrainLossHistory();
            const double score = !vloss.empty()
                ? vloss.back()
                : (!tloss.empty() ? tloss.back() : std::numeric_limits<double>::infinity());
            if (score < bestScore) {
                bestScore = score;
                best = cand;
            }
        }

        hiddenLayers = best.layers;
        hidden = best.hidden;
        hp.activation = best.activation;
        hp.dropoutRate = best.dropout;
        topology = buildTopology(hiddenLayers, hidden);
        nn = NeuralNet(topology);
    }

    nn.setInputL2Scales(inputL2Scales);

    if (config.neuralStreamingMode) {
        nn.trainIncremental(Xnn, Ynn, hp, std::max<size_t>(16, config.neuralStreamingChunkRows));
    } else {
        nn.train(Xnn, Ynn, hp);
    }

    const double stabilityProbe = estimateTrainingStability(nn.getTrainLossHistory(), nn.getValLossHistory());
    if (stabilityProbe < 0.70 && hidden > 4 && config.neuralFixedHiddenNodes == 0) {
        const size_t reducedHidden = std::max<size_t>(4, static_cast<size_t>(std::llround(static_cast<double>(hidden) * 0.65)));
        if (reducedHidden < hidden) {
            hidden = reducedHidden;
            topology = buildTopology(hiddenLayers, hidden);
            nn = NeuralNet(topology);
            nn.setInputL2Scales(inputL2Scales);
            hp.epochs = std::min<size_t>(hp.epochs, 180);
            hp.earlyStoppingPatience = std::min(hp.earlyStoppingPatience, 10);
            if (config.neuralStreamingMode) {
                nn.trainIncremental(Xnn, Ynn, hp, std::max<size_t>(16, config.neuralStreamingChunkRows));
            } else {
                nn.train(Xnn, Ynn, hp);
            }
            analysis.policyUsed = policy.name + "+stability_shrink";
        }
    }

    analysis.inputNodes = inputNodes;
    analysis.hiddenNodes = hidden;
    analysis.outputNodes = outputNodes;
    analysis.binaryTarget = (outputNodes == 1 && classificationTarget);
    analysis.classificationTarget = classificationTarget;
    analysis.hiddenActivation = activationToString(hp.activation);
    analysis.outputActivation = activationToString(hp.outputActivation);
    analysis.lossName = (hp.loss == NeuralNet::LossFunction::CROSS_ENTROPY) ? "cross_entropy" : "mse";
    analysis.epochs = hp.epochs;
    analysis.batchSize = hp.batchSize;
    analysis.valSplit = hp.valSplit;
    analysis.l2Lambda = hp.l2Lambda;
    analysis.dropoutRate = hp.dropoutRate;
    analysis.earlyStoppingPatience = hp.earlyStoppingPatience;
    if (analysis.policyUsed.empty()) analysis.policyUsed = policy.name;
    analysis.explainabilityMethod = config.neuralExplainability;
    analysis.trainingRowsUsed = Xnn.size();
    {
        std::ostringstream topo;
        for (size_t i = 0; i < topology.size(); ++i) {
            if (i > 0) topo << " -> ";
            topo << topology[i];
        }
        analysis.topology = topo.str();
    }
    analysis.trainLoss = nn.getTrainLossHistory();
    analysis.valLoss = nn.getValLossHistory();
    analysis.gradientNorm = nn.getGradientNormHistory();
    analysis.weightStd = nn.getWeightStdHistory();
    analysis.weightMeanAbs = nn.getWeightMeanAbsHistory();

    const size_t importanceTrials = (config.neuralImportanceTrials > 0)
        ? config.neuralImportanceTrials
        : static_cast<size_t>(inputNodes > 10 ? 7 : 5);

    std::vector<double> rawImportance;
    const std::string explainability = CommonUtils::toLower(config.neuralExplainability);
    const size_t explainabilitySteps = (explainability == "hybrid") ? 2 : 1;
    size_t explainabilityStep = 0;
    printProgressBar("feature-extraction", explainabilityStep, explainabilitySteps);

    auto advanceExplainability = [&]() {
        explainabilityStep = std::min(explainabilityStep + 1, explainabilitySteps);
        printProgressBar("feature-extraction", explainabilityStep, explainabilitySteps);
    };

    if (explainability == "integrated_gradients") {
        rawImportance = nn.calculateIntegratedGradients(Xnn,
                                                        config.neuralIntegratedGradSteps,
                                                        config.neuralImportanceMaxRows);
        advanceExplainability();
    } else if (explainability == "hybrid") {
        const std::vector<double> perm = nn.calculateFeatureImportance(Xnn,
                                                                        Ynn,
                                                                        importanceTrials,
                                                                        config.neuralImportanceMaxRows,
                                                                        config.neuralImportanceParallel);
        advanceExplainability();
        const std::vector<double> ig = nn.calculateIntegratedGradients(Xnn,
                                                                        config.neuralIntegratedGradSteps,
                                                                        config.neuralImportanceMaxRows);
        advanceExplainability();
        const double wp = std::max(0.0, config.tuning.hybridExplainabilityWeightPermutation);
        const double wi = std::max(0.0, config.tuning.hybridExplainabilityWeightIntegratedGradients);
        const double wsum = std::max(config.tuning.numericEpsilon, wp + wi);
        rawImportance.assign(std::max(perm.size(), ig.size()), 0.0);
        for (size_t i = 0; i < rawImportance.size(); ++i) {
            const double p = (i < perm.size()) ? perm[i] : 0.0;
            const double g = (i < ig.size()) ? ig[i] : 0.0;
            rawImportance[i] = (wp * p + wi * g) / wsum;
        }
    } else {
        rawImportance = nn.calculateFeatureImportance(Xnn,
                                                      Ynn,
                                                      importanceTrials,
                                                      config.neuralImportanceMaxRows,
                                                      config.neuralImportanceParallel);
        advanceExplainability();
    }

    if (!Xnn.empty()) {
        const size_t uncertaintyRows = std::min<size_t>(std::min<size_t>(Xnn.size(), 128), static_cast<size_t>(512));
        std::vector<double> outStd(outputNodes, 0.0);
        std::vector<double> outCiWidth(outputNodes, 0.0);
        for (size_t r = 0; r < uncertaintyRows; ++r) {
            const auto unc = nn.predictWithUncertainty(Xnn[r], config.neuralUncertaintySamples, hp.dropoutRate);
            for (size_t j = 0; j < outputNodes && j < unc.stddev.size(); ++j) {
                outStd[j] += unc.stddev[j];
                const double width = (j < unc.ciLow.size() && j < unc.ciHigh.size()) ? (unc.ciHigh[j] - unc.ciLow[j]) : 0.0;
                outCiWidth[j] += width;
            }
        }
        if (uncertaintyRows > 0) {
            for (size_t j = 0; j < outputNodes; ++j) {
                outStd[j] /= static_cast<double>(uncertaintyRows);
                outCiWidth[j] /= static_cast<double>(uncertaintyRows);
            }
        }
        analysis.uncertaintyStd = std::move(outStd);
        analysis.uncertaintyCiWidth = std::move(outCiWidth);
    }

    if (config.neuralEnsembleMembers > 1 && Xnn.size() >= 32 && Ynn.size() == Xnn.size()) {
        const size_t members = std::min<size_t>(config.neuralEnsembleMembers, 7);
        const size_t trainRows = std::min<size_t>(Xnn.size(), std::max<size_t>(32, config.neuralEnsembleProbeRows));
        std::vector<size_t> rows(Xnn.size());
        std::iota(rows.begin(), rows.end(), 0);
        std::mt19937 ensRng(config.neuralSeed ^ 0x517cc1b7U);
        std::shuffle(rows.begin(), rows.end(), ensRng);
        rows.resize(trainRows);

        std::vector<std::vector<double>> ensX;
        std::vector<std::vector<double>> ensY;
        ensX.reserve(trainRows);
        ensY.reserve(trainRows);
        for (size_t idx : rows) {
            ensX.push_back(Xnn[idx]);
            ensY.push_back(Ynn[idx]);
        }

        const size_t probeRows = std::min<size_t>(ensX.size(), static_cast<size_t>(128));
        std::vector<std::vector<double>> memberPredSums(members, std::vector<double>(outputNodes, 0.0));
        for (size_t m = 0; m < members; ++m) {
            NeuralNet ens(topology);
            ens.setInputL2Scales(inputL2Scales);
            NeuralNet::Hyperparameters ehp = hp;
            ehp.epochs = std::min<size_t>(hp.epochs, std::max<size_t>(8, config.neuralEnsembleProbeEpochs));
            ehp.batchSize = std::min<size_t>(hp.batchSize, static_cast<size_t>(64));
            ehp.earlyStoppingPatience = std::min(hp.earlyStoppingPatience, 10);
            ehp.verbose = false;
            ehp.seed = config.neuralSeed + static_cast<uint32_t>(101 + 37 * m);
            ens.train(ensX, ensY, ehp);

            for (size_t r = 0; r < probeRows; ++r) {
                const std::vector<double> pred = ens.predict(ensX[r]);
                for (size_t j = 0; j < outputNodes && j < pred.size(); ++j) {
                    memberPredSums[m][j] += pred[j];
                }
            }
        }

        std::vector<double> ensStd(outputNodes, 0.0);
        for (size_t j = 0; j < outputNodes; ++j) {
            double mean = 0.0;
            for (size_t m = 0; m < members; ++m) {
                mean += memberPredSums[m][j] / static_cast<double>(probeRows);
            }
            mean /= static_cast<double>(members);

            double var = 0.0;
            for (size_t m = 0; m < members; ++m) {
                const double v = memberPredSums[m][j] / static_cast<double>(probeRows);
                const double d = v - mean;
                var += d * d;
            }
            ensStd[j] = std::sqrt(var / static_cast<double>(std::max<size_t>(1, members - 1)));
        }
        analysis.ensembleStd = std::move(ensStd);
    }

    const OodDriftDiagnostics ood = computeOodDriftDiagnostics(Xnn, config);
    analysis.oodRate = ood.oodRate;
    analysis.oodMeanDistance = ood.meanDistance;
    analysis.oodMaxDistance = ood.maxDistance;
    analysis.oodReferenceRows = ood.referenceRows;
    analysis.oodMonitorRows = ood.monitorRows;
    analysis.driftPsiMean = ood.psiMean;
    analysis.driftPsiMax = ood.psiMax;
    analysis.driftBand = ood.driftBand;
    analysis.driftWarning = ood.warning;
    analysis.confidenceScore = computeConfidenceScore(analysis.uncertaintyStd,
                                                      analysis.ensembleStd,
                                                      analysis.oodRate,
                                                      analysis.driftBand);

    analysis.featureImportance.assign(featureIdx.size(), 0.0);

    double totalImportance = 0.0;
    double categoricalImportance = 0.0;
    for (size_t i = 0; i < rawImportance.size(); ++i) {
        const double imp = std::max(0.0, std::isfinite(rawImportance[i]) ? rawImportance[i] : 0.0);
        totalImportance += imp;

        if (i >= encoded.sourceNumericFeaturePos.size()) {
            categoricalImportance += imp;
            continue;
        }

        const int numericPos = encoded.sourceNumericFeaturePos[i];
        if (numericPos >= 0) {
            const size_t numericPosU = static_cast<size_t>(numericPos);
            if (numericPosU < analysis.featureImportance.size()) {
                analysis.featureImportance[numericPosU] += imp;
            }
        } else {
            categoricalImportance += imp;
        }
    }
    analysis.categoricalImportanceShare = (totalImportance > config.tuning.numericEpsilon)
        ? (categoricalImportance / totalImportance)
        : 0.0;

    auto isStrictMonotonicIndexLike = [&](const std::vector<double>& values) {
        if (values.size() < 6) return false;

        size_t finiteCount = 0;
        bool increasing = true;
        bool decreasing = true;
        std::vector<double> diffs;
        diffs.reserve(values.size() - 1);

        for (size_t i = 1; i < values.size(); ++i) {
            const double prev = values[i - 1];
            const double cur = values[i];
            if (!std::isfinite(prev) || !std::isfinite(cur)) return false;
            ++finiteCount;

            const double d = cur - prev;
            if (d <= 0.0) increasing = false;
            if (d >= 0.0) decreasing = false;
            diffs.push_back(std::abs(d));
        }

        if (finiteCount + 1 != values.size()) return false;
        if (!increasing && !decreasing) return false;

        const double meanStep = std::accumulate(diffs.begin(), diffs.end(), 0.0) / std::max<size_t>(1, diffs.size());
        if (meanStep <= 1e-12) return false;

        double stepVar = 0.0;
        for (double d : diffs) {
            const double dd = d - meanStep;
            stepVar += dd * dd;
        }
        stepVar /= std::max<size_t>(1, diffs.size() - 1);
        const double stepStd = std::sqrt(stepVar);

        const double relStepStd = stepStd / meanStep;
        return relStepStd <= 0.10;
    };

    bool reweightApplied = false;
    for (size_t featurePos = 0; featurePos < featureIdx.size() && featurePos < analysis.featureImportance.size(); ++featurePos) {
        const int colIdx = featureIdx[featurePos];
        if (colIdx < 0 || static_cast<size_t>(colIdx) >= data.columns().size()) continue;
        if (data.columns()[static_cast<size_t>(colIdx)].type != ColumnType::NUMERIC) continue;

        const auto& vals = std::get<std::vector<double>>(data.columns()[static_cast<size_t>(colIdx)].values);
        if (!isStrictMonotonicIndexLike(vals)) continue;

        analysis.featureImportance[featurePos] *= 0.10;
        reweightApplied = true;
    }

    if (reweightApplied) {
        const double adjustedSum = std::accumulate(analysis.featureImportance.begin(), analysis.featureImportance.end(), 0.0);
        if (adjustedSum > config.tuning.numericEpsilon) {
            for (double& imp : analysis.featureImportance) {
                imp /= adjustedSum;
            }
        }
    }

    return analysis;
}

BivariateScoringPolicy chooseBivariatePolicy(const AutoConfig& config, const NeuralAnalysis& neural) {
    const std::unordered_map<std::string, BivariateScoringPolicy> registry = {
        {std::string(StrategyKeys::kBalanced), {std::string(StrategyKeys::kBalanced), 0.50, 0.35, 0.15, 0.60, 1.0, 0.60, 0.25}},
        {std::string(StrategyKeys::kCorrHeavy), {std::string(StrategyKeys::kCorrHeavy), 0.30, 0.55, 0.15, 0.55, 1.0, 0.55, 0.20}},
        {std::string(StrategyKeys::kImportanceHeavy), {std::string(StrategyKeys::kImportanceHeavy), 0.65, 0.20, 0.15, 0.65, 1.0, 0.70, 0.35}},
    };

    std::string mode = CommonUtils::toLower(config.bivariateStrategy);
    auto applySelectionQuantileOverride = [&](BivariateScoringPolicy policy) {
        if (config.tuning.bivariateSelectionQuantileOverride >= 0.0 &&
            config.tuning.bivariateSelectionQuantileOverride <= 1.0) {
            policy.selectionQuantile = config.tuning.bivariateSelectionQuantileOverride;
        }
        return policy;
    };
    if (mode != StrategyKeys::kAuto) {
        auto it = registry.find(mode);
        if (it != registry.end()) return applySelectionQuantileOverride(it->second);
        return applySelectionQuantileOverride(registry.at(std::string(StrategyKeys::kBalanced)));
    }

    double maxImp = 0.0;
    double sumImpSq = 0.0;
    for (double v : neural.featureImportance) {
        maxImp = std::max(maxImp, v);
        sumImpSq += v * v;
    }
    double concentration = std::sqrt(sumImpSq);

    if (maxImp > config.tuning.corrHeavyMaxImportanceThreshold ||
        concentration > config.tuning.corrHeavyConcentrationThreshold) {
        return applySelectionQuantileOverride(registry.at(std::string(StrategyKeys::kCorrHeavy)));
    }
    if (maxImp < config.tuning.importanceHeavyMaxImportanceThreshold &&
        concentration < config.tuning.importanceHeavyConcentrationThreshold) {
        return applySelectionQuantileOverride(registry.at(std::string(StrategyKeys::kImportanceHeavy)));
    }
    return applySelectionQuantileOverride(registry.at(std::string(StrategyKeys::kBalanced)));
}

std::unordered_set<size_t> computeNeuralApprovedNumericFeatures(const TypedDataset& data,
                                                                int targetIdx,
                                                                const std::vector<int>& featureIdx,
                                                                const NeuralAnalysis& neural,
                                                                const NumericStatsCache& statsCache) {
    std::unordered_set<size_t> approved;
    if (featureIdx.empty() || neural.featureImportance.empty()) return approved;

    std::vector<double> absCorr = computeFeatureTargetAbsCorr(data, targetIdx, featureIdx, statsCache);

    std::vector<double> positiveImportance;
    positiveImportance.reserve(neural.featureImportance.size());
    for (double imp : neural.featureImportance) {
        if (std::isfinite(imp) && imp > 0.0) positiveImportance.push_back(imp);
    }

    double minImportance = 0.01;
    if (!positiveImportance.empty()) {
        std::sort(positiveImportance.begin(), positiveImportance.end());
        const size_t pos = static_cast<size_t>(std::floor(0.60 * static_cast<double>(positiveImportance.size() - 1)));
        minImportance = std::max(0.01, positiveImportance[pos]);
    }

    const size_t n = data.rowCount();
    for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size() && i < absCorr.size(); ++i) {
        const int col = featureIdx[i];
        if (col < 0) continue;

        const double imp = neural.featureImportance[i];
        const double corr = absCorr[i];
        const bool statSig = MathUtils::calculateSignificance(corr, n).is_significant;
        if (std::isfinite(imp) && imp >= minImportance && statSig) {
            approved.insert(static_cast<size_t>(col));
        }
    }

    if (approved.empty()) {
        double bestImp = -1.0;
        size_t bestIdx = 0;
        for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size(); ++i) {
            if (featureIdx[i] < 0) continue;
            if (neural.featureImportance[i] > bestImp) {
                bestImp = neural.featureImportance[i];
                bestIdx = static_cast<size_t>(featureIdx[i]);
            }
        }
        if (bestImp >= 0.0) approved.insert(bestIdx);
    }

    return approved;
}

struct StructuralRelation {
    size_t a = 0;
    size_t b = 0;
    size_t c = 0;
    std::string op;
};

std::vector<StructuralRelation> detectStructuralRelations(const TypedDataset& data,
                                                          const std::vector<size_t>& numericIdx,
                                                          size_t capColumns = 18,
                                                          size_t capRows = 512) {
    std::vector<StructuralRelation> out;
    if (numericIdx.size() < 3) return out;

    std::vector<size_t> cols = numericIdx;
    if (cols.size() > capColumns) cols.resize(capColumns);

    std::unordered_set<uint64_t> correlatedPairs;
    correlatedPairs.reserve(cols.size() * cols.size());
    for (size_t ia = 0; ia < cols.size(); ++ia) {
        const auto& av = std::get<std::vector<double>>(data.columns()[cols[ia]].values);
        const ColumnStats sa = Statistics::calculateStats(av);
        for (size_t ib = ia + 1; ib < cols.size(); ++ib) {
            const auto& bv = std::get<std::vector<double>>(data.columns()[cols[ib]].values);
            const ColumnStats sb = Statistics::calculateStats(bv);
            const double ar = std::abs(MathUtils::calculatePearson(av, bv, sa, sb).value_or(0.0));
            if (ar < 0.20) continue;
            const uint64_t key = (static_cast<uint64_t>(std::min(ia, ib)) << 32) | static_cast<uint64_t>(std::max(ia, ib));
            correlatedPairs.insert(key);
        }
    }

    for (size_t ia = 0; ia < cols.size(); ++ia) {
        const auto& av = std::get<std::vector<double>>(data.columns()[cols[ia]].values);
        for (size_t ib = ia + 1; ib < cols.size(); ++ib) {
            const uint64_t pairKey = (static_cast<uint64_t>(ia) << 32) | static_cast<uint64_t>(ib);
            if (correlatedPairs.find(pairKey) == correlatedPairs.end()) continue;
            const auto& bv = std::get<std::vector<double>>(data.columns()[cols[ib]].values);
            for (size_t ic = 0; ic < cols.size(); ++ic) {
                if (ic == ia || ic == ib) continue;
                const auto& cv = std::get<std::vector<double>>(data.columns()[cols[ic]].values);
                const size_t n = std::min({av.size(), bv.size(), cv.size(), data.rowCount(), capRows});
                if (n < 24) continue;

                size_t valid = 0;
                size_t addHits = 0;
                size_t mulHits = 0;
                for (size_t r = 0; r < n; ++r) {
                    if (data.columns()[cols[ia]].missing[r] || data.columns()[cols[ib]].missing[r] || data.columns()[cols[ic]].missing[r]) {
                        continue;
                    }
                    const double a = av[r];
                    const double b = bv[r];
                    const double c = cv[r];
                    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) continue;
                    const double addErr = std::abs((a + b) - c);
                    const double addScale = std::max(1.0, std::abs(c));
                    if (addErr <= (1e-5 * addScale + 1e-8)) addHits++;

                    const double mulErr = std::abs((a * b) - c);
                    const double mulScale = std::max(1.0, std::abs(c));
                    if (mulErr <= (1e-4 * mulScale + 1e-8)) mulHits++;
                    ++valid;
                }
                if (valid < 24) continue;

                const double addRate = static_cast<double>(addHits) / static_cast<double>(valid);
                const double mulRate = static_cast<double>(mulHits) / static_cast<double>(valid);
                if (addRate >= 0.985) {
                    out.push_back({cols[ia], cols[ib], cols[ic], "sum"});
                } else if (mulRate >= 0.985) {
                    out.push_back({cols[ia], cols[ib], cols[ic], "product"});
                }
            }
        }
    }

    return out;
}

std::unordered_map<size_t, size_t> buildInformationClusters(const TypedDataset& data,
                                                            const std::vector<size_t>& numericIdx,
                                                            const NumericStatsCache& statsCache,
                                                            const std::unordered_map<size_t, double>& importanceByIndex,
                                                            double absCorrThreshold = 0.97) {
    std::unordered_map<size_t, size_t> representativeByFeature;
    if (numericIdx.empty()) return representativeByFeature;

    std::unordered_map<size_t, size_t> parent;
    for (size_t idx : numericIdx) parent[idx] = idx;

    std::function<size_t(size_t)> findRoot = [&](size_t x) -> size_t {
        auto it = parent.find(x);
        if (it == parent.end()) return x;
        if (it->second == x) return x;
        it->second = findRoot(it->second);
        return it->second;
    };
    auto unite = [&](size_t a, size_t b) {
        const size_t ra = findRoot(a);
        const size_t rb = findRoot(b);
        if (ra == rb) return;
        parent[rb] = ra;
    };

    for (size_t i = 0; i < numericIdx.size(); ++i) {
        for (size_t j = i + 1; j < numericIdx.size(); ++j) {
            const size_t ia = numericIdx[i];
            const size_t ib = numericIdx[j];
            const auto& va = std::get<std::vector<double>>(data.columns()[ia].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[ib].values);
            auto aIt = statsCache.find(ia);
            auto bIt = statsCache.find(ib);
            std::optional<ColumnStats> aFallback;
            std::optional<ColumnStats> bFallback;
            const ColumnStats* sa = nullptr;
            const ColumnStats* sb = nullptr;
            if (aIt != statsCache.end()) sa = &aIt->second;
            else {
                aFallback = Statistics::calculateStats(va);
                sa = &(*aFallback);
            }
            if (bIt != statsCache.end()) sb = &bIt->second;
            else {
                bFallback = Statistics::calculateStats(vb);
                sb = &(*bFallback);
            }
            if (sa == nullptr || sb == nullptr) continue;
            const double r = std::abs(MathUtils::calculatePearson(va, vb, *sa, *sb).value_or(0.0));
            if (r >= absCorrThreshold) unite(ia, ib);
        }
    }

    std::unordered_map<size_t, std::vector<size_t>> members;
    for (size_t idx : numericIdx) members[findRoot(idx)].push_back(idx);

    for (const auto& kv : members) {
        const auto& cluster = kv.second;
        size_t rep = cluster.front();
        double best = -1.0;
        for (size_t idx : cluster) {
            const double s = importanceByIndex.count(idx) ? importanceByIndex.at(idx) : 0.0;
            if (s > best) {
                best = s;
                rep = idx;
            }
        }
        for (size_t idx : cluster) representativeByFeature[idx] = rep;
    }
    return representativeByFeature;
}

double computeFoldStabilityFromCorrelation(const std::vector<double>& a,
                                           const std::vector<double>& b,
                                           size_t folds = 5) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 30 || folds < 2) return 0.0;
    const size_t foldCount = std::min(folds, n / 6);
    if (foldCount < 2) return 0.0;

    std::vector<double> foldScores;
    foldScores.reserve(foldCount);
    for (size_t f = 0; f < foldCount; ++f) {
        const size_t s = (f * n) / foldCount;
        const size_t e = ((f + 1) * n) / foldCount;
        std::vector<double> xa;
        std::vector<double> xb;
        xa.reserve(e - s);
        xb.reserve(e - s);
        for (size_t i = s; i < e; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) continue;
            xa.push_back(a[i]);
            xb.push_back(b[i]);
        }
        if (xa.size() < 8) continue;
        const ColumnStats sa = Statistics::calculateStats(xa);
        const ColumnStats sb = Statistics::calculateStats(xb);
        const double r = std::abs(MathUtils::calculatePearson(xa, xb, sa, sb).value_or(0.0));
        if (std::isfinite(r)) foldScores.push_back(r);
    }
    if (foldScores.size() < 2) return 0.0;

    const double mean = std::accumulate(foldScores.begin(), foldScores.end(), 0.0) / static_cast<double>(foldScores.size());
    if (mean <= 1e-12) return 0.0;
    double var = 0.0;
    for (double v : foldScores) {
        const double d = v - mean;
        var += d * d;
    }
    var /= static_cast<double>(std::max<size_t>(1, foldScores.size() - 1));
    const double cv = std::sqrt(var) / mean;
    return std::clamp(1.0 - cv, 0.0, 1.0);
}

std::vector<PairInsight> analyzeBivariatePairs(const TypedDataset& data,
                                               const std::unordered_map<size_t, double>& importanceByIndex,
                                               const std::unordered_set<size_t>& modeledIndices,
                                               const BivariateScoringPolicy& policy,
                                               GnuplotEngine* plotter,
                                               bool verbose,
                                               const NumericStatsCache& statsCache,
                                               double numericEpsilon,
                                               const HeuristicTuningConfig& tuning,
                                               size_t maxPairs,
                                               size_t maxSelectedPairs) {
    std::vector<PairInsight> pairs;
    const auto numericIdx = data.numericColumnIndices();
    const size_t n = data.rowCount();
    constexpr double kHardMissingnessPrune = 0.90;
    constexpr size_t kHardMinNonMissing = 24;
    std::vector<size_t> evalNumericIdx;
    evalNumericIdx.reserve(numericIdx.size());
    for (size_t idx : numericIdx) {
        const auto it = statsCache.find(idx);
        if (it != statsCache.end()) {
            const double var = std::isfinite(it->second.variance) ? it->second.variance : 0.0;
            if (var <= numericEpsilon) continue;
        }
        const auto& col = data.columns()[idx];
        const size_t miss = std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1));
        const double missRatio = col.missing.empty() ? 0.0 : static_cast<double>(miss) / static_cast<double>(col.missing.size());
        const size_t nonMissing = col.missing.size() > miss ? (col.missing.size() - miss) : 0;
        if (missRatio > kHardMissingnessPrune || nonMissing < kHardMinNonMissing) continue;
        evalNumericIdx.push_back(idx);
    }
    if (evalNumericIdx.size() < 2) return pairs;

    // Two-stage pruning for very wide datasets: keep highest-utility columns before O(n^2) pair expansion.
    if (maxPairs == 0 && evalNumericIdx.size() > 512) {
        std::vector<std::pair<size_t, double>> ranked;
        ranked.reserve(evalNumericIdx.size());
        for (size_t idx : evalNumericIdx) {
            double imp = 0.0;
            if (const auto it = importanceByIndex.find(idx); it != importanceByIndex.end()) imp = it->second;
            const auto& col = data.columns()[idx];
            const size_t miss = std::count(col.missing.begin(), col.missing.end(), static_cast<uint8_t>(1));
            const double missRatio = col.missing.empty() ? 0.0 : static_cast<double>(miss) / static_cast<double>(col.missing.size());
            const double modeledBoost = (modeledIndices.find(idx) != modeledIndices.end()) ? 0.20 : 0.0;
            const double quality = std::clamp(1.0 - missRatio, 0.0, 1.0);
            ranked.push_back({idx, std::max(0.0, imp) * 0.70 + quality * 0.20 + modeledBoost * 0.10});
        }
        std::sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
            if (a.second == b.second) return data.columns()[a.first].name < data.columns()[b.first].name;
            return a.second > b.second;
        });

        evalNumericIdx.clear();
        evalNumericIdx.reserve(512);
        for (size_t i = 0; i < 512 && i < ranked.size(); ++i) {
            evalNumericIdx.push_back(ranked[i].first);
        }
    }

    const auto representativeByFeature = buildInformationClusters(data, evalNumericIdx, statsCache, importanceByIndex);
    const auto structuralRelations = detectStructuralRelations(data, evalNumericIdx);
    std::unordered_map<std::string, std::string> structuralPairLabel;
    for (const auto& sr : structuralRelations) {
        const size_t x = std::min(sr.a, sr.b);
        const size_t y = std::max(sr.a, sr.b);
        const std::string key = std::to_string(x) + "|" + std::to_string(y);
        const std::string label = (sr.op == "sum")
            ? (data.columns()[sr.a].name + " + " + data.columns()[sr.b].name + " ≈ " + data.columns()[sr.c].name)
            : (data.columns()[sr.a].name + " × " + data.columns()[sr.b].name + " ≈ " + data.columns()[sr.c].name);
        structuralPairLabel[key] = label;
    }

    const size_t totalPairs = evalNumericIdx.size() < 2 ? 0 : (evalNumericIdx.size() * (evalNumericIdx.size() - 1)) / 2;
    if (maxPairs > 0 && totalPairs > maxPairs) {
        std::vector<std::pair<size_t, double>> ranked;
        ranked.reserve(evalNumericIdx.size());
        for (size_t idx : evalNumericIdx) {
            double imp = 0.0;
            auto it = importanceByIndex.find(idx);
            if (it != importanceByIndex.end()) imp = it->second;
            double modeledBoost = (modeledIndices.find(idx) != modeledIndices.end()) ? 0.25 : 0.0;
            ranked.push_back({idx, imp + modeledBoost});
        }
        std::sort(ranked.begin(), ranked.end(), [&](const auto& a, const auto& b) {
            if (a.second == b.second) return data.columns()[a.first].name < data.columns()[b.first].name;
            return a.second > b.second;
        });

        const size_t keepColumns = std::clamp<size_t>(
            static_cast<size_t>(std::floor((1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(maxPairs))) / 2.0)),
            2,
            ranked.size());
        evalNumericIdx.clear();
        evalNumericIdx.reserve(keepColumns);
        for (size_t i = 0; i < keepColumns; ++i) {
            evalNumericIdx.push_back(ranked[i].first);
        }
    }

    if (verbose) {
        size_t activePairs = evalNumericIdx.size() < 2 ? 0 : (evalNumericIdx.size() * (evalNumericIdx.size() - 1)) / 2;
        std::cout << "[Seldon][Bivariate] Evaluating " << activePairs << " pair combinations with policy '" << policy.name << "'";
        if (activePairs < totalPairs) {
            std::cout << " (fast cap from " << totalPairs << ")";
        }
        std::cout << "...\n";
    }

    const size_t evalCount = evalNumericIdx.size();
    auto pairSlot = [evalCount](size_t i, size_t j) {
        return i * evalCount + j;
    };

    std::vector<ColumnStats> evalStats(evalCount);
    std::vector<std::vector<double>> evalRanks(evalCount);
    std::vector<ColumnStats> evalRankStats(evalCount);

    auto computeAverageRanks = [](const std::vector<double>& values) {
        std::vector<double> ranks(values.size(), 0.0);
        if (values.empty()) return ranks;

        std::vector<size_t> order(values.size(), 0);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (values[a] == values[b]) return a < b;
            return values[a] < values[b];
        });

        size_t i = 0;
        while (i < order.size()) {
            size_t j = i + 1;
            while (j < order.size() && values[order[j]] == values[order[i]]) ++j;
            const double rank = (static_cast<double>(i) + static_cast<double>(j - 1)) * 0.5 + 1.0;
            for (size_t k = i; k < j; ++k) ranks[order[k]] = rank;
            i = j;
        }
        return ranks;
    };

    for (size_t pos = 0; pos < evalCount; ++pos) {
        const size_t colIdx = evalNumericIdx[pos];
        const auto cacheIt = statsCache.find(colIdx);
        if (cacheIt != statsCache.end()) {
            evalStats[pos] = cacheIt->second;
        } else {
            const auto& colVals = std::get<std::vector<double>>(data.columns()[colIdx].values);
            evalStats[pos] = Statistics::calculateStats(colVals);
        }

        const auto& colVals = std::get<std::vector<double>>(data.columns()[colIdx].values);
        evalRanks[pos] = computeAverageRanks(colVals);
        evalRankStats[pos] = Statistics::calculateStats(evalRanks[pos]);
    }

    std::vector<float> pearsonMatrix(evalCount * evalCount, 0.0f);
    std::vector<float> spearmanMatrix(evalCount * evalCount, 0.0f);
    if (n > 1 && evalCount > 0) {
        MathUtils::Matrix standardized(n, evalCount);
        MathUtils::Matrix rankStandardized(n, evalCount);

        for (size_t c = 0; c < evalCount; ++c) {
            const auto& values = std::get<std::vector<double>>(data.columns()[evalNumericIdx[c]].values);
            const double mean = evalStats[c].mean;
            const double sd = std::abs(evalStats[c].stddev) > numericEpsilon ? evalStats[c].stddev : 1.0;

            const double rankMean = evalRankStats[c].mean;
            const double rankSd = std::abs(evalRankStats[c].stddev) > numericEpsilon ? evalRankStats[c].stddev : 1.0;

            for (size_t r = 0; r < n; ++r) {
                const double v = (r < values.size() && std::isfinite(values[r])) ? values[r] : mean;
                const double rv = (r < evalRanks[c].size() && std::isfinite(evalRanks[c][r])) ? evalRanks[c][r] : rankMean;
                standardized.at(r, c) = (v - mean) / sd;
                rankStandardized.at(r, c) = (rv - rankMean) / rankSd;
            }
        }

        const MathUtils::Matrix pearsonCov = standardized.transpose().multiply(standardized);
        const MathUtils::Matrix spearmanCov = rankStandardized.transpose().multiply(rankStandardized);
        const double denom = static_cast<double>(n - 1);
        for (size_t i = 0; i < evalCount; ++i) {
            for (size_t j = 0; j < evalCount; ++j) {
                pearsonMatrix[pairSlot(i, j)] = static_cast<float>(pearsonCov.at(i, j) / denom);
                spearmanMatrix[pairSlot(i, j)] = static_cast<float>(spearmanCov.at(i, j) / denom);
            }
        }
    } else {
        for (size_t i = 0; i < evalCount; ++i) {
            pearsonMatrix[pairSlot(i, i)] = 1.0f;
            spearmanMatrix[pairSlot(i, i)] = 1.0f;
        }
    }

    auto buildPair = [&](size_t aPos, size_t bPos) {
            size_t ia = evalNumericIdx[aPos];
            size_t ib = evalNumericIdx[bPos];
            const auto& va = std::get<std::vector<double>>(data.columns()[ia].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[ib].values);

            PairInsight p;
            p.idxA = ia;
            p.idxB = ib;
            p.featureA = data.columns()[ia].name;
            p.featureB = data.columns()[ib].name;
            p.r = static_cast<double>(pearsonMatrix[pairSlot(aPos, bPos)]);
            p.spearman = 0.0;
            p.kendallTau = 0.0;
            p.r2 = p.r * p.r;
            // Keep regression parameter derivation centralized in MathUtils.
            auto fit = MathUtils::simpleLinearRegression(va, vb, evalStats[aPos], evalStats[bPos], p.r);
            p.slope = fit.first;
            p.intercept = fit.second;

            auto sig = MathUtils::calculateSignificance(p.r, n);
            p.pValue = sig.p_value;
            p.tStat = sig.t_stat;
            p.statSignificant = sig.is_significant;

            const double absR = std::abs(p.r);
            if (p.statSignificant || absR >= 0.20) {
                p.spearman = static_cast<double>(spearmanMatrix[pairSlot(aPos, bPos)]);
                if (va.size() <= 256) {
                    p.kendallTau = MathUtils::calculateKendallTau(va, vb).value_or(0.0);
                } else {
                    const double rho = std::clamp(p.spearman, -1.0, 1.0);
                    p.kendallTau = (2.0 / 3.14159265358979323846) * std::asin(rho);
                }
            }

            const bool monotonicIdentity = (std::abs(p.spearman) >= 0.999 && std::abs(p.r) < 0.9985);
            const bool nearIdentity = std::abs(p.r) >= 0.9995;
            if (nearIdentity || monotonicIdentity) {
                p.filteredAsRedundant = true;
                p.relationLabel = nearIdentity ? "Near-identical signal" : "Monotonic transform proxy";
            }

            if (!p.filteredAsRedundant && isEngineeredLineagePair(p.featureA, p.featureB) && std::abs(p.r) >= 0.98) {
                p.filteredAsRedundant = true;
                p.relationLabel = "Engineered lineage duplicate";
                p.redundancyGroup = canonicalEngineeredBaseName(p.featureA);
            }

            const size_t repA = representativeByFeature.count(ia) ? representativeByFeature.at(ia) : ia;
            const size_t repB = representativeByFeature.count(ib) ? representativeByFeature.at(ib) : ib;
            if (repA == repB) {
                p.filteredAsRedundant = true;
                p.redundancyGroup = data.columns()[repA].name;
                if (p.relationLabel.empty()) p.relationLabel = "Information cluster duplicate";
            }

            const size_t kx = std::min(ia, ib);
            const size_t ky = std::max(ia, ib);
            const std::string structuralKey = std::to_string(kx) + "|" + std::to_string(ky);
            auto sit = structuralPairLabel.find(structuralKey);
            if (sit != structuralPairLabel.end()) {
                p.filteredAsStructural = true;
                p.relationLabel = sit->second;
            }

            double impA = 0.0;
            double impB = 0.0;
            auto ita = importanceByIndex.find(ia);
            auto itb = importanceByIndex.find(ib);
            if (ita != importanceByIndex.end()) impA = ita->second;
            if (itb != importanceByIndex.end()) impB = itb->second;
            double impScore = std::clamp((impA + impB) / 2.0, 0.0, 1.0);
            p.effectSize = std::clamp(p.r2, 0.0, 1.0);
            p.foldStability = (p.statSignificant || absR >= 0.25)
                ? computeFoldStabilityFromCorrelation(va, vb, 5)
                : 0.0;
            bool aModeled = modeledIndices.find(ia) != modeledIndices.end();
            bool bModeled = modeledIndices.find(ib) != modeledIndices.end();
            double coverageFactor = (aModeled && bModeled) ? policy.coverageBoth : ((aModeled || bModeled) ? policy.coverageOne : policy.coverageNone);
            const double effectWeighted = 0.60 * p.effectSize + 0.40 * std::clamp(std::abs(p.r), 0.0, 1.0);
            p.neuralScore = (policy.wImportance * impScore + policy.wCorrelation * effectWeighted + policy.wSignificance * p.foldStability) * coverageFactor;

            const bool leakageNameHint =
                containsToken(CommonUtils::toLower(p.featureA), {"future", "post", "outcome", "label", "resolved", "actual"}) ||
                containsToken(CommonUtils::toLower(p.featureB), {"future", "post", "outcome", "label", "resolved", "actual"});
            p.leakageRisk = (std::abs(p.r) >= 0.995 && (leakageNameHint || p.filteredAsStructural));
            if (p.leakageRisk && p.relationLabel.empty()) {
                p.relationLabel = "Potential leakage proxy";
            }

            if (p.relationLabel.empty()) {
                const double ar = std::abs(p.r);
                const std::string strength =
                    (ar >= 0.85) ? "Very strong" :
                    (ar >= 0.65) ? "Strong" :
                    (ar >= 0.45) ? "Moderate" :
                    (ar >= 0.25) ? "Weak" : "Very weak";
                const std::string sign = (p.r >= 0.0) ? "positive" : "negative";
                p.relationLabel = strength + " " + sign + " association";
            }

            return p;
    };

    auto appendPairsForRange = [&](size_t startAPos, size_t endAPos, std::vector<PairInsight>& out, bool emitVerboseRows) {
        for (size_t aPos = startAPos; aPos < endAPos; ++aPos) {
            for (size_t bPos = aPos + 1; bPos < evalNumericIdx.size(); ++bPos) {
                PairInsight p = buildPair(aPos, bPos);
                if (emitVerboseRows) {
                    std::cout << "[Seldon][Bivariate] " << p.featureA << " vs " << p.featureB
                              << " | r=" << toFixed(p.r, 6)
                              << " p=" << toFixed(p.pValue, 8)
                              << " t=" << toFixed(p.tStat, 6)
                              << " neural=" << toFixed(p.neuralScore, 8) << "\n";
                }
                out.push_back(std::move(p));
            }
        }
    };

    if (verbose) {
        const size_t reservePairs = evalNumericIdx.size() < 2 ? 0 : (evalNumericIdx.size() * (evalNumericIdx.size() - 1)) / 2;
        pairs.reserve(reservePairs);
        for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
            appendPairsForRange(aPos, aPos + 1, pairs, true);
            printProgressBar("pair-generation", aPos + 1, evalNumericIdx.size());
        }
    } else {
        #ifdef USE_OPENMP
        const int threadCount = std::max(1, omp_get_max_threads());
        std::vector<std::vector<PairInsight>> threadPairs(static_cast<size_t>(threadCount));
        const size_t reservePairs = evalNumericIdx.size() < 2 ? 0 : (evalNumericIdx.size() * (evalNumericIdx.size() - 1)) / 2;
        const size_t reservePerThread = reservePairs / static_cast<size_t>(threadCount) + 16;
        for (auto& local : threadPairs) local.reserve(reservePerThread);

        #pragma omp parallel default(none) shared(threadPairs, evalNumericIdx, appendPairsForRange)
        {
            const int tid = omp_get_thread_num();
            std::vector<PairInsight>& localPairs = threadPairs[static_cast<size_t>(tid)];

            #pragma omp for schedule(dynamic)
            for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
                appendPairsForRange(aPos, aPos + 1, localPairs, false);
            }
        }

        for (auto& local : threadPairs) {
            pairs.insert(pairs.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
        }
        printProgressBar("pair-generation", evalNumericIdx.size(), evalNumericIdx.size());
        #else
        for (size_t aPos = 0; aPos < evalNumericIdx.size(); ++aPos) {
            appendPairsForRange(aPos, aPos + 1, pairs, false);
            if (((aPos + 1) % std::max<size_t>(1, evalNumericIdx.size() / 20)) == 0 || (aPos + 1) == evalNumericIdx.size()) {
                printProgressBar("pair-generation", aPos + 1, evalNumericIdx.size());
            }
        }
        #endif
    }

    std::vector<double> significantScores;
    for (const auto& p : pairs) {
        if (p.statSignificant) significantScores.push_back(p.neuralScore);
    }

    // Dynamic cutoff couples statistical significance with neural relevance.
    double dynamicCutoff = 0.0;
    if (!significantScores.empty()) {
        std::sort(significantScores.begin(), significantScores.end());
        size_t pos = static_cast<size_t>(std::floor(policy.selectionQuantile * static_cast<double>(significantScores.size() - 1)));
        dynamicCutoff = significantScores[pos];
    }

    std::vector<size_t> selectedCandidateIdx;
    selectedCandidateIdx.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        if (pairs[i].filteredAsRedundant || pairs[i].filteredAsStructural || pairs[i].leakageRisk) continue;
        if (pairs[i].statSignificant && pairs[i].neuralScore >= dynamicCutoff) {
            selectedCandidateIdx.push_back(i);
        }
    }

    std::sort(selectedCandidateIdx.begin(), selectedCandidateIdx.end(), [&](size_t lhs, size_t rhs) {
        if (pairs[lhs].neuralScore == pairs[rhs].neuralScore) {
            return pairs[lhs].effectSize > pairs[rhs].effectSize;
        }
        return pairs[lhs].neuralScore > pairs[rhs].neuralScore;
    });

    std::unordered_set<size_t> finalSelected;
    const size_t keepCap = std::max<size_t>(1, maxSelectedPairs);
    const size_t keepCount = std::min(selectedCandidateIdx.size(), keepCap);
    for (size_t i = 0; i < keepCount; ++i) {
        finalSelected.insert(selectedCandidateIdx[i]);
    }

    std::vector<size_t> statOnlyIdx;
    statOnlyIdx.reserve(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        const auto& p = pairs[i];
        if (p.leakageRisk) continue;
        if (!p.statSignificant || finalSelected.count(i) > 0) continue;
        statOnlyIdx.push_back(i);
    }

    std::sort(statOnlyIdx.begin(), statOnlyIdx.end(), [&](size_t lhs, size_t rhs) {
        const auto& a = pairs[lhs];
        const auto& b = pairs[rhs];
        const double pa = std::clamp(-std::log10(std::max(1e-12, a.pValue)) / 12.0, 0.0, 1.0);
        const double pb = std::clamp(-std::log10(std::max(1e-12, b.pValue)) / 12.0, 0.0, 1.0);
        const double sa = 0.45 * a.effectSize + 0.25 * a.foldStability + 0.20 * pa + 0.10 * std::abs(a.r);
        const double sb = 0.45 * b.effectSize + 0.25 * b.foldStability + 0.20 * pb + 0.10 * std::abs(b.r);
        if (sa == sb) {
            if (a.pValue == b.pValue) return a.neuralScore > b.neuralScore;
            return a.pValue < b.pValue;
        }
        return sa > sb;
    });

    const double tier3Aggressiveness = std::clamp(tuning.bivariateTier3FallbackAggressiveness, 0.0, 3.0);
    size_t minSelectedFloor = 0;
    if (tier3Aggressiveness > 0.0) {
        const size_t baseFloor = std::max<size_t>(3, std::min<size_t>(12, pairs.size() / 25 + 1));
        const size_t scaledFloor = static_cast<size_t>(std::llround(static_cast<double>(baseFloor) * tier3Aggressiveness));
        minSelectedFloor = std::min(keepCap, std::max<size_t>(1, scaledFloor));
    }
    if (finalSelected.size() < minSelectedFloor) {
        const size_t deficit = minSelectedFloor - finalSelected.size();
        const size_t promote = std::min(deficit, statOnlyIdx.size());
        for (size_t i = 0; i < promote; ++i) {
            finalSelected.insert(statOnlyIdx[i]);
        }
    }

    for (size_t pairIdx = 0; pairIdx < pairs.size(); ++pairIdx) {
        auto& p = pairs[pairIdx];
        p.selected = finalSelected.find(pairIdx) != finalSelected.end();
        if (!p.leakageRisk && p.statSignificant) {
            p.significanceTier = 1;
            p.selectionReason = "statistical";
            if (p.neuralScore >= dynamicCutoff) {
                p.significanceTier = 2;
                p.selectionReason = "statistical+neural";
            }
            if (p.selected && p.significanceTier < 2) {
                p.significanceTier = 3;
                if (p.filteredAsRedundant || p.filteredAsStructural) {
                    p.selectionReason = "statistical+identity_validation";
                } else {
                    p.selectionReason = "statistical+domain_fallback";
                }
            }
        }
        if (p.selected && plotter) {
            const auto& va = std::get<std::vector<double>>(data.columns()[p.idxA].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[p.idxB].values);
            const size_t sampleSize = std::min(va.size(), vb.size());
            std::string id = "sig_" + p.featureA + "_" + p.featureB;
            const bool addFit = shouldOverlayFittedLine(p.r,
                                                        p.statSignificant,
                                                        va,
                                                        vb,
                                                        p.slope,
                                                        p.intercept,
                                                        tuning);
            const bool addBand = shouldAddConfidenceBand(p.r,
                                                         p.statSignificant,
                                                         sampleSize,
                                                         tuning);
            p.fitLineAdded = addFit;
            p.confidenceBandAdded = addBand;
            p.plotPath = plotter->scatter(id,
                                          va,
                                          vb,
                                          p.featureA + " vs " + p.featureB,
                                          addFit,
                                          p.slope,
                                          p.intercept,
                                          "Fitted line",
                                          addBand,
                                          1.96,
                                          tuning.scatterDownsampleThreshold);
            if (p.plotPath.empty()) {
                p.fitLineAdded = false;
                p.confidenceBandAdded = false;
            }

            if (shouldAddResidualPlot(p.r, p.selected, sampleSize, tuning)) {
                std::vector<double> fitted;
                std::vector<double> residuals;
                fitted.reserve(sampleSize);
                residuals.reserve(sampleSize);
                for (size_t i = 0; i < sampleSize; ++i) {
                    if (!std::isfinite(va[i]) || !std::isfinite(vb[i])) continue;
                    const double yHat = p.slope * va[i] + p.intercept;
                    fitted.push_back(yHat);
                    residuals.push_back(vb[i] - yHat);
                }
                p.residualPlotPath = plotter->residual("resid_" + p.featureA + "_" + p.featureB,
                                                       fitted,
                                                       residuals,
                                                       "Residuals: " + p.featureA + " -> " + p.featureB);
            }

            if (auto facetIdx = chooseFacetingColumn(data, p.idxA, p.idxB, tuning); facetIdx.has_value()) {
                const auto& facetVals = std::get<std::vector<std::string>>(data.columns()[*facetIdx].values);
                p.facetedPlotPath = plotter->facetedScatter("facet_" + p.featureA + "_" + p.featureB,
                                                            va,
                                                            vb,
                                                            facetVals,
                                                            p.featureA + " vs " + p.featureB + " by " + data.columns()[*facetIdx].name,
                                                            tuning.facetMaxCategories);
            }

            const BivariateStackedBarData stacked = buildBivariateStackedBar(va, vb);
            if (stacked.valid) {
                p.stackedPlotPath = plotter->stackedBar("sigstack_" + p.featureA + "_" + p.featureB,
                                                        stacked.categories,
                                                        {p.featureB + " <= median", p.featureB + " > median"},
                                                        {stacked.lowCounts, stacked.highCounts},
                                                        "Stacked Profile: " + p.featureA + " x " + p.featureB,
                                                        "Count");
            }
        }
        if (verbose) {
            std::cout << "[Seldon][Bivariate] Decision " << p.featureA << " vs " << p.featureB
                      << " => " << (p.selected ? "SELECTED" : "REJECTED")
                      << " (cutoff=" << toFixed(dynamicCutoff, 8) << ")\n";
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairInsight& lhs, const PairInsight& rhs) {
        if (lhs.selected != rhs.selected) return lhs.selected > rhs.selected;
        if (lhs.effectSize == rhs.effectSize) return std::abs(lhs.r) > std::abs(rhs.r);
        return lhs.effectSize > rhs.effectSize;
    });

    return pairs;
}

struct ContingencyInsight {
    std::string catA;
    std::string catB;
    double chi2 = 0.0;
    double pValue = 1.0;
    double cramerV = 0.0;
    double oddsRatio = 1.0;
    double oddsCiLow = 1.0;
    double oddsCiHigh = 1.0;
};

std::vector<ContingencyInsight> analyzeContingencyPairs(const TypedDataset& data) {
    std::vector<ContingencyInsight> out;
    const auto cats = data.categoricalColumnIndices();
    for (size_t i = 0; i < cats.size(); ++i) {
        for (size_t j = i + 1; j < cats.size(); ++j) {
            const auto& aCol = std::get<std::vector<std::string>>(data.columns()[cats[i]].values);
            const auto& bCol = std::get<std::vector<std::string>>(data.columns()[cats[j]].values);
            const size_t n = std::min(aCol.size(), bCol.size());
            if (n < 20) continue;

            std::map<std::string, size_t> aMap;
            std::map<std::string, size_t> bMap;
            std::vector<std::pair<std::string, std::string>> rows;
            rows.reserve(n);
            for (size_t r = 0; r < n; ++r) {
                if (data.columns()[cats[i]].missing[r] || data.columns()[cats[j]].missing[r]) continue;
                rows.push_back({aCol[r], bCol[r]});
                aMap.emplace(aCol[r], aMap.size());
                bMap.emplace(bCol[r], bMap.size());
            }
            if (rows.size() < 20 || aMap.size() < 2 || bMap.size() < 2) continue;

            const size_t R = aMap.size();
            const size_t C = bMap.size();
            std::vector<std::vector<double>> table(R, std::vector<double>(C, 0.0));
            for (const auto& row : rows) {
                table[aMap[row.first]][bMap[row.second]] += 1.0;
            }

            std::vector<double> rowSum(R, 0.0), colSum(C, 0.0);
            double total = 0.0;
            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    rowSum[r] += table[r][c];
                    colSum[c] += table[r][c];
                    total += table[r][c];
                }
            }
            if (total <= 0.0) continue;

            double chi2 = 0.0;
            for (size_t r = 0; r < R; ++r) {
                for (size_t c = 0; c < C; ++c) {
                    const double expected = (rowSum[r] * colSum[c]) / total;
                    if (expected <= 1e-12) continue;
                    const double d = table[r][c] - expected;
                    chi2 += (d * d) / expected;
                }
            }
            const double p = std::exp(-0.5 * chi2); // simple approximation
            const double v = std::sqrt(std::max(0.0, chi2 / (total * static_cast<double>(std::min(R - 1, C - 1)))));

            double oratio = 1.0, lo = 1.0, hi = 1.0;
            if (R == 2 && C == 2) {
                const double a = table[0][0] + 0.5;
                const double b = table[0][1] + 0.5;
                const double c = table[1][0] + 0.5;
                const double d = table[1][1] + 0.5;
                oratio = (a * d) / (b * c);
                const double se = std::sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d);
                const double l = std::log(oratio);
                lo = std::exp(l - 1.96 * se);
                hi = std::exp(l + 1.96 * se);
            }

            out.push_back({data.columns()[cats[i]].name, data.columns()[cats[j]].name, chi2, p, v, oratio, lo, hi});
            if (out.size() >= 12) return out;
        }
    }
    return out;
}

struct AnovaInsight {
    std::string categorical;
    std::string numeric;
    double fStat = 0.0;
    double pValue = 1.0;
    double eta2 = 0.0;
    std::string tukeySummary;
};

std::vector<AnovaInsight> analyzeAnovaPairs(const TypedDataset& data) {
    std::vector<AnovaInsight> out;
    const auto cats = data.categoricalColumnIndices();
    const auto nums = data.numericColumnIndices();
    for (size_t cidx : cats) {
        const auto& cv = std::get<std::vector<std::string>>(data.columns()[cidx].values);
        for (size_t nidx : nums) {
            const auto& nv = std::get<std::vector<double>>(data.columns()[nidx].values);
            const size_t n = std::min(cv.size(), nv.size());
            if (n < 24) continue;

            std::map<std::string, std::vector<double>> groups;
            for (size_t i = 0; i < n; ++i) {
                if (data.columns()[cidx].missing[i] || data.columns()[nidx].missing[i] || !std::isfinite(nv[i])) continue;
                groups[cv[i]].push_back(nv[i]);
            }
            if (groups.size() < 2 || groups.size() > 10) continue;

            std::vector<double> all;
            for (const auto& kv : groups) all.insert(all.end(), kv.second.begin(), kv.second.end());
            if (all.size() < 24) continue;
            const double grand = std::accumulate(all.begin(), all.end(), 0.0) / static_cast<double>(all.size());

            double ssb = 0.0;
            double ssw = 0.0;
            std::vector<std::pair<std::string, double>> means;
            for (const auto& kv : groups) {
                if (kv.second.size() < 2) continue;
                const double mu = std::accumulate(kv.second.begin(), kv.second.end(), 0.0) / static_cast<double>(kv.second.size());
                means.push_back({kv.first, mu});
                const double dm = mu - grand;
                ssb += static_cast<double>(kv.second.size()) * dm * dm;
                for (double v : kv.second) {
                    const double d = v - mu;
                    ssw += d * d;
                }
            }
            const double dfb = static_cast<double>(groups.size() - 1);
            const double dfw = static_cast<double>(all.size() - groups.size());
            if (dfb <= 0.0 || dfw <= 0.0) continue;
            const double msb = ssb / dfb;
            const double msw = ssw / dfw;
            const double f = (msw <= 1e-12) ? 0.0 : (msb / msw);
            const double p = std::exp(-0.5 * f);
            const double eta2 = (ssb + ssw <= 1e-12) ? 0.0 : (ssb / (ssb + ssw));

            std::sort(means.begin(), means.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string tukey = "n/a";
            if (p < 0.05 && means.size() >= 2) {
                const double delta = std::abs(means[0].second - means[1].second);
                tukey = means[0].first + " vs " + means[1].first + " Δ=" + toFixed(delta, 4);
            }

            out.push_back({data.columns()[cidx].name, data.columns()[nidx].name, f, p, eta2, tukey});
            if (out.size() >= 12) return out;
        }
    }
    return out;
}

struct AdvancedAnalyticsOutputs {
    std::vector<std::vector<std::string>> orderedRows;
    std::vector<std::vector<std::string>> mahalanobisRows;
    std::vector<std::vector<std::string>> pdpRows;
    std::vector<std::vector<std::string>> causalDagRows;
    std::vector<std::vector<std::string>> globalConditionalRows;
    std::vector<std::vector<std::string>> temporalDriftRows;
    std::vector<std::vector<std::string>> contextualDeadZoneRows;
    std::vector<std::string> narrativeRows;
    std::vector<std::string> priorityTakeaways;
    std::optional<std::string> interactionEvidence;
    std::optional<std::string> causalDagMermaid;
    std::unordered_map<size_t, double> mahalanobisByRow;
    double mahalanobisThreshold = 0.0;
    std::optional<std::string> executiveSummary;
};

struct ConditionalDriftAssessment {
    bool signFlip = false;
    bool magnitudeCollapse = false;
    double collapseRatio = 1.0;
    std::string label = "stable";
};

ConditionalDriftAssessment assessGlobalConditionalDrift(double globalR,
                                                        double conditionalR,
                                                        double minGlobalAbs = 0.15,
                                                        double collapseRatioThreshold = 0.50,
                                                        double collapseAbsDrop = 0.12) {
    ConditionalDriftAssessment out;
    if (!std::isfinite(globalR) || !std::isfinite(conditionalR)) {
        out.label = "insufficient";
        return out;
    }

    const double absGlobal = std::abs(globalR);
    const double absConditional = std::abs(conditionalR);
    out.collapseRatio = (absGlobal > 1e-12) ? (absConditional / absGlobal) : 1.0;

    if (absGlobal < minGlobalAbs) {
        out.label = "weak-global-signal";
        return out;
    }

    out.signFlip = (globalR * conditionalR < 0.0) && (absConditional >= 0.05);
    out.magnitudeCollapse =
        (absConditional <= absGlobal * collapseRatioThreshold) &&
        ((absGlobal - absConditional) >= collapseAbsDrop);

    if (out.signFlip && out.magnitudeCollapse) out.label = "flip+collapse";
    else if (out.signFlip) out.label = "sign-flip";
    else if (out.magnitudeCollapse) out.label = "magnitude-collapse";
    else out.label = "stable";
    return out;
}

std::string driftPatternLabel(const std::string& raw) {
    if (raw == "flip+collapse") return "sign reversal + magnitude collapse";
    if (raw == "sign-flip") return "sign reversal";
    if (raw == "magnitude-collapse") return "magnitude collapse";
    return raw;
}

std::string cramerStrengthLabel(double v) {
    if (v >= 0.60) return "very strong";
    if (v >= 0.40) return "strong";
    if (v >= 0.20) return "moderate";
    if (v >= 0.10) return "weak";
    return "very weak";
}

struct TemporalAxisDescriptor {
    std::vector<double> axis;
    std::string name;
};

TemporalAxisDescriptor detectTemporalAxis(const TypedDataset& data) {
    TemporalAxisDescriptor out;

    const auto datetimeIdx = data.datetimeColumnIndices();
    for (size_t idx : datetimeIdx) {
        const auto& col = data.columns()[idx];
        const auto& vals = std::get<std::vector<int64_t>>(col.values);
        if (vals.size() < 16) continue;

        out.axis.assign(vals.size(), 0.0);
        std::unordered_set<int64_t> unique;
        unique.reserve(vals.size());
        for (size_t r = 0; r < vals.size(); ++r) {
            out.axis[r] = static_cast<double>(vals[r]);
            if (r < col.missing.size() && !col.missing[r]) unique.insert(vals[r]);
        }
        if (unique.size() >= 8) {
            out.name = col.name;
            return out;
        }
    }

    const auto numericIdx = data.numericColumnIndices();
    for (size_t idx : numericIdx) {
        const std::string lower = CommonUtils::toLower(data.columns()[idx].name);
        if (!(containsToken(lower, {"time", "date", "index", "idx", "step", "order", "epoch", "year", "month", "day", "week"}) ||
              lower == "t" || lower == "ts")) {
            continue;
        }

        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        if (vals.size() < 16) continue;
        out.axis.assign(vals.begin(), vals.end());
        out.name = data.columns()[idx].name;
        return out;
    }

    out.axis.resize(data.rowCount(), 0.0);
    for (size_t r = 0; r < out.axis.size(); ++r) out.axis[r] = static_cast<double>(r + 1);
    out.name = "row_index";
    return out;
}

double absCorrAligned(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<size_t>& rows) {
    std::vector<double> xa;
    std::vector<double> ya;
    xa.reserve(rows.size());
    ya.reserve(rows.size());
    for (size_t r : rows) {
        if (r >= x.size() || r >= y.size()) continue;
        if (!std::isfinite(x[r]) || !std::isfinite(y[r])) continue;
        xa.push_back(x[r]);
        ya.push_back(y[r]);
    }
    if (xa.size() < 10) return 0.0;
    const ColumnStats xs = Statistics::calculateStats(xa);
    const ColumnStats ys = Statistics::calculateStats(ya);
    return std::abs(MathUtils::calculatePearson(xa, ya, xs, ys).value_or(0.0));
}

struct ContextualDeadZoneInsight {
    std::string feature;
    std::string strongCluster;
    std::string weakCluster;
    double strongCorr = 0.0;
    double weakCorr = 0.0;
    double dropRatio = 1.0;
    size_t support = 0;
};

std::vector<ContextualDeadZoneInsight> detectContextualDeadZones(const TypedDataset& data,
                                                                 size_t targetIdx,
                                                                 const std::vector<size_t>& candidateFeatures,
                                                                 size_t maxRows = 10) {
    std::vector<ContextualDeadZoneInsight> out;
    if (targetIdx >= data.columns().size() || data.columns()[targetIdx].type != ColumnType::NUMERIC) return out;

    std::vector<size_t> anchors;
    for (size_t idx : candidateFeatures) {
        if (idx == targetIdx) continue;
        if (data.columns()[idx].type != ColumnType::NUMERIC) continue;
        anchors.push_back(idx);
        if (anchors.size() >= 2) break;
    }
    if (anchors.size() < 2) return out;

    const auto& ax = std::get<std::vector<double>>(data.columns()[anchors[0]].values);
    const auto& ay = std::get<std::vector<double>>(data.columns()[anchors[1]].values);
    const auto& target = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    std::vector<size_t> validRows;
    validRows.reserve(data.rowCount());
    for (size_t r = 0; r < data.rowCount(); ++r) {
        if (r >= ax.size() || r >= ay.size() || r >= target.size()) continue;
        if (!std::isfinite(ax[r]) || !std::isfinite(ay[r]) || !std::isfinite(target[r])) continue;
        if (r < data.columns()[anchors[0]].missing.size() && data.columns()[anchors[0]].missing[r]) continue;
        if (r < data.columns()[anchors[1]].missing.size() && data.columns()[anchors[1]].missing[r]) continue;
        if (r < data.columns()[targetIdx].missing.size() && data.columns()[targetIdx].missing[r]) continue;
        validRows.push_back(r);
    }
    if (validRows.size() < 40) return out;

    std::pair<double, double> c0 = {ax[validRows.front()], ay[validRows.front()]};
    std::pair<double, double> c1 = {ax[validRows[validRows.size() / 2]], ay[validRows[validRows.size() / 2]]};
    std::unordered_map<size_t, int> clusterByRow;
    for (int iter = 0; iter < 20; ++iter) {
        double s0x = 0.0, s0y = 0.0, n0 = 0.0;
        double s1x = 0.0, s1y = 0.0, n1 = 0.0;
        for (size_t r : validRows) {
            const double d0 = (ax[r] - c0.first) * (ax[r] - c0.first) + (ay[r] - c0.second) * (ay[r] - c0.second);
            const double d1 = (ax[r] - c1.first) * (ax[r] - c1.first) + (ay[r] - c1.second) * (ay[r] - c1.second);
            const int cid = (d0 <= d1) ? 0 : 1;
            clusterByRow[r] = cid;
            if (cid == 0) {
                s0x += ax[r];
                s0y += ay[r];
                n0 += 1.0;
            } else {
                s1x += ax[r];
                s1y += ay[r];
                n1 += 1.0;
            }
        }
        if (n0 < 10.0 || n1 < 10.0) return out;
        c0 = {s0x / n0, s0y / n0};
        c1 = {s1x / n1, s1y / n1};
    }

    std::vector<size_t> rows0;
    std::vector<size_t> rows1;
    rows0.reserve(validRows.size());
    rows1.reserve(validRows.size());
    for (size_t r : validRows) {
        if (clusterByRow[r] == 0) rows0.push_back(r);
        else rows1.push_back(r);
    }
    if (rows0.size() < 12 || rows1.size() < 12) return out;

    for (size_t featureIdx : candidateFeatures) {
        if (featureIdx == targetIdx || featureIdx == anchors[0] || featureIdx == anchors[1]) continue;
        if (featureIdx >= data.columns().size() || data.columns()[featureIdx].type != ColumnType::NUMERIC) continue;
        const auto& fv = std::get<std::vector<double>>(data.columns()[featureIdx].values);

        const double cA = absCorrAligned(fv, target, rows0);
        const double cB = absCorrAligned(fv, target, rows1);
        const double strong = std::max(cA, cB);
        const double weak = std::min(cA, cB);
        const double delta = strong - weak;
        if (strong < 0.35 || weak > 0.10 || delta < 0.25) continue;

        const bool aStrong = cA >= cB;
        out.push_back({
            data.columns()[featureIdx].name,
            aStrong ? "Cluster A" : "Cluster B",
            aStrong ? "Cluster B" : "Cluster A",
            strong,
            weak,
            (strong > 1e-9) ? (weak / strong) : 1.0,
            std::min(rows0.size(), rows1.size())
        });
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        const double da = a.strongCorr - a.weakCorr;
        const double db = b.strongCorr - b.weakCorr;
        if (da == db) return a.dropRatio < b.dropRatio;
        return da > db;
    });
    if (out.size() > maxRows) out.resize(maxRows);
    return out;
}

