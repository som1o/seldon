#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
#include "GnuplotEngine.h"
#include "MathUtils.h"
#include "NeuralNet.h"
#include "Preprocessor.h"
#include "ReportEngine.h"
#include "SeldonExceptions.h"
#include "Statistics.h"
#include "TypedDataset.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace {
std::string toFixed(double v, int prec = 4) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

int resolveTargetIndex(const TypedDataset& data, const AutoConfig& config) {
    if (!config.targetColumn.empty()) {
        int idx = data.findColumnIndex(config.targetColumn);
        if (idx < 0) {
            throw Seldon::ConfigurationException("Target column not found: " + config.targetColumn);
        }
        if (data.columns()[idx].type != ColumnType::NUMERIC) {
            throw Seldon::ConfigurationException("Target column must be numeric: " + config.targetColumn);
        }
        return idx;
    }

    auto numeric = data.numericColumnIndices();
    if (numeric.empty()) {
        throw Seldon::ConfigurationException("Could not resolve a numeric target column");
    }
    return static_cast<int>(numeric.back());
}

bool isBinaryTargetRaw(const TypedDataset& ds, int targetIdx) {
    if (targetIdx < 0 || ds.columns()[targetIdx].type != ColumnType::NUMERIC) return false;
    const auto& y = std::get<std::vector<double>>(ds.columns()[targetIdx].values);
    for (double v : y) {
        if (std::abs(v) > 1e-9 && std::abs(v - 1.0) > 1e-9) return false;
    }
    return true;
}

std::vector<int> collectFeatureIndices(const TypedDataset& data, int targetIdx, const AutoConfig& config) {
    auto numeric = data.numericColumnIndices();
    std::vector<int> featureIdx;
    for (size_t idx : numeric) {
        if (static_cast<int>(idx) == targetIdx) continue;
        const std::string& name = data.columns()[idx].name;
        if (std::find(config.excludedColumns.begin(), config.excludedColumns.end(), name) != config.excludedColumns.end()) continue;
        featureIdx.push_back(static_cast<int>(idx));
    }
    if (featureIdx.empty()) {
        throw Seldon::ConfigurationException("No numeric input features available after exclusions");
    }
    return featureIdx;
}

struct PairInsight {
    size_t idxA = 0;
    size_t idxB = 0;
    std::string featureA;
    std::string featureB;
    double r = 0.0;
    double r2 = 0.0;
    double slope = 0.0;
    double intercept = 0.0;
    double tStat = 0.0;
    double pValue = 1.0;
    bool statSignificant = false;
    double neuralScore = 0.0;
    bool selected = false;
    std::string plotPath;
};

struct NeuralAnalysis {
    size_t inputNodes = 0;
    size_t hiddenNodes = 0;
    size_t outputNodes = 0;
    bool binaryTarget = false;
    std::vector<double> trainLoss;
    std::vector<double> valLoss;
    std::vector<double> featureImportance;
};

struct NumericDetailedStats {
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

double quantileSorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    if (q <= 0.0) return sorted.front();
    if (q >= 1.0) return sorted.back();
    double pos = q * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sorted[lo];
    double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

NumericDetailedStats detailedNumeric(const std::vector<double>& values) {
    NumericDetailedStats out;
    if (values.empty()) return out;

    ColumnStats st = Statistics::calculateStats(values);
    out.mean = st.mean;
    out.median = st.median;
    out.variance = st.variance;
    out.stddev = st.stddev;
    out.skewness = st.skewness;
    out.kurtosis = st.kurtosis;

    out.min = *std::min_element(values.begin(), values.end());
    out.max = *std::max_element(values.begin(), values.end());
    out.range = out.max - out.min;

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    out.q1 = quantileSorted(sorted, 0.25);
    out.q3 = quantileSorted(sorted, 0.75);
    out.iqr = out.q3 - out.q1;
    out.p05 = quantileSorted(sorted, 0.05);
    out.p95 = quantileSorted(sorted, 0.95);

    std::vector<double> absDev(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        absDev[i] = std::abs(values[i] - out.median);
        out.sum += values[i];
        if (std::abs(values[i]) > 1e-12) out.nonZero++;
    }
    std::sort(absDev.begin(), absDev.end());
    out.mad = quantileSorted(absDev, 0.5);

    if (std::abs(out.mean) > 1e-12) out.coeffVar = out.stddev / std::abs(out.mean);
    return out;
}

std::string plotSubdir(const AutoConfig& cfg, const std::string& name) {
    return cfg.assetsDir + "/" + name;
}

void cleanupOutputs(const AutoConfig& config) {
    namespace fs = std::filesystem;

    const std::vector<std::string> rootFiles = {
        "seldon_report.html",
        "univariate.html",
        "univaraite.html",
        "bivariate.html",
        "seldon_model.seldon",
        "univaraite.txt",
        "bivariate.txt",
        "neural_synthesis.txt",
        "final_analysis.txt"
    };

    std::error_code ec;
    for (const auto& f : rootFiles) {
        fs::remove(f, ec);
    }
    fs::remove(config.reportFile, ec);

    fs::remove_all(config.assetsDir, ec);
    fs::create_directories(config.assetsDir, ec);
}

void addUnivariateDetailedSection(ReportEngine& report, const TypedDataset& data, const PreprocessReport& prep, bool verbose) {
    std::vector<std::vector<std::string>> summaryRows;
    std::vector<std::vector<std::string>> categoricalRows;
    summaryRows.reserve(data.colCount());

    if (verbose) {
        std::cout << "[Seldon][Univariate] Starting per-column statistical profiling...\n";
    }

    for (const auto& col : data.columns()) {
        const size_t missing = prep.missingCounts.count(col.name) ? prep.missingCounts.at(col.name) : 0;
        const size_t outliers = prep.outlierCounts.count(col.name) ? prep.outlierCounts.at(col.name) : 0;

        if (col.type == ColumnType::NUMERIC) {
            const auto& vals = std::get<std::vector<double>>(col.values);
            NumericDetailedStats st = detailedNumeric(vals);

            summaryRows.push_back({
                col.name,
                "numeric",
                toFixed(st.mean),
                toFixed(st.median),
                toFixed(st.min),
                toFixed(st.max),
                toFixed(st.range),
                toFixed(st.q1),
                toFixed(st.q3),
                toFixed(st.iqr),
                toFixed(st.p05),
                toFixed(st.p95),
                toFixed(st.mad),
                toFixed(st.variance),
                toFixed(st.stddev),
                toFixed(st.skewness),
                toFixed(st.kurtosis),
                toFixed(st.coeffVar),
                toFixed(st.sum),
                std::to_string(st.nonZero),
                std::to_string(missing),
                std::to_string(outliers)
            });
            if (verbose) {
                std::cout << "[Seldon][Univariate] Numeric " << col.name
                          << " | n=" << vals.size()
                          << " mean=" << toFixed(st.mean)
                          << " std=" << toFixed(st.stddev)
                          << " iqr=" << toFixed(st.iqr)
                          << " missing=" << missing
                          << " outliers=" << outliers << "\n";
            }
        } else if (col.type == ColumnType::CATEGORICAL) {
            const auto& vals = std::get<std::vector<std::string>>(col.values);
            std::map<std::string, size_t> freq;
            for (const auto& v : vals) freq[v]++;

            std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            std::string mode = top.empty() ? "-" : top.front().first;
            size_t modeCount = top.empty() ? 0 : top.front().second;
            double modeRatio = vals.empty() ? 0.0 : static_cast<double>(modeCount) / static_cast<double>(vals.size());

            summaryRows.push_back({
                col.name,
                "categorical",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                std::to_string(freq.size()),
                std::to_string(missing),
                "-"
            });

            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                categoricalRows.push_back({col.name, top[i].first, std::to_string(top[i].second), toFixed(static_cast<double>(top[i].second) / static_cast<double>(std::max<size_t>(1, vals.size())), 6)});
            }

            if (verbose) {
                std::cout << "[Seldon][Univariate] Categorical " << col.name
                          << " | unique=" << freq.size()
                          << " mode='" << mode << "'"
                          << " mode_ratio=" << toFixed(modeRatio, 6)
                          << " missing=" << missing << "\n";
            }
        } else {
            const auto& vals = std::get<std::vector<int64_t>>(col.values);
            int64_t minTs = vals.empty() ? 0 : *std::min_element(vals.begin(), vals.end());
            int64_t maxTs = vals.empty() ? 0 : *std::max_element(vals.begin(), vals.end());
            summaryRows.push_back({
                col.name,
                "datetime",
                "-",
                "-",
                "-",
                std::to_string(minTs),
                std::to_string(maxTs),
                std::to_string(maxTs - minTs),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                std::to_string(vals.size()),
                std::to_string(missing),
                "-"
            });
            if (verbose) {
                std::cout << "[Seldon][Univariate] Datetime " << col.name
                          << " | min_ts=" << minTs
                          << " max_ts=" << maxTs
                          << " missing=" << missing << "\n";
            }
        }
    }

    report.addTable("Column Statistical Super-Summary", {
        "Column", "Type", "Mean", "Median", "Min", "Max", "Range", "Q1", "Q3", "IQR",
        "P05", "P95", "MAD", "Variance", "StdDev", "Skew", "Kurtosis", "CoeffVar", "Sum", "NonZero/Unique", "Missing", "Outliers"
    }, summaryRows);

    if (!categoricalRows.empty()) {
        report.addTable("Categorical Top Frequencies", {"Column", "Category", "Count", "Share"}, categoricalRows);
    }
}

void addBenchmarkSection(ReportEngine& report, const std::vector<BenchmarkResult>& benchmarks) {
    std::vector<std::vector<std::string>> benchRows;
    for (const auto& b : benchmarks) {
        benchRows.push_back({b.model, toFixed(b.rmse), toFixed(b.r2), toFixed(b.accuracy)});
    }
    report.addTable("Baseline Benchmark (k-fold)", {"Model", "RMSE", "R2", "Accuracy"}, benchRows);
}

NeuralAnalysis runNeuralAnalysis(const TypedDataset& data,
                                 int targetIdx,
                                 const std::vector<int>& featureIdx,
                                 bool binaryTarget,
                                 bool verbose,
                                 const AutoConfig& config) {
    const auto& y = std::get<std::vector<double>>(data.columns()[targetIdx].values);

    std::vector<std::vector<double>> Xnn(data.rowCount(), std::vector<double>(featureIdx.size(), 0.0));
    std::vector<std::vector<double>> Ynn(data.rowCount(), std::vector<double>(1, 0.0));

    for (size_t r = 0; r < data.rowCount(); ++r) {
        for (size_t c = 0; c < featureIdx.size(); ++c) {
            const auto& fv = std::get<std::vector<double>>(data.columns()[featureIdx[c]].values);
            Xnn[r][c] = fv[r];
        }
        Ynn[r][0] = y[r];
    }

    size_t inputNodes = featureIdx.size();
    size_t outputNodes = 1;
    size_t hidden = std::max<size_t>(2, (inputNodes + outputNodes) / 2);

    NeuralNet nn({inputNodes, hidden, outputNodes});
    NeuralNet::Hyperparameters hp;
    hp.epochs = 200;
    hp.batchSize = 32;
    hp.earlyStoppingPatience = 12;
    hp.lrStepSize = 50;
    hp.lrDecay = 0.5;
    hp.activation = NeuralNet::Activation::RELU;
    hp.outputActivation = NeuralNet::Activation::SIGMOID;
    hp.loss = binaryTarget ? NeuralNet::LossFunction::CROSS_ENTROPY : NeuralNet::LossFunction::MSE;
    hp.verbose = verbose;
    hp.seed = config.neuralSeed;
    hp.gradientClipNorm = config.gradientClipNorm;

    nn.train(Xnn, Ynn, hp);

    NeuralAnalysis analysis;
    analysis.inputNodes = inputNodes;
    analysis.hiddenNodes = hidden;
    analysis.outputNodes = outputNodes;
    analysis.binaryTarget = binaryTarget;
    analysis.trainLoss = nn.getTrainLossHistory();
    analysis.valLoss = nn.getValLossHistory();
    analysis.featureImportance = nn.calculateFeatureImportance(Xnn, Ynn, 5);
    return analysis;
}

std::vector<PairInsight> analyzeBivariatePairs(const TypedDataset& data,
                                               const std::unordered_map<size_t, double>& importanceByIndex,
                                               GnuplotEngine* plotter,
                                               bool verbose) {
    std::vector<PairInsight> pairs;
    const auto numericIdx = data.numericColumnIndices();
    const size_t n = data.rowCount();

    if (verbose) {
        size_t totalPairs = numericIdx.size() < 2 ? 0 : (numericIdx.size() * (numericIdx.size() - 1)) / 2;
        std::cout << "[Seldon][Bivariate] Evaluating " << totalPairs << " pair combinations...\n";
    }

    for (size_t aPos = 0; aPos < numericIdx.size(); ++aPos) {
        for (size_t bPos = aPos + 1; bPos < numericIdx.size(); ++bPos) {
            size_t ia = numericIdx[aPos];
            size_t ib = numericIdx[bPos];
            const auto& va = std::get<std::vector<double>>(data.columns()[ia].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[ib].values);

            PairInsight p;
            p.idxA = ia;
            p.idxB = ib;
            p.featureA = data.columns()[ia].name;
            p.featureB = data.columns()[ib].name;
            ColumnStats statsA = Statistics::calculateStats(va);
            ColumnStats statsB = Statistics::calculateStats(vb);
            p.r = MathUtils::calculatePearson(va, vb, statsA, statsB).value_or(0.0);
            p.r2 = p.r * p.r;
            // Keep regression parameter derivation centralized in MathUtils.
            auto fit = MathUtils::simpleLinearRegression(va, vb, statsA, statsB, p.r);
            p.slope = fit.first;
            p.intercept = fit.second;

            auto sig = MathUtils::calculateSignificance(p.r, n);
            p.pValue = sig.p_value;
            p.tStat = sig.t_stat;
            p.statSignificant = sig.is_significant;

            double impA = 0.0;
            double impB = 0.0;
            auto ita = importanceByIndex.find(ia);
            auto itb = importanceByIndex.find(ib);
            if (ita != importanceByIndex.end()) impA = ita->second;
            if (itb != importanceByIndex.end()) impB = itb->second;
            p.neuralScore = std::abs(p.r) * ((impA + impB) / 2.0 + 1e-9);

            pairs.push_back(std::move(p));

            if (verbose) {
                const auto& ref = pairs.back();
                std::cout << "[Seldon][Bivariate] " << ref.featureA << " vs " << ref.featureB
                          << " | r=" << toFixed(ref.r, 6)
                          << " p=" << toFixed(ref.pValue, 8)
                          << " t=" << toFixed(ref.tStat, 6)
                          << " neural=" << toFixed(ref.neuralScore, 8) << "\n";
            }
        }
    }

    std::vector<double> significantScores;
    for (const auto& p : pairs) {
        if (p.statSignificant) significantScores.push_back(p.neuralScore);
    }

    // Dynamic cutoff couples statistical significance with neural relevance.
    double dynamicCutoff = 0.0;
    if (!significantScores.empty()) {
        for (double s : significantScores) dynamicCutoff += s;
        dynamicCutoff /= static_cast<double>(significantScores.size());
    }

    for (auto& p : pairs) {
        p.selected = p.statSignificant && (p.neuralScore >= dynamicCutoff * 0.75);
        if (p.selected && plotter) {
            const auto& va = std::get<std::vector<double>>(data.columns()[p.idxA].values);
            const auto& vb = std::get<std::vector<double>>(data.columns()[p.idxB].values);
            std::string id = "sig_" + p.featureA + "_" + p.featureB;
            p.plotPath = plotter->scatter(id, va, vb, p.featureA + " vs " + p.featureB);
        }
        if (verbose) {
            std::cout << "[Seldon][Bivariate] Decision " << p.featureA << " vs " << p.featureB
                      << " => " << (p.selected ? "SELECTED" : "REJECTED")
                      << " (cutoff=" << toFixed(dynamicCutoff * 0.75, 8) << ")\n";
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairInsight& lhs, const PairInsight& rhs) {
        if (lhs.selected != rhs.selected) return lhs.selected > rhs.selected;
        return std::abs(lhs.r) > std::abs(rhs.r);
    });

    return pairs;
}

void addOverallSections(ReportEngine& report,
                        const TypedDataset& data,
                        const PreprocessReport& prep,
                        const std::vector<BenchmarkResult>& benchmarks,
                        const NeuralAnalysis& neural,
                        GnuplotEngine* overallPlotter,
                        bool canPlotOverall,
                        bool verbose) {
    report.addParagraph("Overall analysis aggregates dataset health, model baselines, neural training behavior, and optional global visual diagnostics.");

    size_t totalMissing = 0;
    size_t totalOutliers = 0;
    for (const auto& kv : prep.missingCounts) totalMissing += kv.second;
    for (const auto& kv : prep.outlierCounts) totalOutliers += kv.second;

    report.addTable("Dataset Health", {"Metric", "Value"}, {
        {"Rows", std::to_string(data.rowCount())},
        {"Columns", std::to_string(data.colCount())},
        {"Numeric Columns", std::to_string(data.numericColumnIndices().size())},
        {"Categorical Columns", std::to_string(data.categoricalColumnIndices().size())},
        {"Datetime Columns", std::to_string(data.datetimeColumnIndices().size())},
        {"Total Missing Cells", std::to_string(totalMissing)},
        {"Total Outlier Flags", std::to_string(totalOutliers)}
    });

    addBenchmarkSection(report, benchmarks);

    std::vector<std::vector<std::string>> lossSummary;
    if (!neural.trainLoss.empty()) {
        double minTrain = *std::min_element(neural.trainLoss.begin(), neural.trainLoss.end());
        double maxTrain = *std::max_element(neural.trainLoss.begin(), neural.trainLoss.end());
        double endTrain = neural.trainLoss.back();
        lossSummary.push_back({"train", toFixed(minTrain, 6), toFixed(maxTrain, 6), toFixed(endTrain, 6)});
    }
    if (!neural.valLoss.empty()) {
        double minVal = *std::min_element(neural.valLoss.begin(), neural.valLoss.end());
        double maxVal = *std::max_element(neural.valLoss.begin(), neural.valLoss.end());
        double endVal = neural.valLoss.back();
        lossSummary.push_back({"validation", toFixed(minVal, 6), toFixed(maxVal, 6), toFixed(endVal, 6)});
    }
    if (!lossSummary.empty()) {
        report.addTable("Neural Loss Summary", {"Series", "Min", "Max", "Final"}, lossSummary);
    }

    if (overallPlotter && canPlotOverall) {
        std::vector<std::string> labels;
        std::vector<double> missingCounts;
        for (const auto& col : data.columns()) {
            labels.push_back(col.name);
            auto it = prep.missingCounts.find(col.name);
            missingCounts.push_back(it == prep.missingCounts.end() ? 0.0 : static_cast<double>(it->second));
        }
        std::string missingImg = overallPlotter->bar("overall_missingness", labels, missingCounts, "Overall Missingness by Column");
        if (!missingImg.empty()) report.addImage("Overall Missingness", missingImg);

        auto numericIdx = data.numericColumnIndices();
        if (numericIdx.size() >= 2) {
            std::vector<std::vector<double>> corr(numericIdx.size(), std::vector<double>(numericIdx.size(), 1.0));
            for (size_t i = 0; i < numericIdx.size(); ++i) {
                for (size_t j = i + 1; j < numericIdx.size(); ++j) {
                    const auto& a = std::get<std::vector<double>>(data.columns()[numericIdx[i]].values);
                    const auto& b = std::get<std::vector<double>>(data.columns()[numericIdx[j]].values);
                    ColumnStats statsA = Statistics::calculateStats(a);
                    ColumnStats statsB = Statistics::calculateStats(b);
                    double r = MathUtils::calculatePearson(a, b, statsA, statsB).value_or(0.0);
                    corr[i][j] = r;
                    corr[j][i] = r;
                }
            }
            std::string img = overallPlotter->heatmap("overall_corr_heatmap", corr, "Overall Correlation Heatmap");
            if (!img.empty()) report.addImage("Overall Correlation Heatmap", img);
        }

        if (!neural.trainLoss.empty()) {
            std::vector<double> epochs(neural.trainLoss.size(), 0.0);
            for (size_t i = 0; i < epochs.size(); ++i) epochs[i] = static_cast<double>(i + 1);
            std::string trainImg = overallPlotter->line("overall_nn_train_loss", epochs, neural.trainLoss, "NN Train Loss");
            if (!trainImg.empty()) report.addImage("Overall NN Train Loss", trainImg);

            if (!neural.valLoss.empty() && neural.valLoss.size() == neural.trainLoss.size()) {
                std::string valImg = overallPlotter->line("overall_nn_val_loss", epochs, neural.valLoss, "NN Validation Loss");
                if (!valImg.empty()) report.addImage("Overall NN Validation Loss", valImg);
            }
        }

        if (verbose) {
            std::cout << "[Seldon][Overall] Optional overall plots generated in dedicated folder.\n";
        }
    } else {
        report.addParagraph("Overall plots were skipped (supervised mode disabled or gnuplot unavailable).");
    }
}
} // namespace

int AutomationPipeline::run(const AutoConfig& config) {
    AutoConfig runCfg = config;
    runCfg.plot.format = "png";

    cleanupOutputs(runCfg);

    TypedDataset data(config.datasetPath, config.delimiter);
    data.load();
    if (data.rowCount() == 0 || data.colCount() == 0) {
        throw Seldon::DatasetException("Dataset has no usable rows/columns");
    }

    for (const auto& ex : config.excludedColumns) {
        if (data.findColumnIndex(ex) < 0) {
            throw Seldon::ConfigurationException("Excluded column not found: " + ex);
        }
    }

    int targetIdx = resolveTargetIndex(data, config);
    bool targetBinaryOriginal = isBinaryTargetRaw(data, targetIdx);

    PreprocessReport prep = Preprocessor::run(data, runCfg);

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Univariate] Preparing deeply detailed univariate analysis...\n";
    }

    ReportEngine univariate;
    univariate.addTitle("Univariate Analysis");
    univariate.addParagraph("Dataset: " + runCfg.datasetPath);
    univariate.addParagraph("Rows: " + std::to_string(data.rowCount()) + " | Columns: " + std::to_string(data.colCount()));
    addUnivariateDetailedSection(univariate, data, prep, runCfg.verboseAnalysis);

    GnuplotEngine plotterBivariate(plotSubdir(runCfg, "bivariate"), runCfg.plot);
    GnuplotEngine plotterUnivariate(plotSubdir(runCfg, "univariate"), runCfg.plot);
    GnuplotEngine plotterOverall(plotSubdir(runCfg, "overall"), runCfg.plot);
    bool canPlot = plotterBivariate.isAvailable();
    univariate.addParagraph(canPlot ? "Gnuplot detected." : "Gnuplot not available: plot generation skipped.");

    if (runCfg.plotUnivariate && canPlot) {
        univariate.addParagraph("Supervised setting enabled: univariate plots generated in a dedicated folder.");
        if (runCfg.verboseAnalysis) {
            std::cout << "[Seldon][Univariate] Generating supervised univariate plots...\n";
        }

        for (size_t idx : data.numericColumnIndices()) {
            const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
            std::string img = plotterUnivariate.histogram("uni_hist_" + std::to_string(idx), vals, "Histogram: " + data.columns()[idx].name);
            if (!img.empty()) {
                univariate.addImage("Histogram: " + data.columns()[idx].name, img);
                if (runCfg.verboseAnalysis) std::cout << "[Seldon][Univariate] Plot generated: " << img << "\n";
            }
        }

        for (size_t idx : data.categoricalColumnIndices()) {
            const auto& vals = std::get<std::vector<std::string>>(data.columns()[idx].values);
            std::map<std::string, size_t> freq;
            for (const auto& v : vals) freq[v]++;
            std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            if (top.size() > 12) top.resize(12);
            std::vector<std::string> labels;
            std::vector<double> counts;
            for (const auto& kv : top) {
                labels.push_back(kv.first);
                counts.push_back(static_cast<double>(kv.second));
            }
            std::string img = plotterUnivariate.bar("uni_cat_" + std::to_string(idx), labels, counts, "Category Frequencies: " + data.columns()[idx].name);
            if (!img.empty()) {
                univariate.addImage("Category Plot: " + data.columns()[idx].name, img);
                if (runCfg.verboseAnalysis) std::cout << "[Seldon][Univariate] Plot generated: " << img << "\n";
            }
        }
    } else {
        univariate.addParagraph("Supervised setting disabled: univariate plots skipped.");
    }

    std::vector<int> featureIdx = collectFeatureIndices(data, targetIdx, runCfg);
    auto benchmarks = BenchmarkEngine::run(data, targetIdx, featureIdx, runCfg.kfold);

    if (runCfg.verboseAnalysis) {
        std::cout << "[Seldon][Neural] Starting neural lattice training with verbose trace...\n";
    }
    NeuralAnalysis neural = runNeuralAnalysis(data, targetIdx, featureIdx, targetBinaryOriginal, runCfg.verboseAnalysis, runCfg);

    std::unordered_map<size_t, double> importanceByIndex;
    double fallbackImportance = 0.0;
    for (double v : neural.featureImportance) fallbackImportance += v;
    if (!neural.featureImportance.empty()) fallbackImportance /= neural.featureImportance.size();
    for (size_t i = 0; i < featureIdx.size() && i < neural.featureImportance.size(); ++i) {
        importanceByIndex[static_cast<size_t>(featureIdx[i])] = neural.featureImportance[i];
    }
    importanceByIndex[static_cast<size_t>(targetIdx)] = fallbackImportance;

    const auto numeric = data.numericColumnIndices();
    for (size_t idx : numeric) {
        if (importanceByIndex.find(idx) == importanceByIndex.end()) {
            importanceByIndex[idx] = fallbackImportance;
        }
    }

    GnuplotEngine* bivariatePlotter = (canPlot && runCfg.plotBivariateSignificant) ? &plotterBivariate : nullptr;
    auto bivariatePairs = analyzeBivariatePairs(data, importanceByIndex, bivariatePlotter, runCfg.verboseAnalysis);

    ReportEngine bivariate;
    bivariate.addTitle("Bivariate Analysis");
    bivariate.addParagraph("All numeric pair combinations are included below (nC2). Significant table is dynamically filtered using statistical significance + neural relevance score.");
    bivariate.addParagraph("Neural-lattice significance score drives final pair selection; only selected pairs are considered significant findings.");

    std::vector<std::vector<std::string>> allRows;
    std::vector<std::vector<std::string>> sigRows;
    size_t statSigCount = 0;
    for (const auto& p : bivariatePairs) {
        if (p.statSignificant) statSigCount++;
        allRows.push_back({
            p.featureA,
            p.featureB,
            toFixed(p.r),
            toFixed(p.r2),
            toFixed(p.slope),
            toFixed(p.intercept),
            toFixed(p.tStat, 6),
            toFixed(p.pValue, 6),
            p.statSignificant ? "yes" : "no",
            toFixed(p.neuralScore, 6),
            p.selected ? "yes" : "no",
            p.plotPath.empty() ? "-" : p.plotPath
        });

        if (p.selected) {
            sigRows.push_back({
                p.featureA,
                p.featureB,
                toFixed(p.r),
                toFixed(p.r2),
                toFixed(p.slope),
                toFixed(p.intercept),
                toFixed(p.tStat, 6),
                toFixed(p.pValue, 6),
                toFixed(p.neuralScore, 6),
                p.plotPath.empty() ? "-" : p.plotPath
            });
            if (!p.plotPath.empty()) {
                bivariate.addImage("Significant Pair: " + p.featureA + " vs " + p.featureB, p.plotPath);
            }
        }
    }

    bivariate.addParagraph("Total pairs evaluated: " + std::to_string(allRows.size()));
    bivariate.addParagraph("Statistically significant pairs (p<0.05): " + std::to_string(statSigCount));
    bivariate.addParagraph("Final neural-selected significant pairs: " + std::to_string(sigRows.size()));
    bivariate.addTable("All Pairwise Results", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "stat_sig", "neural_score", "selected", "plot"}, allRows);
    bivariate.addTable("Final Significant Results", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "neural_score", "plot"}, sigRows);

    ReportEngine neuralReport;
    neuralReport.addTitle("Neural Synthesis");
    neuralReport.addParagraph("This synthesis captures detailed lattice training traces and how neural relevance influenced bivariate selection.");
    neuralReport.addTable("Neural Network Auto-Defaults", {"Setting", "Value"}, {
        {"Input Nodes", std::to_string(neural.inputNodes)},
        {"Hidden Nodes", std::to_string(neural.hiddenNodes)},
        {"Output Nodes", std::to_string(neural.outputNodes)},
        {"Loss", neural.binaryTarget ? "cross_entropy" : "mse"},
        {"Activation", "relu"},
        {"Output Activation", "sigmoid"}
    });

    std::vector<std::vector<std::string>> fiRows;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
        fiRows.push_back({name, toFixed(imp, 6)});
    }
    neuralReport.addTable("Feature Importance (Permutation, Neural)", {"Feature", "Importance"}, fiRows);

    std::vector<std::vector<std::string>> lossRows;
    for (size_t e = 0; e < neural.trainLoss.size(); ++e) {
        double val = (e < neural.valLoss.size() ? neural.valLoss[e] : 0.0);
        lossRows.push_back({
            std::to_string(e + 1),
            toFixed(neural.trainLoss[e], 6),
            toFixed(val, 6)
        });
    }
    neuralReport.addTable("Neural Lattice Training Trace", {"Epoch", "Train Loss", "Validation Loss"}, lossRows);

    addOverallSections(neuralReport, data, prep, benchmarks, neural, &plotterOverall, canPlot && runCfg.plotOverall, runCfg.verboseAnalysis);

    ReportEngine finalAnalysis;
    finalAnalysis.addTitle("Final Analysis - Significant Findings Only");
    finalAnalysis.addParagraph("This report contains only neural-net-approved significant findings (dynamic decision engine). Non-selected findings are excluded by design.");
    finalAnalysis.addTable("Neural-Selected Significant Bivariate Findings", {"Feature A", "Feature B", "r", "r2", "slope", "intercept", "t_stat", "p_value", "neural_score", "plot"}, sigRows);

    std::vector<std::vector<std::string>> topFeatures;
    std::vector<std::pair<std::string, double>> fiPairs;
    for (size_t i = 0; i < featureIdx.size(); ++i) {
        const std::string name = data.columns()[featureIdx[i]].name;
        double imp = (i < neural.featureImportance.size()) ? neural.featureImportance[i] : 0.0;
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
        {"Pairs Evaluated", std::to_string(allRows.size())},
        {"Pairs Statistically Significant", std::to_string(statSigCount)},
        {"Pairs Neural-Selected", std::to_string(sigRows.size())},
        {"Training Epochs Executed", std::to_string(neural.trainLoss.size())}
    });

    univariate.save("univaraite.txt");
    bivariate.save("bivariate.txt");
    neuralReport.save("neural_synthesis.txt");
    finalAnalysis.save("final_analysis.txt");

    std::cout << "[Seldon] Automated pipeline complete.\n";
    std::cout << "[Seldon] Reports: univaraite.txt, bivariate.txt, neural_synthesis.txt, final_analysis.txt\n";
    std::cout << "[Seldon] Plot folders: "
              << plotSubdir(runCfg, "univariate") << ", "
              << plotSubdir(runCfg, "bivariate") << ", "
              << plotSubdir(runCfg, "overall") << "\n";

    return 0;
}
