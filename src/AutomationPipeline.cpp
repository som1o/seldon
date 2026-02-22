#include "AutomationPipeline.h"

#include "BenchmarkEngine.h"
#include "GnuplotEngine.h"
#include "NeuralNet.h"
#include "Preprocessor.h"
#include "ReportEngine.h"
#include "SeldonExceptions.h"
#include "StatsEngine.h"
#include "TypedDataset.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace {
std::string toFixed(double v, int prec = 4) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    double meanX = 0.0, meanY = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= x.size();
    meanY /= y.size();

    double num = 0.0, denX = 0.0, denY = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }

    if (denX <= 1e-12 || denY <= 1e-12) return 0.0;
    return num / std::sqrt(denX * denY);
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

void addSummarySection(ReportEngine& report, const TypedDataset& data, const PreprocessReport& prep) {
    std::vector<std::vector<std::string>> summaryRows;
    summaryRows.reserve(data.colCount());

    for (const auto& col : data.columns()) {
        if (col.type == ColumnType::NUMERIC) {
            ColumnStats st = StatsEngine::calculateStats(std::get<std::vector<double>>(col.values));
            summaryRows.push_back({
                col.name,
                "numeric",
                toFixed(st.mean),
                toFixed(st.median),
                toFixed(st.stddev),
                std::to_string(prep.missingCounts.at(col.name)),
                std::to_string(prep.outlierCounts.at(col.name))
            });
        } else if (col.type == ColumnType::CATEGORICAL) {
            summaryRows.push_back({
                col.name,
                "categorical",
                "-",
                "-",
                "-",
                std::to_string(prep.missingCounts.at(col.name)),
                "-"
            });
        } else {
            summaryRows.push_back({
                col.name,
                "datetime",
                "-",
                "-",
                "-",
                std::to_string(prep.missingCounts.at(col.name)),
                "-"
            });
        }
    }

    report.addTable("Column Summary", {"Column", "Type", "Mean", "Median", "StdDev", "Missing", "Outliers"}, summaryRows);
}

void addPlotSections(ReportEngine& report, GnuplotEngine& plotter, const TypedDataset& data, const PreprocessReport& prep) {
    std::vector<std::string> labels;
    std::vector<double> missingCounts;
    for (const auto& col : data.columns()) {
        labels.push_back(col.name);
        auto it = prep.missingCounts.find(col.name);
        missingCounts.push_back(it == prep.missingCounts.end() ? 0.0 : static_cast<double>(it->second));
    }

    std::string missingImg = plotter.bar("missingness", labels, missingCounts, "Missing Values by Column");
    if (!missingImg.empty()) {
        report.addImage("Missing Data", missingImg);
    }

    auto numericIdx = data.numericColumnIndices();
    size_t plotted = 0;
    for (size_t idx : numericIdx) {
        if (plotted++ >= 8) break;
        const auto& vals = std::get<std::vector<double>>(data.columns()[idx].values);
        std::string img = plotter.histogram("hist_" + std::to_string(idx), vals, data.columns()[idx].name);
        if (!img.empty()) {
            report.addImage("Histogram: " + data.columns()[idx].name, img);
        }
    }

    if (numericIdx.size() >= 2) {
        std::vector<std::vector<double>> corr(numericIdx.size(), std::vector<double>(numericIdx.size(), 1.0));
        for (size_t i = 0; i < numericIdx.size(); ++i) {
            for (size_t j = i + 1; j < numericIdx.size(); ++j) {
                const auto& a = std::get<std::vector<double>>(data.columns()[numericIdx[i]].values);
                const auto& b = std::get<std::vector<double>>(data.columns()[numericIdx[j]].values);
                double r = pearson(a, b);
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }
        std::string img = plotter.heatmap("corr_heatmap", corr, "Correlation Heatmap");
        if (!img.empty()) {
            report.addImage("Correlation Heatmap", img);
        }
    }

    size_t plottedCats = 0;
    for (size_t ci : data.categoricalColumnIndices()) {
        if (plottedCats++ >= 5) break;

        const auto& vals = std::get<std::vector<std::string>>(data.columns()[ci].values);
        std::map<std::string, size_t> freq;
        for (const auto& v : vals) freq[v]++;

        std::vector<std::pair<std::string, size_t>> top(freq.begin(), freq.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        if (top.size() > 10) top.resize(10);

        std::vector<std::string> catLabels;
        std::vector<double> catCounts;
        for (const auto& kv : top) {
            catLabels.push_back(kv.first);
            catCounts.push_back(static_cast<double>(kv.second));
        }

        std::string img = plotter.bar("cat_" + std::to_string(ci), catLabels, catCounts, data.columns()[ci].name);
        if (!img.empty()) {
            report.addImage("Category Frequency: " + data.columns()[ci].name, img);
        }
    }
}

void addBenchmarkSection(ReportEngine& report, GnuplotEngine* plotter, const std::vector<BenchmarkResult>& benchmarks) {
    std::vector<std::vector<std::string>> benchRows;
    for (const auto& b : benchmarks) {
        benchRows.push_back({b.model, toFixed(b.rmse), toFixed(b.r2), toFixed(b.accuracy)});
    }
    report.addTable("Baseline Benchmark (k-fold)", {"Model", "RMSE", "R2", "Accuracy"}, benchRows);

    if (plotter && !benchmarks.empty()) {
        const auto& first = benchmarks.front();
        std::vector<double> residuals(first.actual.size(), 0.0);
        for (size_t i = 0; i < first.actual.size() && i < first.predicted.size(); ++i) {
            residuals[i] = first.actual[i] - first.predicted[i];
        }
        std::string img = plotter->scatter("residuals_linear", first.predicted, residuals, "Residuals");
        if (!img.empty()) {
            report.addImage("Residual Plot (Linear)", img);
        }
    }
}

void runNeuralAndReport(ReportEngine& report, GnuplotEngine* plotter, const TypedDataset& data,
                        int targetIdx, const std::vector<int>& featureIdx, bool binaryTarget) {
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
    hp.verbose = false;

    nn.train(Xnn, Ynn, hp);
    nn.saveModelBinary("seldon_model.seldon");

    report.addTable("Neural Network Auto-Defaults", {"Setting", "Value"}, {
        {"Input Nodes", std::to_string(inputNodes)},
        {"Hidden Nodes", std::to_string(hidden)},
        {"Output Nodes", std::to_string(outputNodes)},
        {"Loss", binaryTarget ? "cross_entropy" : "mse"},
        {"Activation", "relu"}
    });

    if (plotter) {
        const auto& trainLoss = nn.getTrainLossHistory();
        const auto& valLoss = nn.getValLossHistory();
        std::vector<double> epochs(trainLoss.size(), 0.0);
        for (size_t i = 0; i < epochs.size(); ++i) epochs[i] = static_cast<double>(i + 1);

        std::string trainImg = plotter->line("nn_train_loss", epochs, trainLoss, "NN Train Loss");
        if (!trainImg.empty()) report.addImage("Training Loss Curve", trainImg);

        if (!valLoss.empty()) {
            std::string valImg = plotter->line("nn_val_loss", epochs, valLoss, "NN Validation Loss");
            if (!valImg.empty()) report.addImage("Validation Loss Curve", valImg);
        }
    }
}
} // namespace

int AutomationPipeline::run(const AutoConfig& config) {
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

    PreprocessReport prep = Preprocessor::run(data, config);

    ReportEngine report;
    report.addTitle("Seldon Automated Analytics Report");
    report.addParagraph("Dataset: " + config.datasetPath);
    report.addParagraph("Rows: " + std::to_string(data.rowCount()) + " | Columns: " + std::to_string(data.colCount()));

    addSummarySection(report, data, prep);

    GnuplotEngine plotter(config.assetsDir, config.plot);
    bool canPlot = plotter.isAvailable();
    report.addParagraph(canPlot ? "Gnuplot detected: plots generated." : "Gnuplot not available: plot generation skipped.");

    if (canPlot) {
        addPlotSections(report, plotter, data, prep);
    }

    std::vector<int> featureIdx = collectFeatureIndices(data, targetIdx, config);
    auto benchmarks = BenchmarkEngine::run(data, targetIdx, featureIdx, config.kfold);
    addBenchmarkSection(report, canPlot ? &plotter : nullptr, benchmarks);

    runNeuralAndReport(report, canPlot ? &plotter : nullptr, data, targetIdx, featureIdx, targetBinaryOriginal);

    report.save(config.reportFile);

    std::cout << "[Seldon] Automated pipeline complete.\n";
    std::cout << "[Seldon] Report: " << config.reportFile << "\n";
    std::cout << "[Seldon] Assets: " << config.assetsDir << "\n";

    return 0;
}
