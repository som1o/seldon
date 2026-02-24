#pragma once
#include "AutoConfig.h"
#include <cstdint>
#include <string>
#include <vector>

class GnuplotEngine {
public:
    struct ScatterOptions {
        bool withFitLine = false;
        double fitSlope = 0.0;
        double fitIntercept = 0.0;
        std::string fitLabel = "Linear fit";
        bool withConfidenceBand = false;
        double confidenceZ = 1.96;
        size_t downsampleThreshold = 10000;
    };

    struct HistogramOptions {
        bool adaptiveBinning = true;
        bool withDensityOverlay = true;
    };

    /**
     * @brief Initializes plotting backend and asset directory.
     * @post assets directory is created if possible.
     */
    GnuplotEngine(std::string assetsDir, PlotConfig cfg);

    /**
     * @brief Checks whether gnuplot executable is available in PATH.
     */
    bool isAvailable() const;

    /**
     * @brief Generates histogram plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string histogram(const std::string& id,
                          const std::vector<double>& values,
                          const std::string& title);

    std::string histogram(const std::string& id,
                          const std::vector<double>& values,
                          const std::string& title,
                          const HistogramOptions& options);

    /**
     * @brief Generates bar plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string bar(const std::string& id, const std::vector<std::string>& labels, const std::vector<double>& values, const std::string& title);

    /**
     * @brief Generates stacked bar plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string stackedBar(const std::string& id,
                           const std::vector<std::string>& categories,
                           const std::vector<std::string>& seriesLabels,
                           const std::vector<std::vector<double>>& seriesValues,
                           const std::string& title,
                           const std::string& yLabel = "Count");

    /**
     * @brief Generates scatter plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string scatter(const std::string& id,
                        const std::vector<double>& x,
                        const std::vector<double>& y,
                        const std::string& title,
                        bool withFitLine = false,
                        double fitSlope = 0.0,
                        double fitIntercept = 0.0,
                        const std::string& fitLabel = "Linear fit",
                        bool withConfidenceBand = false,
                        double confidenceZ = 1.96,
                        size_t downsampleThreshold = 10000);

    /**
     * @brief Generates residual-vs-fitted plot image.
     */
    std::string residual(const std::string& id,
                         const std::vector<double>& fitted,
                         const std::vector<double>& residuals,
                         const std::string& title);

    /**
     * @brief Generates line plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string line(const std::string& id, const std::vector<double>& x, const std::vector<double>& y, const std::string& title);

    /**
     * @brief Generates multi-series line plot image sharing the same x-axis.
     */
    std::string multiLine(const std::string& id,
                          const std::vector<double>& x,
                          const std::vector<std::vector<double>>& series,
                          const std::vector<std::string>& labels,
                          const std::string& title,
                          const std::string& yLabel = "Value");

    /**
     * @brief Generates auto-selected violin-like or boxen-like plot for categorical-numeric distributions.
     */
    std::string categoricalDistribution(const std::string& id,
                                        const std::vector<std::string>& categories,
                                        const std::vector<double>& values,
                                        const std::string& title);

    /**
     * @brief Generates faceted scatter plot split by category.
     */
    std::string facetedScatter(const std::string& id,
                               const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::vector<std::string>& facets,
                               const std::string& title,
                               size_t maxFacets = 6);

    /**
     * @brief Generates parallel coordinates plot for normalized numeric dimensions.
     */
    std::string parallelCoordinates(const std::string& id,
                                    const std::vector<std::vector<double>>& matrix,
                                    const std::vector<std::string>& axisLabels,
                                    const std::string& title);

    /**
     * @brief Generates time-series plot with automatic trend-line selection.
     */
    std::string timeSeriesTrend(const std::string& id,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::string& title,
                                bool xIsUnixTime = false);

    /**
     * @brief Generates ogive (cumulative frequency) plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string ogive(const std::string& id, const std::vector<double>& values, const std::string& title);

    /**
     * @brief Generates a single-column box plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string box(const std::string& id, const std::vector<double>& values, const std::string& title);

    /**
     * @brief Generates pie chart image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string pie(const std::string& id,
                    const std::vector<std::string>& labels,
                    const std::vector<double>& values,
                    const std::string& title);

    /**
     * @brief Generates lightweight Gantt chart image from task windows.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string gantt(const std::string& id,
                      const std::vector<std::string>& taskNames,
                      const std::vector<int64_t>& startUnixSeconds,
                      const std::vector<int64_t>& endUnixSeconds,
                      const std::vector<std::string>& semantics,
                      const std::string& title);

    /**
     * @brief Generates heatmap image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string heatmap(const std::string& id,
                        const std::vector<std::vector<double>>& matrix,
                        const std::string& title,
                        const std::vector<std::string>& labels = {});

private:
    std::string assetsDir_;
    PlotConfig cfg_;

    static std::string sanitizeId(const std::string& id);
    static std::string quoteForGnuplot(const std::string& value);
    static std::string terminalForFormat(const std::string& format, int width, int height);
    std::string styledHeader(const std::string& id, const std::string& title) const;
    std::string runScript(const std::string& id, const std::string& dataContent, const std::string& scriptContent);
};
