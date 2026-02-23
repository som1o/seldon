#pragma once
#include "AutoConfig.h"
#include <cstdint>
#include <string>
#include <vector>

class GnuplotEngine {
public:
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
    std::string histogram(const std::string& id, const std::vector<double>& values, const std::string& title);

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
                        const std::string& fitLabel = "Linear fit");

    /**
     * @brief Generates line plot image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string line(const std::string& id, const std::vector<double>& x, const std::vector<double>& y, const std::string& title);

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
                      const std::string& title);

    /**
     * @brief Generates heatmap image.
     * @post Returns output image path, or empty string on generation failure.
     */
    std::string heatmap(const std::string& id, const std::vector<std::vector<double>>& matrix, const std::string& title);

private:
    std::string assetsDir_;
    PlotConfig cfg_;

    static std::string sanitizeId(const std::string& id);
    static std::string quoteForGnuplot(const std::string& value);
    static std::string terminalForFormat(const std::string& format, int width, int height);
    std::string styledHeader(const std::string& id, const std::string& title) const;
    std::string runScript(const std::string& id, const std::string& dataContent, const std::string& scriptContent);
};
