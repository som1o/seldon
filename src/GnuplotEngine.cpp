#include "GnuplotEngine.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

std::string GnuplotEngine::sanitizeId(const std::string& id) {
    std::string out = id;
    std::replace_if(out.begin(), out.end(), [](unsigned char c) {
        return !(std::isalnum(c) || c == '_' || c == '-');
    }, '_');
    if (out.empty()) out = "plot";
    return out;
}

std::string GnuplotEngine::quoteForGnuplot(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('\'');
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string normalizePlotLabel(const std::string& label, size_t maxLen = 28) {
    std::string out;
    out.reserve(label.size());
    for (unsigned char ch : label) {
        if (ch >= 32 && ch <= 126) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('?');
        }
    }
    if (out.size() > maxLen) {
        out = out.substr(0, maxLen - 3) + "...";
    }
    return out;
}

std::string quoteForDatafileString(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\\\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string GnuplotEngine::terminalForFormat(const std::string& format, int width, int height) {
    if (format == "svg") return "svg size " + std::to_string(width) + "," + std::to_string(height);
    if (format == "pdf") return "pdfcairo size 11in,8in";
    return "pngcairo size " + std::to_string(width) + "," + std::to_string(height);
}

std::string GnuplotEngine::styledHeader(const std::string& id, const std::string& title) const {
    const std::string safeId = sanitizeId(id);
    const bool darkTheme = (cfg_.theme == "dark");
    const std::string titleColor = darkTheme ? "#f9fafb" : "#1f2937";
    const std::string borderColor = darkTheme ? "#6b7280" : "#9ca3af";
    const std::string ticColor = darkTheme ? "#e5e7eb" : "#374151";
    const std::string gridColor = darkTheme ? "#374151" : "#e5e7eb";
    const std::string bgColor = darkTheme ? "#111827" : "#ffffff";

    std::ostringstream script;
    script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << " enhanced\n";
    script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
    script << "set object 999 rect from graph 0,0 to graph 1,1 behind fc rgb " << quoteForGnuplot(bgColor) << " fs solid 1.0 noborder\n";
    script << "set title " << quoteForGnuplot(title) << " tc rgb " << quoteForGnuplot(titleColor) << "\n";
    script << "set border linewidth " << cfg_.lineWidth << " lc rgb " << quoteForGnuplot(borderColor) << "\n";
    script << "set tics textcolor rgb " << quoteForGnuplot(ticColor) << "\n";
    if (cfg_.showGrid) {
        script << "set grid back lc rgb " << quoteForGnuplot(gridColor) << " lw 1\n";
    } else {
        script << "unset grid\n";
    }
    script << "set key top left\n";
    script << "set style line 1 lc rgb '#2563eb' lw " << cfg_.lineWidth << " pt 7 ps " << cfg_.pointSize << "\n";
    script << "set style line 2 lc rgb '#dc2626' lw " << cfg_.lineWidth << "\n";
    script << "set style line 3 lc rgb '#059669' lw " << cfg_.lineWidth << "\n";
    script << "set style line 4 lc rgb '#7c3aed' lw " << cfg_.lineWidth << "\n";
    return script.str();
}

GnuplotEngine::GnuplotEngine(std::string assetsDir, PlotConfig cfg)
    : assetsDir_(std::move(assetsDir)), cfg_(std::move(cfg)) {
    std::filesystem::create_directories(assetsDir_);
}

bool GnuplotEngine::isAvailable() const {
    int code = std::system("command -v gnuplot > /dev/null 2>&1");
    return code == 0;
}

std::string GnuplotEngine::runScript(const std::string& id, const std::string& dataContent, const std::string& scriptContent) {
    const std::string safeId = sanitizeId(id);
    const std::string dataFile = assetsDir_ + "/" + safeId + ".dat";
    const std::string scriptFile = assetsDir_ + "/" + safeId + ".plt";
    const std::string outputFile = assetsDir_ + "/" + safeId + "." + cfg_.format;

    std::ofstream dout(dataFile);
    if (!dout) return "";
    dout << dataContent;
    if (!dout.good()) return "";
    dout.close();

    std::ofstream sout(scriptFile);
    if (!sout) return "";
    sout << scriptContent;
    if (!sout.good()) return "";
    sout.close();

    std::string cmd = "gnuplot \"" + scriptFile + "\"";
    int rc = std::system(cmd.c_str());

    std::error_code ec;
    std::filesystem::remove(dataFile, ec);
    std::filesystem::remove(scriptFile, ec);

    if (rc != 0 || !std::filesystem::exists(outputFile)) {
        std::cerr << "[Seldon][Plot] Generation failed for id='" << safeId
                  << "' rc=" << rc
                  << " output='" << outputFile << "'\n";
        return "";
    }

    return outputFile;
}

std::string GnuplotEngine::histogram(const std::string& id, const std::vector<double>& values, const std::string& title) {
    if (values.empty()) return "";

    std::ostringstream data;
    for (double v : values) data << v << "\n";

    double minV = 0.0;
    double maxV = 0.0;
    if (!values.empty()) {
        auto mm = std::minmax_element(values.begin(), values.end());
        minV = *mm.first;
        maxV = *mm.second;
    }
    double range = maxV - minV;
    double binwidth = 1.0;
    if (range > 1e-12) {
        binwidth = range / 20.0;
        if (binwidth < 1e-6) binwidth = 1e-6;
    }

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set xlabel 'Value' tc rgb '#374151'\n";
    script << "set ylabel 'Frequency' tc rgb '#374151'\n";
    script << "set yrange [0:*]\n";
    if (range <= 1e-12) {
        script << "set xrange [" << (minV - 1.0) << ":" << (maxV + 1.0) << "]\n";
    }
    script << "set boxwidth 0.9 relative\nset style fill solid 0.85 border lc rgb '#1d4ed8'\n";
    script << "binwidth=" << binwidth << "\nbin(x,width)=width*floor(x/width)\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using (bin($1,binwidth)):(1.0) smooth freq with boxes ls 1 title 'Histogram'\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::bar(const std::string& id, const std::vector<std::string>& labels, const std::vector<double>& values, const std::string& title) {
    std::ostringstream data;
    double maxValue = 0.0;
    for (size_t i = 0; i < labels.size() && i < values.size(); ++i) {
        if (values[i] > maxValue) maxValue = values[i];
        data << i << " " << values[i] << "\n";
    }
    if (labels.empty() || values.empty()) return "";
    if (maxValue < 1.0) maxValue = 1.0;

    std::ostringstream xtics;
    xtics << "set xtics (";
    for (size_t i = 0; i < labels.size() && i < values.size(); ++i) {
        if (i > 0) xtics << ", ";
        xtics << quoteForGnuplot(normalizePlotLabel(labels[i])) << " " << i;
    }
    xtics << ") rotate by -25 font ',8'\n";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set style fill solid 0.85 border lc rgb '#1d4ed8'\n";
    script << "set boxwidth 0.8\nset yrange [0:" << (maxValue * 1.12) << "]\n";
    script << "set xlabel ''\nset ylabel 'Count' tc rgb '#374151'\n";
    script << xtics.str();
    script << "set xrange [-1:" << (labels.size()) << "]\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with boxes ls 1 notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::stackedBar(const std::string& id,
                                      const std::vector<std::string>& categories,
                                      const std::vector<std::string>& seriesLabels,
                                      const std::vector<std::vector<double>>& seriesValues,
                                      const std::string& title,
                                      const std::string& yLabel) {
    const size_t c = categories.size();
    const size_t s = seriesLabels.size();
    if (c == 0 || s == 0 || seriesValues.size() != s) return "";
    for (const auto& series : seriesValues) {
        if (series.size() != c) return "";
    }

    std::ostringstream data;
    double maxStack = 0.0;
    for (size_t i = 0; i < c; ++i) {
        data << quoteForDatafileString(normalizePlotLabel(categories[i], 20));
        double stackSum = 0.0;
        for (size_t j = 0; j < s; ++j) {
            const double v = std::max(0.0, std::isfinite(seriesValues[j][i]) ? seriesValues[j][i] : 0.0);
            stackSum += v;
            data << " " << v;
        }
        data << "\n";
        if (stackSum > maxStack) maxStack = stackSum;
    }
    if (maxStack <= 0.0) return "";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set style data histograms\n";
    script << "set style histogram rowstacked\n";
    script << "set style fill solid 0.88 border lc rgb '#ffffff'\n";
    script << "set boxwidth 0.82\n";
    script << "set ylabel " << quoteForGnuplot(yLabel) << " tc rgb '#374151'\n";
    script << "set yrange [0:" << (maxStack * 1.15) << "]\n";
    script << "set xtics rotate by -25 font ',8'\n";
    script << "set key top right\n";
    script << "plot ";
    for (size_t j = 0; j < s; ++j) {
        if (j > 0) script << ", ";
        script << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
               << " using " << (j + 2) << ":xtic(1) title " << quoteForGnuplot(normalizePlotLabel(seriesLabels[j], 24))
               << " ls " << ((j % 4) + 1);
    }
    script << "\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::scatter(const std::string& id,
                                   const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const std::string& title,
                                   bool withFitLine,
                                   double fitSlope,
                                   double fitIntercept,
                                   const std::string& fitLabel) {
    if (x.empty() || y.empty()) return "";

    std::ostringstream data;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) data << x[i] << " " << y[i] << "\n";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set xlabel 'X' tc rgb '#374151'\n";
    script << "set ylabel 'Y' tc rgb '#374151'\n";
    script << "set key top left\n";
    script << "f(x)=" << fitSlope << "*x+" << fitIntercept << "\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using 1:2 with points ls 1 title 'Observations'";
    if (withFitLine) {
        script << ", f(x) with lines ls 2 title " << quoteForGnuplot(fitLabel);
    }
    script << "\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::line(const std::string& id, const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    if (x.empty() || y.empty()) return "";

    std::ostringstream data;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) data << x[i] << " " << y[i] << "\n";

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set xlabel 'Epoch' tc rgb '#374151'\n";
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with lines ls 1 title 'Series'\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::ogive(const std::string& id, const std::vector<double>& values, const std::string& title) {
    if (values.size() < 2) return "";

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream data;
    const size_t n = sorted.size();
    for (size_t i = 0; i < n; ++i) {
        const double cumulativePct = (100.0 * static_cast<double>(i + 1)) / static_cast<double>(n);
        data << sorted[i] << " " << cumulativePct << "\n";
    }

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel 'Value' tc rgb '#374151'\n";
    script << "set ylabel 'Cumulative %' tc rgb '#374151'\n";
    script << "set yrange [0:100]\n";
    script << "set key top left\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
            << " using 1:2 with linespoints ls 3 pt 7 ps " << std::max(0.1, cfg_.pointSize * 0.65) << " title 'Ogive'\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::box(const std::string& id, const std::vector<double>& values, const std::string& title) {
    if (values.size() < 4) return "";

    std::ostringstream data;
    for (double v : values) data << v << "\n";

    const std::string safeId = sanitizeId(id);
    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xlabel ''\n";
    script << "set ylabel 'Value' tc rgb '#374151'\n";
    script << "set style data boxplot\n";
    script << "set style boxplot outliers pointtype 7\n";
    script << "set style fill solid 0.45 border lc rgb '#1d4ed8'\n";
    script << "set boxwidth 0.35\n";
    script << "set xrange [0.5:1.5]\n";
    script << "set xtics (" << quoteForGnuplot("Distribution") << " 1)\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat")
           << " using (1):1 title 'Box Plot' lc rgb '#2563eb'\n";

    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::pie(const std::string& id,
                               const std::vector<std::string>& labels,
                               const std::vector<double>& values,
                               const std::string& title) {
    const size_t n = std::min(labels.size(), values.size());
    if (n < 2) return "";

    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (values[i] > 0.0) total += values[i];
    }
    if (total <= 0.0) return "";

    const std::vector<std::string> palette = {
        "#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4f46e5"
    };

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "unset key\n";
    script << "unset border\n";
    script << "unset xtics\n";
    script << "unset ytics\n";
    script << "unset grid\n";
    script << "set size square\n";
    script << "set xrange [-1.25:1.25]\n";
    script << "set yrange [-1.25:1.25]\n";

    constexpr double kPi = 3.14159265358979323846;
    double angleStart = 0.0;
    int objectId = 1;
    int labelId = 1;

    for (size_t i = 0; i < n; ++i) {
        if (values[i] <= 0.0) continue;
        const double frac = values[i] / total;
        const double angleEnd = angleStart + frac * 2.0 * kPi;

        script << "set object " << objectId++ << " polygon from 0,0 ";
        const int seg = 36;
        for (int s = 0; s <= seg; ++s) {
            const double t = angleStart + (angleEnd - angleStart) * static_cast<double>(s) / static_cast<double>(seg);
            const double px = std::cos(t);
            const double py = std::sin(t);
            script << "to " << px << "," << py << " ";
        }
        script << "fs solid 0.93 border lc rgb '#ffffff' fc rgb "
               << quoteForGnuplot(palette[i % palette.size()]) << "\n";

        const double mid = 0.5 * (angleStart + angleEnd);
        const double lx = 1.12 * std::cos(mid);
        const double ly = 1.12 * std::sin(mid);
        const int pct = static_cast<int>(std::round(frac * 100.0));
        script << "set label " << labelId++ << " "
               << quoteForGnuplot(normalizePlotLabel(labels[i], 20) + " (" + std::to_string(pct) + "%)")
               << " at " << lx << "," << ly << " center tc rgb '#111827' font ',8'\n";

        angleStart = angleEnd;
    }

    script << "plot 1/0 notitle\n";
    return runScript(id, "", script.str());
}

std::string GnuplotEngine::gantt(const std::string& id,
                                 const std::vector<std::string>& taskNames,
                                 const std::vector<int64_t>& startUnixSeconds,
                                 const std::vector<int64_t>& endUnixSeconds,
                                 const std::string& title) {
    const size_t n = std::min(taskNames.size(), std::min(startUnixSeconds.size(), endUnixSeconds.size()));
    if (n == 0) return "";

    int64_t minStart = startUnixSeconds[0];
    int64_t maxEnd = endUnixSeconds[0];
    for (size_t i = 0; i < n; ++i) {
        minStart = std::min(minStart, startUnixSeconds[i]);
        maxEnd = std::max(maxEnd, endUnixSeconds[i]);
    }
    if (maxEnd <= minStart) return "";

    const std::vector<std::string> palette = {
        "#2563eb", "#059669", "#dc2626", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4f46e5"
    };

    std::ostringstream ytics;
    ytics << "set ytics (";
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) ytics << ", ";
        ytics << quoteForGnuplot(normalizePlotLabel(taskNames[i], 32)) << " " << (i + 1);
    }
    ytics << ")\n";

    std::ostringstream script;
    script << styledHeader(id, title);
    script << "set xdata time\n";
    script << "set timefmt '%s'\n";
    script << "set format x '%Y-%m-%d'\n";
    script << "set xtics rotate by -25\n";
    script << "set xlabel 'Timeline' tc rgb '#374151'\n";
    script << "set ylabel 'Task' tc rgb '#374151'\n";
    script << "set yrange [0.5:" << (static_cast<double>(n) + 0.5) << "]\n";
    script << ytics.str();
    script << "set xrange [" << quoteForGnuplot(std::to_string(minStart)) << ":"
           << quoteForGnuplot(std::to_string(maxEnd + 86400)) << "]\n";
    script << "unset key\n";

    int objectId = 1;
    for (size_t i = 0; i < n; ++i) {
        const double yLow = static_cast<double>(i + 1) - 0.35;
        const double yHigh = static_cast<double>(i + 1) + 0.35;
        script << "set object " << objectId++
               << " rect from " << quoteForGnuplot(std::to_string(startUnixSeconds[i])) << "," << yLow
               << " to " << quoteForGnuplot(std::to_string(endUnixSeconds[i])) << "," << yHigh
               << " fc rgb " << quoteForGnuplot(palette[i % palette.size()])
               << " fs solid 0.85 border lc rgb '#ffffff'\n";
    }

    script << "plot 1/0 notitle\n";
    return runScript(id, "", script.str());
}

std::string GnuplotEngine::heatmap(const std::string& id, const std::vector<std::vector<double>>& matrix, const std::string& title) {
    if (matrix.empty() || matrix.front().empty()) return "";

    std::ostringstream data;
    for (size_t r = 0; r < matrix.size(); ++r) {
        for (size_t c = 0; c < matrix[r].size(); ++c) {
            data << c << " " << r << " " << matrix[r][c] << "\n";
        }
        data << "\n";
    }

    std::ostringstream script;
    const std::string safeId = sanitizeId(id);
    script << styledHeader(id, title);
    script << "set view map\nset pm3d at b\nunset key\n";
    script << "set palette defined (-1 '#1e3a8a', 0 '#f3f4f6', 1 '#991b1b')\n";
    script << "set cbrange [-1:1]\n";
    script << "unset grid\n";
    script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2:3 with pm3d\n";
    return runScript(id, data.str(), script.str());
}
