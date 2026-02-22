#include "GnuplotEngine.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <cctype>

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

std::string GnuplotEngine::terminalForFormat(const std::string& format, int width, int height) {
    if (format == "svg") return "svg size " + std::to_string(width) + "," + std::to_string(height);
    if (format == "pdf") return "pdfcairo size 11in,8in";
    return "pngcairo size " + std::to_string(width) + "," + std::to_string(height);
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

    std::ofstream sout(scriptFile);
    if (!sout) return "";
    sout << scriptContent;
    if (!sout.good()) return "";

    std::string cmd = "gnuplot \"" + scriptFile + "\" > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !std::filesystem::exists(outputFile)) return "";
    return outputFile;
}

std::string GnuplotEngine::histogram(const std::string& id, const std::vector<double>& values, const std::string& title) {
    std::ostringstream data;
    for (double v : values) data << v << "\n";

    std::ostringstream script;
        const std::string safeId = sanitizeId(id);
        script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << "\n";
        script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
        script << "set title " << quoteForGnuplot(title) << "\n";
    script << "set boxwidth 0.9 relative\nset style fill solid 1.0\n";
    script << "binwidth=1\nbin(x,width)=width*floor(x/width)\n";
    script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using (bin($1,binwidth)):(1.0) smooth freq with boxes notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::bar(const std::string& id, const std::vector<std::string>& labels, const std::vector<double>& values, const std::string& title) {
    std::ostringstream data;
    for (size_t i = 0; i < labels.size() && i < values.size(); ++i) data << i << " " << values[i] << "\n";

    std::ostringstream script;
        const std::string safeId = sanitizeId(id);
        script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << "\n";
        script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
        script << "set title " << quoteForGnuplot(title) << "\nset style data histograms\nset style fill solid\n";
        script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 2:xtic(1) with boxes notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::scatter(const std::string& id, const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    std::ostringstream data;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) data << x[i] << " " << y[i] << "\n";

    std::ostringstream script;
        const std::string safeId = sanitizeId(id);
        script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << "\n";
        script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
        script << "set title " << quoteForGnuplot(title) << "\n";
        script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with points pt 7 ps 0.6 notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::line(const std::string& id, const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    std::ostringstream data;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) data << x[i] << " " << y[i] << "\n";

    std::ostringstream script;
        const std::string safeId = sanitizeId(id);
        script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << "\n";
        script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
        script << "set title " << quoteForGnuplot(title) << "\n";
        script << "plot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2 with lines lw 2 notitle\n";
    return runScript(id, data.str(), script.str());
}

std::string GnuplotEngine::heatmap(const std::string& id, const std::vector<std::vector<double>>& matrix, const std::string& title) {
    std::ostringstream data;
    for (size_t r = 0; r < matrix.size(); ++r) {
        for (size_t c = 0; c < matrix[r].size(); ++c) {
            data << c << " " << r << " " << matrix[r][c] << "\n";
        }
        data << "\n";
    }

    std::ostringstream script;
        const std::string safeId = sanitizeId(id);
        script << "set terminal " << terminalForFormat(cfg_.format, cfg_.width, cfg_.height) << "\n";
        script << "set output " << quoteForGnuplot(assetsDir_ + "/" + safeId + "." + cfg_.format) << "\n";
        script << "set title " << quoteForGnuplot(title) << "\nset view map\nset pm3d at b\nunset key\n";
        script << "splot " << quoteForGnuplot(assetsDir_ + "/" + safeId + ".dat") << " using 1:2:3 with pm3d\n";
    return runScript(id, data.str(), script.str());
}
