#include "ReportEngine.h"
#include <fstream>
#include <sstream>
#include <algorithm>

void ReportEngine::addTitle(const std::string& title) {
    body_ += "# " + title + "\n\n";
}

void ReportEngine::addParagraph(const std::string& text) {
    body_ += text + "\n\n";
}

void ReportEngine::addTable(const std::string& title, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
    body_ += "## " + title + "\n";
    if (headers.empty()) {
        body_ += "(no columns)\n\n";
        return;
    }

    std::vector<size_t> widths(headers.size(), 0);
    for (size_t i = 0; i < headers.size(); ++i) {
        widths[i] = headers[i].size();
    }

    for (const auto& row : rows) {
        for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }

    auto appendRow = [&](const std::vector<std::string>& row) {
        for (size_t i = 0; i < headers.size(); ++i) {
            std::string cell = (i < row.size() ? row[i] : "");
            body_ += cell;
            if (widths[i] > cell.size()) body_ += std::string(widths[i] - cell.size(), ' ');
            body_ += (i + 1 == headers.size() ? "\n" : " | ");
        }
    };

    appendRow(headers);
    for (size_t i = 0; i < headers.size(); ++i) {
        body_ += std::string(widths[i], '-');
        body_ += (i + 1 == headers.size() ? "\n" : "-+-");
    }
    for (const auto& row : rows) appendRow(row);
    body_ += "\n";
}

void ReportEngine::addImage(const std::string& title, const std::string& imagePath) {
    body_ += "[PLOT] " + title + " => " + imagePath + "\n";
}

void ReportEngine::save(const std::string& filePath) const {
    std::ofstream out(filePath);
    out << body_;
}
