#include "ReportEngine.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace {
std::string escapeMarkdownTableCell(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char ch : value) {
        if (ch == '|') {
            escaped += "\\|";
        } else if (ch == '\n') {
            escaped += "<br>";
        } else if (ch != '\r') {
            escaped.push_back(ch);
        }
    }
    return escaped;
}
} // namespace

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

    body_ += "|";
    for (const auto& h : headers) {
        body_ += " " + escapeMarkdownTableCell(h) + " |";
    }
    body_ += "\n|";
    for (size_t i = 0; i < headers.size(); ++i) {
        body_ += " --- |";
    }
    body_ += "\n";

    for (const auto& row : rows) {
        body_ += "|";
        for (size_t i = 0; i < headers.size(); ++i) {
            body_ += " " + escapeMarkdownTableCell(i < row.size() ? row[i] : "") + " |";
        }
        body_ += "\n";
    }
    body_ += "\n";
}

void ReportEngine::addImage(const std::string& title, const std::string& imagePath) {
    std::string portablePath = imagePath;
    std::error_code ec;
    const std::filesystem::path rel = std::filesystem::relative(std::filesystem::path(imagePath), std::filesystem::current_path(), ec);
    if (!ec && !rel.empty()) {
        portablePath = rel.string();
    }
    body_ += "### " + title + "\n";
    body_ += "![" + title + "](" + portablePath + ")\n\n";
}

void ReportEngine::save(const std::string& filePath) const {
    std::ofstream out(filePath);
    out << body_;
}
