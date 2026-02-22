#include "ReportEngine.h"
#include <fstream>
#include <sstream>

std::string ReportEngine::escapeHtml(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

void ReportEngine::addTitle(const std::string& title) {
    body_ += "<h1>" + escapeHtml(title) + "</h1>\n";
}

void ReportEngine::addParagraph(const std::string& text) {
    body_ += "<p>" + escapeHtml(text) + "</p>\n";
}

void ReportEngine::addTable(const std::string& title, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
    body_ += "<h2>" + escapeHtml(title) + "</h2>\n<table><thead><tr>";
    for (const auto& h : headers) body_ += "<th>" + escapeHtml(h) + "</th>";
    body_ += "</tr></thead><tbody>";
    for (const auto& row : rows) {
        body_ += "<tr>";
        for (const auto& cell : row) body_ += "<td>" + escapeHtml(cell) + "</td>";
        body_ += "</tr>";
    }
    body_ += "</tbody></table>\n";
}

void ReportEngine::addImage(const std::string& title, const std::string& imagePath) {
    body_ += "<h3>" + escapeHtml(title) + "</h3><img src='" + escapeHtml(imagePath) + "' style='max-width:100%;height:auto;'/>\n";
}

void ReportEngine::save(const std::string& filePath) const {
    std::ofstream out(filePath);
    out << "<!doctype html><html><head><meta charset='utf-8'/>"
        << "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}"
        << "th,td{border:1px solid #ddd;padding:8px;}th{background:#f5f5f5;text-align:left;}"
        << "h1,h2,h3{margin-top:24px;}</style></head><body>";
    out << body_;
    out << "</body></html>";
}
