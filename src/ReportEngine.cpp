#include "ReportEngine.h"
#include "SeldonExceptions.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
std::string escapeHtml(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 16);
    for (char ch : value) {
        switch (ch) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\n': escaped += "<br>"; break;
            case '\r': break;
            default: escaped.push_back(ch); break;
        }
    }
    return escaped;
}

void appendHtmlTable(std::string& body,
                     const std::vector<std::string>& headers,
                     const std::vector<std::vector<std::string>>& rows) {
    body += "<div style=\"overflow-x:auto; max-width:100%;\">\n";
    body += "<table>\n";
    body += "  <thead>\n";
    body += "    <tr>\n";
    for (const auto& h : headers) {
        body += "      <th>" + escapeHtml(h) + "</th>\n";
    }
    body += "    </tr>\n";
    body += "  </thead>\n";
    body += "  <tbody>\n";
    for (const auto& row : rows) {
        body += "    <tr>\n";
        for (size_t i = 0; i < headers.size(); ++i) {
            body += "      <td>" + escapeHtml(i < row.size() ? row[i] : "") + "</td>\n";
        }
        body += "    </tr>\n";
    }
    body += "  </tbody>\n";
    body += "</table>\n";
    body += "</div>\n\n";
}

std::string buildStandaloneHtml(const std::string& body) {
    std::ostringstream out;
    out << "<!doctype html><html><head><meta charset=\"utf-8\"/>"
        << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
        << "<title>Seldon Report</title>"
        << "<style>body{font-family:Inter,system-ui,sans-serif;line-height:1.45;margin:0;background:#0d1018;color:#c4cad6;}"
        << "main{max-width:1220px;margin:0 auto;padding:18px;}h1,h2,h3,h4{color:#e8940a;}"
        << "table{border-collapse:collapse;width:100%;}th,td{border:1px solid rgba(255,255,255,.08);padding:4px 6px;}"
        << "img{max-width:100%;height:auto;display:block;background:#0d1018;margin:6px 0;}"
        << "p,li{color:#c4cad6;}code{background:rgba(255,255,255,.08);padding:1px 4px;}</style></head><body><main>"
        << body
        << "</main></body></html>";
    return out.str();
}
} // namespace

void ReportEngine::addTitle(const std::string& title) {
    body_ += "<h1>" + escapeHtml(title) + "</h1>\n\n";
}

void ReportEngine::addParagraph(const std::string& text) {
    body_ += "<p>" + escapeHtml(text) + "</p>\n\n";
}

void ReportEngine::addTable(const std::string& title,
                            const std::vector<std::string>& headers,
                            const std::vector<std::vector<std::string>>& rows) {
    body_ += "<h2>" + escapeHtml(title) + "</h2>\n";
    if (headers.empty()) {
        body_ += "<p><em>(no columns)</em></p>\n\n";
        return;
    }
    appendHtmlTable(body_, headers, rows);
}

void ReportEngine::addImage(const std::string& title, const std::string& imagePath) {
    body_ += "<h3>" + escapeHtml(title) + "</h3>\n";
    body_ += "<img src=\"" + escapeHtml(imagePath) + "\" alt=\"" + escapeHtml(title) + "\"/>\n\n";
}

std::string ReportEngine::renderBody() const {
    return body_;
}

void ReportEngine::save(const std::string& filePath) const {
    const std::filesystem::path outPath(filePath);
    const std::filesystem::path parent = outPath.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            throw Seldon::IOException("Unable to create report directory '" + parent.string() + "': " + ec.message());
        }
    }

    const bool htmlOut = outPath.extension() == ".html";
    const std::string payload = htmlOut ? buildStandaloneHtml(body_) : body_;
    if (payload.size() > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw Seldon::IOException("Report payload exceeds stream write bounds for: " + filePath);
    }

    std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw Seldon::IOException("Unable to open report output file: " + filePath);
    }

    std::vector<char> ioBuffer(1 << 20, '\0');
    out.rdbuf()->pubsetbuf(ioBuffer.data(), static_cast<std::streamsize>(ioBuffer.size()));
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!out.good()) {
        throw Seldon::IOException("Failed to write report output file: " + filePath);
    }

    out.flush();
    if (!out.good()) {
        throw Seldon::IOException("Failed to flush report output file: " + filePath);
    }
}
