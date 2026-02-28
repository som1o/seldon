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

constexpr size_t kTallTableRowCap = 120;

void appendMarkdownTable(std::string& body,
                         const std::vector<std::string>& headers,
                         const std::vector<std::vector<std::string>>& rows) {
    body += "|";
    for (const auto& h : headers) {
        body += " " + escapeMarkdownTableCell(h) + " |";
    }
    body += "\n|";
    for (size_t i = 0; i < headers.size(); ++i) {
        body += " --- |";
    }
    body += "\n";

    for (const auto& row : rows) {
        body += "|";
        for (size_t i = 0; i < headers.size(); ++i) {
            body += " " + escapeMarkdownTableCell(i < row.size() ? row[i] : "") + " |";
        }
        body += "\n";
    }
    body += "\n";
}

void appendWideHtmlTable(std::string& body,
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

bool hasUriScheme(const std::string& target) {
    const size_t colon = target.find(':');
    if (colon == std::string::npos) return false;
    if (colon == 0) return false;
    for (size_t i = 0; i < colon; ++i) {
        const char ch = target[i];
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') ||
                        ch == '+' || ch == '-' || ch == '.';
        if (!ok) return false;
    }
    return true;
}

bool isExternalOrAnchorLink(const std::string& target) {
    if (target.empty()) return true;
    if (target[0] == '#') return true;
    if (hasUriScheme(target)) return true;
    return false;
}

std::string toPortablePathString(const std::filesystem::path& path) {
    return path.generic_string();
}

std::string normalizeLocalLinkTarget(const std::string& target,
                                     const std::filesystem::path& reportDir) {
    if (target.empty() || isExternalOrAnchorLink(target)) return target;

    std::string base = target;
    std::string suffix;
    const size_t fragPos = base.find('#');
    if (fragPos != std::string::npos) {
        suffix = base.substr(fragPos);
        base = base.substr(0, fragPos);
    }
    if (base.empty()) return target;

    std::error_code ec;
    const std::filesystem::path rawPath(base);
    if (rawPath.is_absolute()) {
        const std::filesystem::path rel = std::filesystem::relative(rawPath, reportDir, ec);
        if (!ec && !rel.empty()) return toPortablePathString(rel) + suffix;
        return toPortablePathString(rawPath) + suffix;
    }

    const std::filesystem::path reportRelative = reportDir / rawPath;
    if (std::filesystem::exists(reportRelative, ec) && !ec) {
        return toPortablePathString(rawPath) + suffix;
    }

    ec.clear();
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (ec) return target;

    const std::filesystem::path cwdRelative = cwd / rawPath;
    if (std::filesystem::exists(cwdRelative, ec) && !ec) {
        ec.clear();
        const std::filesystem::path rel = std::filesystem::relative(cwdRelative, reportDir, ec);
        if (!ec && !rel.empty()) return toPortablePathString(rel) + suffix;
    }

    return toPortablePathString(rawPath) + suffix;
}

std::string normalizeMarkdownLinksForReport(const std::string& markdown,
                                            const std::filesystem::path& reportDir) {
    std::string out;
    out.reserve(markdown.size() + 64);

    size_t cursor = 0;
    while (true) {
        const size_t start = markdown.find("](", cursor);
        if (start == std::string::npos) {
            out.append(markdown.substr(cursor));
            break;
        }

        const size_t targetStart = start + 2;
        const size_t end = markdown.find(')', targetStart);
        if (end == std::string::npos) {
            out.append(markdown.substr(cursor));
            break;
        }

        out.append(markdown.substr(cursor, targetStart - cursor));
        const std::string target = markdown.substr(targetStart, end - targetStart);
        out.append(normalizeLocalLinkTarget(target, reportDir));
        out.push_back(')');
        cursor = end + 1;
    }

    return out;
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

    const bool wideTable = headers.size() >= 10;
    const bool tallTable = rows.size() > kTallTableRowCap;

    if (wideTable) {
        body_ += "_Wide table rendered in a scrollable block for readability._\n\n";
    }
    if (tallTable) {
        body_ += "_Tall table preview shown (" + std::to_string(kTallTableRowCap) + " of " + std::to_string(rows.size()) + " rows)._\n\n";
    }

    const size_t previewCount = tallTable ? kTallTableRowCap : rows.size();
    const std::vector<std::vector<std::string>> previewRows(rows.begin(), rows.begin() + static_cast<long>(previewCount));

    if (wideTable) {
        appendWideHtmlTable(body_, headers, previewRows);
    } else {
        appendMarkdownTable(body_, headers, previewRows);
    }

    if (tallTable) {
        body_ += "<details>\n";
        body_ += "<summary>Show full table (" + std::to_string(rows.size()) + " rows)</summary>\n\n";
        if (wideTable) {
            appendWideHtmlTable(body_, headers, rows);
        } else {
            appendMarkdownTable(body_, headers, rows);
        }
        body_ += "</details>\n\n";
    }
}

void ReportEngine::addImage(const std::string& title, const std::string& imagePath) {
    body_ += "### " + title + "\n";
    body_ += "![" + title + "](" + imagePath + ")\n\n";
}

void ReportEngine::save(const std::string& filePath) const {
    std::error_code ec;
    const std::filesystem::path reportDir = std::filesystem::path(filePath).parent_path().empty()
        ? std::filesystem::current_path(ec)
        : std::filesystem::path(filePath).parent_path();

    const std::string normalized = normalizeMarkdownLinksForReport(body_, reportDir);
    std::ofstream out(filePath);
    out << normalized;
}
