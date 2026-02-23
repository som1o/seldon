#include "CSVUtils.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>

namespace CSVUtils {
std::string trimUnquotedField(const std::string& value) {
    if (value.empty()) return value;
    size_t start = value.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    size_t end = value.find_last_not_of(" \t");
    return value.substr(start, end - start + 1);
}

void skipBOM(std::istream& is) {
    char bom[3];
    if (is.read(bom, 3)) {
        if (!((unsigned char)bom[0] == 0xEF && (unsigned char)bom[1] == 0xBB && (unsigned char)bom[2] == 0xBF)) {
            is.seekg(0);
        }
    } else {
        is.clear();
        is.seekg(0);
    }
}

std::vector<std::string> parseCSVLine(std::istream& is, char delimiter, bool* malformed, size_t* consumedLines) {
    if (malformed) *malformed = false;
    if (consumedLines) *consumedLines = 0;
    if (is.peek() == EOF) return {};

    std::vector<std::string> row;
    std::vector<uint8_t> fieldQuoted;
    std::string val;
    bool inQuotes = false;
    bool currentFieldQuoted = false;
    bool hasRecordData = false;
    bool hadDelimiter = false;
    char c;

    while (is.get(c)) {
        if (c == '"') {
            if (!inQuotes && val.empty()) {
                inQuotes = true;
                currentFieldQuoted = true;
                hasRecordData = true;
            } else if (inQuotes) {
                if (is.peek() == '"') {
                    val += '"';
                    is.get();
                } else {
                    int next = is.peek();
                    if (next == EOF || next == delimiter || next == '\n' || next == '\r') {
                        inQuotes = false;
                    } else {
                        val += c;
                    }
                }
            } else {
                val += c;
                hasRecordData = true;
            }
        } else if (c == delimiter && !inQuotes) {
            row.push_back(currentFieldQuoted ? val : trimUnquotedField(val));
            fieldQuoted.push_back(static_cast<uint8_t>(currentFieldQuoted ? 1 : 0));
            val.clear();
            currentFieldQuoted = false;
            hadDelimiter = true;
            hasRecordData = true;
        } else if (c == '\r') {
            if (is.peek() == '\n') is.get();
            if (consumedLines) ++(*consumedLines);
            if (inQuotes) {
                val += '\n';
            } else {
                break;
            }
        } else if (c == '\n') {
            if (consumedLines) ++(*consumedLines);
            if (inQuotes) {
                val += '\n';
            } else {
                break;
            }
        } else {
            val += c;
            hasRecordData = true;
        }
    }

    if (inQuotes && malformed) {
        *malformed = true;
    }

    if (hasRecordData || hadDelimiter || !val.empty()) {
        row.push_back(currentFieldQuoted ? val : trimUnquotedField(val));
        fieldQuoted.push_back(static_cast<uint8_t>(currentFieldQuoted ? 1 : 0));
    }

    if (row.size() == 1 && row[0].empty() && fieldQuoted[0] == 0 && !hadDelimiter) {
        return {};
    }

    return row;
}

std::vector<std::string> normalizeHeader(const std::vector<std::string>& header) {
    std::vector<std::string> out = header;
    std::unordered_set<std::string> seen;

    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].empty()) {
            out[i] = "column_" + std::to_string(i + 1);
        }

        std::string original = out[i];
        if (seen.find(out[i]) != seen.end()) {
            size_t suffix = 2;
            while (seen.find(original + "_" + std::to_string(suffix)) != seen.end()) {
                ++suffix;
            }
            out[i] = original + "_" + std::to_string(suffix);
        }
        seen.insert(out[i]);
    }

    return out;
}
} // namespace CSVUtils
