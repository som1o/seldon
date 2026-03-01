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
    if (!is.good()) return;

    const int first = is.peek();
    if (first == EOF || static_cast<unsigned char>(first) != 0xEF) {
        return;
    }

    is.get();
    const int second = is.peek();
    if (second == EOF || static_cast<unsigned char>(second) != 0xBB) {
        is.clear(is.rdstate() & ~std::ios::eofbit);
        is.unget();
        return;
    }

    is.get();
    const int third = is.peek();
    if (third == EOF || static_cast<unsigned char>(third) != 0xBF) {
        is.clear(is.rdstate() & ~std::ios::eofbit);
        is.unget();
        is.unget();
        return;
    }

    is.get();
}

std::vector<std::string> parseCSVLine(std::istream& is,
                                      char delimiter,
                                      bool* malformed,
                                      size_t* consumedLines,
                                      bool* limitExceeded,
                                      const ParseLimits& limits) {
    if (malformed) *malformed = false;
    if (consumedLines) *consumedLines = 0;
    if (limitExceeded) *limitExceeded = false;
    if (is.peek() == EOF) return {};

    std::vector<std::string> row;
    std::string val;
    bool inQuotes = false;
    bool currentFieldQuoted = false;
    bool lastPushedFieldQuoted = false;
    bool hasRecordData = false;
    bool hadDelimiter = false;
    bool recordHadAnyNewline = false;
    size_t recordBytes = 0;
    size_t physicalLineCount = 1;
    char c;

    auto markLimitExceeded = [&]() {
        if (limitExceeded) *limitExceeded = true;
    };

    auto exceedsRecordBytes = [&](size_t delta) {
        if (limits.maxRecordBytes == 0) return false;
        if (recordBytes > limits.maxRecordBytes - delta) {
            markLimitExceeded();
            return true;
        }
        recordBytes += delta;
        return false;
    };

    auto exceedsFieldBytes = [&](size_t fieldSize) {
        if (limits.maxFieldBytes == 0) return false;
        if (fieldSize > limits.maxFieldBytes) {
            markLimitExceeded();
            return true;
        }
        return false;
    };

    auto pushField = [&]() {
        row.push_back(currentFieldQuoted ? val : trimUnquotedField(val));
        lastPushedFieldQuoted = currentFieldQuoted;
        if (limits.maxColumns > 0 && row.size() > limits.maxColumns) {
            markLimitExceeded();
        }
    };

    while (is.get(c)) {
        if (exceedsRecordBytes(1)) break;

        if (c == '"') {
            if (!inQuotes && val.empty()) {
                inQuotes = true;
                currentFieldQuoted = true;
                hasRecordData = true;
            } else if (inQuotes) {
                if (is.peek() == '"') {
                    is.get();
                    if (exceedsRecordBytes(1)) break;
                    val += '"';
                    if (exceedsFieldBytes(val.size())) break;
                } else {
                    int next = is.peek();
                    if (next == EOF || next == delimiter || next == '\n' || next == '\r') {
                        inQuotes = false;
                    } else {
                        val += c;
                        if (exceedsFieldBytes(val.size())) break;
                    }
                }
            } else {
                val += c;
                if (exceedsFieldBytes(val.size())) break;
                hasRecordData = true;
            }
        } else if (c == delimiter && !inQuotes) {
            pushField();
            if (limitExceeded && *limitExceeded) break;
            val.clear();
            currentFieldQuoted = false;
            hadDelimiter = true;
            hasRecordData = true;
        } else if (c == '\r') {
            if (is.peek() == '\n') is.get();
            if (consumedLines) ++(*consumedLines);
            recordHadAnyNewline = true;
            if (inQuotes) {
                ++physicalLineCount;
                if (limits.maxPhysicalLinesPerRecord > 0 && physicalLineCount > limits.maxPhysicalLinesPerRecord) {
                    markLimitExceeded();
                    break;
                }
            }
            if (inQuotes) {
                val += '\n';
                if (exceedsFieldBytes(val.size())) break;
            } else {
                break;
            }
        } else if (c == '\n') {
            if (consumedLines) ++(*consumedLines);
            recordHadAnyNewline = true;
            if (inQuotes) {
                ++physicalLineCount;
                if (limits.maxPhysicalLinesPerRecord > 0 && physicalLineCount > limits.maxPhysicalLinesPerRecord) {
                    markLimitExceeded();
                    break;
                }
            }
            if (inQuotes) {
                val += '\n';
                if (exceedsFieldBytes(val.size())) break;
            } else {
                break;
            }
        } else {
            val += c;
            if (exceedsFieldBytes(val.size())) break;
            hasRecordData = true;
        }

        if (limitExceeded && *limitExceeded) break;
    }

    if (inQuotes && malformed) {
        *malformed = true;
    }

    if (hasRecordData || hadDelimiter || !val.empty()) {
        pushField();
    }

    if (row.size() == 1 && row[0].empty() && !lastPushedFieldQuoted && !hadDelimiter && recordHadAnyNewline) {
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

CSVChunkReader::CSVChunkReader(std::istream& is, char delimiter, ParseLimits limits)
    : is_(is), delimiter_(delimiter), limits_(limits) {}

std::vector<std::vector<std::string>> CSVChunkReader::readChunk(size_t maxRows,
                                                                bool* malformed,
                                                                bool* limitExceeded) {
    if (malformed) *malformed = false;
    if (limitExceeded) *limitExceeded = false;

    std::vector<std::vector<std::string>> rows;
    rows.reserve(maxRows);
    while (rows.size() < maxRows && is_.peek() != EOF) {
        bool rowMalformed = false;
        bool rowLimitExceeded = false;
        auto row = parseCSVLine(is_, delimiter_, &rowMalformed, nullptr, &rowLimitExceeded, limits_);
        if (rowLimitExceeded) {
            if (limitExceeded) *limitExceeded = true;
            break;
        }
        if (rowMalformed) {
            if (malformed) *malformed = true;
            continue;
        }
        if (row.empty()) continue;
        rows.push_back(std::move(row));
    }
    return rows;
}
} // namespace CSVUtils
