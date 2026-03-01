#include "CSVUtils.h"

#include <algorithm>
#include <cctype>
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

    enum class FieldState {
        LEADING,
        UNQUOTED,
        QUOTED,
        AFTER_QUOTE
    };

    std::vector<std::string> row;
    std::string value;
    FieldState state = FieldState::LEADING;
    bool currentFieldQuoted = false;
    bool hadDelimiter = false;
    bool hasAnyData = false;
    bool endedByNewline = false;
    bool atRecordStart = true;
    size_t recordBytes = 0;
    size_t physicalLines = 1;

    auto markLimitExceeded = [&]() {
        if (limitExceeded) *limitExceeded = true;
    };

    auto addRecordBytes = [&](size_t delta) {
        if (limits.maxRecordBytes > 0 && recordBytes > limits.maxRecordBytes - delta) {
            markLimitExceeded();
            return false;
        }
        recordBytes += delta;
        return true;
    };

    auto appendValueChar = [&](char ch) {
        value.push_back(ch);
        if (limits.maxFieldBytes > 0 && value.size() > limits.maxFieldBytes) {
            markLimitExceeded();
            return false;
        }
        return true;
    };

    auto pushField = [&]() {
        row.push_back(currentFieldQuoted ? value : trimUnquotedField(value));
        if (limits.maxColumns > 0 && row.size() > limits.maxColumns) {
            markLimitExceeded();
            return false;
        }
        value.clear();
        state = FieldState::LEADING;
        currentFieldQuoted = false;
        return true;
    };

    while (true) {
        int peeked = is.peek();
        if (peeked == EOF) break;

        char c = static_cast<char>(peeked);
        if (state == FieldState::QUOTED) {
            is.get();
            if (!addRecordBytes(1)) break;
            atRecordStart = false;

            if (c == '"') {
                if (is.peek() == '"') {
                    is.get();
                    if (!addRecordBytes(1)) break;
                    if (!appendValueChar('"')) break;
                    hasAnyData = true;
                    continue;
                }
                state = FieldState::AFTER_QUOTE;
                continue;
            }

            if (c == '\r') {
                if (is.peek() == '\n') {
                    is.get();
                    if (!addRecordBytes(1)) break;
                }
                if (consumedLines) ++(*consumedLines);
                ++physicalLines;
                if (limits.maxPhysicalLinesPerRecord > 0 && physicalLines > limits.maxPhysicalLinesPerRecord) {
                    markLimitExceeded();
                    break;
                }
                if (!appendValueChar('\n')) break;
                hasAnyData = true;
                continue;
            }

            if (c == '\n') {
                if (consumedLines) ++(*consumedLines);
                ++physicalLines;
                if (limits.maxPhysicalLinesPerRecord > 0 && physicalLines > limits.maxPhysicalLinesPerRecord) {
                    markLimitExceeded();
                    break;
                }
                if (!appendValueChar('\n')) break;
                hasAnyData = true;
                continue;
            }

            if (!appendValueChar(c)) break;
            hasAnyData = true;
            continue;
        }

        if (state == FieldState::AFTER_QUOTE) {
            if (c == delimiter) {
                is.get();
                if (!addRecordBytes(1)) break;
                atRecordStart = false;
                hadDelimiter = true;
                hasAnyData = true;
                if (!pushField()) break;
                continue;
            }

            if (c == '\r') {
                is.get();
                if (!addRecordBytes(1)) break;
                if (is.peek() == '\n') {
                    is.get();
                    if (!addRecordBytes(1)) break;
                }
                if (consumedLines) ++(*consumedLines);
                endedByNewline = true;
                if (!pushField()) break;
                break;
            }

            if (c == '\n') {
                is.get();
                if (!addRecordBytes(1)) break;
                if (consumedLines) ++(*consumedLines);
                endedByNewline = true;
                if (!pushField()) break;
                break;
            }

            if (std::isspace(static_cast<unsigned char>(c))) {
                is.get();
                if (!addRecordBytes(1)) break;
                atRecordStart = false;
                continue;
            }

            if (malformed) *malformed = true;
            is.get();
            if (!addRecordBytes(1)) break;
            atRecordStart = false;
            if (!appendValueChar(c)) break;
            state = FieldState::UNQUOTED;
            hasAnyData = true;
            continue;
        }

        if (c == delimiter) {
            is.get();
            if (!addRecordBytes(1)) break;
            atRecordStart = false;
            hadDelimiter = true;
            hasAnyData = true;
            if (!pushField()) break;
            continue;
        }

        if (c == '\r') {
            is.get();
            if (!addRecordBytes(1)) break;
            if (is.peek() == '\n') {
                is.get();
                if (!addRecordBytes(1)) break;
            }
            if (consumedLines) ++(*consumedLines);
            endedByNewline = true;
            if (!atRecordStart || hadDelimiter || !value.empty()) {
                if (!pushField()) break;
            }
            break;
        }

        if (c == '\n') {
            is.get();
            if (!addRecordBytes(1)) break;
            if (consumedLines) ++(*consumedLines);
            endedByNewline = true;
            if (!atRecordStart || hadDelimiter || !value.empty()) {
                if (!pushField()) break;
            }
            break;
        }

        if (state == FieldState::LEADING && std::isspace(static_cast<unsigned char>(c))) {
            is.get();
            if (!addRecordBytes(1)) break;
            atRecordStart = false;
            if (!appendValueChar(c)) break;
            continue;
        }

        if (state == FieldState::LEADING && c == '"' && trimUnquotedField(value).empty()) {
            is.get();
            if (!addRecordBytes(1)) break;
            atRecordStart = false;
            value.clear();
            state = FieldState::QUOTED;
            currentFieldQuoted = true;
            hasAnyData = true;
            continue;
        }

        is.get();
        if (!addRecordBytes(1)) break;
        atRecordStart = false;
        if (!appendValueChar(c)) break;
        state = FieldState::UNQUOTED;
        hasAnyData = true;
    }

    if (state == FieldState::QUOTED) {
        if (malformed) *malformed = true;
    }

    if (!row.empty() || hadDelimiter || !value.empty() || state != FieldState::LEADING || hasAnyData) {
        pushField();
    }

    if (row.size() == 1 && row[0].empty() && !hadDelimiter && endedByNewline) {
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
