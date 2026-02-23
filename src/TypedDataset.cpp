#include "TypedDataset.h"
#include "CSVUtils.h"
#include "CommonUtils.h"
#include "SeldonExceptions.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>

TypedDataset::TypedDataset(std::string filename, char delimiter)
    : filename_(std::move(filename)), delimiter_(delimiter) {}

namespace {
bool isMissingToken(const std::string& raw) {
    std::string s = CommonUtils::trim(raw);
    if (s.empty()) return true;
    s = CommonUtils::toLower(std::move(s));
    return s == "na" || s == "n/a" || s == "null" || s == "none" || s == "nan" || s == "missing";
}

bool parseFixedInt(const std::string& s, size_t offset, size_t len, int& out) {
    if (offset + len > s.size()) return false;
    int value = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char ch = static_cast<unsigned char>(s[offset + i]);
        if (ch < '0' || ch > '9') return false;
        value = value * 10 + static_cast<int>(ch - '0');
    }
    out = value;
    return true;
}

bool isLeapYear(int year) {
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    return (year % 4 == 0);
}

int daysInMonth(int year, int month) {
    static const int kMonthDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) return 0;
    if (month == 2) return isLeapYear(year) ? 29 : 28;
    return kMonthDays[month - 1];
}

int64_t daysFromCivil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const int mp = static_cast<int>(m) + (m > 2 ? -3 : 9);
    const unsigned doy = (153 * static_cast<unsigned>(mp) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

bool parseTimePart(const std::string& timePart, int& hour, int& minute, int& second) {
    if (timePart.empty()) {
        hour = minute = second = 0;
        return true;
    }
    if (timePart.size() != 8) return false;
    if (!parseFixedInt(timePart, 0, 2, hour) || timePart[2] != ':' ||
        !parseFixedInt(timePart, 3, 2, minute) || timePart[5] != ':' ||
        !parseFixedInt(timePart, 6, 2, second)) {
        return false;
    }
    return true;
}

bool parseDatePart(const std::string& datePart, int& year, int& month, int& day) {
    // ISO: YYYY-MM-DD
    if (datePart.size() == 10 && datePart[4] == '-' && datePart[7] == '-') {
        return parseFixedInt(datePart, 0, 4, year) &&
               parseFixedInt(datePart, 5, 2, month) &&
               parseFixedInt(datePart, 8, 2, day);
    }

    // US: MM/DD/YYYY
    if (datePart.size() == 10 && datePart[2] == '/' && datePart[5] == '/') {
        return parseFixedInt(datePart, 0, 2, month) &&
               parseFixedInt(datePart, 3, 2, day) &&
               parseFixedInt(datePart, 6, 4, year);
    }

    // DMY: DD-MM-YYYY
    if (datePart.size() == 10 && datePart[2] == '-' && datePart[5] == '-') {
        return parseFixedInt(datePart, 0, 2, day) &&
               parseFixedInt(datePart, 3, 2, month) &&
               parseFixedInt(datePart, 6, 4, year);
    }

    return false;
}

std::string normalizeNumericToken(const std::string& input, TypedDataset::NumericSeparatorPolicy policy) {
    std::string cleaned;
    cleaned.reserve(input.size());
    for (char ch : input) {
        if (!std::isspace(static_cast<unsigned char>(ch)) && ch != '_') {
            cleaned.push_back(ch);
        }
    }

    const auto toUS = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char ch : s) {
            if (ch != ',') out.push_back(ch);
        }
        return out;
    };

    const auto toEuropean = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char ch : s) {
            if (ch == '.') {
                continue;
            }
            if (ch == ',') {
                out.push_back('.');
            } else {
                out.push_back(ch);
            }
        }
        return out;
    };

    if (policy == TypedDataset::NumericSeparatorPolicy::US_THOUSANDS) {
        return toUS(cleaned);
    }
    if (policy == TypedDataset::NumericSeparatorPolicy::EUROPEAN) {
        return toEuropean(cleaned);
    }

    const size_t dotPos = cleaned.find('.');
    const size_t commaPos = cleaned.find(',');
    if (dotPos != std::string::npos && commaPos != std::string::npos) {
        const size_t lastDot = cleaned.find_last_of('.');
        const size_t lastComma = cleaned.find_last_of(',');
        if (lastComma > lastDot) {
            return toEuropean(cleaned);
        }
        return toUS(cleaned);
    }

    if (commaPos != std::string::npos) {
        if (cleaned.find(',', commaPos + 1) == std::string::npos) {
            const size_t digitsAfter = cleaned.size() - commaPos - 1;
            if (digitsAfter >= 1 && digitsAfter <= 6) {
                std::string out = cleaned;
                out[commaPos] = '.';
                return out;
            }
        }
        return toUS(cleaned);
    }

    return cleaned;
}
}

bool TypedDataset::parseDouble(const std::string& v, double& out) const {
    auto sv = CommonUtils::trim(v);
    if (isMissingToken(sv)) return false;

    std::string cleaned = normalizeNumericToken(sv, numericSeparatorPolicy_);

    if (!cleaned.empty() && cleaned.front() == '+') {
        cleaned.erase(cleaned.begin());
    }
    if (!cleaned.empty() && cleaned.back() == '%') {
        cleaned.pop_back();
    }

    if (cleaned.empty()) return false;

    const char* b = cleaned.data();
    const char* e = b + cleaned.size();
    auto [p, ec] = std::from_chars(b, e, out, std::chars_format::general);
    return ec == std::errc{} && p == e && std::isfinite(out);
}

bool TypedDataset::parseDateTime(const std::string& v, int64_t& outUnixSeconds) {
    std::string s = CommonUtils::trim(v);
    if (s.empty() || isMissingToken(s)) return false;

    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;

    std::string datePart = s;
    std::string timePart;
    const size_t sep = s.find(' ');
    if (sep != std::string::npos) {
        datePart = s.substr(0, sep);
        timePart = s.substr(sep + 1);
    }

    if (!parseDatePart(datePart, year, month, day)) {
        return false;
    }
    if (!parseTimePart(timePart, hour, minute, second)) {
        return false;
    }

    if (month < 1 || month > 12) return false;
    const int dim = daysInMonth(year, month);
    if (day < 1 || day > dim) return false;
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59 || second < 0 || second > 60) return false;

    const int64_t days = daysFromCivil(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
    outUnixSeconds = days * 86400 + static_cast<int64_t>(hour) * 3600 + static_cast<int64_t>(minute) * 60 + second;
    return true;
}

std::vector<std::string> TypedDataset::parseCSVLine(std::istream& is, bool& malformed) const {
    return CSVUtils::parseCSVLine(is, delimiter_, &malformed, nullptr);
}

void TypedDataset::load() {
    std::ifstream in(filename_, std::ios::binary);
    if (!in) throw Seldon::IOException("Could not open file: " + filename_);

    CSVUtils::skipBOM(in);

    bool malformed = false;
    auto header = parseCSVLine(in, malformed);
    if (malformed || header.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");
    header = CSVUtils::normalizeHeader(header);

    std::vector<size_t> numericHits(header.size(), 0);
    std::vector<size_t> datetimeHits(header.size(), 0);
    std::vector<size_t> nonMissing(header.size(), 0);
    rowCount_ = 0;

    while (in.peek() != EOF) {
        auto row = parseCSVLine(in, malformed);
        if (row.empty() || malformed) continue;
        ++rowCount_;

        const size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < cols; ++c) {
            if (isMissingToken(row[c])) continue;
            ++nonMissing[c];
            double dv = 0.0;
            int64_t tv = 0;
            if (parseDouble(row[c], dv)) {
                ++numericHits[c];
            } else if (parseDateTime(row[c], tv)) {
                ++datetimeHits[c];
            }
        }
    }

    columns_.clear();
    columns_.reserve(header.size());

    for (size_t c = 0; c < header.size(); ++c) {
        TypedColumn col;
        col.name = header[c];
        col.missing.assign(rowCount_, static_cast<uint8_t>(0));

        if (nonMissing[c] > 0 && numericHits[c] >= std::max<size_t>(3, (nonMissing[c] * 8) / 10)) {
            col.type = ColumnType::NUMERIC;
            col.values = std::vector<double>(rowCount_, std::numeric_limits<double>::quiet_NaN());
        } else if (nonMissing[c] > 0 && datetimeHits[c] >= std::max<size_t>(3, nonMissing[c] * 8 / 10)) {
            col.type = ColumnType::DATETIME;
            col.values = std::vector<int64_t>(rowCount_, 0);
        } else {
            col.type = ColumnType::CATEGORICAL;
            col.values = std::vector<std::string>(rowCount_);
        }
        columns_.push_back(std::move(col));
    }

    std::vector<size_t> datetimeObserved(columns_.size(), 0);
    std::vector<size_t> datetimeParseFailures(columns_.size(), 0);

    std::unordered_map<size_t, std::vector<std::string>> datetimeRaw;
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (columns_[c].type == ColumnType::DATETIME) {
            datetimeRaw.emplace(c, std::vector<std::string>(rowCount_));
        }
    }

    in.clear();
    in.seekg(0, std::ios::beg);
    CSVUtils::skipBOM(in);
    malformed = false;
    auto headerSecondPass = parseCSVLine(in, malformed);
    if (malformed || headerSecondPass.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");

    size_t r = 0;
    while (in.peek() != EOF) {
        auto row = parseCSVLine(in, malformed);
        if (row.empty() || malformed) continue;
        if (r >= rowCount_) break;

        size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < header.size(); ++c) {
            std::string s = (c < cols) ? row[c] : "";
            if (isMissingToken(s)) {
                columns_[c].missing[r] = static_cast<uint8_t>(1);
                continue;
            }

            if (columns_[c].type == ColumnType::NUMERIC) {
                auto& numericValues = std::get<std::vector<double>>(columns_[c].values);
                double dv = 0.0;
                if (parseDouble(s, dv)) {
                    numericValues[r] = dv;
                } else {
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            } else if (columns_[c].type == ColumnType::DATETIME) {
                auto& datetimeValues = std::get<std::vector<int64_t>>(columns_[c].values);
                int64_t ts = 0;
                datetimeObserved[c]++;
                auto rawIt = datetimeRaw.find(c);
                if (rawIt != datetimeRaw.end()) {
                    rawIt->second[r] = CommonUtils::trim(s);
                }
                if (parseDateTime(s, ts)) {
                    datetimeValues[r] = ts;
                } else {
                    datetimeParseFailures[c]++;
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            } else {
                auto& categoricalValues = std::get<std::vector<std::string>>(columns_[c].values);
                categoricalValues[r] = CommonUtils::trim(s);
                if (categoricalValues[r].empty()) {
                    columns_[c].missing[r] = static_cast<uint8_t>(1);
                }
            }
        }
        ++r;
    }

    constexpr double kDatetimeFallbackFailureRatio = 0.15;
    for (size_t c = 0; c < columns_.size(); ++c) {
        if (columns_[c].type != ColumnType::DATETIME) continue;
        if (datetimeObserved[c] == 0 || datetimeParseFailures[c] == 0) continue;

        const double failureRatio = static_cast<double>(datetimeParseFailures[c]) /
                                    static_cast<double>(datetimeObserved[c]);
        if (failureRatio <= kDatetimeFallbackFailureRatio) continue;

        columns_[c].type = ColumnType::CATEGORICAL;
        columns_[c].values = std::vector<std::string>(rowCount_);
        auto& categoricalValues = std::get<std::vector<std::string>>(columns_[c].values);
        columns_[c].missing.assign(rowCount_, static_cast<uint8_t>(0));

        auto rawIt = datetimeRaw.find(c);
        if (rawIt == datetimeRaw.end()) continue;
        const auto& rawCol = rawIt->second;
        for (size_t row = 0; row < rawCol.size(); ++row) {
            const std::string s = rawCol[row];
            if (isMissingToken(s) || s.empty()) {
                columns_[c].missing[row] = static_cast<uint8_t>(1);
                categoricalValues[row].clear();
            } else {
                categoricalValues[row] = s;
            }
        }
    }
}

std::vector<size_t> TypedDataset::numericColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::NUMERIC) out.push_back(i);
    return out;
}

std::vector<size_t> TypedDataset::categoricalColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::CATEGORICAL) out.push_back(i);
    return out;
}

std::vector<size_t> TypedDataset::datetimeColumnIndices() const {
    std::vector<size_t> out;
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].type == ColumnType::DATETIME) out.push_back(i);
    return out;
}

int TypedDataset::findColumnIndex(const std::string& name) const {
    for (size_t i = 0; i < columns_.size(); ++i) if (columns_[i].name == name) return static_cast<int>(i);
    return -1;
}

void TypedDataset::removeRows(const MissingMask& keepMask) {
    if (keepMask.size() != rowCount_) throw Seldon::DatasetException("Row mask size mismatch");

    for (auto& col : columns_) {
        MissingMask newMissing;
        newMissing.reserve(rowCount_);

        if (col.type == ColumnType::NUMERIC) {
            auto& values = std::get<std::vector<double>>(col.values);
            std::vector<double> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        } else if (col.type == ColumnType::CATEGORICAL) {
            auto& values = std::get<std::vector<std::string>>(col.values);
            std::vector<std::string> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        } else {
            auto& values = std::get<std::vector<int64_t>>(col.values);
            std::vector<int64_t> next;
            next.reserve(rowCount_);
            for (size_t i = 0; i < rowCount_; ++i) {
                if (!keepMask[i]) continue;
                next.push_back(values[i]);
                newMissing.push_back(col.missing[i]);
            }
            values = std::move(next);
        }

        col.missing = std::move(newMissing);
    }

    rowCount_ = std::count(keepMask.begin(), keepMask.end(), static_cast<uint8_t>(1));
}
