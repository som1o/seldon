#include "TypedDataset.h"
#include "CSVUtils.h"
#include "SeldonExceptions.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

TypedDataset::TypedDataset(std::string filename, char delimiter)
    : filename_(std::move(filename)), delimiter_(delimiter) {}

std::string TypedDataset::trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

namespace {
std::string trimLocal(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool isMissingToken(const std::string& raw) {
    std::string s = trimLocal(raw);
    if (s.empty()) return true;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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
}

bool TypedDataset::parseDouble(const std::string& v, double& out) {
    auto sv = trim(v);
    if (isMissingToken(sv)) return false;

    std::string cleaned;
    cleaned.reserve(sv.size());
    for (char ch : sv) {
        if (ch != ',') cleaned.push_back(ch);
    }

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
    std::string s = trim(v);
    if (s.empty() || isMissingToken(s)) return false;

    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;

    const bool hasDateOnly = (s.size() == 10);
    const bool hasDateTime = (s.size() == 19);
    if (!hasDateOnly && !hasDateTime) return false;

    if (!parseFixedInt(s, 0, 4, year) || s[4] != '-' ||
        !parseFixedInt(s, 5, 2, month) || s[7] != '-' ||
        !parseFixedInt(s, 8, 2, day)) {
        return false;
    }

    if (hasDateTime) {
        if (s[10] != ' ' ||
            !parseFixedInt(s, 11, 2, hour) || s[13] != ':' ||
            !parseFixedInt(s, 14, 2, minute) || s[16] != ':' ||
            !parseFixedInt(s, 17, 2, second)) {
            return false;
        }
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
    
    size_t record = 1;
    size_t validRows = 0;
    
    // Pass 1: Determine column types and count rows
    auto pos = in.tellg();
    while (in.peek() != EOF) {
        auto row = parseCSVLine(in, malformed);
        if (row.empty()) continue;
        record++;
        if (malformed) continue;

        validRows++;
        size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < cols; ++c) {
            if (isMissingToken(row[c])) continue;
            nonMissing[c]++;
            double dv = 0.0;
            int64_t tv = 0;
            if (parseDouble(row[c], dv)) numericHits[c]++;
            else if (parseDateTime(row[c], tv)) datetimeHits[c]++;
        }
    }

    rowCount_ = validRows;
    columns_.clear();
    columns_.reserve(header.size());

    for (size_t c = 0; c < header.size(); ++c) {
        TypedColumn col;
        col.name = header[c];
        col.missing.assign(rowCount_, false);

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

    // Pass 2: Parse and store data
    in.clear();
    in.seekg(pos);
    
    size_t r = 0;
    while (in.peek() != EOF && r < rowCount_) {
        auto row = parseCSVLine(in, malformed);
        if (row.empty()) continue;
        if (malformed) continue;

        size_t cols = std::min(row.size(), header.size());
        for (size_t c = 0; c < header.size(); ++c) {
            std::string s = (c < cols) ? row[c] : "";
            if (isMissingToken(s)) {
                columns_[c].missing[r] = true;
                continue;
            }

            if (columns_[c].type == ColumnType::NUMERIC) {
                auto& numericValues = std::get<std::vector<double>>(columns_[c].values);
                double dv = 0.0;
                if (parseDouble(s, dv)) {
                    numericValues[r] = dv;
                } else {
                    columns_[c].missing[r] = true;
                }
            } else if (columns_[c].type == ColumnType::DATETIME) {
                auto& datetimeValues = std::get<std::vector<int64_t>>(columns_[c].values);
                int64_t ts = 0;
                if (parseDateTime(s, ts)) {
                    datetimeValues[r] = ts;
                } else {
                    columns_[c].missing[r] = true;
                }
            } else {
                auto& categoricalValues = std::get<std::vector<std::string>>(columns_[c].values);
                categoricalValues[r] = trim(s);
                if (categoricalValues[r].empty()) {
                    columns_[c].missing[r] = true;
                }
            }
        }
        r++;
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

void TypedDataset::removeRows(const std::vector<bool>& keepMask) {
    if (keepMask.size() != rowCount_) throw Seldon::DatasetException("Row mask size mismatch");

    for (auto& col : columns_) {
        std::vector<bool> newMissing;
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

    rowCount_ = std::count(keepMask.begin(), keepMask.end(), true);
}
