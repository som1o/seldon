#include "TypedDataset.h"
#include "CSVUtils.h"
#include "SeldonExceptions.h"
#include <algorithm>
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

bool TypedDataset::parseDouble(const std::string& v, double& out) {
    auto sv = trim(v);
    if (sv.empty()) return false;
    const char* b = sv.data();
    const char* e = b + sv.size();
    auto [p, ec] = std::from_chars(b, e, out, std::chars_format::general);
    return ec == std::errc{} && p == e && std::isfinite(out);
}

bool TypedDataset::parseDateTime(const std::string& v, int64_t& outUnixSeconds) {
    std::string s = trim(v);
    if (s.empty()) return false;

    std::tm tm{};
    std::istringstream ss1(s);
    ss1 >> std::get_time(&tm, "%Y-%m-%d");
    if (!ss1.fail()) {
        outUnixSeconds = static_cast<int64_t>(std::mktime(&tm));
        return true;
    }

    std::tm tm2{};
    std::istringstream ss2(s);
    ss2 >> std::get_time(&tm2, "%Y-%m-%d %H:%M:%S");
    if (!ss2.fail()) {
        outUnixSeconds = static_cast<int64_t>(std::mktime(&tm2));
        return true;
    }

    return false;
}

std::vector<std::string> TypedDataset::parseCSVLine(std::istream& is, bool& malformed, size_t& consumedLines) const {
    return CSVUtils::parseCSVLine(is, delimiter_, &malformed, &consumedLines);
}

void TypedDataset::load() {
    std::ifstream in(filename_, std::ios::binary);
    if (!in) throw Seldon::IOException("Could not open file: " + filename_);

    CSVUtils::skipBOM(in);

    bool malformed = false;
    size_t consumedLines = 0;
    auto header = parseCSVLine(in, malformed, consumedLines);
    if (malformed || header.empty()) throw Seldon::DatasetException("Malformed or empty CSV header");

    header = CSVUtils::normalizeHeader(header);

    std::vector<std::vector<std::string>> raw(header.size());
    size_t record = 1;
    while (in.peek() != EOF) {
        auto row = parseCSVLine(in, malformed, consumedLines);
        if (row.empty()) continue;
        record++;
        if (malformed) continue;

        if (row.size() < header.size()) row.resize(header.size(), "");
        if (row.size() > header.size()) row.resize(header.size());
        for (size_t c = 0; c < header.size(); ++c) raw[c].push_back(row[c]);
    }

    rowCount_ = raw.empty() ? 0 : raw[0].size();
    columns_.clear();
    columns_.reserve(header.size());

    for (size_t c = 0; c < header.size(); ++c) {
        size_t numericHits = 0;
        size_t datetimeHits = 0;
        size_t nonMissing = 0;
        for (const auto& s : raw[c]) {
            if (trim(s).empty()) continue;
            nonMissing++;
            double dv = 0.0;
            int64_t tv = 0;
            if (parseDouble(s, dv)) numericHits++;
            else if (parseDateTime(s, tv)) datetimeHits++;
        }

        TypedColumn col;
        col.name = header[c];
        col.missing.assign(rowCount_, false);

        if (nonMissing > 0 && numericHits == nonMissing) {
            col.type = ColumnType::NUMERIC;
            std::vector<double> values(rowCount_, std::numeric_limits<double>::quiet_NaN());
            for (size_t r = 0; r < rowCount_; ++r) {
                double dv = 0.0;
                if (parseDouble(raw[c][r], dv)) values[r] = dv;
                else col.missing[r] = true;
            }
            col.values = std::move(values);
        } else if (nonMissing > 0 && datetimeHits >= std::max<size_t>(3, nonMissing * 8 / 10)) {
            col.type = ColumnType::DATETIME;
            std::vector<int64_t> values(rowCount_, 0);
            for (size_t r = 0; r < rowCount_; ++r) {
                int64_t ts = 0;
                if (parseDateTime(raw[c][r], ts)) values[r] = ts;
                else col.missing[r] = true;
            }
            col.values = std::move(values);
        } else {
            col.type = ColumnType::CATEGORICAL;
            std::vector<std::string> values(rowCount_);
            for (size_t r = 0; r < rowCount_; ++r) {
                values[r] = trim(raw[c][r]);
                if (values[r].empty()) col.missing[r] = true;
            }
            col.values = std::move(values);
        }

        columns_.push_back(std::move(col));
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
