#include "Dataset.h"
#include "CSVUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <charconv>
#include <string_view>

namespace {
constexpr size_t kMinTypeScanRows = 1000;
constexpr size_t kMaxTypeScanRows = 50000;
constexpr std::streamsize kLargeDatasetWarningBytes = 500LL * 1024 * 1024;

bool parseFiniteDouble(std::string_view input, double& out) {
    double parsed = 0.0;
    const char* begin = input.data();
    const char* end = begin + input.size();
    auto [ptr, ec] = std::from_chars(begin, end, parsed, std::chars_format::general);
    if (ec != std::errc{} || ptr != end || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

size_t requiredRowWidth(const std::vector<size_t>& indices) {
    if (indices.empty()) return 0;
    return *std::max_element(indices.begin(), indices.end()) + 1;
}

size_t computeAdaptiveScanLimit(bool exhaustiveScan, size_t expectedCols, std::streamsize fileSize) {
    if (exhaustiveScan) return std::numeric_limits<size_t>::max();

    size_t fileFactor = 0;
    if (fileSize > 0) {
        fileFactor = static_cast<size_t>(fileSize / (1024 * 1024));
    }

    size_t estimate = expectedCols * 256 + fileFactor * 8;
    return std::clamp(estimate, kMinTypeScanRows, kMaxTypeScanRows);
}

}

Dataset::Dataset(const std::string& filename) : filename_(filename) {}

void Dataset::setDelimiter(char delimiter) {
    if (delimiter == '"' || delimiter == '\n' || delimiter == '\r' || delimiter == '\0') {
        throw Seldon::DatasetException("Invalid delimiter character");
    }
    delimiter_ = delimiter;
}

void Dataset::analyzeMetadata() {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);
    
    CSVUtils::skipBOM(file);
    allColumnNames = readHeader(file);
    size_t expectedCols = allColumnNames.size();
    isNumeric.assign(expectedCols, true);
    
    std::vector<bool> hasValue(expectedCols, false);
    std::vector<std::vector<double>> dummyRaw(expectedCols);
    size_t skipped = 0;
    
    // Heuristic: scan first 5000 rows to detect types
    detectColumnTypes(file, expectedCols, false, isNumeric, hasValue, dummyRaw, true, skipped);
    
    // Reset rowCount because detectColumnTypes increments it
    rowCount = 0;
}

void Dataset::skipBOM(std::istream& is) {
    CSVUtils::skipBOM(is);
}

std::vector<std::string> Dataset::readHeader(std::ifstream& file) {
    bool malformed = false;
    auto header = CSVUtils::parseCSVLine(file, delimiter_, &malformed);
    if (header.empty() || malformed) {
        throw Seldon::DatasetException("Empty or malformed header in " + filename_);
    }
    return CSVUtils::normalizeHeader(header);
}

void Dataset::detectColumnTypes(std::ifstream& file, size_t expectedCols, bool exhaustiveScan, 
                                 std::vector<bool>& isNumeric, std::vector<bool>& hasValue, 
                                 std::vector<std::vector<double>>& rawData, 
                                 bool skipMalformed, size_t& totalSkipped) {
    rowCount = 0;
    totalSkipped = 0;
    size_t scanLimit = kMinTypeScanRows;

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    scanLimit = computeAdaptiveScanLimit(exhaustiveScan, expectedCols, fileSize);
    file.seekg(0, std::ios::beg);
    CSVUtils::skipBOM(file);
    bool headerMalformed = false;
    CSVUtils::parseCSVLine(file, delimiter_, &headerMalformed); // Skip header
    if (headerMalformed) {
        throw Seldon::DatasetException("Malformed CSV header in " + filename_);
    }

    size_t recordNumber = 1; // header
    while (file.peek() != EOF) {
        std::streamsize currentPos = file.tellg();
        if (rowCount % 100 == 0 && fileSize > 0) {
            double progress = (double)currentPos / fileSize * 100.0;
            std::cout << "\r[Agent] Ingesting Dataspace: [" << std::fixed << std::setprecision(1) << progress << "%] " << std::flush;
        }

        bool malformed = false;
        auto rawRow = CSVUtils::parseCSVLine(file, delimiter_, &malformed);
        ++recordNumber;
        if (rawRow.empty()) continue;
        if (malformed) {
            totalSkipped++;
            if (!skipMalformed) {
                throw Seldon::DatasetException("Malformed quoted field at record " + std::to_string(recordNumber));
            }
            continue;
        }

        if (rawRow.size() != expectedCols) {
            bool recovered = false;
            if (rawRow.size() > expectedCols) {
                size_t tail = rawRow.size();
                while (tail > expectedCols && rawRow[tail - 1].empty()) tail--;
                if (tail == expectedCols) {
                    rawRow.resize(expectedCols);
                    recovered = true;
                }
            }
            if (!recovered) {
                totalSkipped++;
                if (!skipMalformed) {
                    throw Seldon::DatasetException("Column mismatch at record " + std::to_string(recordNumber));
                }
                continue;
            }
        }

        std::vector<double> rowValues(expectedCols);
        bool rowTypeMatch = true;

        for (size_t i = 0; i < expectedCols; ++i) {
            if (rawRow[i].empty()) {
                rowValues[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double parsed = 0.0;
                if (parseFiniteDouble(rawRow[i], parsed)) {
                    rowValues[i] = parsed;
                    hasValue[i] = true;
                } else {
                    if (rowCount + totalSkipped < scanLimit) {
                        isNumeric[i] = false;
                    }
                    rowValues[i] = std::numeric_limits<double>::quiet_NaN();
                    if (isNumeric[i]) {
                        rowTypeMatch = false;
                    }
                }
            }
        }

        if (rowTypeMatch) {
            for (size_t i = 0; i < expectedCols; ++i) {
                rawData[i].push_back(rowValues[i]);
            }
            rowCount++;
        } else {
            totalSkipped++;
            if (!skipMalformed) {
                throw Seldon::DatasetException("Non-numeric value in numeric column at record " + std::to_string(recordNumber));
            }
        }
    }
}

void Dataset::processImputations(const std::vector<std::vector<double>>& rawData, 
                                 const std::vector<bool>& isNumeric, 
                                 const std::vector<std::string>& allColumnNames, 
                                 ImputationStrategy strategy) {
    numericIndices.clear();
    columnNames.clear();
    for (size_t i = 0; i < isNumeric.size(); ++i) {
        if (isNumeric[i]) {
            numericIndices.push_back(i);
            columnNames.push_back(allColumnNames[i]);
        }
    }

    if (numericIndices.empty()) {
        throw Seldon::DatasetException("No numeric columns detected");
    }

    std::vector<std::vector<double>> filteredData(numericIndices.size());
    for (size_t i = 0; i < numericIndices.size(); ++i) {
        filteredData[i] = std::move(rawData[numericIndices[i]]);
    }

    imputationValues.clear();
    calculateImputations(filteredData, imputationValues, strategy);

    columns.assign(numericIndices.size(), std::vector<double>(rowCount));
    for (size_t i = 0; i < numericIndices.size(); ++i) {
        for (size_t r = 0; r < rowCount; ++r) {
            if (std::isnan(filteredData[i][r])) {
                columns[i][r] = imputationValues[i];
            } else {
                columns[i][r] = filteredData[i][r];
            }
        }
    }

    if (strategy == ImputationStrategy::SKIP) {
        std::vector<std::vector<double>> finalColumns(numericIndices.size());
        size_t finalRowCount = 0;
        for (size_t r = 0; r < rowCount; ++r) {
            bool hasNaN = false;
            for (size_t i = 0; i < numericIndices.size(); ++i) {
                if (std::isnan(filteredData[i][r])) {
                    hasNaN = true;
                    break;
                }
            }
            if (!hasNaN) {
                for (size_t i = 0; i < numericIndices.size(); ++i) {
                    finalColumns[i].push_back(filteredData[i][r]);
                }
                finalRowCount++;
            }
        }
        columns = std::move(finalColumns);
        rowCount = finalRowCount;
    }
}

void Dataset::load(bool skipMalformed, bool exhaustiveScan, ImputationStrategy strategy) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);

    if (allColumnNames.empty()) {
        CSVUtils::skipBOM(file);
        allColumnNames = readHeader(file);
        isNumeric.assign(allColumnNames.size(), true);
    }
    
    size_t expectedCols = allColumnNames.size();
    std::vector<bool> hasValue(expectedCols, false);
    std::vector<std::vector<double>> rawData(expectedCols);
    size_t totalSkipped = 0;

    detectColumnTypes(file, expectedCols, exhaustiveScan, isNumeric, hasValue, rawData, skipMalformed, totalSkipped);
    processImputations(rawData, isNumeric, allColumnNames, strategy);

    if (totalSkipped > 0) {
        std::cout << "\n[Agent Warning] Filtered/Skipped " << totalSkipped << " row(s). Isolated " 
                  << numericIndices.size() << " numeric features.\n";
    }
}

void Dataset::loadAuto(bool skipMalformed, bool exhaustiveScan, ImputationStrategy strategy) {
    // Check file size
    std::ifstream file(filename_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);
    
    std::streamsize size = file.tellg();
    file.close();

    if (size > kLargeDatasetWarningBytes) {
        std::cout << "[Agent Warning] Large dataset detected (" << (size / (1024 * 1024)) << " MB). "
                  << "Consider using --chunked mode for out-of-core processing if memory is limited.\n";
    }

    load(skipMalformed, exhaustiveScan, strategy);
}

void Dataset::loadChunk(size_t startRow, size_t numRows) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);

    CSVUtils::skipBOM(file);
    auto allColNames = readHeader(file);

    if (numericIndices.empty()) {
        for (size_t i = 0; i < allColNames.size(); ++i) {
            numericIndices.push_back(i);
            columnNames.push_back(allColNames[i]);
        }
        imputationValues.assign(numericIndices.size(), 0.0);
    }

    // Skip to startRow
    for (size_t i = 0; i < startRow; ++i) {
        if (file.peek() == EOF) {
            columns.clear();
            rowCount = 0;
            return;
        }
        bool ignoredMalformed = false;
        CSVUtils::parseCSVLine(file, delimiter_, &ignoredMalformed);
    }

    columns.assign(numericIndices.size(), std::vector<double>());
    for (auto& col : columns) col.reserve(numRows);

    rowCount = 0;
    while (rowCount < numRows && file.peek() != EOF) {
        bool malformed = false;
        auto rawRow = CSVUtils::parseCSVLine(file, delimiter_, &malformed);
        if (rawRow.empty()) continue;
        if (malformed) continue;

        size_t minWidth = requiredRowWidth(numericIndices);
        if (rawRow.size() < minWidth) {
            rawRow.resize(minWidth, "");
        }

        for (size_t i = 0; i < numericIndices.size(); ++i) {
            size_t idx = numericIndices[i];
            double val;
            if (idx >= rawRow.size() || rawRow[idx].empty()) {
                val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
            } else {
                double parsed = 0.0;
                if (parseFiniteDouble(rawRow[idx], parsed)) {
                    val = parsed;
                } else {
                    val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
                }
            }
            columns[i].push_back(val);
        }
        rowCount++;
    }
    
    if (rowCount == 0) {
         throw Seldon::DatasetException("No rows loaded in chunk starting at " + std::to_string(startRow));
    }
}

void Dataset::openStream() {
    if (streamFile.is_open()) streamFile.close();
    streamFile.open(filename_, std::ios::binary);
    if (!streamFile.is_open()) throw Seldon::IOException("Could not open file " + filename_);
    CSVUtils::skipBOM(streamFile);
    
    if (allColumnNames.empty()) {
        allColumnNames = readHeader(streamFile);
        // Fallback if analyzeMetadata wasn't called: assume all columns are numeric
        if (numericIndices.empty()) {
            for (size_t i = 0; i < allColumnNames.size(); ++i) {
                numericIndices.push_back(i);
                columnNames.push_back(allColumnNames[i]);
            }
            imputationValues.assign(numericIndices.size(), 0.0);
        }
    } else {
        readHeader(streamFile); // skip header line
    }
}

bool Dataset::fetchNextChunk(size_t chunkSize) {
    if (!streamFile.is_open()) return false;
    
    if (numericIndices.empty() && !allColumnNames.empty()) {
        for (size_t i = 0; i < allColumnNames.size(); ++i) {
            numericIndices.push_back(i);
            columnNames.push_back(allColumnNames[i]);
        }
        imputationValues.assign(numericIndices.size(), 0.0);
    }

    columns.assign(numericIndices.size(), std::vector<double>());
    for (auto& col : columns) col.reserve(chunkSize);

    rowCount = 0;
    while (rowCount < chunkSize && streamFile.peek() != EOF) {
        bool malformed = false;
        auto rawRow = CSVUtils::parseCSVLine(streamFile, delimiter_, &malformed);
        if (rawRow.empty()) continue;
        if (malformed) continue;

        size_t minWidth = requiredRowWidth(numericIndices);
        if (rawRow.size() < minWidth) {
            rawRow.resize(minWidth, "");
        }

        for (size_t i = 0; i < numericIndices.size(); ++i) {
            size_t idx = numericIndices[i];
            double val;
            if (idx >= rawRow.size() || rawRow[idx].empty()) {
                val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
            } else {
                double parsed = 0.0;
                if (parseFiniteDouble(rawRow[idx], parsed)) {
                    val = parsed;
                } else {
                    val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
                }
            }
            columns[i].push_back(val);
        }
        rowCount++;
    }
    
    return rowCount > 0;
}

void Dataset::closeStream() {
    if (streamFile.is_open()) {
        streamFile.close();
    }
}

std::vector<std::string> Dataset::parseCSVLine(std::istream& is, bool* malformed, size_t* consumedLines) {
    return CSVUtils::parseCSVLine(is, delimiter_, malformed, consumedLines);
}

void Dataset::calculateImputations(const std::vector<std::vector<double>>& rawNumericData, 
                                 std::vector<double>& imputationValues, 
                                 ImputationStrategy strategy) {
    imputationValues.assign(rawNumericData.size(), 0.0);
    if (strategy == ImputationStrategy::SKIP || strategy == ImputationStrategy::ZERO) return;

    for (size_t i = 0; i < rawNumericData.size(); ++i) {
        std::vector<double> validValues;
        for (double v : rawNumericData[i]) {
            if (!std::isnan(v)) validValues.push_back(v);
        }

        if (validValues.empty()) {
            imputationValues[i] = 0.0;
            continue;
        }

        if (strategy == ImputationStrategy::MEAN) {
            double sum = 0;
            for (double v : validValues) sum += v;
            imputationValues[i] = sum / validValues.size();
        } else if (strategy == ImputationStrategy::MEDIAN) {
            size_t sz = validValues.size();
            size_t mid = sz / 2;
            std::nth_element(validValues.begin(), validValues.begin() + mid, validValues.end());
            double upper = validValues[mid];
            if (sz % 2 == 0) {
                std::nth_element(validValues.begin(), validValues.begin() + (mid - 1), validValues.begin() + mid);
                imputationValues[i] = (validValues[mid - 1] + upper) / 2.0;
            } else {
                imputationValues[i] = upper;
            }
        }
    }
}

void Dataset::printSummary() const {
    std::cout << "Dataset Loaded: " << filename_ << "\n";
    std::cout << "Rows: " << getRowCount() << ", Columns: " << getColCount() << "\n";
}
