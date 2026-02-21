#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>

Dataset::Dataset(const std::string& filename) : filename_(filename) {}

void Dataset::analyzeMetadata() {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);
    
    skipBOM(file);
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

std::vector<std::string> Dataset::readHeader(std::ifstream& file) {
    auto header = parseCSVLine(file);
    if (header.empty()) {
        throw Seldon::DatasetException("Empty or malformed header in " + filename_);
    }
    return header;
}

void Dataset::detectColumnTypes(std::ifstream& file, size_t expectedCols, bool exhaustiveScan, 
                                 std::vector<bool>& isNumeric, std::vector<bool>& hasValue, 
                                 std::vector<std::vector<double>>& rawData, 
                                 bool skipMalformed, size_t& totalSkipped) {
    rowCount = 0;
    totalSkipped = 0;
    size_t scanLimit = exhaustiveScan ? std::numeric_limits<size_t>::max() : 1000;

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    skipBOM(file);
    parseCSVLine(file); // Skip header

    while (file.peek() != EOF) {
        std::streamsize currentPos = file.tellg();
        if (rowCount % 100 == 0 && fileSize > 0) {
            double progress = (double)currentPos / fileSize * 100.0;
            std::cout << "\r[Agent] Ingesting Dataspace: [" << std::fixed << std::setprecision(1) << progress << "%] " << std::flush;
        }

        auto rawRow = parseCSVLine(file);
        if (rawRow.empty()) continue;

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
                    throw Seldon::DatasetException("Column mismatch at row " + std::to_string(rowCount + totalSkipped));
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
                try {
                    rowValues[i] = std::stod(rawRow[i]);
                    hasValue[i] = true;
                } catch (...) {
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
                throw Seldon::DatasetException("Non-numeric value in numeric column at row " + std::to_string(rowCount + totalSkipped));
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
    if (allColumnNames.empty()) analyzeMetadata();

    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);
    
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

    if (size > 500 * 1024 * 1024) {
        std::cout << "[Agent Warning] Large dataset detected (" << (size / (1024 * 1024)) << " MB). "
                  << "Consider using --chunked mode for out-of-core processing if memory is limited.\n";
    }

    load(skipMalformed, exhaustiveScan, strategy);
}

void Dataset::loadChunk(size_t startRow, size_t numRows) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) throw Seldon::IOException("Could not open file " + filename_);

    skipBOM(file);
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
        parseCSVLine(file);
    }

    columns.assign(numericIndices.size(), std::vector<double>());
    for (auto& col : columns) col.reserve(numRows);

    rowCount = 0;
    while (rowCount < numRows && file.peek() != EOF) {
        auto rawRow = parseCSVLine(file);
        if (rawRow.empty()) continue;

        if (rawRow.size() < numericIndices.size()) {
            rawRow.resize(numericIndices.size(), "");
        }

        for (size_t i = 0; i < numericIndices.size(); ++i) {
            size_t idx = numericIndices[i];
            double val;
            if (idx >= rawRow.size() || rawRow[idx].empty()) {
                val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
            } else {
                try {
                    val = std::stod(rawRow[idx]);
                } catch (...) {
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
    skipBOM(streamFile);
    
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
        auto rawRow = parseCSVLine(streamFile);
        if (rawRow.empty()) continue;

        if (rawRow.size() < numericIndices.size()) {
            rawRow.resize(numericIndices.size(), "");
        }

        for (size_t i = 0; i < numericIndices.size(); ++i) {
            size_t idx = numericIndices[i];
            double val;
            if (idx >= rawRow.size() || rawRow[idx].empty()) {
                val = (i < imputationValues.size()) ? imputationValues[i] : 0.0;
            } else {
                try {
                    val = std::stod(rawRow[idx]);
                } catch (...) {
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

std::vector<std::string> Dataset::parseCSVLine(std::istream& is) {
    if (is.peek() == EOF) return {};
    
    std::vector<std::string> row;
    std::string val;
    bool in_quotes = false;
    char c;
    bool has_content = false;

    while (is.get(c)) {
        has_content = true;
        if (c == '"') {
            if (in_quotes && is.peek() == '"') {
                val += '"';
                is.get(); // Consume next escaped quote
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            row.push_back(val);
            val.clear();
        } else if (c == '\r') {
            if (is.peek() == '\n') is.get(); // Handle CRLF
            if (in_quotes) {
                val += '\n'; // Normalize multiline line endings to LF
            } else {
                break;
            }
        } else if (c == '\n') {
            if (in_quotes) {
                val += '\n';
            } else {
                break;
            }
        } else {
            val += c;
        }
    }
    
    // Final value after last comma or newline
    if (has_content || !row.empty()) {
        row.push_back(val);
    }

    // Trim whitespace only for non-empty fields and handle potential surrounding spaces outside quotes
    for (auto& v : row) {
        if (v.empty()) continue;
        size_t start = v.find_first_not_of(" \t");
        size_t end = v.find_last_not_of(" \t");
        if (start == std::string::npos) {
            v.clear();
        } else {
            v = v.substr(start, end - start + 1);
        }
    }
    return row;
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
            std::sort(validValues.begin(), validValues.end());
            size_t sz = validValues.size();
            if (sz % 2 == 0) imputationValues[i] = (validValues[sz / 2 - 1] + validValues[sz / 2]) / 2.0;
            else imputationValues[i] = validValues[sz / 2];
        }
    }
}

void Dataset::printSummary() const {
    std::cout << "Dataset Loaded: " << filename_ << "\n";
    std::cout << "Rows: " << getRowCount() << ", Columns: " << getColCount() << "\n";
}
