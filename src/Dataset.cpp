#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>

Dataset::Dataset(const std::string& filename) : filename_(filename) {}

std::vector<std::string> Dataset::parseCSVLine(std::istream& is) {
    std::vector<std::string> row;
    std::string val;
    bool in_quotes = false;
    char c;

    while (is.get(c)) {
        if (c == '"') {
            if (in_quotes && is.peek() == '"') {
                val += '"';
                is.get(); // Consume next quote
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            row.push_back(val);
            val.clear();
        } else if (c == '\r') {
            if (is.peek() == '\n') is.get();
            if (!in_quotes) break;
            val += '\n';
        } else if (c == '\n' && !in_quotes) {
            break;
        } else {
            val += c;
        }
    }
    
    if (!val.empty() || (is.eof() && !row.empty())) {
         row.push_back(val);
    }

    for (auto& v : row) {
        if (v.empty()) continue;
        size_t start = v.find_first_not_of(" \n\r\t");
        if (start == std::string::npos) {
            v.clear();
        } else {
            size_t end = v.find_last_not_of(" \n\r\t");
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

bool Dataset::load(bool skipMalformed, bool exhaustiveScan, ImputationStrategy strategy) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename_ << "\n";
        return false;
    }

    // Check for UTF-8 BOM (0xEF, 0xBB, 0xBF)
    char bom[3];
    bool hasBOM = false;
    if (file.read(bom, 3)) {
        if ((unsigned char)bom[0] == 0xEF && (unsigned char)bom[1] == 0xBB && (unsigned char)bom[2] == 0xBF) {
            hasBOM = true;
        }
    }
    
    if (!hasBOM) {
        file.seekg(0);
    }

    std::vector<std::string> allColumnNames = parseCSVLine(file);
    size_t expectedCols = allColumnNames.size();
    if (expectedCols == 0) return false;

    // Phase 1: Single Pass Read & Metadata Extraction
    std::vector<bool> isNumeric(expectedCols, true);
    std::vector<bool> hasValue(expectedCols, false);
    std::vector<std::vector<double>> rawData(expectedCols);
    
    rowCount = 0;
    size_t skippedRows = 0;
    size_t scanLimit = exhaustiveScan ? std::numeric_limits<size_t>::max() : 1000;

    // Progress Tracking
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    // Skip BOM if present (already checked, but seekg(0) might have moved it back)
    if (fileSize > 3) {
        char checkBOM[3];
        if (file.read(checkBOM, 3) && (unsigned char)checkBOM[0] == 0xEF && (unsigned char)checkBOM[1] == 0xB && (unsigned char)checkBOM[2] == 0xBF) {}
        else file.seekg(0);
    }

    // Skip Header again after seek
    parseCSVLine(file);

    while (file.peek() != EOF) {
        std::streamsize currentPos = file.tellg();
        if (rowCount % 100 == 0 && fileSize > 0) {
            double progress = (double)currentPos / fileSize * 100.0;
            std::cout << "\r[Agent] Ingesting Dataspace: [" << std::fixed << std::setprecision(1) << progress << "%] " << std::flush;
        }

        auto rawRow = parseCSVLine(file);
        if (rawRow.empty()) continue;

        bool validRow = true;
        if (rawRow.size() != expectedCols) {
            if (rawRow.size() > expectedCols) {
                size_t tail = rawRow.size();
                while (tail > expectedCols && rawRow[tail - 1].empty()) tail--;
                if (tail == expectedCols) rawRow.resize(expectedCols);
                else validRow = false;
            } else {
                validRow = false;
            }
        }

        if (!validRow) {
            skippedRows++;
            if (!skipMalformed) {
                std::cerr << "Error: Column mismatch at row " << rowCount + skippedRows << "\n";
                return false;
            }
            continue;
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
                    if (rowCount + skippedRows < scanLimit) {
                        isNumeric[i] = false;
                    }
                    rowValues[i] = std::numeric_limits<double>::quiet_NaN();
                    if (isNumeric[i]) {
                        rowTypeMatch = false; // Numeric column but bad value
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
            skippedRows++;
            if (!skipMalformed) {
                std::cerr << "Error: Non-numeric value in numeric column at row " << rowCount + skippedRows << "\n";
                return false;
            }
        }
    }

    // Phase 2: Post-processing & Imputation
    numericIndices.clear();
    columnNames.clear();
    for (size_t i = 0; i < expectedCols; ++i) {
        if (isNumeric[i]) {
            numericIndices.push_back(i);
            columnNames.push_back(allColumnNames[i]);
            if (!hasValue[i]) {
                std::cerr << "[Agent Warning] Column '" << allColumnNames[i] << "' is empty. Imputation defaulting to zero.\n";
            }
        }
    }

    if (numericIndices.empty()) {
        std::cerr << "Error: No numeric columns detected.\n";
        return false;
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
                if (strategy == ImputationStrategy::SKIP) {
                    // This is tricky: SKIP means we should throw out the whole row.
                    // But we already loaded the row.
                    // Let's refine the row filtering if strategy is SKIP.
                }
                columns[i][r] = imputationValues[i];
            } else {
                columns[i][r] = filteredData[i][r];
            }
        }
    }

    // Handle strategy == SKIP: We need to remove rows that have ANY NaN in numeric columns
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
        skippedRows += (rowCount - finalRowCount);
        rowCount = finalRowCount;
    }

    if (skippedRows > 0) {
        std::cerr << "[Agent Warning] Filtered/Skipped " << skippedRows << " row(s). Isolated " 
                  << numericIndices.size() << " numeric features.\n";
    }

    return true;
}

bool Dataset::loadChunk(size_t startRow, size_t numRows) {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) return false;

    // Handle BOM if present
    char bom[3];
    if (file.read(bom, 3)) {
        if (!((unsigned char)bom[0] == 0xEF && (unsigned char)bom[1] == 0xBB && (unsigned char)bom[2] == 0xBF)) {
            file.seekg(0);
        }
    }

    auto allColNames = parseCSVLine(file);
    if (allColNames.empty()) return false;

    // If metadata doesn't exist, perform a quick detection from the header/first row
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
            return false;
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
            // Fill short rows with NaN or imputation (best effort)
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

void Dataset::printSummary() const {
    std::cout << "Dataset Loaded: " << filename_ << "\n";
    std::cout << "Rows: " << getRowCount() << ", Columns: " << getColCount() << "\n";
}
