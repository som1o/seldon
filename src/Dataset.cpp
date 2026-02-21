#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

Dataset::Dataset(const std::string& filename) : filename_(filename) {}

bool Dataset::load(bool skipMalformed) {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename_ << "\n";
        return false;
    }

    auto parseCSVLine = [](const std::string& s) {
        std::vector<std::string> row;
        std::string val;
        bool in_quotes = false;
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                row.push_back(val);
                val.clear();
            } else {
                val += c;
            }
        }
        row.push_back(val);
        for(auto& v : row) {
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
    };

    std::string line;
    std::vector<std::string> allColumnNames;
    if (std::getline(file, line)) {
        if (line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF') {
            line = line.substr(3);
        }
        allColumnNames = parseCSVLine(line);
    }

    size_t expectedCols = allColumnNames.size();
    if (expectedCols == 0) return false;

    rowCount = 0;
    size_t skippedRows = 0;

    std::vector<std::vector<std::string>> allRows;
    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \n\r\t") == std::string::npos) continue;

        std::vector<std::string> rawRow = parseCSVLine(line);

        if (rawRow.size() != expectedCols) {
            if (rawRow.size() > expectedCols && rawRow.back().empty()) {
                while (rawRow.size() > expectedCols && rawRow.back().empty()) {
                    rawRow.pop_back();
                }
            }
            if (rawRow.size() != expectedCols) {
                skippedRows++;
                continue; 
            }
        }
        allRows.push_back(std::move(rawRow));
    }

    std::vector<size_t> numericIndices;
    for (size_t i = 0; i < expectedCols; ++i) {
        bool isNumeric = false;
        for (size_t r = 0; r < allRows.size(); ++r) {
            if (!allRows[r][i].empty()) {
                try {
                    std::stod(allRows[r][i]);
                    isNumeric = true;
                } catch (...) {
                    isNumeric = false;
                }
                break;
            }
        }
        if (isNumeric) {
            numericIndices.push_back(i);
            columnNames.push_back(allColumnNames[i]);
        }
    }

    columns.resize(numericIndices.size());

    for (size_t r = 0; r < allRows.size(); ++r) {
        std::vector<double> parsedRow;
        bool validRow = true;
        for (size_t i : numericIndices) {
            if (allRows[r][i].empty()) {
                validRow = false; break;
            }
            try {
                parsedRow.push_back(std::stod(allRows[r][i]));
            } catch (...) {
                validRow = false; break;
            }
        }

        if (validRow && parsedRow.size() == numericIndices.size() && !numericIndices.empty()) {
            for (size_t c = 0; c < parsedRow.size(); ++c) {
                columns[c].push_back(parsedRow[c]);
            }
            rowCount++;
        } else {
            skippedRows++;
            if (!skipMalformed && validRow) {
                std::cerr << "Error: Malformed row encountered and skipMalformed is false.\n";
                return false;
            }
        }
    }

    if (skippedRows > 0) {
        std::cerr << "[Agent Warning] Skipped " << skippedRows << " malformed row(s). Isolated " 
                  << numericIndices.size() << " valid mathematical features out of " << expectedCols << " total columns.\n";
    }

    return true;
}


void Dataset::printSummary() const {
    std::cout << "Dataset Loaded: " << filename_ << "\n";
    std::cout << "Rows: " << getRowCount() << ", Columns: " << getColCount() << "\n";
}
